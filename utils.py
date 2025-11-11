"""
Shared utility functions for the predict_power_consumption application.

This module contains common functions used across multiple modules to avoid code duplication.
"""

import requests
import http.client
import urllib.parse
from pathlib import Path
from typing import Optional
import pandas as pd
import pytz
from datetime import datetime
from config import settings, logger


def export_to_parquet(df: pd.DataFrame, filename: str, path: Optional[Path] = None) -> Path:
    """
    Export a DataFrame to Parquet format.

    Args:
        df: DataFrame to export
        filename: Name of the output file (should end with .parquet)
        path: Directory path for export (default: settings.data_dir)

    Returns:
        Path: Full path to the exported file

    Raises:
        ValueError: If DataFrame is empty or filename is invalid
        IOError: If file cannot be written
    """
    if df.empty:
        raise ValueError("Cannot export empty DataFrame")

    if not filename.endswith('.parquet'):
        filename = f"{filename}.parquet"

    target_path = path or settings.data_dir
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    filepath = target_path / filename

    try:
        df.to_parquet(filepath, index=False)
        logger.info(f"Data exported successfully: {filepath} ({len(df)} rows)")
        return filepath
    except Exception as e:
        logger.error(f"Failed to export data to {filepath}: {e}")
        raise IOError(f"Failed to write parquet file: {e}") from e


def read_parquet(filename: str, path: Optional[Path] = None) -> pd.DataFrame:
    """
    Read a Parquet file into a DataFrame.

    Args:
        filename: Name of the file to read
        path: Directory path (default: settings.data_dir)

    Returns:
        pd.DataFrame: Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    target_path = path or settings.data_dir
    filepath = Path(target_path) / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    try:
        df = pd.read_parquet(filepath)
        logger.info(f"Data loaded successfully: {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to read parquet file {filepath}: {e}")
        raise IOError(f"Failed to read parquet file: {e}") from e


def send_pushover_notification(
    message: str,
    title: Optional[str] = None,
    image_path: Optional[Path] = None,
    user: Optional[str] = None
) -> bool:
    """
    Send a notification via Pushover.

    Args:
        message: Notification message text
        title: Optional notification title
        image_path: Optional path to image attachment
        user: Pushover user/group key (default: from settings)

    Returns:
        bool: True if notification sent successfully, False otherwise

    Raises:
        ValueError: If notifications are disabled or credentials missing
    """
    if not settings.enable_notifications:
        logger.info("Notifications disabled, skipping")
        return False

    if not settings.pushover_api_token or not settings.pushover_user_group:
        logger.warning("Pushover credentials not configured, skipping notification")
        return False

    user_key = user or settings.pushover_user_group

    try:
        data = {
            "token": settings.pushover_api_token,
            "user": user_key,
            "message": message,
        }

        if title:
            data["title"] = title

        files = None
        if image_path:
            image_path = Path(image_path)
            if image_path.exists():
                files = {"attachment": open(image_path, "rb")}
            else:
                logger.warning(f"Image file not found: {image_path}")

        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data=data,
            files=files,
            timeout=settings.http_timeout_seconds
        )

        if files:
            files["attachment"].close()

        response.raise_for_status()
        logger.info(f"Pushover notification sent successfully to {user_key}")
        return True

    except requests.exceptions.Timeout:
        logger.error("Pushover notification timed out")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Pushover notification: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending notification: {e}")
        return False


def get_timezone_aware_datetime(dt: Optional[datetime] = None) -> datetime:
    """
    Convert a datetime to timezone-aware datetime using configured timezone.

    Args:
        dt: Datetime to convert (default: current time)

    Returns:
        datetime: Timezone-aware datetime
    """
    if dt is None:
        dt = datetime.now()

    tz = pytz.timezone(settings.timezone)

    if dt.tzinfo is None:
        # Naive datetime - localize it
        return tz.localize(dt)
    else:
        # Already timezone-aware - convert to target timezone
        return dt.astimezone(tz)


def get_tomorrow_date(tz: Optional[str] = None) -> datetime.date:
    """
    Get tomorrow's date in the configured timezone.

    Args:
        tz: Timezone string (default: from settings)

    Returns:
        datetime.date: Tomorrow's date
    """
    timezone = pytz.timezone(tz or settings.timezone)
    now = datetime.now(timezone)
    tomorrow = now.date() + pd.Timedelta(days=1)
    return tomorrow


def validate_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Validate that a DataFrame is not empty.

    Args:
        df: DataFrame to validate
        name: Name for error message

    Raises:
        ValueError: If DataFrame is empty
    """
    if df.empty:
        raise ValueError(f"{name} is empty")
    logger.debug(f"{name} validation passed: {len(df)} rows, {len(df.columns)} columns")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], name: str = "DataFrame") -> None:
    """
    Validate that a DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error message

    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{name} missing required columns: {missing_columns}")
    logger.debug(f"{name} has all required columns: {required_columns}")


def retry_with_backoff(func, max_attempts: Optional[int] = None, *args, **kwargs):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts (default: from settings)
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        The return value of func if successful

    Raises:
        The last exception encountered if all attempts fail
    """
    import time

    attempts = max_attempts or settings.http_retry_attempts
    last_exception = None

    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < attempts:
                wait_time = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s, 4s, ...
                logger.warning(f"Attempt {attempt}/{attempts} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {attempts} attempts failed")

    raise last_exception
