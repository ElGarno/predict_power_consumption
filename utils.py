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


def get_awtrix_client():
    """
    Get an AWTRIX client instance configured from settings.

    Returns:
        AwtrixClient: Configured AWTRIX client

    Raises:
        ImportError: If awtrix_client module is not available
    """
    try:
        from awtrix_client import AwtrixClient
        return AwtrixClient(settings.awtrix_host, settings.awtrix_port)
    except ImportError as e:
        logger.error(f"Failed to import AwtrixClient: {e}")
        raise


def send_awtrix_countdown(hours: int, minutes: int) -> bool:
    """
    Send countdown to next forecast to AWTRIX display.

    Args:
        hours: Hours until next forecast
        minutes: Minutes until next forecast

    Returns:
        bool: True if successful, False otherwise
    """
    if not settings.awtrix_enabled:
        logger.debug("AWTRIX notifications disabled, skipping countdown")
        return False

    try:
        from awtrix_client import AwtrixMessage

        # Format the time remaining
        if hours > 0:
            time_str = f"{hours}h {minutes}m"
        else:
            time_str = f"{minutes}m"

        target_time = f"{settings.daily_prediction_hour:02d}:{settings.daily_prediction_minute:02d}"
        text = f"Next forecast: {target_time} ({time_str})"

        message = AwtrixMessage(
            text=text,
            icon="27464",  # Sun emoji icon
            color="#FFD700",  # Gold color
            duration=10
        )

        client = get_awtrix_client()
        success = client.send_notification(message)

        if success:
            logger.debug(f"AWTRIX countdown sent: {time_str} until forecast")
        else:
            logger.warning("Failed to send AWTRIX countdown")

        return success

    except Exception as e:
        logger.error(f"Error sending AWTRIX countdown: {e}")
        return False


def send_awtrix_forecast_summary(total_energy_kwh: float, overproduction_hours: list = None) -> bool:
    """
    Send tomorrow's forecast summary to AWTRIX display.

    Args:
        total_energy_kwh: Total expected solar energy in kWh
        overproduction_hours: List of hour tuples (start, end) with overproduction

    Returns:
        bool: True if successful, False otherwise
    """
    if not settings.awtrix_enabled:
        logger.debug("AWTRIX notifications disabled, skipping forecast summary")
        return False

    try:
        from awtrix_client import AwtrixMessage

        # Format the forecast message
        text = f"Tomorrow: {total_energy_kwh:.1f} kWh expected"

        message = AwtrixMessage(
            text=text,
            icon="27464",  # Sun emoji icon
            color="#00FF00",  # Green color for good forecast
            duration=15,
            priority=1
        )

        client = get_awtrix_client()
        success = client.send_notification(message)

        if success:
            logger.info(f"AWTRIX forecast summary sent: {total_energy_kwh:.1f} kWh")
        else:
            logger.warning("Failed to send AWTRIX forecast summary")

        # If there are overproduction hours, send a follow-up message
        if overproduction_hours and len(overproduction_hours) > 0:
            # Format overproduction time windows
            time_windows = []
            for start, end in overproduction_hours:
                time_windows.append(f"{start:02d}:00-{end:02d}:00")

            overproduction_text = f"Overproduction: {', '.join(time_windows)}"

            overproduction_message = AwtrixMessage(
                text=overproduction_text,
                icon="32491",  # Electric plug icon
                color="#FF6600",  # Orange color
                duration=15,
                priority=1
            )

            client.send_notification(overproduction_message)
            logger.info(f"AWTRIX overproduction alert sent: {len(overproduction_hours)} time windows")

        return success

    except Exception as e:
        logger.error(f"Error sending AWTRIX forecast summary: {e}")
        return False


def send_awtrix_simple_status(message: str, icon: str = "27464", color: str = "#FFD700") -> bool:
    """
    Send a simple status message to AWTRIX display.

    Args:
        message: Status message text
        icon: LaMetric icon code (default: sun emoji)
        color: Hex color code (default: gold)

    Returns:
        bool: True if successful, False otherwise
    """
    if not settings.awtrix_enabled:
        logger.debug("AWTRIX notifications disabled, skipping status")
        return False

    try:
        from awtrix_client import AwtrixMessage

        status_message = AwtrixMessage(
            text=message,
            icon=icon,
            color=color,
            duration=10
        )

        client = get_awtrix_client()
        success = client.send_notification(status_message)

        if success:
            logger.debug(f"AWTRIX status sent: {message}")
        else:
            logger.warning(f"Failed to send AWTRIX status: {message}")

        return success

    except Exception as e:
        logger.error(f"Error sending AWTRIX status: {e}")
        return False
