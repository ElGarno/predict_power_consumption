"""
Module for preparing and merging weather and power consumption data for ML training.

This module combines historical weather data with power consumption measurements,
performs feature selection, and creates the final training dataset.
"""

import pandas as pd
import pytz
from typing import Optional
from datetime import timedelta

from get_power_consumption_data import get_power
from get_weather_data_adorn_hourly_api import get_weather_data_pivot
from config import settings, logger
from utils import export_to_parquet, validate_dataframe_not_empty, validate_required_columns


def preprocess_weather_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess weather data by selecting relevant features.

    Performs the following steps:
    1. Drops columns with all missing values
    2. Drops rows with any missing values
    3. Drops constant-value columns
    4. Selects highly important features for solar prediction

    Args:
        data: Raw weather data DataFrame

    Returns:
        pd.DataFrame: Preprocessed weather data with selected features

    Raises:
        ValueError: If preprocessing results in empty DataFrame
    """
    logger.info(f"Preprocessing weather data: {len(data)} rows, {len(data.columns)} columns")

    # Drop columns with all missing values
    df_cleaned = data.dropna(axis=1, how='all')
    cols_dropped = len(data.columns) - len(df_cleaned.columns)
    if cols_dropped > 0:
        logger.info(f"Dropped {cols_dropped} columns with all missing values")

    # Drop rows where any value is missing
    original_rows = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(how='any')
    rows_dropped = original_rows - len(df_cleaned)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} rows with missing values")

    # Drop columns with only one unique value (constants)
    original_cols = len(df_cleaned.columns)
    df_cleaned = df_cleaned.loc[:, df_cleaned.nunique() != 1]
    const_cols_dropped = original_cols - len(df_cleaned.columns)
    if const_cols_dropped > 0:
        logger.info(f"Dropped {const_cols_dropped} constant-value columns")

    # Select highly important features for prediction
    highly_important_features = ['date'] + settings.prediction_features

    # Check which features are available
    available_features = [f for f in highly_important_features if f in df_cleaned.columns]
    missing_features = set(highly_important_features) - set(available_features)

    if missing_features:
        logger.warning(f"Some requested features not available: {missing_features}")

    if 'date' not in available_features:
        raise ValueError("'date' column missing from weather data")

    df_cleaned = df_cleaned[available_features]

    validate_dataframe_not_empty(df_cleaned, "Preprocessed weather data")
    logger.info(f"Weather preprocessing complete: {len(df_cleaned)} rows, {len(df_cleaned.columns)} columns")

    return df_cleaned


def preprocess_pv_data(
    data: pd.DataFrame,
    df_relevant: pd.DataFrame
) -> pd.DataFrame:
    """
    Preprocess power consumption data and align with weather data timeframe.

    Converts timestamps to match weather data timezone, filters to relevant
    date range, and resamples to hourly intervals grouped by device.

    Args:
        data: Raw power consumption data
        df_relevant: Weather data with date range to match

    Returns:
        pd.DataFrame: Hourly power consumption pivoted by device

    Raises:
        ValueError: If timestamp column is missing or data is empty
    """
    logger.info(f"Preprocessing power consumption data: {len(data)} rows")

    validate_required_columns(data, ['timestamp', 'device_name', 'power_watts'], "Power data")

    df_power = data.copy()

    # Convert timestamps to timezone-aware datetimes
    tz = pytz.timezone(settings.timezone)

    # Make timestamp timezone-aware if it isn't already
    if df_power['timestamp'].dt.tz is None:
        logger.debug("Localizing naive timestamps to configured timezone")
        df_power['timestamp'] = df_power['timestamp'].dt.tz_localize('UTC')

    # Convert to target timezone
    df_power['timestamp_local'] = df_power['timestamp'].dt.tz_convert(tz)

    # Remove timezone info for easier merging with weather data
    df_power['timestamp_naive'] = df_power['timestamp_local'].dt.tz_localize(None)

    # Filter to relevant time range (with small buffer for hourly aggregation)
    # Make sure to convert to naive datetime for comparison
    start_time = pd.to_datetime(df_relevant['date'].min()).tz_localize(None) - timedelta(hours=1)
    end_time = pd.to_datetime(df_relevant['date'].max()).tz_localize(None) + timedelta(hours=1)

    df_power_relevant = df_power[
        (df_power['timestamp_naive'] >= start_time) &
        (df_power['timestamp_naive'] <= end_time)
    ]

    logger.info(f"Filtered to relevant time range: {len(df_power_relevant)} rows")

    if df_power_relevant.empty:
        raise ValueError("No power consumption data in weather data time range")

    # Resample to hourly data grouped by device_name
    # Use mean() since power_watts is instantaneous power, not energy
    # (multiple samples per hour should be averaged, not summed)
    logger.debug("Resampling power data to hourly intervals")
    df_power_hourly = (
        df_power_relevant[['timestamp_naive', 'device_name', 'power_watts']]
        .set_index('timestamp_naive')
        .groupby(['device_name', pd.Grouper(freq='h')])
        .mean()
        .reset_index()
    )

    # Pivot so each device is a column
    df_power_pivot = df_power_hourly.pivot(
        index='timestamp_naive',
        columns='device_name',
        values='power_watts'
    )

    validate_dataframe_not_empty(df_power_pivot, "Hourly power consumption data")
    logger.info(f"Power preprocessing complete: {len(df_power_pivot)} rows, {len(df_power_pivot.columns)} devices")

    return df_power_pivot


def merge_data(weather_data: pd.DataFrame, pv_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather data with power consumption data on timestamp.

    Args:
        weather_data: Preprocessed weather data with 'date' column
        pv_data: Preprocessed power data with 'timestamp_naive' as index

    Returns:
        pd.DataFrame: Merged dataset ready for ML training

    Raises:
        ValueError: If merge results in empty DataFrame
    """
    logger.info("Merging weather and power consumption data")

    # Ensure date column is datetime and timezone-naive
    if not pd.api.types.is_datetime64_any_dtype(weather_data['date']):
        weather_data['date'] = pd.to_datetime(weather_data['date'])

    # Convert to timezone-naive if needed (DWD data comes with UTC timezone)
    if weather_data['date'].dt.tz is not None:
        logger.debug("Converting weather dates to timezone-naive for merging")
        weather_data['date'] = weather_data['date'].dt.tz_localize(None)

    # Merge on timestamps
    df_merged = pd.merge(
        weather_data,
        pv_data,
        how='inner',
        left_on='date',
        right_on='timestamp_naive'
    )

    validate_dataframe_not_empty(df_merged, "Merged data")
    logger.info(f"Merge complete: {len(df_merged)} rows, {len(df_merged.columns)} columns")

    return df_merged


def get_merged_data_for_training(
    get_power_from_db: Optional[bool] = None,
    get_weather_data_from_api: Optional[bool] = None
) -> pd.DataFrame:
    """
    Get merged weather and power data ready for ML training.

    Main entry point for data preparation pipeline. Fetches or loads data,
    preprocesses it, and merges into final training dataset.

    Args:
        get_power_from_db: Fetch power data from InfluxDB (default: from settings)
        get_weather_data_from_api: Fetch weather data from API (default: from settings)

    Returns:
        pd.DataFrame: Merged and preprocessed training data

    Raises:
        ValueError: If data fetching or processing fails
    """
    get_power_from_db = get_power_from_db if get_power_from_db is not None else settings.get_power_from_db
    get_weather_data_from_api = get_weather_data_from_api if get_weather_data_from_api is not None else settings.get_weather_from_api

    # Load weather data
    if get_weather_data_from_api:
        logger.info("Fetching weather data from API")
        weather_data = get_weather_data_pivot()
    else:
        logger.info(f"Loading weather data from file: {settings.weather_data_file}")
        weather_data = pd.read_parquet(settings.get_full_path(settings.weather_data_file))

    # Load power consumption data
    if get_power_from_db:
        logger.info("Fetching power data from InfluxDB")
        pv_data = get_power()
    else:
        logger.info(f"Loading power data from file: {settings.power_consumption_file}")
        pv_data = pd.read_parquet(settings.get_full_path(settings.power_consumption_file))

    # Preprocess data
    weather_data_relevant = preprocess_weather_data(weather_data)
    pv_data_hourly = preprocess_pv_data(pv_data, weather_data_relevant)

    # Merge datasets
    merged_data = merge_data(weather_data_relevant, pv_data_hourly)

    logger.info(f"Training data preparation complete: {len(merged_data)} samples")
    return merged_data


def export_merged_data(
    df: Optional[pd.DataFrame] = None,
    filename: Optional[str] = None
) -> None:
    """
    Export merged training data to Parquet format.

    Args:
        df: DataFrame to export (if None, generates fresh data)
        filename: Output filename (default: from settings)

    Raises:
        ValueError: If DataFrame is empty
        IOError: If export fails
    """
    if df is None:
        logger.info("No DataFrame provided, generating merged data")
        df = get_merged_data_for_training()

    validate_dataframe_not_empty(df, "Merged training data")

    filename = filename or settings.merged_data_file
    export_to_parquet(df, filename)


if __name__ == "__main__":
    try:
        logger.info("Starting merged data preparation")
        df_merged = get_merged_data_for_training(
            get_power_from_db=False,
            get_weather_data_from_api=True
        )
        export_merged_data(df_merged)
        logger.info("Merged data preparation completed successfully")
    except Exception as e:
        logger.error(f"Merged data preparation failed: {e}", exc_info=True)
        raise
