"""
Module for fetching historical weather data from DWD (German Weather Service).

Uses the Wetterdienst library to query DWD observation stations and retrieve
hourly weather measurements for power prediction features.
"""

from wetterdienst.provider.dwd.observation import DwdObservationRequest
from wetterdienst import Settings as WetterdientSettings
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime

from config import settings, logger
from utils import export_to_parquet, validate_dataframe_not_empty, get_timezone_aware_datetime


def get_data_by_api(
    wetter_settings: WetterdientSettings,
    parameters: list[Tuple[str, str]],
    location: Tuple[float, float],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_stations: int = 5
) -> pd.DataFrame:
    """
    Fetch weather data from DWD API for specified parameters.

    Args:
        wetter_settings: Wetterdienst settings instance
        parameters: List of (resolution, dataset) tuples, e.g., ("hourly", "temperature_air")
        location: (latitude, longitude) tuple for location
        start_date: Start date in YYYY-MM-DD format (default: from settings)
        end_date: End date in YYYY-MM-DD format (default: today)
        num_stations: Number of closest stations to consider

    Returns:
        pd.DataFrame: Combined weather data from all parameters

    Raises:
        ValueError: If no data is returned from API
        ConnectionError: If API request fails
    """
    start_date = start_date or settings.historical_data_start_date
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    logger.info(f"Fetching weather data for location {location}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Parameters: {[f'{r}/{d}' for r, d in parameters]}")

    df_list = []

    for resolution, dataset in parameters:
        try:
            logger.debug(f"Querying {resolution}/{dataset}")

            request = DwdObservationRequest(
                parameters=(resolution, dataset),
                start_date=start_date,
                end_date=end_date,
                settings=wetter_settings,
            )

            # Find closest weather stations to the specified location
            stations = request.filter_by_rank(latlon=location, rank=num_stations)

            # Query data from all selected stations
            df = stations.values.all().df.drop_nulls()

            if df.is_empty():
                logger.warning(f"No data found for {resolution}/{dataset}")
                continue

            # Convert Polars to Pandas for compatibility (preserve column names)
            df_pandas = df.to_pandas()
            df_list.append(df_pandas)
            logger.info(f"Retrieved {len(df_pandas)} rows for {resolution}/{dataset}")

        except Exception as e:
            logger.error(f"Failed to fetch {resolution}/{dataset}: {e}")
            # Continue with other parameters rather than failing completely
            continue

    if not df_list:
        raise ValueError(f"No weather data retrieved for location {location}")

    # Combine all dataframes
    df_climate = pd.concat(df_list, ignore_index=True)
    logger.info(f"Successfully combined weather data: {len(df_climate)} total rows")

    return df_climate


def get_pivot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot weather data from long to wide format.

    Transforms data so each parameter becomes a column, with date as index.
    Removes rows with missing values and constant-value columns.

    Args:
        df: Weather data in long format with 'date', 'parameter', 'value' columns

    Returns:
        pd.DataFrame: Pivoted weather data with one column per parameter

    Raises:
        ValueError: If required columns are missing or result is empty
    """
    required_columns = ['date', 'parameter', 'value']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    logger.debug("Pivoting weather data from long to wide format")

    # Create pivot table
    df_pivot = df.pivot_table(
        index='date',
        columns='parameter',
        values='value',
        aggfunc='mean'  # Average if multiple values for same parameter/date
    )

    # Reset index to make date a column
    df_pivot = df_pivot.reset_index()

    original_rows = len(df_pivot)
    original_cols = len(df_pivot.columns)

    # Remove rows where any value is missing
    df_cleaned = df_pivot.dropna(how='any')
    rows_dropped = original_rows - len(df_cleaned)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} rows with missing values")

    # Remove columns with only one unique value (constant columns)
    df_cleaned = df_cleaned.loc[:, df_cleaned.nunique() != 1]
    cols_dropped = original_cols - len(df_cleaned.columns)
    if cols_dropped > 0:
        logger.info(f"Dropped {cols_dropped} constant-value columns")

    validate_dataframe_not_empty(df_cleaned, "Cleaned weather data")

    logger.info(f"Pivot complete: {len(df_cleaned)} rows, {len(df_cleaned.columns)} columns")
    return df_cleaned


def get_weather_data_pivot(
    location: Optional[Tuple[float, float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch and pivot historical weather data from DWD.

    Main entry point for getting weather data. Fetches data from DWD API
    and returns it in pivoted format ready for merging with power data.

    Args:
        location: (latitude, longitude) tuple (default: from settings)
        start_date: Start date in YYYY-MM-DD format (default: from settings)
        end_date: End date in YYYY-MM-DD format (default: today)

    Returns:
        pd.DataFrame: Pivoted weather data with one row per datetime

    Raises:
        ValueError: If no data available
        ConnectionError: If API request fails
    """
    location = location or (settings.latitude, settings.longitude)

    # Configure Wetterdienst to skip empty data
    wetter_settings = WetterdientSettings(ts_skip_empty=True)

    # Define weather parameters to fetch (relevant for solar power prediction)
    # Format: (resolution, dataset) tuples
    parameters = [
        ("hourly", "temperature_air"),
        ("hourly", "precipitation"),
        ("hourly", "wind"),
        ("hourly", "cloudiness"),  # Fixed: was cloud_cover
        ("hourly", "solar"),
    ]

    # Fetch data from API
    df_climate = get_data_by_api(
        wetter_settings,
        parameters,
        location,
        start_date,
        end_date
    )

    # Pivot to wide format
    df_climate_pivot = get_pivot_df(df_climate)

    return df_climate_pivot


def export_weather_data(
    df: Optional[pd.DataFrame] = None,
    filename: Optional[str] = None
) -> None:
    """
    Export weather data to Parquet format.

    Args:
        df: DataFrame to export (if None, fetches fresh data)
        filename: Output filename (default: from settings)

    Raises:
        ValueError: If DataFrame is empty
        IOError: If export fails
    """
    if df is None:
        logger.info("No DataFrame provided, fetching fresh weather data")
        df = get_weather_data_pivot()

    validate_dataframe_not_empty(df, "Weather data")

    filename = filename or settings.weather_data_file
    export_to_parquet(df, filename)


if __name__ == "__main__":
    try:
        logger.info("Starting weather data export")
        df_climate_pivot = get_weather_data_pivot()
        export_weather_data(df_climate_pivot)
        logger.info("Weather data export completed successfully")
    except Exception as e:
        logger.error(f"Weather data export failed: {e}", exc_info=True)
        raise
