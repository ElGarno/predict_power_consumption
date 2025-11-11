"""
Module for fetching power consumption data from InfluxDB.

This module provides functions to query power consumption measurements from an InfluxDB
time-series database and export them to Parquet format for analysis.
"""

from influxdb_client import InfluxDBClient
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from config import settings, logger
from utils import export_to_parquet, validate_dataframe_not_empty


def get_power(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fetch power consumption data from InfluxDB.

    Queries InfluxDB in chunks to avoid timeout issues with large date ranges.
    Returns all power consumption measurements for the specified time period.

    Args:
        start_time: Start of time range (default: from settings.historical_data_start_date)
        end_time: End of time range (default: current time)

    Returns:
        pd.DataFrame: Power consumption data with columns:
            - timestamp: Measurement timestamp
            - device_name: Name of the device
            - power_watts: Power consumption in watts

    Raises:
        ConnectionError: If unable to connect to InfluxDB
        ValueError: If no data found or invalid time range
    """
    # Set default time range
    if start_time is None:
        start_time = datetime.strptime(settings.historical_data_start_date, "%Y-%m-%d")
    if end_time is None:
        end_time = datetime.now()

    if start_time >= end_time:
        raise ValueError(f"start_time ({start_time}) must be before end_time ({end_time})")

    logger.info(f"Fetching power data from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")

    # Query in chunks to avoid timeouts
    delta = timedelta(days=settings.influxdb_query_chunk_days)
    chunked_dfs = []
    current_start = start_time

    try:
        with InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org
        ) as client:
            query_api = client.query_api()

            while current_start < end_time:
                chunk_end = min(current_start + delta, end_time)

                # Flux query to get power consumption data
                query = f'''
                from(bucket: "{settings.influxdb_bucket}")
                |> range(start: {current_start.strftime("%Y-%m-%dT%H:%M:%SZ")},
                        stop: {chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")})
                |> filter(fn: (r) => r._measurement == "power_consumption")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                '''

                logger.debug(f"Querying chunk: {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")

                try:
                    result = query_api.query_data_frame(query)

                    if isinstance(result, list):
                        df = pd.concat(result) if result else pd.DataFrame()
                    else:
                        df = result

                    if not df.empty:
                        # Select and rename relevant columns
                        df = df[['_time', 'device', 'power']]
                        df.rename(
                            columns={
                                '_time': 'timestamp',
                                'device': 'device_name',
                                'power': 'power_watts'
                            },
                            inplace=True
                        )
                        chunked_dfs.append(df)
                        logger.info(f"Retrieved {len(df)} rows for chunk ending {chunk_end.strftime('%Y-%m-%d')}")
                    else:
                        logger.warning(f"No data found for chunk ending {chunk_end.strftime('%Y-%m-%d')}")

                except Exception as e:
                    logger.error(f"Error querying InfluxDB chunk: {e}")
                    # Continue to next chunk rather than failing completely
                    continue

                current_start = chunk_end

    except Exception as e:
        logger.error(f"Failed to connect to InfluxDB at {settings.influxdb_url}: {e}")
        raise ConnectionError(f"InfluxDB connection failed: {e}") from e

    # Concatenate all chunks
    if chunked_dfs:
        final_df = pd.concat(chunked_dfs, ignore_index=True)
        logger.info(f"Successfully fetched {len(final_df)} total power consumption records")
        return final_df
    else:
        logger.error(f"No power consumption data found for period {start_time} to {end_time}")
        raise ValueError("No power consumption data found for specified time range")


def export_power_data(
    df: Optional[pd.DataFrame] = None,
    filename: Optional[str] = None
) -> None:
    """
    Export power consumption data to Parquet format.

    Args:
        df: DataFrame to export (if None, fetches fresh data)
        filename: Output filename (default: from settings)

    Raises:
        ValueError: If DataFrame is empty
        IOError: If export fails
    """
    if df is None:
        logger.info("No DataFrame provided, fetching fresh power data")
        df = get_power()

    validate_dataframe_not_empty(df, "Power consumption data")

    filename = filename or settings.power_consumption_file
    export_to_parquet(df, filename)


if __name__ == "__main__":
    try:
        logger.info("Starting power consumption data export")
        df = get_power()
        export_power_data(df)
        logger.info("Power consumption data export completed successfully")
    except Exception as e:
        logger.error(f"Power consumption data export failed: {e}", exc_info=True)
        raise
