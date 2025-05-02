from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


def get_power():
    # InfluxDB connection parameters
    influx_url = "http://192.168.178.114:8088"
    influx_token = os.environ.get("INFLUXDB_TOKEN")
    influx_org = "None"
    influx_bucket = "power_consumption"

    # Time range for data export (adjust as needed)
    start_time = datetime(2024, 12, 15)  # December 15, 2024
    end_time = datetime.now()  # Current time
    
    start = start_time
    end = end_time
    delta = timedelta(days=7)
    chunked_dfs = []
    
    while start < end:
        chunk_end = min(start + delta, end)
        # Query to get all power consumption data
        query = f'''
        from(bucket: "{influx_bucket}")
        |> range(start: {start.strftime("%Y-%m-%dT%H:%M:%SZ")}, stop: {chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")})
        |> filter(fn: (r) => r._measurement == "power_consumption")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        # Connect to InfluxDB and execute query
        with InfluxDBClient(url=influx_url, token=influx_token, org=influx_org) as client:
            query_api = client.query_api()
            print(f"Querying InfluxDB for data from {start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            try:
                # Set a timeout for the query to prevent hanging
                result = query_api.query_data_frame(query)  # 60 seconds timeout in milliseconds
            except Exception as e:
                print(f"Error querying InfluxDB: {e}")
                # Create empty DataFrame with expected columns if query fails
                result = pd.DataFrame(columns=['_time', 'device', 'power'])
            
            if isinstance(result, list):
                # If multiple DataFrames are returned, concatenate them
                df = pd.concat(result)
            else:
                df = result
            
            # Clean up the DataFrame
            if not df.empty:
                # Select and rename relevant columns
                df = df[['_time', 'device', 'power']]
                df.rename(columns={'_time': 'timestamp', 'device': 'device_name', 'power': 'power_watts'}, inplace=True)
                # Append to the list of DataFrames
                chunked_dfs.append(df)
            else:
                print("No data found for the specified time range")
        start = chunk_end
        
    # Concatenate all chunked DataFrames into one
    if chunked_dfs:
        final_df = pd.concat(chunked_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=['timestamp', 'device_name', 'power_watts'])
    return final_df
    
def export_to_parquet(df, path="data"):
    # Export DataFrame to Parquet
    parquet_filename = f"power_consumption_export.parquet"
    parquet_filepath = os.path.join(path, parquet_filename)
    df.to_parquet(parquet_filepath, index=False)    
    print(f"Data exported to {parquet_filepath}")

            
if __name__ == "__main__":
    df = get_power()
    export_to_parquet(df)
    