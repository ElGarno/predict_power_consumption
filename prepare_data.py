import pandas as pd
from get_power_consumption_data import get_power
from get_weather_data_adorn_hourly_api import get_weather_data_pivot
import os


def preprocess_weather_data(data):
    #drop columns with all missing values
    df_cleaned = data.dropna(axis=1, how='all')
    # drop rows where all values are missing
    df_cleaned = df_cleaned.dropna(how='any')
    # drop any column which contains only the same value
    df_cleaned = df_cleaned.loc[:, df_cleaned.nunique() != 1]
    highly_important_features = ['date', 'radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 'sunshine_duration', 'sun_zenith_angle', 'temperature_air_mean_2m', 'cloud_cover_total', 'humidity']
    return df_cleaned[highly_important_features]

def preprocess_pv_data(data, df_relevant):
    df_power_consumption = data.copy()
    df_power_consumption['timestamp_naive'] = df_power_consumption['timestamp'].dt.tz_localize(None) + pd.Timedelta(hours=1)

    # Get the rows where timestamp is within the timestamp range of df_relevant
    df_power_consumption_relevant = df_power_consumption[
        (df_power_consumption['timestamp_naive'] >= df_relevant['date'].min()) & 
        (df_power_consumption['timestamp_naive'] <= df_relevant['date'].max() + pd.Timedelta(hours=1))
    ]
    # resample the power consumption data to hourly data grouped by device_name
    # Method 1: Using pd.Grouper (recommended approach)
    df_power_consumption_hourly = (df_power_consumption_relevant[['timestamp_naive', 'device_name', 'power_watts']]
                .set_index('timestamp_naive')
                .groupby(['device_name', pd.Grouper(freq='h')])
                .sum()
                .reset_index())
    df_power_consumption_hourly_pivot = df_power_consumption_hourly.pivot(index='timestamp_naive', columns='device_name', values='power_watts')
    return df_power_consumption_hourly_pivot

def merge_data(weather_data, pv_data):
    # Merge the weather data with the power consumption data
    df_merged = pd.merge(weather_data, pv_data, how='inner', left_on='date', right_on='timestamp_naive')
    return df_merged


def get_merged_data_for_training(get_power_from_db=False, get_weather_data_from_api=True):
    if get_weather_data_from_api:
        # Load weather data from API
        weather_data = get_weather_data_pivot()
    else:
        # Load weather data from parquet
        weather_data = pd.read_parquet('data/attendorn_hourly_weather_data_api.parquet')
    # Load PV data
    if get_power_from_db:
        # Load the data from the database
        pv_data = get_power()
    else:
        # Load the data from a file
        pv_data = pd.read_parquet("data/power_consumption_export.parquet")
    
    # Preprocess weather data
    weather_data_relevant = preprocess_weather_data(weather_data)
    # Preprocess PV data
    pv_data_hourly = preprocess_pv_data(pv_data, weather_data_relevant)
    # Merge the data
    merged_data = merge_data(weather_data_relevant, pv_data_hourly)
    return merged_data

def export_to_parquet(df, path="data"):
    # Export DataFrame to Parquet
    parquet_filename = f"merged_data_weather_power.parquet"
    parquet_filepath = os.path.join(path, parquet_filename)
    df.to_parquet(parquet_filepath, index=False)
    print(f"Weather data successfully exported with {len(df)} rows to {parquet_filename}.")
    
    

if __name__ == "__main__":
    df_merged = get_merged_data_for_training(get_power_from_db=False, get_weather_data_from_api=True)
    # export to parquet
    export_to_parquet(df_merged)