import pandas as pd


def preprocess_weather_data(data):
    #drop columns with all missing values
    df_cleaned = data.dropna(axis=1, how='all')
    # drop rows where all values are missing
    df_cleaned = df_cleaned.dropna(how='any')
    # drop any column which contains only the same value
    df_cleaned = df_cleaned.loc[:, df_cleaned.nunique() != 1]
    highly_important_features = ['timestamp', 'global_solar_Jcm2', 'diffuse_solar_Jcm2', 'sunshine_duration_min', 'zenith_angle_deg', 'temp_c', 'cloud_cover_8ths', 'humidity_pct']
    return df_cleaned[highly_important_features]

def preprocess_pv_data(data, df_relevant):
    df_power_consumption = data.copy()
    df_power_consumption['timestamp_naive'] = df_power_consumption['timestamp'].dt.tz_localize(None) + pd.Timedelta(hours=1)

    # Get the rows where timestamp is within the timestamp range of df_relevant
    df_power_consumption_relevant = df_power_consumption[
        (df_power_consumption['timestamp_naive'] >= df_relevant['timestamp'].min()) & 
        (df_power_consumption['timestamp_naive'] <= df_relevant['timestamp'].max() + pd.Timedelta(hours=1))
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
    df_merged = pd.merge(weather_data, pv_data, how='inner', left_on='timestamp', right_on='timestamp_naive')
    return df_merged


def main():
    # Load weather data
    weather_data = pd.read_parquet('data/attendorn_hourly_weather_data_api.parquet')
    # Load PV data
    pv_data = pd.read_parquet("data/power_consumption_export_20250424.parquet")
    
    # Preprocess weather data
    weather_data_relevant = preprocess_weather_data(weather_data)
    print(weather_data_relevant.info())
    # Preprocess PV data
    pv_data_hourly = preprocess_pv_data(pv_data, weather_data_relevant)
    # Merge the data
    merged_data = merge_data(weather_data_relevant, pv_data_hourly)
    # save merged data to parquet
    merged_data.to_parquet("data/merged_data.parquet")
    
if __name__ == "__main__":
    main()