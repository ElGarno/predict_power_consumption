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

def preprocess_pv_data(data):
    df_power_consumption = pd.read_parquet('data/power_consumption_export_20250321.parquet')
    df_power_consumption['timestamp_naive'] = df_power_consumption['timestamp'].dt.tz_localize(None) + pd.Timedelta(hours=1)

    # Get the rows where timestamp is within the timestamp range of df_relevant
    df_power_consumption_relevant = df_power_consumption[
        (df_power_consumption['timestamp_naive'] >= df_relevant['timestamp'].min()) & 
        (df_power_consumption['timestamp_naive'] <= df_relevant['timestamp'].max() + pd.Timedelta(hours=1))
    ]


def main():
    # Load weather data
    weather_data = pd.read_csv("attendorn_hourly_weather_data.csv")
    
    # Preprocess weather data
    weather_data = preprocess_weather_data(weather_data)
    
    
if __name__ == "__main__":
    main()