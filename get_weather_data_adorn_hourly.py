import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import zipfile
import io
import glob
import re
import pyarrow.parquet as pq

# Define the time period (last 6 months)
end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
start_date = end_date - timedelta(days=365)

# DWD FTP base URL for hourly data
base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"

# Parameters of interest
parameters = {
    "air_temperature": {"name": "Temperature", "prefix": "TU", "postfix": "akt", "recent_folder": True},
    "precipitation": {"name": "Precipitation", "prefix": "RR", "postfix": "akt", "recent_folder": True},
    "solar": {"name": "Solar Radiation", "prefix": "ST", "postfix": "row", "recent_folder": False},
    "wind": {"name": "Wind", "prefix": "FF", "postfix": "akt", "recent_folder": True},
    "cloudiness": {"name": "Cloudiness", "prefix": "N", "postfix": "akt", "recent_folder": True}
}

# Attendorn coordinates (approximate)
attendorn_lat, attendorn_lon = 51.1279, 7.9022

# Function to rename columns
def rename_columns(df):
    rename_dict = {
        "QN_9": "quality_temp",
        "TT_TU": "temp_c",
        "RF_TU": "humidity_pct",
        "QN_8_cloudiness": "quality_cloud",
        "  R1": "precip_mm",
        "RS_IND": "precip_indicator",
        "WRTR": "precip_type",
        "QN_3": "quality_wind",
        "   F": "wind_speed_ms",
        "   D": "wind_dir_deg",
        # "QN_8": "quality_precip",
        "V_N_I": "cloud_measure_type",
        " V_N": "cloud_cover_8ths",
        "QN_592": "quality_solar",
        "ATMO_LBERG": "atm_radiation_Jcm2",
        "FD_LBERG": "diffuse_solar_Jcm2",
        "FG_LBERG": "global_solar_Jcm2",
        "SD_LBERG": "sunshine_duration_min",
        "ZENIT": "zenith_angle_deg",
        "MESS_DATUM_WOZ": "local_solar_time"
    }
    df.rename(columns=rename_dict, inplace=True)
    df.rename(columns={'QN_8': 'quality_precip'}, inplace=True)
    return df

# Function to find nearby stations
def get_nearby_stations(parameter, start_date=start_date, end_date=end_date):
    param_info = parameters[parameter]
    
    if param_info["recent_folder"]:
        stations_url = f"{base_url}/{parameter}/recent/{param_info['prefix']}_Stundenwerte_Beschreibung_Stationen.txt"
    else:
        stations_url = f"{base_url}/{parameter}/{param_info['prefix']}_Stundenwerte_Beschreibung_Stationen.txt"
    
    print(f"Requesting station data from: {stations_url}")
    response = requests.get(stations_url)
    
    if response.status_code != 200:
        print(f"Failed to get station data for {parameter}, status code: {response.status_code}")
        return []
    
    content_lines = response.text.split('\n')
    if param_info["recent_folder"]:
        pattern = re.compile(r"(\d+)\s+(\d{8})\s+(\d{8})\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+(.+?)\s+([A-Za-z-]+)\s+(\w+)")
        parsed_data = [pattern.match(line).groups() for line in content_lines[2:] if pattern.match(line)]
        df_temp = pd.DataFrame(parsed_data, columns=["station_id", "start_date", "end_date", "height", "lat", "lon", "name", "state", "type"])
    else:
        # Define regex pattern
        pattern = re.compile(r"(\d+)\s+(\d{8})\s+(\d{8})\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+(.+?)\s+([A-Za-z-]+)")
    # pattern = re.compile(r"(\d+)\s+(\d{8})\s+(\d{8})\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+(.+?)\s+([A-Za-z-]+)\s+(\w+)")
        parsed_data = [pattern.match(line).groups() for line in content_lines[2:] if pattern.match(line)]
        df_temp = pd.DataFrame(parsed_data, columns=["station_id", "start_date", "end_date", "height", "lat", "lon", "name", "state"])
    df_temp["lat"] = df_temp["lat"].astype(float)
    df_temp["lon"] = df_temp["lon"].astype(float)
    df_temp["distance"] = ((df_temp["lat"] - attendorn_lat)**2 + (df_temp["lon"] - attendorn_lon)**2)**0.5
    # select the station that is closest to the coordinates of Attendorn and is within the time range
    # first select all stations that are within the time range
    df_temp["start_date"] = pd.to_datetime(df_temp["start_date"], format="%Y%m%d")
    df_temp["end_date"] = pd.to_datetime(df_temp["end_date"], format="%Y%m%d")
    max_end_date = df_temp["end_date"].max()
    if param_info["recent_folder"]:
        df_temp = df_temp[(df_temp["start_date"] <= start_date) & (df_temp["end_date"] >= end_date)]
    else:
        df_temp = df_temp[(df_temp["start_date"] <= start_date) & (df_temp["end_date"] == max_end_date)]
    # then select the one that is closest to the coordinates of Attendorn
    closest_station = df_temp.loc[df_temp["distance"].idxmin()]
    return closest_station["station_id"]
    
    
    
    
    # df_temp = df_temp.sort_values(by="distance")
    # return df_temp["station_id"].astype(str).tolist()

# Fix timestamp format for solar data
def parse_solar_timestamp(timestamp):
    """Handles timestamps in solar data that contain HH:MM instead of just HH."""
    try:
        return datetime.strptime(timestamp, "%Y%m%d%H:%M")
    except ValueError:
        return pd.NaT  # If parsing fails, return NaT (Not a Time)
    
# Function to adjust timestamps to the nearest hour
def adjust_timestamps(df):
    df["timestamp"] = df["timestamp"].dt.floor("h")
    return df

# Function to download and process data for a given parameter
def get_weather_data(parameter):
    param_info = parameters[parameter]
    # station_list = get_nearby_stations(parameter)
    station_id = get_nearby_stations(parameter)
    
    # for station_id in station_list:
    if param_info["recent_folder"]:
        url = f"{base_url}/{parameter}/recent/stundenwerte_{param_info['prefix']}_{station_id}_{param_info['postfix']}.zip"
    else:
        url = f"{base_url}/{parameter}/stundenwerte_{param_info['prefix']}_{station_id}_{param_info['postfix']}.zip"
    
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Using station {station_id} for {parameter} data.")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            try:
                data_file = [name for name in z.namelist() if "produkt" in name][0]
                with z.open(data_file) as f:
                    df = pd.read_csv(f, sep=";", encoding="latin1", na_values="-999")
                    
                    # Remove unwanted columns to prevent merge conflicts
                    df = df.loc[:, ~df.columns.isin(["STATIONS_ID", "eor"])]
                    df = rename_columns(df)
                    
                    if "MESS_DATUM" in df.columns:
                        if parameter == "solar":
                            df["MESS_DATUM"] = df["MESS_DATUM"].apply(parse_solar_timestamp)
                            print(f"solar-data: {df.tail()}")
                        else:
                            df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", errors="coerce")
                    
                    df = df[(df["MESS_DATUM"] >= start_date) & (df["MESS_DATUM"] <= end_date)]
                    # if df.empty try next closest station
                    df.rename(columns={"MESS_DATUM": "timestamp"}, inplace=True)
                    df = adjust_timestamps(df)
                    if not df.empty:
                        return df
            except Exception as e:
                print(f"Error extracting data for station {station_id}: {e}")
    else:
        print(f"Station {station_id} not available for {parameter}, trying next closest station...")
    
    print(f"No valid weather data found for {parameter}.")
    return pd.DataFrame()

# Main execution
def main():
    all_data = pd.DataFrame()
    
    for param in parameters.keys():
        print(f"Processing {parameters[param]['name']} data...")
        data = get_weather_data(param)
        
        if not data.empty:
            data.rename(columns={"MESS_DATUM": "timestamp"}, inplace=True)
            if all_data.empty:
                all_data = data
            else:
                all_data = pd.merge(all_data, data, on="timestamp", how="outer", suffixes=(f"_{param}", ""))
    
    if not all_data.empty:
        all_data.to_csv("attendorn_hourly_weather_data.csv")
        # save also as parquer file
        all_data.to_parquet("data/attendorn_hourly_weather_data.parquet")
        print(f"Weather data successfully saved with {len(all_data)} rows.")
    else:
        print("No weather data was retrieved.")

if __name__ == "__main__":
    main()
