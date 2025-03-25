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
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

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

# Function to find the closest station
def get_closest_station(parameter):
    param_info = parameters[parameter]
    
    # Construct the URL based on the parameter's specific format
    if param_info["recent_folder"]:
        stations_url = f"{base_url}/{parameter}/recent/{param_info['prefix']}_Stundenwerte_Beschreibung_Stationen.txt"
    else:
        stations_url = f"{base_url}/{parameter}/{param_info['prefix']}_Stundenwerte_Beschreibung_Stationen.txt"
    
    print(f"Requesting station data from: {stations_url}")
    response = requests.get(stations_url)
    
    if response.status_code != 200:
        print(f"Failed to get station data for {parameter}, status code: {response.status_code}")
        return None
    
    # print(f"Test output of response: {response.text[:500]}")
    # First, examine the file structure
    content_lines = response.text.split('\n')
    # for i, line in enumerate(content_lines[:5]):  # Print first few lines to debug
    #     print(f"Line {i}: {line}")
    
    # Try to dynamically determine the file structure
    try:
        df_columns = pd.read_csv(io.StringIO(response.text), sep='\s+', nrows=1).columns
        # Use pandas to automatically detect columns without specifying usecols
        # df_temp = pd.read_csv(io.StringIO(response.text), sep='\t', skiprows=2)
        # for i, col in enumerate(df_columns):
        #     print(f"Column {i}: {col}")
        
        # Map column names to indices based on common patterns in DWD files
        id_col = next((i for i, col in enumerate(df_columns) if 'Stations_id' in col or 'ID' in col.upper()), 0)
        lat_col = next((i for i, col in enumerate(df_columns) if 'geoBreite' in col or 'lat' in col.lower() or 'geoBreite' in col), None)
        lon_col = next((i for i, col in enumerate(df_columns) if 'geoLaenge' in col or 'lon' in col.lower() or 'geoLaenge' in col), None)
        
        print(f"Using columns: ID={id_col}, LAT={lat_col}, LON={lon_col}")
        
        # print(f"Test output of df_temp: {df_temp.head()}")
        # # print the first 3 columns and 2 rows of df_temp
        # text_data = response.text

        # # Use StringIO to read the text as if it were a file
        # df_temp = pd.read_fwf(io.StringIO(text_data), skiprows=1)  # Skip the separator line
        # df_temp = pd.read_csv(io.StringIO(response.text), sep='\s+', skiprows=2, ncols=5)
        
        
        # Define regex pattern (adjust based on spacing issues)
        pattern = re.compile(r"(\d+)\s+(\d{8})\s+(\d{8})\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+(.+?)\s+([A-Za-z-]+)\s+(\w+)")

        # Extract data
        parsed_data = [pattern.match(line).groups() for line in content_lines[2:] if pattern.match(line)]  # Skip the header
        # Display the first rows
        # print(df_temp.head())
        # Create DataFrame from parsed_data and use the detected column names
        df_temp = pd.DataFrame(parsed_data)
        if len(df_temp.columns) == len(df_columns):
            df_temp.columns = df_columns
        else:
            # Fallback if the number of columns doesn't match
            df_temp.columns = ["station_id", "start_date", "end_date", "height", "lat", "lon", "name", "state", "type"]
        
        print(df_temp.iloc[:2, :2])
        # df_temp = pd.read_table(io.StringIO(response.text), skiprows=2, encoding='latin1')
        # Now use the correct columns
        stations = df_temp.iloc[:, [id_col, lat_col, lon_col]]
        stations.columns = ["station_id", "lat", "lon"]
        # convert lat/lon to float
        stations["lat"] = stations["lat"].astype(float)
        stations["lon"] = stations["lon"].astype(float)
        
    except Exception as e:
        print(f"Error parsing station data: {e}")
        print("Trying alternative approach with fixed column structure...")
        try:
            # Use a simpler approach with fixed column structure
            stations = pd.read_csv(io.StringIO(response.text), sep="\s+", header=None, skiprows=2, encoding='latin1')
            # Use the first column as station_id and whatever columns are available for lat/lon
            if len(stations.columns) >= 3:
                stations = stations.iloc[:, [0, 1, 2]]
                stations.columns = ["station_id", "lat", "lon"]
            else:
                print(f"Not enough columns in file (found {len(stations.columns)})")
                return None
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")
            return None
    
    # Continue with distance calculation
    stations["distance"] = ((stations["lat"] - attendorn_lat)**2 + (stations["lon"] - attendorn_lon)**2)**0.5
    closest_station = stations.loc[stations["distance"].idxmin()]
    print(f"Selected station {closest_station['station_id']} at distance {closest_station['distance']:.4f}")
    return str(closest_station["station_id"])

# Fix timestamp format for solar data
def parse_solar_timestamp(timestamp):
    """Handles timestamps in solar data that contain HH:MM instead of just HH."""
    try:
        return datetime.strptime(timestamp, "%Y%m%d%H:%M")
    except ValueError:
        return pd.NaT  # If parsing fails, return NaT (Not a Time)
    
# Function to download and process data for a given parameter
def get_weather_data(parameter, station_id):
    param_info = parameters[parameter]
    if param_info["recent_folder"]:
        url = f"{base_url}/{parameter}/recent/stundenwerte_{param_info['prefix']}_{station_id}_akt.zip"
    else:
        url = f"{base_url}/{parameter}/stundenwerte_{param_info['prefix']}_{station_id}_akt.zip"

    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find the data file within the ZIP archive
            data_file = [name for name in z.namelist() if "produkt" in name][0]
            with z.open(data_file) as f:
                df = pd.read_csv(f, sep=";", encoding="latin1", na_values="-999")
                print(f"zip-data: {df.head()}")
                
                # Convert the date column differently for solar data
                if "MESS_DATUM" in df.columns:
                    if parameter == "solar":
                        # Apply the special timestamp parsing function for solar data
                        df["MESS_DATUM"] = df["MESS_DATUM"].apply(parse_solar_timestamp)
                    else:
                        df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", errors="coerce")
                
                # Filter by date range
                df = df[(df["MESS_DATUM"] >= start_date) & (df["MESS_DATUM"] <= end_date)]
                return df
    else:
        print(f"Data for {parameter} not available for station {station_id}.")
        return pd.DataFrame()

# Main execution
# Main execution
def main():
    all_data = pd.DataFrame()
    
    for param in parameters.keys():
        print(f"Processing {parameters[param]['name']} data...")
        
        station_id = get_closest_station(param)
        if not station_id:
            print(f"No valid station found for {param}, skipping...")
            continue
        
        data = get_weather_data(param, station_id)
        
        if not data.empty:
            # Ensure we keep all relevant columns
            relevant_columns = [col for col in data.columns if col != "MESS_DATUM"]  # All except MESS_DATUM
            relevant_columns.insert(0, "MESS_DATUM")  # Keep date column at the beginning
            
            data = data[relevant_columns]
            
            # Rename columns to include parameter type to avoid duplicate names
            new_column_names = {"MESS_DATUM": "timestamp"}
            new_column_names.update({col: f"{param}_{col}" for col in data.columns if col != "MESS_DATUM"})
            
            data.rename(columns=new_column_names, inplace=True)
            
            if all_data.empty:
                all_data = data
            else:
                all_data = pd.merge(all_data, data, on="timestamp", how="outer")
    
    if not all_data.empty:
        all_data["timestamp"] = pd.to_datetime(all_data["timestamp"])
        all_data.set_index("timestamp", inplace=True)
        all_data.sort_index(inplace=True)
        
        # Save to CSV or Parquet
        all_data.to_csv("attendorn_hourly_weather_data.csv")
        # all_data.to_parquet("attendorn_hourly_weather_data.parquet")
        
        print(f"Weather data successfully saved with {len(all_data)} rows.")
    else:
        print("No weather data was retrieved.")

if __name__ == "__main__":
    main()