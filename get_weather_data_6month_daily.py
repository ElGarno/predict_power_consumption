import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import zipfile
import io
import glob
import pyarrow.parquet as pq

# Define the time period (last 6 months)
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

# Find the closest weather station to Attendorn
def get_closest_station():
    # DWD station list URL
    stations_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/recent/KL_Tageswerte_Beschreibung_Stationen.txt"
    
    # Attendorn coordinates (approximate)
    attendorn_lat, attendorn_lon = 51.1279, 7.9022
    
    # Download station list
    response = requests.get(stations_url)
    
    # Skip header lines and parse stations
    stations = []
    for line in response.text.splitlines()[2:]:  # Skip first two header lines
        if line.strip():
            parts = line.split()
            if len(parts) >= 7:
                try:
                    station_id = parts[0]
                    lat = float(parts[4])
                    lon = float(parts[5])
                    height = parts[6]
                    name = " ".join(parts[7:])
                    
                    # Calculate simple distance (this is approximate)
                    dist = ((lat - attendorn_lat) ** 2 + (lon - attendorn_lon) ** 2) ** 0.5
                    stations.append((station_id, name, lat, lon, height, dist))
                except:
                    continue
    
    # Sort by distance and return closest
    stations.sort(key=lambda x: x[5])
    return stations[0]  # Return the closest station

# Get weather data for a station
def get_weather_data(station_id):
    # Base URL for daily climate data
    base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily"
    
    # Parameters of interest for PV prediction
    parameters = {
        "kl": "Climate data",  # General climate data (temperature, precipitation, etc.)
        "solar": "Solar data"  # Solar radiation data
    }
    
    all_data = pd.DataFrame()
    
    for param, desc in parameters.items():
        print(f"Fetching {desc}...")
        
        # Check both recent and historical data
        for timespan in ["recent", "historical"]:
            url = f"{base_url}/{param}/{timespan}"
            
            # First get the list of available zip files
            try:
                response = requests.get(f"{url}/")
                if response.status_code != 200:
                    continue
                
                # Look for the station ID in the response
                for line in response.text.splitlines():
                    if station_id in line and ".zip" in line:
                        zip_filename = line.split('"')[1].split('"')[0]
                        if ">" in zip_filename:  # Handle HTML formatting
                            zip_filename = zip_filename.split(">")[1].split("<")[0]
                        
                        # Download the zip file
                        zip_url = f"{url}/{zip_filename}"
                        print(f"Downloading {zip_url}")
                        
                        try:
                            response = requests.get(zip_url)
                            z = zipfile.ZipFile(io.BytesIO(response.content))
                            
                            # Extract all files to a temp directory
                            temp_dir = f"temp_dwd_{param}"
                            os.makedirs(temp_dir, exist_ok=True)
                            z.extractall(temp_dir)
                            
                            # Look for data files (usually produkt_*.txt)
                            data_files = glob.glob(f"{temp_dir}/produkt_*.txt")
                            
                            for data_file in data_files:
                                # Read the data file
                                df = pd.read_csv(data_file, sep=";")
                                
                                # Convert date column
                                if "MESS_DATUM" in df.columns:
                                    df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d")
                                    
                                    # Filter for our date range
                                    df = df[(df["MESS_DATUM"] >= start_date) & (df["MESS_DATUM"] <= end_date)]
                                    
                                    if not df.empty:
                                        # Add parameter type
                                        df["PARAMETER_TYPE"] = param
                                        
                                        # Append to our collection
                                        if all_data.empty:
                                            all_data = df
                                        else:
                                            # Merge on date if data already exists
                                            all_data = pd.merge(all_data, df, on="MESS_DATUM", how="outer", suffixes=("", f"_{param}"))
                        except Exception as e:
                            print(f"Error processing {zip_url}: {str(e)}")
            except Exception as e:
                print(f"Error accessing {url}: {str(e)}")
    
    # Clean up temp files
    for param in parameters:
        temp_dir = f"temp_dwd_{param}"
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
    
    return all_data

# Function to merge weather data with your PV power data
def merge_with_pv_data(weather_data, pv_data_path):
    # Load your PV data from parquet
    pv_data = pq.read_table(pv_data_path).to_pandas()
    
    # Ensure datetime formats match
    # Adjust this based on your parquet file's timestamp format
    if "timestamp" in pv_data.columns:
        pv_data["timestamp"] = pd.to_datetime(pv_data["timestamp"])
        
        # Convert to daily data if needed
        pv_daily = pv_data.resample("D", on="timestamp").mean()
        pv_daily = pv_daily.reset_index()
        
        # Rename for merging
        pv_daily = pv_daily.rename(columns={"timestamp": "MESS_DATUM"})
        
        # Merge with weather data
        merged_data = pd.merge(weather_data, pv_daily, on="MESS_DATUM", how="inner")
        
        return merged_data
    else:
        print("Could not find timestamp column in PV data")
        return None

# Main execution
def main():
    print("Finding closest weather station to Attendorn...")
    station_id, name, lat, lon, height, dist = get_closest_station()
    print(f"Closest station: {name} (ID: {station_id}, Distance: {dist:.2f} degrees)")
    
    print(f"Fetching weather data for the last 6 months...")
    weather_data = get_weather_data(station_id)
    
    if not weather_data.empty:
        # Save the weather data
        weather_data.to_csv(f"attendorn_weather_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv", index=False)
        print(f"Weather data saved with {len(weather_data)} rows")
        
        # Example of merging with your PV data (uncomment and adjust path)
        # pv_data_path = "path/to/your/pv_data.parquet"
        # merged_data = merge_with_pv_data(weather_data, pv_data_path)
        # if merged_data is not None:
        #     merged_data.to_parquet("merged_pv_weather_data.parquet")
        #     print(f"Merged data saved with {len(merged_data)} rows")
    else:
        print("No weather data found for the specified period")

if __name__ == "__main__":
    main()