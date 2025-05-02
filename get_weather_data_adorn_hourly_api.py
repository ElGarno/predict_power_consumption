from wetterdienst.provider.dwd.observation import DwdObservationRequest
from wetterdienst import Settings
import pandas as pd
import os



def get_data_by_api(settings, parameters):
    df_list = []
    for parameter in parameters:
        request = DwdObservationRequest(
            parameters=parameter,
            start_date="2024-12-15",  # Starting where also the power consumption data starts
            end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),  # Current date
            settings=settings,
        )
        attendorn = (51.1279, 7.9022)
        stations = request.filter_by_rank(latlon=attendorn, rank=5)

        # Query data all together
        df = stations.values.all().df.drop_nulls()
        df_list.append(df)
    # Combine all dataframes into one
    df_list_df = []
    for i in range(len(df_list)):
        i_df = pd.DataFrame(df_list[i])
        df_list_df.append(i_df)
    df_climate = pd.concat(df_list_df)
    df_climate.columns =stations.values.all().df.drop_nulls().columns
    return df_climate


def get_pivot_df(df):
    # Extract datetime from the date column (assuming it's in column 4 based on earlier cells)
    df_climate_processed = df.copy()

    # Create the pivot table
    df_climate_pivot = df_climate_processed.pivot_table(
        index='date',
        columns='parameter',
        values='value',
        aggfunc='mean'  # Use mean in case there are multiple values for the same parameter and date
    )

    # Reset index to make date a column again
    df_climate_pivot = df_climate_pivot.reset_index()
    # drop rows where all values are missing
    df_climate_cleaned = df_climate_pivot.dropna(how='any')
    # drop any column which contains only the same value
    df_climate_cleaned = df_climate_cleaned.loc[:, df_climate_cleaned.nunique() != 1]
    return df_climate_cleaned
            
def export_to_parquet(df, path="data"):
    # Export DataFrame to Parquet
    parquet_filename = f"attendorn_hourly_weather_data_api.parquet"
    parquet_filepath = os.path.join(path, parquet_filename)
    df.to_parquet(parquet_filepath, index=False)
    print(f"Weather data successfully exported with {len(df)} rows to {parquet_filename}.")
    

def get_weather_data_pivot():
    # if no settings are provided, default settings are used which are
    # Settings(ts_shape="long", ts_humanize=True, ts_si_units=True)
    settings = Settings(ts_skip_empty=True)
    parameters = [
        ("hourly", "air_temperature"),
        ("hourly", "precipitation"),
        ("hourly", "wind"),
        ("hourly", "cloudiness"),
        ("hourly", "solar"),
    ]
    
    df_climate = get_data_by_api(settings, parameters)
    df_climate_pivot = get_pivot_df(df_climate)
    return df_climate_pivot
    
    
# Main execution
if __name__ == "__main__":
    df_climate_pivot = get_weather_data_pivot()
    export_path = "data"
    export_to_parquet(df_climate_pivot, export_path)
    