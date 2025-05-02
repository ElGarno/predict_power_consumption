from wetterdienst.provider.dwd.observation import DwdObservationRequest
from wetterdienst import Settings
import pandas as pd



def get_data_by_api(settings, parameters):
    df_list = []
    for parameter in parameters:
        request = DwdObservationRequest(
            parameters=parameter,
            start_date="2024-12-15",
            end_date="2025-04-27",
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
            
    # Main execution
def main():
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
    # # Save the pivoted DataFrame to a CSV file
    # df_climate_pivot.to_csv("data/attendorn_hourly_weather_data_api.csv", index=False)
    # Save the pivoted DataFrame to a Parquet file
    df_climate_pivot.to_parquet("data/attendorn_hourly_weather_data_api.parquet", index=False)
    print(f"Weather data successfully saved with {len(df_climate_pivot)} rows.")
    

if __name__ == "__main__":
    main()