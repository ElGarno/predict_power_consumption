import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import joblib

import polars as pl
from datetime import datetime, timedelta
from wetterdienst import Settings
from wetterdienst.provider.dwd.mosmix import DwdForecastDate, DwdMosmixRequest
import plotly.express as px

def read_data(data_path):
    # Read the data from the parquet file
    data = pd.read_parquet(data_path)
    return data

def train_model(data, features_prediction_only):
    import pandas as pd

    # Prepare data
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.date)
    df = df.set_index("timestamp").sort_index()
    X = df[features_prediction_only]
    y = df["solar"]

    # Split off final test set (last 10%)
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Cross-validation on training data
    tscv = TimeSeriesSplit(n_splits=5)
    model_cv = RandomForestRegressor(n_estimators=40, random_state=42)
    
    rmse_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        model_cv.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model_cv.predict(X_train.iloc[val_idx])
        rmse = root_mean_squared_error(y_train.iloc[val_idx], preds)
        rmse_scores.append(rmse)

    print(f"Cross-Validation RMSE (on training set): {np.mean(rmse_scores)}")

    # Final model trained on 90% training data
    final_model = RandomForestRegressor(n_estimators=40, random_state=42)
    final_model.fit(X_train, y_train)

    # Optional: Evaluate final model on 10% holdout
    test_preds = final_model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    print(f"Final Test Set RMSE: {test_rmse}")

    return final_model

def test_models(data, features_prediction_only):
    # Test your models here
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.date)
    df = df.set_index("timestamp").sort_index()
    X = df[features_prediction_only]
    y = df["solar"]

    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1),
        "RandomForest": RandomForestRegressor(n_estimators=40, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    results = {}
    for name, model in models.items():
        rmses = []
        for train_idx, val_idx in tscv.split(X):
            # print(f"train_idx: {train_idx}, val_idx: {val_idx}")
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            rmses.append(root_mean_squared_error(y.iloc[val_idx], preds))
        results[name] = np.mean(rmses)

    print(pd.Series(results).sort_values().rename("CV_RMSE"))
    
def test_predict(model, data, features_prediction_only):
    # Test your model here
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.date)
    df = df.set_index("timestamp").sort_index()
    X = df[features_prediction_only]
    y = df["solar"]

    preds = model.predict(X)
    rmse = root_mean_squared_error(y, preds)
    print(f"RMSE: {rmse}")
    df['predicted'] = preds
    df["predicted"] = df["predicted"].clip(lower=0)
    df[["date", "solar", "predicted"]].set_index("date").loc["2025-03-20":"2025-03-29"].plot(title="Prediction vs Actual")
    plt.savefig("data/prediction_vs_actual_lessFeatures.png")
    
def get_forecast_data(data, features_prediction_only):
    # get the data from https://api.open-meteo.com/v1/dwd-icon?latitude=51.12&longitude=7.90&hourly=shortwave_radiation,diffuse_radiation,sunshine_duration,temperature_2m,cloud_cover,relative_humidity_2m&timezone=Europe%2FBerlin
    import requests

    # Define the API endpoint and parameters
    url = "https://api.open-meteo.com/v1/dwd-icon"
    params = {
        "latitude": 51.14,
        "longitude": 7.92,
        "hourly": "shortwave_radiation,diffuse_radiation,sunshine_duration,temperature_2m,cloud_cover,relative_humidity_2m",
        "timezone": "Europe/Berlin"
    }

    # Make the API request
    response = requests.get(url, params=params)
    data = response.json()

    # Extract the hourly data
    hourly_data = data["hourly"]
    # Create the DataFrame
    df_forecast2 = pd.DataFrame(hourly_data)
    # Convert the 'time' column to datetime format
    df_forecast2["time"] = pd.to_datetime(df_forecast2["time"])
    
    # feature mapping
    feature_mapping = {
        "shortwave_radiation": "radiation_global",
        "diffuse_radiation": "radiation_sky_short_wave_diffuse",
        "sunshine_duration": "sunshine_duration",
        "temperature_2m": "temperature_air_mean_2m",
        "cloud_cover": "cloud_cover_total",
        "relative_humidity_2m": "humidity"
    }
    # Rename the columns in df_forecast2
    df_forecast2.rename(columns=feature_mapping, inplace=True)
    
    # use only the forecast for the next day
    df_forecast2 = df_forecast2[df_forecast2["time"].dt.date == tomorrow]

    # Check what columns are actually available in the result
    print("\nAvailable columns in old:", df_forecast.columns.tolist())

    # Get only the columns that exist in the dataframe
    available_features = [col for col in features_prediction_only if col in df_forecast.columns]

    # Create plot with only available columns
    if 'date' in df_forecast.columns and len(available_features) > 0:
        fig = px.line(df_forecast, x='date', y=available_features, 
                    title=f"Weather forecast for {tomorrow}")
        fig.show()
    else:
        print("Required columns not found in the dataframe. Available columns:", df_forecast.columns)
        
    # same for new forecast
    # Check what columns are actually available in the result
    print("\nAvailable columns in new:", df_forecast2.columns.tolist())
    available_features_new = [col for col in features_prediction_only if col in df_forecast2.columns]
    if 'time' in df_forecast2.columns and len(available_features_new) > 0:
        fig = px.line(df_forecast2, x='time', y=available_features_new, 
                    title=f"Weather forecast for {tomorrow}")
        fig.show()
    else:
        print("Required columns not found in the dataframe. Available columns:", df_forecast2.columns)    
    
    # get solar prediction with model
    df_forecast2["predicted"] = model.predict(df_forecast2[available_features_new])
    df_forecast2["predicted"] = df_forecast2["predicted"].clip(lower=0)
    # plot the prediction
    if 'time' in df_forecast2.columns:
        fig = px.line(df_forecast2, x='time', y='predicted', 
                    title=f"Solar prediction for {tomorrow}")
        fig.show()
    else:
        print("Date column not found in the dataframe. Available columns:", df_forecast2.columns)

    
def predict_tomorrow(model, data, features_prediction_only):
    def next_day_mosmix(tomorrow: datetime, features: list[str]):
        # 1. Compute start/end datetimes for midnight→midnight UTC
        start = datetime(tomorrow.year, tomorrow.month, tomorrow.day)
        end   = start + timedelta(days=1)

        # 2. Settings: wide table with humanized, non-SI names
        settings = Settings(ts_shape="wide", ts_humanize=True)

        # 3. Request only the 10-day forecast from the latest run, 
        #    but *limit* it to tomorrow via start_date/end_date
        # Use a list of parameters instead of None
        parameters = []
        for feature in features:
            parameters.append(("hourly", "large", feature))
        
        request = DwdMosmixRequest(
            parameters=parameters,           # Specify parameters to fetch
            issue=DwdForecastDate.LATEST,    # latest model run
            start_date=start,                # ← here
            end_date=end,                    # ← and here
            settings=settings,
        )

        # 4. Restrict to your stations
        attendorn = (51.1279, 7.9022)
        stations = request.filter_by_rank(latlon=attendorn, rank=5)

        # 5. Pull the first (and only) result
        response = next(stations.values.query())
        
        # 6. Convert to pandas DataFrame for easier handling with plotly
        df = response.df.to_pandas()
        # plot df and save as png
        df.plot(title="Tomorrow's forecast")
        plt.savefig("data/forecast.png")
        plt.show()
        
        return df

    # Tomorrow in UTC
    tomorrow = datetime.now().date() + timedelta(days=1)

    # Get forecast data
    df_forecast = next_day_mosmix(tomorrow, features_prediction_only)
    
    # other api
    # get the data from https://api.open-meteo.com/v1/dwd-icon?latitude=51.12&longitude=7.90&hourly=shortwave_radiation,diffuse_radiation,sunshine_duration,temperature_2m,cloud_cover,relative_humidity_2m&timezone=Europe%2FBerlin
    import requests

    # Define the API endpoint and parameters
    url = "https://api.open-meteo.com/v1/dwd-icon"
    params = {
        "latitude": 51.14,
        "longitude": 7.92,
        "hourly": "shortwave_radiation,diffuse_radiation,sunshine_duration,temperature_2m,cloud_cover,relative_humidity_2m",
        "timezone": "Europe/Berlin"
    }

    # Make the API request
    response = requests.get(url, params=params)
    data = response.json()

    # Extract the hourly data
    hourly_data = data["hourly"]
    # Create the DataFrame
    df_forecast2 = pd.DataFrame(hourly_data)
    # Convert the 'time' column to datetime format
    df_forecast2["time"] = pd.to_datetime(df_forecast2["time"])
    
    # feature mapping
    feature_mapping = {
        "shortwave_radiation": "radiation_global",
        "diffuse_radiation": "radiation_sky_short_wave_diffuse",
        "sunshine_duration": "sunshine_duration",
        "temperature_2m": "temperature_air_mean_2m",
        "cloud_cover": "cloud_cover_total",
        "relative_humidity_2m": "humidity"
    }
    # Rename the columns in df_forecast2
    df_forecast2.rename(columns=feature_mapping, inplace=True)
    
    # use only the forecast for the next day
    df_forecast2 = df_forecast2[df_forecast2["time"].dt.date == tomorrow]

    # Check what columns are actually available in the result
    print("\nAvailable columns in old:", df_forecast.columns.tolist())

    # Get only the columns that exist in the dataframe
    available_features = [col for col in features_prediction_only if col in df_forecast.columns]

    # Create plot with only available columns
    if 'date' in df_forecast.columns and len(available_features) > 0:
        fig = px.line(df_forecast, x='date', y=available_features, 
                    title=f"Weather forecast for {tomorrow}")
        fig.show()
    else:
        print("Required columns not found in the dataframe. Available columns:", df_forecast.columns)
        
    # same for new forecast
    # Check what columns are actually available in the result
    print("\nAvailable columns in new:", df_forecast2.columns.tolist())
    available_features_new = [col for col in features_prediction_only if col in df_forecast2.columns]
    if 'time' in df_forecast2.columns and len(available_features_new) > 0:
        fig = px.line(df_forecast2, x='time', y=available_features_new, 
                    title=f"Weather forecast for {tomorrow}")
        fig.show()
    else:
        print("Required columns not found in the dataframe. Available columns:", df_forecast2.columns)    
    
    # get solar prediction with model
    df_forecast2["predicted"] = model.predict(df_forecast2[available_features_new])
    df_forecast2["predicted"] = df_forecast2["predicted"].clip(lower=0)
    # plot the prediction
    if 'time' in df_forecast2.columns:
        fig = px.line(df_forecast2, x='time', y='predicted', 
                    title=f"Solar prediction for {tomorrow}")
        fig.show()
    else:
        print("Date column not found in the dataframe. Available columns:", df_forecast2.columns)
    
    
def main():
    # Read the data
    data = read_data("data/merged_data.parquet")
    # Features
    features = ['radiation_global', 'radiation_sky_short_wave_diffuse', 
                'sunshine_duration', 'temperature_air_mean_2m', 
                'cloud_cover_total', 'humidity']
    features_prediction_only = ['sunshine_duration', 'temperature_air_mean_2m', 'cloud_cover_total']
    # Test the models
    test_models(data, features)
    model = train_model(data, features)
    joblib.dump(model, "data/model_2.pkl")
    # test prediction
    test_predict(model, data, features)
    get_forcast_data
    # predict tomorrow
    predict_tomorrow(model, data, features)
    
if __name__ == "__main__":
    main()