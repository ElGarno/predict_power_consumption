import pandas as pd
import numpy as np
import asyncio
from dotenv import load_dotenv
import http.client
import urllib.parse
import requests
from prepare_data import get_merged_data_for_training, export_to_parquet
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

import polars as pl
from datetime import datetime, timedelta
import plotly.express as px

def send_pushover_notification_new(user, message):
    load_dotenv()
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    pushover_api_token = os.getenv("PUSHOVER_TAPO_API_TOKEN")
    conn.request("POST", "/1/messages.json",
                 urllib.parse.urlencode({
                     "token": pushover_api_token,
                     "user": user,
                     "message": message,
                 }), {"Content-type": "application/x-www-form-urlencoded"})
    conn.getresponse()
    
def send_pushover_notification_img(user, message, image_path="data/prediction_overproduction.png"):
    load_dotenv()
    pushover_api_token = os.getenv("PUSHOVER_TAPO_API_TOKEN")
    
    data = {
        "token": pushover_api_token,
        "user": user,
        "message": message,
    }

    files = {"attachment": open(image_path, "rb")} if image_path else None

    response = requests.post("https://api.pushover.net/1/messages.json", data=data, files=files)
    response.raise_for_status() 

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

# def test_models(data, features_prediction_only):
#     # Test your models here
#     df = data.copy()
#     df["timestamp"] = pd.to_datetime(df.date)
#     df = df.set_index("timestamp").sort_index()
#     X = df[features_prediction_only]
#     y = df["solar"]

#     tscv = TimeSeriesSplit(n_splits=5)
#     models = {
#         "Linear": LinearRegression(),
#         "Ridge": Ridge(alpha=1),
#         "RandomForest": RandomForestRegressor(n_estimators=40, random_state=42),
#         "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
#     }
    
#     print(f"X shape: {X.shape}, y shape: {y.shape}")

#     results = {}
#     for name, model in models.items():
#         rmses = []
#         for train_idx, val_idx in tscv.split(X):
#             # print(f"train_idx: {train_idx}, val_idx: {val_idx}")
#             model.fit(X.iloc[train_idx], y.iloc[train_idx])
#             preds = model.predict(X.iloc[val_idx])
#             rmses.append(root_mean_squared_error(y.iloc[val_idx], preds))
#         results[name] = np.mean(rmses)

#     print(pd.Series(results).sort_values().rename("CV_RMSE"))
    
# def test_predict(model, data, features_prediction_only):
#     # Test your model here
#     df = data.copy()
#     df["timestamp"] = pd.to_datetime(df.date)
#     df = df.set_index("timestamp").sort_index()
#     X = df[features_prediction_only]
#     y = df["solar"]

#     preds = model.predict(X)
#     rmse = root_mean_squared_error(y, preds)
#     print(f"RMSE: {rmse}")
#     df['predicted'] = preds
#     df["predicted"] = df["predicted"].clip(lower=0)
#     df[["date", "solar", "predicted"]].set_index("date").loc["2025-03-20":"2025-03-29"].plot(title="Prediction vs Actual")
#     plt.savefig("data/prediction_vs_actual_lessFeatures.png")
    
def get_forecast_data():
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
    df_forecast = pd.DataFrame(hourly_data)
    # Convert the 'time' column to datetime format
    df_forecast["time"] = pd.to_datetime(df_forecast["time"])
    
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
    df_forecast.rename(columns=feature_mapping, inplace=True)
    # Tomorrow in UTC
    tomorrow = datetime.now().date() + timedelta(days=1)
    # use only the forecast for the next day
    df_forecast = df_forecast[df_forecast["time"].dt.date == tomorrow]
        
    return df_forecast

    
def predict_tomorrow(model, weather_forecast, features_prediction_only, plot_images=False):

    # Check what columns are actually available in the result
    print("\nAvailable columns:", weather_forecast.columns.tolist())
    available_features = [col for col in features_prediction_only if col in weather_forecast.columns]
    # Tomorrow in UTC
    tomorrow = datetime.now().date() + timedelta(days=1)
    if plot_images:
        if 'time' in weather_forecast.columns and len(available_features) > 0:
            fig = px.line(weather_forecast, x='time', y=available_features, 
                        title=f"Weather forecast for {tomorrow}")
            fig.show()
        else:
            print("Required columns not found in the dataframe. Available columns:", weather_forecast.columns)    
    
    # get solar prediction with model
    weather_forecast["predicted"] = model.predict(weather_forecast[available_features])
    weather_forecast["predicted"] = weather_forecast["predicted"].clip(lower=0)
    # plot the prediction
    if plot_images:
        if 'time' in weather_forecast.columns:
            fig = px.line(weather_forecast, x='time', y='predicted', 
                        title=f"Solar prediction for {tomorrow}")
            fig.show()
        else:
            print("Date column not found in the dataframe. Available columns:", weather_forecast.columns)
    return weather_forecast


def compute_enegery_overproduction(df, threshold=20000):
    overproduction_df = df.copy()
    # Compute the energy overproduction
    overproduction_df["overproduction"] = overproduction_df["predicted"] - threshold
    # what are suitable times to consume energy?
    overproduction_df["overproduction"] = overproduction_df["overproduction"].clip(lower=0)
    # get the time of the overproduction
    overproduction_df["overproduction_time"] = overproduction_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    overproduction_df = overproduction_df[overproduction_df["overproduction"] > 0]
    
    # Tomorrow in UTC
    tomorrow = datetime.now().date() + timedelta(days=1)
    # create a matplotlib plot where the x-axis is the time and the y-axis is the predicted value and the threshold is 20000.
    # if the predicted value is greater than 20000, then the value is red, otherwise it is green so plot all in one plot
    fig, ax = plt.subplots()
    # Extract hour from datetime for x-axis
    hours = df["time"].dt.hour
    
    ax.plot(hours, df["predicted"], label="Predicted", color="blue")
    ax.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    ax.fill_between(hours, threshold, df["predicted"], 
                    where=(df["predicted"] > threshold), color='green', alpha=0.5)
    ax.fill_between(hours, threshold, df["predicted"], 
                    where=(df["predicted"] <= threshold), color='red', alpha=0.5)
    ax.set_title(f"Solar prediction for {tomorrow}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Predicted Power (W)")
    ax.grid(True, linestyle='--', alpha=0.7)  # Add grid lines
    plt.legend()
    # plt.show()
    # Save the plot
    plt.savefig("data/prediction_overproduction.png")
    return overproduction_df[["overproduction_time", "overproduction"]].reset_index(drop=True)


async def get_predicted_power(merged_data_file_path, update_data=True, get_power_from_db=False, get_weather_data_from_api=True, update_model=True):
    while True:
        if update_data:
            # Get the merged data for training
            merged_data = get_merged_data_for_training(get_power_from_db=get_power_from_db, get_weather_data_from_api=get_weather_data_from_api)
            # export the data to parquet
            export_to_parquet(merged_data, path="data")
        else:
            merged_data = read_data(merged_data_file_path)
        # Features
        features = ['radiation_global', 'radiation_sky_short_wave_diffuse', 
                    'sunshine_duration', 'temperature_air_mean_2m', 
                    'cloud_cover_total', 'humidity']
        # # Test the models
        # test_models(data, features)
        if update_model:
            # Train the model
            model = train_model(merged_data, features)
            # Save the model
            joblib.dump(model, "data/model_weather_solar_power.pkl")
        else:
            # Load the model
            model = joblib.load("data/model_weather_solar_power.pkl")
        # # test prediction
        # test_predict(model, data, features)
        weather_forecast = get_forecast_data()
        # predict tomorrow
        forecast_data = predict_tomorrow(model, weather_forecast, features)
        # The predicted column is the energy production in W for the next day per hour
        # if predicted value is > 20000 W then we produce more than we consume
        df_overproduction = compute_enegery_overproduction(forecast_data, threshold=20000)
        output_string = f"Overproduction for tomorrow: {df_overproduction.to_string(index=False)}"
        print(output_string)
        # send pushover notification
        pushover_user_group = os.getenv("PUSHOVER_USER_GROUP_WOERIS")
        if (datetime.now().hour == 21) and (datetime.now().minute <= 10):
            send_pushover_notification_img(pushover_user_group, output_string, "data/prediction_overproduction.png")
        await asyncio.sleep(600)
    
    
async def main():
    # Read the data
    path = "data"
    merged_data_file = "merged_data_weather_power.parquet"
    merged_data_file_path = os.path.join(path, merged_data_file)
    
    update_data = True
    get_power_from_db = False
    get_weather_data_from_api = True
    
    update_model = True
    await get_predicted_power(merged_data_file_path, update_data=update_data, get_power_from_db=get_power_from_db, get_weather_data_from_api=get_weather_data_from_api, update_model=update_model)
        
    
if __name__ == "__main__":
    asyncio.run(main())
    
