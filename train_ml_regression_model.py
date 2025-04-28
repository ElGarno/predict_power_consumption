import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

def read_data(data_path):
    # Read the data from the parquet file
    data = pd.read_parquet(data_path)
    return data

def train_model(data):
    # Train your model here
    #Use Random Forest to predict solar power generation
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.date)
    df = df.set_index("timestamp").sort_index()
    highly_important_features = ['date', 'radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 'sunshine_duration', 'sun_zenith_angle', 'temperature_air_mean_2m', 'cloud_cover_total', 'humidity']
    features = ['radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 'sunshine_duration', 'temperature_air_mean_2m', 'cloud_cover_total', 'humidity']
    X = df[features]
    y = df["solar"]
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(n_estimators=40, random_state=42)
    # predict using Randowm Forest
    rmse = []
    for train_idx, val_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmse.append(root_mean_squared_error(y.iloc[val_idx], preds))
    rmse_mean = np.mean(rmse)
    print(f"RMSE: {rmse_mean}")
    # plot prediction vs actual for 23.03.2025 - 25.03.2025
    df["predicted"] = model.predict(X)
    df["predicted"] = df["predicted"].clip(lower=0)
    # df["predicted"] = df["predicted"].clip(upper=10000)
    # plt.figure(figsize=(12, 6))
    df[["date", "solar", "predicted"]].set_index("date").loc["2025-03-20":"2025-03-25"].plot(title="Prediction vs Actual")
    # save the plot
    plt.savefig("data/prediction_vs_actual.png")
    plt.show()
    
    return model

def test_models(data):
    # Test your models here
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.date)
    df = df.set_index("timestamp").sort_index()
    highly_important_features = ['date', 'radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 'sunshine_duration', 'sun_zenith_angle', 'temperature_air_mean_2m', 'cloud_cover_total', 'humidity']
    features = ['radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 'sunshine_duration', 'temperature_air_mean_2m', 'cloud_cover_total', 'humidity']

    X = df[features]
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
    
    
def main():
    # Read the data
    data = read_data("data/merged_data.parquet")
    # Test the models
    test_models(data)
    model = train_model(data)
    
if __name__ == "__main__":
    main()