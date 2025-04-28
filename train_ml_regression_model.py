import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import joblib

def read_data(data_path):
    # Read the data from the parquet file
    data = pd.read_parquet(data_path)
    return data

def train_model(data):
    import pandas as pd

    # Prepare data
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.date)
    df = df.set_index("timestamp").sort_index()
    features = ['radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 
                'sunshine_duration', 'sun_zenith_angle', 'temperature_air_mean_2m', 
                'cloud_cover_total', 'humidity']
    X = df[features]
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

def test_models(data):
    # Test your models here
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.date)
    df = df.set_index("timestamp").sort_index()
    highly_important_features = ['date', 'radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 'sunshine_duration', 'sun_zenith_angle', 'temperature_air_mean_2m', 'cloud_cover_total', 'humidity']
    features = ['radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 'sunshine_duration', 'sun_zenith_angle', 'temperature_air_mean_2m', 'cloud_cover_total', 'humidity']

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
    
def test_predict(model, data):
    # Test your model here
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.date)
    df = df.set_index("timestamp").sort_index()
    features = ['radiation_global', 'radiation_sky_long_wave', 'radiation_sky_short_wave_diffuse', 'sunshine_duration', 'sun_zenith_angle', 'temperature_air_mean_2m', 'cloud_cover_total', 'humidity']

    X = df[features]
    y = df["solar"]

    preds = model.predict(X)
    rmse = root_mean_squared_error(y, preds)
    print(f"RMSE: {rmse}")
    df['predicted'] = preds
    df["predicted"] = df["predicted"].clip(lower=0)
    df[["date", "solar", "predicted"]].set_index("date").loc["2025-03-20":"2025-03-29"].plot(title="Prediction vs Actual")
    plt.savefig("data/prediction_vs_actual.png")
    
def main():
    # Read the data
    data = read_data("data/merged_data.parquet")
    # Test the models
    test_models(data)
    model = train_model(data)
    joblib.dump(model, "data/model.pkl")
    # test prediction
    test_predict(model, data)
    
    
if __name__ == "__main__":
    main()