import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

def read_data(data_path):
    # Read the data from the parquet file
    data = pd.read_parquet(data_path)
    return data

def train_model(data):
    # Train your model here
    pass

def test_models(data):
    # Test your models here
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.timestamp)
    df = df.set_index("timestamp").sort_index()
    features = ['global_solar_Jcm2', 'diffuse_solar_Jcm2', 'sunshine_duration_min', 'temp_c', 'cloud_cover_8ths', 'humidity_pct']
    X = df[features]
    y = df["solar"]

    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
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
    
if __name__ == "__main__":
    main()