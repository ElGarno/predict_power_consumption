"""
ML model training and prediction module for solar power forecasting.

This module handles:
- Training RandomForest models on historical weather/power data
- Fetching weather forecasts from Open-Meteo API
- Predicting solar power production for next day
- Computing overproduction periods above consumption threshold
- Sending daily notifications with predictions
- Running continuously with graceful shutdown support

For NAS deployment, this runs as a long-lived process with configurable intervals.
"""

import sys
print("DEBUG: Starting imports...", file=sys.stderr, flush=True)

import pandas as pd
print("DEBUG: pandas imported", file=sys.stderr, flush=True)
import numpy as np
print("DEBUG: numpy imported", file=sys.stderr, flush=True)
import asyncio
import signal
import requests
print("DEBUG: stdlib imports done", file=sys.stderr, flush=True)

from sklearn.model_selection import TimeSeriesSplit
print("DEBUG: TimeSeriesSplit imported", file=sys.stderr, flush=True)
from sklearn.ensemble import RandomForestRegressor
print("DEBUG: RandomForestRegressor imported", file=sys.stderr, flush=True)
from sklearn.metrics import root_mean_squared_error
print("DEBUG: sklearn metrics imported", file=sys.stderr, flush=True)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for NAS
import matplotlib.pyplot as plt
print("DEBUG: matplotlib imported", file=sys.stderr, flush=True)

import joblib
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import pytz
print("DEBUG: remaining stdlib imports done", file=sys.stderr, flush=True)

from prepare_data import get_merged_data_for_training, export_merged_data
print("DEBUG: prepare_data imported", file=sys.stderr, flush=True)

from config import settings, logger
print("DEBUG: config imported", file=sys.stderr, flush=True)

from utils import (
    send_pushover_notification,
    read_parquet,
    validate_dataframe_not_empty,
    validate_required_columns,
    get_tomorrow_date,
    retry_with_backoff,
    send_awtrix_countdown,
    send_awtrix_forecast_summary
)
print("DEBUG: utils imported", file=sys.stderr, flush=True)

# Debug: Log that imports completed successfully
logger.info("All imports completed successfully")

# Global flag for graceful shutdown
shutdown_flag = False

# Debug: Log module initialization complete
logger.info("Module initialization complete, ready to start")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True


def train_model(
    data: pd.DataFrame,
    features: Optional[list[str]] = None
) -> RandomForestRegressor:
    """
    Train a RandomForest regression model for solar power prediction.

    Uses time-series cross-validation for robust evaluation and trains
    final model on configured train/test split.

    Args:
        data: Training data with features and 'solar' target column
        features: List of feature columns (default: from settings)

    Returns:
        RandomForestRegressor: Trained model

    Raises:
        ValueError: If data is missing required columns or is empty
    """
    features = features or settings.prediction_features

    logger.info(f"Training model with {len(data)} samples")

    # Validate data
    validate_dataframe_not_empty(data, "Training data")
    required_cols = features + ['date', 'solar']
    validate_required_columns(data, required_cols, "Training data")

    # Prepare data
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.date)
    df = df.set_index("timestamp").sort_index()
    X = df[features]
    y = df["solar"]

    logger.info(f"Training features: {features}")
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")

    # Split off final test set
    split_ratio = settings.model_train_test_split
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train/test split: {split_ratio:.0%} = {len(X_train)}/{len(X_test)} samples")

    # Cross-validation on training data
    tscv = TimeSeriesSplit(n_splits=settings.model_cv_splits)
    model_cv = RandomForestRegressor(
        n_estimators=settings.model_n_estimators,
        random_state=settings.model_random_state,
        n_jobs=-1  # Use all CPU cores
    )

    rmse_scores = []
    logger.info(f"Running {settings.model_cv_splits}-fold time-series cross-validation")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        model_cv.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model_cv.predict(X_train.iloc[val_idx])
        rmse = root_mean_squared_error(y_train.iloc[val_idx], preds)
        rmse_scores.append(rmse)
        logger.debug(f"Fold {fold}/{settings.model_cv_splits}: RMSE = {rmse:.2f}")

    mean_cv_rmse = np.mean(rmse_scores)
    logger.info(f"Cross-Validation RMSE: {mean_cv_rmse:.2f} (±{np.std(rmse_scores):.2f})")

    # Train final model on full training set
    logger.info("Training final model on full training set")
    final_model = RandomForestRegressor(
        n_estimators=settings.model_n_estimators,
        random_state=settings.model_random_state,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    test_preds = final_model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    logger.info(f"Final Test Set RMSE: {test_rmse:.2f}")

    # Log feature importances
    feature_importance = sorted(
        zip(features, final_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    logger.info("Feature importances: " + ", ".join([f"{name}={imp:.3f}" for name, imp in feature_importance]))

    return final_model


def get_forecast_data() -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo DWD-ICON API.

    Retrieves hourly forecast for tomorrow for the configured location,
    maps API fields to model feature names.

    Returns:
        pd.DataFrame: Tomorrow's hourly weather forecast

    Raises:
        ConnectionError: If API request fails
        ValueError: If response is invalid or missing data
    """
    logger.info(f"Fetching weather forecast for location ({settings.latitude}, {settings.longitude})")

    params = {
        "latitude": settings.latitude,
        "longitude": settings.longitude,
        "hourly": "shortwave_radiation,diffuse_radiation,sunshine_duration,temperature_2m,cloud_cover,relative_humidity_2m",
        "timezone": settings.timezone
    }

    try:
        response = requests.get(
            settings.weather_api_url,
            params=params,
            timeout=settings.http_timeout_seconds
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        raise ConnectionError(f"Weather API request timed out after {settings.http_timeout_seconds}s")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Weather API request failed: {e}")
    except ValueError as e:
        raise ValueError(f"Failed to parse weather API response as JSON: {e}")

    # Extract and validate hourly data
    if "hourly" not in data:
        raise ValueError("Weather API response missing 'hourly' data")

    hourly_data = data["hourly"]
    df_forecast = pd.DataFrame(hourly_data)

    # Convert time column to datetime
    df_forecast["time"] = pd.to_datetime(df_forecast["time"])

    # Map API field names to model feature names
    feature_mapping = {
        "shortwave_radiation": "radiation_global",
        "diffuse_radiation": "radiation_sky_short_wave_diffuse",
        "sunshine_duration": "sunshine_duration",
        "temperature_2m": "temperature_air_mean_2m",
        "cloud_cover": "cloud_cover_total",
        "relative_humidity_2m": "humidity"
    }
    df_forecast.rename(columns=feature_mapping, inplace=True)

    # Filter to tomorrow only
    tomorrow = get_tomorrow_date(settings.timezone)
    df_forecast = df_forecast[df_forecast["time"].dt.date == tomorrow]

    validate_dataframe_not_empty(df_forecast, "Tomorrow's forecast")
    logger.info(f"Retrieved forecast for {tomorrow}: {len(df_forecast)} hourly data points")

    return df_forecast


def predict_tomorrow(
    model: RandomForestRegressor,
    weather_forecast: pd.DataFrame,
    features: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Predict solar power production for tomorrow using weather forecast.

    Args:
        model: Trained RandomForest model
        weather_forecast: Tomorrow's weather forecast
        features: Feature names (default: from settings)

    Returns:
        pd.DataFrame: Forecast with 'predicted' power column added

    Raises:
        ValueError: If required features are missing
    """
    features = features or settings.prediction_features

    # Check which features are available
    available_features = [f for f in features if f in weather_forecast.columns]
    missing_features = set(features) - set(available_features)

    if missing_features:
        logger.warning(f"Missing features in forecast: {missing_features}")

    if not available_features:
        raise ValueError("No prediction features available in weather forecast")

    logger.debug(f"Predicting with features: {available_features}")

    # Generate predictions
    forecast = weather_forecast.copy()
    forecast["predicted"] = model.predict(forecast[available_features])

    # Clip negative predictions to zero (can't have negative power)
    forecast["predicted"] = forecast["predicted"].clip(lower=0)

    tomorrow = get_tomorrow_date(settings.timezone)
    total_predicted = forecast["predicted"].sum()
    logger.info(f"Solar prediction for {tomorrow}: {total_predicted/1000:.1f} kWh total, "
                f"peak {forecast['predicted'].max():.0f}W")

    return forecast


def compute_energy_overproduction(
    forecast_df: pd.DataFrame,
    threshold: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute periods when solar production exceeds consumption threshold.

    Also generates and saves a visualization plot.

    Args:
        forecast_df: Forecast with 'time' and 'predicted' columns
        threshold: Power threshold in watts (default: from settings)

    Returns:
        pd.DataFrame: Rows where overproduction occurs with columns:
            - overproduction_time: Timestamp string
            - overproduction: Watts above threshold

    Raises:
        ValueError: If required columns are missing
    """
    threshold = threshold or settings.overproduction_threshold_watts

    validate_required_columns(forecast_df, ['time', 'predicted'], "Forecast data")

    logger.info(f"Computing overproduction with threshold {threshold}W")

    df = forecast_df.copy()

    # Calculate overproduction
    df["overproduction"] = (df["predicted"] - threshold).clip(lower=0)
    df["overproduction_time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Filter to only overproduction periods
    overproduction_df = df[df["overproduction"] > 0]

    total_overproduction = overproduction_df["overproduction"].sum()
    num_hours = len(overproduction_df)

    logger.info(f"Overproduction found: {num_hours} hours, {total_overproduction/1000:.1f} kWh total")

    # Create visualization
    tomorrow = get_tomorrow_date(settings.timezone)

    fig, ax = plt.subplots(figsize=(12, 6))
    hours = df["time"].dt.hour

    # Plot prediction line
    ax.plot(hours, df["predicted"], label="Predicted Power", color="blue", linewidth=2)

    # Plot threshold line
    ax.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold}W)", linewidth=2)

    # Fill areas
    ax.fill_between(
        hours, threshold, df["predicted"],
        where=(df["predicted"] > threshold),
        color='green', alpha=0.3, label="Overproduction"
    )
    ax.fill_between(
        hours, 0, df["predicted"],
        where=(df["predicted"] <= threshold),
        color='orange', alpha=0.2, label="Normal Production"
    )

    ax.set_title(f"Solar Power Prediction for {tomorrow}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Predicted Power (W)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    ax.set_xlim(0, 23)

    # Save plot
    plot_path = settings.get_full_path(settings.prediction_plot_file)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved prediction plot to {plot_path}")

    return overproduction_df[["overproduction_time", "overproduction"]].reset_index(drop=True)


async def prediction_loop():
    """
    Main prediction loop that runs continuously.

    Supports two modes:
    - Daily: Runs prediction once per day at scheduled time
    - Interval: Runs prediction every N seconds (legacy mode)

    Also handles weekly model retraining if enabled.
    Respects graceful shutdown signal.
    """
    global shutdown_flag

    logger.info("Starting prediction loop")

    if settings.prediction_mode == "daily":
        logger.info(f"Mode: Daily prediction at {settings.daily_prediction_hour}:{settings.daily_prediction_minute:02d}")
        if settings.model_retrain_enabled:
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            logger.info(f"Model retraining: {days[settings.model_retrain_day]} at {settings.model_retrain_hour}:{settings.model_retrain_minute:02d}")
    else:
        logger.info(f"Mode: Interval prediction every {settings.prediction_interval_seconds}s")

    logger.info(f"Notification time: {settings.notification_hour}:{settings.notification_minute_window:02d}")

    # Track last execution times to avoid duplicate runs
    last_prediction_date = None
    last_retrain_date = None

    # Load model once at startup if it exists
    model = None
    if settings.model_path.exists():
        logger.info(f"Loading existing model from {settings.model_path}")
        try:
            load_start = datetime.now()
            model = joblib.load(settings.model_path)
            load_duration = (datetime.now() - load_start).total_seconds()
            logger.info(f"Model loaded successfully in {load_duration:.1f}s")
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}. Will train new one.")
            model = None

    # Log initial status
    if settings.prediction_mode == "daily":
        now = datetime.now(pytz.timezone(settings.timezone))
        next_prediction = f"{settings.daily_prediction_hour:02d}:{settings.daily_prediction_minute:02d}"
        logger.info(f"Service ready. Waiting for next prediction at {next_prediction} (currently {now.strftime('%H:%M')})")

    while not shutdown_flag:
        try:
            now = datetime.now(pytz.timezone(settings.timezone))
            current_date = now.date()
            current_day = now.weekday()  # 0=Monday, 6=Sunday

            # Log status in daily mode (once per hour to show it's alive)
            if settings.prediction_mode == "daily" and now.minute == 0:
                next_prediction = f"{settings.daily_prediction_hour:02d}:{settings.daily_prediction_minute:02d}"
                logger.info(f"Service running. Next prediction at {next_prediction} (currently {now.strftime('%H:%M')})")

            # Send AWTRIX countdown every 5 minutes in daily mode
            if settings.prediction_mode == "daily":
                # Calculate time until next prediction
                target_time = now.replace(
                    hour=settings.daily_prediction_hour,
                    minute=settings.daily_prediction_minute,
                    second=0,
                    microsecond=0
                )

                # If target time is in the past, move to tomorrow
                if target_time <= now:
                    target_time += timedelta(days=1)

                time_diff = target_time - now
                hours_until = int(time_diff.total_seconds() // 3600)
                minutes_until = int((time_diff.total_seconds() % 3600) // 60)

                # Send countdown to AWTRIX
                send_awtrix_countdown(hours_until, minutes_until)

            # Check if it's time for weekly model retraining
            should_retrain = (
                settings.model_retrain_enabled and
                model is not None and  # Only retrain if we have a model
                current_day == settings.model_retrain_day and
                now.hour == settings.model_retrain_hour and
                now.minute == settings.model_retrain_minute and
                last_retrain_date != current_date
            )

            if should_retrain:
                logger.info("=" * 60)
                logger.info(f"Weekly model retraining triggered at {now}")
                logger.info("=" * 60)
                try:
                    # Fetch fresh training data
                    logger.info("Fetching fresh training data from sources")
                    merged_data = get_merged_data_for_training(
                        get_power_from_db=settings.get_power_from_db,
                        get_weather_data_from_api=settings.get_weather_from_api
                    )
                    export_merged_data(merged_data)

                    # Train new model
                    logger.info("Training new model (this may take a while on NAS...)")
                    model = train_model(merged_data, settings.prediction_features)

                    # Save with compression to reduce file size and I/O time
                    save_start = datetime.now()
                    joblib.dump(model, settings.model_path, compress=3)
                    save_duration = (datetime.now() - save_start).total_seconds()
                    logger.info(f"Model saved to {settings.model_path} in {save_duration:.1f}s")

                    last_retrain_date = current_date
                    logger.info("Weekly model retraining completed successfully")
                except Exception as e:
                    logger.error(f"Model retraining failed: {e}", exc_info=True)

            # Check if it's time for prediction
            if settings.prediction_mode == "daily":
                should_predict = (
                    now.hour == settings.daily_prediction_hour and
                    now.minute == settings.daily_prediction_minute and
                    last_prediction_date != current_date
                )
            else:
                # Interval mode: always predict (controlled by sleep interval)
                should_predict = True

            if should_predict:
                logger.info("=" * 60)
                logger.info(f"Prediction triggered at {now}")
                logger.info("=" * 60)

                # Ensure we have a model
                if model is None:
                    logger.warning("No model available, training initial model")
                    merged_data = get_merged_data_for_training(
                        get_power_from_db=settings.get_power_from_db,
                        get_weather_data_from_api=settings.get_weather_from_api
                    )
                    model = train_model(merged_data, settings.prediction_features)
                    joblib.dump(model, settings.model_path)
                    logger.info(f"Initial model saved to {settings.model_path}")

                # Get tomorrow's forecast
                weather_forecast = get_forecast_data()

                # Generate prediction
                forecast_data = predict_tomorrow(model, weather_forecast, settings.prediction_features)

                # Compute overproduction
                df_overproduction = compute_energy_overproduction(
                    forecast_data,
                    settings.overproduction_threshold_watts
                )

                # Calculate total energy and overproduction windows for AWTRIX
                total_energy_kwh = forecast_data["predicted"].sum() / 1000

                # Extract overproduction time windows (consecutive hours)
                overproduction_windows = []
                if len(df_overproduction) > 0:
                    # Parse hours from overproduction times
                    df_overproduction_copy = df_overproduction.copy()
                    df_overproduction_copy["hour"] = pd.to_datetime(df_overproduction_copy["overproduction_time"]).dt.hour

                    # Group consecutive hours into time windows
                    current_start = None
                    current_end = None

                    for hour in sorted(df_overproduction_copy["hour"].unique()):
                        if current_start is None:
                            current_start = hour
                            current_end = hour
                        elif hour == current_end + 1:
                            current_end = hour
                        else:
                            overproduction_windows.append((current_start, current_end + 1))
                            current_start = hour
                            current_end = hour

                    # Add final window
                    if current_start is not None:
                        overproduction_windows.append((current_start, current_end + 1))

                # Send AWTRIX forecast summary
                send_awtrix_forecast_summary(total_energy_kwh, overproduction_windows)

                # Prepare notification message
                tomorrow = get_tomorrow_date(settings.timezone)
                if len(df_overproduction) > 0:
                    message = f"Overproduction forecast for {tomorrow}:\n\n{df_overproduction.to_string(index=False)}"
                else:
                    message = f"No overproduction expected for {tomorrow} (all below {settings.overproduction_threshold_watts}W threshold)"

                logger.info(f"Prediction summary: {message}")

                # Send notification if in configured time window and notifications enabled
                should_notify = (
                    settings.enable_notifications and
                    now.hour == settings.notification_hour and
                    now.minute <= settings.notification_minute_window
                )

                if should_notify:
                    logger.info("In notification window, sending Pushover notification")
                    plot_path = settings.get_full_path(settings.prediction_plot_file)
                    success = send_pushover_notification(
                        message=message,
                        title=f"Solar Forecast {tomorrow}",
                        image_path=plot_path if plot_path.exists() else None
                    )
                    if success:
                        logger.info("Notification sent successfully")
                    else:
                        logger.warning("Notification failed (check logs above)")
                else:
                    logger.debug(f"Outside notification window or notifications disabled")

                last_prediction_date = current_date
                logger.info("Prediction cycle completed successfully")

        except Exception as e:
            logger.error(f"Error in prediction loop: {e}", exc_info=True)

        # Sleep until next check
        if not shutdown_flag:
            if settings.prediction_mode == "daily":
                # In daily mode, check every 5 minutes
                sleep_seconds = 300
            else:
                # In interval mode, use configured interval
                sleep_seconds = settings.prediction_interval_seconds

            for _ in range(sleep_seconds):
                if shutdown_flag:
                    break
                await asyncio.sleep(1)

    logger.info("Prediction loop terminated gracefully")


async def main():
    """Main entry point for the application."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 60)
    logger.info("Solar Power Prediction Service Starting")
    logger.info(f"Location: ({settings.latitude}, {settings.longitude})")
    logger.info(f"Timezone: {settings.timezone}")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info(f"Model: RandomForest(n_estimators={settings.model_n_estimators})")
    logger.info(f"Overproduction threshold: {settings.overproduction_threshold_watts}W")
    logger.info("=" * 60)

    try:
        await prediction_loop()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise
    finally:
        logger.info("Solar Power Prediction Service Stopped")


if __name__ == "__main__":
    asyncio.run(main())
