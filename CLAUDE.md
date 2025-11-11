# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project predicts solar power consumption using machine learning based on historical weather data and power consumption patterns. It fetches weather data from DWD (German Weather Service), combines it with power consumption data from an InfluxDB instance, trains a RandomForest regression model, and generates daily predictions with Pushover notifications for energy overproduction.

## Development Commands

### Package Management
- **Install dependencies**: `uv sync` or `poetry install`
- **Add dependency**: `uv add <package>` or `poetry add <package>`
- The project uses both `uv.lock` and `poetry.lock` - prefer `uv` for faster operations

### Running the Application
- **Main prediction loop**: `python train_ml_regression_model.py`
  - Runs continuously, updating predictions every 10 minutes
  - Sends Pushover notifications at 21:00 with next-day overproduction forecast
- **Data collection**:
  - `python get_power_consumption_data.py` - Fetch power data from InfluxDB
  - `python get_weather_data_adorn_hourly_api.py` - Fetch historical weather via Wetterdienst API
  - `python prepare_data.py` - Merge weather and power data for training
- **Docker**: `docker build -t predict-power .` then `docker run predict-power`

### Environment Configuration
- Requires `.env` file with:
  - `INFLUXDB_TOKEN` - InfluxDB authentication
  - `PUSHOVER_TAPO_API_TOKEN` - Pushover API token
  - `PUSHOVER_USER_GROUP_WOERIS` - Pushover user/group ID

## Architecture

### Data Pipeline Flow
1. **Data Collection** (separate scripts, run manually or scheduled):
   - `get_power_consumption_data.py` queries InfluxDB in 7-day chunks to avoid timeouts
   - `get_weather_data_adorn_hourly_api.py` fetches DWD weather via Wetterdienst library
   - Both export to Parquet files in `data/` directory

2. **Data Preparation** (`prepare_data.py`):
   - Filters weather data to key features: radiation_global, radiation_sky_short_wave_diffuse, sunshine_duration, temperature_air_mean_2m, cloud_cover_total, humidity
   - Resamples power consumption to hourly intervals, pivots by device_name
   - Merges datasets on timestamp with timezone-aware handling (+1 hour offset)
   - Outputs `merged_data_weather_power.parquet`

3. **Model Training & Prediction** (`train_ml_regression_model.py`):
   - Trains RandomForestRegressor (40 estimators) using TimeSeriesSplit cross-validation
   - Saves model to `data/model_weather_solar_power.pkl`
   - Fetches next-day forecast from Open-Meteo DWD-ICON API
   - Predicts hourly solar production, computes overproduction >20000W threshold
   - Generates matplotlib plot saved to `data/prediction_overproduction.png`

### Key Design Patterns
- **Async Main Loop**: Uses `asyncio` for continuous 10-minute prediction cycles
- **Chunked Queries**: InfluxDB queries split into 7-day chunks to prevent timeout (get_power_consumption_data.py:22-63)
- **Feature Mapping**: Open-Meteo forecast fields mapped to DWD historical feature names (train_ml_regression_model.py:163-171)
- **Timezone Handling**: Power data timestamps localized and shifted +1 hour to align with weather data (prepare_data.py:19)

### Data Storage
- All intermediate data stored as Parquet in `data/` directory
- Model persistence via joblib in `data/model_weather_solar_power.pkl`
- InfluxDB stores raw power measurements in `power_consumption` bucket

## Important Implementation Details

### InfluxDB Configuration
- Hardcoded URL: `http://192.168.178.114:8088`
- Bucket: `power_consumption`
- Measurement: `power_consumption` with fields `device` and `power`
- Queries use Flux syntax with pivot for wide-format output

### Weather Data Sources
- **Historical**: Wetterdienst library querying DWD stations near Attendorn (51.1279, 7.9022)
- **Forecast**: Open-Meteo DWD-ICON API (51.14, 7.92) with 48h hourly resolution
- Start date for historical data: 2024-12-15 (hardcoded in multiple files)

### Model Configuration
- Algorithm: RandomForestRegressor with 40 trees, random_state=42
- Train/test split: 90/10 temporal split
- Cross-validation: 5-fold TimeSeriesSplit
- Target variable: `solar` column (hourly solar power in watts)
- Predictions clipped to non-negative values

### Notification Logic
- Overproduction threshold: 20,000W (indicates excess production over consumption)
- Daily notification: Sent only when current time is 21:00-21:10
- Message includes hourly overproduction times and values
- Image attachment shows prediction plot with threshold visualization

## Known Constraints
- Weather forecast limited to next day only (tomorrow in Europe/Berlin timezone)
- Model retrains on every loop iteration when `update_model=True` (expensive)
- No automated testing infrastructure
- Geographic location hardcoded to Attendorn, Germany region
