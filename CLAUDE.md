# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready solar power consumption prediction system that uses machine learning to forecast solar power generation based on weather data. It runs continuously on a NAS, fetches weather forecasts, trains/uses RandomForest models, and sends daily notifications about expected energy overproduction periods.

## Development Commands

### Package Management
- **Install dependencies**: `uv sync` (preferred) or `poetry install`
- **Add dependency**: `uv add <package>` or `poetry add <package>`
- The project supports both uv (faster) and poetry

### Running the Application

#### Local Development
```bash
# Copy environment template and fill in your credentials
cp .env.example .env
# Edit .env with your actual values

# Run main prediction service
python train_ml_regression_model.py

# Or run individual data collection scripts
python get_power_consumption_data.py  # Fetch from InfluxDB
python get_weather_data_adorn_hourly_api.py  # Fetch from DWD
python prepare_data.py  # Merge and prepare training data
```

#### Docker Deployment (NAS Production)
```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f solar-prediction

# Stop service
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Configuration
- **Environment variables**: Copy `.env.example` to `.env` and configure
- **Application settings**: All configurable via environment variables (see `config.py`)
- **Key settings**:
  - `INFLUXDB_URL`, `INFLUXDB_TOKEN`: Database connection
  - `LATITUDE`, `LONGITUDE`: Location for weather data
  - `OVERPRODUCTION_THRESHOLD_WATTS`: Power threshold (default: 20000W)
  - `PREDICTION_INTERVAL_SECONDS`: How often to run predictions (default: 600s)
  - `UPDATE_MODEL_ON_START`: Retrain model each cycle (default: false, expensive!)
  - `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Architecture

### Module Structure

```
├── config.py              # Centralized configuration with pydantic-settings
├── utils.py               # Shared utilities (logging, notifications, validation)
├── get_power_consumption_data.py   # InfluxDB data fetching
├── get_weather_data_adorn_hourly_api.py  # DWD weather API client
├── prepare_data.py        # Data preprocessing and merging pipeline
├── train_ml_regression_model.py    # ML training and prediction service (main)
└── data/                  # Persistent storage for models and datasets
```

### Data Pipeline Flow

1. **Historical Data Collection** (manual or scheduled):
   - `get_power_consumption_data.py`: Queries InfluxDB in 7-day chunks, exports to Parquet
   - `get_weather_data_adorn_hourly_api.py`: Fetches DWD hourly weather via Wetterdienst
   - Both cache data as Parquet files in `data/` directory

2. **Data Preparation** (`prepare_data.py`):
   - Loads cached or fetches fresh data based on config
   - Filters weather features: radiation, temperature, humidity, cloudiness, sunshine
   - Converts power data timestamps with proper timezone handling (pytz)
   - Resamples power to hourly intervals, pivots by device
   - Merges on timezone-aware timestamps
   - Exports `merged_data_weather_power.parquet`

3. **Model Training & Prediction** (`train_ml_regression_model.py`):
   - Main service runs continuously in async loop
   - Configurable whether to retrain model each cycle (expensive!) or load cached
   - Trains RandomForestRegressor with TimeSeriesSplit cross-validation
   - Fetches next-day forecast from Open-Meteo DWD-ICON API
   - Predicts hourly solar production for tomorrow
   - Computes overproduction periods above threshold
   - Generates matplotlib visualization plot
   - Sends Pushover notification at configured time (default: 21:00)
   - Supports graceful shutdown on SIGTERM/SIGINT

### Key Design Improvements (Production-Ready)

#### Configuration Management
- All hardcoded values moved to `config.py` using pydantic-settings
- Environment variable-driven with sensible defaults
- Easy to switch between dev/prod/NAS environments

#### Logging Infrastructure
- Replaced all `print()` with proper logging module
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Structured logging with timestamps and module names
- Optional file logging for persistent logs

#### Error Handling
- Comprehensive try-catch blocks with proper error types
- Input validation using pydantic and custom validators
- HTTP timeouts on all external API calls
- Graceful degradation (continues if single chunk fails)

#### Security
- Non-root Docker user (appuser:1000)
- Proper file permissions (755 not 777)
- Secrets via environment variables only
- .env excluded from git with .gitignore

#### Resource Optimization (NAS-friendly)
- Model training optional per cycle (avoid constant retraining)
- Cached model/data reuse between iterations
- Chunked InfluxDB queries to avoid timeouts/memory issues
- n_jobs=-1 for parallel RandomForest training

#### Timezone Handling
- Proper pytz timezone conversion (no more manual +1hr offset)
- Timezone-aware datetimes throughout
- UTC → local timezone conversion with DST support

#### Docker & Deployment
- Updated Dockerfile uses uv (faster than poetry)
- Pinned base image version (python:3.13.0-slim)
- Health check monitors model file existence
- docker-compose.yml with resource limits, volume persistence
- Graceful 30s shutdown period

### Data Storage
- **Parquet format**: Efficient columnar storage for all datasets
- **Model persistence**: joblib for scikit-learn models
- **Caching strategy**: Reuse data/models between cycles to reduce API/DB load
- **Volume mounting**: Docker volume for persistence across container restarts

## Important Implementation Details

### InfluxDB Configuration
- **URL**: Configurable via `INFLUXDB_URL` (default: localhost:8086)
- **Chunking**: Queries split into 7-day chunks (configurable) to prevent timeouts
- **Bucket**: `power_consumption` measurement with `device` and `power` fields
- **Flux query**: Uses pivot for wide-format output

### Weather Data Sources
- **Historical**: Wetterdienst library querying DWD stations (configurable location)
- **Forecast**: Open-Meteo DWD-ICON API (free, no API key needed)
- **Features**: radiation_global, radiation_sky_short_wave_diffuse, sunshine_duration, temperature_air_mean_2m, cloud_cover_total, humidity
- **Timezone**: All data aligned to configured timezone (default: Europe/Berlin)

### ML Model Configuration
- **Algorithm**: RandomForestRegressor (fast, interpretable, works well on tabular data)
- **Hyperparameters**: Configurable n_estimators (default: 40), random_state=42
- **Cross-validation**: TimeSeriesSplit with configurable splits (default: 5)
- **Train/test split**: Configurable ratio (default: 90/10)
- **Target**: Hourly solar power production in watts
- **Predictions**: Clipped to non-negative values

### Notification Logic
- **Service**: Pushover (requires API token and user key in .env)
- **Timing**: Configurable notification hour and minute window (default: 21:00-21:10)
- **Message**: Includes overproduction hours with timestamps and wattage
- **Image**: Attaches prediction plot visualization
- **Threshold**: Configurable (default: 20,000W)

### Graceful Shutdown
- Handles SIGTERM and SIGINT signals
- Completes current prediction cycle before exiting
- No data loss or corruption on container restart
- Docker stop_grace_period: 30s

## Common Development Tasks

### Adding New Features
1. **New weather features**: Update `prediction_features` in `config.py` or `.env`
2. **New model algorithm**: Modify `train_model()` in `train_ml_regression_model.py`
3. **New notification channel**: Add function to `utils.py`, call in prediction loop

### Testing Locally
```bash
# Use cached data (fast, no API calls)
export UPDATE_DATA_ON_START=false
export UPDATE_MODEL_ON_START=false
export PREDICTION_INTERVAL_SECONDS=60
python train_ml_regression_model.py
```

### Debugging
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python train_ml_regression_model.py

# Or set in .env:
LOG_LEVEL=DEBUG
```

### Model Retraining
```bash
# Force model retraining (run once, then disable)
export UPDATE_MODEL_ON_START=true
python train_ml_regression_model.py
# After first successful run, set back to false

# Or separate training script:
python -c "
from prepare_data import get_merged_data_for_training
from train_ml_regression_model import train_model
import joblib
from config import settings

data = get_merged_data_for_training()
model = train_model(data)
joblib.dump(model, settings.model_path)
print(f'Model saved to {settings.model_path}')
"
```

## Known Constraints & Notes

- **Geographic scope**: Optimized for Germany/Central Europe (DWD data, Europe/Berlin timezone)
- **Forecast horizon**: Next day only (Open-Meteo DWD-ICON limitation)
- **Historical data**: Starts from `HISTORICAL_DATA_START_DATE` (default: 2024-12-15)
- **NAS performance**: Model training can take 30-60s on low-power NAS CPUs
- **API rate limits**: Open-Meteo is generous but avoid excessive requests
- **No automated testing**: Tests not yet implemented (TODO)

## Troubleshooting

### "No data found" errors
- Check InfluxDB connectivity and credentials
- Verify date ranges overlap between power and weather data
- Check DWD API availability

### Model prediction errors
- Ensure all required features present in forecast data
- Check feature names match between training and inference
- Verify model file exists and is not corrupted

### Notifications not sending
- Verify Pushover API token and user key in .env
- Check `ENABLE_NOTIFICATIONS=true` in config
- Ensure within notification time window
- Check internet connectivity from container

### Docker container crashes
- Check logs: `docker-compose logs solar-prediction`
- Verify .env file exists with all required variables
- Ensure data volume has write permissions
- Check resource limits aren't too restrictive

## Future Improvements (TODO)

- Add unit tests (pytest) for data pipeline
- Implement model performance monitoring and drift detection
- Add Prometheus metrics export
- Support multiple locations/models
- Web dashboard for predictions
- Automated model retraining scheduler (weekly)
- Integration tests with mock InfluxDB
