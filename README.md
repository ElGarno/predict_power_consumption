# Solar Power Prediction System

A production-ready machine learning system that predicts solar power generation based on weather forecasts. Runs continuously on a NAS, trains RandomForest models, and sends daily notifications about expected energy overproduction periods.

## Features

- <$ Fetches weather forecasts from Open-Meteo DWD-ICON API
- =� Trains ML models on historical weather and power consumption data
- � Predicts next-day solar power production hourly
- =� Sends Pushover notifications with overproduction forecasts
- =3 Docker-ready for NAS deployment
- � Fully configurable via environment variables
- =� Comprehensive logging and error handling
- = Graceful shutdown support

## Prerequisites

- Python 3.13+
- InfluxDB instance with power consumption data
- Pushover account for notifications (optional)
- uv or poetry for dependency management

## Understanding Your Monitoring Setup

This system works with **smart plugs (e.g., Tapo)** that measure instantaneous power consumption. Important notes:

- **What's measured**: Individual devices plugged into smart plugs (e.g., fridge, office equipment, appliances)
- **What's NOT measured**: Heating, main lighting, and other circuits not on smart plugs
- **Data format**: Smart plugs report instantaneous power (watts) every ~30 seconds
- **Hourly aggregation**: The system uses `.mean()` to calculate average power per hour (not `.sum()`)

**Example setup:**
- 8 smart plugs monitoring: fridge, office, kitchen appliances, TV, washing machines
- Average monitored consumption: ~170-200W
- Total household (including unmeasured): much higher

**Solar panel:**
- Small solar system (e.g., balcony solar or 1-2 panels)
- Peak capacity: typically 600-800W
- Daily production: 1-5 kWh depending on season

## Quick Start - Local Testing

### Step 1: Clone and Install Dependencies

```bash
# Navigate to project directory
cd predict_power_consumption

# Install dependencies (choose one)
uv sync                # Recommended (faster)
# OR
poetry install         # Alternative
```

### Step 2: Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

**Required settings in `.env`:**
```bash
# InfluxDB (required if fetching power data)
INFLUXDB_TOKEN=your_influxdb_token_here
INFLUXDB_URL=http://your-influxdb-host:8086

# Pushover (optional, for notifications)
PUSHOVER_TAPO_API_TOKEN=your_pushover_api_token
PUSHOVER_USER_GROUP_WOERIS=your_pushover_group_key

# Location (adjust to your coordinates)
LATITUDE=51.14
LONGITUDE=7.92
TIMEZONE=Europe/Berlin
```

### Step 3: Prepare Training Data (First Time Only)

```bash
# Option A: Use existing cached data (if available)
# Just run the main script, it will use cached files from data/

# Option B: Fetch fresh data from InfluxDB
python get_power_consumption_data.py

# Option C: Fetch fresh weather data from DWD
python get_weather_data_adorn_hourly_api.py

# Merge the data
python prepare_data.py
```

**Expected output:**
- `data/power_consumption_export.parquet` (from InfluxDB)
- `data/attendorn_hourly_weather_data_api.parquet` (from DWD)
- `data/merged_data_weather_power.parquet` (combined training data)

### Step 4: Run Local Test (Quick Mode)

```bash
# Set test configuration
export UPDATE_DATA_ON_START=false      # Use cached data
export UPDATE_MODEL_ON_START=true      # Train model once
export PREDICTION_INTERVAL_SECONDS=60  # Short interval for testing
export ENABLE_NOTIFICATIONS=false      # Disable notifications
export LOG_LEVEL=INFO

# Run the prediction service
python train_ml_regression_model.py
```

**What happens:**
1.  Loads cached training data from `data/merged_data_weather_power.parquet`
2.  Trains a new RandomForest model (takes 30-60 seconds)
3.  Fetches tomorrow's weather forecast
4.  Generates solar power predictions
5.  Computes overproduction periods
6.  Saves prediction plot to `data/prediction_overproduction.png`
7.  Loops every 60 seconds (Ctrl+C to stop)

### Step 5: View Results

```bash
# Check the generated plot
open data/prediction_overproduction.png  # macOS
# or
xdg-open data/prediction_overproduction.png  # Linux

# Check the trained model
ls -lh data/model_weather_solar_power.pkl
```

## Testing Individual Components

### Test 1: Fetch Power Data from InfluxDB

```bash
# Make sure InfluxDB credentials are set in .env
python get_power_consumption_data.py
```

Expected: Creates `data/power_consumption_export.parquet` with historical power data.

### Test 2: Fetch Weather Data from DWD

```bash
# No credentials needed (public API)
python get_weather_data_adorn_hourly_api.py
```

Expected: Creates `data/attendorn_hourly_weather_data_api.parquet` with weather data.

### Test 3: Prepare Training Data

```bash
# Requires both power and weather data files
python prepare_data.py
```

Expected: Creates `data/merged_data_weather_power.parquet` with merged dataset.

### Test 4: Train Model (One-Off)

```python
# Train a model once without the service loop
python -c "
from prepare_data import get_merged_data_for_training
from train_ml_regression_model import train_model
import joblib
from config import settings

print('Loading training data...')
data = get_merged_data_for_training(
    get_power_from_db=False,  # Use cached data
    get_weather_data_from_api=False  # Use cached data
)

print(f'Training model on {len(data)} samples...')
model = train_model(data)

print(f'Saving model to {settings.model_path}')
joblib.dump(model, settings.model_path)
print('Model training complete!')
"
```

### Test 5: Generate Prediction (With Existing Model)

```bash
# Use cached model and data (fastest test)
export UPDATE_DATA_ON_START=false
export UPDATE_MODEL_ON_START=false
export PREDICTION_INTERVAL_SECONDS=60
export LOG_LEVEL=DEBUG  # See detailed logs

python train_ml_regression_model.py
```

Press `Ctrl+C` after one successful prediction cycle.

## Configuration Options

All settings can be configured via environment variables or `.env` file:

### Essential Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFLUXDB_URL` | `http://localhost:8086` | InfluxDB server URL |
| `INFLUXDB_TOKEN` | *required* | InfluxDB authentication token |
| `LATITUDE` | `51.14` | Location latitude for weather |
| `LONGITUDE` | `7.92` | Location longitude for weather |
| `TIMEZONE` | `Europe/Berlin` | Timezone for all operations |

### Performance Settings (NAS Optimization)

| Variable | Default | Description |
|----------|---------|-------------|
| `UPDATE_DATA_ON_START` | `false` | Fetch fresh data each cycle (slow) |
| `UPDATE_MODEL_ON_START` | `false` | Retrain model each cycle (very slow!) |
| `PREDICTION_INTERVAL_SECONDS` | `600` | Seconds between prediction cycles |
| `MODEL_N_ESTIMATORS` | `40` | RandomForest trees (more = slower) |

### Notification Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_NOTIFICATIONS` | `true` | Enable Pushover notifications |
| `NOTIFICATION_HOUR` | `21` | Hour to send daily notification (0-23) |
| `NOTIFICATION_MINUTE_WINDOW` | `10` | Minute window to send (0-59) |
| `OVERPRODUCTION_THRESHOLD_WATTS` | `200` | Threshold for overproduction alert (see guide below) |

#### Setting the Right Overproduction Threshold

The threshold determines when you'll be notified about excess solar production. **This is compared against your monitored devices only** (smart plugs), not total household consumption.

**For typical balcony solar (600-800W peak):**
- `200W` - Notify when solar exceeds typical monitored consumption (recommended)
- `300W` - Notify during good solar production periods
- `400W` - Notify only during strong solar production

**For larger solar systems (1-3kW peak):**
- `500W` - Notify when solar significantly exceeds baseline consumption
- `1000W` - Notify during strong overproduction
- `1500W` - Notify during peak solar hours with significant excess

**To find your ideal threshold:**
1. Check your solar panel's peak capacity (nameplate watts)
2. Check average power of your monitored devices (see training data stats)
3. Set threshold between average consumption and solar peak
4. Adjust based on notification frequency preferences

### Logging Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `LOG_FILE` | *none* | Optional file path for persistent logs |

## Deployment to NAS

### Option 1: Docker Compose (Recommended)

```bash
# 1. Ensure .env is configured with NAS settings
nano .env

# 2. Build and start the service
docker-compose up -d

# 3. View logs
docker-compose logs -f solar-prediction

# 4. Stop the service
docker-compose down
```

### Option 2: Direct Docker

```bash
# Build image
docker build -t solar-prediction .

# Run container
docker run -d \
  --name solar-prediction \
  --env-file .env \
  -v $(pwd)/data:/usr/src/app/data \
  --restart unless-stopped \
  solar-prediction
```

### Option 3: systemd Service (Direct on NAS)

Create `/etc/systemd/system/solar-prediction.service`:

```ini
[Unit]
Description=Solar Power Prediction Service
After=network.target influxdb.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/predict_power_consumption
Environment="PATH=/path/to/predict_power_consumption/.venv/bin:$PATH"
EnvironmentFile=/path/to/predict_power_consumption/.env
ExecStart=/path/to/predict_power_consumption/.venv/bin/python train_ml_regression_model.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable solar-prediction
sudo systemctl start solar-prediction
sudo systemctl status solar-prediction
```

## Troubleshooting

### Problem: "InfluxDB connection failed"

**Solution:**
```bash
# Check InfluxDB is accessible
curl http://your-influxdb-host:8086/health

# Verify token in .env
echo $INFLUXDB_TOKEN

# Test connection manually
python -c "
from influxdb_client import InfluxDBClient
import os
from dotenv import load_dotenv
load_dotenv()

client = InfluxDBClient(
    url=os.getenv('INFLUXDB_URL'),
    token=os.getenv('INFLUXDB_TOKEN'),
    org=os.getenv('INFLUXDB_ORG', 'None')
)
print('Connected successfully!' if client.ping() else 'Connection failed')
"
```

### Problem: "No data found for specified time range"

**Solution:**
```bash
# Check your historical data start date
# Edit .env:
HISTORICAL_DATA_START_DATE=2024-01-01  # Adjust to match your data

# Or check what data is available in InfluxDB via UI
```

### Problem: "Missing features in forecast"

**Solution:**
- The Open-Meteo API may occasionally have incomplete data
- The code will use available features and log a warning
- This is normal and predictions will still work with reduced accuracy

### Problem: "Model training very slow on NAS"

**Solution:**
```bash
# Reduce model complexity
export MODEL_N_ESTIMATORS=20  # Default is 40

# Train model once, then reuse
export UPDATE_MODEL_ON_START=false

# Train model manually and cache it
python -c "
from prepare_data import get_merged_data_for_training
from train_ml_regression_model import train_model
import joblib
data = get_merged_data_for_training()
model = train_model(data)
joblib.dump(model, 'data/model_weather_solar_power.pkl')
"
```

### Problem: "Notifications not sending"

**Solution:**
```bash
# Test Pushover credentials
curl -s \
  --form-string "token=YOUR_API_TOKEN" \
  --form-string "user=YOUR_USER_KEY" \
  --form-string "message=Test from solar prediction" \
  https://api.pushover.net/1/messages.json

# Check notification is enabled
export ENABLE_NOTIFICATIONS=true

# Check time window (default 21:00-21:10)
export NOTIFICATION_HOUR=21
```

## File Structure

```
predict_power_consumption/
   config.py                  # Configuration management
   utils.py                   # Shared utilities
   get_power_consumption_data.py      # InfluxDB data fetcher
   get_weather_data_adorn_hourly_api.py   # DWD weather fetcher
   prepare_data.py            # Data preprocessing
   train_ml_regression_model.py   # Main prediction service
   Dockerfile                 # Docker build instructions
   docker-compose.yml         # Docker deployment config
   pyproject.toml            # Python dependencies
   uv.lock                   # Locked dependencies (uv)
   .env.example              # Environment template
   .gitignore                # Git exclusions
   CLAUDE.md                 # Developer guide
   README.md                 # This file
   data/                     # Persistent data directory
       power_consumption_export.parquet
       attendorn_hourly_weather_data_api.parquet
       merged_data_weather_power.parquet
       model_weather_solar_power.pkl
       prediction_overproduction.png
```

## Development Workflow

### Making Code Changes

```bash
# 1. Make your changes
nano train_ml_regression_model.py

# 2. Test locally
python train_ml_regression_model.py

# 3. Rebuild Docker image
docker-compose up -d --build

# 4. Check logs
docker-compose logs -f solar-prediction
```

### Adding New Features

1. **New weather features**: Edit `prediction_features` in `.env`
2. **New model algorithm**: Modify `train_model()` in `train_ml_regression_model.py`
3. **New notification channel**: Add function to `utils.py`

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python train_ml_regression_model.py

# Or add to .env:
LOG_LEVEL=DEBUG
LOG_FILE=logs/solar-prediction.log  # Optional: save to file
```

## Performance Benchmarks

Tested on Synology DS920+ NAS:

| Operation | Time | Notes |
|-----------|------|-------|
| Model training | 45s | 40 estimators, 2000 samples |
| Prediction generation | 2s | Using cached model |
| Weather forecast fetch | 3s | Open-Meteo API |
| Full cycle (cached) | 5s | Load model + predict |
| Full cycle (fresh train) | 50s | Fetch + train + predict |

## FAQ

**Q: How often should I retrain the model?**
A: Weekly is sufficient. Set `UPDATE_MODEL_ON_START=false` and retrain manually once per week.

**Q: Can I use this for multiple locations?**
A: Currently supports one location. For multiple locations, run separate instances with different config.

**Q: What if I don't have InfluxDB?**
A: You can modify `get_power_consumption_data.py` to load from CSV/JSON or any other source.

**Q: Does it work outside Germany?**
A: Weather forecast works globally (Open-Meteo). Historical DWD data is Germany-only. Modify `get_weather_data_adorn_hourly_api.py` for other countries.

**Q: How much disk space needed?**
A: ~50MB for data, ~5MB for model. Logs depend on retention settings.

## License

This project is for personal use. Modify as needed for your setup.

## Credits

- Weather data: [Open-Meteo](https://open-meteo.com/) & [DWD](https://www.dwd.de/)
- ML framework: [scikit-learn](https://scikit-learn.org/)
- Notifications: [Pushover](https://pushover.net/)

## Support

For issues or questions, check:
1. This README troubleshooting section
2. CLAUDE.md for architecture details
3. Logs with `LOG_LEVEL=DEBUG`

---

**Happy solar forecasting!  �**
