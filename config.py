"""
Configuration management for predict_power_consumption application.

Uses pydantic-settings to load configuration from environment variables
with sensible defaults for local development and NAS deployment.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from pathlib import Path
from typing import Optional
import logging


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # InfluxDB Configuration
    influxdb_url: str = Field(
        default="http://localhost:8086",
        description="InfluxDB server URL"
    )
    influxdb_token: str = Field(
        ...,
        description="InfluxDB authentication token"
    )
    influxdb_org: str = Field(
        default="None",
        description="InfluxDB organization name"
    )
    influxdb_bucket: str = Field(
        default="power_consumption",
        description="InfluxDB bucket name"
    )
    influxdb_query_chunk_days: int = Field(
        default=7,
        description="Days per chunk when querying InfluxDB (to avoid timeouts)"
    )

    # Pushover Notification Configuration
    pushover_api_token: str = Field(
        default="",
        alias="PUSHOVER_TAPO_API_TOKEN",
        description="Pushover API token"
    )
    pushover_user_group: str = Field(
        default="",
        alias="PUSHOVER_USER_GROUP_WOERIS",
        description="Pushover user/group key"
    )
    enable_notifications: bool = Field(
        default=True,
        description="Enable/disable Pushover notifications"
    )
    notification_hour: int = Field(
        default=21,
        ge=0,
        le=23,
        description="Hour to send daily prediction notification (0-23)"
    )
    notification_minute_window: int = Field(
        default=10,
        ge=1,
        le=59,
        description="Minute window for sending notification"
    )

    # Location Configuration (for weather data)
    latitude: float = Field(
        default=51.14,
        description="Latitude for weather data location"
    )
    longitude: float = Field(
        default=7.92,
        description="Longitude for weather data location"
    )
    timezone: str = Field(
        default="Europe/Berlin",
        description="Timezone for all datetime operations"
    )

    # Data Configuration
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for data storage"
    )
    power_consumption_file: str = Field(
        default="power_consumption_export.parquet",
        description="Filename for power consumption export"
    )
    weather_data_file: str = Field(
        default="attendorn_hourly_weather_data_api.parquet",
        description="Filename for weather data export"
    )
    merged_data_file: str = Field(
        default="merged_data_weather_power.parquet",
        description="Filename for merged training data"
    )

    # Historical data configuration
    historical_data_start_date: str = Field(
        default="2024-12-15",
        description="Start date for historical data (YYYY-MM-DD)"
    )

    # ML Model Configuration
    model_path: Optional[Path] = Field(
        default=None,
        description="Path to trained model file (auto-generated if None)"
    )
    model_n_estimators: int = Field(
        default=40,
        description="Number of estimators for RandomForest model"
    )
    model_random_state: int = Field(
        default=42,
        description="Random state for reproducibility"
    )
    model_train_test_split: float = Field(
        default=0.9,
        ge=0.5,
        le=0.95,
        description="Train/test split ratio (e.g., 0.9 = 90% train, 10% test)"
    )
    model_cv_splits: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of cross-validation splits"
    )

    # Feature configuration
    prediction_features: list[str] = Field(
        default=[
            "radiation_global",
            "radiation_sky_short_wave_diffuse",
            "sunshine_duration",
            "temperature_air_mean_2m",
            "cloud_cover_total",
            "humidity"
        ],
        description="Features used for prediction"
    )

    # Prediction Configuration
    overproduction_threshold_watts: int = Field(
        default=200,
        description="Power threshold in watts for overproduction detection (compared to monitored devices only)"
    )
    prediction_plot_file: str = Field(
        default="prediction_overproduction.png",
        description="Filename for prediction visualization"
    )

    # Scheduling Configuration
    prediction_mode: str = Field(
        default="daily",
        description="Prediction mode: 'daily' (once per day) or 'interval' (every N seconds)"
    )
    prediction_interval_seconds: int = Field(
        default=600,
        ge=60,
        description="Interval between prediction runs when using 'interval' mode (seconds)"
    )
    daily_prediction_hour: int = Field(
        default=21,
        ge=0,
        le=23,
        description="Hour to run daily prediction (0-23), typically same as notification hour"
    )
    daily_prediction_minute: int = Field(
        default=0,
        ge=0,
        le=59,
        description="Minute to run daily prediction (0-59)"
    )

    # Model Retraining Configuration
    model_retrain_enabled: bool = Field(
        default=True,
        description="Enable automatic weekly model retraining"
    )
    model_retrain_day: int = Field(
        default=0,
        ge=0,
        le=6,
        description="Day of week to retrain model (0=Monday, 6=Sunday)"
    )
    model_retrain_hour: int = Field(
        default=3,
        ge=0,
        le=23,
        description="Hour to retrain model (0-23), typically early morning"
    )
    model_retrain_minute: int = Field(
        default=0,
        ge=0,
        le=59,
        description="Minute to retrain model (0-59)"
    )

    # Pipeline Configuration
    update_data_on_start: bool = Field(
        default=False,
        description="Fetch fresh data from sources on startup"
    )
    update_model_on_start: bool = Field(
        default=False,
        description="Retrain model on startup (expensive, use sparingly)"
    )
    get_power_from_db: bool = Field(
        default=False,
        description="Fetch power data from InfluxDB (vs. using cached file)"
    )
    get_weather_from_api: bool = Field(
        default=True,
        description="Fetch weather data from API (vs. using cached file)"
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path (None = console only)"
    )

    # HTTP Configuration
    http_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout for HTTP requests"
    )
    http_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for failed HTTP requests"
    )

    # Weather API Configuration
    weather_api_url: str = Field(
        default="https://api.open-meteo.com/v1/dwd-icon",
        description="Open-Meteo DWD-ICON API URL for forecasts"
    )

    @validator("model_path", always=True)
    def set_model_path(cls, v, values):
        """Auto-generate model path if not provided."""
        if v is None:
            data_dir = values.get("data_dir", Path("./data"))
            return data_dir / "model_weather_solar_power.pkl"
        return v

    @validator("data_dir", always=True)
    def ensure_data_dir_exists(cls, v):
        """Ensure data directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v

    def get_full_path(self, filename: str) -> Path:
        """Get full path for a file in the data directory."""
        return self.data_dir / filename

    def setup_logging(self) -> logging.Logger:
        """Configure logging based on settings."""
        logger = logging.getLogger("predict_power_consumption")
        logger.setLevel(getattr(logging, self.log_level))

        # Remove existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level))
        formatter = logging.Formatter(self.log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if specified)
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(getattr(logging, self.log_level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


# Global settings instance
settings = Settings()

# Setup logging on module import
logger = settings.setup_logging()
