# Production-ready Dockerfile for Solar Power Prediction Service
# Optimized for NAS deployment

FROM python:3.13.0-slim

# Set working directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using pip (more reliable for Docker builds)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pandas==2.2.3 \
    pyarrow==18.0.0 \
    scikit-learn==1.5.2 \
    matplotlib==3.9.2 \
    requests==2.32.3 \
    python-dotenv==1.0.1 \
    pydantic-settings==2.6.1 \
    influxdb-client==1.47.0 \
    wetterdienst==0.101.0 \
    pytz==2024.2 \
    joblib==1.4.2

# Create data directory with proper permissions
RUN mkdir -p /usr/src/app/data && \
    chmod 755 /usr/src/app/data

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /usr/src/app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Run the prediction service
CMD ["python", "./train_ml_regression_model.py"]
