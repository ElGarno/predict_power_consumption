# Production-ready Dockerfile for Solar Power Prediction Service
# Uses uv for faster dependency management optimized for NAS deployment

FROM python:3.13.0-slim

# Set working directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.cargo/bin/uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

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
    PATH="/usr/src/app/.venv/bin:$PATH" \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from pathlib import Path; import sys; sys.exit(0 if Path('data/model_weather_solar_power.pkl').exists() else 1)"

# Run the prediction service
CMD ["python", "./train_ml_regression_model.py"]
