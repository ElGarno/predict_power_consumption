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
COPY requirements.txt ./

# Install dependencies using pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create data directory with proper permissions
RUN mkdir -p /usr/src/app/data && \
    chmod 755 /usr/src/app/data

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /usr/src/app

# Copy application code and make entrypoint executable
COPY --chown=appuser:appuser . .
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Run the prediction service
CMD ["./entrypoint.sh"]
