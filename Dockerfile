# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl build-essential

# Install Poetry and add to PATH
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

# Create data directory if it doesn't exist
RUN mkdir -p /usr/src/app/data

# Set permissions so the app can write to it
RUN chmod 777 /usr/src/app/data

# Copy application code
COPY . .

# Run app.py when the container launches
CMD ["python", "./train_ml_regression_model.py"]
