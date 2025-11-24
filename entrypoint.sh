#!/bin/bash
# Entrypoint script to diagnose startup issues

echo "Container starting at $(date)"
echo "Memory available: $(free -h 2>/dev/null || echo 'N/A')"
echo "Python version: $(python --version)"
echo ""

echo "Starting Python application..."
exec python ./train_ml_regression_model.py