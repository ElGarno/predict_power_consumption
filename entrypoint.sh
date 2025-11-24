#!/bin/bash
# Entrypoint script to diagnose startup issues

echo "Container starting at $(date)"
echo "Memory limits:"
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "  No cgroup memory limit"
echo "Memory usage:"
cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo "  No cgroup info"
echo "Python version: $(python --version)"
echo ""

echo "Starting Python application..."
echo "PYTHONUNBUFFERED=1 to ensure immediate log output"

# Enable core dumps to catch crashes
ulimit -c unlimited 2>/dev/null || true

# Reduce Python memory pressure
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=2

exec python -u ./train_ml_regression_model.py