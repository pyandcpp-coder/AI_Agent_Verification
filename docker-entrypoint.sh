#!/bin/bash
# Production startup script for Docker deployment

set -e  # Exit on error

echo "=========================================="
echo "AI Agent Verification System - Startup"
echo "=========================================="

# Print environment info
echo "Environment:"
echo "  - Host: ${APP_HOST:-0.0.0.0}"
echo "  - Port: ${APP_PORT:-8101}"
echo "  - Workers: ${WORKERS:-1}"
echo "  - GPU: ${USE_GPU:-true}"
echo "  - Log Level: ${LOG_LEVEL:-INFO}"
echo "  - Redis: ${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}"
echo ""

# Wait for Redis to be ready
if [ "${REDIS_HOST}" != "" ]; then
    REDIS_PORT=${REDIS_PORT:-6379}
    echo "Waiting for Redis at ${REDIS_HOST}:${REDIS_PORT}..."
    timeout 30 bash -c "until nc -z \${REDIS_HOST} \${REDIS_PORT}; do sleep 1; done" || {
        echo "WARNING: Redis not available, continuing without cache"
    }
    echo "Redis is ready"
fi

# Check if models exist
echo "Checking models..."
if [ ! -f "${MODEL1_PATH:-models/best4.pt}" ]; then
    echo "ERROR: Model file not found: ${MODEL1_PATH:-models/best4.pt}"
    exit 1
fi
if [ ! -f "${MODEL2_PATH:-models/best.pt}" ]; then
    echo "ERROR: Model file not found: ${MODEL2_PATH:-models/best.pt}"
    exit 1
fi
echo "Models found"

# Check GPU availability
if [ "${USE_GPU}" = "true" ]; then
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    else
        echo "WARNING: nvidia-smi not found, GPU may not be available"
    fi
fi

# Create necessary directories
mkdir -p /app/temp /app/logs /app/data /app/debug_output

# Start the application
echo ""
echo "Starting application..."
echo "=========================================="

exec "$@"
