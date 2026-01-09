#!/bin/bash
# Development/Testing startup script

set -e

echo "Starting AI Agent Verification System (Development Mode)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create necessary directories
mkdir -p temp logs data debug_output models

# Check if models exist
if [ ! -f "models/best4.pt" ] || [ ! -f "models/best.pt" ]; then
    echo "WARNING: Model files not found in models/ directory"
    echo "Please ensure best4.pt and best.pt are present"
fi

# Set environment variables for development
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}
export ENABLE_DEBUG_IMAGES=${ENABLE_DEBUG_IMAGES:-false}
export REDIS_HOST=${REDIS_HOST:-localhost}
export REDIS_PORT=${REDIS_PORT:-6379}

# Start with uvicorn for development (hot reload)
python main.py
