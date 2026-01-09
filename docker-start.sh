#!/bin/bash
# Build and run with Docker Compose

set -e

echo "Building and starting AI Agent Verification System..."

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "ERROR: models/ directory not found"
    echo "Please create models/ and add required model files:"
    echo "  - models/best4.pt"
    echo "  - models/best.pt"
    exit 1
fi

# Check if model files exist
if [ ! -f "models/best4.pt" ]; then
    echo "ERROR: models/best4.pt not found"
    exit 1
fi

if [ ! -f "models/best.pt" ]; then
    echo "ERROR: models/best.pt not found"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Build and start services
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo ""
echo "Services started successfully!"
echo ""
echo "Application: http://localhost:8101"
echo "Health Check: http://localhost:8101/health"
echo "Redis Commander (debug): http://localhost:8081"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f app"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
