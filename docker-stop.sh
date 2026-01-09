#!/bin/bash
# Stop all services

docker-compose down

echo "All services stopped"
echo ""
echo "To remove volumes (WARNING: This will delete Redis cache):"
echo "  docker-compose down -v"
