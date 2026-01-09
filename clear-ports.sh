#!/bin/bash
# Clear ports used by the application

echo "Checking and clearing ports..."

# Stop Redis system service if running
echo ""
echo "Checking Redis system service..."
if systemctl is-active --quiet redis-server 2>/dev/null; then
    echo "Stopping Redis system service..."
    sudo systemctl stop redis-server
    sudo systemctl disable redis-server 2>/dev/null
    echo "✓ Redis service stopped and disabled"
elif systemctl is-active --quiet redis 2>/dev/null; then
    echo "Stopping Redis system service..."
    sudo systemctl stop redis
    sudo systemctl disable redis 2>/dev/null
    echo "✓ Redis service stopped and disabled"
else
    echo "Redis service not running"
fi

# Required ports
PORTS=(6379 8101 8081)

for PORT in "${PORTS[@]}"; do
    echo ""
    echo "Checking port $PORT..."
    
    # Find process using the port
    PID=$(sudo lsof -ti:$PORT 2>/dev/null)
    
    if [ -n "$PID" ]; then
        echo "Port $PORT is in use by PID: $PID"
        echo "Process details:"
        ps -p $PID -o pid,cmd 2>/dev/null || echo "Process already terminated"
        
        echo "Killing process $PID..."
        sudo kill -9 $PID 2>/dev/null
        
        # Verify it's killed
        sleep 1
        if sudo lsof -ti:$PORT > /dev/null 2>&1; then
            echo "WARNING: Port $PORT is still in use!"
        else
            echo "✓ Port $PORT is now free"
        fi
    else
        echo "✓ Port $PORT is already free"
    fi
done

echo ""
echo "All ports checked. You can now run: docker compose up -d"
