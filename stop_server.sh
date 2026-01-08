#!/bin/bash
if [ -f server.pid ]; then
    PID=$(cat server.pid)
    kill $PID 2>/dev/null
    sleep 2
    if ps -p $PID > /dev/null 2>&1; then
        kill -9 $PID 2>/dev/null
    fi
    rm -f server.pid
    echo "✓ Server stopped"
else
    pkill -f "python.*main.py"
    echo "✓ Processes killed"
fi
