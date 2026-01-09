#!/bin/bash
echo "Monitoring server logs (Ctrl+C to exit)..."
tail -f server.log | grep --line-buffered -E "INFO|ERROR|WARNING" | \
  while read line; do
    if echo "$line" | grep -q "ERROR"; then
        echo -e "\033[0;31m$line\033[0m"
    elif echo "$line" | grep -q "WARNING"; then
        echo -e "\033[0;33m$line\033[0m"
    else
        echo -e "\033[0;32m$line\033[0m"
    fi
  done
