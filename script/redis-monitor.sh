#!/bin/bash

# Redis Background Monitor - Logs all activity to a file
# Usage: ./redis-monitor.sh start|stop|status|tail
# This script runs in background and logs Redis metrics continuously

MONITOR_LOG="/var/log/redis-monitor.log"
PID_FILE="/tmp/redis-monitor.pid"
INTERVAL=30  # Check every 30 seconds

# Check if running with sudo
check_sudo() {
  if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Some commands require sudo. Please run with: sudo $0 $1"
    exit 1
  fi
}

# Initialize log file
init_log() {
  if [ ! -f "$MONITOR_LOG" ]; then
    touch "$MONITOR_LOG"
    chmod 666 "$MONITOR_LOG"
  fi
}

# Monitor function
monitor_redis() {
  init_log
  
  while true; do
    {
      echo ""
      echo "╔═══════════════════════════════════════════════════════════"
      echo "║ $(date '+%Y-%m-%d %H:%M:%S')"
      echo "╚═══════════════════════════════════════════════════════════"
      
      # Check if Redis is running
      if ! redis-cli ping > /dev/null 2>&1; then
        echo "❌ ERROR: Redis is not running!"
        continue
      fi
      
      echo "✅ Redis Status: Running"
      echo ""
      
      # Memory info
      echo "📊 MEMORY:"
      redis-cli INFO memory | grep -E "used_memory_human|used_memory_rss_human|maxmemory_human"
      
      # Key counts
      echo ""
      echo "📝 KEYS:"
      TOTAL=$(redis-cli DBSIZE | grep -o '[0-9]*')
      KYC=$(redis-cli KEYS "kyc_verification:*" 2>/dev/null | wc -l)
      echo "  Total Keys: $TOTAL"
      echo "  KYC Keys: $KYC"
      
      # Performance
      echo ""
      echo "⚡ PERFORMANCE:"
      redis-cli INFO stats | grep -E "total_commands_processed|instantaneous_ops_per_sec|total_net_input_bytes|total_net_output_bytes"
      
      # Clients
      echo ""
      echo "👥 CLIENTS:"
      redis-cli INFO clients | grep -E "connected_clients|blocked_clients"
      
      # Replication
      echo ""
      echo "🔄 REPLICATION:"
      redis-cli INFO replication | grep -E "role|connected_slaves"
      
      # Persistence
      echo ""
      echo "💾 PERSISTENCE:"
      redis-cli INFO persistence | grep -E "rdb_last_save_time|rdb_changes_since_last_save|aof_enabled"
      
      # Check for errors
      ERRORS=$(redis-cli INFO stats | grep total_error_replies | cut -d: -f2 | tr -d '\r')
      if [ "$ERRORS" != "0" ]; then
        echo ""
        echo "⚠️  ALERTS:"
        echo "  ERROR COUNT: $ERRORS"
      fi
      
      # Sample API responses
      echo ""
      echo "📨 SAMPLE API RESPONSES:"
      redis-cli KEYS "kyc_verification:*" 2>/dev/null | head -3 | while read key; do
        USER_ID=$(echo "$key" | cut -d: -f2)
        API_MSG=$(redis-cli GET "$key" 2>/dev/null | jq -r '.api_response.message // "N/A"' 2>/dev/null || echo "Error")
        API_SUCCESS=$(redis-cli GET "$key" 2>/dev/null | jq -r '.api_response.success // "N/A"' 2>/dev/null || echo "Error")
        echo "  User $USER_ID: success=$API_SUCCESS, msg=$API_MSG"
      done
      
    } >> "$MONITOR_LOG"
    
    sleep "$INTERVAL"
  done
}

# Start monitoring in background
start_monitor() {
  if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
      echo "⚠️  Monitor is already running with PID $OLD_PID"
      return
    fi
  fi
  
  init_log
  
  # Start monitoring in background
  nohup bash -c "
    echo \$$ > '$PID_FILE'
    $(declare -f monitor_redis)
    monitor_redis
  " > /dev/null 2>&1 &
  
  sleep 1
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "✅ Monitor started with PID: $PID"
    echo "📝 Logs saved to: $MONITOR_LOG"
    echo "   Tail logs with: tail -f $MONITOR_LOG"
  else
    echo "❌ Failed to start monitor"
  fi
}

# Stop monitoring
stop_monitor() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID"
      rm "$PID_FILE"
      echo "✅ Monitor stopped (PID: $PID)"
    else
      echo "⚠️  Monitor is not running"
      rm "$PID_FILE"
    fi
  else
    echo "⚠️  Monitor is not running"
  fi
}

# Check status
status_monitor() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "✅ Monitor is running (PID: $PID)"
      echo "📝 Log file: $MONITOR_LOG"
      echo "📊 Log size: $(du -h "$MONITOR_LOG" 2>/dev/null | cut -f1)"
      echo "📝 Last entries:"
      tail -5 "$MONITOR_LOG" | sed 's/^/   /'
    else
      echo "❌ Monitor PID file exists but process is not running"
      rm "$PID_FILE"
    fi
  else
    echo "❌ Monitor is not running"
  fi
}

# Tail logs
tail_logs() {
  if [ ! -f "$MONITOR_LOG" ]; then
    echo "❌ Log file not found: $MONITOR_LOG"
    echo "Start monitor first with: $0 start"
    exit 1
  fi
  
  echo "📝 Following Redis monitor logs (Ctrl+C to stop)..."
  echo ""
  tail -f "$MONITOR_LOG"
}

# Analyze logs
analyze_logs() {
  if [ ! -f "$MONITOR_LOG" ]; then
    echo "❌ Log file not found: $MONITOR_LOG"
    exit 1
  fi
  
  echo "📊 REDIS MONITOR ANALYSIS"
  echo ""
  
  echo "Total entries: $(grep -c "^║" "$MONITOR_LOG")"
  echo ""
  
  echo "Average KYC Keys over last 10 samples:"
  grep "KYC Keys:" "$MONITOR_LOG" | tail -10 | cut -d: -f2 | tr -d ' ' | awk '{sum+=$1} END {print "  " sum/NR " keys"}'
  
  echo ""
  echo "Average Memory Used (last 10 samples):"
  grep "used_memory_human:" "$MONITOR_LOG" | tail -10 | cut -d' ' -f3 | awk '{print $1}' | grep -o '[0-9.]*' | awk '{sum+=$1; count++} END {if(count>0) print "  " sum/count " units"}'
  
  echo ""
  echo "Error occurrences:"
  ERROR_COUNT=$(grep -c "ERROR:" "$MONITOR_LOG" || echo "0")
  echo "  Total errors logged: $ERROR_COUNT"
  
  echo ""
  echo "Top API responses:"
  grep "msg=" "$MONITOR_LOG" | grep -o "msg=[^,]*" | sort | uniq -c | sort -rn | head -5 | sed 's/^/  /'
}

# Show help
show_help() {
  cat << EOF
Redis Background Monitor Script
================================

Usage: sudo $0 [command]

Commands:
  start      - Start background monitoring (logs to $MONITOR_LOG)
  stop       - Stop background monitoring
  status     - Show monitor status
  tail       - Follow monitor logs in real-time
  analyze    - Analyze collected monitor data
  help       - Show this help message

Examples:
  sudo $0 start          # Start monitoring in background
  sudo $0 status         # Check if running
  sudo $0 tail           # View logs in real-time (Ctrl+C to exit)
  sudo $0 stop           # Stop monitoring

Log Location: $MONITOR_LOG

The monitor collects Redis metrics every $INTERVAL seconds including:
  - Memory usage
  - Total and KYC keys
  - Commands processed
  - Connected clients
  - Persistence status
  - Sample API responses
  - Errors and warnings

EOF
}

# Main
case "${1:-help}" in
  start)
    start_monitor
    ;;
  stop)
    stop_monitor
    ;;
  status)
    status_monitor
    ;;
  tail)
    tail_logs
    ;;
  analyze)
    analyze_logs
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    echo "❌ Unknown command: $1"
    show_help
    exit 1
    ;;
esac
