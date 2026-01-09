#!/bin/bash

# Redis Monitoring Dashboard for Headless Ubuntu Server
# Usage: ./redis-dashboard.sh
# Press Ctrl+C to exit

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear_screen() {
  clear
}

print_header() {
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}     Redis Monitoring Dashboard${NC}"
  echo -e "${BLUE}========================================${NC}"
  echo ""
}

check_redis() {
  if ! redis-cli ping > /dev/null 2>&1; then
    echo -e "${RED}❌ Redis is not running!${NC}"
    echo "Start Redis with: sudo systemctl start redis-server"
    exit 1
  fi
}

print_connection_status() {
  echo -e "${GREEN}✅ Redis Connected${NC}"
  echo ""
}

print_storage_info() {
  echo -e "${YELLOW}📊 STORAGE INFORMATION${NC}"
  
  # Total keys
  TOTAL_KEYS=$(redis-cli DBSIZE | grep -o '[0-9]*')
  echo "  Total Keys in Redis: $TOTAL_KEYS"
  
  # KYC Keys
  KYC_KEYS=$(redis-cli KEYS "kyc_verification:*" 2>/dev/null | wc -l)
  echo "  KYC Verification Keys: $KYC_KEYS"
  
  # Memory
  USED_MEM=$(redis-cli INFO memory 2>/dev/null | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')
  MAX_MEM=$(redis-cli CONFIG GET "maxmemory" 2>/dev/null | tail -1)
  echo "  Memory Used: $USED_MEM"
  
  if [ "$MAX_MEM" != "0" ]; then
    echo "  Max Memory: $MAX_MEM bytes"
  fi
  
  echo ""
}

print_persistence_info() {
  echo -e "${YELLOW}💾 PERSISTENCE INFORMATION${NC}"
  
  # RDB file
  if [ -f "/var/lib/redis/dump.rdb" ]; then
    RDB_SIZE=$(du -h /var/lib/redis/dump.rdb 2>/dev/null | cut -f1)
    RDB_TIME=$(stat /var/lib/redis/dump.rdb 2>/dev/null | grep Modify | cut -d: -f2-)
    echo "  RDB File: $RDB_SIZE"
    echo "  Last Save: $RDB_TIME"
  else
    echo "  RDB File: Not found"
  fi
  
  # Check if AOF is enabled
  AOF_STATUS=$(redis-cli CONFIG GET "appendonly" 2>/dev/null | tail -1)
  echo "  AOF Enabled: $AOF_STATUS"
  
  echo ""
}

print_performance_info() {
  echo -e "${YELLOW}⚡ PERFORMANCE INFORMATION${NC}"
  
  # Connected clients
  CLIENTS=$(redis-cli INFO clients 2>/dev/null | grep connected_clients | cut -d: -f2 | tr -d '\r')
  echo "  Connected Clients: $CLIENTS"
  
  # Uptime
  UPTIME=$(redis-cli INFO server 2>/dev/null | grep uptime_in_seconds | cut -d: -f2 | tr -d '\r')
  UPTIME_DAYS=$((UPTIME / 86400))
  UPTIME_HOURS=$(((UPTIME % 86400) / 3600))
  echo "  Uptime: ${UPTIME_DAYS}d ${UPTIME_HOURS}h"
  
  # Commands processed
  COMMANDS=$(redis-cli INFO stats 2>/dev/null | grep total_commands_processed | cut -d: -f2 | tr -d '\r')
  echo "  Commands Processed: $COMMANDS"
  
  echo ""
}

print_api_responses() {
  echo -e "${YELLOW}📨 API RESPONSES SAMPLE${NC}"
  
  # Get first 3 KYC entries
  COUNT=0
  redis-cli KEYS "kyc_verification:*" 2>/dev/null | while read key; do
    if [ $COUNT -lt 3 ]; then
      USER_ID=$(echo "$key" | cut -d: -f2)
      API_RESPONSE=$(redis-cli GET "$key" 2>/dev/null | jq -r '.api_response.message // "No message"' 2>/dev/null || echo "Error parsing")
      echo "  User $USER_ID: $API_RESPONSE"
      COUNT=$((COUNT + 1))
    fi
  done
  
  echo ""
}

print_recent_logs() {
  echo -e "${YELLOW}📝 RECENT LOGS (Last 5 lines)${NC}"
  
  if [ -f "/var/log/redis/redis-server.log" ]; then
    tail -5 /var/log/redis/redis-server.log | while read line; do
      echo "  $line"
    done
  else
    echo "  Log file not found at /var/log/redis/redis-server.log"
  fi
  
  echo ""
}

print_menu() {
  echo -e "${BLUE}COMMANDS:${NC}"
  echo "  1. Refresh dashboard (auto-refresh every 5s)"
  echo "  2. View all KYC users"
  echo "  3. View specific user data"
  echo "  4. Monitor all Redis commands in real-time"
  echo "  5. View Redis logs (tail -f)"
  echo "  6. Check Redis health"
  echo "  q. Quit"
  echo ""
}

view_all_users() {
  clear_screen
  print_header
  
  echo -e "${YELLOW}📋 ALL CACHED KYC USERS${NC}"
  echo ""
  
  COUNT=$(redis-cli KEYS "kyc_verification:*" 2>/dev/null | wc -l)
  echo "Total Users: $COUNT"
  echo ""
  
  redis-cli KEYS "kyc_verification:*" 2>/dev/null | while read key; do
    USER_ID=$(echo "$key" | cut -d: -f2)
    DECISION=$(redis-cli GET "$key" 2>/dev/null | jq -r '.verification_result.final_decision // "Unknown"' 2>/dev/null || echo "Unknown")
    API_SUCCESS=$(redis-cli GET "$key" 2>/dev/null | jq -r '.api_response.success // "Unknown"' 2>/dev/null || echo "Unknown")
    
    if [ "$API_SUCCESS" = "true" ]; then
      API_ICON="✅"
    elif [ "$API_SUCCESS" = "false" ]; then
      API_ICON="❌"
    else
      API_ICON="⚠️"
    fi
    
    echo "  $USER_ID | Decision: $DECISION | API: $API_ICON"
  done
  
  echo ""
  read -p "Press Enter to go back..."
}

view_specific_user() {
  clear_screen
  print_header
  
  read -p "Enter User ID: " USER_ID
  
  DATA=$(redis-cli GET "kyc_verification:$USER_ID" 2>/dev/null)
  
  if [ -z "$DATA" ]; then
    echo -e "${RED}❌ User $USER_ID not found in cache${NC}"
  else
    echo -e "${YELLOW}📦 DATA FOR USER: $USER_ID${NC}"
    echo ""
    echo "$DATA" | jq '.' 2>/dev/null || echo "$DATA"
  fi
  
  echo ""
  read -p "Press Enter to go back..."
}

monitor_commands() {
  clear_screen
  print_header
  echo -e "${YELLOW}⏱️  Monitoring Redis commands (Ctrl+C to stop)${NC}"
  echo ""
  
  redis-cli MONITOR 2>/dev/null | grep -i "kyc_verification" || redis-cli MONITOR 2>/dev/null
}

view_logs() {
  clear_screen
  print_header
  
  if [ -f "/var/log/redis/redis-server.log" ]; then
    echo -e "${YELLOW}📝 Redis Logs (Ctrl+C to stop)${NC}"
    echo ""
    tail -f /var/log/redis/redis-server.log
  else
    echo -e "${RED}❌ Log file not found at /var/log/redis/redis-server.log${NC}"
    read -p "Press Enter to go back..."
  fi
}

check_health() {
  clear_screen
  print_header
  
  echo -e "${YELLOW}🏥 REDIS HEALTH CHECK${NC}"
  echo ""
  
  # Ping test
  PING=$(redis-cli PING 2>/dev/null)
  if [ "$PING" = "PONG" ]; then
    echo -e "${GREEN}✅ Ping: OK${NC}"
  else
    echo -e "${RED}❌ Ping: FAILED${NC}"
  fi
  
  # Memory check
  USED=$(redis-cli INFO memory 2>/dev/null | grep "used_memory:" | cut -d: -f2 | tr -d '\r')
  MAX=$(redis-cli CONFIG GET "maxmemory" 2>/dev/null | tail -1)
  
  if [ "$MAX" != "0" ]; then
    USAGE=$((USED * 100 / MAX))
    echo "Memory Usage: $USAGE%"
    
    if [ $USAGE -gt 90 ]; then
      echo -e "${RED}⚠️  WARNING: Memory usage is critical!${NC}"
    elif [ $USAGE -gt 70 ]; then
      echo -e "${YELLOW}⚠️  WARNING: Memory usage is high!${NC}"
    else
      echo -e "${GREEN}✅ Memory: OK${NC}"
    fi
  fi
  
  # Error check
  ERRORS=$(redis-cli INFO stats 2>/dev/null | grep total_error_replies | cut -d: -f2 | tr -d '\r')
  if [ "$ERRORS" != "0" ]; then
    echo -e "${YELLOW}⚠️  Errors detected: $ERRORS${NC}"
  else
    echo -e "${GREEN}✅ Errors: None${NC}"
  fi
  
  # Connection check
  CLIENTS=$(redis-cli INFO clients 2>/dev/null | grep connected_clients | cut -d: -f2 | tr -d '\r')
  echo -e "${GREEN}✅ Connected Clients: $CLIENTS${NC}"
  
  echo ""
  read -p "Press Enter to go back..."
}

auto_refresh_dashboard() {
  while true; do
    clear_screen
    print_header
    print_connection_status
    print_storage_info
    print_persistence_info
    print_performance_info
    print_api_responses
    print_recent_logs
    
    echo -e "${BLUE}Next refresh in 5 seconds... (Press Ctrl+C to stop)${NC}"
    sleep 5
  done
}

# Main loop
main() {
  check_redis
  
  while true; do
    clear_screen
    print_header
    print_connection_status
    print_storage_info
    print_persistence_info
    print_performance_info
    print_api_responses
    print_recent_logs
    print_menu
    
    read -p "Choose option: " choice
    
    case $choice in
      1)
        auto_refresh_dashboard
        ;;
      2)
        view_all_users
        ;;
      3)
        view_specific_user
        ;;
      4)
        monitor_commands
        ;;
      5)
        view_logs
        ;;
      6)
        check_health
        ;;
      q|Q)
        echo "Goodbye!"
        exit 0
        ;;
      *)
        echo "Invalid option"
        read -p "Press Enter to continue..."
        ;;
    esac
  done
}

# Run main
main
