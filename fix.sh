#!/bin/bash

# Emergency Fix Script - Resolves PyTorch/Transformers conflict
# Run with: bash emergency_fix.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Emergency Fix - Version Conflict${NC}"
echo -e "${BLUE}================================${NC}\n"

# Stop server
echo -e "${YELLOW}Stopping server...${NC}"
./stop_server.sh 2>/dev/null || pkill -f main.py || true
sleep 2

# Ask user for fix approach
echo -e "\n${BLUE}Choose fix approach:${NC}"
echo "  1. Quick Fix - Disable Qwen fallback (30 seconds) ${GREEN}[RECOMMENDED]${NC}"
echo "  2. Full Fix - Downgrade PyTorch (5 minutes)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo -e "\n${BLUE}Applying Quick Fix...${NC}"
    
    # Disable Qwen fallback
    if grep -q "enable_qwen_fallback=True" main.py; then
        sed -i 's/enable_qwen_fallback=True/enable_qwen_fallback=False/' main.py
        echo -e "${GREEN}✓${NC} Disabled Qwen fallback"
    else
        echo -e "${YELLOW}⚠${NC} Qwen fallback already disabled or not found"
    fi
    
elif [ "$choice" = "2" ]; then
    echo -e "\n${BLUE}Applying Full Fix (this will take a few minutes)...${NC}"
    
    # Activate venv
    source venv/bin/activate
    
    # Uninstall problematic versions
    echo -e "\n${YELLOW}Uninstalling current PyTorch...${NC}"
    pip uninstall torch torchvision torchaudio -y
    
    # Install stable versions
    echo -e "\n${YELLOW}Installing PyTorch 2.1.2...${NC}"
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --quiet
    
    echo -e "\n${YELLOW}Installing compatible transformers...${NC}"
    pip install transformers==4.37.2 --quiet
    pip install qwen-vl-utils --quiet 2>/dev/null || echo "qwen-vl-utils optional, skipping"
    
    echo -e "${GREEN}✓${NC} PyTorch downgraded to 2.1.2"
    echo -e "${GREEN}✓${NC} Transformers updated to 4.37.2"
    
else
    echo -e "${RED}Invalid choice${NC}"
    exit 1
fi

# Test imports
echo -e "\n${BLUE}Testing imports...${NC}"
python3 << 'EOF'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

errors = []

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch failed: {e}")
    errors.append("torch")

try:
    from ultralytics import YOLO
    print("✓ YOLO OK")
except Exception as e:
    print(f"✗ YOLO failed: {e}")
    errors.append("yolo")

try:
    import tensorflow as tf
    print("✓ TensorFlow OK")
except Exception as e:
    print(f"✗ TensorFlow failed: {e}")
    errors.append("tensorflow")

if errors:
    print(f"\n⚠ Some imports failed: {', '.join(errors)}")
    exit(1)
else:
    print("\n✓ All critical imports working!")
    exit(0)
EOF

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}================================${NC}"
    echo -e "${GREEN}Fix Applied Successfully!${NC}"
    echo -e "${GREEN}================================${NC}"
    
    # Start server
    echo -e "\n${BLUE}Starting server...${NC}"
    ./start_server.sh
    
    # Wait and test
    echo -e "\n${YELLOW}Waiting 10 seconds for server to start...${NC}"
    sleep 10
    
    echo -e "\n${BLUE}Testing health endpoint...${NC}"
    HEALTH=$(curl -s http://localhost:8101/health 2>/dev/null)
    
    if echo "$HEALTH" | grep -q '"status":"healthy"'; then
        echo -e "${GREEN}✓ Server is healthy!${NC}\n"
        echo "$HEALTH" | python -m json.tool
    else
        echo -e "${YELLOW}⚠ Server started but health check unclear${NC}"
        echo "Check logs: tail -f server.log"
    fi
    
else
    echo -e "\n${RED}================================${NC}"
    echo -e "${RED}Import test failed!${NC}"
    echo -e "${RED}================================${NC}"
    echo -e "\nTry the other fix approach or check logs"
    exit 1
fi
