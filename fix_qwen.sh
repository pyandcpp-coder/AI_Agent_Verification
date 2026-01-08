#!/bin/bash

# Complete Fix for Qwen + PyTorch + Transformers
# This script downgrades PyTorch and upgrades transformers to compatible versions

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════╗
║     Fixing Qwen + PyTorch + Transformers             ║
║     This will take 5-10 minutes                      ║
╚══════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Stop server
echo -e "${YELLOW}Step 1: Stopping server...${NC}"
./stop_server.sh 2>/dev/null || pkill -f main.py || true
sleep 2
echo -e "${GREEN}✓${NC} Server stopped\n"

# Activate venv
echo -e "${YELLOW}Step 2: Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated\n"

# Backup current versions
echo -e "${YELLOW}Step 3: Checking current versions...${NC}"
python3 -c "import torch; print(f'Current PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch not importable"
python3 -c "import transformers; print(f'Current Transformers: {transformers.__version__}')" 2>/dev/null || echo "Transformers not importable"
echo ""

# Uninstall conflicting packages
echo -e "${YELLOW}Step 4: Uninstalling conflicting packages...${NC}"
pip uninstall torch torchvision torchaudio transformers -y
echo -e "${GREEN}✓${NC} Removed old versions\n"

# Install compatible PyTorch (2.1.2 - stable and well-tested)
echo -e "${YELLOW}Step 5: Installing PyTorch 2.1.2 (stable version)...${NC}"
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
echo -e "${GREEN}✓${NC} PyTorch 2.1.2 installed\n"

# Install transformers with Qwen2VL support
echo -e "${YELLOW}Step 6: Installing transformers with Qwen2VL support...${NC}"
pip install transformers>=4.37.0
pip install accelerate
pip install qwen-vl-utils 2>/dev/null || echo "qwen-vl-utils optional, continuing..."
echo -e "${GREEN}✓${NC} Transformers installed with Qwen2VL support\n"

# Install other required packages
echo -e "${YELLOW}Step 7: Ensuring all dependencies are installed...${NC}"
pip install sentencepiece protobuf
echo -e "${GREEN}✓${NC} Dependencies installed\n"

# Verify installations
echo -e "${YELLOW}Step 8: Verifying installations...${NC}"
python3 << 'EOFTEST'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("\n" + "="*60)
print("Verification Test")
print("="*60 + "\n")

errors = []

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    
    # Test basic operation
    x = torch.randn(2, 2)
    y = x + 1
    print(f"  - Tensor operations: OK")
except Exception as e:
    print(f"✗ PyTorch failed: {e}")
    errors.append("torch")

# Test Transformers
try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers failed: {e}")
    errors.append("transformers")

# Test Qwen2VL import (the critical one)
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"✓ Qwen2VL import successful!")
    print(f"  - Qwen2VLForConditionalGeneration: Available")
    print(f"  - AutoProcessor: Available")
except ImportError as e:
    print(f"✗ Qwen2VL import failed: {e}")
    errors.append("qwen2vl")

# Test YOLO
try:
    from ultralytics import YOLO
    print(f"✓ YOLO import successful")
except Exception as e:
    print(f"✗ YOLO failed: {e}")
    errors.append("yolo")

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow failed: {e}")
    errors.append("tensorflow")

print("\n" + "="*60)

if errors:
    print(f"⚠ WARNING: Some components failed: {', '.join(errors)}")
    print("="*60 + "\n")
    exit(1)
else:
    print("✅ ALL COMPONENTS VERIFIED SUCCESSFULLY!")
    print("="*60 + "\n")
    exit(0)
EOFTEST

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}${BOLD}║           Fix Applied Successfully! ✓                ║${NC}"
    echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "${BLUE}Installed Versions:${NC}"
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
    python3 -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
    python3 -c "from transformers import Qwen2VLForConditionalGeneration; print('  Qwen2VL: ✓ Available')"
    
    echo -e "\n${BLUE}Starting server...${NC}"
    ./start_server.sh
    
    echo -e "\n${YELLOW}Waiting 15 seconds for server initialization...${NC}"
    sleep 15
    
    echo -e "\n${BLUE}Testing health endpoint...${NC}"
    HEALTH=$(curl -s http://localhost:8101/health 2>/dev/null)
    
    if echo "$HEALTH" | grep -q '"status":"healthy"'; then
        echo -e "${GREEN}✓ Server is healthy!${NC}\n"
        echo "$HEALTH" | python -m json.tool
        
        echo -e "\n${GREEN}${BOLD}System is ready for production!${NC}"
        echo -e "${BLUE}Monitor logs: ${NC}tail -f server.log"
    else
        echo -e "${YELLOW}⚠ Health check response:${NC}"
        echo "$HEALTH"
        echo -e "\n${YELLOW}Check logs for details:${NC} tail -f server.log"
    fi
    
else
    echo -e "\n${RED}${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}${BOLD}║              Verification Failed!                    ║${NC}"
    echo -e "${RED}${BOLD}╚══════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "${YELLOW}Troubleshooting steps:${NC}"
    echo "1. Check if you have enough disk space: df -h"
    echo "2. Try manual installation:"
    echo "   pip install torch==2.1.2 torchvision==0.16.2"
    echo "   pip install 'transformers>=4.37.0'"
    echo "   pip install accelerate sentencepiece"
    echo "3. Check Python version: python3 --version (need 3.8+)"
    
    exit 1
fi
