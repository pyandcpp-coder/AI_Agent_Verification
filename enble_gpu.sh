#!/bin/bash

# Enable GPU Mode - Now that packages are fixed

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}
╔══════════════════════════════════════════════════════╗
║           Enabling GPU Mode                          ║
╚══════════════════════════════════════════════════════╝
${NC}\n"

# Check if GPU is available
echo -e "${YELLOW}Checking GPU availability...${NC}"
if nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} GPU detected: ${GPU_NAME}\n"
else
    echo -e "${RED}✗${NC} No GPU detected. Cannot enable GPU mode."
    exit 1
fi

# Stop server
echo -e "${YELLOW}Stopping server...${NC}"
./stop_server.sh 2>/dev/null || pkill -f main.py || true
sleep 2
echo -e "${GREEN}✓${NC} Server stopped\n"

# Backup main.py
echo -e "${YELLOW}Creating backup...${NC}"
cp main.py main.py.backup_cpu
echo -e "${GREEN}✓${NC} Backup created: main.py.backup_cpu\n"

# Modify main.py to enable GPU
echo -e "${YELLOW}Modifying main.py to enable GPU...${NC}"

# Remove the line that forces CPU mode
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ''/# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU mode enabled/" main.py

# Change USE_GPU to True by default
sed -i 's/USE_GPU = False/USE_GPU = True  # GPU mode enabled/' main.py

echo -e "${GREEN}✓${NC} main.py updated for GPU mode\n"

# Test GPU with TensorFlow
echo -e "${YELLOW}Testing GPU with TensorFlow...${NC}"
source venv/bin/activate

python3 << 'EOFTEST'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ TensorFlow can see {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("✗ TensorFlow cannot see GPU")
        exit(1)
except Exception as e:
    print(f"✗ TensorFlow GPU test failed: {e}")
    exit(1)
EOFTEST

if [ $? -ne 0 ]; then
    echo -e "\n${RED}GPU test failed. Reverting to CPU mode...${NC}"
    cp main.py.backup_cpu main.py
    exit 1
fi

# Test GPU with PyTorch
echo -e "\n${YELLOW}Testing GPU with PyTorch...${NC}"

python3 << 'EOFTEST2'
import torch

if torch.cuda.is_available():
    print(f"✓ PyTorch CUDA available")
    print(f"  - Device: {torch.cuda.get_device_name(0)}")
    print(f"  - CUDA Version: {torch.version.cuda}")
    
    # Test actual GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"✓ GPU computation test passed")
else:
    print("✗ PyTorch CUDA not available")
    exit(1)
EOFTEST2

if [ $? -ne 0 ]; then
    echo -e "\n${RED}PyTorch GPU test failed. Reverting to CPU mode...${NC}"
    cp main.py.backup_cpu main.py
    exit 1
fi

echo -e "\n${GREEN}${BOLD}✓ GPU tests passed!${NC}\n"

# Start server
echo -e "${BLUE}Starting server in GPU mode...${NC}"
./start_server.sh

echo -e "\n${YELLOW}Waiting 20 seconds for server to initialize...${NC}"
for i in {20..1}; do
    echo -ne "  ${i}...\r"
    sleep 1
done
echo ""

# Check health
echo -e "\n${BLUE}Testing health endpoint...${NC}"
HEALTH=$(curl -s http://localhost:8101/health 2>/dev/null)

if echo "$HEALTH" | grep -q '"status":"healthy"'; then
    echo -e "\n${GREEN}${BOLD}
╔══════════════════════════════════════════════════════╗
║       🚀 GPU MODE ENABLED SUCCESSFULLY! 🚀           ║
╚══════════════════════════════════════════════════════╝
${NC}\n"
    
    echo "$HEALTH" | python -m json.tool
    
    # Check GPU usage
    echo -e "\n${BLUE}GPU Usage:${NC}"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    
    echo -e "\n${GREEN}${BOLD}System is now running in GPU mode!${NC}"
    echo -e "${BLUE}Monitor GPU: ${NC}watch -n 1 nvidia-smi"
    echo -e "${BLUE}Monitor logs: ${NC}tail -f server.log"
    
    echo -e "\n${YELLOW}Note: If you see CUDA errors in logs, run:${NC}"
    echo -e "  ${GREEN}./stop_server.sh${NC}"
    echo -e "  ${GREEN}cp main.py.backup_cpu main.py${NC}"
    echo -e "  ${GREEN}./start_server.sh${NC}"
    
else
    echo -e "\n${YELLOW}Health check response:${NC}"
    echo "$HEALTH"
    echo -e "\n${YELLOW}Check logs for any CUDA errors:${NC}"
    echo -e "  tail -50 server.log | grep -i 'cuda\\|gpu\\|error'"
fi
