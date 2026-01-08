#!/bin/bash

# Final Fix: Upgrade PyTorch to 2.2.0 for transformers 4.37 compatibility

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}
╔══════════════════════════════════════════════════════╗
║   Final Fix: PyTorch 2.2.0 + Transformers 4.37      ║
╚══════════════════════════════════════════════════════╝
${NC}\n"

source venv/bin/activate

echo -e "${YELLOW}Step 1: Uninstalling PyTorch 2.1.2...${NC}"
pip uninstall torch torchvision torchaudio -y
echo -e "${GREEN}✓${NC} Uninstalled\n"

echo -e "${YELLOW}Step 2: Installing PyTorch 2.2.0 (with register_pytree_node support)...${NC}"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
echo -e "${GREEN}✓${NC} PyTorch 2.2.0 installed\n"

echo -e "${YELLOW}Step 3: Reinstalling transformers 4.37.2...${NC}"
pip install transformers==4.37.2
echo -e "${GREEN}✓${NC} Transformers 4.37.2 installed\n"

echo -e "${YELLOW}Step 4: Verifying compatibility...${NC}"
python3 << 'EOFTEST'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("\n" + "="*60)
print("Final Compatibility Test")
print("="*60 + "\n")

errors = []

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    
    # Test if register_pytree_node exists
    if hasattr(torch.utils._pytree, 'register_pytree_node'):
        print(f"  ✓ register_pytree_node: Available")
    else:
        print(f"  ✗ register_pytree_node: Missing")
        errors.append("pytree")
except Exception as e:
    print(f"✗ PyTorch failed: {e}")
    errors.append("torch")

# Test Transformers (this was failing before)
try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers failed: {e}")
    errors.append("transformers")

# Test Qwen2VL import
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"✓ Qwen2VL imports successful")
except ImportError as e:
    print(f"✗ Qwen2VL import failed: {e}")
    errors.append("qwen2vl")

# Test YOLO
try:
    from ultralytics import YOLO
    print(f"✓ YOLO OK")
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
    print(f"⚠ Some components failed: {', '.join(errors)}")
    exit(1)
else:
    print("✅ ALL TESTS PASSED!")
    exit(0)
EOFTEST

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}
╔══════════════════════════════════════════════════════╗
║            ✓ FIX SUCCESSFUL ✓                        ║
╚══════════════════════════════════════════════════════╝
${NC}\n"
    
    echo -e "${BLUE}Final Configuration:${NC}"
    python3 << 'EOFCONFIG'
import torch
import transformers
print(f"  PyTorch: {torch.__version__}")
print(f"  Transformers: {transformers.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
print(f"  Qwen2VL: ✓ Ready")
EOFCONFIG
    
    echo -e "\n${BLUE}Starting server...${NC}"
    ./start_server.sh
    
    echo -e "\n${YELLOW}Waiting 15 seconds for initialization...${NC}"
    sleep 15
    
    echo -e "\n${BLUE}Testing health endpoint...${NC}"
    HEALTH=$(curl -s http://localhost:8101/health 2>/dev/null)
    
    if echo "$HEALTH" | grep -q '"status":"healthy"'; then
        echo -e "\n${GREEN}✓✓✓ SERVER IS HEALTHY ✓✓✓${NC}\n"
        echo "$HEALTH" | python -m json.tool
        
        echo -e "\n${GREEN}${BOLD}SYSTEM READY FOR PRODUCTION!${NC}"
        echo -e "${BLUE}Monitor: ${NC}tail -f server.log"
        echo -e "${BLUE}Health:  ${NC}curl http://localhost:8101/health"
    else
        echo -e "${YELLOW}Health response:${NC}"
        echo "$HEALTH"
        echo -e "\n${YELLOW}Check logs:${NC} tail -f server.log"
    fi
    
else
    echo -e "\n${RED}${BOLD}
╔══════════════════════════════════════════════════════╗
║            ✗ VERIFICATION FAILED ✗                   ║
╚══════════════════════════════════════════════════════╝
${NC}\n"
    
    echo -e "${YELLOW}Last resort options:${NC}"
    echo "1. Try PyTorch 2.3.0:"
    echo "   pip install torch==2.3.0 torchvision==0.18.0"
    echo ""
    echo "2. Use older transformers (without Qwen):"
    echo "   pip install transformers==4.35.0"
    echo "   # Then disable Qwen in main.py: enable_qwen_fallback=False"
    
    exit 1
fi
