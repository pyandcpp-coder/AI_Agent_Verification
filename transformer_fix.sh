#!/bin/bash

# Final Fix: Upgrade transformers to 4.38+ for Qwen2VL support

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}
╔══════════════════════════════════════════════════════╗
║   Upgrading Transformers to 4.38+ (Qwen2VL)         ║
╚══════════════════════════════════════════════════════╝
${NC}\n"

source venv/bin/activate

echo -e "${YELLOW}Upgrading transformers to latest with Qwen2VL support...${NC}"
pip install --upgrade 'transformers>=4.38.0'
echo -e "${GREEN}✓${NC} Transformers upgraded\n"

echo -e "${YELLOW}Verifying Qwen2VL availability...${NC}"
python3 << 'EOFTEST'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("\n" + "="*60)
print("Qwen2VL Verification")
print("="*60 + "\n")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch: {e}")
    exit(1)

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers: {e}")
    exit(1)

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"✓ Qwen2VLForConditionalGeneration: Available")
    print(f"✓ AutoProcessor: Available")
    print(f"\n✅ QWEN2VL READY!")
except ImportError as e:
    print(f"✗ Qwen2VL import failed: {e}")
    print(f"\nInstalled transformers version may still be too old.")
    print(f"Try: pip install transformers==4.45.0")
    exit(1)

print("="*60)
exit(0)
EOFTEST

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}
╔══════════════════════════════════════════════════════╗
║         ✓✓✓ QWEN2VL WORKING ✓✓✓                      ║
╚══════════════════════════════════════════════════════╝
${NC}\n"
    
    echo -e "${BLUE}Final Stack:${NC}"
    python3 << 'EOFCONFIG'
import torch
import transformers
print(f"  PyTorch: {torch.__version__}")
print(f"  Transformers: {transformers.__version__}")
print(f"  CUDA: {'Yes' if torch.cuda.is_available() else 'No (CPU mode)'}")
from transformers import Qwen2VLForConditionalGeneration
print(f"  Qwen2VL: ✓ Ready")
EOFCONFIG
    
    echo -e "\n${BLUE}Starting server...${NC}"
    ./start_server.sh
    
    echo -e "\n${YELLOW}Waiting 20 seconds for server startup...${NC}"
    for i in {20..1}; do
        echo -ne "  ${i} seconds remaining...\r"
        sleep 1
    done
    echo ""
    
    echo -e "\n${BLUE}Testing health endpoint...${NC}"
    HEALTH=$(curl -s http://localhost:8101/health 2>/dev/null)
    
    if echo "$HEALTH" | grep -q '"status":"healthy"'; then
        echo -e "\n${GREEN}${BOLD}
╔══════════════════════════════════════════════════════╗
║     🎉 SERVER IS HEALTHY AND READY! 🎉               ║
╚══════════════════════════════════════════════════════╝
${NC}\n"
        echo "$HEALTH" | python -m json.tool
        
        echo -e "\n${GREEN}${BOLD}ALL SYSTEMS GO!${NC}"
        echo -e "${BLUE}Commands:${NC}"
        echo -e "  Monitor logs: ${GREEN}tail -f server.log${NC}"
        echo -e "  View health:  ${GREEN}curl http://localhost:8101/health${NC}"
        echo -e "  Stop server:  ${GREEN}./stop_server.sh${NC}"
        
        echo -e "\n${BLUE}Test with batch dispatcher:${NC}"
        echo -e "  ${GREEN}python batch_dispatcher.py${NC}"
    else
        echo -e "\n${YELLOW}Health check response:${NC}"
        echo "$HEALTH"
        echo -e "\n${YELLOW}Server may still be starting. Check logs:${NC}"
        echo -e "  ${GREEN}tail -f server.log${NC}"
    fi
    
else
    echo -e "\n${RED}${BOLD}Verification failed${NC}"
    echo -e "\nManual fix:"
    echo -e "  pip install transformers==4.45.0"
    exit 1
fi
