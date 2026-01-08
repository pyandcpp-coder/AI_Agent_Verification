#!/bin/bash

# Quick Setup Script for KYC Verification System
# Fixes all common issues and gets the system running

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

clear
echo -e "${BOLD}${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════╗
║   KYC Verification System - Quick Setup & Fix         ║
║   Automated System Recovery & Configuration           ║
╚════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Log file
LOG_FILE="setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

print_step() {
    echo -e "\n${BOLD}${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    print_error "Please don't run as root"
    exit 1
fi

# ============================================================
# STEP 1: System Cleanup
# ============================================================
print_step "Step 1: System Cleanup"

echo "Stopping existing processes..."
pkill -f main.py 2>/dev/null || true
pkill -f batch_dispatcher.py 2>/dev/null || true
pkill -f uvicorn 2>/dev/null || true
sleep 2
print_success "Stopped all processes"

echo "Cleaning temporary files..."
rm -rf temp/* 2>/dev/null || true
rm -f server.pid nohup.out 2>/dev/null || true
print_success "Cleaned temporary files"

# ============================================================
# STEP 2: Check Prerequisites
# ============================================================
print_step "Step 2: Checking Prerequisites"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python3 not found"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_success "pip3 found"
else
    print_error "pip3 not found"
    echo "Install with: sudo apt-get install python3-pip"
    exit 1
fi

# Check virtual environment
if [ -d "venv" ]; then
    print_success "Virtual environment exists"
    source venv/bin/activate
else
    print_warning "Virtual environment not found"
    read -p "Create virtual environment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -m venv venv
        source venv/bin/activate
        print_success "Created and activated virtual environment"
    else
        print_error "Virtual environment required"
        exit 1
    fi
fi

# ============================================================
# STEP 3: GPU/CUDA Detection & Configuration
# ============================================================
print_step "Step 3: GPU/CUDA Configuration"

USE_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_success "GPU detected: $GPU_NAME"
        
        # Test CUDA compatibility
        echo "Testing CUDA/TensorFlow compatibility..."
        python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('COMPATIBLE')
        exit(0)
    else:
        print('NO_GPU')
        exit(1)
except Exception as e:
    print(f'ERROR: {e}')
    exit(1)
" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            print_success "CUDA/TensorFlow compatible - GPU mode enabled"
            USE_GPU=true
        else
            print_warning "CUDA/TensorFlow incompatible - switching to CPU mode"
            export CUDA_VISIBLE_DEVICES=''
            USE_GPU=false
        fi
    else
        print_warning "nvidia-smi failed - using CPU mode"
        export CUDA_VISIBLE_DEVICES=''
    fi
else
    print_warning "No GPU detected - using CPU mode"
    export CUDA_VISIBLE_DEVICES=''
fi

# Save GPU setting
if [ "$USE_GPU" = false ]; then
    echo 'export CUDA_VISIBLE_DEVICES=""' >> ~/.bashrc
    print_success "CPU mode will persist after reboot"
fi

# ============================================================
# STEP 4: Install/Update Dependencies
# ============================================================
print_step "Step 4: Installing Dependencies"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_warning "requirements.txt not found"
    read -p "Create from template? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Create basic requirements.txt
        cat > requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
tensorflow==2.15.0
torch==2.1.2
torchvision==0.16.2
opencv-python==4.9.0.80
deepface==0.0.79
ultralytics==8.1.9
aiohttp==3.9.1
aiofiles==23.2.1
cloudscraper==1.2.71
redis==5.0.1
pytesseract==0.3.10
paddleocr==2.7.0.3
numpy==1.26.3
pandas==2.1.4
Pillow==10.2.0
pydantic==2.5.3
EOF
        print_success "Created requirements.txt"
    fi
fi

echo "Installing Python packages (this may take a few minutes)..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q --no-cache-dir

if [ $? -eq 0 ]; then
    print_success "All packages installed successfully"
else
    print_error "Package installation failed"
    echo "Try manually: pip install -r requirements.txt"
    exit 1
fi

# ============================================================
# STEP 5: Redis Check & Setup
# ============================================================
print_step "Step 5: Redis Configuration"

if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        REDIS_VERSION=$(redis-cli info server | grep redis_version | cut -d: -f2 | tr -d '\r')
        print_success "Redis is running (version $REDIS_VERSION)"
    else
        print_warning "Redis not responding"
        
        if command -v redis-server &> /dev/null; then
            echo "Starting Redis..."
            redis-server --daemonize yes 2>/dev/null
            sleep 2
            
            if redis-cli ping &> /dev/null; then
                print_success "Redis started successfully"
            else
                print_error "Failed to start Redis"
            fi
        else
            print_error "Redis not installed"
            echo "Install with: sudo apt-get install redis-server"
        fi
    fi
else
    print_warning "Redis not found - some features may be limited"
    echo "Install with: sudo apt-get install redis-server"
fi

# ============================================================
# STEP 6: Verify Model Files
# ============================================================
print_step "Step 6: Verifying Model Files"

MODELS_OK=true
if [ -f "models/best4.pt" ]; then
    SIZE=$(du -h models/best4.pt | cut -f1)
    print_success "Detection model found (${SIZE})"
else
    print_error "Detection model missing: models/best4.pt"
    MODELS_OK=false
fi

if [ -f "models/best.pt" ]; then
    SIZE=$(du -h models/best.pt | cut -f1)
    print_success "Extraction model found (${SIZE})"
else
    print_error "Extraction model missing: models/best.pt"
    MODELS_OK=false
fi

if [ "$MODELS_OK" = false ]; then
    print_error "Model files missing - system will not work properly"
    echo "Please ensure model files are in the models/ directory"
fi

# ============================================================
# STEP 7: Create Necessary Scripts
# ============================================================
print_step "Step 7: Creating Helper Scripts"

# Create start script
cat > start_server.sh << 'EOFSTART'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=''  # Force CPU mode for stability
source venv/bin/activate
echo "Starting KYC Verification Server..."
nohup python3 main.py > server.log 2>&1 &
echo $! > server.pid
sleep 3
if ps -p $(cat server.pid) > /dev/null; then
    echo "✓ Server started (PID: $(cat server.pid))"
    echo "✓ Logs: tail -f server.log"
    echo "✓ Health: curl http://localhost:8101/health"
else
    echo "✗ Server failed to start - check server.log"
fi
EOFSTART

chmod +x start_server.sh
print_success "Created start_server.sh"

# Create stop script  
cat > stop_server.sh << 'EOFSTOP'
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
EOFSTOP

chmod +x stop_server.sh
print_success "Created stop_server.sh"

# Create monitoring script
cat > monitor.sh << 'EOFMON'
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
EOFMON

chmod +x monitor.sh
print_success "Created monitor.sh"

# ============================================================
# STEP 8: Test System Components
# ============================================================
print_step "Step 8: Testing System Components"

echo "Testing imports..."
python3 << 'EOFTEST'
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

errors = []

try:
    import tensorflow as tf
    print("✓ TensorFlow OK")
except Exception as e:
    print(f"✗ TensorFlow failed: {e}")
    errors.append("tensorflow")

try:
    import torch
    print("✓ PyTorch OK")
except Exception as e:
    print(f"✗ PyTorch failed: {e}")
    errors.append("torch")

try:
    import cv2
    print("✓ OpenCV OK")
except Exception as e:
    print(f"✗ OpenCV failed: {e}")
    errors.append("opencv")

try:
    import deepface
    print("✓ DeepFace OK")
except Exception as e:
    print(f"✗ DeepFace failed: {e}")
    errors.append("deepface")

try:
    from ultralytics import YOLO
    print("✓ YOLO OK")
except Exception as e:
    print(f"✗ YOLO failed: {e}")
    errors.append("yolo")

sys.exit(len(errors))
EOFTEST

if [ $? -eq 0 ]; then
    print_success "All components working"
else
    print_warning "Some components have issues"
fi

# ============================================================
# STEP 9: Final Configuration
# ============================================================
print_step "Step 9: Final Configuration"

# Create directories
mkdir -p temp logs models
print_success "Created required directories"

# Set permissions
chmod -R 755 temp logs
print_success "Set directory permissions"

# ============================================================
# Summary & Next Steps
# ============================================================
echo -e "\n${BOLD}${GREEN}"
cat << "EOF"
╔════════════════════════════════════════════════════════╗
║             Setup Complete!                            ║
╚════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BOLD}System Configuration:${NC}"
echo "  GPU Mode: $([ "$USE_GPU" = true ] && echo '✓ Enabled' || echo '✗ Disabled (CPU mode)')"
echo "  Redis: $(redis-cli ping &>/dev/null && echo '✓ Running' || echo '✗ Not running')"
echo "  Models: $([ "$MODELS_OK" = true ] && echo '✓ Found' || echo '✗ Missing')"
echo

echo -e "${BOLD}Quick Commands:${NC}"
echo "  Start server:   ${GREEN}./start_server.sh${NC}"
echo "  Stop server:    ${GREEN}./stop_server.sh${NC}"
echo "  Monitor logs:   ${GREEN}./monitor.sh${NC}"
echo "  Check health:   ${GREEN}curl http://localhost:8101/health${NC}"
echo

echo -e "${BOLD}Next Steps:${NC}"
echo "  1. Start the server: ${GREEN}./start_server.sh${NC}"
echo "  2. Check health: ${GREEN}curl http://localhost:8101/health${NC}"
echo "  3. Monitor logs: ${GREEN}tail -f server.log${NC}"
echo

if [ "$MODELS_OK" = false ]; then
    echo -e "${BOLD}${RED}⚠ WARNING: Model files are missing!${NC}"
    echo "  Please add model files to the models/ directory before starting"
    echo
fi

echo -e "${BOLD}Setup log saved to:${NC} $LOG_FILE"
echo
