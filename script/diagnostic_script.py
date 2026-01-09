#!/usr/bin/env python3
"""
System Diagnostic Script for KYC Verification System
Checks all components and suggests fixes
"""

import sys
import subprocess
import os
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def check_status(condition, message):
    status = f"{Colors.GREEN}✅" if condition else f"{Colors.RED}❌"
    print(f"{status} {message}{Colors.END}")
    return condition

def run_command(cmd, capture=True):
    """Run shell command and return output"""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.returncode == 0, result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True, timeout=10)
            return result.returncode == 0, ""
    except Exception as e:
        return False, str(e)

def check_python_packages():
    """Check if required packages are installed"""
    print_section("Python Package Check")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'torch': 'PyTorch',
        'fastapi': 'FastAPI',
        'aiohttp': 'AsyncIO HTTP',
        'redis': 'Redis Client',
        'cloudscraper': 'CloudScraper',
        'deepface': 'DeepFace',
        'ultralytics': 'YOLO/Ultralytics',
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            check_status(True, f"{name} installed")
        except ImportError:
            check_status(False, f"{name} NOT installed")
            all_ok = False
    
    return all_ok

def check_gpu_cuda():
    """Check GPU and CUDA setup"""
    print_section("GPU & CUDA Check")
    
    # Check NVIDIA driver
    success, nvidia_output = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    if success:
        print(f"{Colors.GREEN}✅ NVIDIA Driver detected:{Colors.END}")
        print(f"   {nvidia_output}")
    else:
        check_status(False, "NVIDIA Driver not detected")
    
    # Check CUDA version
    success, cuda_version = run_command("nvcc --version | grep 'release' | awk '{print $5}'")
    if success:
        check_status(True, f"CUDA Toolkit: {cuda_version}")
    else:
        check_status(False, "CUDA Toolkit not found (nvcc not available)")
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            check_status(True, f"TensorFlow can see {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            check_status(False, "TensorFlow cannot see any GPUs")
    except Exception as e:
        check_status(False, f"TensorFlow GPU check failed: {e}")
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            check_status(True, f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            check_status(False, "PyTorch CUDA not available")
    except Exception as e:
        check_status(False, f"PyTorch check failed: {e}")

def check_compatibility():
    """Check version compatibility"""
    print_section("Version Compatibility Check")
    
    try:
        import tensorflow as tf
        import torch
        
        tf_version = tf.__version__
        torch_version = torch.__version__
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
        
        print(f"TensorFlow version: {tf_version}")
        print(f"PyTorch version: {torch_version}")
        print(f"PyTorch CUDA version: {cuda_version}")
        
        # Check for known incompatibilities
        if tf_version.startswith('2.17') or tf_version.startswith('2.18'):
            print(f"{Colors.YELLOW}⚠️  TensorFlow {tf_version} may have CUDA compatibility issues{Colors.END}")
            print(f"{Colors.YELLOW}   Recommendation: Downgrade to TensorFlow 2.15.0{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error checking versions: {e}{Colors.END}")

def check_redis():
    """Check Redis connection"""
    print_section("Redis Check")
    
    # Check if Redis is running
    success, _ = run_command("redis-cli ping")
    check_status(success, "Redis server responding to ping")
    
    if success:
        # Check Redis info
        success, info = run_command("redis-cli info server | grep 'redis_version'")
        if success:
            print(f"   {info}")

def check_model_files():
    """Check if model files exist"""
    print_section("Model Files Check")
    
    model_paths = [
        "models/best4.pt",
        "models/best.pt",
    ]
    
    all_ok = True
    for path in model_paths:
        exists = Path(path).exists()
        check_status(exists, f"Model file: {path}")
        if not exists:
            all_ok = False
    
    return all_ok

def check_directories():
    """Check required directories"""
    print_section("Directory Structure Check")
    
    dirs = ['temp', 'logs', 'models', 'app']
    for directory in dirs:
        exists = Path(directory).exists()
        check_status(exists, f"Directory: {directory}")

def generate_fixes():
    """Generate fix commands"""
    print_section("Suggested Fixes")
    
    print(f"{Colors.BOLD}1. Fix CUDA/TensorFlow Compatibility:{Colors.END}")
    print("   Option A - Use CPU mode (fastest fix):")
    print("     export CUDA_VISIBLE_DEVICES=''")
    print("     # Add to your startup script or .bashrc")
    print()
    print("   Option B - Downgrade TensorFlow (if using GPU):")
    print("     pip uninstall tensorflow tensorflow-gpu")
    print("     pip install tensorflow==2.15.0")
    print()
    
    print(f"{Colors.BOLD}2. Install Missing Packages:{Colors.END}")
    print("   pip install -r requirements.txt")
    print()
    
    print(f"{Colors.BOLD}3. Start Redis (if not running):{Colors.END}")
    print("   sudo systemctl start redis")
    print("   # or")
    print("   redis-server --daemonize yes")
    print()
    
    print(f"{Colors.BOLD}4. Test the Fixed System:{Colors.END}")
    print("   # Start the server")
    print("   python main.py")
    print()
    print("   # In another terminal, test health endpoint")
    print("   curl http://localhost:8101/health")
    print()
    
    print(f"{Colors.BOLD}5. Monitor Logs:{Colors.END}")
    print("   # Watch for CUDA errors")
    print("   tail -f nohup.out | grep -i 'cuda\\|gpu\\|error'")

def check_env_variables():
    """Check important environment variables"""
    print_section("Environment Variables Check")
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'LD_LIBRARY_PATH',
        'CUDA_HOME',
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")

def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}KYC Verification System - Diagnostic Tool{Colors.END}\n")
    
    issues_found = []
    
    # Run all checks
    if not check_python_packages():
        issues_found.append("Missing Python packages")
    
    check_gpu_cuda()
    check_compatibility()
    check_redis()
    
    if not check_model_files():
        issues_found.append("Missing model files")
    
    check_directories()
    check_env_variables()
    
    # Summary
    print_section("Diagnostic Summary")
    if issues_found:
        print(f"{Colors.RED}Issues found:{Colors.END}")
        for issue in issues_found:
            print(f"  • {issue}")
    else:
        print(f"{Colors.GREEN}No critical issues detected{Colors.END}")
    
    # Generate fixes
    generate_fixes()
    
    print(f"\n{Colors.BOLD}Diagnostic complete!{Colors.END}\n")

if __name__ == "__main__":
    main()