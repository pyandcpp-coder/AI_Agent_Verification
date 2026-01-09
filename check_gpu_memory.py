#!/usr/bin/env python3
"""
GPU Memory Status Checker
Displays current GPU memory usage and provides recommendations
"""
import torch

def check_gpu_memory():
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return
    
    print("=" * 60)
    print("GPU Memory Status")
    print("=" * 60)
    
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    
    # Get memory stats
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    free = total_memory - (reserved / total_memory * total_memory)
    
    print(f"Device: {device_name}")
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Allocated: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
    print(f"Reserved: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
    print(f"Free: {free:.2f} GB ({free/total_memory*100:.1f}%)")
    print()
    
    # Recommendations
    if allocated / total_memory > 0.85:
        print("⚠️  WARNING: GPU memory usage is very high (>85%)")
        print("   Recommendations:")
        print("   - Clear GPU cache: torch.cuda.empty_cache()")
        print("   - Reduce batch sizes")
        print("   - Use model quantization or mixed precision")
    elif allocated / total_memory > 0.70:
        print("⚠️  CAUTION: GPU memory usage is moderate-high (>70%)")
        print("   Consider clearing cache periodically")
    else:
        print("✅ GPU memory usage is healthy")
    
    print("=" * 60)

if __name__ == "__main__":
    check_gpu_memory()
