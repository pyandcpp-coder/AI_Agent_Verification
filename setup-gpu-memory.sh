#!/bin/bash
# GPU Memory Optimization Setup Script

echo "=================================================="
echo "GPU Memory Optimization Setup"
echo "=================================================="
echo ""

# Set environment variable for PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "✓ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

# Check if we should add it to bashrc
if ! grep -q "PYTORCH_CUDA_ALLOC_CONF" ~/.bashrc; then
    echo "Adding to ~/.bashrc for persistence..."
    echo "" >> ~/.bashrc
    echo "# PyTorch GPU Memory Management" >> ~/.bashrc
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" >> ~/.bashrc
    echo "✓ Added to ~/.bashrc"
else
    echo "✓ Already in ~/.bashrc"
fi

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Applied fixes:"
echo "  1. ✓ PyTorch memory fragmentation reduction"
echo "  2. ✓ GPU cache clearing after model operations"
echo "  3. ✓ Low memory mode for model loading"
echo "  4. ✓ Memory synchronization after inference"
echo ""
echo "To apply the environment variable to current session:"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""
echo "Or restart your terminal/reload bashrc:"
echo "  source ~/.bashrc"
echo ""
