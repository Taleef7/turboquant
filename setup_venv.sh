#!/bin/bash
# TurboQuant CUDA Setup Script
# Creates a Python 3.12 virtual environment with CUDA-enabled PyTorch

set -e  # Exit on error

echo "=========================================="
echo "TurboQuant Virtual Environment Setup"
echo "=========================================="
echo ""

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "ERROR: Python 3.12 not found"
    echo "Install with: choco install python312 -y"
    exit 1
fi

PYTHON312=$(which python3.12)
echo "✓ Found Python 3.12: $PYTHON312"
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv312" ]; then
    echo "Creating virtual environment..."
    $PYTHON312 -m venv venv312
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source venv312/Scripts/activate
echo "✓ Activated: $(which python)"
echo "✓ Python version: $(python --version)"

echo ""
echo "Installing PyTorch with CUDA 12.4 support..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo "✓ PyTorch installed: $(pip show torch | grep Version)"

echo ""
echo "Installing dependencies from requirements.txt..."
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv312/Scripts/activate"
echo ""
echo "To run tests:"
echo "  python -m pytest scripts/test_math.py -v"
echo ""
echo "Note: RTX 5070 Ti (sm_120) requires PyTorch update or source build"
echo "See SETUP_CUDA.md for details"
echo ""
