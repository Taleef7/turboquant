# TurboQuant CUDA Setup Guide

## Status: WSL2 GPU Setup Complete

**Updated:** 2026-03-31
**Platform:** Ubuntu 24.04 (WSL2)
**Python:** 3.12.3
**PyTorch:** 2.12.0.dev20260330+cu128 (nightly)
**Triton:** 3.6.0
**CUDA:** 13.1 (driver), 12.8 (PyTorch runtime)
**RTX 5070 Ti:** 12.8 GB VRAM, sm_120 (Blackwell)

---

## Current Status: GPU Fully Working

The RTX 5070 Ti (Blackwell architecture, sm_120) is now **fully supported** using PyTorch nightly with CUDA 12.8:

```
PyTorch: 2.12.0.dev20260330+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
Compute Capability: sm_120
VRAM: 12.8 GB
Simple CUDA matmul: SUCCESS
Triton: 3.6.0
```

### Tests Passing

```
✓ 25/25 tests passing (math + kernels)
✓ Triton GPU kernels compiling and running
✓ TurboQuantCache working on both CPU and GPU
```

---

## Quick Start (WSL2)

### 1. Activate the Virtual Environment

```bash
cd ~/projects/turboquant
source venv312/bin/activate
```

### 2. Verify GPU

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### 3. Run Tests

```bash
# Full test suite
python -m pytest scripts/test_math.py scripts/test_kernels.py -v

# GPU benchmark
python scripts/run_baseline.py
```

---

## Installation from Scratch (WSL2)

### Prerequisites

1. **WSL2 with Ubuntu 24.04** installed
2. **NVIDIA drivers 591.86+** installed on Windows
3. **Python 3.12** with venv support:
   ```bash
   sudo apt update
   sudo apt install -y python3.12 python3.12-venv python3.12-dev build-essential
   ```

### Setup Steps

```bash
# 1. Clone repo (if not already done)
cd ~
git clone https://github.com/Taleef7/turboquant.git
cd turboquant

# 2. Create virtual environment
python3.12 -m venv venv312
source venv312/bin/activate

# 3. Install PyTorch nightly with CUDA 12.8 (required for sm_120)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Install project dependencies
pip install -r requirements.txt
pip install pytest

# 5. Verify setup
python -m pytest scripts/test_math.py scripts/test_kernels.py -v
```

---

## Key Discovery: sm_120 Support

**RTX 5070 Ti (Blackwell, sm_120) requires PyTorch nightly with CUDA 12.8**

| PyTorch Version | CUDA | sm_120 Support |
|-----------------|------|----------------|
| 2.6.0 stable | cu124 | ❌ No |
| 2.7.0 nightly | cu124 | ❌ No |
| 2.12.0 nightly | cu126 | ❌ No |
| **2.12.0 nightly** | **cu128** | ✅ **Yes** |

Install the correct version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Virtual Environment Details

Location: `./venv312/`

### Installed Packages

```
✓ torch 2.12.0.dev20260330+cu128
✓ triton 3.6.0
✓ transformers 5.4.0
✓ bitsandbytes 0.49.2
✓ accelerate 1.13.0
✓ datasets 4.8.4
✓ einops 0.8.2
✓ pytest 9.0.2
```

### Activate

**WSL2 / Linux:**
```bash
source venv312/bin/activate
```

---

## Running Tests

### Unit Tests

```bash
# Math primitives (10 tests)
python -m pytest scripts/test_math.py -v

# Kernel tests (15 tests)  
python -m pytest scripts/test_kernels.py -v

# Full suite (25 tests)
python -m pytest scripts/test_math.py scripts/test_kernels.py -v
```

### GPU Benchmarks

```bash
# Baseline (standard DynamicCache)
python scripts/run_baseline.py

# TurboQuant (compressed KV cache)
python scripts/run_turboquant_v2.py

# NIAH paired baseline-vs-TurboQuant
python scripts/test_long_context.py --test niah --mode paired --max-context 8192 --key-bits 8 --value-bits 6 --buffer-size 16384 --trials 2
```

---

## Code Fix Applied

A fix was applied to `core/turboquant_cache.py` to properly handle CPU vs GPU contexts:

```python
# Use Triton only if globally available AND cache is on CUDA
use_triton = _USE_TRITON and self.device.type == 'cuda'
```

This ensures:
- Tests with `device='cpu'` use Python fallback kernels
- Production code with `device='cuda'` uses fast Triton kernels

---

## Troubleshooting

### "CUDA error: no kernel image is available"
- **Cause:** PyTorch doesn't support your GPU's compute capability
- **Solution:** Install PyTorch nightly with CUDA 12.8:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128
  ```

### "Pointer argument cannot be accessed from Triton (cpu tensor?)"
- **Cause:** Triton kernels called with CPU tensors
- **Solution:** Ensure cache is initialized with `device=torch.device('cuda')`

### "nvidia-smi not found" in WSL2
- **Cause:** NVIDIA drivers not installed on Windows host
- **Solution:** Install latest GeForce drivers from nvidia.com

### Triton compilation slow (first run)
- **Expected:** Triton JIT compiles kernels on first use (10-60 seconds)
- **Subsequent runs:** Kernels are cached and load instantly

---

## Summary

| Component | Status |
|-----------|--------|
| WSL2 Ubuntu 24.04 | ✓ Working |
| Python 3.12 | ✓ Ready |
| PyTorch 2.12.0+cu128 | ✓ Installed |
| Triton 3.6.0 | ✓ Working |
| CUDA 13.1 drivers | ✓ Detected |
| RTX 5070 Ti (sm_120) | ✓ **Fully Supported** |
| Unit tests | ✓ 25/25 passing |
| GPU benchmarks | ✓ Running |

---

## References

- PyTorch Nightly: https://pytorch.org/get-started/locally/
- Triton Documentation: https://triton-lang.org/
- WSL2 CUDA Guide: https://docs.nvidia.com/cuda/wsl-user-guide/
- TurboQuant Paper: https://arxiv.org/abs/2504.19874
