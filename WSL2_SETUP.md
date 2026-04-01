# WSL2 GPU Setup Guide for TurboQuant

## Why WSL2?

Windows PyTorch doesn't have native Triton support, which we need for the fully-fused compression kernels. WSL2 (Windows Subsystem for Linux 2) provides a native Linux environment with full CUDA support and Triton availability.

**Benefits:**
- ✓ Native Triton compilation support
- ✓ Full CUDA toolkit integration
- ✓ GPU passthrough (RTX 5070 Ti)
- ✓ Faster kernel compilation

---

## Prerequisites

1. **Windows 11 Home/Pro** with virtualization enabled
2. **Docker Desktop** (optional, but recommended) or native WSL2
3. **Administrative access** to install WSL2

---

## Step 1: Install WSL2

### Option A: Using PowerShell (Recommended)

Run PowerShell as Administrator and execute:

```powershell
wsl --install
```

This installs WSL2 with Ubuntu 22.04 by default.

### Option B: Manual Installation

1. Open "Turn Windows Features On or Off"
2. Enable:
   - "Windows Subsystem for Linux"
   - "Virtual Machine Platform"
3. Restart your computer
4. Download Ubuntu 22.04 from Microsoft Store
5. Launch Ubuntu and complete initial setup

---

## Step 2: Set Up Ubuntu Environment

### 1. Launch WSL2 Ubuntu Terminal

```bash
# From PowerShell or CMD
wsl
```

You should now be in the Ubuntu shell.

### 2. Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
```

### 3. Install Python 3.12 and Dependencies

```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev \
  build-essential curl wget git
```

### 4. Verify Python Installation

```bash
python3.12 --version
```

Expected output: `Python 3.12.x`

---

## Step 3: Clone the Repository

### 1. Choose a Linux Location

**Important:** Clone into the WSL2 Linux filesystem (`/home/username/`), NOT the Windows mounted path (`/mnt/c/`).

```bash
cd ~
git clone https://github.com/Taleef7/turboquant.git
cd turboquant
```

### 2. Verify Repository Structure

```bash
ls -la
# Expected files: utils/, kernels/, core/, scripts/, README.md, etc.
```

---

## Step 4: Create Python Virtual Environment

```bash
python3.12 -m venv venv312
source venv312/bin/activate
```

Verify activation:
```bash
which python
# Should show: /home/username/turboquant/venv312/bin/python
```

---

## Step 5: Install PyTorch with CUDA Support

### 1. Install PyTorch Nightly (CUDA 12.8 for RTX 5070 Ti / Blackwell)

**IMPORTANT:** RTX 5070 Ti (Blackwell, sm_120) requires PyTorch nightly with CUDA 12.8:

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

This will take 5-10 minutes (downloads ~2GB).

> **Note:** Standard cu124 builds do NOT support sm_120. You must use cu128 nightly.

### 2. Verify CUDA Detection

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Test CUDA operation
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'CUDA matmul test: SUCCESS')
"
```

Expected output:
```
PyTorch: 2.12.0.dev...+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
VRAM: 12.8 GB
CUDA matmul test: SUCCESS
```

---

## Step 6: Install Triton

### 1. Verify Triton Installation

Triton is installed automatically with PyTorch nightly (cu128). Verify:

```bash
python -c "import triton; print(f'Triton: {triton.__version__}')"
```

Expected: `Triton: 3.6.0` (or similar)

---

## Step 7: Install Project Dependencies

```bash
pip install -r requirements.txt
pip install pytest  # For running tests
```

This installs:
- transformers
- bitsandbytes
- accelerate
- datasets
- einops
- pytest

---

## Step 8: Run Tests

### 1. Math Unit Tests (No GPU Required)

```bash
python -m pytest scripts/test_math.py -v
```

Expected: **10/10 PASSED**

### 2. Kernel Tests (CPU Fallback)

```bash
python -m pytest scripts/test_kernels.py -v
```

Expected: **15/15 PASSED**

### 3. Full Test Suite

```bash
python -m pytest scripts/test_math.py scripts/test_kernels.py -v
```

Expected: **25/25 PASSED**

---

## Step 9: Run GPU Benchmarks

### 1. Baseline Benchmark

```bash
python scripts/run_baseline.py
```

Expected output:
```
=== BASELINE RESULTS ===
Peak VRAM:         ~5.5 GB
KV Cache ~VRAM:    ~1.8 GB
Throughput (TPS):  ~45.2 tok/s
...
```

### 2. TurboQuant Benchmark

```bash
python scripts/run_turboquant.py
```

Expected output:
```
=== TURBOQUANT RESULTS ===
Peak VRAM:            ~4.2 GB
Compressed KV size:   ~400 KB
Throughput (TPS):     ~38.5 tok/s
...
```

### 3. NIAH Evaluation

```bash
# TurboQuant only
python scripts/eval_niah.py

# Compare baseline vs TurboQuant
python scripts/eval_niah.py --both
```

---

## Troubleshooting

### Issue: `WSL2 not found`

**Solution:**
```bash
# Verify WSL2 is installed
wsl --list --verbose
```

If nothing appears, run: `wsl --install` (requires admin)

### Issue: `CUDA not available in WSL2`

**Solution:** Ensure you have:
- Latest NVIDIA driver (591.86+)
- WSL2 CUDA Toolkit (installed automatically)
- GPU passthrough enabled in WSL2 config

Check: `nvidia-smi` inside WSL2

### Issue: `Triton compilation timeout`

**Solution:** Triton compilation can take 10+ minutes on first use. Be patient.

```bash
# Monitor compilation progress
export TRITON_CACHE_DIR=/tmp/triton
python -c "import triton; print('Triton ready!')"
```

### Issue: `pip install fails with permission denied`

**Solution:** Ensure venv is activated:
```bash
source venv312/bin/activate
```

---

## Using Files Between Windows and WSL2

### Access WSL2 Files from Windows

WSL2 filesystem is located at:
```
\\wsl.localhost\Ubuntu\home\username\turboquant
```

Open in Windows Explorer or VS Code Remote.

### Access Windows Files from WSL2

Windows C: drive is mounted at:
```bash
/mnt/c/
cd /mnt/c/Users/talee/OneDrive...  # Access Windows files
```

**Note:** File I/O is slower from `/mnt/c/`. Keep project in `/home/` for best performance.

---

## VS Code Integration

### 1. Install Remote Development Extension

In VS Code:
- Open Extensions
- Search "Remote - WSL"
- Install Microsoft's "Remote - WSL" extension

### 2. Open Project in WSL2

```bash
code .
# Or from Windows:
# wsl code .
```

VS Code will connect to WSL2 automatically and allow you to edit files with full IntelliSense.

---

## Deactivating Venv / Exiting WSL2

```bash
# Exit WSL2 shell
exit

# Or switch to another terminal
# The venv is automatically deactivated when you exit
```

Re-entering:
```bash
wsl
cd ~/turboquant
source venv312/bin/activate
```

---

## Performance Tips

1. **Keep files in Linux filesystem** (`/home/`) not `/mnt/c/`
2. **Use WSL2 terminal** not Windows PowerShell for best performance
3. **GPU compilation is slow first time** — patience required for Triton
4. **Cache compilation** — Triton caches kernels; subsequent runs are fast

---

## Summary Checklist

- [ ] WSL2 installed with Ubuntu 22.04
- [ ] Python 3.12 installed in WSL2
- [ ] Repository cloned to `/home/username/turboquant`
- [ ] Virtual environment created and activated
- [ ] PyTorch installed with CUDA 12.4
- [ ] Triton installed (source compilation complete)
- [ ] All dependencies installed from requirements.txt
- [ ] Tests passing (25/25)
- [ ] Benchmarks running successfully
- [ ] GPU (RTX 5070 Ti) fully functional

---

## Next Steps

Once WSL2 setup is complete:

1. **Run full test suite**: `pytest scripts/ -v`
2. **Profile GPU usage**: Monitor with `nvidia-smi` while running benchmarks
3. **Measure compression ratio**: Compare baseline vs TurboQuant VRAM
4. **Evaluate accuracy**: Run NIAH tests to verify correctness

---

## References

- WSL2 Official Docs: https://learn.microsoft.com/en-us/windows/wsl/
- NVIDIA CUDA in WSL2: https://docs.nvidia.com/cuda/wsl-user-guide/
- Triton Documentation: https://triton-lang.org/
- PyTorch Installation: https://pytorch.org/get-started/locally/
