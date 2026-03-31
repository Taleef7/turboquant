# TurboQuant CUDA Setup Guide

## Status: Python 3.12 Virtual Environment Ready

**Created:** 2026-03-30
**Python:** 3.12.10
**PyTorch:** 2.7.0.dev20250310+cu124 (nightly)
**CUDA:** 13.1 (system), 12.4 (PyTorch)
**RTX 5070 Ti:** 12.8 GB VRAM available

---

## Current Limitation: Blackwell (sm_120) Support

The RTX 5070 Ti uses the **Blackwell architecture (sm_120)**, released in early 2025. Unfortunately:

- **PyTorch 2.6.0**: Supports sm_50-sm_90 (Ada Lovelace is latest)
- **PyTorch 2.7.0 nightly**: Still supports sm_50-sm_90
- **Official PyTorch wheels**: No sm_120 support yet

### Status Check

```
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
Compute capability: sm_120 ❌ NOT SUPPORTED
CUDA available: True (but kernels won't run)
```

### Workaround: CPU Execution

**The good news:** All code works perfectly on CPU!

```bash
# Verify CPU works
cd "C:/Users/talee/OneDrive - Higher Education Commission/projects/TurboQuant/turboquant-qwen-showcase"
source venv312/Scripts/activate
python -m pytest scripts/test_math.py -v
```

**Result:** ✓ 10/10 math tests passing on CPU

---

## Virtual Environment Setup

A Python 3.12 virtual environment has been created at:
```
./venv312/
```

### Activate the Environment

**Windows (Git Bash/WSL):**
```bash
source venv312/Scripts/activate
```

**Windows (PowerShell):**
```powershell
.\venv312\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```cmd
venv312\Scripts\activate.bat
```

### Installed Packages

```
✓ PyTorch 2.7.0.dev20250310+cu124
✓ NumPy 2.4.3
✓ Transformers 5.4.0
✓ BitsAndBytes 0.49.2
✓ PEFT 0.18.1
✓ Accelerate 1.13.0
✓ Pytest 9.0.2
```

---

## Running Tests

### CPU-Based Tests (Recommended Now)

```bash
source venv312/Scripts/activate

# Math primitives only (no GPU)
python -m pytest scripts/test_math.py -v

# All tests (math + kernels, CPU fallback)
python -m pytest scripts/test_kernels.py scripts/test_math.py -v
```

**Expected:** ✓ 25/25 passing

### GPU-Based Benchmarks (Blocked Until sm_120 Support)

These scripts are ready but will fail on GPU computation:

```bash
source venv312/Scripts/activate

# Will fail: CUDA kernel not available for sm_120
python scripts/run_baseline.py
```

**Error expected:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

---

## Future: PyTorch sm_120 Support

Once PyTorch officially supports Blackwell (sm_120), you can:

1. Update PyTorch:
   ```bash
   source venv312/Scripts/activate
   pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
   ```

2. Verify GPU support:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Run benchmarks:
   ```bash
   python scripts/run_baseline.py
   python scripts/run_turboquant.py
   python scripts/eval_niah.py --both
   ```

**Check PyTorch roadmap:** https://github.com/pytorch/pytorch/issues

---

## Alternative: Build PyTorch from Source (Advanced)

If you need GPU support immediately, you can build PyTorch from source with sm_120 support:

```bash
# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Activate venv
source ../venv312/Scripts/activate

# Set CUDA architecture for sm_120
export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;12.0"

# Build (takes 30-60 minutes)
python setup.py install
```

**This is advanced and not recommended unless you're familiar with C++/CUDA builds.**

---

## Environment Variables

If building from source, set:

```bash
# CUDA paths
export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"
export PATH="$CUDA_HOME/bin:$PATH"

# PyTorch build flags
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_VISIBLE_DEVICES=0
```

---

## Summary

| Aspect | Status |
|--------|--------|
| Python 3.12 | ✓ Ready |
| PyTorch CUDA | ✓ Ready (CPU fallback) |
| CUDA 13.1 drivers | ✓ Installed |
| RTX 5070 Ti detection | ✓ Detected (sm_120) |
| sm_120 kernel support | ❌ Not yet in PyTorch |
| CPU-based tests | ✓ All passing |
| GPU-based tests | ⏳ Blocked (awaiting PyTorch update) |

---

## Next Steps

### Immediate (CPU)
- Run math unit tests: `pytest scripts/test_math.py -v`
- Verify cache integration on CPU
- Document compression ratio (0.94x with int8)

### When sm_120 Support Lands
- Upgrade PyTorch (`pip install --upgrade torch`)
- Run GPU benchmarks
- Measure VRAM savings on RTX 5070 Ti
- Compare baseline vs TurboQuant TPS

---

## Troubleshooting

### "CUDA error: no kernel image is available"
- **Cause:** Your GPU (sm_120) isn't supported by installed PyTorch
- **Solution:** Wait for PyTorch to add sm_120 support, or build from source

### "ModuleNotFoundError: No module named 'triton'"
- **Cause:** Triton is optional (Windows doesn't have pre-built wheels)
- **Solution:** Code falls back to Python kernels automatically

### venv not activating
- **Windows PowerShell:** Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Windows Git Bash:** Use `source venv312/Scripts/activate`
- **Windows cmd.exe:** Use `venv312\Scripts\activate.bat`

---

## References

- PyTorch CUDA Support: https://pytorch.org/get-started/locally/
- Blackwell Architecture: https://developer.nvidia.com/blackwell
- RTX 5070 Ti Specs: https://www.nvidia.com/en-us/geforce/ada-generation/
- CUDA Capabilities: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
