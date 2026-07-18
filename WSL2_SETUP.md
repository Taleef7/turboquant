# WSL2 setup for TurboQuant

TurboQuant's Triton kernels require Linux. Work from the WSL filesystem (for
example `~/projects/turboquant`), rather than a `/mnt/c` checkout, for reliable
virtual-environment and file-system performance.

## Install WSL and prerequisites

From an elevated Windows PowerShell:

```powershell
wsl --install
```

Then, in Ubuntu:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv git build-essential
nvidia-smi
```

`nvidia-smi` must show the Windows-host NVIDIA driver before installing Python
packages.

## Create the native runtime

```bash
git clone https://github.com/Taleef7/turboquant.git ~/projects/turboquant
cd ~/projects/turboquant
python3.12 -m venv ~/.venvs/turboquant-forge
source ~/.venvs/turboquant-forge/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/runtime.lock
```

The lock selects official stable PyTorch CUDA 12.8 wheels; it does not use
nightly packages. Verify the installed runtime and run the CPU test lane:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import accelerate, bitsandbytes, einops, numpy, safetensors, scipy, transformers, triton; print('imports: OK')"
python -m pip install pytest ruff
python -m pytest tests -m "not gpu" -q
python -m compileall core scripts tests
```

Use the benchmark and reproduction scripts only after this smoke check passes.
Generated measurements belong under `artifacts/`; fixtures under
`artifacts/fixtures/` may be committed.
