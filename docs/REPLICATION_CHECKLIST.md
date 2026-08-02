# TurboQuant Forge Replication Checklist

This checklist verifies implementation contracts and reproduces the evidence
ladder. A passed unit test or completed process is not permission to describe a
profile as qualified; use the status rules in
[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).

Sections 1–6 are the native-WSL/local workflow. For Gilbreth, do not reuse the
local `/home/taleef` model-cache path or run commands on the login node; use
[GILBRETH_HANDOFF.md](GILBRETH_HANDOFF.md).

## 1. Prepare the environment

Use Python 3.11+ and the pinned runtime lock. The GPU path is supported on
native WSL with CUDA and Triton.

```bash
python3.12 -m venv ~/.venvs/turboquant-forge
source ~/.venvs/turboquant-forge/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/runtime.lock
python -m pip install pytest ruff
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

For a GPU run, record the `nvidia-smi` output with the generated artifact.

### Native WSL worktree Git provenance

Qualified evidence rejects an unresolved Git commit; it never promotes
`git_commit: unknown` to a valid result. In this Windows-hosted worktree,
native WSL Git needs the worktree metadata expressed as WSL paths before the
release command. From this checkout root, run:

```bash
export GIT_DIR="/mnt/c/Users/talee/OneDrive - Higher Education Commission/projects/turboquant/.git/worktrees/turboquant-forge"
export GIT_WORK_TREE="$PWD"
git rev-parse --verify HEAD
```

The final command must print a 40-character commit SHA. For a different
checkout, set `GIT_DIR` to that repository's WSL-translated
`.git/worktrees/<worktree-name>` path and keep `GIT_WORK_TREE` at the checkout
root. Leave these variables set for the `reproduce_release.py` command below.

### Gilbreth continuation

For the current branch/artifact state, Slurm commands, CUDA/module checks,
durable artifact transfer, and the ordered 16K/32K follow-up, use the
[Gilbreth handoff](GILBRETH_HANDOFF.md). It is separate from the native-WSL
instructions above. The local Qwen2.5 fixture is pinned to snapshot
`a09a35458c702b33eeacc393d103063234e8bc28` with PyTorch `2.7.0+cu128`,
Transformers `4.57.3`, and Triton `3.3.0`; do not substitute a model or assume
Gilbreth exposes CUDA 12.8 without recording the replacement environment.

## 2. Run download-free CPU verification

```bash
python -m pytest tests -m "not gpu" -q
python -m pytest tests/test_documentation_claims.py -q
git diff --check
```

These tests cover the packed storage/reference/capability/profile and artifact
contracts. They do not establish model-level retrieval quality.

## 3. Run GPU mechanics verification

```bash
python -m pytest tests/test_packed_attention_kernels.py -m gpu -q
python -m pytest tests/test_packed_attention_benchmark.py -m gpu -q
```

The first command is direct-kernel parity against the packed PyTorch reference.
The benchmark command labels its output as a kernel microbenchmark; it is not
end-to-end throughput evidence.

For a manual memory-safety pass when Compute Sanitizer is installed:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest tests/test_packed_attention_kernels.py -m gpu -q
```

## 4. Run the pinned model smoke test

```bash
TURBOQUANT_RUN_MODEL_TESTS=1 python -m pytest tests/test_qwen_adapter_gpu.py -m gpu -q
```

This is opt-in because it downloads the pinned Qwen2.5 weights. It must fail,
not silently skip, if explicitly requested dependencies are unavailable.

## 5. Reproduce release evidence

```bash
python scripts/reproduce_release.py --mode quick --output-dir artifacts/repro_quick
python scripts/reproduce_release.py --mode qualified --output-dir artifacts/repro_qualified
```

Inspect the timestamped `repro_report.json`, `quality`/`systems` JSON files,
persisted 32K decision, and stdout/stderr logs. The qualified command returns
nonzero unless all gates pass. The current named artifact,
`qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0`, is structurally valid
and has qualified 8K quality evidence, but its top-level report is `failed`
because 32K capacity admission stopped the run. A fresh result must not inherit
that partial outcome as a 32K or release claim. The historical v4 artifact is
retired invalid evidence.

## 6. Interpret before publishing

- Run `python scripts/validate_artifact.py path/to/artifact.json` for structural
  and binding validation.
- Treat `failed`, `invalid`, and `inconclusive` as non-qualifying states.
- Cite the raw artifact ID with any physical-byte, peak-memory, latency, or
  quality statement.
- Raw JSON under `artifacts/` is ignored by Git. Copy it and its logs/hashes to
  durable storage before switching machines; a documentation summary alone is
  not enough to validate a claim.
- Do not promote Qwen3, Qwen3.5, or Gemma 4 from a manifest-only entry without
  the adapter/evidence steps in [PORTABILITY.md](PORTABILITY.md).
