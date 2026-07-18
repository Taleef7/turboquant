# TurboQuant Forge Replication Checklist

This checklist verifies implementation contracts and reproduces the evidence
ladder. A passed unit test or completed process is not permission to describe a
profile as qualified; use the status rules in
[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).

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
nonzero unless all gates pass. The retained checked evidence is invalid under
the current context/provenance contract, so a fresh run must not inherit its
diagnostic outcomes as a success claim.

## 6. Interpret before publishing

- Run `python scripts/validate_artifact.py path/to/artifact.json` for structural
  and binding validation.
- Treat `failed`, `invalid`, and `inconclusive` as non-qualifying states.
- Cite the raw artifact ID with any physical-byte, peak-memory, latency, or
  quality statement.
- Do not promote Qwen3, Qwen3.5, or Gemma 4 from a manifest-only entry without
  the adapter/evidence steps in [PORTABILITY.md](PORTABILITY.md).
