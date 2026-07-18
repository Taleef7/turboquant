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
nonzero unless all gates pass. The current checked evidence is a failed quality
result, so reproducing that current profile should retain failure evidence—not
produce a success claim.

## 6. Interpret before publishing

- Run `python scripts/validate_artifact.py path/to/artifact.json` for structural
  and binding validation.
- Treat `failed`, `invalid`, and `inconclusive` as non-qualifying states.
- Cite the raw artifact ID with any physical-byte, peak-memory, latency, or
  quality statement.
- Do not promote Qwen3, Qwen3.5, or Gemma 4 from a manifest-only entry without
  the adapter/evidence steps in [PORTABILITY.md](PORTABILITY.md).
