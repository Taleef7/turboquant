# Contributing

Thanks for contributing to TurboQuant replication.

This repository is currently optimized for a **Qwen-focused reproducibility path**. Please keep contributions aligned with reproducibility, clarity, and auditable claims.

## Canonical Workflow

## 1) Environment

```bash
python -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt
```

For RTX 50-series (`sm_120`), use PyTorch nightly `cu128` before installing requirements.

## 2) Core verification commands

```bash
# Fast check
python scripts/reproduce_release.py --mode quick --output-dir results/repro_quick

# Release-check path (Qwen scope)
python scripts/reproduce_release.py --mode full --output-dir results/repro_full

# Unit tests
pytest scripts/test_math.py scripts/test_kernels.py -q
pytest scripts/test_long_context_harness.py scripts/test_cache_config.py scripts/test_qjl_config.py -q
pytest scripts/test_repro_utils.py scripts/test_reproduce_release.py -q
```

## 3) Claims discipline

Before changing benchmark claims, update all of:
- `README.md`
- `TESTING_RESULTS.md`
- `docs/PAPER_CLAIMS_STATUS.md`
- `docs/REPLICATION_CHECKLIST.md`

Never claim parity for benchmarks not implemented in this repo yet.

## Script scope policy

Keep `scripts/` focused on actively used, reproducible paths. Avoid adding one-off debug/profiling scripts unless they are promoted into maintained tooling.

## Pull request checklist

- [ ] Tests pass locally
- [ ] Repro command path still works
- [ ] Docs and claims are synchronized
- [ ] No stale references to removed scripts/files
