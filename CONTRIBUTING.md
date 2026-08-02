# Contributing

Thanks for contributing to TurboQuant replication.

This repository is currently optimized for a **Qwen-focused reproducibility path**. Please keep contributions aligned with reproducibility, clarity, and auditable claims.

## Canonical Workflow

## 1) Environment

```bash
python3.12 -m venv ~/.venvs/turboquant-forge
source ~/.venvs/turboquant-forge/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/runtime.lock
python -m pip install pytest ruff
```

The lock pins the supported local WSL runtime, including stable PyTorch
`2.7.0+cu128`. Do not add an unpinned nightly wheel to a reproducibility run;
on Gilbreth, follow [the cluster handoff](docs/GILBRETH_HANDOFF.md) and record
any compatible replacement environment.

## 2) Core verification commands

```bash
# Fast check
python scripts/reproduce_release.py --mode quick --output-dir artifacts/repro_quick

# Release-check path (Qwen scope)
python scripts/reproduce_release.py --mode qualified --output-dir artifacts/repro_qualified

# Unit tests
python -m pytest tests -m "not gpu" -q
python -m pytest tests/test_documentation_claims.py -q
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
