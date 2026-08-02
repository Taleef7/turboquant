# Codec reproduction report

This report describes the first public, local codec experiment in TurboQuant
Forge. It is a clean-room implementation of the mathematical families needed
for a bounded TurboQuant-style comparison; it is not a claim that Forge is the
paper implementation or that the paper's exact experimental conditions were
recovered.

## Canonical artifact

The report is bound to artifact `codec-local-v1`, generated from the clean
commit `871cbc9058d4befe4b0f5e35b687d1b5f525cb2c`.

| Check | Result |
| --- | --- |
| Artifact status | `qualified` |
| Persisted rows | `1,408/1,408` passed |
| Across-seed summaries | `588/588` complete, each with the exact 16-seed roster |
| Paired Product-vs-MSE comparisons | `108/108` complete |
| Resolved claim references | `3,858` in `claims.json` |
| Source tree | clean (`dirty: false`) |

The manifest and compact summary are written to the ignored local directory
`artifacts/codec-local-v1/`. The artifact is intentionally generated rather
than checked into Git: it contains many row records and a dataset-derived
sidecar. The manifest, summary, and ledger are the authoritative evidence
objects when the experiment is rerun.

Validate the copied artifact with:

```bash
python scripts/validate_artifact.py artifacts/codec-local-v1/manifest.json
```

The command must print `{"valid": true, "errors": []}`. To rebuild the
reference-checked claim ledger:

```bash
python -c "import json; from pathlib import Path; from scripts.claim_ledger import build_claim_ledger; p=Path('artifacts/codec-local-v1/manifest.json'); out=build_claim_ledger([p]); Path('artifacts/codec-local-v1/claims.json').write_text(json.dumps(out, sort_keys=True, indent=2)+'\\n', encoding='utf-8'); print(len(out['claims']))"
```

## What was measured

The synthetic matrix contains four distributions (`sphere`, `gaussian`,
`heavy_tail`, and `outlier`), three dimensions (64, 128, and 256), 16 paired
rotation/projection seeds, and 10,000 vectors per fixture. MSE uses nominal
widths 1, 2, 3, and 4; Product uses widths 2, 3, and 4. This produces 1,344
synthetic rows. Each row records distortion, independent-query error,
high-inner-product error and signed bias, plus byte accounting and timing
metadata.

The GloVe portion uses the pinned 962,819,488-byte
`glove-200-angular.hdf5` (SHA-256
`4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20`), a
deterministic 100,000-vector base subset, and 10,000 queries. It adds 64 rows:
MSE and Product at nominal widths 2 and 4, each repeated over the 16 paired
seeds. The runner recomputes exact top-1 neighbors over the selected base and
uses bounded chunked scoring; it never materializes a 10,000-by-100,000 score
matrix.

## Retrieval results

The following are 16-seed means from the manifest's `across_seed_statistics`
sidecar. They are recall of the exact selected-base neighbor at cutoffs 1, 10,
and 100, not recall against an unknown paper sample. Every value is directly
queryable in `artifacts/codec-local-v1/claim-summary.json`; the corresponding
claim references are in `claims.json`.

| Codec | Nominal width | Recall@1 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| MSE | 2 | 0.507875 | 0.9087875 | 0.99686875 |
| Product | 2 | 0.2900125 | 0.64278125 | 0.8959375 |
| MSE | 4 | 0.82224375 | 0.99935625 | 1.00000000 |
| Product | 4 | 0.6634375 | 0.98195000 | 0.99999375 |

For example, the MSE width-2 Recall@1 row is addressed by:

```text
sidecar:artifacts/codec-local-v1/claim-summary.json#across_seed_statistics.across-glove200-None-200-mse-2-retrieval-1-1.statistics.mean
```

The paired bootstrap comparisons cover the three shared Product/MSE widths,
four distributions, three dimensions, and three metrics: 108 complete groups.
They use 10,000 deterministic paired resamples (`seed=1729`) and are exposed
under `product_minus_mse_comparisons` in the same summary.

## Runtime conditions

The canonical run was executed in WSL2 using the pinned Python 3.12.3 runtime,
NumPy 2.5.1, SciPy 1.18.0, h5py 3.16.0, and PyTorch 2.7.0+cu128. The GPU was
an NVIDIA GeForce RTX 5070 Ti Laptop GPU with 12,820,480,000 reported bytes of
VRAM, driver 610.47, and CUDA 12.8. Codec encode/decode timing is explicitly
classified `local_reference_cpu_timing`; only the exact chunked retrieval
scoring ran on CUDA. The manifest records the CPU, thread settings, allocator
environment, runtime lock hash, GPU temperature/power observations, and the
retrieval policy so another run can explain environmental differences.

## Scope and limitations

- This is a v1 partial reproduction. The original paper's sample IDs, exact
  codec variant, and accepted-revision protocol are not all available here.
- Nominal width is an experiment label; it is not a claim of physical bit
  parity with a paper implementation.
- The GloVe result is a deterministic local subset experiment, not a claim
  about the full paper-scale benchmark.
- These codec measurements do not establish an end-to-end model serving
  advantage. The separate Qwen2.5 systems artifact remains the source of the
  current negative 8K systems result, and 32K is not qualified.
- RaBitQ/PQ baselines, QJL residuals, NIAH/RULER/LongBench-E, and a Gilbreth
  hardware run are separate follow-on work. The current cluster execution
  sequence and transfer rules are in [GILBRETH_HANDOFF.md](GILBRETH_HANDOFF.md).

## Reproduce from a clean environment

Use the locked WSL runtime and the pinned dataset. `--no-download` makes a
missing or mismatched file fail rather than silently changing the experiment.

```bash
python scripts/run_codec_reproduction.py \
  --config configs/benchmarks/codec-local-v1.json \
  --output-dir artifacts/codec-local-v1 \
  --dataset-cache ~/.cache/turboquant \
  --retrieval-device cuda \
  --retrieval-base-chunk-size 2048 \
  --no-download
python scripts/validate_artifact.py artifacts/codec-local-v1/manifest.json
```

The run is expected to produce the counts above on the same inputs. A missing
GloVe file is represented as `unavailable` rows and cannot be promoted to a
qualified result.
