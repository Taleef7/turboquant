# Evidence and claim ledger

Every public numeric statement about the codec experiment should be traceable
to the clean `codec-local-v1` artifact and, where applicable, to the generated
`claims.json` ledger. The ledger resolves the manifest's exact queries and
verifies the SHA-256 of every referenced sidecar before returning a value.

Canonical source identity:

```text
artifact_id: codec-local-v1
git_commit: 871cbc9058d4befe4b0f5e35b687d1b5f525cb2c
authoritative manifest: artifacts/codec-local-v1/manifest.json
summary sidecar: artifacts/codec-local-v1/claim-summary.json
resolved ledger: artifacts/codec-local-v1/claims.json
```

The source tree was clean when the artifact was generated. The ledger contains
3,858 resolved claim records. A claim is not considered release evidence if it
comes from a dirty source, a failed/invalid/OOM row, an incomplete 16-seed
block, or a sidecar whose hash no longer matches the manifest.

## Safe statements and exact evidence

| Statement | Exact evidence query or check | Interpretation |
| --- | --- | --- |
| The canonical artifact is qualified with 1,408 passed rows. | `sidecar:artifacts/codec-local-v1/claim-summary.json#pass_state_counts.passed` | The runner's complete v1 matrix passed its schema and row gates. |
| No row failed, was invalid, OOM, unavailable, or unsupported. | `sidecar:artifacts/codec-local-v1/claim-summary.json#pass_state_counts.failed` (and the sibling status keys) | This describes this input/environment, not a universal guarantee. |
| All 588 across-seed groups are complete. | `manifest.json` → `extensions.across_seed_statistics[*].status` | Each summary uses exactly the declared 16 paired seeds. |
| All 108 Product/MSE comparison groups are complete. | `manifest.json` → `extensions.product_minus_mse_comparisons[*].status` | Only shared nominal widths 2, 3, and 4 are compared. |
| GloVe MSE width-2 Recall@1 mean is 0.507875. | `sidecar:artifacts/codec-local-v1/claim-summary.json#across_seed_statistics.across-glove200-None-200-mse-2-retrieval-1-1.statistics.mean` | Mean over 16 paired seeds on the pinned 100k-base/10k-query subset. |
| GloVe Product width-4 Recall@10 mean is 0.98195. | `sidecar:artifacts/codec-local-v1/claim-summary.json#across_seed_statistics.across-glove200-None-200-product-4-retrieval-1-10.statistics.mean` | Same subset and exact selected-base neighbor protocol. |
| A Product/MSE difference is paired and uncertainty-estimated. | `sidecar:artifacts/codec-local-v1/claim-summary.json#product_minus_mse_comparisons.comparison-all-gaussian-128-all-2-queries-independent-rmse` | The object records the mean, 95% bootstrap interval, seed, and 10,000 iterations. |
| Retrieval scoring used bounded exact chunking. | `manifest.json` → `extensions.retrieval_policy` | The policy records CUDA scoring, a 10,000×2,048 maximum block, and no full score matrix. |

The query strings above are intentionally literal. A report may summarize a
value, but it must preserve the artifact ID and query path so a reader can
resolve the same value independently.

## Statements deliberately not made

- This ledger does not claim the paper's exact implementation, sample IDs, or
  accepted-revision protocol.
- It does not turn nominal widths into physical bit parity or claim paper-scale
  behavior from a local subset.
- It does not claim a Qwen2.5 serving improvement, 32K qualification, or
  support for Qwen3, Qwen3.5, or Gemma 4. Those require separate adapters and
  evidence artifacts.
- It does not use the codec artifact to overwrite the separate negative 8K
  systems result; each experiment keeps its own scope and conditions.

## Rebuilding and checking the ledger

The runner emits `claim-summary.json` as a compact deterministic bridge. Build
the reference-checked ledger with:

```bash
python -c "import json; from pathlib import Path; from scripts.claim_ledger import build_claim_ledger; p=Path('artifacts/codec-local-v1/manifest.json'); out=build_claim_ledger([p]); Path('artifacts/codec-local-v1/claims.json').write_text(json.dumps(out, sort_keys=True, indent=2)+'\\n', encoding='utf-8'); print(len(out['claims']))"
```

Then validate the manifest:

```bash
python scripts/validate_artifact.py artifacts/codec-local-v1/manifest.json
```

Both checks must succeed before copying numbers into a résumé, report, or
benchmark comparison.
