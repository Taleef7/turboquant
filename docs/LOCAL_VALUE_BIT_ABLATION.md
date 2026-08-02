# Local value-bit ablation

This diagnostic compares the production direct profile (6-bit keys and values)
with an opt-in 8-bit-value profile. Both candidates use the same pinned
Qwen2.5-7B-Instruct model, fixed-token streams, and serial direct backend.
The parallel fused lane is intentionally excluded so value precision is the
only experimental variable.

## Reproducibility

| Check | Value |
| --- | --- |
| Raw artifact | `artifacts/qwen25-value-ablation-local-v1-clean-eba1c5b.json` |
| Bound commit | `eba1c5b0239971ab2b78e1acceadda0885b6cb28` |
| Source tree | clean detached worktree (`dirty: false`) |
| Model | `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28` |
| Matrix | 4,096 and 8,192 context; 3 measured tokens × 3 repeats |
| Quality gates | max `0.01`, mean `0.001`, KL `0.0001`, top-1 `1.0`, greedy `true` |

Run and validate it with:

```bash
python scripts/benchmark_qwen25_value_ablation.py \
  --config configs/benchmarks/qwen25-value-ablation-local-v1.json \
  --model-path /home/taleef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --output artifacts/qwen25-value-ablation-local-v1.json
```

The artifact fingerprints both profile files, their internal profile hashes,
the model fixture manifest, the runner, and the clean source commit.

## Results

| Context / lane | Max abs logit | Mean abs logit | KL divergence | Top-1 | Greedy | Gate |
| --- | ---: | ---: | ---: | ---: | :---: | :---: |
| 4K baseline | 0 | 0 | 0 | 1.0 | true | pass |
| 4K v6 serial | 3.560547 | 0.373152 | 2.6749e-05 | 1.0 | true | fail |
| 4K v8 serial | 3.524902 | 0.378567 | 2.9547e-05 | 1.0 | true | fail |
| 8K baseline | 0 | 0 | 0 | 1.0 | true | pass |
| 8K v6 serial | 0.796875 | 0.074644 | 1.1998e-08 | 1.0 | true | fail |
| 8K v8 serial | 2.109375 | 0.104040 | 5.7609e-08 | 1.0 | true | fail |

| Context | v6 logical bytes | v8 logical bytes | v8 vs v6 storage |
| ---: | ---: | ---: | ---: |
| 4K | 118,988,800 | 132,751,360 | 11.6% larger |
| 8K | 230,006,784 | 258,449,408 | 12.4% larger |

| Context | v6 decode p50 (ms) | v8 decode p50 (ms) | v8 vs v6 latency |
| ---: | ---: | ---: | ---: |
| 4K | 59.398157 | 94.762252 | 59.5% slower |
| 8K | 95.636049 | 142.537450 | 49.0% slower |

## Conclusion

The 8-bit-value experiment is a useful negative result: it preserves exact
top-1 and greedy choices in this tiny matrix but does not satisfy the strict
full-logit gates, increases cache storage by roughly 12%, and is substantially
slower. It should remain opt-in and must not replace the production v6 profile
without a larger quality study and a measured systems benefit.
