# Local long-context warmup results

This page records one reproducible, diagnostic-only Qwen2.5 run at 4,096 and
8,192 prompt tokens. It is intentionally separate from the canonical v1
qualification matrix and makes no end-to-end speed or VRAM advantage claim.

## Artifact and environment

| Check | Value |
| --- | --- |
| Raw artifact | `artifacts/qwen25-fixed-warmup-local-v2-clean-14f2fdd.json` |
| Artifact status | `diagnostic` |
| Bound commit | `14f2fdd988e763aa9ac9470d60ce7f481b2404c9` |
| Source tree | clean (`dirty: false`) |
| Model | `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28` |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12,820,480,000 reported bytes |
| Runtime | PyTorch 2.7.0+cu128, Transformers 4.57.3, Triton 3.3.0 |
| Matrix | contexts `[4096, 8192]`; 1 warmup token; 3 measured tokens; 3 repeats |
| Stream | fixed stream seed `26009`; four declared backends |

The raw artifact is intentionally kept in the ignored `artifacts/` directory.
Re-run and validate it from the repository root with:

```bash
python scripts/benchmark_qwen25_fixed_warmup.py \
  --config configs/benchmarks/qwen25-fixed-warmup-local-v2.json \
  --model-path /home/taleef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --output artifacts/qwen25-fixed-warmup-local-v2.json
python scripts/validate_qwen_fixed_warmup.py \
  --config configs/benchmarks/qwen25-fixed-warmup-local-v2.json \
  artifacts/qwen25-fixed-warmup-local-v2.json
```

The clean artifact contains 8/8 completed rows: 2/8 strict quality-gate
passes (the two baseline rows) and 6/8 strict quality-gate failures (all
packed rows). Every packed row keeps top-1 agreement `1.0`, greedy agreement
`true`, and KL below the configured threshold, but every packed row exceeds at
least one full-logit threshold. Thus this run does not qualify compressed
logits as baseline-equivalent under the declared contract.

## Quality matrix

The gates are `max_abs_logit <= 0.01`, `mean_abs_logit <= 0.001`,
`kl_divergence <= 0.0001`, `top1_agreement >= 1.0`, and
`greedy_agreement == true`. Values below are copied from the artifact (small
floating-point values are shown in scientific notation).

| Row | Gate | Max abs logit | Mean abs logit | KL divergence | Top-1 | Greedy |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| 4096 baseline | pass | 0 | 0 | 0 | 1.0 | true |
| 4096 packed_reference | fail | 3.581787109375 | 0.376474291086 | 2.740086893027e-05 | 1.0 | true |
| 4096 serial_direct | fail | 3.560546875 | 0.373152226210 | 2.674931056390e-05 | 1.0 | true |
| 4096 parallel_direct | fail | 3.560302734375 | 0.373278826475 | 2.671105175978e-05 | 1.0 | true |
| 8192 baseline | pass | 0 | 0 | 0 | 1.0 | true |
| 8192 packed_reference | fail | 1.140625 | 0.088638372719 | 0 | 1.0 | true |
| 8192 serial_direct | fail | 0.796875 | 0.074643783271 | 1.199773169702e-08 | 1.0 | true |
| 8192 parallel_direct | fail | 0.84375 | 0.077557057142 | 0 | 1.0 | true |

All six packed lanes therefore fail the strict full-logit gates while passing
the top-1, greedy, and KL checks. The baseline rows are exact-zero references
by construction; they are not independent quality evidence.

## Cache and allocator bytes

All byte values are measured integer storage values from the artifact, not
nominal bit-rate estimates. `logical_payload_bytes` counts used logical KV
payload (including the representation's used metadata/tail accounting).
`physical_cache_bytes` counts the actual cache/store allocation. For an
arena-backed parallel row it includes the arena storage and store workspace;
for non-arena rows it is the store's total measured storage. `unique_resident_bytes`
deduplicates storage and adds the parallel page workspace where applicable.

| Row | Logical payload | Physical cache | Arena capacity | Tail capacity | Workspace | Unique resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 baseline | 235,110,400 | 235,110,400 | 0 | 0 | 0 | 235,110,400 |
| 4096 packed_reference | 118,988,800 | 118,988,800 | 0 | 0 | 0 | 118,988,800 |
| 4096 serial_direct | 118,988,800 | 118,988,800 | 0 | 0 | 0 | 118,988,800 |
| 4096 parallel_direct | 118,988,800 | 148,004,864 | 104,079,360 | 21,962,752 | 247,296 | 148,252,160 |
| 8192 baseline | 469,991,424 | 469,991,424 | 0 | 0 | 0 | 469,991,424 |
| 8192 packed_reference | 230,006,784 | 230,006,784 | 0 | 0 | 0 | 230,006,784 |
| 8192 serial_direct | 230,006,784 | 230,006,784 | 0 | 0 | 0 | 230,006,784 |
| 8192 parallel_direct | 230,006,784 | 259,022,848 | 215,097,344 | 21,962,752 | 480,256 | 259,503,104 |

The runner also records CUDA allocator peaks. Those peaks include model and
runtime allocations and must not be substituted for the cache-store columns:

| Row | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| 4096 baseline | 7,350,514,688 | 7,908,360,192 |
| 4096 packed_reference | 7,221,051,904 | 7,704,936,448 |
| 4096 serial_direct | 7,488,783,360 | 7,981,760,512 |
| 4096 parallel_direct | 7,996,485,632 | 8,401,190,912 |
| 8192 baseline | 9,036,626,944 | 10,252,976,128 |
| 8192 packed_reference | 8,726,489,088 | 9,783,214,080 |
| 8192 serial_direct | 8,649,229,312 | 9,724,493,824 |
| 8192 parallel_direct | 9,449,949,184 | 10,515,120,128 |

## Decode timing

Each timing row contains 9 measured token samples (3 measured tokens × 3
repeats). The warmup decode is outside this sample set. `p50` is the median,
`p95` is the inclusive 95th percentile, and `mean` is the arithmetic mean of
the per-token millisecond samples.

| Row | Decode p50 (ms) | Decode p95 (ms) | Decode mean (ms) |
| --- | ---: | ---: | ---: |
| 4096 baseline | 35.911885 | 50.607248 | 38.186000 |
| 4096 packed_reference | 537.599579 | 644.405826 | 486.532809 |
| 4096 serial_direct | 66.566217 | 101.180962 | 70.860440 |
| 4096 parallel_direct | 41.491307 | 59.779783 | 47.121943 |
| 8192 baseline | 32.683900 | 36.615455 | 33.336203 |
| 8192 packed_reference | 741.899146 | 1,252.948109 | 832.049827 |
| 8192 serial_direct | 104.796978 | 161.947371 | 119.201555 |
| 8192 parallel_direct | 41.764538 | 61.793438 | 48.139988 |

These timings are a small fixed-token diagnostic, not a throughput or
end-to-end latency qualification. The artifact has an empty `claims` list.

## Scope and limitations

- This is one pinned Qwen2.5-7B-Instruct fixture, one fixed stream seed, two
  contexts, three repeats, and three measured decode tokens. It is not a broad
  long-context benchmark or evidence for other model families.
- The six packed rows preserve token choice in this matrix but fail the strict
  full-logit thresholds. No compressed-quality equivalence claim is made.
- Cache-store bytes and CUDA allocator peaks answer different questions; the
  latter include model/runtime allocations. This artifact does not establish
  an end-to-end speed or VRAM advantage.
- The artifact is diagnostic-only, bound to commit `14f2fdd`, and should be
  revalidated after any source, profile, model, or benchmark-config change.
