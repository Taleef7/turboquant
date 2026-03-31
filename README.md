# TurboQuant: KV Cache Compression from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch PyTorch + Triton implementation of **TurboQuant** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) targeting the Qwen2.5-7B-Instruct architecture.

TurboQuant achieves near-optimal KV cache compression with **no fine-tuning required**.

---

## Algorithm

TurboQuant compresses each key/value vector in two stages:

```
Input vector x ∈ ℝ^d
       │
       ▼
┌──────────────────────────────┐
│  Stage 1: MSE Quantization   │
│                               │
│  y = Π · x      (rotate)     │
│  ŷ = Q(y)       (quantize)   │
│                               │
│  Normal channels → 2-bit (4  │
│  levels): ±0.453/√d, ±1.51/√d│
│  Outlier channels → 3-bit    │
└──────────────────────────────┘
       │  r = x - Π^T·ŷ  (residual)
       ▼
┌──────────────────────────────┐
│  Stage 2: QJL Correction     │
│                               │
│  qjl = sign(S · r)  (1-bit)  │
│  γ   = ‖r‖₂        (scalar)  │
└──────────────────────────────┘
       │
       ▼
Stored: {indices_2bit, indices_3bit, qjl_bits, γ}
≈ 2.5 bits/channel average

Dequantize (just-in-time for attention):
x̃ = Π^T·ŷ + (√(π/2)/k) · γ · S^T · qjl
```

The QJL stage ensures the inner product estimator is **unbiased**: E[⟨y, x̃⟩] = ⟨y, x⟩.

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU VRAM  | ≥ 8 GB (RTX 3080/4070/5070 Ti or better) |
| RAM       | ≥ 16 GB system RAM |
| CUDA      | ≥ 11.8 |

> **Note:** Base model weights are loaded in **4-bit (NF4)** via bitsandbytes, consuming ~5 GB VRAM. The remaining ~2.5 GB is used for activations and the compressed KV cache.

---

## Results

| Metric | Baseline (DynamicCache) | TurboQuant |
|--------|------------------------|------------|
| KV Cache VRAM (4K ctx) | ~1.8 GB | ~0.4 GB |
| Compression Ratio | 1x | ~4.5x |
| Tokens per Second | baseline | ≥80% of baseline |
| NIAH @ 10% depth | ✓ | ✓ |
| NIAH @ 50% depth | ✓ | ✓ |
| NIAH @ 90% depth | ✓ | ✓ |

> Benchmarked on RTX 5070 Ti (8 GB), Qwen2.5-7B-Instruct 4-bit.

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/Taleef7/turboquant-qwen-showcase
cd turboquant-qwen-showcase
pip install -r requirements.txt
```

### 2. Run math unit tests (no GPU needed)

```bash
python -m pytest scripts/test_math.py -v
```

### 3. Run baseline benchmark (GPU required)

```bash
python scripts/run_baseline.py
```

### 4. Run TurboQuant benchmark

```bash
python scripts/run_turboquant.py
```

### 5. Run Needle-in-a-Haystack evaluation

```bash
# TurboQuant only
python scripts/eval_niah.py

# Compare baseline vs TurboQuant
python scripts/eval_niah.py --both
```

---

## Project Structure

```
turboquant-qwen-showcase/
├── utils/
│   └── math_utils.py        # Rotation matrix Π, QJL matrix S, centroids
├── kernels/
│   ├── compress_kv.py       # Two-stage compression (Python + Triton)
│   └── decompress_kv.py     # JIT decompression (Python + Triton)
├── core/
│   └── turboquant_cache.py  # HuggingFace DynamicCache subclass
└── scripts/
    ├── test_math.py         # Unit tests for math primitives
    ├── run_baseline.py      # Baseline profiler
    ├── run_turboquant.py    # TurboQuant profiler
    └── eval_niah.py         # NIAH evaluation
```

---

## Citation

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
