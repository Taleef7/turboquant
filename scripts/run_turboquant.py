"""
TurboQuant KV Cache Benchmark — Compressed KV with TurboQuantCache.

Loads Qwen2.5-7B-Instruct in 4-bit, injects TurboQuantCache, generates
4,000 tokens, and reports the same metrics as run_baseline.py for comparison.

Usage:
  python scripts/run_turboquant.py

Hardware requirement: NVIDIA GPU with ≥8GB VRAM
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.turboquant_cache import TurboQuantCache, print_profile_stats

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TARGET_NEW_TOKENS = 200
PROMPT_TOKENS = 3800
HEAD_DIM = 128  # Qwen2.5-7B head dimension
N_QJL = 128  # QJL projections


def warmup_triton_kernels(
    head_dim: int = 128,
    device: torch.device = None,
    prefill_seq_len: int = 3800,
    n_heads: int = 28,
    batch_size: int = 1,
):
    """
    Trigger Triton JIT compilation for all kernel variants BEFORE timing.

    This compiles kernels for:
    - Prefill: large seq_len (uses different grid than decode)
    - Decode: seq_len=1 (most common during generation)
    - Both quantize and dequantize kernels

    Triton caches compiled kernels by (kernel_function, constexpr_args).
    The constexpr params are BLOCK_SEQ=32 and BLOCK_D=32.
    Grid dimensions vary with input size: (cdiv(seq_len, 32), cdiv(head_dim, 32)).

    IMPORTANT: Triton compiles a unique binary for each grid dimension,
    so we need to exercise the exact grid sizes used during inference.
    """
    from kernels.quantize_triton import triton_quantize, triton_dequantize

    if device is None:
        device = torch.device("cuda")

    print("Warming up Triton kernels (JIT compilation)...")

    # Create dummy inputs matching TurboQuantCache internals
    centroids_2bit = torch.tensor(
        [-1.5, -0.5, 0.5, 1.5], dtype=torch.float32, device=device
    )
    centroids_3bit = torch.tensor(
        [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=torch.float32, device=device
    )
    is_outlier = torch.zeros(head_dim, dtype=torch.bool, device=device)
    is_outlier[:32] = True  # 32 outlier channels

    # Calculate the actual sizes used during inference
    # During prefill: (batch * heads * seq_len, head_dim) = (batch * n_heads * prefill_seq_len, head_dim)
    # During decode: (batch * heads * 1, head_dim) = (batch * n_heads, head_dim)

    n_slices = batch_size * n_heads  # Qwen2.5-7B has 28 attention heads

    # Key sequence lengths to warm up (covering all grid dimensions used during inference)
    warmup_seq_lens = [
        n_slices * 1,  # Decode: single token per slice (28 total rows)
        n_slices * 5,  # Model warmup with 5 tokens
        n_slices * prefill_seq_len,  # Prefill: 3800 tokens * 28 heads = 106400 rows
    ]

    # Also add smaller sizes for incremental variations
    warmup_seq_lens.extend([1, 8, 32, 128, 256, 512, 1024, 2048, 4096])

    # Remove duplicates and sort
    warmup_seq_lens = sorted(set(warmup_seq_lens))

    print(
        f"  Warming up for seq_lens: {warmup_seq_lens[:5]}... (and {len(warmup_seq_lens)} total)"
    )

    for seq_len in warmup_seq_lens:
        y = torch.randn(seq_len, head_dim, dtype=torch.float32, device=device)

        # Warmup quantize kernel
        idx, y_hat = triton_quantize(y, centroids_2bit, centroids_3bit, is_outlier)

        # Warmup dequantize kernel
        _ = triton_dequantize(idx, centroids_2bit, centroids_3bit, is_outlier)

    # Sync to ensure all compilation is done
    torch.cuda.synchronize()
    print("Triton kernel warmup complete.")


def get_dummy_prompt(n_tokens: int, tokenizer) -> str:
    filler = "The quick brown fox jumps over the lazy dog. " * 500
    tokens = tokenizer.encode(filler)[:n_tokens]
    return tokenizer.decode(tokens)


def profile_memory():
    class MemProfiler:
        def __enter__(self):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            self.mem_before = torch.cuda.memory_allocated() / 1024**2
            return self

        def __exit__(self, *_):
            self.peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            self.current_mb = torch.cuda.memory_allocated() / 1024**2

    return MemProfiler()


def run_turboquant():
    print(f"Loading {MODEL_ID} in 4-bit with TurboQuant KV cache...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    torch.cuda.empty_cache()
    mem_after_load = torch.cuda.memory_allocated() / 1024**2
    print(f"Model loaded. VRAM used: {mem_after_load:.1f} MB")

    prompt = get_dummy_prompt(PROMPT_TOKENS, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    print(f"Prompt tokens: {input_len}")

    # ====================================================================
    # WARMUP PHASE: Trigger all Triton JIT compilation BEFORE timing
    # ====================================================================

    # 1. Direct kernel warmup (exercises all seq_len grid variants)
    warmup_triton_kernels(head_dim=HEAD_DIM, device=torch.device("cuda"))

    # 2. Model-level warmup: Run full prefill + a few decode steps
    #    This ensures HuggingFace's internal caching is also warmed up
    print("Warming up model with TurboQuant cache (prefill + decode)...")
    tq_cache_warmup = TurboQuantCache(
        head_dim=HEAD_DIM, n_qjl=N_QJL, device=torch.device("cuda"), dtype=torch.float16
    )
    with torch.no_grad():
        # Generate a few tokens to warm up both prefill and decode paths
        _ = model.generate(
            **inputs, max_new_tokens=5, do_sample=False, past_key_values=tq_cache_warmup
        )

    # Clean up warmup cache and free memory
    del tq_cache_warmup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Warmup complete. Starting timed benchmark...\n")

    # ====================================================================
    # TIMED RUN: Now measure actual performance
    # ====================================================================

    # Initialize fresh TurboQuantCache for the benchmark
    tq_cache = TurboQuantCache(
        head_dim=HEAD_DIM,
        n_qjl=N_QJL,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    # Timed run
    with profile_memory() as prof:
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=TARGET_NEW_TOKENS,
                do_sample=False,
                past_key_values=tq_cache,
                return_dict_in_generate=True,
            )
        t_total = time.perf_counter() - t0

    n_generated = outputs.sequences.shape[1] - input_len
    tps = n_generated / t_total
    ttft_ms = (t_total / n_generated) * 1000
    compressed_kb = tq_cache.compressed_size_bytes() / 1024

    print("\n=== TURBOQUANT RESULTS ===")
    print(f"Peak VRAM:            {prof.peak_mb:.1f} MB")
    print(f"VRAM delta (KV):      {(prof.peak_mb - mem_after_load):.1f} MB")
    print(f"Compressed KV size:   {compressed_kb:.1f} KB")
    print(f"Tokens generated:     {n_generated}")
    print(f"Total time:           {t_total:.2f}s")
    print(f"Approx TTFT:          {ttft_ms:.1f} ms/token")
    print(f"Throughput (TPS):     {tps:.2f} tok/s")
    print("==========================")

    # Print profiling stats if enabled
    print_profile_stats()

    return {"peak_vram_mb": prof.peak_mb, "tps": tps, "compressed_kb": compressed_kb}


if __name__ == "__main__":
    run_turboquant()
