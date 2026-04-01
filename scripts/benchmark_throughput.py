#!/usr/bin/env python3
"""
Throughput benchmark: Compare baseline vs TurboQuant with 6-bit keys.
Measures tokens/second for generation with various prompt lengths.
"""

import torch
import time
import sys

sys.path.insert(0, "/home/taleef/projects/turboquant")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.turboquant_cache_v2 import TurboQuantCacheV2


def benchmark_generation(
    model, tokenizer, input_ids, cache, num_tokens=200, warmup_runs=2
):
    """Run generation and measure throughput."""
    device = input_ids.device

    # Warmup runs (JIT compilation, cache warmup)
    for _ in range(warmup_runs):
        if cache is not None:
            test_cache = TurboQuantCacheV2(
                head_dim=128,
                bits=cache.bits,
                device=device,
                buffer_size=cache.buffer_size,
            )
        else:
            test_cache = None

        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                past_key_values=test_cache,
                use_cache=True,
            )

    # Actual benchmark
    if cache is not None:
        bench_cache = TurboQuantCacheV2(
            head_dim=128,
            bits=cache.bits,
            device=device,
            buffer_size=cache.buffer_size,
        )
    else:
        bench_cache = None

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=num_tokens,
            do_sample=False,
            past_key_values=bench_cache,
            use_cache=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_tokens = outputs.shape[1] - input_ids.shape[1]
    throughput = generated_tokens / elapsed

    return throughput, generated_tokens, elapsed


def main():
    print("Loading Qwen/Qwen2.5-7B-Instruct...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    device = torch.device("cuda")

    # Test configurations
    prompt_lengths = [100, 500, 1000]
    num_generate = 200

    print(f"\n{'=' * 70}")
    print(f"Throughput Benchmark: Generating {num_generate} tokens")
    print(f"{'=' * 70}\n")

    for prompt_len in prompt_lengths:
        # Create prompt of target length
        base_text = "The quick brown fox jumps over the lazy dog. " * 100
        tokens = tokenizer.encode(base_text)[:prompt_len]
        input_ids = torch.tensor([tokens], device=device)
        actual_prompt_len = input_ids.shape[1]

        print(f"Prompt length: {actual_prompt_len} tokens")
        print("-" * 50)

        # Baseline (no compression)
        baseline_throughput, baseline_tokens, baseline_time = benchmark_generation(
            model, tokenizer, input_ids, None, num_generate
        )
        print(
            f"  Baseline:   {baseline_throughput:6.2f} tok/s ({baseline_tokens} tokens in {baseline_time:.2f}s)"
        )

        # TurboQuant 6-bit
        cache_6bit = TurboQuantCacheV2(
            head_dim=128, bits=6, device=device, buffer_size=128
        )
        tq6_throughput, tq6_tokens, tq6_time = benchmark_generation(
            model, tokenizer, input_ids, cache_6bit, num_generate
        )
        speedup_6 = tq6_throughput / baseline_throughput * 100
        print(
            f"  TurboQ-6b:  {tq6_throughput:6.2f} tok/s ({tq6_tokens} tokens in {tq6_time:.2f}s) [{speedup_6:.1f}% of baseline]"
        )

        # TurboQuant 8-bit (for comparison)
        cache_8bit = TurboQuantCacheV2(
            head_dim=128, bits=8, device=device, buffer_size=128
        )
        tq8_throughput, tq8_tokens, tq8_time = benchmark_generation(
            model, tokenizer, input_ids, cache_8bit, num_generate
        )
        speedup_8 = tq8_throughput / baseline_throughput * 100
        print(
            f"  TurboQ-8b:  {tq8_throughput:6.2f} tok/s ({tq8_tokens} tokens in {tq8_time:.2f}s) [{speedup_8:.1f}% of baseline]"
        )

        print()

    # Memory usage comparison
    print(f"{'=' * 70}")
    print("Memory Usage Analysis (500 token prompt + 200 generated)")
    print(f"{'=' * 70}")

    # Estimate memory for baseline
    # KV cache: 2 * num_layers * 2 * seq_len * num_heads * head_dim * bytes_per_element
    num_layers = 28  # Qwen2.5-7B
    num_heads = 28
    head_dim = 128
    seq_len = 700  # 500 prompt + 200 generated
    bytes_fp16 = 2

    baseline_kv_bytes = 2 * num_layers * seq_len * num_heads * head_dim * bytes_fp16
    baseline_kv_mb = baseline_kv_bytes / (1024 * 1024)

    # TurboQuant 6-bit with buffer
    buffer_size = 128
    compressed_len = seq_len - buffer_size
    bits = 6
    # Compressed: indices (bits per value) + norms (fp16) + rotation overhead
    # Simplified: indices = compressed_len * bits / 8, norms = compressed_len * 2
    compressed_bytes_per_head = compressed_len * bits // 8 + compressed_len * 2
    buffer_bytes_per_head = buffer_size * head_dim * bytes_fp16
    tq6_kv_bytes = (
        2 * num_layers * num_heads * (compressed_bytes_per_head + buffer_bytes_per_head)
    )
    tq6_kv_mb = tq6_kv_bytes / (1024 * 1024)

    compression_ratio = baseline_kv_bytes / tq6_kv_bytes

    print(f"  Baseline KV cache:    {baseline_kv_mb:.1f} MB")
    print(f"  TurboQuant-6b cache:  {tq6_kv_mb:.1f} MB")
    print(f"  Compression ratio:    {compression_ratio:.2f}x")


if __name__ == "__main__":
    main()
