#!/usr/bin/env python3
"""Profile where TurboQuant is spending time during generation."""

import torch
import time
import sys

sys.path.insert(0, "/home/taleef/projects/turboquant")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.turboquant_cache_v2 import TurboQuantCacheV2


class ProfilingCache(TurboQuantCacheV2):
    """TurboQuantCacheV2 with detailed timing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compress_times = []
        self.decompress_times = []
        self.update_count = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.update_count += 1

        # Time the full update
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        self.compress_times.append(elapsed)
        return result

    def get_stats(self):
        if not self.compress_times:
            return {}

        return {
            "update_calls": self.update_count,
            "avg_update_ms": sum(self.compress_times) * 1000 / len(self.compress_times),
            "total_update_ms": sum(self.compress_times) * 1000,
            "min_update_ms": min(self.compress_times) * 1000,
            "max_update_ms": max(self.compress_times) * 1000,
        }


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

    # Create prompt
    prompt_len = 100
    base_text = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = tokenizer.encode(base_text)[:prompt_len]
    input_ids = torch.tensor([tokens], device=device)

    print(f"\nProfiling with {prompt_len} token prompt, generating 50 tokens")
    print("=" * 60)

    # Warmup run (triggers JIT)
    print("Warmup run (JIT compilation)...")
    warmup_cache = ProfilingCache(head_dim=128, bits=6, device=device, buffer_size=128)
    with torch.no_grad():
        _ = model.generate(
            input_ids, max_new_tokens=10, do_sample=False, past_key_values=warmup_cache
        )
    print(f"  Warmup stats: {warmup_cache.get_stats()}")

    # Actual profiling run
    print("\nProfiling run (after JIT warmup)...")
    profile_cache = ProfilingCache(head_dim=128, bits=6, device=device, buffer_size=128)

    torch.cuda.synchronize()
    total_start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids, max_new_tokens=50, do_sample=False, past_key_values=profile_cache
        )

    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - total_start

    generated_tokens = outputs.shape[1] - input_ids.shape[1]

    print(f"\nResults:")
    print(f"  Generated: {generated_tokens} tokens in {total_elapsed * 1000:.1f}ms")
    print(f"  Throughput: {generated_tokens / total_elapsed:.2f} tok/s")
    print(f"\nCache stats:")
    stats = profile_cache.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # Calculate overhead
    cache_overhead = stats["total_update_ms"]
    model_time = total_elapsed * 1000 - cache_overhead
    print(f"\nTime breakdown:")
    print(
        f"  Model forward: {model_time:.1f}ms ({model_time / (total_elapsed * 1000) * 100:.1f}%)"
    )
    print(
        f"  Cache update:  {cache_overhead:.1f}ms ({cache_overhead / (total_elapsed * 1000) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
