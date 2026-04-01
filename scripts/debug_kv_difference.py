#!/usr/bin/env python3
"""
Debug why K has worse compression quality than V.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TURBOQUANT_FORCE_PYTHON"] = "1"

from core.turboquant_cache import TurboQuantCache


def test_separate_kv():
    """Test K and V compression separately to isolate the issue."""

    # Create TWO caches with same seed to ensure identical rotation matrices
    cache_k = TurboQuantCache(
        head_dim=128,
        n_qjl=128,
        n_outliers=32,
        device=torch.device("cuda"),
        dtype=torch.float16,
        use_qjl=False,
        bit_width=3,
        seed=42,
    )

    cache_v = TurboQuantCache(
        head_dim=128,
        n_qjl=128,
        n_outliers=32,
        device=torch.device("cuda"),
        dtype=torch.float16,
        use_qjl=False,
        bit_width=3,
        seed=42,
    )

    batch, heads, seq, head_dim = 1, 4, 5, 128

    # Use same seed for K and V generation
    torch.manual_seed(100)
    k_original = torch.randn(
        batch, heads, seq, head_dim, dtype=torch.float16, device="cuda"
    )

    torch.manual_seed(200)
    v_original = torch.randn(
        batch, heads, seq, head_dim, dtype=torch.float16, device="cuda"
    )

    print(
        f"K stats: mean={k_original.mean():.4f}, std={k_original.std():.4f}, norm={k_original.norm():.4f}"
    )
    print(
        f"V stats: mean={v_original.mean():.4f}, std={v_original.std():.4f}, norm={v_original.norm():.4f}"
    )

    # Compress K
    k_compressed = cache_k._compress(k_original)
    k_roundtrip = cache_k._decompress(k_compressed)

    # Compress V
    v_compressed = cache_v._compress(v_original)
    v_roundtrip = cache_v._decompress(v_compressed)

    # Measure quality
    k_cosine = torch.nn.functional.cosine_similarity(
        k_original.flatten(), k_roundtrip.flatten(), dim=0
    )
    v_cosine = torch.nn.functional.cosine_similarity(
        v_original.flatten(), v_roundtrip.flatten(), dim=0
    )

    k_mse = ((k_original.float() - k_roundtrip.float()) ** 2).mean()
    v_mse = ((v_original.float() - v_roundtrip.float()) ** 2).mean()

    print(
        f"\nK roundtrip: cosine={k_cosine:.6f}, MSE={k_mse:.6f}, norm={k_roundtrip.norm():.4f}"
    )
    print(
        f"V roundtrip: cosine={v_cosine:.6f}, MSE={v_mse:.6f}, norm={v_roundtrip.norm():.4f}"
    )

    # Check if using the same cache for both helps
    print("\n--- Testing with SAME cache object ---")
    cache_shared = TurboQuantCache(
        head_dim=128,
        n_qjl=128,
        n_outliers=32,
        device=torch.device("cuda"),
        dtype=torch.float16,
        use_qjl=False,
        bit_width=3,
        seed=42,
    )

    k_compressed_2 = cache_shared._compress(k_original)
    k_roundtrip_2 = cache_shared._decompress(k_compressed_2)

    v_compressed_2 = cache_shared._compress(v_original)
    v_roundtrip_2 = cache_shared._decompress(v_compressed_2)

    k_cosine_2 = torch.nn.functional.cosine_similarity(
        k_original.flatten(), k_roundtrip_2.flatten(), dim=0
    )
    v_cosine_2 = torch.nn.functional.cosine_similarity(
        v_original.flatten(), v_roundtrip_2.flatten(), dim=0
    )

    print(f"K roundtrip (shared): cosine={k_cosine_2:.6f}")
    print(f"V roundtrip (shared): cosine={v_cosine_2:.6f}")

    # Check mask differences
    _, _, _, _, mask_k, *_ = k_compressed_2
    _, _, _, _, mask_v, *_ = v_compressed_2

    print(f"\nMask comparison:")
    print(f"  K mask sum: {mask_k.sum()}")
    print(f"  V mask sum: {mask_v.sum()}")
    print(f"  Masks equal: {torch.equal(mask_k, mask_v)}")


if __name__ == "__main__":
    test_separate_kv()
