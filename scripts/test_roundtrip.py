#!/usr/bin/env python3
"""
Test if compress->decompress roundtrip preserves enough information
for correct attention computation.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force Python fallback
os.environ["TURBOQUANT_FORCE_PYTHON"] = "1"

from core.turboquant_cache import TurboQuantCache


def test_roundtrip():
    """Test if compress->decompress is invertible enough for correct predictions."""

    # Create cache
    cache = TurboQuantCache(
        head_dim=128,
        n_qjl=128,
        n_outliers=32,
        device=torch.device("cuda"),
        dtype=torch.float16,
        use_qjl=False,
        bit_width=3,
    )

    # Create random KV tensors (simulating prefill)
    batch, heads, seq, head_dim = 1, 4, 5, 128

    torch.manual_seed(100)
    k_original = torch.randn(
        batch, heads, seq, head_dim, dtype=torch.float16, device="cuda"
    )
    torch.manual_seed(200)
    v_original = torch.randn(
        batch, heads, seq, head_dim, dtype=torch.float16, device="cuda"
    )

    print(f"Original K: shape={k_original.shape}, norm={k_original.norm():.4f}")
    print(f"Original V: shape={v_original.shape}, norm={v_original.norm():.4f}")

    # Compress and decompress
    k_compressed = cache._compress(k_original)
    k_roundtrip = cache._decompress(k_compressed)

    v_compressed = cache._compress(v_original)
    v_roundtrip = cache._decompress(v_compressed)

    print(f"\nRoundtrip K: shape={k_roundtrip.shape}, norm={k_roundtrip.norm():.4f}")
    print(f"Roundtrip V: shape={v_roundtrip.shape}, norm={v_roundtrip.norm():.4f}")

    # Check reconstruction quality
    k_mse = ((k_original.float() - k_roundtrip.float()) ** 2).mean()
    v_mse = ((v_original.float() - v_roundtrip.float()) ** 2).mean()

    k_cosine = torch.nn.functional.cosine_similarity(
        k_original.flatten(), k_roundtrip.flatten(), dim=0
    )
    v_cosine = torch.nn.functional.cosine_similarity(
        v_original.flatten(), v_roundtrip.flatten(), dim=0
    )

    print(f"\nReconstruction Quality:")
    print(f"  K MSE: {k_mse:.6f}, Cosine: {k_cosine:.6f}")
    print(f"  V MSE: {v_mse:.6f}, Cosine: {v_cosine:.6f}")

    # Simulate attention computation
    q = torch.randn(batch, heads, 1, head_dim, dtype=torch.float16, device="cuda")

    # Attention with original KV
    attn_original = torch.matmul(q, k_original.transpose(-2, -1)) / (head_dim**0.5)
    probs_original = torch.nn.functional.softmax(attn_original, dim=-1)
    out_original = torch.matmul(probs_original, v_original)

    # Attention with roundtrip KV
    attn_roundtrip = torch.matmul(q, k_roundtrip.transpose(-2, -1)) / (head_dim**0.5)
    probs_roundtrip = torch.nn.functional.softmax(attn_roundtrip, dim=-1)
    out_roundtrip = torch.matmul(probs_roundtrip, v_roundtrip)

    # Compare attention outputs
    attn_diff = (attn_original - attn_roundtrip).abs().max()
    probs_diff = (probs_original - probs_roundtrip).abs().max()
    out_diff = (out_original - out_roundtrip).abs().max()

    print(f"\nAttention Computation:")
    print(f"  Logits max diff: {attn_diff:.6f}")
    print(f"  Probs max diff: {probs_diff:.6f}")
    print(f"  Output max diff: {out_diff:.6f}")

    print(f"\n  Original probs: {probs_original.flatten()[:10]}")
    print(f"  Roundtrip probs: {probs_roundtrip.flatten()[:10]}")

    # Check argmax stability
    argmax_original = probs_original.argmax(dim=-1)
    argmax_roundtrip = probs_roundtrip.argmax(dim=-1)
    argmax_match = (argmax_original == argmax_roundtrip).float().mean()

    print(f"\n  Argmax match rate: {argmax_match:.2%}")

    if argmax_match < 1.0:
        print(f"  WARNING: Argmax mismatch detected!")
        print(f"  Original argmax: {argmax_original}")
        print(f"  Roundtrip argmax: {argmax_roundtrip}")


if __name__ == "__main__":
    test_roundtrip()
