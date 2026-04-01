"""
Test script to verify attention score preservation with TurboQuant compression.

This test checks if the attention scores computed with compressed KV cache
are close enough to baseline attention scores for correct model behavior.
"""

import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.math_utils import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    get_centroids_2bit,
    get_centroids_3bit,
)
from kernels.compress_kv import compress_kv_python, build_outlier_mask
from kernels.decompress_kv import decompress_kv_python


def test_attention_score_preservation():
    """
    Test how well TurboQuant preserves attention scores.

    Attention = softmax(Q @ K.T / sqrt(d)) @ V

    We need both K and V to be accurate for correct output.
    """
    print("=" * 70)
    print("ATTENTION SCORE PRESERVATION TEST")
    print("=" * 70)

    device = torch.device("cuda")
    head_dim = 128
    n_qjl = 128
    n_outliers = 32
    seq_len = 20  # Short sequence for detailed analysis

    torch.manual_seed(42)

    # Generate random Q, K, V with realistic magnitudes
    # Based on diagnostic: real KV norms range 5-500, mean ~200
    Q = torch.randn(1, seq_len, head_dim, device=device) * 10
    K = torch.randn(1, seq_len, head_dim, device=device) * 10
    V = torch.randn(1, seq_len, head_dim, device=device) * 10

    # Setup TurboQuant
    Pi = generate_rotation_matrix(head_dim, dtype=torch.float32, device=device)
    S = generate_qjl_matrix(head_dim, n_qjl, dtype=torch.float32, device=device)
    c2 = get_centroids_2bit(head_dim, device=device)
    c3 = get_centroids_3bit(head_dim, device=device)

    def compress_decompress(x):
        """Full TurboQuant roundtrip with normalization."""
        # x: (1, seq_len, head_dim)
        x_flat = x.squeeze(0).to(torch.float32)  # (seq_len, head_dim)

        # Normalize to unit norm
        norms = torch.norm(x_flat, dim=-1, keepdim=True)
        norms_safe = torch.clamp(norms, min=1e-8)
        x_normalized = x_flat / norms_safe

        # Build mask and compress
        mask = build_outlier_mask(x_normalized, n_outliers)
        idx_all, qjl_bits, gamma = compress_kv_python(x_normalized, Pi, S, c2, c3, mask)

        # Decompress (without QJL for best MSE)
        x_hat_normalized = decompress_kv_python(
            idx_all,
            qjl_bits,
            gamma,
            Pi,
            S,
            c2,
            c3,
            mask,
            target_dtype=torch.float32,
            use_qjl=False,
        )

        # Rescale
        x_hat = x_hat_normalized * norms

        return x_hat.unsqueeze(0)  # (1, seq_len, head_dim)

    # Compress K and V
    K_hat = compress_decompress(K)
    V_hat = compress_decompress(V)

    # Compute baseline attention
    scale = 1.0 / (head_dim**0.5)
    attn_scores_baseline = F.softmax(
        Q @ K.transpose(-2, -1) * scale, dim=-1
    )  # (1, seq_len, seq_len)
    output_baseline = attn_scores_baseline @ V  # (1, seq_len, head_dim)

    # Compute compressed attention
    attn_scores_compressed = F.softmax(Q @ K_hat.transpose(-2, -1) * scale, dim=-1)
    output_compressed = attn_scores_compressed @ V_hat

    # Analysis
    print("\n--- Key (K) Reconstruction ---")
    k_mse = torch.mean((K - K_hat) ** 2).item()
    k_cos = F.cosine_similarity(K.flatten(), K_hat.flatten(), dim=0).item()
    print(f"MSE: {k_mse:.6f}")
    print(f"Cosine Sim: {k_cos:.6f}")

    print("\n--- Value (V) Reconstruction ---")
    v_mse = torch.mean((V - V_hat) ** 2).item()
    v_cos = F.cosine_similarity(V.flatten(), V_hat.flatten(), dim=0).item()
    print(f"MSE: {v_mse:.6f}")
    print(f"Cosine Sim: {v_cos:.6f}")

    print("\n--- Attention Scores (before softmax: Q @ K^T / sqrt(d)) ---")
    raw_scores_baseline = Q @ K.transpose(-2, -1) * scale
    raw_scores_compressed = Q @ K_hat.transpose(-2, -1) * scale
    raw_diff = (raw_scores_baseline - raw_scores_compressed).abs()
    print(f"Max diff: {raw_diff.max().item():.6f}")
    print(f"Mean diff: {raw_diff.mean().item():.6f}")
    print(
        f"Raw score range (baseline): [{raw_scores_baseline.min().item():.3f}, {raw_scores_baseline.max().item():.3f}]"
    )
    print(
        f"Raw score range (compressed): [{raw_scores_compressed.min().item():.3f}, {raw_scores_compressed.max().item():.3f}]"
    )

    print("\n--- Attention Weights (after softmax) ---")
    attn_diff = (attn_scores_baseline - attn_scores_compressed).abs()
    print(f"Max diff: {attn_diff.max().item():.6f}")
    print(f"Mean diff: {attn_diff.mean().item():.6f}")

    # Check if attention distribution is corrupted
    # If softmax is dominated by a few entries, small errors can flip the ordering
    print("\n--- Attention Weight Distribution (row 0) ---")
    print(f"Baseline top-5: {attn_scores_baseline[0, 0].topk(5).values.tolist()}")
    print(f"Compressed top-5: {attn_scores_compressed[0, 0].topk(5).values.tolist()}")
    print(
        f"Baseline top-5 indices: {attn_scores_baseline[0, 0].topk(5).indices.tolist()}"
    )
    print(
        f"Compressed top-5 indices: {attn_scores_compressed[0, 0].topk(5).indices.tolist()}"
    )

    print("\n--- Final Output ---")
    output_diff = (output_baseline - output_compressed).abs()
    print(f"Max diff: {output_diff.max().item():.6f}")
    print(f"Mean diff: {output_diff.mean().item():.6f}")
    output_cos = F.cosine_similarity(
        output_baseline.flatten(), output_compressed.flatten(), dim=0
    ).item()
    print(f"Cosine Sim: {output_cos:.6f}")

    # Detailed per-position analysis
    print("\n--- Per-Position Attention Error ---")
    for i in range(min(5, seq_len)):
        pos_attn_baseline = attn_scores_baseline[0, i]
        pos_attn_compressed = attn_scores_compressed[0, i]
        pos_diff = (pos_attn_baseline - pos_attn_compressed).abs()

        # KL divergence
        kl = F.kl_div(
            pos_attn_compressed.log(), pos_attn_baseline, reduction="sum"
        ).item()

        # Total variation distance
        tvd = 0.5 * pos_diff.sum().item()

        print(
            f"Position {i}: Max Diff={pos_diff.max().item():.4f}, TVD={tvd:.4f}, KL={kl:.4f}"
        )

    return {
        "k_cos": k_cos,
        "v_cos": v_cos,
        "output_cos": output_cos,
        "attn_max_diff": attn_diff.max().item(),
    }


def test_with_real_kv_distribution():
    """
    Test with KV distributions closer to what we observed in the model.

    From diagnostics:
    - Layer 0 Keys: mean ~1.3, std ~27, norms 107-466
    - Layer 0 Values: mean ~0, std ~0.26, norms 0.4-4.4
    """
    print("\n" + "=" * 70)
    print("TEST WITH REALISTIC KV DISTRIBUTION")
    print("=" * 70)

    device = torch.device("cuda")
    head_dim = 128
    n_qjl = 128
    n_outliers = 32
    seq_len = 20

    torch.manual_seed(42)

    # Setup TurboQuant
    Pi = generate_rotation_matrix(head_dim, dtype=torch.float32, device=device)
    S = generate_qjl_matrix(head_dim, n_qjl, dtype=torch.float32, device=device)
    c2 = get_centroids_2bit(head_dim, device=device)
    c3 = get_centroids_3bit(head_dim, device=device)

    # Generate realistic K, V based on observed distributions
    # Keys: high variance, large norms
    K = torch.randn(1, seq_len, head_dim, device=device) * 27 + 1.3

    # Values: low variance, small norms
    V = torch.randn(1, seq_len, head_dim, device=device) * 0.26

    # Query similar to Keys
    Q = torch.randn(1, seq_len, head_dim, device=device) * 27 + 1.3

    print(
        f"K norms: [{torch.norm(K, dim=-1).min().item():.1f}, {torch.norm(K, dim=-1).max().item():.1f}]"
    )
    print(
        f"V norms: [{torch.norm(V, dim=-1).min().item():.1f}, {torch.norm(V, dim=-1).max().item():.1f}]"
    )

    def compress_decompress(x):
        x_flat = x.squeeze(0).to(torch.float32)
        norms = torch.norm(x_flat, dim=-1, keepdim=True)
        norms_safe = torch.clamp(norms, min=1e-8)
        x_normalized = x_flat / norms_safe
        mask = build_outlier_mask(x_normalized, n_outliers)
        idx_all, qjl_bits, gamma = compress_kv_python(x_normalized, Pi, S, c2, c3, mask)
        x_hat_normalized = decompress_kv_python(
            idx_all,
            qjl_bits,
            gamma,
            Pi,
            S,
            c2,
            c3,
            mask,
            target_dtype=torch.float32,
            use_qjl=False,
        )
        x_hat = x_hat_normalized * norms
        return x_hat.unsqueeze(0)

    K_hat = compress_decompress(K)
    V_hat = compress_decompress(V)

    # Compute attention
    scale = 1.0 / (head_dim**0.5)
    attn_baseline = F.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
    attn_compressed = F.softmax(Q @ K_hat.transpose(-2, -1) * scale, dim=-1)

    output_baseline = attn_baseline @ V
    output_compressed = attn_compressed @ V_hat

    print("\n--- Attention Weights Comparison ---")
    attn_diff = (attn_baseline - attn_compressed).abs()
    print(f"Max diff: {attn_diff.max().item():.6f}")
    print(f"Mean diff: {attn_diff.mean().item():.6f}")

    print("\n--- Output Comparison ---")
    output_cos = F.cosine_similarity(
        output_baseline.flatten(), output_compressed.flatten(), dim=0
    ).item()
    print(f"Output cosine sim: {output_cos:.6f}")

    # Check if the attention weights have shifted dramatically
    print("\n--- Attention Weight Distribution Check ---")
    for i in range(min(3, seq_len)):
        baseline_argmax = attn_baseline[0, i].argmax().item()
        compressed_argmax = attn_compressed[0, i].argmax().item()
        match = "MATCH" if baseline_argmax == compressed_argmax else "MISMATCH!"
        print(
            f"Position {i}: Baseline argmax={baseline_argmax}, Compressed argmax={compressed_argmax} [{match}]"
        )


def test_attention_with_causal_mask():
    """
    Test attention with causal mask (like in autoregressive generation).
    """
    print("\n" + "=" * 70)
    print("TEST WITH CAUSAL MASK (AUTOREGRESSIVE)")
    print("=" * 70)

    device = torch.device("cuda")
    head_dim = 128
    n_qjl = 128
    n_outliers = 32
    seq_len = 10

    torch.manual_seed(42)

    # Setup
    Pi = generate_rotation_matrix(head_dim, dtype=torch.float32, device=device)
    S = generate_qjl_matrix(head_dim, n_qjl, dtype=torch.float32, device=device)
    c2 = get_centroids_2bit(head_dim, device=device)
    c3 = get_centroids_3bit(head_dim, device=device)

    # Realistic magnitudes
    Q = torch.randn(1, seq_len, head_dim, device=device) * 15
    K = torch.randn(1, seq_len, head_dim, device=device) * 15
    V = torch.randn(1, seq_len, head_dim, device=device) * 0.5

    def compress_decompress(x):
        x_flat = x.squeeze(0).to(torch.float32)
        norms = torch.norm(x_flat, dim=-1, keepdim=True)
        norms_safe = torch.clamp(norms, min=1e-8)
        x_normalized = x_flat / norms_safe
        mask = build_outlier_mask(x_normalized, n_outliers)
        idx_all, qjl_bits, gamma = compress_kv_python(x_normalized, Pi, S, c2, c3, mask)
        x_hat_normalized = decompress_kv_python(
            idx_all,
            qjl_bits,
            gamma,
            Pi,
            S,
            c2,
            c3,
            mask,
            target_dtype=torch.float32,
            use_qjl=False,
        )
        x_hat = x_hat_normalized * norms
        return x_hat.unsqueeze(0)

    K_hat = compress_decompress(K)
    V_hat = compress_decompress(V)

    # Create causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1
    )

    # Compute attention with causal mask
    scale = 1.0 / (head_dim**0.5)

    raw_baseline = Q @ K.transpose(-2, -1) * scale + causal_mask
    raw_compressed = Q @ K_hat.transpose(-2, -1) * scale + causal_mask

    attn_baseline = F.softmax(raw_baseline, dim=-1)
    attn_compressed = F.softmax(raw_compressed, dim=-1)

    output_baseline = attn_baseline @ V
    output_compressed = attn_compressed @ V_hat

    print("\n--- With Causal Mask ---")

    # Analyze position by position (in autoregressive, later positions have more history)
    for i in range(seq_len):
        # Position i can only attend to positions 0..i
        valid_range = i + 1

        baseline_weights = attn_baseline[0, i, :valid_range]
        compressed_weights = attn_compressed[0, i, :valid_range]

        # Total variation distance
        tvd = 0.5 * (baseline_weights - compressed_weights).abs().sum().item()

        # Check argmax
        baseline_argmax = baseline_weights.argmax().item()
        compressed_argmax = compressed_weights.argmax().item()
        match = "OK" if baseline_argmax == compressed_argmax else "MISMATCH!"

        print(
            f"Position {i} (attends to 0..{i}): TVD={tvd:.4f}, argmax: baseline={baseline_argmax}, compressed={compressed_argmax} [{match}]"
        )

    print(
        f"\nOutput cosine similarity: {F.cosine_similarity(output_baseline.flatten(), output_compressed.flatten(), dim=0).item():.6f}"
    )


if __name__ == "__main__":
    test_attention_score_preservation()
    test_with_real_kv_distribution()
    test_attention_with_causal_mask()
