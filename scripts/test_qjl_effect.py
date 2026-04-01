"""
Test QJL correction effect on attention score preservation.

Hypothesis: TurboQuant_mse (no QJL) has biased inner products.
            TurboQuant_prod (with QJL) should have unbiased inner products.
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


def test_qjl_effect():
    """
    Compare attention preservation with and without QJL correction.
    """
    print("=" * 70)
    print("QJL CORRECTION EFFECT ON ATTENTION")
    print("=" * 70)

    device = torch.device("cuda")
    head_dim = 128
    n_qjl = 128
    n_outliers = 32
    seq_len = 20

    torch.manual_seed(42)

    # Setup
    Pi = generate_rotation_matrix(head_dim, dtype=torch.float32, device=device)
    S = generate_qjl_matrix(head_dim, n_qjl, dtype=torch.float32, device=device)
    c2 = get_centroids_2bit(head_dim, device=device)
    c3 = get_centroids_3bit(head_dim, device=device)

    # Realistic data
    Q = torch.randn(1, seq_len, head_dim, device=device) * 15
    K = torch.randn(1, seq_len, head_dim, device=device) * 15
    V = torch.randn(1, seq_len, head_dim, device=device) * 0.5

    def compress_decompress(x, use_qjl=False):
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
            use_qjl=use_qjl,
        )
        x_hat = x_hat_normalized * norms
        return x_hat.unsqueeze(0)

    # Test without QJL (TurboQuant_mse)
    print("\n--- TurboQuant_mse (NO QJL) ---")
    K_hat_no_qjl = compress_decompress(K, use_qjl=False)
    V_hat_no_qjl = compress_decompress(V, use_qjl=False)

    scale = 1.0 / (head_dim**0.5)
    attn_baseline = F.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
    attn_no_qjl = F.softmax(Q @ K_hat_no_qjl.transpose(-2, -1) * scale, dim=-1)

    output_baseline = attn_baseline @ V
    output_no_qjl = attn_no_qjl @ V_hat_no_qjl

    print(
        f"K cosine sim: {F.cosine_similarity(K.flatten(), K_hat_no_qjl.flatten(), dim=0).item():.6f}"
    )
    print(f"Attention max diff: {(attn_baseline - attn_no_qjl).abs().max().item():.6f}")
    print(
        f"Output cosine sim: {F.cosine_similarity(output_baseline.flatten(), output_no_qjl.flatten(), dim=0).item():.6f}"
    )

    # Count argmax mismatches
    mismatches_no_qjl = 0
    for i in range(seq_len):
        if attn_baseline[0, i].argmax() != attn_no_qjl[0, i].argmax():
            mismatches_no_qjl += 1
    print(f"Attention argmax mismatches: {mismatches_no_qjl}/{seq_len}")

    # Test with QJL (TurboQuant_prod)
    print("\n--- TurboQuant_prod (WITH QJL) ---")
    K_hat_qjl = compress_decompress(K, use_qjl=True)
    V_hat_qjl = compress_decompress(V, use_qjl=True)

    attn_qjl = F.softmax(Q @ K_hat_qjl.transpose(-2, -1) * scale, dim=-1)
    output_qjl = attn_qjl @ V_hat_qjl

    print(
        f"K cosine sim: {F.cosine_similarity(K.flatten(), K_hat_qjl.flatten(), dim=0).item():.6f}"
    )
    print(f"Attention max diff: {(attn_baseline - attn_qjl).abs().max().item():.6f}")
    print(
        f"Output cosine sim: {F.cosine_similarity(output_baseline.flatten(), output_qjl.flatten(), dim=0).item():.6f}"
    )

    mismatches_qjl = 0
    for i in range(seq_len):
        if attn_baseline[0, i].argmax() != attn_qjl[0, i].argmax():
            mismatches_qjl += 1
    print(f"Attention argmax mismatches: {mismatches_qjl}/{seq_len}")

    # Direct inner product comparison
    print("\n--- Inner Product Analysis ---")
    # Compute Q @ K^T for a sample query
    q_sample = Q[0, 0]  # (head_dim,)

    ip_baseline = (q_sample @ K[0].T).cpu()  # (seq_len,)
    ip_no_qjl = (q_sample @ K_hat_no_qjl[0].T).cpu()
    ip_qjl = (q_sample @ K_hat_qjl[0].T).cpu()

    print(f"Baseline inner products: {ip_baseline[:5].tolist()}")
    print(f"No QJL inner products: {ip_no_qjl[:5].tolist()}")
    print(f"With QJL inner products: {ip_qjl[:5].tolist()}")

    print(f"\nNo QJL bias (mean error): {(ip_no_qjl - ip_baseline).mean().item():.4f}")
    print(f"With QJL bias (mean error): {(ip_qjl - ip_baseline).mean().item():.4f}")
    print(f"No QJL RMSE: {((ip_no_qjl - ip_baseline) ** 2).mean().sqrt().item():.4f}")
    print(f"With QJL RMSE: {((ip_qjl - ip_baseline) ** 2).mean().sqrt().item():.4f}")


def test_inner_product_bias():
    """
    Directly test the inner product bias mentioned in the paper.

    Paper says TurboQuant_mse has multiplicative bias of 2/π at b=1.
    This bias diminishes with higher bit widths.
    """
    print("\n" + "=" * 70)
    print("INNER PRODUCT BIAS ANALYSIS")
    print("=" * 70)

    device = torch.device("cuda")
    head_dim = 128
    n_qjl = 128
    n_outliers = 32
    n_samples = 1000  # Many samples for statistical analysis

    torch.manual_seed(42)

    # Setup
    Pi = generate_rotation_matrix(head_dim, dtype=torch.float32, device=device)
    S = generate_qjl_matrix(head_dim, n_qjl, dtype=torch.float32, device=device)
    c2 = get_centroids_2bit(head_dim, device=device)
    c3 = get_centroids_3bit(head_dim, device=device)

    # Generate many unit vectors
    x = torch.randn(n_samples, head_dim, device=device)
    x = x / x.norm(dim=-1, keepdim=True)  # Unit norm

    y = torch.randn(n_samples, head_dim, device=device)
    y = y / y.norm(dim=-1, keepdim=True)  # Unit norm

    # True inner products
    true_ip = (x * y).sum(dim=-1)  # (n_samples,)

    # Compress x with TurboQuant_mse (no QJL)
    mask = build_outlier_mask(x, n_outliers)
    idx_all, qjl_bits, gamma = compress_kv_python(x, Pi, S, c2, c3, mask)
    x_hat_mse = decompress_kv_python(
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

    # Compress x with TurboQuant_prod (with QJL)
    x_hat_prod = decompress_kv_python(
        idx_all,
        qjl_bits,
        gamma,
        Pi,
        S,
        c2,
        c3,
        mask,
        target_dtype=torch.float32,
        use_qjl=True,
    )

    # Compute reconstructed inner products
    ip_mse = (x_hat_mse * y).sum(dim=-1)
    ip_prod = (x_hat_prod * y).sum(dim=-1)

    print(f"\n--- Unit Vectors (||x||=||y||=1) ---")
    print(
        f"True inner products: mean={true_ip.mean().item():.6f}, std={true_ip.std().item():.6f}"
    )

    print(f"\nTurboQuant_mse (no QJL):")
    print(
        f"  Reconstructed IPs: mean={ip_mse.mean().item():.6f}, std={ip_mse.std().item():.6f}"
    )
    print(f"  Bias (mean error): {(ip_mse - true_ip).mean().item():.6f}")
    print(f"  RMSE: {((ip_mse - true_ip) ** 2).mean().sqrt().item():.6f}")

    # Check multiplicative bias: E[<y, x_hat>] / E[<y, x>]
    # For random unit vectors, E[<y, x>] ≈ 0, so we need to look at E[<y,x_hat>|<y,x>]
    # Group by true IP value
    print(f"\n  Multiplicative bias analysis:")
    for threshold in [0.1, 0.2, 0.3]:
        mask_pos = true_ip > threshold
        if mask_pos.sum() > 10:
            ratio = ip_mse[mask_pos].mean() / true_ip[mask_pos].mean()
            print(f"    IP > {threshold}: ratio = {ratio.item():.4f}")

    print(f"\nTurboQuant_prod (with QJL):")
    print(
        f"  Reconstructed IPs: mean={ip_prod.mean().item():.6f}, std={ip_prod.std().item():.6f}"
    )
    print(f"  Bias (mean error): {(ip_prod - true_ip).mean().item():.6f}")
    print(f"  RMSE: {((ip_prod - true_ip) ** 2).mean().sqrt().item():.6f}")

    for threshold in [0.1, 0.2, 0.3]:
        mask_pos = true_ip > threshold
        if mask_pos.sum() > 10:
            ratio = ip_prod[mask_pos].mean() / true_ip[mask_pos].mean()
            print(f"    IP > {threshold}: ratio = {ratio.item():.4f}")


if __name__ == "__main__":
    test_qjl_effect()
    test_inner_product_bias()
