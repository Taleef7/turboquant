"""
Unit tests for TurboQuant compression/decompression kernels (Python reference).
Run: pytest scripts/test_kernels.py -v
"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.math_utils import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    get_centroids_2bit,
    get_centroids_3bit,
)
from kernels.compress_kv import (
    compress_kv_python,
    build_outlier_mask,
    compress_values_group_quant,
)
from kernels.decompress_kv import (
    decompress_kv_python,
    decompress_values_group_quant,
)
from kernels.bitpack_triton import (
    pack_4bit,
    unpack_4bit,
    pack_sign_bits,
    unpack_sign_bits,
)


def test_build_outlier_mask_shape():
    d, seq = 128, 16
    x = torch.randn(seq, d)
    mask = build_outlier_mask(x, n_outliers=32)
    assert mask.shape == (d,)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 32


def test_build_outlier_mask_no_duplicates():
    d, seq = 128, 16
    x = torch.randn(seq, d)
    mask = build_outlier_mask(x, n_outliers=32)
    # Verify high-variance channels are selected
    var = x.float().var(dim=0)
    # All False channels should have lower variance than all True channels
    min_outlier_var = var[mask].min().item()
    max_normal_var = var[~mask].max().item()
    assert min_outlier_var >= max_normal_var


def test_compress_kv_python_output_shapes():
    d, seq, k = 128, 10, 128
    Pi = generate_rotation_matrix(d)
    S = generate_qjl_matrix(d, k)
    c2 = get_centroids_2bit(d)
    c3 = get_centroids_3bit(d)
    x = torch.randn(seq, d)
    x = x / x.norm(dim=-1, keepdim=True)
    mask = build_outlier_mask(x)
    idx_all, qjl_bits, gamma = compress_kv_python(x, Pi, S, c2, c3, mask)
    assert idx_all.shape == (seq, d), f"Expected ({seq},{d}), got {idx_all.shape}"
    assert qjl_bits.shape == (seq, k), f"Expected ({seq},{k}), got {qjl_bits.shape}"
    assert gamma.shape == (seq,), f"Expected ({seq},), got {gamma.shape}"


def test_compress_kv_python_output_dtypes():
    d, seq, k = 128, 10, 128
    Pi = generate_rotation_matrix(d)
    S = generate_qjl_matrix(d, k)
    c2 = get_centroids_2bit(d)
    c3 = get_centroids_3bit(d)
    x = torch.randn(seq, d)
    x = x / x.norm(dim=-1, keepdim=True)
    mask = build_outlier_mask(x)
    idx_all, qjl_bits, gamma = compress_kv_python(x, Pi, S, c2, c3, mask)
    assert idx_all.dtype == torch.int8
    assert qjl_bits.dtype == torch.int8
    assert gamma.dtype == torch.float32


def test_compress_kv_python_idx_ranges():
    """Normal channels: indices in [0,3]; outlier channels: indices in [0,7]."""
    d, seq, k = 128, 10, 128
    Pi = generate_rotation_matrix(d)
    S = generate_qjl_matrix(d, k)
    c2 = get_centroids_2bit(d)
    c3 = get_centroids_3bit(d)
    x = torch.randn(seq, d)
    x = x / x.norm(dim=-1, keepdim=True)
    mask = build_outlier_mask(x)
    idx_all, qjl_bits, gamma = compress_kv_python(x, Pi, S, c2, c3, mask)
    # Normal channels: 4 levels (0-3)
    normal_idx = idx_all[:, ~mask]
    assert (normal_idx >= 0).all() and (normal_idx <= 3).all(), (
        "Normal channel indices out of [0,3]"
    )
    # Outlier channels: 8 levels (0-7)
    outlier_idx = idx_all[:, mask]
    assert (outlier_idx >= 0).all() and (outlier_idx <= 7).all(), (
        "Outlier channel indices out of [0,7]"
    )


def test_compress_qjl_bits_are_plus_minus_one():
    """qjl_bits must contain only +1 and -1 (no zeros)."""
    d, seq, k = 128, 10, 128
    Pi = generate_rotation_matrix(d)
    S = generate_qjl_matrix(d, k)
    c2 = get_centroids_2bit(d)
    c3 = get_centroids_3bit(d)
    x = torch.randn(seq, d)
    x = x / x.norm(dim=-1, keepdim=True)
    mask = build_outlier_mask(x)
    _, qjl_bits, _ = compress_kv_python(x, Pi, S, c2, c3, mask)
    unique_vals = qjl_bits.unique().tolist()
    assert set(unique_vals).issubset({-1, 1}), (
        f"qjl_bits contains unexpected values: {unique_vals}"
    )


def test_compress_gamma_positive():
    """gamma (residual norm) must be positive for non-trivial input."""
    d, seq, k = 128, 10, 128
    Pi = generate_rotation_matrix(d)
    S = generate_qjl_matrix(d, k)
    c2 = get_centroids_2bit(d)
    c3 = get_centroids_3bit(d)
    x = torch.randn(seq, d)
    x = x / x.norm(dim=-1, keepdim=True)
    mask = build_outlier_mask(x)
    _, _, gamma = compress_kv_python(x, Pi, S, c2, c3, mask)
    assert (gamma > 0).all(), "Some gamma values are <= 0"


# ---------------------------------------------------------------------------
# Decompression kernel tests
# ---------------------------------------------------------------------------


def test_decompress_output_shape():
    d, seq, k = 128, 10, 128
    torch.manual_seed(1)
    Pi = generate_rotation_matrix(d)
    S = generate_qjl_matrix(d, k)
    c2 = get_centroids_2bit(d)
    c3 = get_centroids_3bit(d)
    x = torch.randn(seq, d)
    x = x / x.norm(dim=-1, keepdim=True)
    mask = build_outlier_mask(x)
    idx_all, qjl_bits, gamma = compress_kv_python(x, Pi, S, c2, c3, mask)
    x_hat = decompress_kv_python(
        idx_all, qjl_bits, gamma, Pi, S, c2, c3, mask, torch.float32
    )
    assert x_hat.shape == (seq, d)
    assert x_hat.dtype == torch.float32, f"Expected float32, got {x_hat.dtype}"


def test_roundtrip_mse():
    """Round-trip MSE should be < 0.01 for unit-norm vectors."""
    d, seq, k = 128, 16, 128
    torch.manual_seed(0)
    Pi = generate_rotation_matrix(d)
    S = generate_qjl_matrix(d, k)
    c2 = get_centroids_2bit(d)
    c3 = get_centroids_3bit(d)
    x = torch.randn(seq, d)
    x = x / x.norm(dim=-1, keepdim=True)
    mask = build_outlier_mask(x)
    idx_all, qjl_bits, gamma = compress_kv_python(x, Pi, S, c2, c3, mask)
    x_hat = decompress_kv_python(
        idx_all, qjl_bits, gamma, Pi, S, c2, c3, mask, torch.float32
    )
    mse = ((x - x_hat) ** 2).mean().item()
    assert mse < 0.01, f"Round-trip MSE too high: {mse}"


def test_inner_product_preservation():
    """Inner product error should be < 0.05 on average."""
    d, seq, k = 128, 16, 128
    torch.manual_seed(2)
    Pi = generate_rotation_matrix(d)
    S = generate_qjl_matrix(d, k)
    c2 = get_centroids_2bit(d)
    c3 = get_centroids_3bit(d)
    x = torch.randn(seq, d)
    x = x / x.norm(dim=-1, keepdim=True)
    mask = build_outlier_mask(x)
    idx_all, qjl_bits, gamma = compress_kv_python(x, Pi, S, c2, c3, mask)
    x_hat = decompress_kv_python(
        idx_all, qjl_bits, gamma, Pi, S, c2, c3, mask, torch.float32
    )
    ip_error = ((x @ x.T) - (x_hat @ x.T)).abs().mean().item()
    assert ip_error < 0.05, f"Inner product error too high: {ip_error}"


def test_compress_values_group_quant_output_shapes():
    seq, d, group_size = 10, 128, 32
    x = torch.randn(seq, d)

    idx_all, mins, scales = compress_values_group_quant(x, group_size=group_size)

    assert idx_all.shape == (seq, d)
    assert mins.shape == (seq, d // group_size)
    assert scales.shape == (seq, d // group_size)
    assert idx_all.dtype == torch.int8
    assert mins.dtype == torch.float32
    assert scales.dtype == torch.float32


def test_values_group_quant_roundtrip_cosine_similarity():
    seq, d, group_size = 64, 128, 32
    torch.manual_seed(123)
    # Values in real models tend to have compact dynamic ranges.
    x = 0.5 * torch.randn(seq, d)

    idx_all, mins, scales = compress_values_group_quant(x, group_size=group_size)
    x_hat = decompress_values_group_quant(
        idx_all,
        mins,
        scales,
        group_size=group_size,
        target_dtype=torch.float32,
    )

    cosine = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
    assert cosine > 0.99, f"Value roundtrip cosine similarity too low: {cosine}"


def test_bitpack_4bit_roundtrip():
    torch.manual_seed(7)
    idx = torch.randint(0, 16, (11, 129), dtype=torch.int8)
    packed = pack_4bit(idx)
    unpacked = unpack_4bit(packed, n_cols=idx.shape[1])
    assert packed.dtype == torch.uint8
    assert unpacked.dtype == torch.int8
    assert torch.equal(idx, unpacked)


def test_bitpack_sign_roundtrip():
    torch.manual_seed(11)
    bits = torch.randint(0, 2, (13, 257), dtype=torch.int8)
    signs = torch.where(bits > 0, torch.ones_like(bits), -torch.ones_like(bits))
    packed = pack_sign_bits(signs)
    unpacked = unpack_sign_bits(packed, n_cols=signs.shape[1])
    assert packed.dtype == torch.uint8
    assert unpacked.dtype == torch.int8
    assert torch.equal(signs, unpacked)


# NOTE: TurboQuantCache v1 integration tests removed during cleanup.
# The v1 cache has been archived. Use TurboQuantCacheV2 for new tests.
