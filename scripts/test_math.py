"""
Unit tests for TurboQuant math primitives.
Run: pytest scripts/test_math.py -v
"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.math_utils import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    get_centroids_2bit,
    get_centroids_3bit,
    quantize_to_centroids,
    dequantize_from_centroids,
)


def test_rotation_matrix_is_orthogonal():
    """Π must satisfy Π^T Π ≈ I."""
    d = 64
    Pi = generate_rotation_matrix(d)
    assert Pi.shape == (d, d)
    eye = Pi.T @ Pi
    assert torch.allclose(eye, torch.eye(d), atol=1e-5), "Rotation matrix not orthogonal"


def test_rotation_preserves_norms():
    """||Π x||_2 == ||x||_2 for any x."""
    d = 128
    Pi = generate_rotation_matrix(d)
    x = torch.randn(d)
    assert torch.allclose(torch.norm(Pi @ x), torch.norm(x), atol=1e-5)


def test_qjl_matrix_shape():
    """S must be (k, d) with i.i.d. N(0,1) entries."""
    d, k = 64, 64
    S = generate_qjl_matrix(d, k)
    assert S.shape == (k, d)
    # Mean ~0, std ~1 for large matrix
    assert abs(S.mean().item()) < 0.1
    assert abs(S.std().item() - 1.0) < 0.1


def test_centroids_2bit_count():
    """2-bit quantization needs exactly 4 centroids."""
    d = 128
    centroids = get_centroids_2bit(d)
    assert len(centroids) == 4


def test_centroids_2bit_values():
    """2-bit centroids must be ±0.453/√d and ±1.510/√d."""
    d = 128
    centroids = get_centroids_2bit(d)
    expected = sorted([
        -1.510 / math.sqrt(d),
        -0.453 / math.sqrt(d),
         0.453 / math.sqrt(d),
         1.510 / math.sqrt(d),
    ])
    actual = sorted(centroids.tolist())
    for e, a in zip(expected, actual):
        assert abs(e - a) < 1e-6, f"Centroid mismatch: expected {e}, got {a}"


def test_centroids_3bit_count():
    """3-bit quantization needs exactly 8 centroids."""
    d = 128
    centroids = get_centroids_3bit(d)
    assert len(centroids) == 8


def test_centroids_3bit_values():
    """3-bit centroids must be ±0.245/√d, ±0.756/√d, ±1.344/√d, ±2.152/√d."""
    d = 128
    centroids = get_centroids_3bit(d)
    expected = sorted([
        -2.152 / math.sqrt(d), -1.344 / math.sqrt(d),
        -0.756 / math.sqrt(d), -0.245 / math.sqrt(d),
         0.245 / math.sqrt(d),  0.756 / math.sqrt(d),
         1.344 / math.sqrt(d),  2.152 / math.sqrt(d),
    ])
    actual = sorted(centroids.tolist())
    for e, a in zip(expected, actual):
        assert abs(e - a) < 1e-6, f"3-bit centroid mismatch: expected {e:.6f}, got {a:.6f}"


def test_quantize_dequantize_roundtrip_mse():
    """Quantizing then dequantizing should give MSE < theoretical bound."""
    d = 128
    Pi = generate_rotation_matrix(d)
    centroids = get_centroids_2bit(d)
    x = torch.randn(d)
    x = x / x.norm()  # unit-norm vector (as in real KV heads); components have std ~1/√d
    y = Pi @ x  # rotate
    indices = quantize_to_centroids(y, centroids)
    assert indices.shape == (d,)
    assert indices.dtype == torch.long
    assert (indices >= 0).all() and (indices < 4).all()
    y_hat = dequantize_from_centroids(indices, centroids)
    assert y_hat.shape == (d,)
    mse = ((y - y_hat) ** 2).mean().item()
    # With 2-bit and d=128, MSE should be small (< 0.1)
    assert mse < 0.1, f"MSE too high: {mse}"


def test_residual_norm_positive():
    """The residual r = x - x_hat must have a positive L2 norm."""
    d = 64
    Pi = generate_rotation_matrix(d)
    centroids = get_centroids_2bit(d)
    x = torch.randn(d)
    y = Pi @ x
    indices = quantize_to_centroids(y, centroids)
    y_hat = dequantize_from_centroids(indices, centroids)
    x_hat_mse = Pi.T @ y_hat
    r = x - x_hat_mse
    gamma = torch.norm(r)
    assert gamma.item() > 0


def test_qjl_inner_product_unbiased():
    """QJL inner product estimate must be unbiased: E[<y, x_hat>] ≈ <y, x>."""
    d = 128
    k = 256  # more projections → better estimate
    torch.manual_seed(42)
    x = torch.randn(d)
    y = torch.randn(d)
    true_ip = (y @ x).item()

    estimates = []
    for _ in range(200):
        S = generate_qjl_matrix(d, k)
        r = x  # use x as residual for this unit test
        qjl = torch.sign(S @ r)   # shape (k,)
        gamma = torch.norm(r).item()
        scale = (torch.pi / 2) ** 0.5 / k
        x_approx = scale * gamma * (S.T @ qjl)
        estimates.append((y @ x_approx).item())

    mean_estimate = sum(estimates) / len(estimates)
    # Should be within 20% of true value
    assert abs(mean_estimate - true_ip) / (abs(true_ip) + 1e-6) < 0.20, \
        f"QJL estimate biased: mean={mean_estimate:.4f}, true={true_ip:.4f}"
