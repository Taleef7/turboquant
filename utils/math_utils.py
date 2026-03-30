"""
TurboQuant mathematical primitives.

Provides:
  - generate_rotation_matrix(d): random orthogonal matrix Π via QR decomposition
  - generate_qjl_matrix(d, k): random Gaussian matrix S for QJL transform
  - get_centroids_2bit(d): optimal Lloyd-Max centroids for 2-bit (4-level) quantization
  - get_centroids_3bit(d): optimal Lloyd-Max centroids for 3-bit (8-level) quantization
  - quantize_to_centroids(y, centroids): nearest-centroid assignment
  - dequantize_from_centroids(indices, centroids): index → centroid value

References: TurboQuant paper (arXiv:2504.19874), Section 3.
"""

import math
import torch


def generate_rotation_matrix(d: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """
    Generate a random orthogonal rotation matrix Π ∈ R^{d×d}.

    Uses QR decomposition of a standard normal matrix. The resulting Q
    is Haar-distributed (uniformly random orthogonal matrix).

    Args:
        d: Dimension of the square matrix.

    Returns:
        Tensor of shape (d, d), orthogonal: Π^T Π = I.
    """
    A = torch.randn(d, d, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    # Fix sign ambiguity so distribution is truly Haar-uniform
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)
    return Q


def generate_qjl_matrix(d: int, k: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """
    Generate the QJL projection matrix S ∈ R^{k×d} with i.i.d. N(0,1) entries.

    Args:
        d: Input dimension.
        k: Number of projections (output dimension).

    Returns:
        Tensor of shape (k, d).
    """
    return torch.randn(k, d, dtype=dtype, device=device)


def get_centroids_2bit(d: int, dtype=torch.float32) -> torch.Tensor:
    """
    Optimal 4-level (2-bit) Lloyd-Max centroids for Beta-distributed coordinates
    after random rotation in dimension d.

    For high d, each rotated coordinate follows Beta(d/2-1/2, d/2-1/2) ≈ N(0, 1/d).
    The theoretical optimal centroids are: ±0.453/√d and ±1.510/√d.

    Args:
        d: Head dimension of the key/value vectors.

    Returns:
        Tensor of shape (4,) sorted in ascending order.
    """
    scale = 1.0 / math.sqrt(d)
    return torch.tensor(
        [-1.510 * scale, -0.453 * scale, 0.453 * scale, 1.510 * scale],
        dtype=dtype,
    )


def get_centroids_3bit(d: int, dtype=torch.float32) -> torch.Tensor:
    """
    Optimal 8-level (3-bit) Lloyd-Max centroids for Beta-distributed coordinates
    after random rotation in dimension d.

    Centroids derived from Lloyd-Max optimal quantizer tables for Gaussian N(0, 1/d).
    Values are scaled by 1/√d from the standard N(0,1) 8-level table:
    {±0.245, ±0.756, ±1.344, ±2.152}.

    Args:
        d: Head dimension.

    Returns:
        Tensor of shape (8,) sorted in ascending order.
    """
    scale = 1.0 / math.sqrt(d)
    return torch.tensor(
        [
            -2.152 * scale, -1.344 * scale, -0.756 * scale, -0.245 * scale,
             0.245 * scale,  0.756 * scale,  1.344 * scale,  2.152 * scale,
        ],
        dtype=dtype,
    )


def quantize_to_centroids(y: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Assign each element of y to its nearest centroid index.

    Args:
        y: Tensor of shape (..., d) — rotated coordinates.
        centroids: Tensor of shape (n_levels,) — sorted centroid values.

    Returns:
        Integer tensor of shape (..., d) with values in [0, n_levels).
    """
    # Compute distance from each element to each centroid: (..., d, n_levels)
    diff = y.unsqueeze(-1) - centroids  # broadcast
    indices = diff.abs().argmin(dim=-1)
    return indices


def dequantize_from_centroids(indices: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct quantized values from centroid indices.

    Args:
        indices: Integer tensor of shape (..., d).
        centroids: Tensor of shape (n_levels,).

    Returns:
        Float tensor of shape (..., d) — reconstructed values.
    """
    return centroids[indices]
