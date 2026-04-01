"""
TurboQuant mathematical primitives.

Provides:
  - generate_rotation_matrix(d, seed): random orthogonal matrix Π via QR decomposition
  - generate_qjl_matrix(d, k, seed): random Gaussian matrix S for QJL transform
  - get_codebook(d, bits): optimal Lloyd-Max codebook for Beta distribution
  - quantize_to_centroids(y, centroids): nearest-centroid or searchsorted assignment
  - dequantize_from_centroids(indices, centroids): index → centroid value

References: TurboQuant paper (arXiv:2504.19874), Section 3.
Reference implementation: https://github.com/0xSero/turboquant
"""

import os
import json
import math
import torch
import numpy as np
from scipy import integrate, special

# ──────────────────────────────────────────────────────────────────────────────
# Random Matrix Generation (per-layer seeding)
# ──────────────────────────────────────────────────────────────────────────────


def generate_rotation_matrix(
    d: int,
    dtype=torch.float32,
    device="cpu",
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a random orthogonal rotation matrix Π ∈ R^{d×d}.

    Uses QR decomposition of a standard normal matrix with explicit CPU generator
    for reproducibility across devices (per reference implementation).

    Args:
        d: Dimension of the square matrix.
        dtype: Output dtype.
        device: Output device.
        seed: Random seed for reproducibility (use different seeds per layer).

    Returns:
        Tensor of shape (d, d), orthogonal: Π^T Π = I.
    """
    # Use CPU generator for reproducibility (reference implementation pattern)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # Generate on CPU, then move to device
    A = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)

    # Fix sign ambiguity so distribution is truly Haar-uniform
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)  # scales each column j by sign(R[j,j])

    return Q.to(device=device, dtype=dtype)


def generate_qjl_matrix(
    d: int,
    k: int,
    dtype=torch.float32,
    device="cpu",
    seed: int = 12345,
) -> torch.Tensor:
    """
    Generate the QJL projection matrix S ∈ R^{k×d} with i.i.d. N(0,1) entries.

    Args:
        d: Input dimension.
        k: Number of projections (output dimension).
        dtype: Output dtype.
        device: Output device.
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (k, d).
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(k, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Lloyd-Max Codebook Computation
# ──────────────────────────────────────────────────────────────────────────────


def _beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """
    PDF of a single coordinate of a uniform random point on S^{d-1}.

    After random rotation, each coordinate of a unit-norm vector follows:
        f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    This is a scaled Beta distribution on [-1, 1].
    """
    if d <= 2:
        raise ValueError(f"Dimension d={d} too small, need d >= 3")

    log_const = (
        special.gammaln(d / 2.0) - 0.5 * np.log(np.pi) - special.gammaln((d - 1) / 2.0)
    )
    exponent = (d - 3) / 2.0

    # Clip x to avoid numerical issues at boundaries
    x = np.clip(x, -1 + 1e-15, 1 - 1e-15)
    log_val = log_const + exponent * np.log(1 - x**2)
    return np.exp(log_val)


def _conditional_mean(lo: float, hi: float, d: int) -> float:
    """E[X | lo < X < hi] under the Beta PDF on [-1, 1]."""
    num, _ = integrate.quad(lambda x: x * _beta_pdf(np.array([x]), d)[0], lo, hi)
    den, _ = integrate.quad(lambda x: _beta_pdf(np.array([x]), d)[0], lo, hi)
    if den < 1e-30:
        return (lo + hi) / 2.0
    return num / den


def _mse_cost(centroids: np.ndarray, d: int) -> float:
    """Compute MSE cost for a given set of sorted centroids."""
    n = len(centroids)
    boundaries = np.zeros(n + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    cost = 0.0
    for i in range(n):
        lo, hi = boundaries[i], boundaries[i + 1]
        c = centroids[i]
        val, _ = integrate.quad(
            lambda x: (x - c) ** 2 * _beta_pdf(np.array([x]), d)[0], lo, hi
        )
        cost += val
    return cost


def compute_lloyd_max_codebook(
    d: int, bits: int, max_iter: int = 200, tol: float = 1e-12
) -> dict:
    """
    Compute optimal Lloyd-Max codebook for the Beta distribution on [-1, 1]
    arising from random rotation of d-dimensional unit vectors.

    This implements the core algorithm from the TurboQuant paper:
    - The distribution is the marginal of a coordinate after random rotation
    - For high d, converges to N(0, 1/d)
    - Lloyd-Max finds optimal scalar quantizers by alternating:
      1. Update boundaries as midpoints between centroids
      2. Update centroids as conditional means in each region

    Args:
        d: dimension of the embedding space (e.g., head_dim = 128)
        bits: number of bits per coordinate (1, 2, 3, or 4)
        max_iter: max Lloyd-Max iterations
        tol: convergence tolerance

    Returns:
        dict with keys:
            'centroids': sorted list of 2^bits centroids
            'boundaries': sorted list of 2^bits + 1 boundaries (includes -1 and 1)
            'mse_per_coord': achieved MSE cost per coordinate
            'd': dimension
            'bits': bit-width
    """
    n_clusters = 2**bits

    # Initialize centroids using quantiles of the distribution
    x_grid = np.linspace(-1 + 1e-10, 1 - 1e-10, 10000)
    pdf_vals = _beta_pdf(x_grid, d)
    cdf_vals = np.cumsum(pdf_vals) * (x_grid[1] - x_grid[0])
    cdf_vals /= cdf_vals[-1]

    # Place initial centroids at quantile midpoints
    quantile_edges = np.linspace(0, 1, n_clusters + 1)
    centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        q_lo = quantile_edges[i]
        q_hi = quantile_edges[i + 1]
        q_mid = (q_lo + q_hi) / 2.0
        idx = np.searchsorted(cdf_vals, q_mid)
        idx = min(idx, len(x_grid) - 1)
        centroids[i] = x_grid[idx]

    # Lloyd-Max iterations
    prev_cost = float("inf")
    for _iteration in range(max_iter):
        # Compute boundaries (midpoints between consecutive centroids)
        boundaries = np.zeros(n_clusters + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_clusters - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

        # Update centroids as conditional means
        new_centroids = np.zeros(n_clusters)
        for i in range(n_clusters):
            new_centroids[i] = _conditional_mean(boundaries[i], boundaries[i + 1], d)

        cost = _mse_cost(new_centroids, d)
        centroids = new_centroids

        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost

    # Recompute final boundaries
    boundaries = np.zeros(n_clusters + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n_clusters - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    return {
        "centroids": centroids.tolist(),
        "boundaries": boundaries.tolist(),
        "mse_per_coord": float(cost),
        "d": d,
        "bits": bits,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Codebook Cache (compute once, reuse)
# ──────────────────────────────────────────────────────────────────────────────

_CODEBOOK_CACHE: dict[tuple[int, int], dict] = {}
_CODEBOOK_DIR = os.path.join(os.path.dirname(__file__), "..", "codebooks")


def get_codebook(d: int, bits: int) -> dict:
    """
    Get or compute a Lloyd-Max codebook, with on-disk caching.

    First call for a (d, bits) pair will compute the codebook (~1-2 seconds)
    and save it to disk. Subsequent calls load from cache.

    Args:
        d: Head dimension (e.g., 128)
        bits: Quantization bits (2, 3, or 4)

    Returns:
        dict with 'centroids', 'boundaries', 'mse_per_coord', 'd', 'bits'
    """
    key = (d, bits)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]

    # Try loading from disk
    os.makedirs(_CODEBOOK_DIR, exist_ok=True)
    path = os.path.join(_CODEBOOK_DIR, f"codebook_d{d}_b{bits}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            cb = json.load(f)
        _CODEBOOK_CACHE[key] = cb
        return cb

    # Compute and save
    print(f"[TurboQuant] Computing Lloyd-Max codebook for d={d}, bits={bits}...")
    cb = compute_lloyd_max_codebook(d, bits)
    with open(path, "w") as f:
        json.dump(cb, f, indent=2)
    print(f"[TurboQuant] MSE per coord = {cb['mse_per_coord']:.6e}")
    _CODEBOOK_CACHE[key] = cb
    return cb


def get_codebook_tensors(
    d: int,
    bits: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get codebook as GPU tensors ready for quantization.

    Args:
        d: Head dimension
        bits: Quantization bits
        device: Target device
        dtype: Target dtype

    Returns:
        (centroids, boundaries) as tensors
    """
    cb = get_codebook(d, bits)
    centroids = torch.tensor(cb["centroids"], device=device, dtype=dtype)
    boundaries = torch.tensor(cb["boundaries"], device=device, dtype=dtype)
    return centroids, boundaries


# ──────────────────────────────────────────────────────────────────────────────
# Legacy API (for backward compatibility with existing code)
# These now use the proper Lloyd-Max codebooks
# ──────────────────────────────────────────────────────────────────────────────


def get_centroids_2bit(d: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """
    Get optimal 4-level (2-bit) Lloyd-Max centroids.

    This now uses the proper Beta distribution codebook instead of
    hardcoded N(0,1) approximations.
    """
    centroids, _ = get_codebook_tensors(d, bits=2, device=device, dtype=dtype)
    return centroids


def get_centroids_3bit(d: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """
    Get optimal 8-level (3-bit) Lloyd-Max centroids.

    This now uses the proper Beta distribution codebook instead of
    hardcoded N(0,1) approximations.
    """
    centroids, _ = get_codebook_tensors(d, bits=3, device=device, dtype=dtype)
    return centroids


def get_centroids_4bit(d: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """
    Get optimal 16-level (4-bit) Lloyd-Max centroids.

    This now uses the proper Beta distribution codebook instead of
    hardcoded N(0,1) approximations.
    """
    centroids, _ = get_codebook_tensors(d, bits=4, device=device, dtype=dtype)
    return centroids


# ──────────────────────────────────────────────────────────────────────────────
# Quantization / Dequantization
# ──────────────────────────────────────────────────────────────────────────────


def quantize_to_centroids(
    y: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor = None,
) -> torch.Tensor:
    """
    Assign each element of y to its nearest centroid index.

    Uses searchsorted on decision boundaries (more efficient) if boundaries
    are provided, otherwise falls back to argmin distance.

    Args:
        y: Tensor of shape (..., d) — rotated coordinates.
        centroids: Tensor of shape (n_levels,) — sorted centroid values.
        boundaries: Tensor of shape (n_levels+1,) — decision boundaries.
                   If None, computed as midpoints between centroids.

    Returns:
        Integer tensor of shape (..., d) with values in [0, n_levels).
    """
    if boundaries is None:
        # Compute boundaries as midpoints (for backward compatibility)
        n = centroids.shape[0]
        boundaries = torch.zeros(n + 1, device=centroids.device, dtype=centroids.dtype)
        boundaries[0] = -float("inf")
        boundaries[-1] = float("inf")
        for i in range(n - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    # Use searchsorted for efficient quantization (reference implementation)
    # boundaries[1:-1] are the decision boundaries between centroids
    decision_boundaries = boundaries[1:-1]  # shape: (n_levels - 1,)

    # searchsorted returns index where y would be inserted to maintain order
    # This gives us the centroid index directly
    indices = torch.searchsorted(decision_boundaries.contiguous(), y.contiguous())

    return indices


def dequantize_from_centroids(
    indices: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct quantized values from centroid indices.

    Args:
        indices: Integer tensor of shape (..., d).
        centroids: Tensor of shape (n_levels,).

    Returns:
        Float tensor of shape (..., d) — reconstructed values.
    """
    return centroids[indices]
