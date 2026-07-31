"""Deterministic CPU reference rotations derived from the TurboQuant paper."""

import torch


def seeded_haar_rotation(
    dimension: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a seeded Gaussian-QR rotation with a positive R diagonal.

    Exact repeatability is scoped to the pinned runtime and platform. PyTorch
    does not guarantee bit-identical random or linear-algebra results across
    releases, platforms, or CPU/GPU execution.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn(
        (dimension, dimension),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    q, r = torch.linalg.qr(gaussian)
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), 1)
    return (q * signs).to(dtype=dtype)
