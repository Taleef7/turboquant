"""
TurboQuant Simple: Minimal MSE-only implementation.

Based on tonbistudio/turboquant-pytorch reference.
Key insight: MUST normalize to unit length before rotation.

Pipeline:
  COMPRESS:   x → ||x|| (store) → x/||x|| → rotate → quantize → indices
  DECOMPRESS: indices → centroids → unrotate → scale by ||x|| → x_hat
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from utils.math_utils import get_codebook_tensors


def generate_rotation_matrix(
    d: int, seed: int = 42, device: str = "cpu"
) -> torch.Tensor:
    """Generate random orthogonal rotation matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs.unsqueeze(0)
    return Q.to(device)


class TurboQuantMSE(nn.Module):
    """
    MSE-optimal quantizer: rotation + per-coordinate Lloyd-Max quantization.

    CRITICAL: Input vectors are normalized to unit length before rotation.
    This is required because Lloyd-Max centroids are computed for the Beta
    distribution on [-1, 1] arising from random rotation of unit vectors.
    """

    def __init__(
        self, head_dim: int, bits: int = 4, seed: int = 42, device: str = "cuda"
    ):
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.n_levels = 2**bits
        self.device = device

        # Rotation matrix
        self.register_buffer("Pi", generate_rotation_matrix(head_dim, seed, device))

        # Lloyd-Max codebook (precomputed for Beta distribution)
        centroids, boundaries = get_codebook_tensors(
            head_dim, bits, torch.device(device)
        )
        self.register_buffer("centroids", centroids)
        self.register_buffer("boundaries", boundaries)

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compress vectors to indices + stored norms.

        Args:
            x: (batch, head_dim) or (batch, heads, seq, head_dim)

        Returns:
            dict with 'indices' (uint8) and 'norms' (float16)
        """
        orig_shape = x.shape
        flat = x.reshape(-1, self.head_dim).float()

        # 1. Extract and store norm
        norms = torch.norm(flat, dim=-1, keepdim=True)

        # 2. Normalize to unit length (CRITICAL!)
        x_unit = flat / (norms + 1e-8)

        # 3. Rotate
        rotated = x_unit @ self.Pi.T

        # 4. Lloyd-Max quantize (searchsorted for efficiency)
        # boundaries[1:-1] are decision boundaries between centroids
        indices = torch.searchsorted(
            self.boundaries[1:-1].contiguous(), rotated.contiguous()
        )

        return {
            "indices": indices.to(torch.uint8),
            "norms": norms.squeeze(-1).to(torch.float16),
            "shape": orig_shape,
        }

    @torch.no_grad()
    def decompress(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decompress indices + norms back to vectors.

        Returns:
            Tensor with original shape
        """
        indices = compressed["indices"].long()
        norms = compressed["norms"].float().unsqueeze(-1)
        orig_shape = compressed["shape"]

        # 1. Lookup centroids
        values = self.centroids[indices]

        # 2. Unrotate (Pi is orthogonal, so Pi^T = Pi^{-1})
        unrotated = values @ self.Pi

        # 3. Rescale by stored norm
        result = unrotated * norms

        return result.reshape(orig_shape)

    @torch.no_grad()
    def compress_decompress(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Full round-trip for testing."""
        compressed = self.compress(x)
        reconstructed = self.decompress(compressed)
        return reconstructed, compressed


class TurboQuantValueMSE(nn.Module):
    """
    Simpler MSE quantizer for values (no rotation needed per tonbistudio).
    Uses per-group min/max quantization for values.
    """

    def __init__(
        self, head_dim: int, bits: int = 4, group_size: int = 32, device: str = "cuda"
    ):
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.n_levels = 2**bits
        self.group_size = group_size
        self.device = device

        if head_dim % group_size != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by group_size ({group_size})"
            )

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress values with per-group min/max quantization."""
        orig_shape = x.shape
        # Avoid unnecessary float32 conversion if already float16
        flat = x.reshape(-1, self.head_dim)
        n_vecs = flat.shape[0]
        n_groups = self.head_dim // self.group_size

        # Reshape to groups - use view to avoid copy
        grouped = flat.view(n_vecs, n_groups, self.group_size)

        # Per-group min/max (stays in original dtype)
        mins = grouped.amin(dim=-1)
        maxs = grouped.amax(dim=-1)
        scales = (maxs - mins) / (self.n_levels - 1)
        scales = torch.clamp(scales, min=1e-8)

        # Quantize in-place operations where possible
        normalized = (grouped - mins.unsqueeze(-1)) / scales.unsqueeze(-1)
        indices = torch.round(normalized).clamp_(0, self.n_levels - 1).to(torch.uint8)

        return {
            "indices": indices.view(n_vecs, self.head_dim),
            "mins": mins.half(),
            "scales": scales.half(),
            "shape": orig_shape,
        }

    @torch.no_grad()
    def decompress(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decompress values."""
        indices = compressed["indices"]
        mins = compressed["mins"]
        scales = compressed["scales"]
        orig_shape = compressed["shape"]

        n_vecs = indices.shape[0]
        n_groups = self.head_dim // self.group_size

        # Reshape and dequantize (stay in float16 if mins/scales are float16)
        grouped = indices.view(n_vecs, n_groups, self.group_size).to(mins.dtype)
        result = mins.unsqueeze(-1) + grouped * scales.unsqueeze(-1)

        return result.view(orig_shape)
