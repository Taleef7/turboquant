"""
Bit-packing helpers for TurboQuant compressed payloads.

Current implementation provides fast PyTorch vectorized pack/unpack paths:
  - 4-bit index packing (two values per byte)
  - 1-bit sign packing for QJL bits (eight values per byte)

Triton kernels can be added later behind the same API.
"""

from __future__ import annotations

import torch


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack int indices in [0, 15] into uint8 bytes.

    Args:
        indices: (n_rows, n_cols) integer tensor.

    Returns:
        packed: (n_rows, ceil(n_cols/2)) uint8 tensor.
    """
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got shape {indices.shape}")

    idx = indices.to(torch.int16)
    if ((idx < 0) | (idx > 15)).any():
        raise ValueError("pack_4bit expects indices in [0, 15]")

    n_rows, n_cols = idx.shape
    pad = n_cols % 2
    if pad:
        idx = torch.cat(
            [idx, torch.zeros(n_rows, 1, dtype=idx.dtype, device=idx.device)], dim=1
        )

    lo = idx[:, 0::2]
    hi = idx[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed


def unpack_4bit(packed: torch.Tensor, n_cols: int) -> torch.Tensor:
    """
    Unpack uint8 bytes into int8 indices in [0, 15].

    Args:
        packed: (n_rows, ceil(n_cols/2)) uint8 tensor.
        n_cols: Target output column count.

    Returns:
        indices: (n_rows, n_cols) int8 tensor.
    """
    if packed.ndim != 2:
        raise ValueError(f"packed must be 2D, got shape {packed.shape}")
    if n_cols <= 0:
        raise ValueError(f"n_cols must be positive, got {n_cols}")

    b = packed.to(torch.uint8)
    lo = (b & 0x0F).to(torch.int8)
    hi = ((b >> 4) & 0x0F).to(torch.int8)

    out = torch.empty(
        b.shape[0],
        b.shape[1] * 2,
        dtype=torch.int8,
        device=b.device,
    )
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out[:, :n_cols]


def pack_sign_bits(signs: torch.Tensor) -> torch.Tensor:
    """
    Pack sign values (+1/-1) into 1-bit representation.

    Bit value 1 means +1, bit value 0 means -1.

    Args:
        signs: (n_rows, n_cols) int/bool tensor with values in {-1, +1}.

    Returns:
        packed: (n_rows, ceil(n_cols/8)) uint8 tensor.
    """
    if signs.ndim != 2:
        raise ValueError(f"signs must be 2D, got shape {signs.shape}")

    s = signs.to(torch.int16)
    if ((s != 1) & (s != -1)).any():
        raise ValueError("pack_sign_bits expects values in {-1, +1}")

    bits = (s > 0).to(torch.uint8)
    n_rows, n_cols = bits.shape
    pad = (-n_cols) % 8
    if pad:
        bits = torch.cat(
            [bits, torch.zeros(n_rows, pad, dtype=bits.dtype, device=bits.device)],
            dim=1,
        )

    bits = bits.reshape(n_rows, -1, 8)
    packed = (
        bits[:, :, 0]
        | (bits[:, :, 1] << 1)
        | (bits[:, :, 2] << 2)
        | (bits[:, :, 3] << 3)
        | (bits[:, :, 4] << 4)
        | (bits[:, :, 5] << 5)
        | (bits[:, :, 6] << 6)
        | (bits[:, :, 7] << 7)
    ).to(torch.uint8)
    return packed


def unpack_sign_bits(packed: torch.Tensor, n_cols: int) -> torch.Tensor:
    """
    Unpack 1-bit signs back to int8 values (+1/-1).

    Args:
        packed: (n_rows, ceil(n_cols/8)) uint8 tensor.
        n_cols: Target output column count.

    Returns:
        signs: (n_rows, n_cols) int8 tensor in {-1, +1}.
    """
    if packed.ndim != 2:
        raise ValueError(f"packed must be 2D, got shape {packed.shape}")
    if n_cols <= 0:
        raise ValueError(f"n_cols must be positive, got {n_cols}")

    b = packed.to(torch.uint8)
    out = torch.empty(
        b.shape[0],
        b.shape[1] * 8,
        dtype=torch.int8,
        device=b.device,
    )
    for bit in range(8):
        out[:, bit::8] = torch.where(
            ((b >> bit) & 0x01) > 0,
            torch.ones_like(b, dtype=torch.int8),
            -torch.ones_like(b, dtype=torch.int8),
        )
    return out[:, :n_cols]
