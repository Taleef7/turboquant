"""Pure PyTorch, bit-exact payload packing for the reference codec."""

from __future__ import annotations

import math

import torch


_INDEX_BITS = frozenset({1, 2, 3, 4, 5, 6, 7, 8})


def _require_builtin_int(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int:  # noqa: E721
        raise TypeError(f"{name} must be a built-in int.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _require_uint8_payload(payload: torch.Tensor, name: str) -> None:
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if payload.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {tuple(payload.shape)}.")
    if payload.dtype != torch.uint8:
        raise TypeError(f"{name} must use torch.uint8, got {payload.dtype}.")


def _validate_bits(bits: object) -> int:
    bits = _require_builtin_int(bits, "bits")
    if bits not in _INDEX_BITS:
        raise ValueError(f"bits must be one of {sorted(_INDEX_BITS)}.")
    return bits


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 indices along the final dimension, least-significant bit first."""
    bits = _validate_bits(bits)
    _require_uint8_payload(indices, "indices")
    if indices.shape[-1] == 0:
        raise ValueError("indices must have a non-empty final dimension.")
    if bits < 8 and bool((indices >= (1 << bits)).any().item()):
        raise ValueError(f"indices must be in range [0, {1 << bits}).")

    count = indices.shape[-1]
    rows = indices
    shifts = torch.arange(bits, device=indices.device, dtype=torch.uint8)
    individual_bits = (rows.unsqueeze(-1) >> shifts) & 1
    bit_stream = individual_bits.reshape(rows.shape[0], count * bits)
    padded_width = math.ceil(count * bits / 8) * 8
    if padded_width != bit_stream.shape[-1]:
        padding = torch.zeros(
            (rows.shape[0], padded_width - bit_stream.shape[-1]),
            dtype=torch.uint8,
            device=indices.device,
        )
        bit_stream = torch.cat((bit_stream, padding), dim=-1)
    byte_bits = bit_stream.reshape(rows.shape[0], -1, 8)
    weights = (1 << torch.arange(8, device=indices.device, dtype=torch.uint8)).view(
        1, 1, 8
    )
    return (byte_bits * weights).sum(dim=-1, dtype=torch.uint8)


def unpack_indices(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """Unpack a final packed-byte dimension to uint8 indices."""
    bits = _validate_bits(bits)
    count = _require_builtin_int(count, "count")
    _require_uint8_payload(packed, "packed")
    expected_bytes = math.ceil(count * bits / 8)
    if packed.shape[-1] != expected_bytes:
        raise ValueError(
            f"packed final dimension must be {expected_bytes} bytes for count={count}."
        )

    rows = packed
    byte_shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bit_stream = ((rows.unsqueeze(-1) >> byte_shifts) & 1).reshape(rows.shape[0], -1)
    index_shifts = torch.arange(bits, device=packed.device, dtype=torch.uint8)
    values = bit_stream[:, : count * bits].reshape(rows.shape[0], count, bits)
    return (values << index_shifts).sum(dim=-1, dtype=torch.uint8)


def pack_sign_bits(signs: torch.Tensor) -> torch.Tensor:
    """Pack ``-1``/``+1`` signs with zero-valued padding bits."""
    if not isinstance(signs, torch.Tensor):
        raise TypeError("signs must be a torch.Tensor.")
    if signs.ndim != 2:
        raise ValueError(f"signs must be two-dimensional, got shape {tuple(signs.shape)}.")
    if signs.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("signs must use a signed integer dtype.")
    if signs.shape[-1] == 0:
        raise ValueError("signs must have a non-empty final dimension.")
    if bool(((signs != -1) & (signs != 1)).any().item()):
        raise ValueError("signs must contain only -1 and +1.")
    return _pack_one_bit((signs > 0).to(torch.uint8))


def _pack_one_bit(bits: torch.Tensor) -> torch.Tensor:
    """Pack binary values without exposing one-bit indices as a public width."""
    count = bits.shape[-1]
    rows = bits
    padded_width = math.ceil(count / 8) * 8
    if padded_width != count:
        rows = torch.cat(
            (
                rows,
                torch.zeros(
                    (rows.shape[0], padded_width - count),
                    dtype=torch.uint8,
                    device=bits.device,
                ),
            ),
            dim=-1,
        )
    byte_bits = rows.reshape(rows.shape[0], -1, 8)
    weights = (1 << torch.arange(8, device=bits.device, dtype=torch.uint8)).view(1, 1, 8)
    return (byte_bits * weights).sum(dim=-1, dtype=torch.uint8)


def unpack_sign_bits(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack one-bit signs into int8 values in ``{-1, +1}``."""
    count = _require_builtin_int(count, "count")
    _require_uint8_payload(packed, "packed")
    expected_bytes = math.ceil(count / 8)
    if packed.shape[-1] != expected_bytes:
        raise ValueError(
            f"packed final dimension must be {expected_bytes} bytes for count={count}."
        )
    rows = packed
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bit_stream = ((rows.unsqueeze(-1) >> shifts) & 1).reshape(rows.shape[0], -1)
    return torch.where(
        bit_stream[:, :count] == 1,
        torch.ones((), dtype=torch.int8, device=packed.device),
        -torch.ones((), dtype=torch.int8, device=packed.device),
    )


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return logical tensor payload bytes without Python-object overhead."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor.")
    return tensor.numel() * tensor.element_size()
