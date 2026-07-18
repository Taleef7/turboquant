"""Bit-exact contracts for the CPU reference packers."""

from __future__ import annotations

import math

import pytest
import torch

from core.packing import (
    pack_indices,
    pack_sign_bits,
    tensor_nbytes,
    unpack_indices,
    unpack_sign_bits,
)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("width", [64, 128, 129, 256, 257])
def test_pack_indices_round_trip_and_exact_bytes(bits: int, width: int) -> None:
    indices = torch.randint(0, 1 << bits, (3, width), dtype=torch.uint8)

    packed = pack_indices(indices, bits)

    assert packed.dtype == torch.uint8
    assert packed.device == indices.device
    assert packed.shape == (3, math.ceil(width * bits / 8))
    assert tensor_nbytes(packed) == 3 * math.ceil(width * bits / 8)
    assert torch.equal(unpack_indices(packed, bits, width), indices)


@pytest.mark.parametrize("bits", [3, 5, 6, 7])
def test_pack_indices_zeroes_last_byte_padding(bits: int) -> None:
    width = 129
    indices = torch.full((1, width), (1 << bits) - 1, dtype=torch.uint8)

    packed = pack_indices(indices, bits)

    used_bits = (width * bits) % 8
    assert used_bits != 0
    assert int(packed[0, -1]) >> used_bits == 0


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7])
def test_pack_indices_rejects_invalid_values(bits: int) -> None:
    invalid = torch.zeros((1, 8), dtype=torch.uint8)
    invalid[0, 0] = 1 << bits

    with pytest.raises(ValueError, match="range"):
        pack_indices(invalid, bits)


@pytest.mark.parametrize("value", [True, 2.0, "2", None])
def test_pack_indices_requires_builtin_integer_width(value: object) -> None:
    with pytest.raises(TypeError, match="bits"):
        pack_indices(torch.zeros((1, 8), dtype=torch.uint8), value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "indices",
    [
        torch.zeros((8,), dtype=torch.uint8),
        torch.zeros((1, 8), dtype=torch.int64),
        torch.zeros((1, 8), dtype=torch.float32),
    ],
)
def test_pack_indices_rejects_invalid_dtype_or_shape(indices: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        pack_indices(indices, 4)


def test_sign_packing_round_trip_and_padding() -> None:
    signs = torch.tensor([[1, -1, 1, 1, -1, -1, 1, -1, 1]], dtype=torch.int8)

    packed = pack_sign_bits(signs)

    assert packed.dtype == torch.uint8
    assert packed.shape == (1, 2)
    assert int(packed[0, -1]) >> 1 == 0
    assert torch.equal(unpack_sign_bits(packed, 9), signs)


def test_unpack_rejects_wrong_payload_shape_and_width() -> None:
    with pytest.raises((TypeError, ValueError)):
        unpack_indices(torch.zeros((2, 1), dtype=torch.uint8), 4, 8)
    with pytest.raises((TypeError, ValueError)):
        unpack_sign_bits(torch.zeros((1, 1), dtype=torch.uint8), True)  # type: ignore[arg-type]
