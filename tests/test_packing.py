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


class _IntSubclass(int):
    pass


class _SpoofedInt:
    def __init__(self, value: int) -> None:
        self._value = value

    @property
    def __class__(self) -> type[int]:
        return int

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        return self._value == other

    def __lt__(self, other: int) -> bool:
        return self._value < other

    def __index__(self) -> int:
        return self._value

    def __mul__(self, other: int) -> int:
        return self._value * other

    def __rmul__(self, other: int) -> int:
        return other * self._value

    def __truediv__(self, other: int) -> float:
        return self._value / other

    def __rlshift__(self, other: int) -> int:
        return other << self._value


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("width", [64, 128, 129, 256, 257])
def test_pack_indices_round_trip_and_exact_bytes(bits: int, width: int) -> None:
    indices = torch.randint(0, 1 << bits, (3, width), dtype=torch.uint8)

    packed = pack_indices(indices, bits)

    assert packed.dtype == torch.uint8
    assert packed.device == indices.device
    assert packed.shape == (3, math.ceil(width * bits / 8))
    assert tensor_nbytes(packed) == 3 * math.ceil(width * bits / 8)
    assert torch.equal(unpack_indices(packed, bits, width), indices)


def test_one_bit_indices_have_literal_bytes_and_zero_padding() -> None:
    indices = torch.tensor(
        [[1, 0, 1, 1, 0, 0, 1, 0, 1]],
        dtype=torch.uint8,
    )

    packed = pack_indices(indices, 1)

    assert torch.equal(
        packed,
        torch.tensor([[0x4D, 0x01]], dtype=torch.uint8),
    )
    assert torch.equal(unpack_indices(packed, 1, 9), indices)


@pytest.mark.parametrize("bits", [1, 3, 5, 6, 7])
def test_pack_indices_zeroes_last_byte_padding(bits: int) -> None:
    width = 129
    indices = torch.full((1, width), (1 << bits) - 1, dtype=torch.uint8)

    packed = pack_indices(indices, bits)

    used_bits = (width * bits) % 8
    assert used_bits != 0
    assert int(packed[0, -1]) >> used_bits == 0


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7])
def test_pack_indices_rejects_invalid_values(bits: int) -> None:
    invalid = torch.zeros((1, 8), dtype=torch.uint8)
    invalid[0, 0] = 1 << bits

    with pytest.raises(ValueError, match="range"):
        pack_indices(invalid, bits)


@pytest.mark.parametrize("value", [True, 2.0, "2", None, _IntSubclass(2)])
def test_pack_indices_requires_builtin_integer_width(value: object) -> None:
    with pytest.raises(TypeError, match="bits"):
        pack_indices(torch.zeros((1, 8), dtype=torch.uint8), value)  # type: ignore[arg-type]


def test_pack_indices_rejects_object_spoofing_builtin_int_bits() -> None:
    indices = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 0]], dtype=torch.uint8)

    with pytest.raises(TypeError, match="bits"):
        pack_indices(indices, _SpoofedInt(1))  # type: ignore[arg-type]


def test_unpack_indices_rejects_object_spoofing_builtin_int_bits() -> None:
    packed = torch.tensor([[0x5A]], dtype=torch.uint8)

    with pytest.raises(TypeError, match="bits"):
        unpack_indices(packed, _SpoofedInt(1), 8)  # type: ignore[arg-type]


def test_unpack_indices_rejects_object_spoofing_builtin_int_count() -> None:
    packed = torch.tensor([[0x5A]], dtype=torch.uint8)

    with pytest.raises(TypeError, match="count"):
        unpack_indices(packed, 1, _SpoofedInt(8))  # type: ignore[arg-type]


def test_sign_unpacker_rejects_object_spoofing_builtin_int_count() -> None:
    with pytest.raises(TypeError, match="count"):
        unpack_sign_bits(
            torch.tensor([[0x5A]], dtype=torch.uint8),
            _SpoofedInt(8),  # type: ignore[arg-type]
        )


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
