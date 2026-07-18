"""Numerical and byte-accounting contracts for packed reference codecs."""

from __future__ import annotations

import math

import pytest
import torch

from core.codec import KeyCodec, KeyLayout, ValueCodec, ValueLayout
from core.packing import tensor_nbytes, unpack_indices


def test_key_codec_packs_discrete_indices_and_reconstructs_requested_dtype() -> None:
    source = torch.randn((1, 2, 5, 128), dtype=torch.float16)
    codec = KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float32))

    payload = codec.encode(source)
    decoded = payload.unpack_reference()

    unpacked = unpack_indices(payload.indices, 8, 128)
    assert payload.indices.shape == (10, math.ceil(128 * 8 / 8))
    assert torch.equal(unpacked, codec.quantize_indices(source))
    assert decoded.shape == source.shape
    assert decoded.dtype == source.dtype
    assert torch.isfinite(decoded).all()
    assert payload.physical_nbytes() == tensor_nbytes(payload.indices) + tensor_nbytes(
        payload.norms
    )


def test_key_codec_zero_vectors_are_finite() -> None:
    codec = KeyCodec(KeyLayout(head_dim=128, bits=6, norm_dtype=torch.float16))

    decoded = codec.encode(torch.zeros((1, 1, 3, 128), dtype=torch.bfloat16)).unpack_reference()

    assert decoded.dtype == torch.bfloat16
    assert torch.isfinite(decoded).all()


def test_qjl_uses_lower_mse_width_and_one_bit_sign_payload() -> None:
    tokens = 5
    heads = 2
    qjl_dim = 19
    codec = KeyCodec(
        KeyLayout(head_dim=128, bits=6, norm_dtype=torch.float16, qjl_enabled=True),
        qjl_dim=qjl_dim,
    )

    payload = codec.encode(torch.randn((1, heads, tokens, 128), dtype=torch.float16))

    assert payload.qjl_signs is not None
    assert payload.qjl_gamma is not None
    assert payload.indices.shape[-1] == math.ceil(128 * (6 - 1) / 8)
    assert tensor_nbytes(payload.qjl_signs) == heads * math.ceil(tokens * qjl_dim / 8)
    assert payload.physical_nbytes() == sum(
        tensor_nbytes(tensor)
        for tensor in (payload.indices, payload.norms, payload.qjl_signs, payload.qjl_gamma)
    )


def test_value_codec_group_metadata_and_exact_reference_decode() -> None:
    source = torch.linspace(-2, 2, 1 * 2 * 3 * 64, dtype=torch.float32).reshape(1, 2, 3, 64).half()
    codec = ValueCodec(ValueLayout(head_dim=64, bits=6, group_size=16))

    payload = codec.encode(source)
    decoded = payload.unpack_reference()

    assert payload.mins.shape == (6, 4)
    assert payload.scales.shape == (6, 4)
    assert decoded.shape == source.shape
    assert decoded.dtype == source.dtype
    assert torch.isfinite(decoded).all()
    assert payload.physical_nbytes() == sum(
        tensor_nbytes(tensor) for tensor in (payload.indices, payload.mins, payload.scales)
    )


@pytest.mark.parametrize("bits", [True, 1, 9])
def test_layout_rejects_non_integer_or_unsupported_width(bits: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        KeyLayout(head_dim=128, bits=bits, norm_dtype=torch.float16)  # type: ignore[arg-type]
