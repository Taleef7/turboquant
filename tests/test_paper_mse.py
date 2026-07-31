"""Behavioral contracts for the paper-faithful CPU MSE reference codec."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import itertools
import math

import pytest
import torch

from core.packing import unpack_indices
from core.paper_codebook import lloyd_max_sphere_codebook
from core.paper_mse import (
    TurboQuantMSEConfig,
    TurboQuantMSEEncoded,
    TurboQuantMSEReference,
)
from core.paper_rotation import seeded_haar_rotation


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

    def __le__(self, other: int) -> bool:
        return self._value <= other

    def __ge__(self, other: int) -> bool:
        return self._value >= other


@lru_cache(maxsize=None)
def _codec(
    dimension: int,
    bits: int,
    rotation_seed: int = 1729,
) -> TurboQuantMSEReference:
    return TurboQuantMSEReference(
        TurboQuantMSEConfig(
            dimension=dimension,
            bits=bits,
            rotation_seed=rotation_seed,
        )
    )


@pytest.mark.parametrize(
    ("dimension", "bits", "single_index_bytes", "single_total", "batch_total"),
    [
        (64, 1, 8, 12, 36),
        (64, 2, 16, 20, 60),
        (64, 3, 24, 28, 84),
        (64, 4, 32, 36, 108),
        (128, 1, 16, 20, 60),
        (128, 2, 32, 36, 108),
        (128, 3, 48, 52, 156),
        (128, 4, 64, 68, 204),
        (256, 1, 32, 36, 108),
        (256, 2, 64, 68, 204),
        (256, 3, 96, 100, 300),
        (256, 4, 128, 132, 396),
    ],
)
def test_payload_shapes_and_hand_derived_physical_bytes(
    dimension: int,
    bits: int,
    single_index_bytes: int,
    single_total: int,
    batch_total: int,
) -> None:
    codec = _codec(dimension, bits)
    vector = torch.linspace(-1.0, 1.0, dimension, dtype=torch.float64)

    single = codec.encode(vector)
    batch = codec.encode(torch.stack((vector, -vector, vector / 2)))

    assert single.indices.shape == (single_index_bytes,)
    assert single.norm.shape == ()
    assert single.norm.dtype is torch.float32
    assert codec.physical_nbytes(single) == single_total
    assert batch.indices.shape == (3, single_index_bytes)
    assert batch.norm.shape == (3,)
    assert batch.norm.dtype is torch.float32
    assert codec.physical_nbytes(batch) == batch_total
    assert codec.decode(single).shape == (dimension,)
    assert codec.decode(batch).shape == (3, dimension)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_encode_accepts_float32_and_float64_and_stores_float32_norm(
    dtype: torch.dtype,
) -> None:
    vector = torch.linspace(-0.75, 1.25, 64, dtype=dtype)
    expected_norm = torch.linalg.vector_norm(vector.to(torch.float64)).to(torch.float32)

    encoded = _codec(64, 3).encode(vector)
    decoded = _codec(64, 3).decode(encoded)

    assert encoded.norm.dtype is torch.float32
    assert torch.equal(encoded.norm, expected_norm)
    assert decoded.device.type == "cpu"
    assert decoded.dtype is torch.float64


@pytest.mark.parametrize(
    "vector",
    [
        torch.full((64,), 1e39, dtype=torch.float64),
        torch.full(
            (64,),
            torch.finfo(torch.float32).max,
            dtype=torch.float32,
        ),
    ],
)
def test_encode_rejects_norms_that_overflow_float32_storage(
    vector: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="float32"):
        _codec(64, 2).encode(vector)


def test_encode_rejects_nonfinite_float64_norm_from_finite_values() -> None:
    vector = torch.full((64,), 1e308, dtype=torch.float64)
    assert torch.isfinite(vector).all()

    with pytest.raises(ValueError, match="finite norm"):
        _codec(64, 2).encode(vector)


def test_tiny_finite_norm_may_underflow_in_declared_float32_storage() -> None:
    vector = torch.full((64,), 1e-50, dtype=torch.float64)

    encoded = _codec(64, 2).encode(vector)

    assert encoded.norm.item() == 0.0
    assert torch.equal(
        _codec(64, 2).decode(encoded),
        torch.zeros(64, dtype=torch.float64),
    )


def test_zero_vectors_have_deterministic_ties_and_decode_to_exact_zero() -> None:
    for bits in range(1, 5):
        codec = _codec(64, bits)
        single = codec.encode(torch.zeros(64, dtype=torch.float64))
        batch = codec.encode(torch.zeros((3, 64), dtype=torch.float32))
        expected_index = (1 << (bits - 1)) - 1

        unpacked_single = unpack_indices(
            single.indices.unsqueeze(0),
            bits,
            64,
        )[0]
        unpacked_batch = unpack_indices(batch.indices, bits, 64)

        assert torch.all(unpacked_single == expected_index)
        assert torch.all(unpacked_batch == expected_index)
        assert single.norm.item() == 0.0
        assert torch.equal(batch.norm, torch.zeros(3, dtype=torch.float32))
        assert torch.equal(
            codec.decode(single),
            torch.zeros(64, dtype=torch.float64),
        )
        assert torch.equal(
            codec.decode(batch),
            torch.zeros((3, 64), dtype=torch.float64),
        )


def test_row_rotation_nearest_centroid_and_inverse_match_hand_oracle() -> None:
    dimension = 64
    bits = 2
    seed = 23
    rotation = seeded_haar_rotation(dimension, seed)
    rotated_unit = torch.full(
        (dimension,),
        math.sqrt((1.0 - 0.8**2) / (dimension - 1)),
        dtype=torch.float64,
    )
    rotated_unit[0] = 0.8
    vector = 2.5 * (rotated_unit @ rotation)
    expected_indices = torch.full((dimension,), 2, dtype=torch.uint8)
    expected_indices[0] = 3

    codec = _codec(dimension, bits, seed)
    encoded = codec.encode(vector)
    unpacked = unpack_indices(encoded.indices.unsqueeze(0), bits, dimension)[0]
    codebook = lloyd_max_sphere_codebook(dimension, bits)
    expected_decoded = (
        codebook[expected_indices.to(torch.int64)] @ rotation
    ) * 2.5

    assert torch.equal(unpacked, expected_indices)
    torch.testing.assert_close(
        codec.decode(encoded),
        expected_decoded,
        atol=1e-14,
        rtol=0,
    )


def test_encoding_is_bit_exact_across_repeated_and_fresh_instances() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(811)
    vectors = torch.randn((7, 128), generator=generator, dtype=torch.float64)
    config = TurboQuantMSEConfig(128, 4, 909)
    first_codec = TurboQuantMSEReference(config)
    second_codec = TurboQuantMSEReference(config)

    first = first_codec.encode(vectors)
    repeated = first_codec.encode(vectors.clone())
    fresh = second_codec.encode(vectors)

    assert torch.equal(first.indices, repeated.indices)
    assert torch.equal(first.indices, fresh.indices)
    assert torch.equal(first.norm, repeated.norm)
    assert torch.equal(first.norm, fresh.norm)


@pytest.mark.parametrize("dimension", [64, 128, 256])
def test_fixed_seed_sphere_mse_decreases_robustly_with_bits(
    dimension: int,
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1000 + dimension)
    vectors = torch.randn(
        (1024, dimension),
        generator=generator,
        dtype=torch.float64,
    )
    vectors /= torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    distortions: list[float] = []

    for bits in range(1, 5):
        codec = _codec(dimension, bits)
        decoded = codec.decode(codec.encode(vectors))
        distortions.append(
            (vectors - decoded).square().sum(dim=-1).mean().item()
        )

    assert all(
        finer < 0.75 * coarser
        for coarser, finer in itertools.pairwise(distortions)
    )


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("dimension", True, TypeError),
        ("dimension", 1, ValueError),
        ("dimension", 64.0, TypeError),
        ("bits", True, TypeError),
        ("bits", 0, ValueError),
        ("bits", 9, ValueError),
        ("bits", 2.0, TypeError),
        ("rotation_seed", True, TypeError),
        ("rotation_seed", 1.5, TypeError),
    ],
)
def test_config_rejects_values_outside_physical_packer_contract(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "dimension": 64,
        "bits": 2,
        "rotation_seed": 17,
    }
    arguments[field] = value

    with pytest.raises(error_type, match=field):
        TurboQuantMSEConfig(**arguments)  # type: ignore[arg-type]


def test_config_accepts_full_uint8_index_width() -> None:
    assert TurboQuantMSEConfig(64, 8, -1).bits == 8


@pytest.mark.parametrize("field", ["dimension", "bits", "rotation_seed"])
def test_config_rejects_integer_subclasses(field: str) -> None:
    arguments: dict[str, object] = {
        "dimension": 64,
        "bits": 2,
        "rotation_seed": 17,
    }
    arguments[field] = _IntSubclass(arguments[field])

    with pytest.raises(TypeError, match=field):
        TurboQuantMSEConfig(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dimension", 64),
        ("bits", 2),
        ("rotation_seed", 17),
    ],
)
def test_config_rejects_objects_spoofing_builtin_int_class(
    field: str,
    value: int,
) -> None:
    arguments: dict[str, object] = {
        "dimension": 64,
        "bits": 2,
        "rotation_seed": 17,
    }
    arguments[field] = _SpoofedInt(value)

    with pytest.raises(TypeError, match=field):
        TurboQuantMSEConfig(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "seed",
    [
        -0x8000_0000_0000_0000,
        -1,
        0xFFFF_FFFF_FFFF_FFFF,
    ],
)
def test_config_and_reference_accept_documented_seed_boundaries(
    seed: int,
) -> None:
    config = TurboQuantMSEConfig(2, 1, seed)

    reference = TurboQuantMSEReference(config)

    assert reference.config.rotation_seed == seed


@pytest.mark.parametrize(
    "seed",
    [
        -0x8000_0000_0000_0001,
        0x1_0000_0000_0000_0000,
    ],
)
def test_config_rejects_seeds_outside_documented_generator_range(
    seed: int,
) -> None:
    with pytest.raises(ValueError, match="rotation_seed"):
        TurboQuantMSEConfig(64, 2, seed)


def test_reference_requires_exact_config_type() -> None:
    with pytest.raises(TypeError, match="config"):
        TurboQuantMSEReference((64, 2, 17))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "vectors",
    [
        [0.0] * 64,
        torch.zeros((), dtype=torch.float32),
        torch.zeros((1, 1, 64), dtype=torch.float32),
        torch.zeros((0, 64), dtype=torch.float32),
        torch.zeros(63, dtype=torch.float32),
        torch.zeros((2, 65), dtype=torch.float64),
        torch.zeros(64, dtype=torch.float16),
        torch.zeros(64, dtype=torch.int64),
        torch.full((64,), math.nan, dtype=torch.float32),
        torch.full((1, 64), math.inf, dtype=torch.float64),
    ],
)
def test_encode_rejects_malformed_nonfinite_or_unsupported_inputs(
    vectors: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="vectors"):
        _codec(64, 2).encode(vectors)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_encode_rejects_cuda_input() -> None:
    with pytest.raises(ValueError, match="CPU"):
        _codec(64, 2).encode(torch.zeros(64, device="cuda", dtype=torch.float32))


@pytest.mark.parametrize("method_name", ["decode", "physical_nbytes"])
def test_payload_consumers_reject_wrong_type_and_cross_config(
    method_name: str,
) -> None:
    codec = _codec(64, 2)
    encoded = codec.encode(torch.ones(64, dtype=torch.float64))
    method = getattr(codec, method_name)

    with pytest.raises(TypeError, match="encoded"):
        method({"indices": encoded.indices, "norm": encoded.norm})

    mismatched = replace(
        encoded,
        config=TurboQuantMSEConfig(64, 2, 1730),
    )
    with pytest.raises(ValueError, match="config"):
        method(mismatched)

    wrong_config_type = TurboQuantMSEEncoded(
        indices=encoded.indices,
        norm=encoded.norm,
        config=(64, 2, 1729),  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="config"):
        method(wrong_config_type)


@pytest.mark.parametrize("method_name", ["decode", "physical_nbytes"])
def test_payload_consumers_reject_malformed_indices(method_name: str) -> None:
    codec = _codec(65, 1)
    encoded = codec.encode(torch.ones(65, dtype=torch.float64))
    method = getattr(codec, method_name)
    nonzero_padding = encoded.indices.clone()
    nonzero_padding[-1] |= 0b10000000
    malformed = [
        replace(encoded, indices=encoded.indices.to(torch.int64)),
        replace(encoded, indices=encoded.indices[:-1]),
        replace(encoded, indices=encoded.indices.reshape(1, 1, -1)),
        replace(encoded, indices=nonzero_padding),
    ]

    for payload in malformed:
        with pytest.raises((TypeError, ValueError), match="indices"):
            method(payload)


@pytest.mark.parametrize("method_name", ["decode", "physical_nbytes"])
def test_payload_consumers_reject_malformed_norms(method_name: str) -> None:
    codec = _codec(64, 2)
    encoded = codec.encode(torch.ones(64, dtype=torch.float64))
    method = getattr(codec, method_name)
    malformed = [
        replace(encoded, norm=encoded.norm.to(torch.float64)),
        replace(encoded, norm=encoded.norm.reshape(1)),
        replace(encoded, norm=torch.tensor(math.nan, dtype=torch.float32)),
        replace(encoded, norm=torch.tensor(math.inf, dtype=torch.float32)),
        replace(encoded, norm=torch.tensor(-1.0, dtype=torch.float32)),
    ]

    for payload in malformed:
        with pytest.raises((TypeError, ValueError), match="norm"):
            method(payload)


@pytest.mark.parametrize("method_name", ["decode", "physical_nbytes"])
def test_payload_consumers_reject_inconsistent_batch_shapes(
    method_name: str,
) -> None:
    codec = _codec(64, 2)
    encoded = codec.encode(torch.ones((2, 64), dtype=torch.float32))
    method = getattr(codec, method_name)
    malformed = [
        replace(encoded, indices=encoded.indices[:0]),
        replace(encoded, indices=encoded.indices[0]),
        replace(encoded, norm=encoded.norm[:1]),
        replace(encoded, norm=encoded.norm[0]),
    ]

    for payload in malformed:
        with pytest.raises(ValueError, match="indices|norm"):
            method(payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("field", ["indices", "norm"])
def test_payload_consumers_reject_cuda_tensors(field: str) -> None:
    codec = _codec(64, 2)
    encoded = codec.encode(torch.ones(64, dtype=torch.float64))
    cuda_encoded = replace(encoded, **{field: getattr(encoded, field).cuda()})

    with pytest.raises(ValueError, match="CPU"):
        codec.decode(cuda_encoded)
