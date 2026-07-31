"""Paper contracts for the TurboQuant product/QJL reference codec."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math

import pytest
import torch

from core.paper_mse import (
    TurboQuantMSEConfig,
    TurboQuantMSEEncoded,
    TurboQuantMSEReference,
)
from core.paper_product import (
    TurboQuantProductConfig,
    TurboQuantProductEncoded,
    TurboQuantProductReference,
)
from core.packing import unpack_sign_bits


class _IntSubclass(int):
    pass


class _SpoofedInt:
    @property
    def __class__(self) -> type[int]:
        return int


class _ProductConfigSubclass(TurboQuantProductConfig):
    pass


class _ProductEncodedSubclass(TurboQuantProductEncoded):
    pass


@lru_cache(maxsize=None)
def _codec(
    dimension: int = 9,
    total_bits: int = 3,
    rotation_seed: int = 17,
    projection_seed: int = 29,
) -> TurboQuantProductReference:
    return TurboQuantProductReference(
        TurboQuantProductConfig(
            dimension=dimension,
            total_bits=total_bits,
            rotation_seed=rotation_seed,
            projection_seed=projection_seed,
        )
    )


def _projection(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        (dimension, dimension),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )


def test_zero_residual_has_literal_positive_sign_bytes_and_zero_norm() -> None:
    codec = _codec()

    single = codec.encode(torch.zeros(9, dtype=torch.float64))
    batch = codec.encode(torch.zeros((2, 9), dtype=torch.float32))

    assert torch.equal(
        single.residual_signs,
        torch.tensor([0xFF, 0x01], dtype=torch.uint8),
    )
    assert torch.equal(
        batch.residual_signs,
        torch.tensor([[0xFF, 0x01], [0xFF, 0x01]], dtype=torch.uint8),
    )
    assert single.residual_norm.dtype is torch.float32
    assert single.residual_norm.shape == ()
    assert single.residual_norm.item() == 0.0
    assert torch.equal(
        batch.residual_norm,
        torch.zeros(2, dtype=torch.float32),
    )
    assert torch.equal(codec.decode(single), single.mse.norm.new_zeros(9).double())
    assert torch.equal(codec.decode(batch), torch.zeros((2, 9), dtype=torch.float64))


@pytest.mark.parametrize(
    ("dimension", "total_bits", "sign_bytes", "single_bytes", "batch_bytes"),
    [
        (9, 2, 2, 12, 36),
        (9, 3, 2, 13, 39),
        (64, 9, 8, 80, 240),
    ],
)
def test_payload_is_physically_packed_and_byte_count_is_hand_derived(
    dimension: int,
    total_bits: int,
    sign_bytes: int,
    single_bytes: int,
    batch_bytes: int,
) -> None:
    codec = _codec(dimension, total_bits)
    vector = torch.linspace(-1.0, 1.0, dimension, dtype=torch.float64)

    single = codec.encode(vector)
    batch = codec.encode(torch.stack((vector, -vector, vector / 2)))

    assert single.residual_signs.shape == (sign_bytes,)
    assert batch.residual_signs.shape == (3, sign_bytes)
    assert single.residual_signs.dtype is torch.uint8
    assert single.residual_norm.shape == ()
    assert batch.residual_norm.shape == (3,)
    assert codec.physical_nbytes(single) == single_bytes
    assert codec.physical_nbytes(batch) == batch_bytes


@pytest.mark.parametrize("total_bits", range(2, 10))
def test_d2_full_declared_width_domain_round_trips_and_counts_bytes(
    total_bits: int,
) -> None:
    codec = _codec(
        dimension=2,
        total_bits=total_bits,
        rotation_seed=41,
        projection_seed=53,
    )
    zero = codec.encode(torch.zeros(2, dtype=torch.float64))
    nonzero = codec.encode(torch.tensor([0.75, -1.25], dtype=torch.float32))
    expected_bytes = (
        math.ceil(2 * (total_bits - 1) / 8)
        + math.ceil(2 / 8)
        + 8
    )

    assert torch.equal(codec.decode(zero), torch.zeros(2, dtype=torch.float64))
    assert codec.decode(nonzero).shape == (2,)
    assert bool(torch.isfinite(codec.decode(nonzero)).all().item())
    assert codec.physical_nbytes(zero) == expected_bytes
    assert codec.physical_nbytes(nonzero) == expected_bytes


def test_algorithm_two_row_vector_encoding_and_decode_match_independent_oracle() -> None:
    dimension = 9
    codec = _codec(
        dimension=dimension,
        total_bits=3,
        rotation_seed=101,
        projection_seed=211,
    )
    vectors = torch.tensor(
        [
            [1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 1.5, -0.75, 2.25],
            [-0.5, 1.0, 2.0, -1.5, 0.75, 3.0, -2.0, 0.125, 1.25],
        ],
        dtype=torch.float64,
    )

    encoded = codec.encode(vectors)
    mse_reference = TurboQuantMSEReference(
        TurboQuantMSEConfig(dimension, 2, 101)
    )
    mse_decoded = mse_reference.decode(encoded.mse)
    residual = vectors - mse_decoded
    projection = _projection(dimension, 211)
    projected = residual @ projection.T
    expected_signs = torch.where(
        projected >= 0,
        torch.ones((), dtype=torch.int8),
        -torch.ones((), dtype=torch.int8),
    )
    stored_norm = torch.linalg.vector_norm(residual, dim=-1).to(torch.float32)
    expected_decoded = mse_decoded + (
        math.sqrt(math.pi / 2)
        / dimension
        * stored_norm.to(torch.float64).unsqueeze(-1)
        * (expected_signs.to(torch.float64) @ projection)
    )

    assert torch.equal(
        unpack_sign_bits(encoded.residual_signs, dimension),
        expected_signs,
    )
    assert torch.equal(encoded.residual_norm, stored_norm)
    torch.testing.assert_close(
        codec.decode(encoded),
        expected_decoded,
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("queries", "vectors", "expected_shape"),
    [
        (
            torch.arange(9, dtype=torch.float64),
            torch.ones(9, dtype=torch.float64),
            (),
        ),
        (
            torch.arange(9, dtype=torch.float32),
            torch.ones((3, 9), dtype=torch.float64),
            (3,),
        ),
        (
            torch.arange(18, dtype=torch.float64).reshape(2, 9),
            torch.ones(9, dtype=torch.float32),
            (2,),
        ),
        (
            torch.arange(18, dtype=torch.float32).reshape(2, 9),
            torch.ones((3, 9), dtype=torch.float32),
            (2, 3),
        ),
    ],
)
def test_inner_product_matches_query_dot_decoded_for_all_retrieval_shapes(
    queries: torch.Tensor,
    vectors: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    codec = _codec()
    encoded = codec.encode(vectors)
    decoded = codec.decode(encoded)

    actual = codec.estimate_inner_product(queries, encoded)
    query_rows = queries.to(torch.float64).reshape(-1, 9)
    decoded_rows = decoded.reshape(-1, 9)
    expected_matrix = query_rows @ decoded_rows.T
    if queries.ndim == 1 and decoded.ndim == 1:
        expected = expected_matrix[0, 0]
    elif queries.ndim == 1:
        expected = expected_matrix[0]
    elif decoded.ndim == 1:
        expected = expected_matrix[:, 0]
    else:
        expected = expected_matrix

    assert actual.shape == expected_shape
    assert actual.dtype is torch.float64
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=0)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("dimension", True, TypeError),
        ("dimension", 1, ValueError),
        ("dimension", 9.0, TypeError),
        ("total_bits", True, TypeError),
        ("total_bits", 1, ValueError),
        ("total_bits", 10, ValueError),
        ("total_bits", 3.0, TypeError),
        ("rotation_seed", True, TypeError),
        ("rotation_seed", 1.5, TypeError),
        ("projection_seed", True, TypeError),
        ("projection_seed", 1.5, TypeError),
    ],
)
def test_config_rejects_values_outside_exact_contract(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "dimension": 9,
        "total_bits": 3,
        "rotation_seed": 17,
        "projection_seed": 29,
    }
    arguments[field] = value

    with pytest.raises(error_type, match=field):
        TurboQuantProductConfig(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    ["dimension", "total_bits", "rotation_seed", "projection_seed"],
)
@pytest.mark.parametrize("value_factory", [_IntSubclass, lambda _: _SpoofedInt()])
def test_config_rejects_integer_subclasses_and_class_spoofers(
    field: str,
    value_factory: object,
) -> None:
    arguments: dict[str, object] = {
        "dimension": 9,
        "total_bits": 3,
        "rotation_seed": 17,
        "projection_seed": 29,
    }
    arguments[field] = value_factory(arguments[field])  # type: ignore[operator]

    with pytest.raises(TypeError, match=field):
        TurboQuantProductConfig(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "seed",
    [-0x8000_0000_0000_0000, -1, 0, 0xFFFF_FFFF_FFFF_FFFF],
)
@pytest.mark.parametrize("field", ["rotation_seed", "projection_seed"])
def test_config_and_reference_accept_full_generator_seed_range(
    field: str,
    seed: int,
) -> None:
    arguments = {
        "dimension": 2,
        "total_bits": 2,
        "rotation_seed": 17,
        "projection_seed": 29,
    }
    arguments[field] = seed
    config = TurboQuantProductConfig(**arguments)

    assert TurboQuantProductReference(config).config == config


@pytest.mark.parametrize(
    "seed",
    [-0x8000_0000_0000_0001, 0x1_0000_0000_0000_0000],
)
@pytest.mark.parametrize("field", ["rotation_seed", "projection_seed"])
def test_config_rejects_seeds_outside_generator_range(
    field: str,
    seed: int,
) -> None:
    arguments = {
        "dimension": 9,
        "total_bits": 3,
        "rotation_seed": 17,
        "projection_seed": 29,
    }
    arguments[field] = seed

    with pytest.raises(ValueError, match=field):
        TurboQuantProductConfig(**arguments)


def test_reference_requires_exact_config_type_and_does_not_change_global_rng() -> None:
    with pytest.raises(TypeError, match="config"):
        TurboQuantProductReference((9, 3, 17, 29))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        TurboQuantProductReference(_ProductConfigSubclass(9, 3, 17, 29))

    torch.manual_seed(991)
    expected = torch.rand(4)
    torch.manual_seed(991)
    TurboQuantProductReference(TurboQuantProductConfig(9, 3, 17, 29))
    actual = torch.rand(4)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "vectors",
    [
        [0.0] * 9,
        torch.zeros((), dtype=torch.float32),
        torch.zeros((1, 1, 9), dtype=torch.float32),
        torch.zeros((0, 9), dtype=torch.float32),
        torch.zeros(8, dtype=torch.float32),
        torch.zeros((2, 10), dtype=torch.float64),
        torch.zeros(9, dtype=torch.float16),
        torch.zeros(9, dtype=torch.int64),
        torch.full((9,), math.nan, dtype=torch.float32),
        torch.full((1, 9), math.inf, dtype=torch.float64),
    ],
)
def test_encode_rejects_malformed_nonfinite_or_unsupported_inputs(
    vectors: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="vectors"):
        _codec().encode(vectors)  # type: ignore[arg-type]


def test_encode_propagates_upstream_mse_float32_norm_overflow() -> None:
    vector = torch.full((9,), 1e39, dtype=torch.float64)

    with pytest.raises(ValueError, match="float32"):
        _codec().encode(vector)


def test_encode_propagates_upstream_nonfinite_mse_float64_norm() -> None:
    vector = torch.full((9,), 1e308, dtype=torch.float64)
    assert bool(torch.isfinite(vector).all().item())

    with pytest.raises(ValueError, match="finite norm"):
        _codec().encode(vector)


def test_encode_rejects_finite_residual_norm_that_overflows_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _codec(projection_seed=31)
    monkeypatch.setattr(
        codec._mse,
        "decode",
        lambda _: torch.full((9,), -1e39, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="residual.*float32"):
        codec.encode(torch.zeros(9, dtype=torch.float64))


def test_encode_rejects_finite_residual_with_nonfinite_float64_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _codec(projection_seed=37)
    monkeypatch.setattr(
        codec._mse,
        "decode",
        lambda _: torch.full((9,), -1e308, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="residual.*finite norm"):
        codec.encode(torch.zeros(9, dtype=torch.float64))


def test_tiny_residual_norm_uses_ordinary_float32_underflow() -> None:
    vector = torch.full((9,), 1e-50, dtype=torch.float64)

    encoded = _codec().encode(vector)
    mse_decoded = TurboQuantMSEReference(
        TurboQuantMSEConfig(9, 2, 17)
    ).decode(encoded.mse)

    assert encoded.residual_norm.item() == 0.0
    assert torch.equal(_codec().decode(encoded), mse_decoded)


@pytest.mark.parametrize(
    "queries",
    [
        [0.0] * 9,
        torch.zeros((), dtype=torch.float32),
        torch.zeros((1, 1, 9), dtype=torch.float32),
        torch.zeros((0, 9), dtype=torch.float32),
        torch.zeros(8, dtype=torch.float32),
        torch.zeros((2, 10), dtype=torch.float64),
        torch.zeros(9, dtype=torch.float16),
        torch.zeros(9, dtype=torch.int64),
        torch.full((9,), math.nan, dtype=torch.float32),
        torch.full((1, 9), math.inf, dtype=torch.float64),
    ],
)
def test_estimator_rejects_malformed_nonfinite_or_unsupported_queries(
    queries: object,
) -> None:
    encoded = _codec().encode(torch.ones(9, dtype=torch.float64))

    with pytest.raises((TypeError, ValueError), match="queries"):
        _codec().estimate_inner_product(queries, encoded)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "method_name",
    ["decode", "estimate_inner_product", "physical_nbytes"],
)
def test_consumers_reject_wrong_encoded_type_and_outer_config(
    method_name: str,
) -> None:
    codec = _codec()
    encoded = codec.encode(torch.ones(9, dtype=torch.float64))
    method = getattr(codec, method_name)
    args = (torch.ones(9, dtype=torch.float64), encoded) if method_name == "estimate_inner_product" else (encoded,)

    wrong_type_args = list(args)
    wrong_type_args[-1] = {"mse": encoded.mse}
    with pytest.raises(TypeError, match="encoded"):
        method(*wrong_type_args)

    encoded_subclass_args = list(args)
    encoded_subclass_args[-1] = _ProductEncodedSubclass(
        encoded.mse,
        encoded.residual_signs,
        encoded.residual_norm,
        encoded.config,
    )
    with pytest.raises(TypeError, match="encoded"):
        method(*encoded_subclass_args)

    mismatched = replace(
        encoded,
        config=TurboQuantProductConfig(9, 3, 17, 31),
    )
    mismatch_args = list(args)
    mismatch_args[-1] = mismatched
    with pytest.raises(ValueError, match="config"):
        method(*mismatch_args)

    wrong_config = replace(encoded, config=(9, 3, 17, 29))
    wrong_config_args = list(args)
    wrong_config_args[-1] = wrong_config
    with pytest.raises(TypeError, match="config"):
        method(*wrong_config_args)


@pytest.mark.parametrize(
    "method_name",
    ["decode", "estimate_inner_product", "physical_nbytes"],
)
def test_consumers_reject_wrong_or_mismatched_nested_mse_payload(
    method_name: str,
) -> None:
    codec = _codec()
    encoded = codec.encode(torch.ones(9, dtype=torch.float64))
    method = getattr(codec, method_name)
    bad_padding = encoded.mse.indices.clone()
    bad_padding[-1] |= 0b1000_0000

    malformed = [
        replace(encoded, mse={"indices": encoded.mse.indices}),
        replace(
            encoded,
            mse=replace(
                encoded.mse,
                indices=encoded.mse.indices.to(torch.int64),
            ),
        ),
        replace(
            encoded,
            mse=replace(encoded.mse, indices=bad_padding),
        ),
        replace(
            encoded,
            mse=replace(
                encoded.mse,
                norm=encoded.mse.norm.to(torch.float64),
            ),
        ),
        replace(
            encoded,
            mse=replace(
                encoded.mse,
                config=TurboQuantMSEConfig(9, 2, 19),
            ),
        ),
        replace(
            encoded,
            mse=TurboQuantMSEEncoded(
                encoded.mse.indices,
                encoded.mse.norm,
                (9, 2, 17),  # type: ignore[arg-type]
            ),
        ),
    ]

    for payload in malformed:
        args = (
            (torch.ones(9, dtype=torch.float64), payload)
            if method_name == "estimate_inner_product"
            else (payload,)
        )
        with pytest.raises(
            (TypeError, ValueError),
            match="mse|config|indices|norm",
        ):
            method(*args)


@pytest.mark.parametrize(
    "method_name",
    ["decode", "estimate_inner_product", "physical_nbytes"],
)
def test_consumers_reject_malformed_residual_sign_payload(
    method_name: str,
) -> None:
    codec = _codec()
    encoded = codec.encode(torch.ones(9, dtype=torch.float64))
    method = getattr(codec, method_name)
    bad_padding = encoded.residual_signs.clone()
    bad_padding[-1] |= 0b1000_0000
    malformed = [
        replace(encoded, residual_signs=b""),
        replace(encoded, residual_signs=encoded.residual_signs.to(torch.int64)),
        replace(encoded, residual_signs=encoded.residual_signs[:-1]),
        replace(encoded, residual_signs=encoded.residual_signs.reshape(1, 1, -1)),
        replace(encoded, residual_signs=bad_padding),
    ]

    for payload in malformed:
        args = (
            (torch.ones(9, dtype=torch.float64), payload)
            if method_name == "estimate_inner_product"
            else (payload,)
        )
        with pytest.raises((TypeError, ValueError), match="residual signs"):
            method(*args)


@pytest.mark.parametrize(
    "method_name",
    ["decode", "estimate_inner_product", "physical_nbytes"],
)
def test_consumers_reject_malformed_residual_norm_payload(
    method_name: str,
) -> None:
    codec = _codec()
    encoded = codec.encode(torch.ones(9, dtype=torch.float64))
    method = getattr(codec, method_name)
    malformed = [
        replace(encoded, residual_norm=0.0),
        replace(encoded, residual_norm=encoded.residual_norm.to(torch.float64)),
        replace(encoded, residual_norm=encoded.residual_norm.reshape(1)),
        replace(encoded, residual_norm=torch.tensor(math.nan, dtype=torch.float32)),
        replace(encoded, residual_norm=torch.tensor(math.inf, dtype=torch.float32)),
        replace(encoded, residual_norm=torch.tensor(-1.0, dtype=torch.float32)),
    ]

    for payload in malformed:
        args = (
            (torch.ones(9, dtype=torch.float64), payload)
            if method_name == "estimate_inner_product"
            else (payload,)
        )
        with pytest.raises((TypeError, ValueError), match="residual norm"):
            method(*args)


@pytest.mark.parametrize(
    "method_name",
    ["decode", "estimate_inner_product", "physical_nbytes"],
)
def test_consumers_reject_inconsistent_single_and_batch_shapes(
    method_name: str,
) -> None:
    codec = _codec()
    encoded = codec.encode(torch.ones((2, 9), dtype=torch.float64))
    method = getattr(codec, method_name)
    malformed = [
        replace(encoded, residual_signs=encoded.residual_signs[:0]),
        replace(encoded, residual_signs=encoded.residual_signs[0]),
        replace(encoded, residual_norm=encoded.residual_norm[:1]),
        replace(encoded, residual_norm=encoded.residual_norm[0]),
        replace(encoded, mse=replace(encoded.mse, indices=encoded.mse.indices[:1], norm=encoded.mse.norm[:1])),
    ]

    for payload in malformed:
        args = (
            (torch.ones(9, dtype=torch.float64), payload)
            if method_name == "estimate_inner_product"
            else (payload,)
        )
        with pytest.raises(ValueError, match="batch|mse|residual"):
            method(*args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_encode_queries_and_all_payload_tensors_reject_cuda() -> None:
    codec = _codec()
    encoded = codec.encode(torch.ones(9, dtype=torch.float64))

    with pytest.raises(ValueError, match="CPU"):
        codec.encode(torch.ones(9, device="cuda"))
    with pytest.raises(ValueError, match="CPU"):
        codec.estimate_inner_product(torch.ones(9, device="cuda"), encoded)

    fields = [
        ("residual_signs", encoded.residual_signs.cuda()),
        ("residual_norm", encoded.residual_norm.cuda()),
        (
            "mse",
            replace(encoded.mse, indices=encoded.mse.indices.cuda()),
        ),
        ("mse", replace(encoded.mse, norm=encoded.mse.norm.cuda())),
    ]
    for field, value in fields:
        with pytest.raises(ValueError, match="CPU"):
            codec.decode(replace(encoded, **{field: value}))


@pytest.mark.parametrize("dimension", [64, 128])
def test_qjl_signed_error_smoke_gate_is_centered_on_stored_float32_gamma(
    dimension: int,
) -> None:
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(4000 + dimension)
    vectors = torch.randn(
        (8, dimension),
        generator=data_generator,
        dtype=torch.float64,
    )
    vectors /= torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    queries = torch.randn(
        (8, dimension),
        generator=data_generator,
        dtype=torch.float64,
    )
    queries /= torch.linalg.vector_norm(queries, dim=-1, keepdim=True)
    rotation_seed = 7000 + dimension
    mse_codec = TurboQuantMSEReference(
        TurboQuantMSEConfig(dimension, 2, rotation_seed)
    )
    mse_payload = mse_codec.encode(vectors)
    mse_decoded = mse_codec.decode(mse_payload)
    residual = vectors - mse_decoded
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    stored_gamma = residual_norm.to(torch.float32).to(torch.float64)
    expected_vectors = mse_decoded + residual * (
        stored_gamma / residual_norm
    ).unsqueeze(-1)
    expectation = (queries * expected_vectors).sum(dim=-1)
    signed_errors: list[float] = []

    for projection_seed in range(128):
        codec = _codec(
            dimension=dimension,
            total_bits=3,
            rotation_seed=rotation_seed,
            projection_seed=projection_seed,
        )
        estimates = codec.estimate_inner_product(
            queries,
            codec.encode(vectors),
        )
        signed_errors.append(
            (estimates.diagonal() - expectation).mean().item()
        )

    errors = torch.tensor(signed_errors, dtype=torch.float64)
    mean = errors.mean().item()
    sem = (errors.std(unbiased=True) / math.sqrt(errors.numel())).item()
    ci95 = (mean - 1.96 * sem, mean + 1.96 * sem)

    print(
        f"d={dimension} mean={mean:.6g} sem={sem:.6g} "
        f"ci95=({ci95[0]:.6g}, {ci95[1]:.6g})"
    )
    assert abs(mean) <= 4 * sem + 1e-12
