import itertools
import math
import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
import torch

import core.paper_codebook as paper_codebook
from core.paper_codebook import (
    lloyd_max_sphere_codebook,
    sphere_coordinate_density,
)


def _gauss_legendre_integrals(
    dimension: int,
    *,
    codebook: torch.Tensor | None = None,
) -> tuple[float, float]:
    # 512-point Gauss-Legendre integration keeps both analytic moments below
    # 2e-12 error for all three dimensions while avoiding endpoint sampling.
    nodes, weights = np.polynomial.legendre.leggauss(512)
    coordinates = torch.from_numpy(nodes)
    density = sphere_coordinate_density(coordinates, dimension).numpy()
    integral = float(np.dot(weights, density))
    if codebook is None:
        measure = nodes * nodes
    else:
        centroids = codebook.numpy()
        measure = np.min((nodes[:, None] - centroids[None, :]) ** 2, axis=1)
    moment = float(np.dot(weights, measure * density))
    return integral, moment


def _spherical_density_scalar(value: float, dimension: int) -> float:
    if abs(value) >= 1:
        return 0.0
    log_coefficient = (
        math.lgamma(dimension / 2)
        - 0.5 * math.log(math.pi)
        - math.lgamma((dimension - 1) / 2)
    )
    return math.exp(
        log_coefficient
        + ((dimension - 3) / 2) * math.log1p(-(value * value))
    )


def _conditional_mean(lower: float, upper: float, dimension: int) -> float:
    if dimension == 2:
        lower_angle = math.asin(lower)
        upper_angle = math.asin(upper)
        return (
            math.cos(lower_angle) - math.cos(upper_angle)
        ) / (upper_angle - lower_angle)

    probability = quad(
        lambda value: _spherical_density_scalar(value, dimension),
        lower,
        upper,
        epsabs=1e-14,
        epsrel=1e-13,
    )[0]
    first_moment = quad(
        lambda value: value * _spherical_density_scalar(value, dimension),
        lower,
        upper,
        epsabs=1e-14,
        epsrel=1e-13,
    )[0]
    return first_moment / probability


def _equal_probability_initial_codebook(
    dimension: int,
    bits: int,
) -> torch.Tensor:
    levels = 1 << bits
    positive_boundaries = [0.0]
    for index in range(1, levels // 2):
        target = index / levels
        positive_boundaries.append(
            brentq(
                lambda upper: quad(
                    lambda value: _spherical_density_scalar(value, dimension),
                    0.0,
                    upper,
                    epsabs=1e-14,
                    epsrel=1e-13,
                )[0]
                - target,
                0.0,
                1.0,
            )
        )
    positive_boundaries.append(1.0)
    positive = [
        _conditional_mean(lower, upper, dimension)
        for lower, upper in itertools.pairwise(positive_boundaries)
    ]
    return torch.tensor(
        [-value for value in reversed(positive)] + positive,
        dtype=torch.float64,
    )


def _lloyd_max_residual(codebook: torch.Tensor, dimension: int) -> float:
    centroids = codebook.tolist()
    boundaries = [-1.0]
    boundaries.extend(
        (left + right) / 2
        for left, right in itertools.pairwise(centroids)
    )
    boundaries.append(1.0)
    conditional_means = [
        _conditional_mean(lower, upper, dimension)
        for lower, upper in itertools.pairwise(boundaries)
    ]
    return max(
        abs(centroid - conditional_mean)
        for centroid, conditional_mean in zip(
            centroids,
            conditional_means,
            strict=True,
        )
    )


def test_sphere_coordinate_density_has_declared_support_and_d3_value() -> None:
    coordinates = torch.tensor(
        [-1.25, -1.0, -0.25, 0.0, 0.25, 1.0, 1.25],
        dtype=torch.float64,
    )

    density = sphere_coordinate_density(coordinates, 3)

    torch.testing.assert_close(
        density,
        torch.tensor(
            [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0],
            dtype=torch.float64,
        ),
        atol=1e-15,
        rtol=0,
    )


def test_sphere_coordinate_density_coefficient_is_stable_at_large_dimension() -> None:
    dimension = 10**15
    asymptotic_coefficient = math.sqrt(dimension / (2 * math.pi))

    coefficient = sphere_coordinate_density(
        torch.tensor([0.0], dtype=torch.float64),
        dimension,
    ).item()

    assert coefficient == pytest.approx(asymptotic_coefficient, rel=1e-12)


@pytest.mark.parametrize("dimension", [64, 128, 256])
def test_sphere_coordinate_density_is_symmetric_and_has_exact_moments(
    dimension: int,
) -> None:
    coordinates = torch.linspace(-0.95, 0.95, 39, dtype=torch.float64)

    density = sphere_coordinate_density(coordinates, dimension)
    integral, second_moment = _gauss_legendre_integrals(dimension)

    torch.testing.assert_close(density, density.flip(0), atol=0, rtol=0)
    assert integral == pytest.approx(1.0, abs=2e-12)
    assert second_moment == pytest.approx(1.0 / dimension, abs=2e-12)


@pytest.mark.parametrize("dimension", [True, 1, 3.5])
def test_sphere_coordinate_density_rejects_invalid_dimensions(
    dimension: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="dimension"):
        sphere_coordinate_density(
            torch.tensor([0.0], dtype=torch.float64),
            dimension,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "coordinates",
    [
        [0.0],
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([math.nan], dtype=torch.float64),
        torch.tensor([math.inf], dtype=torch.float64),
    ],
)
def test_sphere_coordinate_density_rejects_invalid_coordinate_inputs(
    coordinates: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="x"):
        sphere_coordinate_density(coordinates, 64)  # type: ignore[arg-type]


def test_one_bit_codebook_matches_hand_derived_conditional_mean() -> None:
    dimension = 64
    log_coefficient = (
        math.lgamma(dimension / 2)
        - 0.5 * math.log(math.pi)
        - math.lgamma((dimension - 1) / 2)
    )
    positive_mean = 2.0 * math.exp(log_coefficient) / (dimension - 1)

    codebook = lloyd_max_sphere_codebook(dimension, 1)

    torch.testing.assert_close(
        codebook,
        torch.tensor([-positive_mean, positive_mean], dtype=torch.float64),
        atol=2e-13,
        rtol=0,
    )


def test_d2_codebooks_are_warning_free_converged_and_deterministic() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        for bits in range(1, 5):
            codebook = lloyd_max_sphere_codebook(2, bits)

            assert torch.all(codebook[1:] > codebook[:-1])
            assert torch.equal(codebook, -codebook.flip(0))
            assert torch.equal(codebook, lloyd_max_sphere_codebook(2, bits))
            assert _lloyd_max_residual(codebook, 2) <= 1e-12

    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, IntegrationWarning)
    ]


def test_integration_warning_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    real_quad = paper_codebook.quad

    def warning_quad(*args: object, **kwargs: object) -> tuple[float, float]:
        warnings.warn(
            "forced quadrature warning",
            IntegrationWarning,
            stacklevel=2,
        )
        return real_quad(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(paper_codebook, "quad", warning_quad)

    with pytest.raises(RuntimeError, match="integration"):
        lloyd_max_sphere_codebook(7, 2)


@pytest.mark.parametrize("dimension", [64, 128, 256])
def test_lloyd_max_codebooks_are_deterministic_symmetric_and_improve_with_bits(
    dimension: int,
) -> None:
    distortions: list[float] = []

    for bits in range(1, 5):
        initial = _equal_probability_initial_codebook(dimension, bits)
        codebook = lloyd_max_sphere_codebook(dimension, bits)

        assert codebook.device.type == "cpu"
        assert codebook.dtype is torch.float64
        assert codebook.shape == (1 << bits,)
        assert torch.all(codebook[1:] > codebook[:-1])
        assert torch.equal(codebook, -codebook.flip(0))
        assert torch.equal(
            codebook,
            lloyd_max_sphere_codebook(dimension, bits),
        )
        integral, distortion = _gauss_legendre_integrals(
            dimension,
            codebook=codebook,
        )
        _, initial_distortion = _gauss_legendre_integrals(
            dimension,
            codebook=initial,
        )
        assert integral == pytest.approx(1.0, abs=2e-12)
        assert _lloyd_max_residual(codebook, dimension) <= 1e-12
        assert distortion <= initial_distortion + 1e-15
        distortions.append(distortion)

    assert all(
        finer < coarser
        for coarser, finer in itertools.pairwise(distortions)
    )


@pytest.mark.parametrize(
    ("argument", "value", "error_type", "message"),
    [
        ("dimension", True, TypeError, "dimension"),
        ("dimension", 1, ValueError, "dimension"),
        ("dimension", 64.0, TypeError, "dimension"),
        ("bits", True, TypeError, "bits"),
        ("bits", 0, ValueError, "bits"),
        ("bits", 2.5, TypeError, "bits"),
        ("iterations", True, TypeError, "iterations"),
        ("iterations", 0, ValueError, "iterations"),
        ("iterations", 4.5, TypeError, "iterations"),
        ("tolerance", True, TypeError, "tolerance"),
        ("tolerance", 1, TypeError, "tolerance"),
        ("tolerance", 0.0, ValueError, "tolerance"),
        ("tolerance", math.inf, ValueError, "tolerance"),
        ("tolerance", math.nan, ValueError, "tolerance"),
    ],
)
def test_lloyd_max_codebook_rejects_invalid_parameters(
    argument: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    parameters: dict[str, object] = {
        "dimension": 64,
        "bits": 2,
        "iterations": 256,
        "tolerance": 1e-12,
    }
    parameters[argument] = value

    with pytest.raises(error_type, match=message):
        lloyd_max_sphere_codebook(**parameters)  # type: ignore[arg-type]


def test_lloyd_max_codebook_reports_non_convergence() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        lloyd_max_sphere_codebook(
            64,
            4,
            iterations=1,
            tolerance=1e-15,
        )
