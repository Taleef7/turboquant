"""CPU reference density and scalar codebooks from the TurboQuant paper."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import itertools
import math
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq, root
from scipy.special import betaln, logsumexp
import torch


@dataclass(frozen=True)
class PaperReference:
    """Identity fields for an exact paper revision used as formula context."""

    paper_id: str
    source_url: str
    retrieved_at: str
    sha256: str


@dataclass(frozen=True)
class _LloydMaxSolution:
    centroids: tuple[float, ...]
    evaluations: int
    max_change: float


class _EvaluationBudgetExhausted(Exception):
    pass


class _InvalidSolverCandidate(Exception):
    pass


def _validate_dimension(dimension: int) -> None:
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension < 2:
        raise ValueError("dimension must be at least 2")


def _log_density_coefficient(dimension: int) -> float:
    return float(-betaln(0.5, (dimension - 1) / 2))


def sphere_coordinate_density(
    x: torch.Tensor,
    dimension: int,
) -> torch.Tensor:
    """Evaluate the spherical-coordinate density, with zero outside [-1, 1]."""

    _validate_dimension(dimension)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type != "cpu":
        raise ValueError("x must be a CPU tensor")
    if not x.dtype.is_floating_point:
        raise TypeError("x must have a floating-point dtype")
    if not torch.isfinite(x).all():
        raise ValueError("x must contain only finite values")

    values = x.to(dtype=torch.float64)
    density = torch.zeros_like(values)
    inside = values.abs() <= 1
    inside_values = values[inside]
    exponent = (dimension - 3) / 2
    log_coefficient = _log_density_coefficient(dimension)

    if exponent == 0:
        density[inside] = math.exp(log_coefficient)
    elif inside_values.numel():
        bases = 1 - inside_values.square()
        endpoint = bases == 0
        inside_density = torch.empty_like(inside_values)
        inside_density[~endpoint] = torch.exp(
            log_coefficient + exponent * torch.log(bases[~endpoint])
        )
        inside_density[endpoint] = math.inf if exponent < 0 else 0.0
        density[inside] = inside_density

    return density.to(dtype=x.dtype)


def _checked_quad(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    quantity: str,
) -> float:
    absolute_tolerance = 1e-14
    relative_tolerance = 1e-13
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            value, error = quad(
                function,
                lower,
                upper,
                epsabs=absolute_tolerance,
                epsrel=relative_tolerance,
                limit=200,
            )
    except IntegrationWarning as warning:
        raise RuntimeError(
            f"integration failed while computing interval {quantity}"
        ) from warning

    allowed_error = max(
        absolute_tolerance,
        relative_tolerance * abs(value),
    )
    if (
        not math.isfinite(value)
        or not math.isfinite(error)
        or error > allowed_error
    ):
        raise RuntimeError(
            f"integration failed reliability check for interval {quantity}"
        )
    return float(value)


def _integrate_interval(
    lower: float,
    upper: float,
    *,
    exponent: float,
    log_coefficient: float,
) -> tuple[float, float]:
    lower_angle = math.asin(lower)
    upper_angle = math.asin(upper)
    cosine_exponent = 2 * exponent + 1

    def transformed_density(angle: float) -> float:
        cosine = math.cos(angle)
        if cosine <= 0:
            return math.exp(log_coefficient) if cosine_exponent == 0 else 0.0
        return math.exp(
            log_coefficient + cosine_exponent * math.log(cosine)
        )

    probability = _checked_quad(
        transformed_density,
        lower_angle,
        upper_angle,
        quantity="probability",
    )
    first_moment = _checked_quad(
        lambda angle: math.sin(angle) * transformed_density(angle),
        lower_angle,
        upper_angle,
        quantity="first moment",
    )
    if not math.isfinite(probability) or probability <= 0:
        raise RuntimeError("Lloyd-Max encountered a zero-probability interval")
    return probability, first_moment


def _equal_probability_positive_boundaries(
    levels: int,
    *,
    exponent: float,
    log_coefficient: float,
) -> list[float]:
    positive_levels = levels // 2

    def positive_probability(upper: float) -> float:
        if upper == 0:
            return 0.0
        return _integrate_interval(
            0.0,
            upper,
            exponent=exponent,
            log_coefficient=log_coefficient,
        )[0]

    boundaries = [0.0]
    for index in range(1, positive_levels):
        target = index / levels
        boundary = brentq(
            lambda value: positive_probability(value) - target,
            0.0,
            1.0,
            xtol=5e-15,
            rtol=4 * np.finfo(float).eps,
        )
        boundaries.append(float(boundary))
    boundaries.append(1.0)
    return boundaries


def _conditional_means(
    boundaries: list[float],
    *,
    exponent: float,
    log_coefficient: float,
) -> list[float]:
    centroids: list[float] = []
    for lower, upper in itertools.pairwise(boundaries):
        probability, first_moment = _integrate_interval(
            lower,
            upper,
            exponent=exponent,
            log_coefficient=log_coefficient,
        )
        centroids.append(first_moment / probability)
    return centroids


def _lloyd_update(
    positive: np.ndarray,
    *,
    exponent: float,
    log_coefficient: float,
) -> np.ndarray:
    midpoint_boundaries = [0.0]
    midpoint_boundaries.extend(
        (left + right) / 2
        for left, right in itertools.pairwise(positive.tolist())
    )
    midpoint_boundaries.append(1.0)
    return np.asarray(
        _conditional_means(
            midpoint_boundaries,
            exponent=exponent,
            log_coefficient=log_coefficient,
        ),
        dtype=np.float64,
    )


def _positive_centroids_to_logits(positive: np.ndarray) -> np.ndarray:
    spacings = np.diff(np.concatenate(([0.0], positive, [1.0])))
    if not np.all(np.isfinite(spacings)) or not np.all(spacings > 0):
        raise _InvalidSolverCandidate
    return np.log(spacings[:-1]) - math.log(float(spacings[-1]))


def _logits_to_positive_centroids(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 1 or not np.all(np.isfinite(logits)):
        raise _InvalidSolverCandidate
    extended = np.concatenate((logits, [0.0]))
    spacings = np.exp(extended - logsumexp(extended))
    positive = np.cumsum(spacings[:-1])
    if (
        not np.all(np.isfinite(positive))
        or positive[0] <= 0
        or positive[-1] >= 1
        or not np.all(np.diff(positive) > 0)
    ):
        raise _InvalidSolverCandidate
    return positive


@lru_cache(maxsize=32)
def _cached_lloyd_max_sphere_codebook(
    dimension: int,
    bits: int,
    iterations: int,
    tolerance: float,
) -> _LloydMaxSolution:
    levels = 1 << bits
    exponent = (dimension - 3) / 2
    log_coefficient = _log_density_coefficient(dimension)
    equal_probability_boundaries = _equal_probability_positive_boundaries(
        levels,
        exponent=exponent,
        log_coefficient=log_coefficient,
    )
    initial_positive = np.asarray(
        _conditional_means(
            equal_probability_boundaries,
            exponent=exponent,
            log_coefficient=log_coefficient,
        ),
        dtype=np.float64,
    )
    initial_logits = _positive_centroids_to_logits(initial_positive)
    evaluations = 0
    residual_cache: dict[tuple[float, ...], np.ndarray] = {}

    def fixed_point_residual(logits: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        key = tuple(float(value) for value in logits)
        cached = residual_cache.get(key)
        if cached is not None:
            return cached.copy()
        if evaluations >= iterations:
            raise _EvaluationBudgetExhausted
        positive = _logits_to_positive_centroids(logits)
        updated = _lloyd_update(
            positive,
            exponent=exponent,
            log_coefficient=log_coefficient,
        )
        residual = updated - positive
        evaluations += 1
        residual_cache[key] = residual
        return residual.copy()

    try:
        initial_residual = fixed_point_residual(initial_logits)
        if float(np.max(np.abs(initial_residual))) <= tolerance:
            solver_logits = initial_logits
            final_residual = initial_residual
        else:
            solver_result = root(
                fixed_point_residual,
                initial_logits,
                method="hybr",
                options={"maxfev": iterations, "xtol": tolerance},
            )
            solver_logits = np.asarray(solver_result.x, dtype=np.float64)
            final_residual = np.asarray(solver_result.fun, dtype=np.float64)
    except (_EvaluationBudgetExhausted, _InvalidSolverCandidate) as error:
        raise RuntimeError(
            "Lloyd-Max did not converge "
            f"for dimension={dimension}, bits={bits} after "
            f"{iterations} evaluations"
        ) from error

    max_change = float(np.max(np.abs(final_residual)))
    if not math.isfinite(max_change) or max_change > tolerance:
        raise RuntimeError(
            "Lloyd-Max did not converge "
            f"for dimension={dimension}, bits={bits} after "
            f"{evaluations} evaluations"
        )

    positive = _logits_to_positive_centroids(solver_logits)
    negative = tuple(-float(value) for value in reversed(positive))
    centroids = negative + tuple(float(value) for value in positive)
    return _LloydMaxSolution(
        centroids=centroids,
        evaluations=evaluations,
        max_change=max_change,
    )


def lloyd_max_sphere_codebook(
    dimension: int,
    bits: int,
    *,
    iterations: int = 256,
    tolerance: float = 1e-12,
) -> torch.Tensor:
    """Return a deterministic, sorted, exactly symmetric Lloyd-Max codebook."""

    _validate_dimension(dimension)
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError("bits must be an integer")
    if bits < 1:
        raise ValueError("bits must be at least 1")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError("iterations must be an integer")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if (
        isinstance(tolerance, (bool, Integral))
        or not isinstance(tolerance, Real)
    ):
        raise TypeError("tolerance must be a floating-point number")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    solution = _cached_lloyd_max_sphere_codebook(
        dimension,
        bits,
        iterations,
        float(tolerance),
    )
    return torch.tensor(
        solution.centroids,
        dtype=torch.float64,
        device="cpu",
    )
