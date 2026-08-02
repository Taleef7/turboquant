"""Build the resumable local TurboQuant codec and GloVe reproduction artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import urllib.request

import numpy as np
from scipy.stats import t as student_t
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paper_mse import (  # noqa: E402
    TurboQuantMSEConfig,
    TurboQuantMSEReference,
)
from core.paper_product import (  # noqa: E402
    TurboQuantProductConfig,
    TurboQuantProductReference,
)
from scripts.benchmark_schema import V2_ROW_STATES, load_strict_json  # noqa: E402
from scripts.repro_utils import (  # noqa: E402
    capture_cuda_metadata,
    capture_env_metadata,
    capture_source_metadata,
    package_version,
    sha256_file,
)

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
RESUME_PROTOCOL = "codec_reproduction_row_v2"
DEFAULT_RETRIEVAL_CHUNK_SIZE = 2_048
_CODEC_SOURCE_PATHS = (
    "scripts/run_codec_reproduction.py",
    "scripts/benchmark_schema.py",
    "scripts/repro_utils.py",
    "scripts/claim_ledger.py",
    "core/paper_mse.py",
    "core/paper_product.py",
    "core/paper_codebook.py",
    "core/paper_rotation.py",
    "core/packing.py",
)
_CODEC_CACHE: dict[tuple[object, ...], object] = {}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _behavior_sha256() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _CODEC_SOURCE_PATHS:
        path = ROOT / relative
        result[relative] = sha256_file(path)
    return result


def codec_identity(config_path: str | Path) -> dict[str, str]:
    """Return schema-v2 identity without pretending a codec has model weights."""
    path = Path(config_path)
    artifact_id = json.loads(path.read_text(encoding="utf-8"))["artifact_id"]
    return {
        "model_id": "not_applicable:codec-only",
        "model_revision": "not_applicable:codec-only",
        "weights_sha256": EMPTY_SHA256,
        "profile_id": artifact_id,
        "profile_sha256": sha256_file(path),
        "kv_dtype": "float32",
    }


def _fixture_lookup(
    config: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    fixtures = config["synthetic"]["fixtures"]
    return {
        (fixture["distribution"], fixture["dimension"]): fixture for fixture in fixtures
    }


def build_row_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Materialize canonical row identities and every input seed they bind."""
    pairs = config["seed_pairs"]
    synthetic = config["synthetic"]
    fixtures = _fixture_lookup(config)
    specs: list[dict[str, Any]] = []
    for distribution in (item["name"] for item in synthetic["distributions"]):
        for dimension in synthetic["dimensions"]:
            fixture = fixtures[(distribution, dimension)]
            for codec_name in ("mse", "product"):
                widths = synthetic["codecs"][codec_name]["nominal_total_bits"]
                for bits in widths:
                    for pair in pairs:
                        pair_id = pair["pair_id"]
                        specs.append(
                            {
                                "row_id": (
                                    f"syn-{distribution}-d{dimension}-"
                                    f"{codec_name}-b{bits}-s{pair_id:02d}"
                                ),
                                "dataset": "synthetic",
                                "distribution": distribution,
                                "dimension": dimension,
                                "codec": codec_name,
                                "nominal_total_bits": bits,
                                "pair_id": pair_id,
                                "rotation_seed": pair["rotation_seed"],
                                "projection_seed": (
                                    pair["projection_seed"]
                                    if codec_name == "product"
                                    else None
                                ),
                                "fixture_id": fixture["fixture_id"],
                                "data_seed": fixture["data_seed"],
                                "independent_query_seed": fixture[
                                    "independent_query_seed"
                                ],
                                "high_inner_product_noise_seed": fixture[
                                    "high_inner_product_noise_seed"
                                ],
                                "vector_count": synthetic["vector_count"],
                                "chunk_size": synthetic["chunk_size"],
                            }
                        )
    glove = config["glove200"]
    dimension = glove["datasets"]["train"]["shape"][1]
    for codec_name in glove["codecs"]:
        for bits in glove["nominal_total_bits"]:
            for pair in pairs:
                pair_id = pair["pair_id"]
                specs.append(
                    {
                        "row_id": (f"glove200-{codec_name}-b{bits}-s{pair_id:02d}"),
                        "dataset": "glove200",
                        "dimension": dimension,
                        "codec": codec_name,
                        "nominal_total_bits": bits,
                        "pair_id": pair_id,
                        "rotation_seed": pair["rotation_seed"],
                        "projection_seed": (
                            pair["projection_seed"] if codec_name == "product" else None
                        ),
                    }
                )
    return sorted(specs, key=lambda item: item["row_id"])


def _validate_config(config: Mapping[str, Any], *, canonical: bool) -> None:
    if config.get("config_version") != 1:
        raise ValueError("config_version must be 1")
    pairs = config.get("seed_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("seed_pairs must be a non-empty list")
    for index, pair in enumerate(pairs):
        if pair != {
            "pair_id": index,
            "rotation_seed": 10_000 + index,
            "projection_seed": 1_000_000 + index,
        }:
            raise ValueError(f"seed pair {index} does not match the declared schedule")
    fixtures = config["synthetic"]["fixtures"]
    namespaces = (
        ("data_seed", 3_000_000),
        ("independent_query_seed", 4_000_000),
        ("high_inner_product_noise_seed", 5_000_000),
    )
    for field, start in namespaces:
        if [fixture[field] for fixture in fixtures] != list(
            range(start, start + len(fixtures))
        ):
            raise ValueError(f"{field} does not match its disjoint seed namespace")
    row_ids = [spec["row_id"] for spec in build_row_specs(config)]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("row specifications contain duplicate row IDs")
    if canonical and (len(row_ids) != 1_408 or len(fixtures) != 12):
        raise ValueError("canonical config must materialize 1,408 rows and 12 fixtures")
    base_ids, query_ids = glove_selection_ids(config["glove200"])
    if (
        hashlib.sha256(base_ids.tobytes()).hexdigest()
        != (config["glove200"]["base_selection"]["sha256"])
    ):
        raise ValueError("GloVe base selection ID SHA-256 mismatch")
    if (
        hashlib.sha256(query_ids.tobytes()).hexdigest()
        != (config["glove200"]["query_selection"]["sha256"])
    ):
        raise ValueError("GloVe query selection ID SHA-256 mismatch")


def load_config(path: str | Path) -> dict[str, Any]:
    payload = load_strict_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("codec reproduction config must contain an object")
    canonical = (
        Path(path).resolve()
        == (ROOT / "configs" / "benchmarks" / "codec-local-v1.json").resolve()
    )
    _validate_config(payload, canonical=canonical)
    return payload


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("vectors must have finite nonzero norms")
    return np.asarray(values / norms, dtype=np.float32)


def _synthetic_vectors(
    distribution: str,
    dimension: int,
    count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    if distribution == "sphere":
        return _normalize_rows(generator.standard_normal((count, dimension)))
    if distribution == "gaussian":
        values = generator.standard_normal((count, dimension)) / math.sqrt(dimension)
    elif distribution == "heavy_tail":
        values = generator.standard_t(3, size=(count, dimension)) / math.sqrt(
            3 * dimension
        )
    elif distribution == "outlier":
        values = generator.standard_normal((count, dimension)) / math.sqrt(dimension)
        coordinates = generator.integers(0, dimension, size=count)
        signs = generator.integers(0, 2, size=count) * 2 - 1
        values[np.arange(count), coordinates] += 8.0 * signs
        values /= math.sqrt(65.0)
    else:
        raise ValueError(f"unknown synthetic distribution: {distribution}")
    return np.asarray(values, dtype=np.float32)


def _query_fixture(
    vectors: np.ndarray,
    independent_generator: np.random.Generator,
    high_ip_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    count, dimension = vectors.shape
    independent = _normalize_rows(
        independent_generator.standard_normal((count, dimension))
    )
    unit = _normalize_rows(vectors)
    noise = high_ip_generator.standard_normal((count, dimension))
    noise -= np.sum(noise * unit, axis=1, keepdims=True) * unit
    noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
    bad = np.flatnonzero(noise_norms[:, 0] <= np.finfo(np.float64).eps)
    for index in bad:
        coordinate = int(np.argmin(np.abs(unit[index])))
        noise[index] = 0
        noise[index, coordinate] = 1
        noise[index] -= float(noise[index] @ unit[index]) * unit[index]
    orthogonal = _normalize_rows(noise)
    high_ip = np.asarray(0.8 * unit + 0.6 * orthogonal, dtype=np.float32)
    return independent, high_ip


@dataclass
class _SyntheticStream:
    distribution: str
    dimension: int
    data: np.random.Generator
    independent: np.random.Generator
    high_ip: np.random.Generator

    def take(self, count: int) -> dict[str, np.ndarray]:
        vectors = _synthetic_vectors(
            self.distribution, self.dimension, count, self.data
        )
        independent, high_ip = _query_fixture(vectors, self.independent, self.high_ip)
        return {
            "vectors": vectors,
            "independent_queries": independent,
            "high_inner_product_queries": high_ip,
        }


def _synthetic_stream(
    *,
    distribution: str,
    dimension: int,
    data_seed: int,
    independent_query_seed: int,
    high_inner_product_noise_seed: int,
) -> _SyntheticStream:
    return _SyntheticStream(
        distribution=distribution,
        dimension=dimension,
        data=np.random.Generator(np.random.PCG64(data_seed)),
        independent=np.random.Generator(np.random.PCG64(independent_query_seed)),
        high_ip=np.random.Generator(np.random.PCG64(high_inner_product_noise_seed)),
    )


def generate_synthetic_fixture(
    *,
    distribution: str,
    dimension: int,
    count: int,
    data_seed: int,
    independent_query_seed: int,
    high_inner_product_noise_seed: int,
) -> dict[str, np.ndarray]:
    """Generate a deterministic in-memory fixture for tests and small probes."""
    return _synthetic_stream(
        distribution=distribution,
        dimension=dimension,
        data_seed=data_seed,
        independent_query_seed=independent_query_seed,
        high_inner_product_noise_seed=high_inner_product_noise_seed,
    ).take(count)


@dataclass
class _Moments:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0

    def add(self, values: np.ndarray | torch.Tensor) -> None:
        if isinstance(values, torch.Tensor):
            array = values.detach().cpu().numpy()
        else:
            array = np.asarray(values)
        array = np.asarray(array, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("statistics require finite non-empty observations")
        self.count += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.total_square += float(np.square(array).sum(dtype=np.float64))

    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("mean requires observations")
        return self.total / self.count

    def signed_summary(self) -> dict[str, object]:
        if self.count < 2:
            raise ValueError("Student-t summary requires at least two observations")
        bias = self.total / self.count
        variance = (self.total_square - self.count * bias * bias) / (self.count - 1)
        standard_deviation = math.sqrt(max(0.0, variance))
        sem = standard_deviation / math.sqrt(self.count)
        critical = float(student_t.ppf(0.975, df=self.count - 1))
        return {
            "count": self.count,
            "rmse": math.sqrt(self.total_square / self.count),
            "signed_bias": bias,
            "sample_standard_deviation": standard_deviation,
            "sem": sem,
            "student_t_95_ci": [
                bias - critical * sem,
                bias + critical * sem,
            ],
        }


def summarize_signed_errors(values: np.ndarray) -> dict[str, object]:
    moments = _Moments()
    moments.add(values)
    return moments.signed_summary()


def physical_bytes_formula(
    codec_name: str, dimension: int, nominal_total_bits: int, count: int
) -> int:
    mse_bits = nominal_total_bits if codec_name == "mse" else nominal_total_bits - 1
    mse_bytes = math.ceil(dimension * mse_bits / 8) + 4
    if codec_name == "mse":
        return count * mse_bytes
    if codec_name != "product":
        raise ValueError(f"unknown codec: {codec_name}")
    return count * (mse_bytes + math.ceil(dimension / 8) + 4)


def product_float32_residual_expectation(
    vectors: torch.Tensor,
    mse_decoded: torch.Tensor,
    stored_residual_norm: torch.Tensor,
    queries: torch.Tensor,
) -> torch.Tensor:
    residual = vectors.to(torch.float64) - mse_decoded
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    ratio = torch.zeros_like(residual_norm)
    nonzero = residual_norm > 0
    ratio[nonzero] = (
        stored_residual_norm.to(torch.float64)[nonzero] / residual_norm[nonzero]
    )
    expected = mse_decoded + residual * ratio.unsqueeze(-1)
    return (queries.to(torch.float64) * expected).sum(dim=-1)


def artifact_status(states: Iterable[str], *, dirty_source: bool) -> str:
    outcomes = set(states)
    if "invalid" in outcomes:
        return "invalid"
    if outcomes.intersection({"failed", "oom"}):
        return "failed"
    if dirty_source or outcomes.intersection({"unsupported", "unavailable"}):
        return "partial"
    return "qualified"


def can_resume_row(
    row: Mapping[str, Any], expected_fingerprint: Mapping[str, Any]
) -> bool:
    return (
        row.get("status") in {"passed", "failed", "unsupported"}
        and row.get("extensions", {}).get("resume_fingerprint") == expected_fingerprint
    )


def result_sha256(row: Mapping[str, Any]) -> str:
    """Hash a persisted row while excluding the seal itself."""
    payload = deepcopy(dict(row))
    extensions = payload.get("extensions")
    if isinstance(extensions, dict):
        extensions.pop("result_sha256", None)
    return _sha256_json(payload)


def seal_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy with a deterministic content seal."""
    payload = deepcopy(dict(row))
    extensions = payload.setdefault("extensions", {})
    if not isinstance(extensions, dict):
        raise ValueError("row extensions must be an object")
    extensions.pop("result_sha256", None)
    extensions["result_sha256"] = result_sha256(payload)
    return payload


_BASE_METRIC_KEYS = {
    "sample_count",
    "dimension",
    "mean_squared_l2_error",
    "relative_squared_error",
    "encode_latency_ms",
    "decode_latency_ms",
    "encode_latency_us_per_vector",
    "decode_latency_us_per_vector",
    "actual_physical_bytes",
    "formula_physical_bytes",
    "effective_physical_bits_per_coordinate",
    "nominal_total_bits_per_coordinate",
}
_SIGNED_SUMMARY_KEYS = {
    "count",
    "rmse",
    "signed_bias",
    "sample_standard_deviation",
    "sem",
    "student_t_95_ci",
}
_GLOVE_METRIC_KEYS = {
    "retrieval",
    "retrieval_scoring_latency_ms",
    "retrieval_scoring_device",
    "retrieval_scoring_chunk_size",
    "retrieval_algorithm",
    "scoring_device",
    "full_score_matrix_materialized",
    "maximum_score_block",
    "gpu_indexing_latency_ms",
}


def _require_finite_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_signed_summary(value: object, *, expected_count: int, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _SIGNED_SUMMARY_KEYS:
        raise ValueError(f"{label} has an unexpected metric contract")
    if value["count"] != expected_count:
        raise ValueError(f"{label}.count does not match sample_count")
    for key in _SIGNED_SUMMARY_KEYS - {"count", "student_t_95_ci"}:
        _require_finite_number(value[key], f"{label}.{key}")
    interval = value["student_t_95_ci"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{label}.student_t_95_ci must contain two bounds")
    for index, bound in enumerate(interval):
        _require_finite_number(bound, f"{label}.student_t_95_ci.{index}")


def _validate_declared_sidecars(
    sidecars: object, *, repo_root: Path, expected_paths: Sequence[str]
) -> None:
    if not isinstance(sidecars, list):
        raise ValueError("row sidecars must be a list")
    actual_paths: list[str] = []
    for sidecar in sidecars:
        if not isinstance(sidecar, Mapping) or set(sidecar) != {
            "path",
            "sha256",
            "available",
        }:
            raise ValueError("row sidecar declaration is malformed")
        relative = sidecar["path"]
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("row sidecar path must be repository-relative")
        target = (repo_root / relative).resolve()
        if not target.is_relative_to(repo_root.resolve()) or not target.is_file():
            raise ValueError(f"row sidecar is missing: {relative}")
        if sidecar["available"] is not True:
            raise ValueError(f"row sidecar is unavailable: {relative}")
        if sidecar["sha256"] != sha256_file(target):
            raise ValueError(f"row sidecar hash mismatch: {relative}")
        actual_paths.append(relative)
    if actual_paths != list(expected_paths):
        raise ValueError("row sidecars do not match the declared metric contract")


def validate_persisted_row(
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
    identity: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    """Strictly validate a row before it may be resumed or collected."""
    if row.get("row_id") != spec["row_id"]:
        raise ValueError("row ID does not match expected row")
    if row.get("identity") != identity:
        raise ValueError("row identity does not match artifact identity")
    extensions = row.get("extensions")
    if not isinstance(extensions, Mapping):
        raise ValueError("row extensions must be an object")
    if extensions.get("result_sha256") != result_sha256(row):
        raise ValueError("row result seal mismatch")
    if extensions.get("row_spec") != spec:
        raise ValueError("row spec does not match expected row")
    if extensions.get("resume_fingerprint") != fingerprint:
        raise ValueError("row resume fingerprint does not match expected provenance")
    if extensions.get("timing_class") != fingerprint.get("timing_class"):
        raise ValueError("row timing class does not match expected provenance")
    if set(extensions) != {
        "row_spec",
        "resume_fingerprint",
        "timing_class",
        "result_sha256",
    } and not (
        spec["dataset"] == "glove200"
        and set(extensions)
        == {
            "row_spec",
            "resume_fingerprint",
            "timing_class",
            "retrieval_scoring",
            "result_sha256",
        }
    ):
        raise ValueError("row extensions do not match the status contract")

    status = row.get("status")
    if status not in V2_ROW_STATES:
        raise ValueError("row status is not a schema-v2 state")
    if status == "passed":
        if row.get("reason") is not None:
            raise ValueError("passed row reason must be null")
    elif not isinstance(row.get("reason"), str) or not row["reason"]:
        raise ValueError("non-passed row must have a non-empty reason")

    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("row metrics must be an object")
    if status != "passed":
        if metrics or row.get("sidecars") != []:
            raise ValueError("non-passed row must not declare measurements")
        return

    expected_keys = set(_BASE_METRIC_KEYS)
    if spec["dataset"] == "synthetic":
        expected_keys.add("queries")
        if spec["codec"] == "product":
            expected_keys.add("finite_float32_residual_expectation")
    else:
        expected_keys.update(_GLOVE_METRIC_KEYS)
    if set(metrics) != expected_keys:
        raise ValueError("passed row metrics do not match the exact contract")

    contract = fingerprint.get("metric_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("resume fingerprint is missing the metric contract")
    count = contract["sample_count"]
    dimension = spec["dimension"]
    bits = spec["nominal_total_bits"]
    if metrics["sample_count"] != count or metrics["dimension"] != dimension:
        raise ValueError("row sample_count or dimension does not match the row spec")
    if metrics["nominal_total_bits_per_coordinate"] != bits:
        raise ValueError("row nominal bit width does not match the row spec")
    formula = physical_bytes_formula(spec["codec"], dimension, bits, count)
    if (
        metrics["actual_physical_bytes"] != formula
        or metrics["formula_physical_bytes"] != formula
    ):
        raise ValueError("row physical bytes do not match the codec formula")
    expected_effective = formula * 8 / (count * dimension)
    if metrics["effective_physical_bits_per_coordinate"] != expected_effective:
        raise ValueError("row effective physical width does not match bytes")
    for key in _BASE_METRIC_KEYS - {
        "sample_count",
        "dimension",
        "actual_physical_bytes",
        "formula_physical_bytes",
        "nominal_total_bits_per_coordinate",
    }:
        _require_finite_number(metrics[key], f"metrics.{key}")

    if spec["dataset"] == "synthetic":
        queries = metrics["queries"]
        if not isinstance(queries, Mapping) or set(queries) != {
            "independent",
            "high_inner_product",
        }:
            raise ValueError("synthetic query metrics have an unexpected contract")
        for name in ("independent", "high_inner_product"):
            _validate_signed_summary(
                queries[name],
                expected_count=count,
                label=f"metrics.queries.{name}",
            )
        if spec["codec"] == "product":
            expectation = metrics["finite_float32_residual_expectation"]
            if not isinstance(expectation, Mapping) or set(expectation) != {
                "independent",
                "high_inner_product",
            }:
                raise ValueError("product expectation metrics are malformed")
            for name in ("independent", "high_inner_product"):
                _validate_signed_summary(
                    expectation[name],
                    expected_count=count,
                    label=("metrics.finite_float32_residual_expectation." f"{name}"),
                )
        _validate_declared_sidecars(
            row.get("sidecars"), repo_root=repo_root, expected_paths=()
        )
        return

    expected_top_k = contract["top_k"]
    retrieval = metrics["retrieval"]
    if not isinstance(retrieval, Mapping) or set(retrieval) != {
        f"1@{value}" for value in expected_top_k
    }:
        raise ValueError("GloVe retrieval metrics have an unexpected contract")
    for key, value in retrieval.items():
        number = _require_finite_number(value, f"metrics.retrieval.{key}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"metrics.retrieval.{key} must be in [0, 1]")
    if (
        metrics["retrieval_scoring_device"] != fingerprint["retrieval_policy"]["device"]
        or metrics["scoring_device"] != fingerprint["retrieval_policy"]["device"]
        or metrics["retrieval_scoring_chunk_size"]
        != fingerprint["retrieval_policy"]["base_chunk_size"]
        or metrics["retrieval_algorithm"] != "exact_chunked_brute_force_dot_product"
        or metrics["full_score_matrix_materialized"] is not False
        or metrics["gpu_indexing_latency_ms"] is not None
    ):
        raise ValueError("GloVe retrieval labels do not match the policy")
    maximum = metrics["maximum_score_block"]
    if maximum != {
        "rows": contract["query_count"],
        "columns": min(fingerprint["retrieval_policy"]["base_chunk_size"], count),
    }:
        raise ValueError("GloVe maximum score block does not match the policy")
    _require_finite_number(
        metrics["retrieval_scoring_latency_ms"],
        "metrics.retrieval_scoring_latency_ms",
    )
    retrieval_extension = extensions["retrieval_scoring"]
    if not isinstance(retrieval_extension, Mapping) or set(retrieval_extension) != {
        "device",
        "base_chunk_size",
        "codec_encode_decode_device",
        "gpu_indexing_performed",
        "retrieval_algorithm",
        "scoring_device",
        "full_score_matrix_materialized",
        "maximum_score_block",
    }:
        raise ValueError("GloVe retrieval extension is malformed")
    if retrieval_extension != {
        "device": fingerprint["retrieval_policy"]["device"],
        "base_chunk_size": fingerprint["retrieval_policy"]["base_chunk_size"],
        "codec_encode_decode_device": "cpu",
        "gpu_indexing_performed": False,
        "retrieval_algorithm": "exact_chunked_brute_force_dot_product",
        "scoring_device": fingerprint["retrieval_policy"]["device"],
        "full_score_matrix_materialized": False,
        "maximum_score_block": maximum,
    }:
        raise ValueError("GloVe retrieval extension does not match the policy")
    expected_paths = contract["sidecar_paths"]
    _validate_declared_sidecars(
        row.get("sidecars"),
        repo_root=repo_root,
        expected_paths=expected_paths,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: str | Path, payload: object) -> None:
    _atomic_write(
        Path(path),
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )


def _atomic_write_if_changed(path: Path, payload: bytes) -> None:
    if path.is_file() and path.read_bytes() == payload:
        return
    _atomic_write(path, payload)


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: row["row_id"])
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in ordered)
    _atomic_write(Path(path), payload)


def _placeholder(
    row_id: str, identity: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": "invalid",
        "reason": reason,
        "identity": dict(identity),
        "metrics": {},
        "sidecars": [],
        "extensions": {},
    }


def collect_rows(
    row_dir: str | Path,
    specs: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    *,
    fingerprints: Mapping[str, Mapping[str, Any]] | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Load every expected row, replacing missing or malformed files explicitly."""
    root = Path(row_dir)
    rows: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: item["row_id"]):
        row_id = spec["row_id"]
        path = root / f"{row_id}.json"
        if not path.is_file():
            rows.append(_placeholder(row_id, identity, "missing expected row file"))
            continue
        try:
            row = load_strict_json(path)
            if not isinstance(row, dict):
                raise ValueError("row payload is not an object")
            if row.get("row_id") != row_id:
                raise ValueError("row ID does not match filename")
            if row.get("identity") != identity:
                raise ValueError("row identity does not match artifact identity")
            if fingerprints is not None:
                validate_persisted_row(
                    row,
                    spec,
                    identity,
                    fingerprints[row_id],
                    repo_root=repo_root or root.parent,
                )
            rows.append(row)
        except (KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
            rows.append(_placeholder(row_id, identity, f"malformed row file: {exc}"))
    return rows


def paired_seed_block_bootstrap(
    product_seed_means: Sequence[float],
    mse_seed_means: Sequence[float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, object]:
    """Percentile CI over paired seed-block mean differences."""
    product = np.asarray(product_seed_means, dtype=np.float64)
    mse = np.asarray(mse_seed_means, dtype=np.float64)
    if product.ndim != 1 or product.shape != mse.shape or product.size < 2:
        raise ValueError("paired bootstrap requires equally shaped seed vectors")
    if not np.all(np.isfinite(product)) or not np.all(np.isfinite(mse)):
        raise ValueError("paired bootstrap requires finite seed means")
    differences = product - mse
    generator = np.random.Generator(np.random.PCG64(seed))
    sampled = generator.integers(
        0, differences.size, size=(iterations, differences.size)
    )
    bootstrap_means = differences[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975), method="linear")
    return {
        "method": "paired_seed_block_percentile_bootstrap",
        "seed_block_count": int(differences.size),
        "iterations": iterations,
        "seed": seed,
        "product_minus_mse_mean": float(differences.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
    }


def _dotted_metric(metrics: Mapping[str, Any], path: str) -> float | None:
    current: object = metrics
    for component in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(component)
    if (
        isinstance(current, (int, float))
        and not isinstance(current, bool)
        and math.isfinite(current)
    ):
        return float(current)
    return None


def _student_t_seed_summary(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("across-seed summary requires finite seed means")
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1))
    sem = standard_deviation / math.sqrt(array.size)
    critical = float(student_t.ppf(0.975, df=array.size - 1))
    return {
        "count": int(array.size),
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "sem": sem,
        "student_t_95_ci": [
            mean - critical * sem,
            mean + critical * sem,
        ],
    }


def aggregate_seed_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    specs: Sequence[Mapping[str, Any]] = (),
    top_k: Sequence[int] = (),
    expected_seed_count: int = 16,
) -> list[dict[str, Any]]:
    """Summarize per-seed row means, never within-seed observation counts."""
    grouped: dict[tuple[object, ...], dict[int, float]] = defaultdict(dict)
    synthetic_metric_paths = (
        "mean_squared_l2_error",
        "relative_squared_error",
        "queries.independent.rmse",
        "queries.independent.signed_bias",
        "queries.high_inner_product.rmse",
        "queries.high_inner_product.signed_bias",
        "finite_float32_residual_expectation.independent.signed_bias",
        "finite_float32_residual_expectation.high_inner_product.signed_bias",
    )
    for spec in specs:
        paths = (
            tuple(f"retrieval.1@{value}" for value in top_k)
            if spec["dataset"] == "glove200"
            else synthetic_metric_paths[
                : (
                    len(synthetic_metric_paths)
                    if spec["codec"] == "product"
                    else len(synthetic_metric_paths) - 2
                )
            ]
        )
        for metric_path in paths:
            key = (
                spec.get("dataset"),
                spec.get("distribution"),
                spec.get("dimension"),
                spec.get("codec"),
                spec.get("nominal_total_bits"),
                metric_path,
            )
            grouped[key]
    for row in rows:
        if row.get("status") != "passed":
            continue
        extensions = row.get("extensions")
        spec = extensions.get("row_spec", {}) if isinstance(extensions, Mapping) else {}
        if not isinstance(spec, Mapping):
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        paths: Iterable[str]
        if spec.get("dataset") == "glove200":
            retrieval = metrics.get("retrieval")
            paths = (
                (f"retrieval.{key}" for key in sorted(retrieval))
                if isinstance(retrieval, Mapping)
                else ()
            )
        elif spec.get("dataset") == "synthetic":
            paths = synthetic_metric_paths
        else:
            continue
        for metric_path in paths:
            value = _dotted_metric(metrics, metric_path)
            pair_id = spec.get("pair_id")
            if value is None or not isinstance(pair_id, int):
                continue
            key = (
                spec.get("dataset"),
                spec.get("distribution"),
                spec.get("dimension"),
                spec.get("codec"),
                spec.get("nominal_total_bits"),
                metric_path,
            )
            if pair_id in grouped[key]:
                raise ValueError("duplicate seed row in across-seed aggregate")
            grouped[key][pair_id] = value
    result: list[dict[str, Any]] = []
    for key, seed_values in sorted(
        grouped.items(), key=lambda item: tuple(str(part) for part in item[0])
    ):
        pair_ids = sorted(seed_values)
        complete = len(pair_ids) == expected_seed_count and pair_ids == list(
            range(expected_seed_count)
        )
        aggregate = {
            "dataset": key[0],
            "distribution": key[1],
            "dimension": key[2],
            "codec": key[3],
            "nominal_total_bits": key[4],
            "metric": key[5],
            "expected_seed_count": expected_seed_count,
            "seed_count": len(pair_ids),
            "status": "complete" if complete else "incomplete",
        }
        if complete:
            aggregate["statistics"] = _student_t_seed_summary(
                [seed_values[pair_id] for pair_id in pair_ids]
            )
        result.append(aggregate)
    return result


def aggregate_product_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    iterations: int,
    specs: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    metric_paths = (
        ("mean_squared_l2_error", ("mean_squared_l2_error",)),
        ("queries.independent.rmse", ("queries", "independent", "rmse")),
        (
            "queries.high_inner_product.rmse",
            ("queries", "high_inner_product", "rmse"),
        ),
    )
    groups: dict[tuple[object, ...], dict[str, dict[int, float]]] = defaultdict(
        lambda: {"mse": {}, "product": {}}
    )
    synthetic_contracts = {
        (
            spec.get("distribution"),
            spec.get("dimension"),
            spec.get("nominal_total_bits"),
        )
        for spec in specs
        if spec.get("dataset") == "synthetic"
    }
    mse_contracts = {
        contract
        for contract in synthetic_contracts
        if any(
            spec.get("dataset") == "synthetic"
            and spec.get("codec") == "mse"
            and (
                spec.get("distribution"),
                spec.get("dimension"),
                spec.get("nominal_total_bits"),
            )
            == contract
            for spec in specs
        )
    }
    product_contracts = {
        contract
        for contract in synthetic_contracts
        if any(
            spec.get("dataset") == "synthetic"
            and spec.get("codec") == "product"
            and (
                spec.get("distribution"),
                spec.get("dimension"),
                spec.get("nominal_total_bits"),
            )
            == contract
            for spec in specs
        )
    }
    for contract in mse_contracts & product_contracts:
        for metric_name, _ in metric_paths:
            groups[(*contract, metric_name)]
    for row in rows:
        if row.get("status") != "passed":
            continue
        spec = row.get("extensions", {}).get("row_spec", {})
        if spec.get("dataset") != "synthetic":
            continue
        for metric_name, components in metric_paths:
            metric: object = row.get("metrics", {})
            for component in components:
                if not isinstance(metric, Mapping):
                    metric = None
                    break
                metric = metric.get(component)
            if isinstance(metric, (int, float)) and math.isfinite(metric):
                key = (
                    spec.get("distribution"),
                    spec.get("dimension"),
                    spec.get("nominal_total_bits"),
                    metric_name,
                )
                groups[key][spec["codec"]][spec["pair_id"]] = float(metric)
    result: list[dict[str, Any]] = []
    for key, codecs in sorted(groups.items()):
        pair_ids = sorted(set(codecs["mse"]) & set(codecs["product"]))
        complete = len(pair_ids) == 16 and pair_ids == list(range(16))
        comparison: dict[str, Any] = {
            "distribution": key[0],
            "dimension": key[1],
            "nominal_total_bits": key[2],
            "metric": key[3],
            "expected_seed_count": 16,
            "seed_block_count": len(pair_ids),
            "status": "complete" if complete else "incomplete",
        }
        if complete:
            comparison.update(
                paired_seed_block_bootstrap(
                    [codecs["product"][pair_id] for pair_id in pair_ids],
                    [codecs["mse"][pair_id] for pair_id in pair_ids],
                    seed=seed,
                    iterations=iterations,
                )
            )
        result.append(
            {
                **comparison,
            }
        )
    return result


def glove_selection_ids(
    glove_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    base_spec = glove_config["base_selection"]
    if base_spec["algorithm"] == "literal_sorted_ids":
        base = np.asarray(base_spec["ids"], dtype="<u4")
    else:
        train_count = glove_config["datasets"]["train"]["shape"][0]
        generator = np.random.Generator(np.random.PCG64(base_spec["seed"]))
        base = np.sort(
            generator.choice(
                train_count,
                base_spec["count"],
                replace=False,
            )
        ).astype("<u4")
    query_spec = glove_config["query_selection"]
    query = np.arange(query_spec["start"], query_spec["stop"], dtype="<u4")
    return base, query


def _json_attribute(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def validate_glove_file(
    path: str | Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the pinned file before opening its datasets read-only."""
    source = Path(path)
    actual_size = source.stat().st_size
    if actual_size != expected["size_bytes"]:
        raise ValueError(
            f"GloVe size mismatch: expected {expected['size_bytes']}, "
            f"got {actual_size}"
        )
    actual_sha = sha256_file(source)
    if actual_sha != expected["sha256"]:
        raise ValueError(
            f"GloVe SHA-256 mismatch: expected {expected['sha256']}, "
            f"got {actual_sha}"
        )
    try:
        import h5py
    except ImportError as exc:
        raise ModuleNotFoundError("h5py is required for GloVe retrieval") from exc
    metadata: dict[str, Any] = {
        "size_bytes": actual_size,
        "sha256": actual_sha,
        "datasets": {},
        "attributes": {},
    }
    with h5py.File(source, "r") as handle:
        metadata["attributes"] = {
            key: _json_attribute(value) for key, value in handle.attrs.items()
        }
        for name, declaration in expected["datasets"].items():
            if name not in handle:
                raise ValueError(f"GloVe dataset is missing: {name}")
            dataset = handle[name]
            actual_shape = list(dataset.shape)
            if actual_shape != declaration["shape"]:
                raise ValueError(
                    f"GloVe {name} shape mismatch: expected "
                    f"{declaration['shape']}, got {actual_shape}"
                )
            metadata["datasets"][name] = {
                "shape": actual_shape,
                "dtype": str(dataset.dtype),
                "chunks": list(dataset.chunks) if dataset.chunks else None,
                "compression": dataset.compression,
            }
    return metadata


def _read_h5_rows(dataset: Any, ids: np.ndarray, *, read_chunk_size: int) -> np.ndarray:
    if ids.ndim != 1 or np.any(ids[:-1] >= ids[1:]):
        raise ValueError("HDF5 row IDs must be strictly increasing")
    output = np.empty((len(ids), dataset.shape[1]), dtype=dataset.dtype)
    for output_start in range(0, len(ids), read_chunk_size):
        output_stop = min(output_start + read_chunk_size, len(ids))
        selected = ids[output_start:output_stop].astype(np.int64, copy=False)
        source_start = int(selected[0])
        source_stop = int(selected[-1]) + 1
        if source_stop - source_start <= 4 * len(selected):
            span = np.empty(
                (source_stop - source_start, dataset.shape[1]),
                dtype=dataset.dtype,
            )
            dataset.read_direct(
                span,
                np.s_[source_start:source_stop, :],
                np.s_[:, :],
            )
            output[output_start:output_stop] = span[selected - source_start]
        else:
            # Bounded, sorted fancy selections avoid h5py's huge point lists.
            output[output_start:output_stop] = dataset[selected]
    return output


def load_normalized_glove_arrays(
    path: str | Path,
    base_ids: np.ndarray,
    query_ids: np.ndarray,
    *,
    read_chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:
        raise ModuleNotFoundError("h5py is required for GloVe retrieval") from exc
    with h5py.File(Path(path), "r") as handle:
        base = _read_h5_rows(handle["train"], base_ids, read_chunk_size=read_chunk_size)
        query = _read_h5_rows(
            handle["test"], query_ids, read_chunk_size=read_chunk_size
        )
    return _normalize_rows(base), _normalize_rows(query)


def _retrieval_device(requested: str) -> torch.device:
    if requested not in {"cpu", "cuda"}:
        raise ValueError("retrieval device must be cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA retrieval scoring was requested but is unavailable")
    return torch.device(requested)


def _stream_top_k(
    base: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    chunk_size: int,
    device: str,
    transform: Any | None = None,
) -> tuple[np.ndarray, float]:
    torch_device = _retrieval_device(device)
    query_tensor = torch.from_numpy(np.asarray(queries, dtype=np.float32)).to(
        torch_device
    )
    best_values: torch.Tensor | None = None
    best_ids: torch.Tensor | None = None
    scoring_seconds = 0.0
    for start in range(0, len(base), chunk_size):
        stop = min(start + chunk_size, len(base))
        block = np.asarray(base[start:stop], dtype=np.float32)
        if transform is not None:
            block = transform(block)
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        scoring_start = time.perf_counter()
        block_tensor = torch.from_numpy(np.ascontiguousarray(block)).to(torch_device)
        scores = query_tensor @ block_tensor.T
        local_k = min(k, stop - start)
        values, indices = torch.topk(scores, k=local_k, dim=1)
        indices = indices + start
        if best_values is None:
            best_values, best_ids = values, indices
        else:
            candidate_values = torch.cat((best_values, values), dim=1)
            candidate_ids = torch.cat((best_ids, indices), dim=1)
            keep = min(k, candidate_values.shape[1])
            best_values, positions = torch.topk(candidate_values, k=keep, dim=1)
            best_ids = torch.gather(candidate_ids, 1, positions)
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        scoring_seconds += time.perf_counter() - scoring_start
    if best_ids is None:
        raise ValueError("retrieval base must be non-empty")
    return best_ids.cpu().numpy(), scoring_seconds


def exact_top_one(
    base: np.ndarray,
    queries: np.ndarray,
    *,
    chunk_size: int,
    device: str,
) -> np.ndarray:
    ranked, _ = _stream_top_k(
        base,
        queries,
        k=1,
        chunk_size=chunk_size,
        device=device,
    )
    return ranked[:, 0].astype("<u4")


def one_at_k(
    true_top_one: np.ndarray,
    approximate_ranked_ids: np.ndarray,
    top_k: Sequence[int],
) -> dict[str, float]:
    truth = np.asarray(true_top_one).reshape(-1)
    ranked = np.asarray(approximate_ranked_ids)
    if ranked.ndim != 2 or ranked.shape[0] != truth.shape[0]:
        raise ValueError("ranked IDs must have one row per exact top-one ID")
    result: dict[str, float] = {}
    for k in top_k:
        if (
            not isinstance(k, int)
            or isinstance(k, bool)
            or not 1 <= k <= ranked.shape[1]
        ):
            raise ValueError("each k must fit within the approximate ranking")
        hits = np.any(ranked[:, :k] == truth[:, None], axis=1)
        result[f"1@{k}"] = float(hits.mean())
    return result


def _codec(
    codec_name: str,
    dimension: int,
    bits: int,
    rotation_seed: int,
    projection_seed: int | None,
) -> TurboQuantMSEReference | TurboQuantProductReference:
    key = (codec_name, dimension, bits, rotation_seed, projection_seed)
    cached = _CODEC_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    if codec_name == "mse":
        value: TurboQuantMSEReference | TurboQuantProductReference = (
            TurboQuantMSEReference(TurboQuantMSEConfig(dimension, bits, rotation_seed))
        )
    elif codec_name == "product":
        if projection_seed is None:
            raise ValueError("product codec requires a projection seed")
        value = TurboQuantProductReference(
            TurboQuantProductConfig(dimension, bits, rotation_seed, projection_seed)
        )
    else:
        raise ValueError(f"unknown codec: {codec_name}")
    _CODEC_CACHE[key] = value
    return value


@dataclass
class _CodecMeasurements:
    squared_error: _Moments
    relative_error: _Moments
    query_errors: dict[str, _Moments]
    expectation_errors: dict[str, _Moments]
    actual_bytes: int = 0
    encode_seconds: float = 0.0
    decode_seconds: float = 0.0


def _new_measurements(
    codec_name: str, *, include_query_metrics: bool = True
) -> _CodecMeasurements:
    return _CodecMeasurements(
        squared_error=_Moments(),
        relative_error=_Moments(),
        query_errors=(
            {
                "independent": _Moments(),
                "high_inner_product": _Moments(),
            }
            if include_query_metrics
            else {}
        ),
        expectation_errors=(
            {
                "independent": _Moments(),
                "high_inner_product": _Moments(),
            }
            if codec_name == "product" and include_query_metrics
            else {}
        ),
    )


def _measure_codec_block(
    codec: TurboQuantMSEReference | TurboQuantProductReference,
    codec_name: str,
    vectors: np.ndarray,
    queries: Mapping[str, np.ndarray] | None,
    measurements: _CodecMeasurements,
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(vectors))
    encode_start = time.perf_counter()
    encoded = codec.encode(tensor)
    measurements.encode_seconds += time.perf_counter() - encode_start
    decode_start = time.perf_counter()
    decoded = codec.decode(encoded)
    measurements.decode_seconds += time.perf_counter() - decode_start
    measurements.actual_bytes += codec.physical_nbytes(encoded)

    source = tensor.to(torch.float64)
    difference = source - decoded
    squared = difference.square().sum(dim=-1)
    denominator = source.square().sum(dim=-1)
    measurements.squared_error.add(squared)
    measurements.relative_error.add(squared / denominator)
    if queries is not None:
        for name, values in queries.items():
            query = torch.from_numpy(np.ascontiguousarray(values)).to(torch.float64)
            truth = (query * source).sum(dim=-1)
            estimate = (query * decoded).sum(dim=-1)
            measurements.query_errors[name].add(estimate - truth)
    if codec_name == "product" and queries is not None:
        assert isinstance(codec, TurboQuantProductReference)
        assert measurements.expectation_errors
        mse_decoded = codec._mse.decode(encoded.mse)
        for name, values in queries.items():
            query = torch.from_numpy(np.ascontiguousarray(values))
            expected = product_float32_residual_expectation(
                source,
                mse_decoded,
                encoded.residual_norm.reshape(-1),
                query,
            )
            approximate = (query.to(torch.float64) * decoded).sum(dim=-1)
            measurements.expectation_errors[name].add(approximate - expected)
    return decoded.to(torch.float32).numpy()


def _metrics_from_measurements(
    measurements: _CodecMeasurements,
    *,
    codec_name: str,
    dimension: int,
    bits: int,
    count: int,
) -> dict[str, Any]:
    formula_bytes = physical_bytes_formula(codec_name, dimension, bits, count)
    if measurements.actual_bytes != formula_bytes:
        raise ValueError(
            "physical byte mismatch: "
            f"actual={measurements.actual_bytes}, formula={formula_bytes}"
        )
    metrics: dict[str, Any] = {
        "sample_count": count,
        "dimension": dimension,
        "mean_squared_l2_error": measurements.squared_error.mean(),
        "relative_squared_error": measurements.relative_error.mean(),
        "encode_latency_ms": measurements.encode_seconds * 1_000,
        "decode_latency_ms": measurements.decode_seconds * 1_000,
        "encode_latency_us_per_vector": (
            measurements.encode_seconds * 1_000_000 / count
        ),
        "decode_latency_us_per_vector": (
            measurements.decode_seconds * 1_000_000 / count
        ),
        "actual_physical_bytes": measurements.actual_bytes,
        "formula_physical_bytes": formula_bytes,
        "effective_physical_bits_per_coordinate": (
            measurements.actual_bytes * 8 / (count * dimension)
        ),
        "nominal_total_bits_per_coordinate": bits,
    }
    if measurements.query_errors:
        metrics["queries"] = {
            name: values.signed_summary()
            for name, values in measurements.query_errors.items()
        }
    if measurements.expectation_errors:
        metrics["finite_float32_residual_expectation"] = {
            name: values.signed_summary()
            for name, values in measurements.expectation_errors.items()
        }
    return metrics


def _run_synthetic_row(
    spec: Mapping[str, Any],
    identity: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    timing_class: str,
) -> dict[str, Any]:
    codec = _codec(
        spec["codec"],
        spec["dimension"],
        spec["nominal_total_bits"],
        spec["rotation_seed"],
        spec["projection_seed"],
    )
    stream = _synthetic_stream(
        distribution=spec["distribution"],
        dimension=spec["dimension"],
        data_seed=spec["data_seed"],
        independent_query_seed=spec["independent_query_seed"],
        high_inner_product_noise_seed=spec["high_inner_product_noise_seed"],
    )
    measurements = _new_measurements(spec["codec"])
    processed = 0
    while processed < spec["vector_count"]:
        block_count = min(spec["chunk_size"], spec["vector_count"] - processed)
        fixture = stream.take(block_count)
        _measure_codec_block(
            codec,
            spec["codec"],
            fixture["vectors"],
            {
                "independent": fixture["independent_queries"],
                "high_inner_product": fixture["high_inner_product_queries"],
            },
            measurements,
        )
        processed += block_count
    metrics = _metrics_from_measurements(
        measurements,
        codec_name=spec["codec"],
        dimension=spec["dimension"],
        bits=spec["nominal_total_bits"],
        count=spec["vector_count"],
    )
    return {
        "row_id": spec["row_id"],
        "status": "passed",
        "reason": None,
        "identity": dict(identity),
        "metrics": metrics,
        "sidecars": [],
        "extensions": {
            "row_spec": dict(spec),
            "resume_fingerprint": dict(fingerprint),
            "timing_class": timing_class,
        },
    }


def score_glove_codec(
    base: np.ndarray,
    queries: np.ndarray,
    exact_top_one_ids: np.ndarray,
    *,
    codec_name: str,
    nominal_total_bits: int,
    rotation_seed: int,
    projection_seed: int,
    chunk_size: int,
    top_k: Sequence[int],
    device: str,
) -> dict[str, Any]:
    codec = _codec(
        codec_name,
        base.shape[1],
        nominal_total_bits,
        rotation_seed,
        projection_seed if codec_name == "product" else None,
    )
    measurements = _new_measurements(codec_name, include_query_metrics=False)

    def transform(block: np.ndarray) -> np.ndarray:
        return _measure_codec_block(
            codec,
            codec_name,
            block,
            None,
            measurements,
        )

    ranked, scoring_seconds = _stream_top_k(
        base,
        queries,
        k=max(top_k),
        chunk_size=chunk_size,
        device=device,
        transform=transform,
    )
    metrics = _metrics_from_measurements(
        measurements,
        codec_name=codec_name,
        dimension=base.shape[1],
        bits=nominal_total_bits,
        count=base.shape[0],
    )
    metrics.update(
        {
            "retrieval": one_at_k(exact_top_one_ids, ranked, top_k),
            "retrieval_scoring_latency_ms": scoring_seconds * 1_000,
            "retrieval_scoring_device": device,
            "retrieval_scoring_chunk_size": chunk_size,
            "retrieval_algorithm": ("exact_chunked_brute_force_dot_product"),
            "scoring_device": device,
            "full_score_matrix_materialized": False,
            "maximum_score_block": {
                "rows": int(queries.shape[0]),
                "columns": min(chunk_size, int(base.shape[0])),
            },
            "gpu_indexing_latency_ms": None,
        }
    )
    return {"ranked_ids": ranked, "metrics": metrics}


def _sidecar(path: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return {}
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "available": True,
    }


def _run_glove_row(
    spec: Mapping[str, Any],
    identity: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    base: np.ndarray,
    queries: np.ndarray,
    exact_ids: np.ndarray,
    top_k: Sequence[int],
    chunk_size: int,
    device: str,
    sidecars: Sequence[Mapping[str, Any]],
    timing_class: str,
) -> dict[str, Any]:
    scored = score_glove_codec(
        base,
        queries,
        exact_ids,
        codec_name=spec["codec"],
        nominal_total_bits=spec["nominal_total_bits"],
        rotation_seed=spec["rotation_seed"],
        projection_seed=spec["projection_seed"] or 0,
        chunk_size=chunk_size,
        top_k=top_k,
        device=device,
    )
    return {
        "row_id": spec["row_id"],
        "status": "passed",
        "reason": None,
        "identity": dict(identity),
        "metrics": scored["metrics"],
        "sidecars": list(sidecars),
        "extensions": {
            "row_spec": dict(spec),
            "resume_fingerprint": dict(fingerprint),
            "timing_class": timing_class,
            "retrieval_scoring": {
                "device": device,
                "base_chunk_size": chunk_size,
                "codec_encode_decode_device": "cpu",
                "gpu_indexing_performed": False,
                "retrieval_algorithm": ("exact_chunked_brute_force_dot_product"),
                "scoring_device": device,
                "full_score_matrix_materialized": False,
                "maximum_score_block": {
                    "rows": int(queries.shape[0]),
                    "columns": min(chunk_size, int(base.shape[0])),
                },
            },
        },
    }


def _unavailable_or_invalid_row(
    spec: Mapping[str, Any],
    identity: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    timing_class: str,
) -> dict[str, Any]:
    return {
        "row_id": spec["row_id"],
        "status": status,
        "reason": reason,
        "identity": dict(identity),
        "metrics": {},
        "sidecars": [],
        "extensions": {
            "row_spec": dict(spec),
            "resume_fingerprint": dict(fingerprint),
            "timing_class": timing_class,
        },
    }


def _download_atomic(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _gpu_conditions() -> dict[str, object]:
    unavailable: dict[str, object] = {
        "temperature_c": "unavailable",
        "power_draw_w": "unavailable",
        "competing_compute_processes": "unavailable",
    }
    try:
        sample = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first = sample.stdout.strip().splitlines()[0].split(",")
        unavailable["temperature_c"] = float(first[0].strip())
        unavailable["power_draw_w"] = float(first[1].strip())
    except (
        FileNotFoundError,
        IndexError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        pass
    try:
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        unavailable["competing_compute_processes"] = [
            line.strip() for line in processes.stdout.splitlines() if line.strip()
        ]
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        pass
    return unavailable


def _execution_class(config: Mapping[str, Any]) -> dict[str, Any]:
    cuda = capture_cuda_metadata()
    cpu_model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "name": config["execution_class"],
        "gpu_name": cuda["gpu_name"] or "not_available",
        "total_vram_bytes": cuda["gpu_total_vram_bytes"] or 1,
        "driver_version": cuda["driver_version"] or "not_available",
        "cuda_runtime": cuda["cuda_runtime"] or "not_available",
        "extensions": {
            "cuda_available": bool(cuda["cuda_available"]),
            "environment": capture_env_metadata(),
            "h5py_version": package_version("h5py"),
            "runtime_packages": {
                name: package_version(name)
                for name in ("h5py", "numpy", "scipy", "torch")
            },
            "runtime_lock_sha256": sha256_file(ROOT / "requirements" / "runtime.lock"),
            "cpu_model": cpu_model,
            "operating_system": platform.platform(),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {
                name: os.environ.get(name, "unavailable")
                for name in (
                    "MALLOC_ARENA_MAX",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "PYTORCH_CUDA_ALLOC_CONF",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "gpu_conditions": _gpu_conditions(),
            "timing_policy": {
                "codec_setup_excluded": True,
                "fixture_generation_excluded": True,
                "base_device_transfer_in_retrieval_scoring": True,
                "query_device_transfer_in_retrieval_scoring": False,
            },
        },
    }


def _invocation_metadata(
    *,
    config_path: Path,
    output_dir: Path,
    dataset_cache: Path,
    allow_download: bool,
    retrieval_device: str,
    retrieval_chunk_size: int,
    argv: Sequence[str] | None,
) -> dict[str, object]:
    normalized = [
        sys.executable,
        str((ROOT / "scripts" / "run_codec_reproduction.py").resolve()),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--dataset-cache",
        str(dataset_cache),
        "--retrieval-device",
        retrieval_device,
        "--retrieval-base-chunk-size",
        str(retrieval_chunk_size),
    ]
    if not allow_download:
        normalized.append("--no-download")
    return {
        "argv": list(argv) if argv is not None else normalized,
        "cwd": str(Path.cwd().resolve()),
        "normalized_command": normalized,
    }


def _source_metadata() -> dict[str, Any]:
    source = capture_source_metadata(ROOT)
    if source["git_commit"] != "unknown":
        return source
    root_text = str(ROOT)
    command = "git"
    if ROOT.as_posix().startswith("/mnt/"):
        parts = ROOT.as_posix().split("/")
        drive = parts[2].upper()
        root_text = drive + ":\\" + "\\".join(parts[3:])
        command = "git.exe"
    base = [command, "-c", "safe.directory=*", "-C", root_text]
    try:
        commit = subprocess.check_output(
            [*base, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            [*base, "status", "--porcelain", "--untracked-files=normal"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return source
    return {"git_commit": commit, "dirty": bool(status.strip())}


def _paper(config: Mapping[str, Any]) -> dict[str, Any]:
    reference = ROOT / config["paper_reference"]
    payload = load_strict_json(reference)
    return {
        key: payload[key]
        for key in ("paper_id", "source_url", "retrieved_at", "sha256")
    }


def build_manifest(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    expected_row_ids: Sequence[str],
    source: Mapping[str, Any],
    execution_class: Mapping[str, Any],
    paper: Mapping[str, Any],
    retrieval_device: str = "cpu",
    retrieval_chunk_size: int = DEFAULT_RETRIEVAL_CHUNK_SIZE,
    claims: Sequence[Mapping[str, Any]] = (),
    across_seed_statistics: Sequence[Mapping[str, Any]] | None = None,
    product_comparisons: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    counts = Counter(row["status"] for row in rows)
    statistics = config["statistics"]
    comparisons = (
        list(product_comparisons)
        if product_comparisons is not None
        else aggregate_product_comparisons(
            rows,
            seed=statistics["seed"],
            iterations=statistics["iterations"],
        )
    )
    across_seeds = (
        list(across_seed_statistics)
        if across_seed_statistics is not None
        else aggregate_seed_statistics(rows, expected_seed_count=16)
    )
    glove = config["glove200"]
    query_count = glove["query_selection"]["stop"] - glove["query_selection"]["start"]
    base_count = glove["base_selection"]["count"]
    return {
        "artifact_type": "codec_reproduction",
        "schema_version": 2,
        "artifact_id": config["artifact_id"],
        "status": artifact_status(
            (row["status"] for row in rows),
            dirty_source=bool(source["dirty"]),
        ),
        "paper": dict(paper),
        "source": dict(source),
        "execution_class": dict(execution_class),
        "identity": codec_identity(config_path),
        "expected_row_ids": sorted(expected_row_ids),
        "statistics": {
            "method": statistics["method"],
            "confidence_level": statistics["confidence_level"],
            "seed": statistics["seed"],
            "iterations": statistics["iterations"],
            "extensions": {
                "bootstrap_interval": "paired_seed_block_percentile_linear",
                "student_t_degrees_of_freedom": "actual_n_minus_1",
            },
        },
        "rows": sorted(rows, key=lambda row: row["row_id"]),
        "pass_state_counts": {state: counts[state] for state in V2_ROW_STATES},
        "claims": list(claims),
        "extensions": {
            "config_sha256": sha256_file(config_path),
            "claim_labels": list(config["claim_labels"]),
            "timing_class": config["timing_class"],
            "across_seed_statistics": across_seeds,
            "product_minus_mse_comparisons": comparisons,
            "retrieval_policy": {
                "device": retrieval_device,
                "base_chunk_size": retrieval_chunk_size,
                "codec_encode_decode_device": "cpu",
                "gpu_indexing_performed": False,
                "retrieval_algorithm": ("exact_chunked_brute_force_dot_product"),
                "scoring_device": retrieval_device,
                "full_score_matrix_materialized": False,
                "maximum_score_block": {
                    "rows": query_count,
                    "columns": min(retrieval_chunk_size, base_count),
                },
            },
        },
    }


def _input_hashes(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    glove_actual_sha256: str | None,
) -> dict[str, str]:
    if spec["dataset"] == "synthetic":
        fixture = {
            key: spec[key]
            for key in (
                "distribution",
                "dimension",
                "fixture_id",
                "data_seed",
                "independent_query_seed",
                "high_inner_product_noise_seed",
                "vector_count",
                "chunk_size",
            )
        }
        return {"synthetic_fixture_sha256": _sha256_json(fixture)}
    glove = config["glove200"]
    return {
        "dataset_sha256": glove_actual_sha256 or glove["sha256"],
        "base_ids_sha256": glove["base_selection"]["sha256"],
        "query_ids_sha256": glove["query_selection"]["sha256"],
    }


def _fingerprint(
    spec: Mapping[str, Any],
    config_path: Path,
    input_hashes: Mapping[str, str],
    *,
    source: Mapping[str, Any],
    execution_class: Mapping[str, Any],
    paper: Mapping[str, Any],
    invocation: Mapping[str, Any],
    retrieval_device: str,
    retrieval_chunk_size: int,
    sidecar_paths: Sequence[str] = (),
) -> dict[str, Any]:
    config = load_config(config_path)
    execution_extensions = execution_class.get("extensions", {})
    stable_execution = {
        key: execution_class[key]
        for key in (
            "name",
            "gpu_name",
            "total_vram_bytes",
            "driver_version",
            "cuda_runtime",
        )
    }
    if isinstance(execution_extensions, Mapping):
        environment = execution_extensions.get("environment")
        stable_environment = (
            {
                key: value
                for key, value in environment.items()
                if key not in {"timestamp_utc", "gpu_free_vram_bytes"}
            }
            if isinstance(environment, Mapping)
            else environment
        )
        stable_execution["extensions"] = {
            key: execution_extensions.get(key)
            for key in (
                "runtime_packages",
                "runtime_lock_sha256",
                "cpu_model",
                "operating_system",
                "torch_intraop_threads",
                "torch_interop_threads",
                "thread_environment",
                "timing_policy",
            )
        }
        stable_execution["extensions"]["environment"] = stable_environment
    glove = config["glove200"]
    metric_contract = {
        "sample_count": (
            spec["vector_count"]
            if spec["dataset"] == "synthetic"
            else glove["base_selection"]["count"]
        ),
        "query_count": (
            spec["vector_count"]
            if spec["dataset"] == "synthetic"
            else glove["query_selection"]["stop"] - glove["query_selection"]["start"]
        ),
        "top_k": list(glove["top_k"]) if spec["dataset"] == "glove200" else [],
        "sidecar_paths": list(sidecar_paths),
    }
    return {
        "protocol": RESUME_PROTOCOL,
        "identity": codec_identity(config_path),
        "config_sha256": sha256_file(config_path),
        "row_spec_sha256": _sha256_json(spec),
        "input_sha256": dict(input_hashes),
        "behavior_sha256": _behavior_sha256(),
        "source": dict(source),
        "paper": dict(paper),
        "execution_class": stable_execution,
        "timing_class": config["timing_class"],
        "metric_contract": metric_contract,
        "retrieval_policy": {
            "device": retrieval_device,
            "base_chunk_size": retrieval_chunk_size,
        },
    }


def _load_resumable(
    path: Path,
    spec: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        row = load_strict_json(path)
    except (KeyError, OSError, TypeError, UnicodeError, ValueError):
        return None
    if isinstance(row, dict) and can_resume_row(row, fingerprint):
        try:
            validate_persisted_row(
                row,
                spec,
                identity,
                fingerprint,
                repo_root=repo_root,
            )
        except ValueError:
            return None
        return row
    return None


def _prepare_glove(
    config: Mapping[str, Any],
    dataset_cache: Path,
    output_dir: Path,
    *,
    allow_download: bool,
    retrieval_device: str,
    retrieval_chunk_size: int,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    glove = config["glove200"]
    path = dataset_cache / glove["filename"]
    if not path.is_file():
        if not allow_download:
            return {"state": "unavailable", "reason": "verified GloVe file unavailable"}
        try:
            _download_atomic(glove["url"], path)
        except (OSError, urllib.error.URLError) as exc:
            return {
                "state": "unavailable",
                "reason": f"GloVe download unavailable: {exc}",
            }
    try:
        metadata = validate_glove_file(path, glove)
    except (MemoryError, torch.OutOfMemoryError) as exc:
        return {
            "state": "oom",
            "reason": f"GloVe validation out of memory: {exc}",
        }
    except ModuleNotFoundError as exc:
        return {"state": "unavailable", "reason": str(exc)}
    except (OSError, ValueError) as exc:
        return {"state": "invalid", "reason": str(exc)}
    base_ids, query_ids = glove_selection_ids(glove)
    try:
        base, queries = load_normalized_glove_arrays(path, base_ids, query_ids)
    except (MemoryError, torch.OutOfMemoryError) as exc:
        return {
            "state": "oom",
            "reason": f"GloVe load out of memory: {exc}",
        }
    except (OSError, ValueError, ModuleNotFoundError) as exc:
        return {"state": "invalid", "reason": f"GloVe computation error: {exc}"}
    try:
        exact = exact_top_one(
            base,
            queries,
            chunk_size=retrieval_chunk_size,
            device=retrieval_device,
        )
    except (MemoryError, torch.OutOfMemoryError) as exc:
        return {
            "state": "oom",
            "reason": f"GloVe exact top1 out of memory: {exc}",
        }
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "state": "invalid",
            "reason": f"GloVe exact top1 computation error: {exc}",
        }
    exact = np.asarray(exact)
    try:
        exact_invalid = (
            exact.ndim != 1
            or not np.issubdtype(exact.dtype, np.integer)
            or exact.shape[0] != queries.shape[0]
            or np.any(exact < 0)
            or np.any(exact >= base.shape[0])
        )
    except (TypeError, ValueError):
        exact_invalid = True
    if exact_invalid:
        return {
            "state": "invalid",
            "reason": "GloVe exact top1 computation returned invalid IDs",
        }
    sidecar_values = {
        "glove200-base-ids.u32": base_ids,
        "glove200-query-ids.u32": query_ids,
        "glove200-subset-exact-top1.u32": exact,
    }
    declared: list[dict[str, Any]] = []
    try:
        for name, values in sidecar_values.items():
            sidecar_path = output_dir / name
            _atomic_write(sidecar_path, values.astype("<u4", copy=False).tobytes())
            declaration = _sidecar(
                sidecar_path,
                repo_root=repo_root or output_dir.parent,
            )
            if not declaration:
                raise ValueError("GloVe sidecar is outside the artifact root")
            declared.append(declaration)
    except (MemoryError, torch.OutOfMemoryError) as exc:
        return {
            "state": "oom",
            "reason": f"GloVe sidecar out of memory: {exc}",
        }
    except (OSError, ValueError) as exc:
        return {
            "state": "invalid",
            "reason": f"GloVe sidecar write error: {exc}",
        }
    return {
        "state": "ready",
        "metadata": metadata,
        "base": base,
        "queries": queries,
        "exact": exact,
        "sidecars": declared,
        "sha256": metadata["sha256"],
    }


def _artifact_repo_root(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    root = ROOT.resolve()
    return root if resolved.is_relative_to(root) else resolved.parent


def _summary_key(kind: str, value: Mapping[str, Any]) -> str:
    parts = [
        kind,
        value.get("dataset", "all"),
        value.get("distribution", "all"),
        value.get("dimension", "all"),
        value.get("codec", "all"),
        value.get("nominal_total_bits", "all"),
        value.get("metric", "all"),
    ]
    return "-".join(
        "".join(character if character.isalnum() else "-" for character in str(part))
        for part in parts
    ).strip("-")


def _summary_claim_paths(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    complete = any(
        value.get("status") == "complete"
        for value in summary["across_seed_statistics"].values()
    ) or any(
        value.get("status") == "complete"
        for value in summary["product_minus_mse_comparisons"].values()
    )
    if not complete:
        return []
    paths = [(f"pass-{state}", f"pass_state_counts.{state}") for state in V2_ROW_STATES]
    for aggregate_id, aggregate in summary["across_seed_statistics"].items():
        if aggregate.get("status") != "complete":
            continue
        for key in (
            "count",
            "mean",
            "sample_standard_deviation",
            "sem",
        ):
            paths.append(
                (
                    f"across-seed-{aggregate_id}-{key}",
                    f"across_seed_statistics.{aggregate_id}.statistics.{key}",
                )
            )
        for bound in range(2):
            paths.append(
                (
                    f"across-seed-{aggregate_id}-ci-{bound}",
                    "across_seed_statistics."
                    f"{aggregate_id}.statistics.student_t_95_ci.{bound}",
                )
            )
    for comparison_id, comparison in summary["product_minus_mse_comparisons"].items():
        if comparison.get("status") != "complete":
            continue
        paths.append(
            (
                f"product-comparison-{comparison_id}-mean",
                "product_minus_mse_comparisons."
                f"{comparison_id}.product_minus_mse_mean",
            )
        )
        for bound in range(2):
            paths.append(
                (
                    f"product-comparison-{comparison_id}-ci-{bound}",
                    "product_minus_mse_comparisons."
                    f"{comparison_id}.bootstrap_95_ci.{bound}",
                )
            )
    return paths


def _claim_summary_bridge(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    across_seed_statistics: Sequence[Mapping[str, Any]],
    product_comparisons: Sequence[Mapping[str, Any]],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    passed = sorted(
        (row for row in rows if row.get("status") == "passed"),
        key=lambda row: row["row_id"],
    )
    copied_rows = [deepcopy(dict(row)) for row in rows]
    if not passed:
        return copied_rows, []
    counts = Counter(row["status"] for row in rows)
    across_summary = {
        _summary_key("across", aggregate): aggregate
        for aggregate in across_seed_statistics
    }
    comparison_summary = {
        _summary_key("comparison", comparison): comparison
        for comparison in product_comparisons
    }
    summary = {
        "summary_version": 1,
        "artifact_id": config["artifact_id"],
        "pass_state_counts": {state: counts[state] for state in V2_ROW_STATES},
        "across_seed_statistics": across_summary,
        "product_minus_mse_comparisons": comparison_summary,
    }
    summary_path = output_dir / "claim-summary.json"
    summary_bytes = (
        json.dumps(
            summary,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write_if_changed(summary_path, summary_bytes)
    declaration = _sidecar(summary_path, repo_root=repo_root)
    if not declaration:
        raise ValueError("claim summary is outside the artifact root")
    anchor_id = passed[0]["row_id"]
    for index, row in enumerate(copied_rows):
        if row["row_id"] == anchor_id:
            row["sidecars"] = [*row["sidecars"], declaration]
            copied_rows[index] = seal_row(row)
            break
    relative = declaration["path"]
    claims = [
        {
            "claim_id": f"codec-local-{suffix}",
            "artifact_id": config["artifact_id"],
            "row_id": anchor_id,
            "query": f"sidecar:{relative}#{json_path}",
            "extensions": {
                "caution_labels": list(config["claim_labels"]),
            },
        }
        for suffix, json_path in _summary_claim_paths(summary)
    ]
    return copied_rows, claims


def _write_aggregate_outputs(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    config_path: Path,
    specs: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    execution_class: Mapping[str, Any],
    paper: Mapping[str, Any],
    retrieval_device: str,
    retrieval_chunk_size: int,
    fingerprints: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    rows = collect_rows(
        output_dir / "rows",
        specs,
        identity,
        fingerprints=fingerprints,
        repo_root=repo_root,
    )
    statistics = config["statistics"]
    across_seeds = aggregate_seed_statistics(
        rows,
        specs=specs,
        top_k=config["glove200"]["top_k"],
        expected_seed_count=16,
    )
    comparisons = aggregate_product_comparisons(
        rows,
        seed=statistics["seed"],
        iterations=statistics["iterations"],
        specs=specs,
    )
    manifest_rows, claims = _claim_summary_bridge(
        output_dir=output_dir,
        rows=rows,
        config=config,
        across_seed_statistics=across_seeds,
        product_comparisons=comparisons,
        repo_root=repo_root,
    )
    atomic_write_jsonl(output_dir / "rows.jsonl", manifest_rows)
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        rows=manifest_rows,
        expected_row_ids=[spec["row_id"] for spec in specs],
        source=source,
        execution_class=execution_class,
        paper=paper,
        retrieval_device=retrieval_device,
        retrieval_chunk_size=retrieval_chunk_size,
        claims=claims,
        across_seed_statistics=across_seeds,
        product_comparisons=comparisons,
    )
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def execute_reproduction(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    dataset_cache: str | Path,
    allow_download: bool = True,
    retrieval_device: str = "cpu",
    retrieval_chunk_size: int = DEFAULT_RETRIEVAL_CHUNK_SIZE,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute or resume all declared rows and atomically rebuild the artifact."""
    path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    cache = Path(dataset_cache).expanduser().resolve()
    if retrieval_chunk_size <= 0:
        raise ValueError("retrieval_chunk_size must be positive")
    _retrieval_device(retrieval_device)
    config = load_config(path)
    specs = build_row_specs(config)
    identity = codec_identity(path)
    source = _source_metadata()
    execution = _execution_class(config)
    paper = _paper(config)
    invocation = _invocation_metadata(
        config_path=path,
        output_dir=output,
        dataset_cache=cache,
        allow_download=allow_download,
        retrieval_device=retrieval_device,
        retrieval_chunk_size=retrieval_chunk_size,
        argv=argv,
    )
    execution["extensions"]["invocation"] = invocation
    repo_root = _artifact_repo_root(output)
    rows_dir = output / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    glove_specs = [spec for spec in specs if spec["dataset"] == "glove200"]
    glove_context = _prepare_glove(
        config,
        cache,
        output,
        allow_download=allow_download,
        retrieval_device=retrieval_device,
        retrieval_chunk_size=retrieval_chunk_size,
        repo_root=repo_root,
    )
    actual_glove_sha = (
        glove_context.get("sha256") if glove_context["state"] == "ready" else None
    )
    glove_sidecar_paths = [
        (output / name).resolve().relative_to(repo_root.resolve()).as_posix()
        for name in (
            "glove200-base-ids.u32",
            "glove200-query-ids.u32",
            "glove200-subset-exact-top1.u32",
        )
    ]
    fingerprints = {
        spec["row_id"]: _fingerprint(
            spec,
            path,
            _input_hashes(
                spec,
                config,
                glove_actual_sha256=actual_glove_sha,
            ),
            source=source,
            execution_class=execution,
            paper=paper,
            invocation=invocation,
            retrieval_device=retrieval_device,
            retrieval_chunk_size=retrieval_chunk_size,
            sidecar_paths=(
                glove_sidecar_paths if spec["dataset"] == "glove200" else ()
            ),
        )
        for spec in specs
    }
    try:
        for spec in specs:
            fingerprint = fingerprints[spec["row_id"]]
            row_path = rows_dir / f"{spec['row_id']}.json"
            if (
                _load_resumable(
                    row_path,
                    spec,
                    fingerprint,
                    identity,
                    repo_root=repo_root,
                )
                is not None
            ):
                continue
            try:
                if spec["dataset"] == "synthetic":
                    row = _run_synthetic_row(
                        spec,
                        identity,
                        fingerprint,
                        timing_class=config["timing_class"],
                    )
                elif glove_context["state"] == "ready":
                    row = _run_glove_row(
                        spec,
                        identity,
                        fingerprint,
                        base=glove_context["base"],
                        queries=glove_context["queries"],
                        exact_ids=glove_context["exact"],
                        top_k=config["glove200"]["top_k"],
                        chunk_size=retrieval_chunk_size,
                        device=retrieval_device,
                        sidecars=glove_context["sidecars"],
                        timing_class=config["timing_class"],
                    )
                else:
                    row = _unavailable_or_invalid_row(
                        spec,
                        identity,
                        fingerprint,
                        status=glove_context["state"],
                        reason=glove_context["reason"],
                        timing_class=config["timing_class"],
                    )
            except (MemoryError, torch.OutOfMemoryError) as exc:
                row = _unavailable_or_invalid_row(
                    spec,
                    identity,
                    fingerprint,
                    status="oom",
                    reason=f"out of memory: {exc}",
                    timing_class=config["timing_class"],
                )
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                row = _unavailable_or_invalid_row(
                    spec,
                    identity,
                    fingerprint,
                    status="invalid",
                    reason=f"computation error: {exc}",
                    timing_class=config["timing_class"],
                )
            atomic_write_json(row_path, seal_row(row))
    finally:
        manifest = _write_aggregate_outputs(
            output_dir=output,
            config=config,
            config_path=path,
            specs=specs,
            identity=identity,
            source=source,
            execution_class=execution,
            paper=paper,
            retrieval_device=retrieval_device,
            retrieval_chunk_size=retrieval_chunk_size,
            fingerprints=fingerprints,
            repo_root=repo_root,
        )
    if len(glove_specs) == 0:
        raise ValueError("config must declare at least one GloVe row")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument(
        "--retrieval-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--retrieval-base-chunk-size",
        type=int,
        default=DEFAULT_RETRIEVAL_CHUNK_SIZE,
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="record missing GloVe input as unavailable without downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = execute_reproduction(
        config_path=args.config,
        output_dir=args.output_dir,
        dataset_cache=args.dataset_cache,
        allow_download=not args.no_download,
        retrieval_device=args.retrieval_device,
        retrieval_chunk_size=args.retrieval_base_chunk_size,
        argv=[sys.executable, *sys.argv],
    )
    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "status": manifest["status"],
                "pass_state_counts": manifest["pass_state_counts"],
                "manifest": str(args.output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
