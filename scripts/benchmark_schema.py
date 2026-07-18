"""Typed, self-validating schemas for benchmark evidence artifacts.

An artifact is intentionally complete-or-invalid: validators never infer a pass
from a partial run or silently drop a failed matrix cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import re
from typing import Any, Mapping, Sequence


REQUIRED_BINDING_KEYS = (
    "model_id",
    "model_revision",
    "model_revision_sha256",
    "model_manifest_sha256",
    "profile_sha256",
    "codebook_sha256",
    "code_sha256",
    "runner_sha256",
)
_GIT_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
# ``context_target`` is the requested model-input token budget, while
# ``context_actual`` is the number of tokens handed to the model after
# tokenization. A shortfall below one percent is tolerated for tokenizer
# boundary effects; anything larger is not evidence at the requested context.
MIN_CONTEXT_FIDELITY = 0.99
PHYSICAL_BYTE_CATEGORIES = ("keys", "values", "metadata", "fp16_window", "padding")
QUALITY_ARTIFACT_TYPE = "quality_matrix"
SYSTEMS_ARTIFACT_TYPE = "systems_benchmark"
QUALIFIED_ARTIFACT_TYPE = "qualified_run"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairedBootstrap:
    """Paired baseline-minus-profile accuracy statistics in percentage points."""

    delta_pp: float
    upper_95_pp: float
    seed: int
    iterations: int
    eligible_cases: int


@dataclass(frozen=True)
class QualityRow:
    context_target: int
    context_actual: int
    depth: float
    needle_depth_actual: float
    filler_token_budget: int
    construction_mode: str
    trial_seed: int
    status: str
    needle_present_after_tokenization: bool
    oom: bool
    exception: str | None
    baseline_correct: bool | None
    profile_correct: bool | None

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "QualityRow":
        return cls(
            context_target=int(row["context_target"]),
            context_actual=int(row["context_actual"]),
            depth=float(row["depth"]),
            needle_depth_actual=float(row["needle_depth_actual"]),
            filler_token_budget=int(row["filler_token_budget"]),
            construction_mode=str(row["construction_mode"]),
            trial_seed=int(row["trial_seed"]),
            status=str(row["status"]),
            needle_present_after_tokenization=bool(
                row["needle_present_after_tokenization"]
            ),
            oom=bool(row["oom"]),
            exception=row.get("exception"),
            baseline_correct=row.get("baseline_correct"),
            profile_correct=row.get("profile_correct"),
        )


@dataclass(frozen=True)
class QualityMatrix:
    schema_version: int
    binding: dict[str, str]
    metadata: dict[str, Any]
    status: str
    expected_matrix: dict[str, list[int | float]]
    rows: tuple[QualityRow, ...]

    @classmethod
    def from_dict(cls, artifact: Mapping[str, Any]) -> "QualityMatrix":
        if artifact.get("artifact_type") != QUALITY_ARTIFACT_TYPE:
            raise ValueError("artifact_type must be quality_matrix")
        expected = artifact["expected_matrix"]
        return cls(
            schema_version=int(artifact["schema_version"]),
            binding=dict(artifact["binding"]),
            metadata=dict(artifact["metadata"]),
            status=str(artifact["status"]),
            expected_matrix={
                "contexts": list(expected["contexts"]),
                "depths": list(expected["depths"]),
                "seeds": list(expected["seeds"]),
            },
            rows=tuple(QualityRow.from_dict(row) for row in artifact["rows"]),
        )

    @property
    def expected_row_count(self) -> int:
        return (
            len(self.expected_matrix["contexts"])
            * len(self.expected_matrix["depths"])
            * len(self.expected_matrix["seeds"])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": QUALITY_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "binding": self.binding,
            "metadata": self.metadata,
            "status": self.status,
            "expected_matrix": self.expected_matrix,
            "rows": [asdict(row) for row in self.rows],
        }


def expected_matrix_keys(expected_matrix: Mapping[str, Sequence[Any]]) -> set[tuple[int, float, int]]:
    return {
        (int(context), float(depth), int(seed))
        for context in expected_matrix["contexts"]
        for depth in expected_matrix["depths"]
        for seed in expected_matrix["seeds"]
    }


def _validate_binding(
    binding: Any, expected_binding: Mapping[str, str] | None
) -> list[str]:
    errors: list[str] = []
    if not isinstance(binding, Mapping):
        return ["binding is required"]
    for key in REQUIRED_BINDING_KEYS:
        value = binding.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"binding.{key} is required")
        elif expected_binding is not None and value != expected_binding.get(key):
            errors.append(f"binding.{key} does not match expected hash")
    return errors


def _validate_metadata(metadata: Any) -> list[str]:
    if not isinstance(metadata, Mapping) or not isinstance(
        metadata.get("timestamp_utc"), str
    ) or not metadata["timestamp_utc"]:
        return ["metadata must include timestamp_utc and git_commit"]
    commit = metadata.get("git_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_SHA.fullmatch(commit):
        return ["metadata.git_commit must be a resolved 40-character SHA"]
    return []


def validate_quality_artifact(
    artifact: Mapping[str, Any], expected_binding: Mapping[str, str] | None = None
) -> ValidationResult:
    """Validate every expected quality cell and all retrieval-safety preconditions."""
    errors: list[str] = []
    if artifact.get("artifact_type") != QUALITY_ARTIFACT_TYPE:
        errors.append("artifact_type must be quality_matrix")
    if artifact.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if artifact.get("status") not in {"completed", "failed", "qualified", "invalid"}:
        errors.append("quality status must be completed, failed, qualified, or invalid")
    for error in _validate_metadata(artifact.get("metadata")):
        errors.append(f"quality {error}")
    errors.extend(_validate_binding(artifact.get("binding"), expected_binding))
    expected = artifact.get("expected_matrix")
    rows = artifact.get("rows")
    if not isinstance(expected, Mapping) or not all(
        isinstance(expected.get(key), list) and expected[key]
        for key in ("contexts", "depths", "seeds")
    ):
        errors.append("expected_matrix must provide non-empty contexts, depths, and seeds")
        return ValidationResult(False, tuple(errors))
    if not isinstance(rows, list):
        errors.append("rows must be a list")
        return ValidationResult(False, tuple(errors))

    expected_keys = expected_matrix_keys(expected)
    if len(rows) != len(expected_keys):
        errors.append(f"expected row count {len(expected_keys)}, got {len(rows)}")

    seen: set[tuple[int, float, int]] = set()
    valid_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"row {index} is not an object")
            continue
        try:
            key = (int(row["context_target"]), float(row["depth"]), int(row["trial_seed"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"row {index} is missing matrix coordinates")
            continue
        if key not in expected_keys:
            errors.append(f"row {index} has unexpected matrix coordinates {key}")
        if key in seen:
            errors.append(f"row {index} duplicates matrix coordinates {key}")
        seen.add(key)
        valid_rows.append(row)
        actual_context = row.get("context_actual")
        if type(actual_context) is not int or actual_context <= 0:
            errors.append(f"row {index} context_actual must be a positive integer")
        else:
            minimum_context = math.ceil(key[0] * MIN_CONTEXT_FIDELITY)
            if actual_context > key[0] or actual_context < minimum_context:
                errors.append(
                    f"row {index} context fidelity requires {minimum_context}-{key[0]} "
                    f"effective tokens for requested {key[0]}, got {actual_context}"
                )
        actual_depth = row.get("needle_depth_actual")
        if (
            not isinstance(actual_depth, (int, float))
            or isinstance(actual_depth, bool)
            or not math.isfinite(actual_depth)
            or not 0.0 <= actual_depth <= 1.0
        ):
            errors.append(
                f"row {index} needle_depth_actual must be finite and within [0, 1]"
            )
        filler_budget = row.get("filler_token_budget")
        if type(filler_budget) is not int or filler_budget <= 0:
            errors.append(f"row {index} filler_token_budget must be a positive integer")
        elif type(actual_context) is int and actual_context > 0 and filler_budget >= actual_context:
            errors.append(
                f"row {index} filler_token_budget must be less than context_actual"
            )
        elif (
            isinstance(actual_depth, (int, float))
            and not isinstance(actual_depth, bool)
            and math.isfinite(actual_depth)
        ):
            expected_depth = round(filler_budget * key[1]) / filler_budget
            if not math.isclose(
                actual_depth, expected_depth, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(
                    f"row {index} needle_depth_actual violates the discrete placement contract"
                )
        if row.get("construction_mode") != "verified_token_ids":
            errors.append(
                f"row {index} construction_mode must be verified_token_ids"
            )
        if row.get("status") != "completed":
            errors.append(f"row {index} status is not completed")
        if row.get("needle_present_after_tokenization") is not True:
            errors.append(f"row {index} lost needle during tokenization")
        if row.get("oom") is True:
            errors.append(f"row {index} recorded OOM")
        if row.get("exception") not in (None, ""):
            errors.append(f"row {index} recorded exception")
        if not isinstance(row.get("baseline_correct"), bool):
            errors.append(f"row {index} baseline correctness is missing")
        if not isinstance(row.get("profile_correct"), bool):
            errors.append(f"row {index} profile correctness is missing")
    missing = expected_keys - seen
    if missing:
        errors.append(f"missing expected matrix rows: {sorted(missing)}")
    return ValidationResult(not errors, tuple(errors))


def calculate_paired_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, seed: int, iterations: int = 10000
) -> PairedBootstrap:
    """Return a deterministic paired-bootstrap upper 95% bound in pp.

    The caller must validate the matrix first.  Keeping this function strict
    prevents partial, missing, or tokenizer-lost cases from becoming a delta.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    deltas: list[float] = []
    for row in rows:
        if row.get("status") != "completed" or not row.get(
            "needle_present_after_tokenization"
        ):
            raise ValueError("bootstrap requires completed eligible rows")
        if row.get("oom") or row.get("exception") not in (None, ""):
            raise ValueError("bootstrap requires rows without OOM or exception")
        baseline, profile = row.get("baseline_correct"), row.get("profile_correct")
        if not isinstance(baseline, bool) or not isinstance(profile, bool):
            raise ValueError("bootstrap requires boolean paired correctness")
        deltas.append(float(baseline) - float(profile))
    if not deltas:
        raise ValueError("bootstrap requires at least one eligible row")

    observed = sum(deltas) / len(deltas) * 100.0
    rng = random.Random(seed)
    distribution = sorted(
        sum(rng.choice(deltas) for _ in range(len(deltas))) / len(deltas) * 100.0
        for _ in range(iterations)
    )
    upper_index = min(iterations - 1, math.ceil(iterations * 0.95) - 1)
    return PairedBootstrap(
        delta_pp=observed,
        upper_95_pp=distribution[upper_index],
        seed=seed,
        iterations=iterations,
        eligible_cases=len(deltas),
    )


def quality_retrieval_safe(
    artifact: Mapping[str, Any], *, bootstrap_seed: int, iterations: int = 10000
) -> tuple[bool, PairedBootstrap | None, ValidationResult]:
    validation = validate_quality_artifact(artifact)
    if not validation.valid:
        return False, None, validation
    rows = artifact["rows"]
    baseline_accuracy = sum(row["baseline_correct"] for row in rows) / len(rows)
    bootstrap = calculate_paired_bootstrap(rows, seed=bootstrap_seed, iterations=iterations)
    safe = baseline_accuracy >= 0.95 and bootstrap.upper_95_pp <= 2.0
    return safe, bootstrap, validation


def validate_systems_artifact(
    artifact: Mapping[str, Any], expected_binding: Mapping[str, str] | None = None
) -> ValidationResult:
    errors: list[str] = []
    if artifact.get("artifact_type") != SYSTEMS_ARTIFACT_TYPE:
        errors.append("artifact_type must be systems_benchmark")
    if artifact.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if artifact.get("status") != "completed":
        errors.append("systems status is not completed")
    for error in _validate_metadata(artifact.get("metadata")):
        errors.append(f"systems {error}")
    errors.extend(_validate_binding(artifact.get("binding"), expected_binding))
    expected = artifact.get("expected_backends")
    trials = artifact.get("trials")
    if not isinstance(expected, list) or not expected:
        errors.append("expected_backends is required")
        return ValidationResult(False, tuple(errors))
    if not isinstance(trials, list) or not trials:
        errors.append("trials must be a non-empty list")
        return ValidationResult(False, tuple(errors))
    found = {trial.get("backend") for trial in trials if isinstance(trial, Mapping)}
    for backend in expected:
        if backend not in found:
            errors.append(f"missing expected backend {backend}")
    required_metrics = (
        "actual_store_bytes", "physical_bytes", "max_memory_allocated",
        "max_memory_reserved", "prefill_ms", "ttft_ms", "decode_token_ms",
        "decode_p50_ms", "decode_p95_ms", "decode_tokens_per_second", "dispatch_counts",
    )
    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping):
            errors.append(f"trial {index} is not an object")
            continue
        if trial.get("status") != "completed":
            errors.append(f"trial {index} status is not completed")
        if trial.get("oom") is True:
            errors.append(f"trial {index} recorded OOM")
        if trial.get("exception") not in (None, ""):
            errors.append(f"trial {index} recorded exception")
        for metric in required_metrics:
            if trial.get(metric) is None:
                errors.append(f"trial {index} missing {metric}")
        for metric in (
            "max_memory_allocated", "max_memory_reserved",
            "prefill_ms", "ttft_ms", "decode_p50_ms", "decode_p95_ms",
            "decode_tokens_per_second",
        ):
            value = trial.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                errors.append(f"trial {index} {metric} must be finite and non-negative")
        decode = trial.get("decode_token_ms")
        if not isinstance(decode, list) or not decode:
            errors.append(f"trial {index} decode_token_ms must be non-empty")
        elif any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in decode):
            errors.append(f"trial {index} decode_token_ms must contain finite non-negative values")
        physical = trial.get("physical_bytes")
        if not isinstance(physical, Mapping):
            errors.append(f"trial {index} physical_bytes is required")
        elif missing_categories := [
            category for category in PHYSICAL_BYTE_CATEGORIES if category not in physical
        ]:
            errors.append(
                f"trial {index} physical_bytes lacks required categories: "
                f"{', '.join(missing_categories)}"
            )
        elif any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in physical.values()):
            errors.append(f"trial {index} physical_bytes must be non-negative integers")
        else:
            actual_store = trial.get("actual_store_bytes")
            if type(actual_store) is not int or actual_store < 0:
                errors.append(f"trial {index} actual_store_bytes must be a non-negative integer")
            elif actual_store != sum(physical.values()):
                errors.append(
                    f"trial {index} actual_store_bytes must equal physical_bytes total"
                )
        dispatch = trial.get("dispatch_counts")
        if not isinstance(dispatch, Mapping) or not dispatch:
            errors.append(f"trial {index} dispatch_counts must be non-empty")
        elif any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in dispatch.values()):
            errors.append(f"trial {index} dispatch_counts must contain non-negative integers")
        elif trial.get("backend") == "triton_direct":
            if dispatch.get("triton_direct", 0) <= 0:
                errors.append(f"trial {index} triton_direct dispatches must be positive")
            if dispatch.get("packed_reference", 0) != 0:
                errors.append(
                    f"trial {index} triton_direct cannot report packed_reference dispatches"
                )
            if isinstance(physical, Mapping) and (
                physical.get("keys", 0) + physical.get("values", 0) > 0
            ) and dispatch.get("packed_blocks_consumed", 0) <= 0:
                errors.append(
                    f"trial {index} packed_blocks_consumed must be positive for packed history"
                )
        elif trial.get("backend") == "packed_reference":
            if dispatch.get("packed_reference", 0) <= 0:
                errors.append(f"trial {index} packed_reference dispatches must be positive")
            if dispatch.get("triton_direct", 0) != 0:
                errors.append(
                    f"trial {index} packed_reference cannot report triton_direct dispatches"
                )
            if dispatch.get("packed_blocks_consumed", 0) != 0:
                errors.append(
                    f"trial {index} packed_reference cannot report packed_blocks_consumed"
                )
    warmups = artifact.get("warmups")
    if not isinstance(warmups, list) or not warmups:
        errors.append("warmups must be a non-empty list")
    else:
        for index, warmup in enumerate(warmups):
            if not isinstance(warmup, Mapping) or warmup.get("backend") not in expected:
                errors.append(f"warmup {index} has an invalid backend")
            elif warmup.get("status") != "completed" or warmup.get("oom") is True or warmup.get("exception") not in (None, ""):
                errors.append(f"warmup {index} did not complete cleanly")
    boundary = artifact.get("oom_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("status") not in ("oom", "not_reached_within_configured_probe_range"):
        errors.append("oom_boundary is required")
    elif not isinstance(boundary.get("attempted_contexts"), list) or len(boundary["attempted_contexts"]) < 2 or any(
        not isinstance(value, int) or value <= 0 for value in boundary["attempted_contexts"]
    ):
        errors.append("oom_boundary must record at least two explicit attempted_contexts")
    probes = boundary.get("probes") if isinstance(boundary, Mapping) else None
    if not isinstance(probes, list) or len(probes) != len(boundary.get("attempted_contexts", [])):
        errors.append("oom_boundary probes must match attempted_contexts")
    elif any(not isinstance(probe, Mapping) or probe.get("context_tokens") != context or probe.get("status") not in {"completed", "oom", "error"} for probe, context in zip(probes, boundary["attempted_contexts"])):
        errors.append("oom_boundary probes must record ordered context/status outcomes")
    elif boundary.get("status") == "not_reached_within_configured_probe_range" and any(probe.get("status") != "completed" or probe.get("oom") is True for probe in probes):
        errors.append("not_reached OOM boundary requires completed non-OOM probes")
    elif boundary.get("status") == "oom" and not any(probe.get("oom") is True or probe.get("status") == "oom" for probe in probes):
        errors.append("OOM boundary requires an OOM probe outcome")
    return ValidationResult(not errors, tuple(errors))


def validate_qualified_run(
    report: Mapping[str, Any], expected_binding: Mapping[str, str] | None = None
) -> ValidationResult:
    errors: list[str] = []
    if report.get("artifact_type") != QUALIFIED_ARTIFACT_TYPE:
        errors.append("artifact_type must be qualified_run")
    if report.get("status") not in {"qualified", "invalid", "failed", "inconclusive"}:
        errors.append("release status must be qualified, invalid, failed, or inconclusive")
    for error in _validate_metadata(report.get("metadata")):
        errors.append(f"release {error}")
    errors.extend(_validate_binding(report.get("binding"), expected_binding))
    decision = report.get("qualified_32k_decision")
    requires_32k_decision = (
        report.get("mode") == "qualified" or report.get("status") == "qualified"
    )
    if not isinstance(decision, Mapping):
        if requires_32k_decision:
            errors.append("qualified_32k_decision is required")
    elif requires_32k_decision:
        decision_inputs = (
            "preflight_valid",
            "eight_k_retrieval_safe",
            "eight_k_systems_valid",
            "eight_k_capacity_admitted",
        )
        if any(type(decision.get(key)) is not bool for key in decision_inputs):
            errors.append("qualified_32k_decision must include boolean admission inputs")
        else:
            preflight, retrieval, systems, capacity = (
                decision["preflight_valid"], decision["eight_k_retrieval_safe"],
                decision["eight_k_systems_valid"], decision["eight_k_capacity_admitted"],
            )
            expected = (
                ("skipped", False, "preflight_invalid") if not preflight else
                ("skipped", False, "8k_retrieval_gate_failed") if not retrieval else
                ("skipped", False, "8k_systems_invalid_or_failed") if not systems else
                ("skipped", False, "8k_capacity_admission_failed") if not capacity else
                ("approved", True, "preflight_and_8k_gate_passed")
            )
            if (decision.get("status"), decision.get("attempt_32k"), decision.get("reason")) != expected:
                errors.append("qualified_32k_decision reason does not match admission inputs")
        if report.get("status") == "qualified" and (
            decision.get("status") != "approved" or decision.get("attempt_32k") is not True
        ):
            errors.append("qualified report requires an approved 32K decision")
    for name, validator, required_status in (("quality", validate_quality_artifact, "qualified"), ("systems", validate_systems_artifact, "completed")):
        child = report.get(name)
        if not isinstance(child, Mapping) or (report.get("status") == "qualified" and child.get("status") != required_status) or not isinstance(child.get("artifact"), Mapping):
            errors.append(f"{name} child artifact is missing or invalid")
            continue
        result = validator(child["artifact"], report.get("binding"))
        if not result.valid:
            errors.extend(f"{name}: {error}" for error in result.errors)
    if report.get("status") == "qualified":
        leg = report.get("legs", {}).get("32k") if isinstance(report.get("legs"), Mapping) else None
        if not isinstance(leg, Mapping):
            errors.append("qualified report requires 32K child legs")
        else:
            quality_leg, systems_leg = leg.get("quality"), leg.get("systems")
            if not isinstance(quality_leg, Mapping) or quality_leg.get("status") != "qualified" or not isinstance(quality_leg.get("artifact"), Mapping):
                errors.append("qualified report requires a qualified 32K quality child")
            else:
                result = validate_quality_artifact(quality_leg["artifact"], report.get("binding"))
                if not result.valid:
                    errors.extend(f"32k quality: {error}" for error in result.errors)
            if not isinstance(systems_leg, Mapping) or systems_leg.get("structurally_valid") is not True or not isinstance(systems_leg.get("artifact"), Mapping):
                errors.append("qualified report requires a valid 32K systems child")
            else:
                result = validate_systems_artifact(systems_leg["artifact"], report.get("binding"))
                if not result.valid:
                    errors.extend(f"32k systems: {error}" for error in result.errors)
    if report.get("status") == "qualified" and errors:
        errors.append("qualified status cannot accompany invalid child artifacts")
    return ValidationResult(not errors, tuple(errors))


def validate_artifact(
    artifact: Mapping[str, Any], expected_binding: Mapping[str, str] | None = None
) -> ValidationResult:
    kind = artifact.get("artifact_type")
    if kind == QUALITY_ARTIFACT_TYPE:
        return validate_quality_artifact(artifact, expected_binding)
    if kind == SYSTEMS_ARTIFACT_TYPE:
        return validate_systems_artifact(artifact, expected_binding)
    if kind == QUALIFIED_ARTIFACT_TYPE:
        return validate_qualified_run(artifact, expected_binding)
    return ValidationResult(False, (f"unsupported artifact_type {kind!r}",))
