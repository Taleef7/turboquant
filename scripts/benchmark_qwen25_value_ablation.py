#!/usr/bin/env python3
"""Diagnostic Qwen2.5 value-bit ablation for the fixed-token decode path.

This runner compares the pinned production v6 profile with the opt-in v8
value profile on the same baseline logits, stream, and serial direct lane. It
reuses the warmup lane implementation but emits a separate artifact contract;
the canonical replay/warmup artifacts and parallel fused path are untouched.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_schema import ValidationResult  # noqa: E402
from scripts.benchmark_qwen25_fixed_warmup import (  # noqa: E402
    _finite_number,
    _fixed_stream,
    _quality,
    _run_lane,
    _stats,
)
from scripts.replay_qwen25_fixed_tokens import (  # noqa: E402
    _fixture_identity,
)
from scripts.repro_utils import (  # noqa: E402
    PINNED_QWEN25_HUB_REVISION,
    PINNED_QWEN25_MODEL_ID,
    benchmark_binding,
    capture_env_metadata,
    capture_source_metadata,
    git_commit_sha,
    seed_everything,
    sha256_file,
    write_json,
)


CONFIG_PATH = ROOT / "configs" / "benchmarks" / "qwen25-value-ablation-local-v1.json"
RUNNER_PATH = Path(__file__).resolve()
WARMUP_RUNNER_PATH = ROOT / "scripts" / "benchmark_qwen25_fixed_warmup.py"
EXPECTED_CONTEXTS = [4096, 8192]
EXPECTED_BACKENDS = ("baseline", "serial_direct")
EXPECTED_PROFILE_IDS = ("v6", "v8")
QUALITY_KEYS = (
    "max_abs_logit",
    "mean_abs_logit",
    "kl_divergence",
    "top1_agreement",
    "greedy_agreement",
)
MEMORY_KEYS = (
    "logical_payload_bytes",
    "physical_cache_bytes",
    "arena_capacity_bytes",
    "tail_capacity_bytes",
    "workspace_bytes",
    "unique_resident_bytes",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _profile_entries(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = config.get("profiles")
    if not isinstance(entries, list):
        raise ValueError("profiles must be a list")
    return [dict(entry) for entry in entries]


def _profile_bindings(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve and fingerprint every profile declared by the config."""
    bindings: list[dict[str, Any]] = []
    for entry in _profile_entries(config):
        path = ROOT / str(entry["path"])
        actual_file_sha = sha256_file(path)
        if actual_file_sha != entry["file_sha256"]:
            raise ValueError(f"profile file hash mismatch for {entry['id']}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("profile_sha256") != entry["profile_sha256"]:
            raise ValueError(f"profile hash mismatch for {entry['id']}")
        if payload.get("value_bits") != entry["value_bits"]:
            raise ValueError(f"value_bits mismatch for {entry['id']}")
        bindings.append(
            {
                "id": entry["id"],
                "path": str(entry["path"]),
                "file_sha256": actual_file_sha,
                "profile_sha256": payload["profile_sha256"],
                "value_bits": int(payload["value_bits"]),
                "key_bits": int(payload["key_bits"]),
                "model_id": payload["model_revision"]["model_id"],
                "model_revision": payload["model_revision"]["revision"],
            }
        )
    return bindings


def load_ablation_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("artifact_type") != "qwen25_value_ablation":
        errors.append("artifact_type must be qwen25_value_ablation")
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("name") != "qwen25-value-ablation-local-v1":
        errors.append("name must be qwen25-value-ablation-local-v1")
    if payload.get("model_id") != PINNED_QWEN25_MODEL_ID:
        errors.append("model_id must bind the pinned Qwen2.5 model")
    if payload.get("model_revision") != PINNED_QWEN25_HUB_REVISION:
        errors.append("model_revision must bind the pinned Qwen2.5 revision")
    digest = payload.get("fixture_manifest_sha256")
    if not _is_sha256(digest):
        errors.append("fixture_manifest_sha256 must be a lowercase SHA-256 digest")
    if payload.get("contexts") != EXPECTED_CONTEXTS:
        errors.append("contexts must be the predeclared [4096, 8192] matrix")
    if tuple(payload.get("backends", ())) != EXPECTED_BACKENDS:
        errors.append("backends must be baseline and serial_direct only")
    if type(payload.get("stream_seed")) is not int:  # noqa: E721
        errors.append("stream_seed must be a built-in int")
    for key in ("warmup_tokens", "measured_decode_tokens", "repeats"):
        if type(payload.get(key)) is not int or payload[key] < 1:  # noqa: E721
            errors.append(f"{key} must be a positive built-in int")
    entries = payload.get("profiles")
    if not isinstance(entries, list) or tuple(
        entry.get("id") for entry in entries if isinstance(entry, Mapping)
    ) != EXPECTED_PROFILE_IDS:
        errors.append("profiles must declare v6 followed by v8")
    else:
        expected_profile_paths = {
            "v6": "configs/profiles/qwen25-7b-direct-v1.json",
            "v8": "configs/profiles/qwen25-7b-direct-v1-v8-value.json",
        }
        for entry in entries:
            if (
                not isinstance(entry.get("path"), str)
                or not _is_sha256(entry.get("file_sha256"))
                or not _is_sha256(entry.get("profile_sha256"))
                or type(entry.get("value_bits")) is not int  # noqa: E721
            ):
                errors.append(f"profile {entry.get('id')} binding is malformed")
            elif entry["path"] != expected_profile_paths[entry["id"]]:
                errors.append(f"profile {entry['id']} path is not the pinned profile")
        if [entry.get("value_bits") for entry in entries] != [6, 8]:
            errors.append("profiles must compare value_bits 6 and 8")
    gates = payload.get("quality_gates")
    if not isinstance(gates, Mapping) or set(gates) != set(QUALITY_KEYS):
        errors.append("quality_gates must declare the complete fixed replay gate")
    if errors:
        raise ValueError("invalid value ablation config: " + "; ".join(errors))
    try:
        _profile_bindings(payload)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid value ablation profile binding: {exc}") from exc
    return payload


def expected_row_ids(config: Mapping[str, Any]) -> list[str]:
    return [
        f"{context}-baseline"
        if profile_id == "baseline"
        else f"{context}-{profile_id}-serial_direct"
        for context in config["contexts"]
        for profile_id in ("baseline", *EXPECTED_PROFILE_IDS)
    ]


def _source_identity(config_path: str | Path) -> dict[str, Any]:
    source = capture_source_metadata(ROOT)
    return {
        **source,
        "runner_sha256": sha256_file(RUNNER_PATH),
        "warmup_runner_sha256": sha256_file(WARMUP_RUNNER_PATH),
        "config_path": Path(config_path).resolve().relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "model_id": PINNED_QWEN25_MODEL_ID,
        "model_revision": PINNED_QWEN25_HUB_REVISION,
    }


def _error_row(
    context: int,
    profile_id: str,
    backend: str,
    profile: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    reason: str,
    status: str = "error",
) -> dict[str, Any]:
    row_id = (
        f"{context}-baseline"
        if profile_id == "baseline"
        else f"{context}-{profile_id}-{backend}"
    )
    row = {
        "row_id": row_id,
        "context_tokens": context,
        "backend": backend,
        "profile_id": profile_id,
        "profile": dict(profile) if profile is not None else None,
        "status": status,
        "reason": reason,
        "source_identity": dict(source),
    }
    if status in {"error", "oom"}:
        row["exception"] = reason
    return row


def build_unavailable_artifact(
    config: Mapping[str, Any], *, config_path: str | Path, reason: str
) -> dict[str, Any]:
    source = _source_identity(config_path)
    bindings = _profile_bindings(config)
    rows = []
    for row_id in expected_row_ids(config):
        parts = row_id.split("-")
        profile_id = "baseline" if parts[-1] == "baseline" else parts[1]
        rows.append(
            {
                "row_id": row_id,
                "context_tokens": int(parts[0]),
                "backend": "baseline" if profile_id == "baseline" else "serial_direct",
                "profile_id": profile_id,
                "profile": None
                if profile_id == "baseline"
                else next(item for item in bindings if item["id"] == profile_id),
                "status": "unavailable",
                "reason": reason,
                "source_identity": source,
            }
        )
    return {
        "artifact_type": "qwen25_value_ablation",
        "schema_version": 1,
        "artifact_id": f"qwen25-value-ablation-{git_commit_sha(ROOT)[:12]}",
        "status": "unavailable",
        "diagnostic_only": True,
        "claims": [],
        "config": dict(config),
        "config_path": source["config_path"],
        "binding": benchmark_binding(ROOT),
        "profile_bindings": bindings,
        "metadata": capture_env_metadata(),
        "source_identity": source,
        "rows": rows,
    }


def _valid_stats(timing: Mapping[str, Any], key: str, count: int) -> bool:
    values = timing.get(key)
    return isinstance(values, list) and len(values) == count and all(
        _finite_number(value) and float(value) >= 0 for value in values
    )


def validate_ablation_artifact(
    artifact: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path = CONFIG_PATH,
) -> ValidationResult:
    errors: list[str] = []
    if not isinstance(artifact, Mapping):
        return ValidationResult(False, ("artifact must be an object",))
    config = dict(config or load_ablation_config(config_path))
    if artifact.get("artifact_type") != "qwen25_value_ablation":
        errors.append("artifact_type must be qwen25_value_ablation")
    if artifact.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if artifact.get("diagnostic_only") is not True:
        errors.append("diagnostic_only must be true")
    if artifact.get("claims") != []:
        errors.append("diagnostic artifact claims must be empty")
    if artifact.get("config") != config:
        errors.append("artifact config does not match the predeclared config")
    expected_config_path = Path(config_path).resolve().relative_to(ROOT).as_posix()
    if artifact.get("config_path") != expected_config_path:
        errors.append("artifact config_path is invalid")
    try:
        expected_bindings = _profile_bindings(config)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return ValidationResult(False, (f"profile binding cannot be resolved: {exc}",))
    if artifact.get("profile_bindings") != expected_bindings:
        errors.append("artifact profile_bindings do not match the checked-in profiles")
    source = artifact.get("source_identity")
    source_valid = isinstance(source, Mapping)
    if not source_valid:
        errors.append("artifact source_identity is missing")
    else:
        commit = source.get("git_commit")
        if not isinstance(commit, str) or len(commit) != 40 or not all(
            character in "0123456789abcdef" for character in commit
        ):
            errors.append("artifact source_identity git_commit is invalid")
        if not isinstance(source.get("dirty"), bool):
            errors.append("artifact source_identity dirty flag is invalid")
        for key, expected in (
            ("model_id", config["model_id"]),
            ("model_revision", config["model_revision"]),
            ("config_path", expected_config_path),
        ):
            if source.get(key) != expected:
                errors.append(f"artifact source_identity {key} is invalid")
        for key, path in (
            ("runner_sha256", RUNNER_PATH),
            ("warmup_runner_sha256", WARMUP_RUNNER_PATH),
        ):
            if source.get(key) != sha256_file(path):
                errors.append(f"artifact source_identity {key} is invalid")
        if source.get("config_sha256") != sha256_file(config_path):
            errors.append("artifact source_identity config_sha256 is invalid")
    rows = artifact.get("rows")
    expected = expected_row_ids(config)
    if not isinstance(rows, list) or [row.get("row_id") for row in rows] != expected:
        errors.append("rows must exactly match the predeclared ablation matrix")
        rows = rows if isinstance(rows, list) else []
    bindings_by_id = {item["id"]: item for item in expected_bindings}
    measured = int(config["measured_decode_tokens"])
    repeats = int(config["repeats"])
    expected_samples = measured * repeats
    expected_layer_calls = (int(config["warmup_tokens"]) + measured) * repeats * 28
    expected_standard_calls = (1 + int(config["warmup_tokens"]) + measured) * repeats * 28
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"row {index} must be an object")
            continue
        profile_id = row.get("profile_id")
        if index < len(expected):
            context, *rest = expected[index].split("-")
            expected_profile_id = "baseline" if rest == ["baseline"] else rest[0]
            expected_backend = "baseline" if expected_profile_id == "baseline" else "serial_direct"
            if row.get("context_tokens") != int(context):
                errors.append(f"row {index} context_tokens do not match row_id")
            if row.get("backend") != expected_backend or profile_id != expected_profile_id:
                errors.append(f"row {index} lane/profile does not match row_id")
            expected_profile = (
                None if expected_profile_id == "baseline" else bindings_by_id[expected_profile_id]
            )
            if row.get("profile") != expected_profile:
                errors.append(f"row {index} profile binding does not match row_id")
        if source_valid and row.get("source_identity") != source:
            errors.append(f"row {index} source_identity disagrees with artifact")
        status = row.get("status")
        if status not in {"completed", "unavailable", "error", "oom"}:
            errors.append(f"row {index} has invalid status")
        if status != "completed":
            continue
        quality = row.get("quality")
        dispatch = row.get("dispatch")
        memory = row.get("memory")
        timing = row.get("timing")
        if not all(isinstance(item, Mapping) for item in (quality, dispatch, memory, timing)):
            errors.append(f"row {index} completed metrics are missing")
            continue
        quality_valid = all(
            _finite_number(quality.get(key)) and float(quality[key]) >= 0
            for key in QUALITY_KEYS[:3]
        )
        top1 = quality.get("top1_agreement")
        greedy = quality.get("greedy_agreement")
        if not quality_valid or not _finite_number(top1) or not 0 <= float(top1) <= 1 or not isinstance(greedy, bool):
            errors.append(f"row {index} quality metrics are invalid")
            continue
        gates = config["quality_gates"]
        gate = (
            float(quality["max_abs_logit"]) <= gates["max_abs_logit"]
            and float(quality["mean_abs_logit"]) <= gates["mean_abs_logit"]
            and float(quality["kl_divergence"]) <= gates["kl_divergence"]
            and float(top1) >= gates["top1_agreement"]
            and greedy is True
        )
        if row.get("quality_gate_pass") is not gate:
            errors.append(f"row {index} quality_gate_pass disagrees with metrics")
        if profile_id == "baseline" and (
            any(float(quality[key]) != 0 for key in QUALITY_KEYS[:3])
            or float(top1) != 1.0
            or greedy is not True
        ):
            errors.append(f"row {index} baseline quality must be exact zero-drift")
        if dispatch.get("fallback") is not False:
            errors.append(f"row {index} fallback must be false")
        if profile_id == "baseline":
            if dispatch.get("standard_attention_calls") != expected_standard_calls:
                errors.append(f"row {index} baseline dispatch count is invalid")
        elif dispatch.get("packed_decode_calls") != expected_layer_calls:
            errors.append(f"row {index} serial dispatch count is invalid")
        if any(dispatch.get(key, 0) != 0 for key in (
            "parallel_decode_calls",
            "parallel_page_launches",
            "parallel_reduction_launches",
            "parallel_tail_launches",
        )):
            errors.append(f"row {index} unexpectedly reports parallel dispatch")
        if any(not isinstance(memory.get(key), int) or memory[key] < 0 for key in MEMORY_KEYS):
            errors.append(f"row {index} memory byte accounting is invalid")
        elif memory["unique_resident_bytes"] < memory["physical_cache_bytes"]:
            errors.append(f"row {index} unique resident bytes undercount physical cache")
        if not all(
            _valid_stats(timing, key, count)
            for key, count in (
                ("prefill_ms", repeats),
                ("warmup_ms", repeats),
                ("decode_token_ms", expected_samples),
                ("first_run_decode_ms", repeats),
                ("steady_state_decode_ms", (measured - 1) * repeats),
            )
        ):
            errors.append(f"row {index} timing samples are invalid")
            continue
        for key, source_key in (
            ("decode", "decode_token_ms"),
            ("first_run", "first_run_decode_ms"),
            ("steady_state", "steady_state_decode_ms"),
        ):
            stats = _stats(timing[source_key])
            for suffix in ("p50_ms", "p95_ms"):
                if not _finite_number(timing.get(f"{key}_{suffix}")) or not math.isclose(
                    float(timing[f"{key}_{suffix}"]),
                    stats[suffix],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    errors.append(f"row {index} timing {key}_{suffix} disagrees with samples")
        if not _finite_number(timing.get("decode_mean_ms")) or not math.isclose(
            float(timing["decode_mean_ms"]),
            _stats(timing["decode_token_ms"])["mean_ms"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append(f"row {index} timing decode_mean_ms disagrees with samples")
    statuses = {row.get("status") for row in rows if isinstance(row, Mapping)}
    expected_status = "unavailable" if statuses == {"unavailable"} else "diagnostic" if statuses == {"completed"} else "partial"
    if artifact.get("status") != expected_status:
        errors.append(f"status must be {expected_status!r} for the row outcomes")
    return ValidationResult(not errors, tuple(errors))


def _load_model(config: Mapping[str, Any], *, model_path: str | Path | None, allow_download: bool, cache_dir: str | Path | None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the value-bit ablation")
    load_kwargs: dict[str, Any] = {
        "local_files_only": not allow_download,
        "trust_remote_code": False,
    }
    model_source = config["model_id"] if model_path is None else str(model_path)
    if model_path is None:
        load_kwargs["revision"] = config["model_revision"]
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **load_kwargs,
        torch_dtype=torch.float16,
        device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    ).eval()
    return model, tokenizer


def run_ablation(
    config_path: str | Path,
    output: str | Path,
    *,
    allow_download: bool = False,
    cache_dir: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = load_ablation_config(config_path)
    bindings = _profile_bindings(config)
    source = _source_identity(config_path)
    artifact = build_unavailable_artifact(config, config_path=config_path, reason="model fixture not attempted")
    artifact["configuration"] = {
        "allow_download": allow_download,
        "local_files_only": not allow_download,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "model_path": None if model_path is None else str(model_path),
        "stream_seed": config["stream_seed"],
    }
    if model_path is not None:
        fixture, fixture_error = _fixture_identity(model_path, config)
        if fixture_error is not None:
            reason = f"model fixture rejected: {fixture_error}"
            artifact["rows"] = [
                {**row, "reason": reason} for row in artifact["rows"]
            ]
            write_json(str(output), artifact)
            return artifact
        source = {**source, "model_fixture": fixture}
        artifact["source_identity"] = source
        for row in artifact["rows"]:
            row["source_identity"] = source
    try:
        model, tokenizer = _load_model(
            config,
            model_path=model_path,
            allow_download=allow_download,
            cache_dir=cache_dir,
        )
        from core.profiles import load_profile

        profiles = {
            binding["id"]: load_profile(ROOT / binding["path"])
            for binding in bindings
        }
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        artifact["rows"] = [{**row, "reason": reason} for row in artifact["rows"]]
        write_json(str(output), artifact)
        return artifact
    import torch

    stream_seed = int(config["stream_seed"])
    seed_everything(stream_seed, deterministic=True)
    rows: list[dict[str, Any]] = []
    for context in config["contexts"]:
        total_tokens = context + int(config["warmup_tokens"]) + int(config["measured_decode_tokens"])
        stream = _fixed_stream(tokenizer, total_tokens, stream_seed + context)
        prompt_ids = stream[:, :context].to(model.device)
        decode_ids = stream[:, context:].to(model.device)
        try:
            baseline_row, baseline_logits = _run_lane(
                model,
                profiles["v6"],
                prompt_ids,
                decode_ids,
                "baseline",
                config,
                torch,
            )
        except torch.cuda.OutOfMemoryError as exc:
            reason = str(exc)
            rows.append(
                _error_row(context, "baseline", "baseline", None, source, reason, "oom")
            )
            for profile_id in EXPECTED_PROFILE_IDS:
                profile_binding = next(item for item in bindings if item["id"] == profile_id)
                rows.append(
                    _error_row(
                        context,
                        profile_id,
                        "serial_direct",
                        profile_binding,
                        source,
                        "baseline unavailable: " + reason,
                        "oom",
                    )
                )
            continue
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            rows.append(
                _error_row(context, "baseline", "baseline", None, source, reason)
            )
            for profile_id in EXPECTED_PROFILE_IDS:
                profile_binding = next(item for item in bindings if item["id"] == profile_id)
                rows.append(
                    _error_row(
                        context,
                        profile_id,
                        "serial_direct",
                        profile_binding,
                        source,
                        "baseline unavailable: " + reason,
                    )
                )
            continue
        baseline_row.update(
            {
                "row_id": f"{context}-baseline",
                "context_tokens": context,
                "backend": "baseline",
                "profile_id": "baseline",
                "profile": None,
                "source_identity": source,
                "quality": {
                    "max_abs_logit": 0.0,
                    "mean_abs_logit": 0.0,
                    "kl_divergence": 0.0,
                    "top1_agreement": 1.0,
                    "greedy_agreement": True,
                },
                "quality_gate_pass": True,
            }
        )
        rows.append(baseline_row)
        for profile_id in EXPECTED_PROFILE_IDS:
            profile_binding = next(item for item in bindings if item["id"] == profile_id)
            try:
                row, candidate_logits = _run_lane(
                    model,
                    profiles[profile_id],
                    prompt_ids,
                    decode_ids,
                    "serial_direct",
                    config,
                    torch,
                )
            except torch.cuda.OutOfMemoryError as exc:
                rows.append(
                    _error_row(
                        context,
                        profile_id,
                        "serial_direct",
                        profile_binding,
                        source,
                        str(exc),
                        "oom",
                    )
                )
                continue
            except Exception as exc:
                rows.append(
                    _error_row(
                        context,
                        profile_id,
                        "serial_direct",
                        profile_binding,
                        source,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            try:
                quality = _quality(baseline_logits, candidate_logits)
            except Exception as exc:
                rows.append(
                    _error_row(
                        context,
                        profile_id,
                        "serial_direct",
                        profile_binding,
                        source,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            gates = config["quality_gates"]
            row.update(
                {
                    "row_id": f"{context}-{profile_id}-serial_direct",
                    "context_tokens": context,
                    "backend": "serial_direct",
                    "profile_id": profile_id,
                    "profile": profile_binding,
                    "source_identity": source,
                    "quality": quality,
                    "quality_gate_pass": (
                        quality["max_abs_logit"] <= gates["max_abs_logit"]
                        and quality["mean_abs_logit"] <= gates["mean_abs_logit"]
                        and quality["kl_divergence"] <= gates["kl_divergence"]
                        and quality["top1_agreement"] >= gates["top1_agreement"]
                        and quality["greedy_agreement"] is True
                    ),
                }
            )
            rows.append(row)
    statuses = {row.get("status") for row in rows}
    artifact.update(
        {
            "rows": rows,
            "source_identity": source,
            "status": "diagnostic"
            if statuses == {"completed"}
            else "unavailable"
            if statuses == {"unavailable"}
            else "partial",
        }
    )
    validation = validate_ablation_artifact(artifact, config, config_path=config_path)
    if not validation.valid:
        artifact["status"] = "invalid"
        artifact["validation_errors"] = list(validation.errors)
    write_json(str(output), artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--model-path")
    args = parser.parse_args()
    artifact = run_ablation(
        args.config,
        args.output,
        allow_download=args.allow_download,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
    )
    validation = validate_ablation_artifact(
        artifact,
        load_ablation_config(args.config),
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "status": artifact.get("status"),
                "valid": validation.valid,
                "rows": len(artifact.get("rows", [])),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if validation.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
