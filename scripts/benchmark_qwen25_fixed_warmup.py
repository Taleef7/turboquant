#!/usr/bin/env python3
"""Fixed-token warmup/repeat benchmark for the Qwen2.5 decode lanes.

This benchmark is deliberately separate from the one-shot replay.  It warms
each lane once outside the measured window, then measures three decode tokens
across three fresh repeats using the same fixed token stream.  Results remain
diagnostic until a later systems gate promotes them to a claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_schema import ValidationResult  # noqa: E402
from scripts.replay_qwen25_fixed_tokens import (  # noqa: E402
    BACKENDS,
    _baseline_memory,
    _direct_memory,
    _finite_number,
    _fixture_identity,
    _fixed_stream,
    _memory_runtime,
    _quality,
    _snapshot_logits,
    _source_identity as _shared_source_identity,
    _timed,
)
from scripts.repro_utils import (  # noqa: E402
    PINNED_QWEN25_HUB_REVISION,
    PINNED_QWEN25_MODEL_ID,
    benchmark_binding,
    capture_env_metadata,
    git_commit_sha,
    sha256_file,
    seed_everything,
    write_json,
)


CONFIG_PATH = ROOT / "configs" / "benchmarks" / "qwen25-fixed-warmup-v1.json"
WARMUP_CONTEXT_ALLOWLIST = {
    "qwen25-fixed-warmup-v1": [1024, 2048],
    "qwen25-fixed-warmup-local-v2": [4096, 8192],
}


def _source_identity(config_path: Path) -> dict[str, Any]:
    """Bind warmup artifacts to both shared replay and warmup runner code."""
    return {
        **_shared_source_identity(config_path),
        "warmup_runner_sha256": sha256_file(Path(__file__)),
    }


def load_warmup_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("artifact_type") != "qwen25_fixed_warmup":
        errors.append("artifact_type must be qwen25_fixed_warmup")
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("model_id") != PINNED_QWEN25_MODEL_ID:
        errors.append("model_id must bind the pinned Qwen2.5 model")
    if payload.get("model_revision") != PINNED_QWEN25_HUB_REVISION:
        errors.append("model_revision must bind the pinned Qwen2.5 revision")
    digest = payload.get("fixture_manifest_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        errors.append("fixture_manifest_sha256 must be a lowercase SHA-256 digest")
    name = payload.get("name")
    expected_contexts = WARMUP_CONTEXT_ALLOWLIST.get(name)
    if expected_contexts is None:
        errors.append(
            "name must select one of the predeclared fixed warmup configs: "
            + ", ".join(WARMUP_CONTEXT_ALLOWLIST)
        )
    contexts = payload.get("contexts")
    if (
        not isinstance(contexts, list)
        or not contexts
        or any(type(context) is not int or context < 1 for context in contexts)  # noqa: E721
        or contexts != sorted(set(contexts))
    ):
        errors.append("contexts must be a sorted list of unique positive built-in ints")
    elif expected_contexts is not None and contexts != expected_contexts:
        errors.append(
            f"contexts must match the predeclared matrix for {name}: {expected_contexts}"
        )
    if type(payload.get("stream_seed")) is not int:  # noqa: E721
        errors.append("stream_seed must be a built-in int")
    for key in ("warmup_tokens", "measured_decode_tokens", "repeats"):
        if type(payload.get(key)) is not int or payload[key] < 1:  # noqa: E721
            errors.append(f"{key} must be a positive built-in int")
    if tuple(payload.get("backends", ())) != BACKENDS:
        errors.append("backends must be the four declared decode lanes")
    gates = payload.get("quality_gates")
    required_gates = {
        "max_abs_logit",
        "mean_abs_logit",
        "kl_divergence",
        "top1_agreement",
        "greedy_agreement",
    }
    if not isinstance(gates, Mapping) or set(gates) != required_gates:
        errors.append("quality_gates must declare the complete fixed replay gate")
    if errors:
        raise ValueError("invalid warmup config: " + "; ".join(errors))
    return payload


def expected_row_ids(config: Mapping[str, Any]) -> list[str]:
    return [
        f"{context}-{backend}"
        for context in config["contexts"]
        for backend in config["backends"]
    ]


def build_unavailable_artifact(
    config: Mapping[str, Any], *, config_path: str | Path, reason: str
) -> dict[str, Any]:
    source = _source_identity(Path(config_path))
    rows = [
        {
            "row_id": row_id,
            "context_tokens": int(row_id.split("-", 1)[0]),
            "backend": row_id.split("-", 1)[1],
            "status": "unavailable",
            "reason": reason,
            "source_identity": source,
        }
        for row_id in expected_row_ids(config)
    ]
    return {
        "artifact_type": "qwen25_fixed_warmup",
        "schema_version": 1,
        "artifact_id": f"qwen25-fixed-warmup-{git_commit_sha(ROOT)[:12]}",
        "status": "unavailable",
        "diagnostic_only": True,
        "claims": [],
        "config": dict(config),
        "binding": benchmark_binding(ROOT),
        "metadata": capture_env_metadata(),
        "source_identity": source,
        "rows": rows,
    }


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not _finite_number(value) or value < 0 for value in values):
        raise ValueError("timing samples must be finite non-negative numbers")
    ordered = sorted(float(value) for value in values)
    p95 = (
        statistics.quantiles(ordered, n=100, method="inclusive")[94]
        if len(ordered) > 1
        else ordered[0]
    )
    return {
        "p50_ms": float(statistics.median(ordered)),
        "p95_ms": float(p95),
        "mean_ms": float(statistics.fmean(ordered)),
    }


def _timing_summary(
    prefill_ms: list[float],
    warmup_ms: list[float],
    first_run_ms: list[float],
    steady_state_ms: list[float],
    decode_ms: list[float],
) -> dict[str, Any]:
    return {
        "prefill_ms": prefill_ms,
        "warmup_ms": warmup_ms,
        "first_run_decode_ms": first_run_ms,
        "steady_state_decode_ms": steady_state_ms,
        "decode_token_ms": decode_ms,
        "decode_p50_ms": _stats(decode_ms)["p50_ms"],
        "decode_p95_ms": _stats(decode_ms)["p95_ms"],
        "decode_mean_ms": _stats(decode_ms)["mean_ms"],
        "first_run_p50_ms": _stats(first_run_ms)["p50_ms"],
        "first_run_p95_ms": _stats(first_run_ms)["p95_ms"],
        "steady_state_p50_ms": _stats(steady_state_ms)["p50_ms"],
        "steady_state_p95_ms": _stats(steady_state_ms)["p95_ms"],
    }


def validate_warmup_artifact(
    artifact: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> ValidationResult:
    errors: list[str] = []
    if not isinstance(artifact, Mapping):
        return ValidationResult(False, ("artifact must be an object",))
    config = dict(config or load_warmup_config())
    if artifact.get("artifact_type") != "qwen25_fixed_warmup":
        errors.append("artifact_type must be qwen25_fixed_warmup")
    if artifact.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if artifact.get("diagnostic_only") is not True:
        errors.append("diagnostic_only must be true")
    if artifact.get("claims") != []:
        errors.append("diagnostic artifact claims must be empty")
    if artifact.get("config") != config:
        errors.append("artifact config does not match the predeclared config")
    artifact_source = artifact.get("source_identity")
    source_valid = isinstance(artifact_source, Mapping)
    if not source_valid:
        errors.append("artifact source_identity is missing")
    else:
        commit = artifact_source.get("git_commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            errors.append("artifact source_identity git_commit is invalid")
        if artifact_source.get("model_id") != config["model_id"]:
            errors.append("artifact source_identity model_id is invalid")
        if artifact_source.get("model_revision") != config["model_revision"]:
            errors.append("artifact source_identity model_revision is invalid")
        if not isinstance(artifact_source.get("dirty"), bool):
            errors.append("artifact source_identity dirty flag is invalid")
        warmup_runner_sha256 = artifact_source.get("warmup_runner_sha256")
        if warmup_runner_sha256 is not None:
            if (
                not isinstance(warmup_runner_sha256, str)
                or len(warmup_runner_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in warmup_runner_sha256
                )
            ):
                errors.append(
                    "artifact source_identity warmup_runner_sha256 is invalid"
                )
            elif warmup_runner_sha256 != sha256_file(Path(__file__)):
                errors.append(
                    "artifact source_identity warmup_runner_sha256 does not match "
                    "the warmup runner"
                )
        fixture = artifact_source.get("model_fixture")
        if fixture is not None:
            if not isinstance(fixture, Mapping):
                errors.append("artifact model_fixture identity is invalid")
            else:
                if fixture.get("snapshot_id") != config["model_revision"]:
                    errors.append("artifact model_fixture snapshot_id is invalid")
                if fixture.get("manifest_sha256") != config["fixture_manifest_sha256"]:
                    errors.append("artifact model_fixture manifest is invalid")
                if not isinstance(fixture.get("resolved_path"), str):
                    errors.append("artifact model_fixture resolved_path is missing")
    expected = expected_row_ids(config)
    rows = artifact.get("rows")
    observed = (
        [row.get("row_id") if isinstance(row, Mapping) else None for row in rows]
        if isinstance(rows, list)
        else None
    )
    if observed != expected:
        errors.append("rows must exactly match the predeclared context/backend matrix")
        rows = rows if isinstance(rows, list) else []
    statuses = {"completed", "unavailable", "error", "oom"}
    measured = int(config["measured_decode_tokens"])
    repeats = int(config["repeats"])
    expected_samples = measured * repeats
    expected_layer_calls = (int(config["warmup_tokens"]) + measured) * repeats * 28
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"row {index} must be an object")
            continue
        status = row.get("status")
        if index < len(expected):
            context, backend = expected[index].split("-", 1)
            if row.get("context_tokens") != int(context):
                errors.append(f"row {index} context_tokens do not match row_id")
            if row.get("backend") != backend:
                errors.append(f"row {index} backend does not match row_id")
        if status not in statuses:
            errors.append(f"row {index} has invalid status")
        if not isinstance(row.get("source_identity"), Mapping):
            errors.append(f"row {index} source_identity is missing")
        elif source_valid and row["source_identity"] != artifact_source:
            errors.append(f"row {index} source_identity disagrees with artifact")
        if status == "unavailable" and not isinstance(row.get("reason"), str):
            errors.append(f"row {index} unavailable reason is missing")
        if status in {"error", "oom"} and not isinstance(row.get("exception"), str):
            errors.append(f"row {index} exception is missing")
        if status != "completed":
            continue
        required = ("quality", "dispatch", "memory", "timing")
        for key in required:
            if not isinstance(row.get(key), Mapping):
                errors.append(f"row {index} completed {key} is missing")
        if not isinstance(row.get("quality_gate_pass"), bool):
            errors.append(f"row {index} quality_gate_pass is missing")
        if not all(isinstance(row.get(key), Mapping) for key in required):
            continue
        quality = row["quality"]
        numeric = ("max_abs_logit", "mean_abs_logit", "kl_divergence")
        quality_valid = all(
            _finite_number(quality.get(key)) and float(quality[key]) >= 0
            for key in numeric
        )
        top1 = quality.get("top1_agreement")
        top1_valid = _finite_number(top1) and 0 <= float(top1) <= 1
        greedy = quality.get("greedy_agreement")
        if not quality_valid:
            errors.append(f"row {index} quality numeric metrics are invalid")
        if not top1_valid:
            errors.append(f"row {index} top1_agreement is invalid")
        if not isinstance(greedy, bool):
            errors.append(f"row {index} greedy_agreement is invalid")
        gate = False
        if quality_valid and top1_valid and isinstance(greedy, bool):
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
        if (
            row.get("backend") == "baseline"
            and quality_valid
            and top1_valid
            and (
                any(float(quality[key]) != 0.0 for key in numeric)
                or top1 != 1.0
                or greedy is not True
            )
        ):
            errors.append(f"row {index} baseline quality must be exact zero-drift")
        dispatch = row["dispatch"]
        if dispatch.get("fallback") is not False:
            errors.append(f"row {index} fallback must be false")
        backend = row.get("backend")
        if backend == "baseline" and dispatch.get("standard_attention_calls", 0) != (
            (1 + int(config["warmup_tokens"]) + measured) * repeats * 28
        ):
            errors.append(f"row {index} baseline attention count is invalid")
        if (
            backend == "packed_reference"
            and dispatch.get("packed_reference_decode_calls", 0) != expected_layer_calls
        ):
            errors.append(f"row {index} packed-reference dispatch count is invalid")
        if (
            backend == "serial_direct"
            and dispatch.get("packed_decode_calls", 0) != expected_layer_calls
        ):
            errors.append(f"row {index} serial dispatch count is invalid")
        if backend == "parallel_direct":
            for key in (
                "packed_decode_calls",
                "parallel_decode_calls",
                "parallel_page_launches",
                "parallel_reduction_launches",
                "parallel_tail_launches",
            ):
                if dispatch.get(key, 0) != expected_layer_calls:
                    errors.append(f"row {index} parallel {key} count is invalid")
        memory = row["memory"]
        memory_keys = (
            "logical_payload_bytes",
            "physical_cache_bytes",
            "arena_capacity_bytes",
            "tail_capacity_bytes",
            "workspace_bytes",
            "unique_resident_bytes",
        )
        memory_valid = not any(
            not isinstance(memory.get(key), int) or memory[key] < 0
            for key in memory_keys
        )
        if not memory_valid:
            errors.append(f"row {index} memory byte accounting is invalid")
        else:
            if memory["unique_resident_bytes"] < memory["physical_cache_bytes"]:
                errors.append(
                    f"row {index} unique resident bytes undercount physical cache"
                )
            if backend == "parallel_direct" and (
                memory["workspace_bytes"] <= 0 or memory["arena_capacity_bytes"] <= 0
            ):
                errors.append(f"row {index} parallel memory reservation is missing")
        timing = row["timing"]
        lists = (
            ("prefill_ms", repeats),
            ("warmup_ms", repeats),
            ("decode_token_ms", expected_samples),
            ("first_run_decode_ms", repeats),
            ("steady_state_decode_ms", (measured - 1) * repeats),
        )
        valid_lists: dict[str, list[float]] = {}
        for key, expected_length in lists:
            values = timing.get(key)
            if not isinstance(values, list) or len(values) != expected_length:
                errors.append(f"row {index} timing {key} count is invalid")
            elif any(not _finite_number(value) or value < 0 for value in values):
                errors.append(f"row {index} timing {key} values are invalid")
            else:
                valid_lists[key] = [float(value) for value in values]
        for key in (
            "decode_p50_ms",
            "decode_p95_ms",
            "decode_mean_ms",
            "first_run_p50_ms",
            "first_run_p95_ms",
            "steady_state_p50_ms",
            "steady_state_p95_ms",
        ):
            if not _finite_number(timing.get(key)) or timing[key] < 0:
                errors.append(f"row {index} timing {key} is invalid")
        for prefix, source_key in (
            ("decode", "decode_token_ms"),
            ("first_run", "first_run_decode_ms"),
            ("steady_state", "steady_state_decode_ms"),
        ):
            values = valid_lists.get(source_key)
            if values is None:
                continue
            expected_stats = _stats(values)
            for suffix in ("p50_ms", "p95_ms"):
                key = f"{prefix}_{suffix}"
                if not _finite_number(timing.get(key)) or not math.isclose(
                    float(timing[key]),
                    expected_stats[suffix],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    errors.append(f"row {index} timing {key} disagrees with samples")
            if prefix == "decode":
                if not _finite_number(timing.get("decode_mean_ms")) or not math.isclose(
                    float(timing["decode_mean_ms"]),
                    _stats(values)["mean_ms"],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    errors.append(
                        f"row {index} timing decode_mean_ms disagrees with samples"
                    )
    statuses_seen = {row.get("status") for row in rows if isinstance(row, Mapping)}
    expected_status = (
        "unavailable"
        if statuses_seen == {"unavailable"}
        else "diagnostic"
        if statuses_seen == {"completed"}
        else "partial"
    )
    if artifact.get("status") != expected_status:
        errors.append(f"status must be {expected_status!r} for the row outcomes")
    return ValidationResult(not errors, tuple(errors))


def _run_lane(
    model: Any,
    profile: Any,
    prompt_ids: Any,
    decode_ids: Any,
    backend: str,
    config: Mapping[str, Any],
    torch_module: Any,
) -> tuple[dict[str, Any], list[Any]]:
    from core.qwen_adapter import install_qwen25_packed_backend

    warmup_tokens = int(config["warmup_tokens"])
    measured_tokens = int(config["measured_decode_tokens"])
    repeats = int(config["repeats"])
    parallel = backend == "parallel_direct"
    captured: list[Any] = []
    prefill_times: list[float] = []
    warmup_times: list[float] = []
    first_run_times: list[float] = []
    steady_state_times: list[float] = []
    decode_times: list[float] = []
    dispatch_totals: dict[str, int | bool] = {
        "standard_attention_calls": 0,
        "standard_prefill_calls": 0,
        "packed_decode_calls": 0,
        "packed_reference_decode_calls": 0,
        "parallel_decode_calls": 0,
        "parallel_page_launches": 0,
        "parallel_reduction_launches": 0,
        "parallel_tail_launches": 0,
        "fallback": False,
    }
    last_memory: dict[str, int] | None = None
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
    with torch_module.inference_mode():
        for _repeat in range(repeats):
            if backend == "baseline":
                prefill, elapsed = _timed(
                    lambda: model(prompt_ids, use_cache=True, logits_to_keep=1),
                    torch_module,
                )
                cache = prefill.past_key_values
                captured.append(_snapshot_logits(prefill.logits))
                prefill_times.append(elapsed)
                warmup_outputs: list[Any] = []
                for index in range(warmup_tokens):
                    token = decode_ids[:, index : index + 1]
                    output, warmup_elapsed = _timed(
                        lambda token=token: model(
                            token, past_key_values=cache, use_cache=True
                        ),
                        torch_module,
                    )
                    cache = output.past_key_values
                    warmup_outputs.append(_snapshot_logits(output.logits))
                    warmup_times.append(warmup_elapsed)
                captured.extend(warmup_outputs)
                for index in range(measured_tokens):
                    token_index = warmup_tokens + index
                    token = decode_ids[:, token_index : token_index + 1]
                    output, measured_elapsed = _timed(
                        lambda token=token: model(
                            token, past_key_values=cache, use_cache=True
                        ),
                        torch_module,
                    )
                    cache = output.past_key_values
                    captured.append(_snapshot_logits(output.logits))
                    decode_times.append(measured_elapsed)
                    (first_run_times if index == 0 else steady_state_times).append(
                        measured_elapsed
                    )
                last_memory = _baseline_memory(cache)
                dispatch_totals["standard_attention_calls"] += int(
                    model.config.num_hidden_layers
                    * (1 + warmup_tokens + measured_tokens)
                )
                dispatch_totals["standard_prefill_calls"] += int(
                    model.config.num_hidden_layers
                )
            else:
                session = install_qwen25_packed_backend(
                    model,
                    profile,
                    maximum_sequence_length=prompt_ids.shape[1]
                    + warmup_tokens
                    + measured_tokens
                    if parallel
                    else None,
                    use_parallel_page_attention=parallel,
                )
                session.use_packed_reference_backend = backend == "packed_reference"
                with session:
                    prefill, elapsed = _timed(
                        lambda session=session: session.prefill(prompt_ids),
                        torch_module,
                    )
                    captured.append(_snapshot_logits(prefill.logits))
                    prefill_times.append(elapsed)
                    for index in range(warmup_tokens):
                        token = decode_ids[:, index : index + 1]
                        output, warmup_elapsed = _timed(
                            lambda token=token, session=session: session.decode_one(
                                token
                            ),
                            torch_module,
                        )
                        captured.append(_snapshot_logits(output.logits))
                        warmup_times.append(warmup_elapsed)
                    for index in range(measured_tokens):
                        token_index = warmup_tokens + index
                        token = decode_ids[:, token_index : token_index + 1]
                        output, measured_elapsed = _timed(
                            lambda token=token, session=session: session.decode_one(
                                token
                            ),
                            torch_module,
                        )
                        captured.append(_snapshot_logits(output.logits))
                        decode_times.append(measured_elapsed)
                        (first_run_times if index == 0 else steady_state_times).append(
                            measured_elapsed
                        )
                    last_memory = _direct_memory(session, parallel=parallel)
                    for key in dispatch_totals:
                        if key != "fallback":
                            dispatch_totals[key] += int(
                                getattr(session.metrics, key, 0)
                            )
    if last_memory is None:
        raise RuntimeError("lane produced no cache memory accounting")
    last_memory["runtime_peak_allocated_bytes"] = 0
    last_memory["runtime_peak_reserved_bytes"] = 0
    runtime = _memory_runtime(torch_module)
    last_memory["runtime_peak_allocated_bytes"] = runtime["peak_allocated_bytes"] or 0
    last_memory["runtime_peak_reserved_bytes"] = runtime["peak_reserved_bytes"] or 0
    return {
        "status": "completed",
        "timing": _timing_summary(
            prefill_times,
            warmup_times,
            first_run_times,
            steady_state_times,
            decode_times,
        ),
        "dispatch": dispatch_totals,
        "memory": last_memory,
    }, captured


def _unavailable_row(
    context: int,
    backend: str,
    source: Mapping[str, Any],
    reason: str,
    status: str = "unavailable",
) -> dict[str, Any]:
    row = {
        "row_id": f"{context}-{backend}",
        "context_tokens": context,
        "backend": backend,
        "status": status,
        "reason": reason,
        "source_identity": dict(source),
    }
    if status in {"error", "oom"}:
        row["exception"] = reason
    return row


def run_warmup(
    config_path: str | Path,
    output: str | Path,
    *,
    allow_download: bool = False,
    cache_dir: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_warmup_config(config_path)
    output_path = Path(output)
    source = _source_identity(Path(config_path))
    artifact = build_unavailable_artifact(
        config, config_path=config_path, reason="model fixture not attempted"
    )
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
                _unavailable_row(context, backend, source, reason)
                for context in config["contexts"]
                for backend in config["backends"]
            ]
            write_json(str(output_path), artifact)
            return artifact
        source = {**source, "model_fixture": fixture}
        artifact["source_identity"] = source
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    stream_seed = int(config["stream_seed"])
    seed_everything(stream_seed, deterministic=True)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the fixed warmup benchmark")
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
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        artifact["rows"] = [
            _unavailable_row(context, backend, source, reason)
            for context in config["contexts"]
            for backend in config["backends"]
        ]
        artifact["status"] = "unavailable"
        write_json(str(output_path), artifact)
        return artifact

    from core.profiles import load_profile

    profile = load_profile(ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json")
    rows: list[dict[str, Any]] = []
    for context in config["contexts"]:
        total_tokens = (
            context
            + int(config["warmup_tokens"])
            + int(config["measured_decode_tokens"])
        )
        stream = _fixed_stream(tokenizer, total_tokens, stream_seed + context)
        prompt_ids = stream[:, :context].to(model.device)
        decode_ids = stream[:, context:].to(model.device)
        baseline_row: dict[str, Any] | None = None
        baseline_logits: list[Any] | None = None
        try:
            baseline_row, baseline_logits = _run_lane(
                model, profile, prompt_ids, decode_ids, "baseline", config, torch
            )
            baseline_row.update(
                {
                    "row_id": f"{context}-baseline",
                    "context_tokens": context,
                    "backend": "baseline",
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
        except torch.cuda.OutOfMemoryError as exc:
            baseline_row = _unavailable_row(
                context, "baseline", source, str(exc), "oom"
            )
            baseline_logits = None
        except Exception as exc:
            baseline_row = _unavailable_row(
                context, "baseline", source, f"{type(exc).__name__}: {exc}", "error"
            )
            baseline_row["traceback"] = traceback.format_exc()
            baseline_logits = None
        rows.append(baseline_row)
        for backend in config["backends"][1:]:
            if baseline_logits is None:
                rows.append(
                    _unavailable_row(
                        context,
                        backend,
                        source,
                        "baseline row unavailable; paired comparison was not run",
                    )
                )
                continue
            try:
                row, candidate_logits = _run_lane(
                    model, profile, prompt_ids, decode_ids, backend, config, torch
                )
                quality = _quality(baseline_logits, candidate_logits)
                gates = config["quality_gates"]
                row.update(
                    {
                        "row_id": f"{context}-{backend}",
                        "context_tokens": context,
                        "backend": backend,
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
            except torch.cuda.OutOfMemoryError as exc:
                row = _unavailable_row(context, backend, source, str(exc), "oom")
            except Exception as exc:
                row = _unavailable_row(
                    context, backend, source, f"{type(exc).__name__}: {exc}", "error"
                )
                row["traceback"] = traceback.format_exc()
            rows.append(row)
    statuses = {row.get("status") for row in rows}
    artifact.update(
        {
            "rows": rows,
            "status": "diagnostic" if statuses == {"completed"} else "partial",
        }
    )
    validation = validate_warmup_artifact(artifact, config)
    if not validation.valid:
        artifact["status"] = "invalid"
        artifact["validation_errors"] = list(validation.errors)
    write_json(str(output_path), artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--model-path")
    args = parser.parse_args()
    artifact = run_warmup(
        args.config,
        args.output,
        allow_download=args.allow_download,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
    )
    validation = validate_warmup_artifact(artifact, load_warmup_config(args.config))
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
