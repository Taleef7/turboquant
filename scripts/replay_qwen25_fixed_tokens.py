#!/usr/bin/env python3
"""Validator-first fixed-token replay for the Qwen2.5 direct decode lanes.

The runner deliberately compares the same token IDs through four routes:
baseline HF attention, packed-reference attention, serial direct attention, and
the opt-in fixed-capacity parallel-page path.  It never substitutes a different
prompt when a model fixture is unavailable; instead every declared row is
persisted as ``unavailable`` with the loader error.  The result is diagnostic
only and publishes no speed, VRAM, or quality claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_schema import ValidationResult  # noqa: E402
from scripts.repro_utils import (  # noqa: E402
    PINNED_QWEN25_HUB_REVISION,
    PINNED_QWEN25_MODEL_ID,
    benchmark_binding,
    capture_env_metadata,
    capture_source_metadata,
    git_commit_sha,
    sha256_file,
    seed_everything,
    write_json,
)


BACKENDS = ("baseline", "packed_reference", "serial_direct", "parallel_direct")
CONFIG_PATH = ROOT / "configs" / "benchmarks" / "qwen25-fixed-replay-v1.json"


def load_replay_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the predeclared replay matrix before model loading."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("artifact_type") != "qwen25_fixed_token_replay":
        errors.append("artifact_type must be qwen25_fixed_token_replay")
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("model_id") != PINNED_QWEN25_MODEL_ID:
        errors.append("model_id must bind the pinned Qwen2.5 model")
    if payload.get("model_revision") != PINNED_QWEN25_HUB_REVISION:
        errors.append("model_revision must bind the pinned Qwen2.5 revision")
    manifest_sha256 = payload.get("fixture_manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        errors.append("fixture_manifest_sha256 must be a lowercase SHA-256 digest")
    contexts = payload.get("contexts")
    if contexts != [1024, 2048]:
        errors.append("contexts must be the predeclared [1024, 2048] matrix")
    if type(payload.get("decode_tokens")) is not int or payload["decode_tokens"] < 1:  # noqa: E721
        errors.append("decode_tokens must be a positive built-in int")
    if type(payload.get("stream_seed")) is not int:  # noqa: E721
        errors.append("stream_seed must be a built-in int")
    if tuple(payload.get("backends", ())) != BACKENDS:
        errors.append("backends must be the four declared replay lanes")
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
        raise ValueError("invalid replay config: " + "; ".join(errors))
    return payload


def expected_row_ids(config: Mapping[str, Any]) -> list[str]:
    return [
        f"{context}-{backend}"
        for context in config["contexts"]
        for backend in config["backends"]
    ]


def _source_identity(config_path: Path) -> dict[str, Any]:
    return {
        **capture_source_metadata(ROOT),
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(config_path),
        "model_id": PINNED_QWEN25_MODEL_ID,
        "model_revision": PINNED_QWEN25_HUB_REVISION,
    }


def _fixture_identity(
    model_path: str | Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Verify and fingerprint a direct HF snapshot before loading any weights.

    A path override is only accepted when it is the exact pinned Hugging Face
    snapshot layout.  This prevents a differently trained local model from
    being reported under the pinned Qwen identity.
    """
    snapshot = Path(model_path).expanduser().resolve()
    model_id = str(config["model_id"])
    revision = str(config["model_revision"])
    expected_cache_name = f"models--{model_id.replace('/', '--')}"
    if (
        not snapshot.is_dir()
        or snapshot.name != revision
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != expected_cache_name
    ):
        return None, (
            "local model path must resolve to the pinned HF snapshot layout "
            f"{expected_cache_name}/snapshots/{revision}"
        )

    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    }
    missing = sorted(name for name in required if not (snapshot / name).is_file())
    if missing:
        return None, "pinned model snapshot is missing required files: " + ", ".join(
            missing
        )
    try:
        model_config = json.loads(
            (snapshot / "config.json").read_text(encoding="utf-8")
        )
        index = json.loads(
            (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"pinned model snapshot metadata is unreadable: {exc}"

    expected_architecture = {
        "model_type": "qwen2",
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
    }
    architecture_mismatches = [
        f"{key}={model_config.get(key)!r} (expected {value!r})"
        for key, value in expected_architecture.items()
        if model_config.get(key) != value
    ]
    if architecture_mismatches:
        return None, "model config architecture mismatch: " + "; ".join(
            architecture_mismatches
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        return None, "model.safetensors.index.json has no weight_map"
    shard_names = sorted({str(name) for name in weight_map.values()})
    missing_shards = [name for name in shard_names if not (snapshot / name).is_file()]
    if missing_shards:
        return None, "pinned model snapshot is missing weight shards: " + ", ".join(
            missing_shards
        )

    files: list[dict[str, Any]] = []
    for relative in sorted(required | set(shard_names)):
        path = snapshot / relative
        try:
            files.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        except OSError as exc:
            return None, f"pinned model snapshot file is unreadable ({relative}): {exc}"
    manifest_bytes = json.dumps(files, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_sha256 = config.get("fixture_manifest_sha256")
    if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
        return None, (
            "local fixture manifest SHA-256 does not match the predeclared "
            f"pinned digest (got {manifest_sha256})"
        )
    return {
        "model_id": model_id,
        "snapshot_id": revision,
        "resolved_path": str(snapshot),
        "architecture": expected_architecture,
        "files": files,
        "manifest_sha256": manifest_sha256,
    }, None


def build_unavailable_artifact(
    config: Mapping[str, Any], *, config_path: str | Path, reason: str
) -> dict[str, Any]:
    """Create a complete diagnostic artifact without inventing measurements."""
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
        "artifact_type": "qwen25_fixed_token_replay",
        "schema_version": 1,
        "artifact_id": f"qwen25-fixed-replay-{git_commit_sha(ROOT)[:12]}",
        "status": "unavailable",
        "diagnostic_only": True,
        "claims": [],
        "config": dict(config),
        "binding": benchmark_binding(ROOT),
        "metadata": capture_env_metadata(),
        "source_identity": source,
        "rows": rows,
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_replay_artifact(
    artifact: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> ValidationResult:
    """Validate exact row identity and prevent unavailable data becoming a claim."""
    errors: list[str] = []
    if not isinstance(artifact, Mapping):
        return ValidationResult(False, ("artifact must be an object",))
    config = dict(config or load_replay_config())
    if artifact.get("artifact_type") != "qwen25_fixed_token_replay":
        errors.append("artifact_type must be qwen25_fixed_token_replay")
    if artifact.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if artifact.get("diagnostic_only") is not True:
        errors.append("diagnostic_only must be true")
    if artifact.get("claims") != []:
        errors.append("diagnostic artifact claims must be empty")
    if artifact.get("config") != config:
        errors.append("artifact config does not match the predeclared replay config")
    expected = expected_row_ids(config)
    rows = artifact.get("rows")
    observed_row_ids = (
        [row.get("row_id") if isinstance(row, Mapping) else None for row in rows]
        if isinstance(rows, list)
        else None
    )
    if observed_row_ids != expected:
        errors.append("rows must exactly match the predeclared context/backend matrix")
        rows = rows if isinstance(rows, list) else []
    statuses = {"completed", "unavailable", "error", "oom"}
    expected_decode_tokens = int(config["decode_tokens"])
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"row {index} must be an object")
            continue
        status = row.get("status")
        if index < len(expected):
            expected_context, expected_backend = expected[index].split("-", 1)
            if row.get("context_tokens") != int(expected_context):
                errors.append(f"row {index} context_tokens do not match row_id")
            if row.get("backend") != expected_backend:
                errors.append(f"row {index} backend does not match row_id")
        if status not in statuses:
            errors.append(f"row {index} has invalid status")
        if not isinstance(row.get("source_identity"), Mapping):
            errors.append(f"row {index} source_identity is missing")
        if status == "unavailable" and not isinstance(row.get("reason"), str):
            errors.append(f"row {index} unavailable reason is missing")
        if status in {"error", "oom"} and not isinstance(row.get("exception"), str):
            errors.append(f"row {index} exception is missing")
        if status == "completed":
            for key in ("quality", "dispatch", "memory", "timing"):
                if not isinstance(row.get(key), Mapping):
                    errors.append(f"row {index} completed {key} is missing")
            if not isinstance(row.get("quality_gate_pass"), bool):
                errors.append(f"row {index} quality_gate_pass is missing")
            if all(
                isinstance(row.get(key), Mapping)
                for key in ("quality", "dispatch", "memory", "timing")
            ):
                quality = row["quality"]
                gates = config["quality_gates"]
                numeric_quality = ("max_abs_logit", "mean_abs_logit", "kl_divergence")
                quality_numbers_valid = all(
                    _finite_number(quality.get(key)) and float(quality[key]) >= 0
                    for key in numeric_quality
                )
                if not quality_numbers_valid:
                    errors.append(f"row {index} quality numeric metrics are invalid")
                top1 = quality.get("top1_agreement")
                top1_valid = _finite_number(top1) and 0.0 <= float(top1) <= 1.0
                if not top1_valid:
                    errors.append(f"row {index} top1_agreement is invalid")
                greedy = quality.get("greedy_agreement")
                if not isinstance(greedy, bool):
                    errors.append(f"row {index} greedy_agreement is invalid")
                computed_gate = False
                if quality_numbers_valid and top1_valid and isinstance(greedy, bool):
                    computed_gate = (
                        float(quality["max_abs_logit"]) <= gates["max_abs_logit"]
                        and float(quality["mean_abs_logit"]) <= gates["mean_abs_logit"]
                        and float(quality["kl_divergence"]) <= gates["kl_divergence"]
                        and float(top1) >= gates["top1_agreement"]
                        and greedy is True
                    )
                if row.get("quality_gate_pass") is not computed_gate:
                    errors.append(
                        f"row {index} quality_gate_pass disagrees with quality metrics"
                    )
                backend = row.get("backend")
                expected_layer_calls = expected_decode_tokens * 28
                if (
                    backend == "baseline"
                    and quality_numbers_valid
                    and top1_valid
                    and (
                        any(float(quality[key]) != 0.0 for key in numeric_quality)
                        or top1 != 1.0
                        or greedy is not True
                    )
                ):
                    errors.append(
                        f"row {index} baseline quality must be exact zero-drift"
                    )
                dispatch = row["dispatch"]
                if dispatch.get("fallback") is not False:
                    errors.append(f"row {index} fallback must be false")
                if backend == "baseline" and (
                    not _finite_number(dispatch.get("standard_attention_calls"))
                    or dispatch["standard_attention_calls"] <= 0
                ):
                    errors.append(f"row {index} baseline attention dispatch is missing")
                if (
                    backend == "packed_reference"
                    and dispatch.get("packed_reference_decode_calls", 0)
                    != expected_layer_calls
                ):
                    errors.append(
                        f"row {index} packed-reference dispatch count is invalid"
                    )
                if (
                    backend == "serial_direct"
                    and dispatch.get("packed_decode_calls", 0) != expected_layer_calls
                ):
                    errors.append(f"row {index} serial dispatch count is invalid")
                if backend == "parallel_direct":
                    for key in ("packed_decode_calls", "parallel_decode_calls"):
                        if dispatch.get(key, 0) != expected_layer_calls:
                            errors.append(
                                f"row {index} parallel {key} count is invalid"
                            )
                    for key in (
                        "parallel_page_launches",
                        "parallel_reduction_launches",
                        "parallel_tail_launches",
                    ):
                        if dispatch.get(key, 0) != expected_layer_calls:
                            errors.append(
                                f"row {index} parallel {key} count is invalid"
                            )
                memory = row["memory"]
                memory_keys = (
                    "logical_payload_bytes",
                    "physical_cache_bytes",
                    "arena_capacity_bytes",
                    "tail_capacity_bytes",
                    "workspace_bytes",
                    "unique_resident_bytes",
                )
                if any(
                    not isinstance(memory.get(key), int) or memory[key] < 0
                    for key in memory_keys
                ):
                    errors.append(f"row {index} memory byte accounting is invalid")
                else:
                    if memory["unique_resident_bytes"] < memory["physical_cache_bytes"]:
                        errors.append(
                            f"row {index} unique resident bytes undercount physical cache"
                        )
                    if backend == "parallel_direct" and (
                        memory["workspace_bytes"] <= 0
                        or memory["arena_capacity_bytes"] <= 0
                    ):
                        errors.append(
                            f"row {index} parallel memory reservation is missing"
                        )
                timing = row["timing"]
                if (
                    not isinstance(timing.get("decode_token_ms"), list)
                    or len(timing["decode_token_ms"]) != expected_decode_tokens
                ):
                    errors.append(f"row {index} timing decode-token count is invalid")
                elif any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                    for value in timing["decode_token_ms"]
                ):
                    errors.append(f"row {index} timing values are invalid")
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


def _fixed_stream(tokenizer: Any, total_tokens: int, seed: int) -> Any:
    """Build a deterministic token stream with no random sampling in the model."""
    import torch

    text = (
        "TurboQuant fixed replay. The retrieval marker is ALPHA-3141. "
        "Measure cache quality and memory without changing the token stream. "
    )
    ids = tokenizer(
        text * max(1, (total_tokens // 8) + 2), return_tensors="pt"
    ).input_ids[0]
    if ids.numel() < total_tokens:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        extra = torch.randint(
            0,
            int(tokenizer.vocab_size),
            (total_tokens - ids.numel(),),
            generator=generator,
        )
        ids = torch.cat((ids, extra), dim=0)
    return ids[:total_tokens].reshape(1, total_tokens)


def _sync_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _timed(fn: Any, torch_module: Any) -> tuple[Any, float]:
    _sync_cuda(torch_module)
    start = time.perf_counter()
    value = fn()
    _sync_cuda(torch_module)
    return value, (time.perf_counter() - start) * 1000.0


def _snapshot_logits(logits: Any) -> Any:
    return logits[:, -1].detach().float().cpu()


def _tensor_storage_bytes(value: Any, seen: set[int] | None = None) -> int:
    import torch

    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        pointer = value.untyped_storage().data_ptr()
        if pointer in seen:
            return 0
        seen.add(pointer)
        return value.untyped_storage().nbytes()
    if isinstance(value, Mapping):
        return sum(_tensor_storage_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_storage_bytes(item, seen) for item in value)
    if hasattr(value, "layers"):
        return _tensor_storage_bytes(getattr(value, "layers"), seen)
    if any(
        hasattr(value, name) for name in ("key_cache", "value_cache", "keys", "values")
    ):
        return sum(
            _tensor_storage_bytes(getattr(value, name, ()), seen)
            for name in ("key_cache", "value_cache", "keys", "values")
        )
    return 0


def _baseline_memory(cache: Any) -> dict[str, int]:
    value = _tensor_storage_bytes(cache)
    return {
        "logical_payload_bytes": value,
        "physical_cache_bytes": value,
        "arena_capacity_bytes": 0,
        "tail_capacity_bytes": 0,
        "workspace_bytes": 0,
        "unique_resident_bytes": value,
    }


def _direct_memory(session: Any, *, parallel: bool) -> dict[str, int]:
    from core.parallel_page_workspace import parallel_page_workspace_nbytes

    cache_bytes = 0
    logical = 0
    arena_capacity = 0
    tail_capacity = 0
    unique = 0
    for layer_idx in range(session.capabilities.layer_count):
        store = session.cache.store_for(layer_idx)
        breakdown = store.physical_nbytes()
        if store.arena is not None:
            arena = store.arena
            arena_breakdown = arena.byte_breakdown()
            # The arena owns all reserved page/tail storage.  Append scratch
            # is a separate per-store allocation; neither includes workspace.
            arena_storage = arena._unique_storage_bytes()
            cache_bytes += arena_storage + breakdown.workspace
            logical += arena_breakdown.logical_payload + arena_breakdown.tail_logical
            arena_capacity += arena_breakdown.arena_capacity
            tail_capacity += arena_breakdown.tail_capacity
            unique += arena_storage + breakdown.workspace
        else:
            cache_bytes += breakdown.total_nbytes
            logical += (
                breakdown.key_payload
                + breakdown.value_payload
                + breakdown.metadata
                + breakdown.fp16_window
            )
            unique += breakdown.total_nbytes
    workspace = 0
    if parallel and session._parallel_workspace is not None:
        workspace = parallel_page_workspace_nbytes(session._parallel_workspace)
        unique += workspace
    return {
        "logical_payload_bytes": logical,
        "physical_cache_bytes": cache_bytes,
        "arena_capacity_bytes": arena_capacity,
        "tail_capacity_bytes": tail_capacity,
        "workspace_bytes": workspace,
        "unique_resident_bytes": unique,
    }


def _quality(
    baseline: Sequence[Any], candidate: Sequence[Any]
) -> dict[str, float | bool]:
    import torch
    import torch.nn.functional as F

    base = torch.cat(tuple(baseline), dim=0)
    actual = torch.cat(tuple(candidate), dim=0)
    difference = (actual - base).abs()
    base_probability = base.softmax(dim=-1)
    kl = F.kl_div(actual.log_softmax(dim=-1), base_probability, reduction="batchmean")
    kl_value = float(kl.item())
    if math.isfinite(kl_value):
        # KL is mathematically non-negative; tiny negative values can arise
        # from the floating-point reduction and must not invalidate a row.
        kl_value = max(0.0, kl_value)
    base_top1 = base.argmax(dim=-1)
    actual_top1 = actual.argmax(dim=-1)
    agreement = (base_top1 == actual_top1).float()
    return {
        "max_abs_logit": float(difference.max().item()),
        "mean_abs_logit": float(difference.mean().item()),
        "kl_divergence": kl_value,
        "top1_agreement": float(agreement.mean().item()),
        "greedy_agreement": bool(torch.equal(base_top1, actual_top1)),
    }


def _memory_runtime(torch_module: Any) -> dict[str, int | None]:
    if not torch_module.cuda.is_available():
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
    }


def _run_backend(
    model: Any,
    profile: Any,
    prompt_ids: Any,
    decode_ids: Any,
    backend: str,
    torch_module: Any,
) -> tuple[dict[str, Any], list[Any]]:
    from core.qwen_adapter import install_qwen25_packed_backend

    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
    context_tokens = int(prompt_ids.shape[1])
    parallel = backend == "parallel_direct"
    session = None
    captured: list[Any] = []
    decode_times: list[float] = []
    try:
        with torch_module.inference_mode():
            if backend == "baseline":
                prefill, prefill_ms = _timed(
                    lambda: model(prompt_ids, use_cache=True, logits_to_keep=1),
                    torch_module,
                )
                captured.append(_snapshot_logits(prefill.logits))
                cache = prefill.past_key_values
                for index in range(int(decode_ids.shape[1])):
                    token = decode_ids[:, index : index + 1]
                    output, elapsed = _timed(
                        lambda token=token: model(
                            token, past_key_values=cache, use_cache=True
                        ),
                        torch_module,
                    )
                    cache = output.past_key_values
                    captured.append(_snapshot_logits(output.logits))
                    decode_times.append(elapsed)
                memory = _baseline_memory(cache)
                dispatch = {
                    "standard_attention_calls": int(
                        model.config.num_hidden_layers * (1 + decode_ids.shape[1])
                    ),
                    "packed_decode_calls": 0,
                    "parallel_page_launches": 0,
                    "fallback": False,
                }
            else:
                session = install_qwen25_packed_backend(
                    model,
                    profile,
                    maximum_sequence_length=context_tokens + int(decode_ids.shape[1])
                    if parallel
                    else None,
                    use_parallel_page_attention=parallel,
                )
                session.use_packed_reference_backend = backend == "packed_reference"
                with session:
                    prefill, prefill_ms = _timed(
                        lambda session=session: session.prefill(prompt_ids),
                        torch_module,
                    )
                    captured.append(_snapshot_logits(prefill.logits))
                    for index in range(int(decode_ids.shape[1])):
                        token = decode_ids[:, index : index + 1]
                        output, elapsed = _timed(
                            lambda token=token, session=session: session.decode_one(
                                token
                            ),
                            torch_module,
                        )
                        captured.append(_snapshot_logits(output.logits))
                        decode_times.append(elapsed)
                    memory = _direct_memory(session, parallel=parallel)
                    dispatch = {
                        "standard_prefill_calls": session.metrics.standard_prefill_calls,
                        "packed_decode_calls": session.metrics.packed_decode_calls,
                        "packed_reference_decode_calls": session.metrics.packed_reference_decode_calls,
                        "parallel_decode_calls": session.metrics.parallel_decode_calls,
                        "parallel_page_launches": session.metrics.parallel_page_launches,
                        "parallel_reduction_launches": session.metrics.parallel_reduction_launches,
                        "parallel_tail_launches": session.metrics.parallel_tail_launches,
                        "fallback": False,
                    }
        row = {
            "status": "completed",
            "timing": {
                "prefill_ms": prefill_ms,
                "decode_token_ms": decode_times,
                "decode_p50_ms": statistics.median(decode_times),
                "decode_mean_ms": statistics.fmean(decode_times),
            },
            "dispatch": dispatch,
            "memory": {
                **memory,
                "runtime": _memory_runtime(torch_module),
            },
        }
        return row, captured
    finally:
        del session
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()


def _unavailable_row(
    context: int, backend: str, source: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    return {
        "row_id": f"{context}-{backend}",
        "context_tokens": context,
        "backend": backend,
        "status": "unavailable",
        "reason": reason,
        "source_identity": dict(source),
    }


def run_replay(
    config_path: str | Path,
    output: str | Path,
    *,
    allow_download: bool = False,
    cache_dir: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_replay_config(config_path)
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
            artifact["status"] = "unavailable"
            validation = validate_replay_artifact(artifact, config)
            if not validation.valid:
                raise RuntimeError(
                    "unavailable artifact failed validation: "
                    + "; ".join(validation.errors)
                )
            write_json(str(output_path), artifact)
            return artifact
        assert fixture is not None
        source = {**source, "model_fixture": fixture}
        artifact["source_identity"] = source
    # PyTorch requires this before its first CUDA/cuBLAS operation for the
    # deterministic algorithms requested by the replay contract.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(config["stream_seed"], deterministic=True)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the 7B fixed-token replay")
        load_kwargs = {
            "local_files_only": not allow_download,
            "trust_remote_code": False,
        }
        model_source = config["model_id"] if model_path is None else str(model_path)
        if model_path is None:
            load_kwargs["revision"] = config["model_revision"]
        if cache_dir is not None:
            load_kwargs["cache_dir"] = str(cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
        model_kwargs: dict[str, Any] = {
            **load_kwargs,
            "torch_dtype": torch.float16,
            "device_map": "cuda",
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_source, **model_kwargs
        ).eval()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        artifact["rows"] = [
            _unavailable_row(context, backend, source, reason)
            for context in config["contexts"]
            for backend in config["backends"]
        ]
        artifact["status"] = "unavailable"
        validation = validate_replay_artifact(artifact, config)
        if not validation.valid:
            raise RuntimeError(
                "unavailable artifact failed validation: "
                + "; ".join(validation.errors)
            )
        write_json(str(output_path), artifact)
        return artifact

    profile_path = ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json"
    from core.profiles import load_profile

    profile = load_profile(profile_path)
    rows: list[dict[str, Any]] = []
    for context in config["contexts"]:
        stream = _fixed_stream(
            tokenizer,
            context + config["decode_tokens"],
            config["stream_seed"] + context,
        )
        prompt_ids = stream[:, :context].to(model.device)
        decode_ids = stream[:, context:].to(model.device)
        try:
            baseline_row, baseline_logits = _run_backend(
                model, profile, prompt_ids, decode_ids, "baseline", torch
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
                context, "baseline", source, "CUDA out of memory"
            )
            baseline_row["status"], baseline_row["exception"] = "oom", str(exc)
            baseline_logits = None
        except Exception as exc:
            baseline_row = _unavailable_row(
                context, "baseline", source, f"{type(exc).__name__}: {exc}"
            )
            baseline_row["status"], baseline_row["exception"] = "error", str(exc)
            baseline_logits = None
        rows.append(baseline_row)
        for backend in config["backends"][1:]:
            row_id = f"{context}-{backend}"
            if baseline_logits is None:
                row = _unavailable_row(
                    context,
                    backend,
                    source,
                    "baseline row unavailable; paired comparison was not run",
                )
                rows.append(row)
                continue
            try:
                row, candidate_logits = _run_backend(
                    model, profile, prompt_ids, decode_ids, backend, torch
                )
                quality = _quality(baseline_logits, candidate_logits)
                gates = config["quality_gates"]
                gate_pass = (
                    quality["max_abs_logit"] <= gates["max_abs_logit"]
                    and quality["mean_abs_logit"] <= gates["mean_abs_logit"]
                    and quality["kl_divergence"] <= gates["kl_divergence"]
                    and quality["top1_agreement"] >= gates["top1_agreement"]
                    and quality["greedy_agreement"] is True
                )
                row.update(
                    {
                        "row_id": row_id,
                        "context_tokens": context,
                        "backend": backend,
                        "source_identity": source,
                        "quality": quality,
                        "quality_gate_pass": gate_pass,
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                row = _unavailable_row(context, backend, source, "CUDA out of memory")
                row["status"], row["exception"] = "oom", str(exc)
            except Exception as exc:
                row = _unavailable_row(
                    context, backend, source, f"{type(exc).__name__}: {exc}"
                )
                row["status"], row["exception"], row["traceback"] = (
                    "error",
                    str(exc),
                    traceback.format_exc(),
                )
            rows.append(row)
    row_statuses = {row.get("status") for row in rows}
    artifact.update(
        {
            "rows": rows,
            "status": "diagnostic" if row_statuses == {"completed"} else "partial",
        }
    )
    validation = validate_replay_artifact(artifact, config)
    if not validation.valid:
        artifact["status"] = "invalid"
        artifact["validation_errors"] = list(validation.errors)
    write_json(str(output_path), artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face network access; default is local-only.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory (useful for WSL-local fixtures).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local snapshot directory; bypasses Hub cache resolution.",
    )
    args = parser.parse_args()
    artifact = run_replay(
        args.config,
        args.output,
        allow_download=args.allow_download,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
    )
    validation = validate_replay_artifact(artifact)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "valid": validation.valid,
                "rows": len(artifact["rows"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if not validation.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
