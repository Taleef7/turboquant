#!/usr/bin/env python3
"""Interleaved, measurement-first systems evidence for the direct adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.profiles import load_profile
from core.qwen_adapter import install_qwen25_packed_backend
from scripts.repro_utils import (
    PINNED_QWEN25_HUB_REVISION,
    PINNED_QWEN25_MODEL_ID,
    benchmark_binding,
    capture_env_metadata,
    git_commit_sha,
    write_json,
    write_markdown,
)


BACKENDS = ("baseline", "packed_reference", "triton_direct")
MODEL_ID = PINNED_QWEN25_MODEL_ID
MODEL_REVISION = PINNED_QWEN25_HUB_REVISION


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    index = (len(values) - 1) * percentile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _store_measurements(session) -> tuple[int, dict[str, int]]:
    totals = {"keys": 0, "values": 0, "metadata": 0, "fp16_window": 0, "padding": 0}
    for layer in range(session.capabilities.layer_count):
        breakdown = session.cache.store_for(layer).physical_nbytes()
        totals["keys"] += breakdown.key_payload
        totals["values"] += breakdown.value_payload
        totals["metadata"] += breakdown.metadata
        totals["fp16_window"] += breakdown.fp16_window
        totals["padding"] += breakdown.padding
    return sum(totals.values()), totals


def _fixed_prompt(tokenizer, tokens: int) -> torch.Tensor:
    text = "Evidence runner prompt. The retrieval token is ALPHA-3141. "
    ids = tokenizer(text * max(1, tokens // 8), return_tensors="pt").input_ids
    return ids[:, :tokens]


def _run_once(model, profile, prompt_ids: torch.Tensor, backend: str, decode_tokens: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trial = {"backend": backend, "status": "completed", "oom": False, "exception": None}
    try:
        with torch.inference_mode():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if backend == "baseline":
                prefill = model(prompt_ids, use_cache=True)
                prefill_ms = _elapsed_ms(start, end)
                cache = prefill.past_key_values
                token = prefill.logits[:, -1:].argmax(dim=-1)
                store_bytes, physical = 0, {"keys": 0, "values": 0, "metadata": 0, "fp16_window": 0, "padding": 0}
                dispatch = {"standard_attention": profile.model_architecture["num_hidden_layers"] * decode_tokens}
                decode_ms: list[float] = []
                ttft = None
                for _ in range(decode_tokens):
                    start.record()
                    output = model(token, past_key_values=cache, use_cache=True)
                    ms = _elapsed_ms(start, end)
                    if ttft is None:
                        ttft = ms
                    decode_ms.append(ms)
                    cache, token = output.past_key_values, output.logits[:, -1:].argmax(dim=-1)
            else:
                session = install_qwen25_packed_backend(model, profile)
                session.use_packed_reference_backend = backend == "packed_reference"
                prefill = session.prefill(prompt_ids)
                prefill_ms = _elapsed_ms(start, end)
                token = prefill.logits[:, -1:].argmax(dim=-1)
                decode_ms = []
                ttft = None
                for _ in range(decode_tokens):
                    start.record()
                    output = session.decode_one(token)
                    ms = _elapsed_ms(start, end)
                    if ttft is None:
                        ttft = ms
                    decode_ms.append(ms)
                    token = output.logits[:, -1:].argmax(dim=-1)
                store_bytes, physical = _store_measurements(session)
                dispatch = {
                    "triton_direct": session.metrics.packed_decode_calls,
                    "packed_reference": session.metrics.packed_reference_decode_calls,
                    "packed_blocks_consumed": session.metrics.packed_blocks_consumed,
                }
        trial.update(
            {
                "actual_store_bytes": store_bytes,
                "physical_bytes": physical,
                "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
                "prefill_ms": prefill_ms,
                "ttft_ms": ttft,
                "decode_token_ms": decode_ms,
                "decode_p50_ms": _percentile(decode_ms, 0.5),
                "decode_p95_ms": _percentile(decode_ms, 0.95),
                "decode_tokens_per_second": 1000.0 / statistics.fmean(decode_ms),
                "dispatch_counts": dispatch,
            }
        )
    except torch.cuda.OutOfMemoryError:
        trial.update({"status": "oom", "oom": True, "exception": "CUDA out of memory"})
        torch.cuda.empty_cache()
    except Exception as exc:  # The artifact must preserve unexpected failures.
        trial.update({"status": "error", "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    return trial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="systems JSON artifact path")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--probe-contexts", type=str, default="", help="comma-separated real OOM probe contexts")
    args = parser.parse_args()
    output = Path(args.output)
    run_dir = output.parent / f"{output.stem}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{git_commit_sha()[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    def emit(message: str, *, error: bool = False) -> None:
        stream = sys.stderr if error else sys.stdout
        print(message, file=stream, flush=True)
        with (stderr_log if error else stdout_log).open("a", encoding="utf-8") as captured:
            print(message, file=captured, flush=True)
    emit(f"systems runner started: prompt_tokens={args.prompt_tokens} probe_contexts={args.probe_contexts or 'default'}")
    profile = load_profile(ROOT / "configs/profiles/qwen25-7b-direct-v1.json")
    if (profile.revision.model_id, profile.revision.revision) != (
        MODEL_ID,
        MODEL_REVISION,
    ):
        raise RuntimeError("direct benchmark profile does not bind the pinned Hub revision")
    artifact = {
        "artifact_type": "systems_benchmark", "schema_version": 1,
        "binding": benchmark_binding(ROOT), "status": "inconclusive",
        "metadata": capture_env_metadata(), "expected_backends": list(BACKENDS),
        "configuration": vars(args), "trials": [], "warmups": [],
        "oom_boundary": {"attempted_contexts": [], "status": "not_reached_within_configured_probe_range", "probes": []},
        "artifact_directory": str(run_dir),
    }
    if not torch.cuda.is_available():
        artifact["status"] = "failed"
        artifact["failure"] = "CUDA is required for systems evidence"
        write_json(str(run_dir / "systems.json"), artifact)
        write_json(str(output), artifact)
        emit(artifact["failure"], error=True)
        raise SystemExit(2)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            torch_dtype=torch.float16, device_map="cuda",
        ).eval()
        prompt = _fixed_prompt(tokenizer, args.prompt_tokens).to(model.device)
        for backend in BACKENDS:
            for _ in range(args.warmups):
                warmup = _run_once(model, profile, prompt, backend, min(2, args.decode_tokens))
                artifact["warmups"].append(warmup)
        for trial_index in range(args.trials):
            for backend in BACKENDS:
                trial = _run_once(model, profile, prompt, backend, args.decode_tokens)
                trial["trial_index"] = trial_index
                artifact["trials"].append(trial)
        probe_contexts = [int(value) for value in args.probe_contexts.split(",") if value] if args.probe_contexts else [max(512, args.prompt_tokens // 2), args.prompt_tokens]
        probe_contexts = sorted(set(probe_contexts))
        for context in probe_contexts:
            probe_prompt = _fixed_prompt(tokenizer, context).to(model.device)
            probe = _run_once(model, profile, probe_prompt, "triton_direct", 1)
            probe["context_tokens"] = context
            artifact["oom_boundary"]["probes"].append(probe)
            artifact["oom_boundary"]["attempted_contexts"].append(context)
        all_outcomes = artifact["warmups"] + artifact["trials"]
        probes = artifact["oom_boundary"]["probes"]
        if any(trial["oom"] for trial in all_outcomes + probes):
            artifact["oom_boundary"]["status"] = "oom"
            artifact["status"] = "invalid"
        elif any(trial["status"] != "completed" for trial in all_outcomes + probes):
            artifact["status"] = "invalid"
        else:
            artifact["status"] = "completed"
    except Exception as exc:
        artifact.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        emit(artifact["failure"], error=True)
    markdown = "# Systems benchmark\n\n```json\n" + json.dumps(artifact, indent=2) + "\n```\n"
    write_json(str(run_dir / "systems.json"), artifact)
    write_markdown(str(run_dir / "systems.md"), markdown)
    write_json(str(run_dir / "command.json"), {"argv": sys.argv, "metadata": artifact["metadata"]})
    emit(f"systems runner finished: status={artifact['status']}")
    write_json(str(output), artifact)
    write_markdown(str(output.with_suffix(".md")), markdown)
    if artifact["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
