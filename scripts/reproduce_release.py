#!/usr/bin/env python3
"""Validator-first release evidence runner for the direct Qwen profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_schema import quality_retrieval_safe, validate_quality_artifact, validate_systems_artifact
from scripts.repro_utils import benchmark_binding, capture_env_metadata, git_commit_sha, write_json, write_markdown


def qualified_ladder_decision(
    *,
    preflight_valid: bool,
    eight_k_retrieval_safe: bool,
    eight_k_systems_valid: bool = False,
    eight_k_capacity_admitted: bool = False,
) -> dict:
    """Return the persisted, mechanical decision for the 32K qualified leg."""
    inputs = {
        "preflight_valid": preflight_valid,
        "eight_k_retrieval_safe": eight_k_retrieval_safe,
        "eight_k_systems_valid": eight_k_systems_valid,
        "eight_k_capacity_admitted": eight_k_capacity_admitted,
    }
    if not preflight_valid:
        return {**inputs, "status": "skipped", "attempt_32k": False, "reason": "preflight_invalid"}
    if not eight_k_retrieval_safe:
        return {**inputs, "status": "skipped", "attempt_32k": False, "reason": "8k_retrieval_gate_failed"}
    if not eight_k_systems_valid:
        return {**inputs, "status": "skipped", "attempt_32k": False, "reason": "8k_systems_invalid_or_failed"}
    if not eight_k_capacity_admitted:
        return {**inputs, "status": "skipped", "attempt_32k": False, "reason": "8k_capacity_admission_failed"}
    return {**inputs, "status": "approved", "attempt_32k": True, "reason": "preflight_and_8k_gate_passed"}


def estimate_qualified_capacity(artifact: dict, *, target_context: int, source_context: int) -> dict:
    """Project 32K reservation from measured 1K packed-cache behavior.

    Assumption: model/static allocations are the measured reserved peak minus
    packed bytes; packed bytes scale linearly with context. A 25% cache growth
    contingency and 10% free-VRAM safety margin make the decision conservative.
    """
    trials = artifact.get("trials", [])
    reserved = max(int(row["max_memory_reserved"]) for row in trials)
    store = max(int(row["actual_store_bytes"]) for row in trials)
    baseline_workspace = max(0, reserved - store)
    projected_cache = int(store * target_context / source_context * 1.25)
    projected = baseline_workspace + projected_cache
    free = int(artifact["metadata"]["gpu_free_vram_bytes"])
    allowed = int(free * 0.90)
    return {"source_context": source_context, "target_context": target_context,
            "source_reserved_peak_bytes": reserved, "source_actual_cache_bytes": store,
            "baseline_workspace_bytes": baseline_workspace, "projected_cache_bytes": projected_cache, "projected_32k_bytes": projected,
            "free_vram_bytes": free, "allowed_bytes_after_safety_margin": allowed,
            "passes": projected <= allowed,
            "formula": "baseline_workspace=(8K reserved_peak-8K actual_cache_bytes); projected=baseline_workspace+(8K actual_cache_bytes*(32K/8K)*1.25); require <= free_vram*0.90",
            "source_binding": artifact["binding"]}


def run_command(command: list[str], log_prefix: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log_prefix.with_suffix(".stdout.log").write_text(result.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(result.stderr, encoding="utf-8")
    return result


def extract_delta_pp(niah_json_path: str) -> float | None:
    """Compatibility helper; partial matrices deliberately return no delta."""
    artifact = _load(Path(niah_json_path))
    rows = artifact.get("rows") if isinstance(artifact, dict) else artifact
    if not isinstance(rows, list):
        return None
    eligible = [row for row in rows if row.get("needle_present_after_tokenization")]
    if not eligible or any(row.get("status", "completed") != "completed" for row in eligible):
        return None
    return (sum(row["baseline_correct"] for row in eligible) - sum(row.get("profile_correct", row.get("turboquant_correct")) for row in eligible)) / len(eligible) * 100.0


def build_niah_command(mode: str, output_prefix: str) -> list[str]:
    if mode in {"quick", "eight_k"}:
        max_context, trials, buffer_size = "8192", "2", "128"
    elif mode == "full":  # Kept only for legacy helper callers.
        max_context, trials, buffer_size = "16384", "6", "12288"
    else:
        max_context, trials, buffer_size = "32768", "3", "128"
    return [sys.executable, "scripts/test_long_context.py", "--test", "niah", "--mode", "paired",
            "--model", "Qwen/Qwen2.5-7B-Instruct", "--max-context", max_context,
            "--key-bits", "8", "--value-bits", "6", "--buffer-size", buffer_size,
            "--trials", trials, "--output-prefix", output_prefix]


def build_systems_command(output: str, context: int) -> list[str]:
    probes = "1024,2048" if context <= 1024 else f"{max(1024, context // 2)},{context}"
    return [sys.executable, "scripts/run_direct_benchmark.py", "--output", output,
            "--prompt-tokens", str(context), "--decode-tokens", "64", "--trials", "1", "--warmups", "1", "--probe-contexts", probes]


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except json.JSONDecodeError:
        return None


def _child_summary(name: str, path: Path, result: subprocess.CompletedProcess[str], validator) -> dict:
    artifact = _load(path)
    validation = validator(artifact, benchmark_binding(ROOT)) if isinstance(artifact, dict) else None
    return {"name": name, "path": str(path), "exit_code": result.returncode,
            "artifact": artifact, "structurally_valid": bool(validation and validation.valid),
            "errors": list(validation.errors) if validation else ["artifact missing or invalid JSON"]}


def _quality_state(summary: dict) -> tuple[str, bool, object | None]:
    if not summary["structurally_valid"]:
        return "invalid", False, None
    safe, bootstrap, _ = quality_retrieval_safe(summary["artifact"], bootstrap_seed=1729)
    return ("qualified" if safe else "failed"), safe, bootstrap


def _run_leg(output_dir: Path, label: str, *, quality_mode: str, systems_context: int) -> tuple[dict, dict, str, bool, object | None]:
    quality_path = output_dir / f"{label}-quality.json"
    systems_path = output_dir / f"{label}-systems.json"
    quality_result = run_command(build_niah_command(quality_mode, str(quality_path.with_suffix(""))), output_dir / f"{label}-quality")
    systems_result = run_command(build_systems_command(str(systems_path), systems_context), output_dir / f"{label}-systems")
    quality = _child_summary("quality", quality_path, quality_result, validate_quality_artifact)
    systems = _child_summary("systems", systems_path, systems_result, validate_systems_artifact)
    quality_status, safe, bootstrap = _quality_state(quality)
    return quality, systems, quality_status, safe, bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "qualified", "full"], default="quick")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    mode = "qualified" if args.mode == "full" else args.mode
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{git_commit_sha()[:12]}"
    output_dir = Path(args.output_dir) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    binding = benchmark_binding(ROOT)

    if mode == "quick":
        quality, systems, quality_status, safe, bootstrap = _run_leg(output_dir, "8k", quality_mode="quick", systems_context=8192)
        status = "inconclusive" if quality_status == "qualified" and systems["structurally_valid"] else ("invalid" if quality_status == "invalid" or not systems["structurally_valid"] else "failed")
        preflight = {"status": "not_requested", "attempt_32k": False}
        ladder = {"status": "not_requested", "attempt_32k": False}
        legs = {"8k": {"quality": quality, "systems": systems, "quality_status": quality_status}}
    else:
        preflight_path = output_dir / "qualified_preflight.json"
        preflight_result = run_command(build_systems_command(str(preflight_path), 1024), output_dir / "preflight")
        preflight_systems = _child_summary("preflight_systems", preflight_path, preflight_result, validate_systems_artifact)
        estimate = estimate_qualified_capacity(preflight_systems["artifact"], target_context=32768, source_context=1024) if preflight_systems["structurally_valid"] else None
        preflight_passes = bool(preflight_systems["structurally_valid"] and estimate and estimate["passes"])
        preflight = {"status": "passed" if preflight_passes else "invalid", "artifact": preflight_systems,
                     "capacity_estimate": estimate}
        write_json(str(output_dir / "qualified_capacity_preflight.json"), preflight)
        if preflight_passes:
            quality, systems, quality_status, safe, bootstrap = _run_leg(output_dir, "8k", quality_mode="eight_k", systems_context=8192)
        else:
            quality, systems, quality_status, safe, bootstrap = ({"structurally_valid": False, "errors": ["not run: preflight failed"], "artifact": None}, {"structurally_valid": False, "errors": ["not run: preflight failed"], "artifact": None}, "invalid", False, None)
        calibrated_estimate = estimate_qualified_capacity(systems["artifact"], target_context=32768, source_context=8192) if systems["structurally_valid"] else None
        eight_k_systems_valid = bool(systems["structurally_valid"])
        eight_k_capacity_admitted = bool(
            calibrated_estimate and calibrated_estimate["passes"]
        )
        preflight["eight_k_calibrated_estimate"] = calibrated_estimate
        admission = {"preflight_passes": preflight_passes, "eight_k_systems_valid": eight_k_systems_valid, "eight_k_capacity_passes": eight_k_capacity_admitted, "capacity_estimate": calibrated_estimate}
        write_json(str(output_dir / "qualified_32k_admission.json"), admission)
        ladder = qualified_ladder_decision(
            preflight_valid=preflight_passes,
            eight_k_retrieval_safe=safe,
            eight_k_systems_valid=eight_k_systems_valid,
            eight_k_capacity_admitted=eight_k_capacity_admitted,
        )
        write_json(str(output_dir / "qualified_32k_decision.json"), ladder)
        legs = {"preflight": preflight_systems, "8k": {"quality": quality, "systems": systems, "quality_status": quality_status}}
        if ladder["attempt_32k"]:
            q32, s32, q32_status, safe32, bootstrap32 = _run_leg(output_dir, "32k", quality_mode="qualified", systems_context=32768)
            legs["32k"] = {"quality": q32, "systems": s32, "quality_status": q32_status}
            quality, systems, quality_status, safe, bootstrap = q32, s32, q32_status, safe32, bootstrap32
        status = "qualified" if ladder["attempt_32k"] and quality_status == "qualified" and systems["structurally_valid"] else ("invalid" if quality_status == "invalid" or not systems["structurally_valid"] else "failed")
    report = {"artifact_type": "qualified_run", "schema_version": 1, "mode": mode, "status": status,
              "binding": binding, "metadata": capture_env_metadata(), "quality": {"status": quality_status, "artifact": quality.get("artifact"), "errors": quality.get("errors", [])},
              "systems": {"status": "completed" if systems.get("structurally_valid") else "invalid", "artifact": systems.get("artifact"), "errors": systems.get("errors", [])},
              "retrieval_safe": safe, "bootstrap": None if bootstrap is None else bootstrap.__dict__,
              "preflight": preflight, "qualified_32k_decision": ladder, "legs": legs}
    write_json(str(output_dir / "repro_report.json"), report)
    write_markdown(str(output_dir / "repro_report.md"), f"# TurboQuant release evidence\n\n- Status: `{status}`\n- Quality: `{quality_status}`\n- 32K decision: `{ladder['status']}`\n")
    print(json.dumps({"status": status, "output_dir": str(output_dir)}, indent=2))
    if status != "qualified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
