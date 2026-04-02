#!/usr/bin/env python3
"""One-command reproducibility runner for current Qwen scope."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from repro_utils import capture_env_metadata, write_json, write_markdown


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True)


def extract_delta_pp(niah_json_path: str) -> float | None:
    p = Path(niah_json_path)
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))
    eligible = [r for r in rows if r.get("needle_present_after_tokenization")]
    if not eligible:
        return None
    baseline = sum(1 for r in eligible if r.get("baseline_correct")) / len(eligible)
    turbo = sum(1 for r in eligible if r.get("turboquant_correct")) / len(eligible)
    return (baseline - turbo) * 100.0


def build_niah_command(mode: str, output_prefix: str) -> list[str]:
    if mode == "quick":
        max_context = "8192"
        buffer_size = "7168"
        trials = "2"
    else:
        max_context = "16384"
        buffer_size = "12288"
        trials = "6"

    return [
        "python",
        "scripts/test_long_context.py",
        "--test",
        "niah",
        "--mode",
        "paired",
        "--model",
        "Qwen/Qwen2.5-7B-Instruct",
        "--max-context",
        max_context,
        "--key-bits",
        "8",
        "--value-bits",
        "6",
        "--buffer-size",
        buffer_size,
        "--trials",
        trials,
        "--output-prefix",
        output_prefix,
    ]


def build_report_md(report: dict) -> str:
    niah = report["runs"]["niah"]
    thr = report["runs"]["throughput"]
    return (
        "# TurboQuant Repro Report\n\n"
        f"- Mode: `{report['mode']}`\n"
        f"- Model: `Qwen/Qwen2.5-7B-Instruct`\n"
        f"- Commit: `{report['metadata']['git_commit']}`\n\n"
        "## NIAH (paired baseline-vs-TurboQuant)\n"
        f"- Exit code: `{niah['exit_code']}`\n"
        f"- Delta pp: `{niah.get('delta_pp')}`\n"
        f"- Output JSON: `{niah['json_path']}`\n\n"
        "## Throughput\n"
        f"- Exit code: `{thr['exit_code']}`\n"
        f"- Output JSON: `{thr['json_path']}`\n\n"
        "## Gate\n"
        f"- Target delta <= {report['gate']['target_delta_pp']}pp\n"
        f"- Pass: `{report['gate']['passed']}`\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce current Qwen TurboQuant claims"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick: 8K low-trial smoke, full: 16K t6 claim path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/repro",
        help="Directory for report artifacts",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    niah_prefix = str(output_dir / f"niah_{args.mode}")
    niah_json = f"{niah_prefix}.json"
    throughput_json = str(output_dir / "throughput.json")

    niah_cmd = build_niah_command(args.mode, niah_prefix)
    thr_cmd = [
        "python",
        "scripts/benchmark_throughput.py",
        "--output-json",
        throughput_json,
    ]

    print("Running NIAH paired benchmark...")
    niah_res = run_command(niah_cmd)
    print(niah_res.stdout)
    if niah_res.returncode != 0:
        print(niah_res.stderr)

    print("Running throughput benchmark...")
    thr_res = run_command(thr_cmd)
    print(thr_res.stdout)
    if thr_res.returncode != 0:
        print(thr_res.stderr)

    delta = extract_delta_pp(niah_json)
    gate_target = 2.0
    gate_passed = (
        delta is not None and delta <= gate_target and niah_res.returncode == 0
    )

    report = {
        "mode": args.mode,
        "metadata": capture_env_metadata(),
        "runs": {
            "niah": {
                "command": niah_cmd,
                "exit_code": niah_res.returncode,
                "json_path": niah_json,
                "delta_pp": delta,
            },
            "throughput": {
                "command": thr_cmd,
                "exit_code": thr_res.returncode,
                "json_path": throughput_json,
            },
        },
        "gate": {
            "target_delta_pp": gate_target,
            "passed": gate_passed,
        },
    }

    report_json = str(output_dir / "repro_report.json")
    report_md = str(output_dir / "repro_report.md")

    write_json(report_json, report)
    write_markdown(report_md, build_report_md(report))

    print(f"Wrote report JSON: {report_json}")
    print(f"Wrote report MD:   {report_md}")

    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
