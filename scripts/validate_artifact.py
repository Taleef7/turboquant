#!/usr/bin/env python3
"""Validate a benchmark JSON artifact and return a machine-useful exit code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_schema import validate_artifact
from repro_utils import benchmark_binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--no-binding-check", action="store_true")
    args = parser.parse_args()
    try:
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "errors": [f"cannot read artifact: {exc}"]}))
        raise SystemExit(2)
    result = validate_artifact(
        artifact, None if args.no_binding_check else benchmark_binding(ROOT)
    )
    print(json.dumps({"valid": result.valid, "errors": list(result.errors)}, indent=2))
    if not result.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
