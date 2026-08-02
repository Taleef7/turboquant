#!/usr/bin/env python3
"""Validate a Qwen2.5 fixed warmup artifact without running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_qwen25_fixed_warmup import (  # noqa: E402
    load_warmup_config,
    validate_warmup_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_warmup_config(args.config) if args.config else load_warmup_config()
    try:
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)]}, indent=2))
        raise SystemExit(1)
    result = validate_warmup_artifact(artifact, config)
    print(json.dumps({"valid": result.valid, "errors": list(result.errors)}, indent=2))
    if not result.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
