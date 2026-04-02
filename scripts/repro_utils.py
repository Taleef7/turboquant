"""Utilities for reproducibility reporting."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def git_commit_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def capture_env_metadata() -> dict:
    cuda_available = torch.cuda.is_available()
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_sha(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "python_executable": os.environ.get("PYTHON_EXECUTABLE", "python"),
    }
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        meta.update(
            {
                "gpu_name": props.name,
                "gpu_vram_gb": round(props.total_memory / (1024**3), 2),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return meta


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: dict) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path: str, text: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
