"""Utilities for reproducibility reporting."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PACKAGE_NAMES = (
    "torch",
    "transformers",
    "triton",
    "bitsandbytes",
    "numpy",
    "scipy",
)
PINNED_QWEN25_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PINNED_QWEN25_HUB_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
# Every file here can affect direct-attention correctness or qualification. The
# relative names are also fed into the digest, avoiding an order-only hash.
DIRECT_EVIDENCE_SOURCE_PATHS = (
    "scripts/test_long_context.py",
    "scripts/run_direct_benchmark.py",
    "scripts/reproduce_release.py",
    "scripts/benchmark_schema.py",
    "scripts/repro_utils.py",
    "core/capabilities.py",
    "core/profiles.py",
    "core/packing.py",
    "core/codec.py",
    "core/packed_store.py",
    "core/reference_attention.py",
    "core/packed_cache.py",
    "core/qwen_adapter.py",
    "core/turboquant_simple.py",
    "kernels/packed_attention.py",
    "utils/math_utils.py",
    "configs/profiles/qwen25-7b-direct-v1.json",
    "configs/models/qwen25-7b.json",
)


def git_commit_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_binding(root: str | Path) -> dict[str, str]:
    """Hash every direct-evidence input that can affect observed behavior.

    The codebook digest is a hash of both direct-profile codebooks in a stable
    order, so a run cannot accidentally compare different quantizers.
    """
    root_path = Path(root)
    profile = root_path / "configs" / "profiles" / "qwen25-7b-direct-v1.json"
    manifest = root_path / "configs" / "models" / "qwen25-7b.json"
    profile_payload = json.loads(profile.read_text(encoding="utf-8"))
    model_revision = profile_payload.get("model_revision")
    if model_revision != {
        "model_id": PINNED_QWEN25_MODEL_ID,
        "revision": PINNED_QWEN25_HUB_REVISION,
    }:
        raise ValueError("direct benchmark profile must bind the pinned Qwen2.5 Hub revision")
    codebooks = (
        root_path / "codebooks" / "codebook_d128_b8.json",
        root_path / "codebooks" / "codebook_d128_b6.json",
    )
    combined = hashlib.sha256()
    for codebook in codebooks:
        combined.update(codebook.name.encode("utf-8"))
        combined.update(codebook.read_bytes())
    behavior = hashlib.sha256()
    for relative_path in DIRECT_EVIDENCE_SOURCE_PATHS:
        source = root_path / relative_path
        behavior.update(source.relative_to(root_path).as_posix().encode("utf-8"))
        behavior.update(source.read_bytes())
    runner = root_path / "scripts" / "run_direct_benchmark.py"
    return {
        # This is a digest of the immutable Hub identity, never a local
        # manifest digest mislabeled as model weights.
        "model_id": PINNED_QWEN25_MODEL_ID,
        "model_revision": PINNED_QWEN25_HUB_REVISION,
        "model_revision_sha256": hashlib.sha256(
            f"{PINNED_QWEN25_MODEL_ID}@{PINNED_QWEN25_HUB_REVISION}".encode("utf-8")
        ).hexdigest(),
        "model_manifest_sha256": sha256_file(manifest),
        "profile_sha256": sha256_file(profile),
        "codebook_sha256": combined.hexdigest(),
        "code_sha256": behavior.hexdigest(),
        "runner_sha256": sha256_file(runner),
    }


def package_version(name: str) -> str | None:
    """Return an installed distribution version or an explicit null equivalent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def _nvidia_driver_version() -> str | None:
    """Return the active NVIDIA driver version when nvidia-smi is available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip().splitlines()[0]
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError):
        return None


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed available random generators and configure deterministic torch modes."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch = _import_torch()
    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic


def capture_cuda_metadata() -> dict[str, bool | int | str | None]:
    """Capture CUDA and determinism state without allocating a GPU tensor."""
    schema: dict[str, bool | int | str | None] = {
        "cuda_available": None,
        "cuda_runtime": None,
        "gpu_name": None,
        "gpu_sm_capability": None,
        "gpu_total_vram_bytes": None,
        "gpu_free_vram_bytes": None,
        "driver_version": None,
        "tf32_allow_tf32": None,
        "cudnn_allow_tf32": None,
        "deterministic_algorithms": None,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
    }
    schema["driver_version"] = _nvidia_driver_version()
    torch = _import_torch()
    if torch is None:
        return schema

    cuda_available = torch.cuda.is_available()
    schema.update(
        {
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "tf32_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
    )
    if not cuda_available:
        return schema

    props = torch.cuda.get_device_properties(0)
    free_bytes, _ = torch.cuda.mem_get_info(0)
    schema.update(
        {
            "gpu_name": props.name,
            "gpu_sm_capability": f"sm_{props.major}{props.minor}",
            "gpu_total_vram_bytes": props.total_memory,
            "gpu_free_vram_bytes": free_bytes,
        }
    )
    return schema


def capture_env_metadata() -> dict[str, bool | int | str | None]:
    """Return the explicit, JSON-serializable reproducibility metadata schema."""
    cuda = capture_cuda_metadata()
    meta: dict[str, bool | int | str | None] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_sha(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "deterministic": cuda["deterministic_algorithms"],
    }
    for package in _PACKAGE_NAMES:
        meta[f"{package}_version"] = package_version(package)
    meta.update(cuda)
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
