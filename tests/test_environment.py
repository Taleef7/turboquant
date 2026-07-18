"""Tests for the reproducibility environment contract.

These tests intentionally avoid model downloads and GPU allocation.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import tomllib
from pathlib import Path

from scripts.repro_utils import (
    capture_cuda_metadata,
    capture_env_metadata,
    package_version,
    seed_everything,
    sha256_file,
)


def test_package_test_layout_is_declared() -> None:
    project = Path("pyproject.toml")
    assert project.is_file()
    contents = project.read_text(encoding="utf-8")
    config = tomllib.loads(contents)
    assert 'testpaths = ["tests"]' in contents
    assert 'pythonpath = ["."]' in contents
    assert any(
        marker.startswith("gpu:")
        for marker in config["tool"]["pytest"]["ini_options"]["markers"]
    )


def test_capture_env_metadata_uses_running_interpreter(monkeypatch) -> None:
    monkeypatch.delenv("PYTHON_EXECUTABLE", raising=False)
    metadata = capture_env_metadata()

    assert metadata["python_executable"] == sys.executable
    assert metadata["python_version"] == platform.python_version()
    assert "torch_version" in metadata
    assert "transformers_version" in metadata
    assert "triton_version" in metadata
    assert "bitsandbytes_version" in metadata
    assert "numpy_version" in metadata
    assert "scipy_version" in metadata
    assert metadata["deterministic"] in {None, False, True}
    assert "tf32_allow_tf32" in metadata


def test_capture_cuda_metadata_has_explicit_cpu_safe_schema() -> None:
    metadata = capture_cuda_metadata()

    assert set(metadata) == {
        "cuda_available",
        "cuda_runtime",
        "gpu_name",
        "gpu_sm_capability",
        "gpu_total_vram_bytes",
        "driver_version",
        "tf32_allow_tf32",
        "cudnn_allow_tf32",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
    }
    assert metadata["cuda_available"] in {None, False, True}
    if metadata["cuda_available"] is None:
        assert all(value is None for value in metadata.values())


def test_sha256_file_is_stable(tmp_path) -> None:
    target = tmp_path / "profile.json"
    target.write_text('{"name":"test"}', encoding="utf-8")

    assert sha256_file(target) == sha256_file(target)


def test_package_version_is_explicit_when_missing() -> None:
    assert package_version("turboquant-package-that-does-not-exist") is None


def test_seed_everything_replays_python_random_sequence(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    seed_everything(123, deterministic=True)
    first = random.random()
    seed_everything(123, deterministic=True)

    assert random.random() == first
    assert os.environ["PYTHONHASHSEED"] == "123"
