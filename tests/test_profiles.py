"""Validation coverage for immutable direct-decode profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.errors import UnsupportedCapabilityError
from core.profiles import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json"
MANIFEST_PATH = ROOT / "configs" / "models" / "qwen25-7b.json"


def canonical_hash(payload: dict[str, object], hash_field: str) -> str:
    copy = dict(payload)
    copy.pop(hash_field, None)
    encoded = json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_profile(tmp_path: Path, **changes: object) -> Path:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    payload["model_manifest_path"] = str(
        ROOT / "configs" / "models" / "qwen25-7b.json"
    )
    payload["model_manifest_sha256"] = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    payload.update(changes)
    payload["profile_sha256"] = canonical_hash(payload, "profile_sha256")
    target = tmp_path / "profile.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def write_rehashed_manifest_and_profile(
    tmp_path: Path, **manifest_changes: object
) -> Path:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.update(manifest_changes)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return write_profile(
        tmp_path,
        model_manifest_path=str(manifest_path),
        model_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


def test_checked_in_direct_profile_validates_its_model_manifest() -> None:
    profile = load_profile(PROFILE_PATH)

    assert profile.name == "qwen25-7b-direct-v1"
    assert profile.block_size == 256
    assert profile.key_bits == 8
    assert profile.value_bits == 6
    assert profile.value_group_size == 32
    assert profile.fp16_window == 128


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"key_bits": 7}, "key_bits"),
        ({"value_bits": 7}, "value_bits"),
        ({"value_group_size": 24}, "divisible"),
        ({"block_size": 128}, "block_size"),
        ({"fp16_window": 257}, "fp16_window"),
        ({"model_manifest_sha256": "0" * 64}, "manifest hash"),
    ],
)
def test_profile_rejects_invalid_tuning_or_manifest(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(UnsupportedCapabilityError, match=message):
        load_profile(write_profile(tmp_path, **changes))


def test_profile_rejects_a_tampered_profile_hash(tmp_path: Path) -> None:
    path = write_profile(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["profile_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnsupportedCapabilityError, match="profile hash"):
        load_profile(path)


def test_profile_never_defaults_a_missing_required_field(tmp_path: Path) -> None:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    payload.pop("fp16_window")
    payload["profile_sha256"] = canonical_hash(payload, "profile_sha256")
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnsupportedCapabilityError, match="fp16_window"):
        load_profile(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"num_hidden_layers": 2, "layer_types": ["full_attention"]}, "layer_types|architecture"),
        ({"num_key_value_heads": 8}, "architecture"),
        ({"head_dim": 64}, "architecture"),
        ({"use_cache": False}, "architecture"),
        ({"is_multimodal": True}, "architecture"),
    ],
)
def test_profile_rejects_a_rehashed_manifest_that_breaks_architecture_binding(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(UnsupportedCapabilityError, match=message):
        load_profile(write_rehashed_manifest_and_profile(tmp_path, **changes))
