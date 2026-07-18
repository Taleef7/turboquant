"""Strict loading for immutable, hash-bound compression profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.capabilities import ModelRevision
from core.errors import UnsupportedCapabilityError


_SCHEMA_VERSION = 1
_SUPPORTED_BITS = frozenset({2, 3, 4, 5, 6, 8})
_DIRECT_BLOCK_SIZE = 256
_REQUIRED_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "adapter_status",
        "model_manifest_path",
        "model_manifest_sha256",
        "model_revision",
        "block_size",
        "key_bits",
        "value_bits",
        "value_group_size",
        "fp16_window",
        "profile_sha256",
    }
)
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {"model_id", "revision", "model_type", "head_dim", "adapter_status"}
)


@dataclass(frozen=True)
class CompressionProfile:
    """Validated settings for the sole direct-decode profile currently shipped."""

    schema_version: int
    name: str
    adapter_status: str
    model_manifest_path: Path
    model_manifest_sha256: str
    revision: ModelRevision
    block_size: int
    key_bits: int
    value_bits: int
    value_group_size: int
    fp16_window: int
    profile_sha256: str


def _canonical_hash(payload: Mapping[str, Any], omitted_field: str) -> str:
    content = dict(payload)
    content.pop(omitted_field, None)
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_fields(payload: Mapping[str, Any], required: frozenset[str], label: str) -> None:
    missing = sorted(required.difference(payload))
    if missing:
        raise UnsupportedCapabilityError(f"{label} is missing required field(s): {', '.join(missing)}.")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedCapabilityError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise UnsupportedCapabilityError(f"{label} at {path} must be a JSON object.")
    return payload


def _manifest_path(profile_path: Path, configured_path: str) -> Path:
    candidate = Path(configured_path)
    return candidate if candidate.is_absolute() else profile_path.parent / candidate


def load_profile(path: str | Path) -> CompressionProfile:
    """Load a complete profile and reject stale, malformed, or unsafe settings."""
    profile_path = Path(path)
    payload = _read_json(profile_path, "profile")
    _require_fields(payload, _REQUIRED_PROFILE_FIELDS, "profile")
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise UnsupportedCapabilityError(
            f"Unsupported schema_version {payload['schema_version']!r}; expected {_SCHEMA_VERSION}."
        )
    expected_profile_hash = _canonical_hash(payload, "profile_sha256")
    if payload["profile_sha256"] != expected_profile_hash:
        raise UnsupportedCapabilityError("profile hash does not match the profile contents.")

    for field in ("key_bits", "value_bits"):
        if payload[field] not in _SUPPORTED_BITS:
            raise UnsupportedCapabilityError(
                f"{field}={payload[field]!r} is unsupported; expected one of {sorted(_SUPPORTED_BITS)}."
            )
    if payload["block_size"] != _DIRECT_BLOCK_SIZE:
        raise UnsupportedCapabilityError(
            f"block_size must be {_DIRECT_BLOCK_SIZE} for the direct adapter."
        )
    if not isinstance(payload["fp16_window"], int) or not 0 <= payload["fp16_window"] <= payload["block_size"]:
        raise UnsupportedCapabilityError("fp16_window must be within 0 and block_size.")

    model_revision = payload["model_revision"]
    if not isinstance(model_revision, dict) or set(model_revision) != {"model_id", "revision"}:
        raise UnsupportedCapabilityError("model_revision must contain exactly model_id and revision.")
    manifest_path = _manifest_path(profile_path, payload["model_manifest_path"])
    try:
        actual_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise UnsupportedCapabilityError(f"Cannot read target model manifest at {manifest_path}: {exc}") from exc
    if payload["model_manifest_sha256"] != actual_manifest_hash:
        raise UnsupportedCapabilityError("target model manifest hash does not match the profile.")
    manifest = _read_json(manifest_path, "model manifest")
    _require_fields(manifest, _REQUIRED_MANIFEST_FIELDS, "model manifest")
    if manifest["model_id"] != model_revision["model_id"] or manifest["revision"] != model_revision["revision"]:
        raise UnsupportedCapabilityError("model_revision does not match the target model manifest.")
    if manifest["adapter_status"] != "direct-adapter" or payload["adapter_status"] != "direct-adapter":
        raise UnsupportedCapabilityError("profile requires a direct-adapter target model manifest.")
    if not isinstance(payload["value_group_size"], int) or payload["value_group_size"] <= 0:
        raise UnsupportedCapabilityError("value_group_size must be a positive integer.")
    if manifest["head_dim"] % payload["value_group_size"] != 0:
        raise UnsupportedCapabilityError(
            "value_group_size must be divisible into the target model head_dim."
        )

    return CompressionProfile(
        schema_version=payload["schema_version"],
        name=payload["name"],
        adapter_status=payload["adapter_status"],
        model_manifest_path=manifest_path,
        model_manifest_sha256=payload["model_manifest_sha256"],
        revision=ModelRevision(
            model_id=model_revision["model_id"],
            revision=model_revision["revision"],
            manifest_sha256=actual_manifest_hash,
        ),
        block_size=payload["block_size"],
        key_bits=payload["key_bits"],
        value_bits=payload["value_bits"],
        value_group_size=payload["value_group_size"],
        fp16_window=payload["fp16_window"],
        profile_sha256=payload["profile_sha256"],
    )
