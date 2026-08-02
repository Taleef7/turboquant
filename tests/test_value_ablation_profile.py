"""Regression tests for the opt-in Qwen value-bit ablation profile.

The production profile remains v6 values.  This profile is deliberately
separate so an experiment cannot silently change the reported baseline.
"""

import hashlib
import json
from pathlib import Path

from core.profiles import load_profile


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json"
ABLATION = ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1-v8-value.json"


def canonical_hash(payload: dict[str, object], hash_field: str) -> str:
    copy = dict(payload)
    copy.pop(hash_field, None)
    encoded = json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_value_bit_ablation_is_explicit_and_model_bound():
    baseline = load_profile(BASELINE)
    ablation = load_profile(ABLATION)

    assert baseline.name == "qwen25-7b-direct-v1"
    assert baseline.value_bits == 6
    assert ablation.name == "qwen25-7b-direct-v1-v8-value"
    assert ablation.value_bits == 8
    assert ablation.key_bits == baseline.key_bits == 8
    assert ablation.revision == baseline.revision
    assert ablation.model_manifest_sha256 == baseline.model_manifest_sha256
    assert json.loads(ABLATION.read_text(encoding="utf-8"))["evidence_codebooks"] == [
        "../../codebooks/codebook_d128_b8.json",
        "../../codebooks/codebook_d128_b8.json",
    ]


def test_value_bit_ablation_profile_hash_is_self_consistent():
    payload = json.loads(ABLATION.read_text(encoding="utf-8"))
    assert payload["profile_sha256"] == canonical_hash(payload, "profile_sha256")
