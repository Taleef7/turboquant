"""Behavioral tests for additive evidence artifact schema v2."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark_schema import validate_artifact, validate_v2_artifact
from scripts.repro_utils import capture_source_metadata
from tests.test_benchmark_schema import complete_matrix


ROW_STATES = ("passed", "failed", "invalid", "oom", "unsupported", "unavailable")
ROOT = Path(__file__).resolve().parents[1]


def make_valid_v2_artifact(repo_root: Path) -> dict[str, object]:
    """Return a hand-written, qualified one-row v2 artifact."""
    sidecar_path = repo_root / "evidence" / "row-pass.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_bytes = json.dumps(
        {"samples": [{"latency_ms": 1.25}]},
        separators=(",", ":"),
    ).encode()
    sidecar_path.write_bytes(sidecar_bytes)
    identity = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "weights_sha256": "1" * 64,
        "profile_id": "paper_mse_d128_b4",
        "profile_sha256": "2" * 64,
        "kv_dtype": "float16",
    }
    return {
        "artifact_type": "codec_reproduction",
        "schema_version": 2,
        "artifact_id": "codec-local-v2",
        "status": "qualified",
        "paper": {
            "paper_id": "arxiv:2504.19874v1",
            "source_url": "https://arxiv.org/pdf/2504.19874v1",
            "retrieved_at": "2026-07-31T12:00:00+00:00",
            "sha256": "0" * 64,
        },
        "source": {
            "git_commit": "a" * 40,
            "dirty": False,
        },
        "execution_class": {
            "name": "local_rtx5070ti_12gb",
            "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
            "total_vram_bytes": 12_884_901_888,
            "driver_version": "576.88",
            "cuda_runtime": "12.8",
        },
        "identity": identity,
        "expected_row_ids": ["row-pass"],
        "statistics": {
            "method": "paired_bootstrap",
            "confidence_level": 0.95,
            "seed": 1729,
            "iterations": 10_000,
        },
        "rows": [
            {
                "row_id": "row-pass",
                "status": "passed",
                "reason": None,
                "identity": deepcopy(identity),
                "metrics": {"mse": 0.125, "nested": {"p95": 1.5}},
                "sidecars": [
                    {
                        "path": "evidence/row-pass.json",
                        "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                        "available": True,
                    }
                ],
                "extensions": {},
            }
        ],
        "pass_state_counts": {
            "passed": 1,
            "failed": 0,
            "invalid": 0,
            "oom": 0,
            "unsupported": 0,
            "unavailable": 0,
        },
        "claims": [
            {
                "claim_id": "row-pass-mse",
                "artifact_id": "codec-local-v2",
                "row_id": "row-pass",
                "query": "metrics.mse",
                "extensions": {},
            }
        ],
        "extensions": {},
    }


@pytest.fixture
def valid_v2_artifact(tmp_path: Path) -> dict[str, object]:
    return make_valid_v2_artifact(tmp_path)


def test_v2_accepts_a_complete_qualified_artifact(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert result.valid
    assert result.errors == ()
    assert validate_artifact(valid_v2_artifact, repo_root=tmp_path).valid


def test_v2_rejects_unknown_execution_class(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    valid_v2_artifact["execution_class"]["name"] = "personal_gpu"

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert "execution_class.name is not declared" in result.errors


def test_v2_rejects_qualified_dirty_source(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    valid_v2_artifact["source"]["dirty"] = True

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert "dirty source cannot qualify" in result.errors


def test_v2_rejects_missing_and_duplicate_row_ids(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    missing = deepcopy(valid_v2_artifact)
    missing["expected_row_ids"].append("row-missing")
    result = validate_v2_artifact(missing, repo_root=tmp_path)
    assert not result.valid
    assert "missing expected row IDs: ['row-missing']" in result.errors

    duplicate = deepcopy(valid_v2_artifact)
    duplicate["rows"].append(deepcopy(duplicate["rows"][0]))
    duplicate["pass_state_counts"]["passed"] = 2
    result = validate_v2_artifact(duplicate, repo_root=tmp_path)
    assert not result.valid
    assert "duplicate row ID: row-pass" in result.errors


def test_v2_rejects_unavailable_missing_and_changed_sidecars(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    unavailable = deepcopy(valid_v2_artifact)
    unavailable["rows"][0]["sidecars"][0]["available"] = False
    result = validate_v2_artifact(unavailable, repo_root=tmp_path)
    assert not result.valid
    assert "row row-pass sidecar is unavailable: evidence/row-pass.json" in result.errors

    missing = deepcopy(valid_v2_artifact)
    missing["rows"][0]["sidecars"][0]["path"] = "evidence/missing.json"
    result = validate_v2_artifact(missing, repo_root=tmp_path)
    assert not result.valid
    assert "row row-pass sidecar does not exist: evidence/missing.json" in result.errors

    changed = deepcopy(valid_v2_artifact)
    changed["rows"][0]["sidecars"][0]["sha256"] = "f" * 64
    result = validate_v2_artifact(changed, repo_root=tmp_path)
    assert not result.valid
    assert "row row-pass sidecar hash mismatch: evidence/row-pass.json" in result.errors


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("model_id", "other/model"),
        ("model_revision", "b" * 40),
        ("weights_sha256", "3" * 64),
        ("profile_id", "forge_direct_v1"),
        ("profile_sha256", "4" * 64),
        ("kv_dtype", "bfloat16"),
    ),
)
def test_v2_rejects_mixed_row_identity(
    valid_v2_artifact: dict[str, object],
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    valid_v2_artifact["rows"][0]["identity"][field] = replacement

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert f"row row-pass identity.{field} does not match artifact identity" in result.errors


def test_v2_preserves_every_declared_row_outcome(tmp_path: Path) -> None:
    artifact = make_valid_v2_artifact(tmp_path)
    template = artifact["rows"][0]
    artifact["status"] = "partial"
    artifact["claims"] = []
    artifact["expected_row_ids"] = [f"row-{state}" for state in ROW_STATES]
    artifact["rows"] = []
    artifact["pass_state_counts"] = {state: 0 for state in ROW_STATES}
    for state in ROW_STATES:
        row = deepcopy(template)
        row["row_id"] = f"row-{state}"
        row["status"] = state
        row["reason"] = None if state == "passed" else f"recorded {state}"
        row["sidecars"] = []
        artifact["rows"].append(row)
        artifact["pass_state_counts"][state] += 1

    result = validate_v2_artifact(artifact, repo_root=tmp_path)

    assert result.valid
    assert [row["status"] for row in artifact["rows"]] == list(ROW_STATES)


def test_v2_requires_exact_pass_state_counts(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    valid_v2_artifact["pass_state_counts"]["passed"] = 0

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert "pass_state_counts do not match row outcomes" in result.errors


def test_v2_requires_paper_and_statistics_specs(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    missing_paper = deepcopy(valid_v2_artifact)
    missing_paper["paper"].pop("sha256")
    result = validate_v2_artifact(missing_paper, repo_root=tmp_path)
    assert not result.valid
    assert "paper.sha256 is required" in result.errors

    bad_statistics = deepcopy(valid_v2_artifact)
    bad_statistics["statistics"]["confidence_level"] = 1.0
    result = validate_v2_artifact(bad_statistics, repo_root=tmp_path)
    assert not result.valid
    assert "statistics.confidence_level must be between 0 and 1" in result.errors


def test_v2_requires_exact_claim_references(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    no_query = deepcopy(valid_v2_artifact)
    no_query["claims"][0].pop("query")
    result = validate_v2_artifact(no_query, repo_root=tmp_path)
    assert not result.valid
    assert "claim query is required" in result.errors

    wrong_artifact = deepcopy(valid_v2_artifact)
    wrong_artifact["claims"][0]["artifact_id"] = "another-artifact"
    result = validate_v2_artifact(wrong_artifact, repo_root=tmp_path)
    assert not result.valid
    assert "claim artifact_id does not match artifact" in result.errors

    missing_metric = deepcopy(valid_v2_artifact)
    missing_metric["claims"][0]["query"] = "metrics.not_recorded"
    result = validate_v2_artifact(missing_metric, repo_root=tmp_path)
    assert not result.valid
    assert "claim query does not resolve for row row-pass" in result.errors


def test_v2_rejects_unknown_structured_keys_but_accepts_extensions(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    unknown_top_level = deepcopy(valid_v2_artifact)
    unknown_top_level["surprise"] = True
    result = validate_v2_artifact(unknown_top_level, repo_root=tmp_path)
    assert not result.valid
    assert "artifact has unknown keys: surprise" in result.errors

    unknown_row_key = deepcopy(valid_v2_artifact)
    unknown_row_key["rows"][0]["ad_hoc_note"] = "not namespaced"
    result = validate_v2_artifact(unknown_row_key, repo_root=tmp_path)
    assert not result.valid
    assert "row row-pass has unknown keys: ad_hoc_note" in result.errors

    valid_v2_artifact["extensions"]["lab"] = {"rack": "west"}
    valid_v2_artifact["identity"]["extensions"] = {"weight_format": "safetensors"}
    valid_v2_artifact["rows"][0]["extensions"]["vendor"] = {"clock_mhz": 2300}
    valid_v2_artifact["rows"][0]["identity"]["extensions"] = {
        "weight_format": "safetensors"
    }
    assert validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path).valid


def test_v2_rejects_sidecar_paths_outside_repo_root(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    valid_v2_artifact["rows"][0]["sidecars"][0]["path"] = "../outside.json"

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert "row row-pass sidecar path must be relative to repo_root" in result.errors


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_v2_rejects_non_finite_metrics_recursively(
    valid_v2_artifact: dict[str, object], tmp_path: Path, value: float
) -> None:
    valid_v2_artifact["rows"][0]["metrics"]["nested"]["p95"] = value

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert "row row-pass metrics.nested.p95 must be finite" in result.errors


def test_v2_rejects_non_json_safe_metric_values(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    valid_v2_artifact["rows"][0]["metrics"]["opaque"] = object()

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert "row row-pass metrics.opaque is not JSON-safe" in result.errors


def test_v2_resolves_explicit_sidecar_queries_during_validation(
    valid_v2_artifact: dict[str, object], tmp_path: Path
) -> None:
    valid_v2_artifact["claims"][0]["query"] = (
        "sidecar:evidence/row-pass.json#samples.1.latency_ms"
    )

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert "claim sidecar query does not resolve for row row-pass" in result.errors


@pytest.mark.parametrize("version", (True, 2.0, 3))
def test_artifact_dispatch_rejects_non_exact_or_unsupported_v2_versions(
    valid_v2_artifact: dict[str, object], tmp_path: Path, version: object
) -> None:
    valid_v2_artifact["schema_version"] = version

    result = validate_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert result.errors == (f"unsupported schema_version {version!r}",)


@pytest.mark.parametrize("version", (True, False, 2.0, 3, "2", None))
def test_direct_v2_validator_requires_exact_integer_version_two(
    valid_v2_artifact: dict[str, object], tmp_path: Path, version: object
) -> None:
    if version is None:
        valid_v2_artifact.pop("schema_version")
    else:
        valid_v2_artifact["schema_version"] = version

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert "schema_version must be the exact integer 2" in result.errors


@pytest.mark.parametrize(
    ("level", "value_kind", "expected_error"),
    (
        ("artifact", "nan", "artifact.extensions.review must be finite"),
        (
            "paper",
            "object",
            "artifact.paper.extensions.review is not JSON-safe",
        ),
        (
            "identity",
            "nan",
            "artifact.identity.extensions.review must be finite",
        ),
        (
            "row",
            "object",
            "artifact.rows.0.extensions.review is not JSON-safe",
        ),
        (
            "sidecar",
            "nan",
            "artifact.rows.0.sidecars.0.extensions.review must be finite",
        ),
        (
            "claim",
            "object",
            "artifact.claims.0.extensions.review is not JSON-safe",
        ),
    ),
)
def test_v2_rejects_unsafe_values_at_every_extension_level(
    valid_v2_artifact: dict[str, object],
    tmp_path: Path,
    level: str,
    value_kind: str,
    expected_error: str,
) -> None:
    value = math.nan if value_kind == "nan" else object()
    if level == "artifact":
        extensions = valid_v2_artifact["extensions"]
    elif level == "paper":
        extensions = valid_v2_artifact["paper"].setdefault("extensions", {})
    elif level == "identity":
        extensions = valid_v2_artifact["identity"].setdefault("extensions", {})
    elif level == "row":
        extensions = valid_v2_artifact["rows"][0]["extensions"]
    elif level == "sidecar":
        extensions = valid_v2_artifact["rows"][0]["sidecars"][0].setdefault(
            "extensions", {}
        )
    else:
        extensions = valid_v2_artifact["claims"][0]["extensions"]
    extensions["review"] = value

    result = validate_v2_artifact(valid_v2_artifact, repo_root=tmp_path)

    assert not result.valid
    assert expected_error in result.errors


@pytest.mark.parametrize("version", (True, 1.0, 3))
def test_artifact_dispatch_rejects_non_exact_or_unsupported_v1_versions(
    version: object,
) -> None:
    artifact = complete_matrix()
    artifact["schema_version"] = version

    result = validate_artifact(artifact)

    assert not result.valid
    assert result.errors == (f"unsupported schema_version {version!r}",)


def test_validate_artifact_cli_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    artifact_path = tmp_path / "duplicate.json"
    artifact_path.write_text(
        '{"artifact_type":"codec_reproduction","schema_version":2,'
        '"status":"qualified","status":"partial"}',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_artifact.py",
            str(artifact_path),
            "--no-binding-check",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "duplicate object key: status" in completed.stdout


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_validate_artifact_cli_rejects_non_standard_constants(
    tmp_path: Path, constant: str
) -> None:
    artifact_path = tmp_path / "constant.json"
    artifact_path.write_text(
        f'{{"artifact_type":"codec_reproduction","schema_version":2,'
        f'"metric":{constant}}}',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_artifact.py",
            str(artifact_path),
            "--no-binding-check",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert f"non-standard JSON constant: {constant}" in completed.stdout


def test_source_metadata_distinguishes_clean_and_dirty_repositories(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(
        ["git", "-C", tmp_path, "config", "user.email", "tests@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", tmp_path, "config", "user.name", "TurboQuant Tests"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", tmp_path, "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", tmp_path, "commit", "-q", "-m", "fixture"],
        check=True,
    )

    clean = capture_source_metadata(tmp_path)
    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = capture_source_metadata(tmp_path)

    assert clean == {"git_commit": clean["git_commit"], "dirty": False}
    assert len(clean["git_commit"]) == 40
    assert dirty == {"git_commit": clean["git_commit"], "dirty": True}
