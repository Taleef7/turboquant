"""Behavioral tests for deterministic claim-ledger generation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.claim_ledger import build_claim_ledger, build_claim_ledger_from_payloads
from tests.test_artifact_schema_v2 import make_valid_v2_artifact


def test_claim_ledger_requires_exact_metric_query(tmp_path: Path) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    manifest["claims"][0].pop("query")

    with pytest.raises(ValueError, match="claim query"):
        build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)


def test_claim_ledger_resolves_manifest_metric_and_verifies_references(
    tmp_path: Path,
) -> None:
    manifest = make_valid_v2_artifact(tmp_path)

    ledger = build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)

    assert ledger == {
        "ledger_version": 1,
        "claims": [
            {
                "claim_id": "row-pass-mse",
                "artifact_id": "codec-local-v2",
                "row_id": "row-pass",
                "query": "metrics.mse",
                "value": 0.125,
            }
        ],
    }


def test_claim_ledger_is_sorted_by_claim_and_artifact_id(tmp_path: Path) -> None:
    second = make_valid_v2_artifact(tmp_path)
    second["artifact_id"] = "artifact-z"
    second["claims"][0].update(
        {
            "claim_id": "claim-a",
            "artifact_id": "artifact-z",
        }
    )
    first = deepcopy(second)
    first["artifact_id"] = "artifact-a"
    first["claims"][0].update(
        {
            "claim_id": "claim-a",
            "artifact_id": "artifact-a",
        }
    )
    later = deepcopy(second)
    later["artifact_id"] = "artifact-b"
    later["claims"][0].update(
        {
            "claim_id": "claim-z",
            "artifact_id": "artifact-b",
        }
    )

    forward = build_claim_ledger_from_payloads(
        [later, second, first], repo_root=tmp_path
    )
    reverse = build_claim_ledger_from_payloads(
        [first, second, later], repo_root=tmp_path
    )

    assert forward == reverse
    assert [
        (claim["claim_id"], claim["artifact_id"]) for claim in forward["claims"]
    ] == [
        ("claim-a", "artifact-a"),
        ("claim-a", "artifact-z"),
        ("claim-z", "artifact-b"),
    ]


def test_claim_ledger_rejects_unknown_rows_and_wrong_artifacts(tmp_path: Path) -> None:
    unknown_row = make_valid_v2_artifact(tmp_path)
    unknown_row["claims"][0]["row_id"] = "row-missing"
    with pytest.raises(ValueError, match="claim row_id"):
        build_claim_ledger_from_payloads([unknown_row], repo_root=tmp_path)

    wrong_artifact = make_valid_v2_artifact(tmp_path)
    wrong_artifact["claims"][0]["artifact_id"] = "artifact-missing"
    with pytest.raises(ValueError, match="claim artifact_id"):
        build_claim_ledger_from_payloads([wrong_artifact], repo_root=tmp_path)


def test_claim_ledger_rejects_dirty_partial_manifests(tmp_path: Path) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    manifest["status"] = "partial"
    manifest["source"]["dirty"] = True

    with pytest.raises(
        ValueError,
        match="claim ledger requires a clean source",
    ):
        build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)


def test_claim_ledger_accepts_clean_partial_artifact_with_passed_claim_row(
    tmp_path: Path,
) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    manifest["status"] = "partial"
    unavailable = deepcopy(manifest["rows"][0])
    unavailable.update(
        {
            "row_id": "row-unavailable",
            "status": "unavailable",
            "reason": "optional input missing",
            "metrics": {},
            "sidecars": [],
        }
    )
    manifest["rows"].append(unavailable)
    manifest["expected_row_ids"].append("row-unavailable")
    manifest["pass_state_counts"]["unavailable"] = 1

    ledger = build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)

    assert ledger["claims"][0]["row_id"] == "row-pass"


@pytest.mark.parametrize("status", ("failed", "invalid"))
def test_claim_ledger_rejects_failed_or_invalid_artifacts(
    tmp_path: Path, status: str
) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    manifest["status"] = status

    with pytest.raises(ValueError, match="claim ledger rejects"):
        build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)


def test_claim_ledger_rejects_partial_with_unclaimed_failed_row(
    tmp_path: Path,
) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    manifest["status"] = "partial"
    failed = deepcopy(manifest["rows"][0])
    failed.update(
        {
            "row_id": "row-failed",
            "status": "failed",
            "reason": "benchmark failed",
            "metrics": {},
            "sidecars": [],
        }
    )
    manifest["rows"].append(failed)
    manifest["expected_row_ids"].append("row-failed")
    manifest["pass_state_counts"]["failed"] = 1

    with pytest.raises(ValueError, match="partial artifacts with failed"):
        build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)


def test_claim_ledger_rejects_claims_for_failed_rows(tmp_path: Path) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    manifest["status"] = "partial"
    manifest["rows"][0]["status"] = "failed"
    manifest["rows"][0]["reason"] = "benchmark process failed"
    manifest["pass_state_counts"]["passed"] = 0
    manifest["pass_state_counts"]["failed"] = 1

    with pytest.raises(ValueError, match="claim row row-pass must be passed"):
        build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)


def test_claim_ledger_rejects_non_json_safe_resolved_values(tmp_path: Path) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    manifest["rows"][0]["metrics"]["mse"] = float("nan")

    with pytest.raises(ValueError, match="metrics.mse must be finite"):
        build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)


def test_claim_ledger_does_not_read_raw_sidecar_for_manifest_query(
    tmp_path: Path,
) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    sidecar = tmp_path / "evidence" / "row-pass.json"
    sidecar.unlink()

    ledger = build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)

    assert ledger["claims"][0]["value"] == 0.125


def test_claim_ledger_reads_sidecar_only_for_an_explicit_sidecar_query(
    tmp_path: Path,
) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    manifest["claims"][0][
        "query"
    ] = "sidecar:evidence/row-pass.json#samples.0.latency_ms"

    ledger = build_claim_ledger_from_payloads([manifest], repo_root=tmp_path)

    assert ledger["claims"][0]["value"] == 1.25


def test_build_claim_ledger_reads_compact_manifests_deterministically(
    tmp_path: Path,
) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    first = build_claim_ledger([manifest_path])
    second = build_claim_ledger([manifest_path])

    assert first == second
    assert first["claims"][0]["value"] == 0.125


def test_build_claim_ledger_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    raw = json.dumps(manifest)
    raw = raw.replace(
        '"status": "qualified"',
        '"status": "qualified", "status": "partial"',
        1,
    )
    manifest_path = tmp_path / "duplicate-manifest.json"
    manifest_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate object key: status"):
        build_claim_ledger([manifest_path])


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_build_claim_ledger_rejects_non_standard_constants(
    tmp_path: Path, constant: str
) -> None:
    manifest = make_valid_v2_artifact(tmp_path)
    raw = json.dumps(manifest)
    raw = raw.replace("0.125", constant, 1)
    manifest_path = tmp_path / "constant-manifest.json"
    manifest_path.write_text(raw, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=f"non-standard JSON constant: {constant}",
    ):
        build_claim_ledger([manifest_path])
