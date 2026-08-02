"""Build deterministic, reference-checked claim ledgers from schema-v2 manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.benchmark_schema import (
    _resolve_dotted_path,
    _safe_sidecar_path,
    _sha256_path,
    _validate_json_safe,
    _validate_v2_artifact,
    load_strict_json,
)


def _resolve_claim_value(row: Mapping[str, Any], query: str, *, repo_root: Path) -> Any:
    if query.startswith("metrics."):
        return _resolve_dotted_path(row, query)
    if query.startswith("sidecar:") and "#" in query:
        relative, json_path = query[len("sidecar:") :].split("#", 1)
        declared = {
            sidecar["path"]: sidecar
            for sidecar in row["sidecars"]
            if isinstance(sidecar, Mapping) and isinstance(sidecar.get("path"), str)
        }
        if relative not in declared:
            raise ValueError(f"claim query references undeclared sidecar {relative}")
        sidecar_path = _safe_sidecar_path(repo_root, relative)
        if sidecar_path is None:
            raise ValueError("claim query sidecar path must be relative to repo_root")
        if not sidecar_path.is_file():
            raise ValueError(f"claim query sidecar does not exist: {relative}")
        if _sha256_path(sidecar_path) != declared[relative]["sha256"]:
            raise ValueError(f"claim query sidecar hash mismatch: {relative}")
        try:
            payload = load_strict_json(sidecar_path)
            return _resolve_dotted_path(payload, json_path)
        except (OSError, UnicodeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"claim query cannot read sidecar {relative}: {exc}"
            ) from exc
    raise ValueError(f"claim query is not exact: {query!r}")


def _claims_from_payload(
    payload: dict[str, object], *, repo_root: Path
) -> list[dict[str, object]]:
    result = _validate_v2_artifact(
        payload,
        repo_root=repo_root,
        verify_sidecar_contents=False,
    )
    if not result.valid:
        raise ValueError("; ".join(result.errors))
    eligibility_errors: list[str] = []
    if payload["status"] in {"failed", "invalid"}:
        eligibility_errors.append(f"claim ledger rejects {payload['status']} artifacts")
    elif payload["status"] not in {"qualified", "partial"}:
        eligibility_errors.append(
            "claim ledger requires a qualified or clean partial artifact"
        )
    if payload["status"] == "partial":
        disallowed = {
            row["status"]
            for row in payload["rows"]
            if row["status"] in {"failed", "invalid", "oom"}
        }
        if disallowed:
            eligibility_errors.append(
                "claim ledger rejects partial artifacts with "
                + ", ".join(sorted(disallowed))
                + " rows"
            )
    if payload["source"]["dirty"]:
        eligibility_errors.append("claim ledger requires a clean source")
    artifact_id = payload["artifact_id"]
    rows_by_id = {row["row_id"]: row for row in payload["rows"]}
    for claim in payload["claims"]:
        row = rows_by_id[claim["row_id"]]
        if row["status"] != "passed":
            eligibility_errors.append(f"claim row {claim['row_id']} must be passed")
    if eligibility_errors:
        raise ValueError("; ".join(eligibility_errors))
    resolved: list[dict[str, object]] = []
    for claim in payload["claims"]:
        row = rows_by_id[claim["row_id"]]
        value = _resolve_claim_value(row, claim["query"], repo_root=repo_root)
        value_errors: list[str] = []
        _validate_json_safe(
            value,
            label=f"claim {claim['claim_id']} value",
            errors=value_errors,
        )
        if value_errors:
            raise ValueError("; ".join(value_errors))
        resolved.append(
            {
                "claim_id": claim["claim_id"],
                "artifact_id": artifact_id,
                "row_id": claim["row_id"],
                "query": claim["query"],
                "value": value,
            }
        )
    return resolved


def _build_ledger(claims: list[dict[str, object]]) -> dict[str, object]:
    claims.sort(key=lambda item: (item["claim_id"], item["artifact_id"]))
    seen: set[tuple[object, object]] = set()
    for claim in claims:
        key = (claim["claim_id"], claim["artifact_id"])
        if key in seen:
            raise ValueError(
                "claim ledger has duplicate claim/artifact reference: "
                f"{claim['claim_id']}/{claim['artifact_id']}"
            )
        seen.add(key)
    return {"ledger_version": 1, "claims": claims}


def build_claim_ledger_from_payloads(
    manifests: Sequence[dict[str, object]], *, repo_root: Path
) -> dict[str, object]:
    """Build a stable ledger from already-loaded compact manifests."""
    claims: list[dict[str, object]] = []
    for payload in manifests:
        claims.extend(_claims_from_payload(payload, repo_root=repo_root))
    return _build_ledger(claims)


def _find_repo_root(manifest: Path) -> Path:
    resolved = manifest.resolve()
    for candidate in resolved.parents:
        if (candidate / ".git").exists():
            return candidate
    return resolved.parent


def build_claim_ledger(manifests: Sequence[Path]) -> dict[str, object]:
    """Read compact manifests and build a stable, reference-checked ledger."""
    claims: list[dict[str, object]] = []
    for manifest in manifests:
        try:
            payload = load_strict_json(manifest)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(f"cannot read manifest {manifest}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"manifest {manifest} must contain an object")
        claims.extend(
            _claims_from_payload(payload, repo_root=_find_repo_root(manifest))
        )
    return _build_ledger(claims)
