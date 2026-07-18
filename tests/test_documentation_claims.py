"""Regression checks for public TurboQuant Forge claims.

These checks intentionally examine the release-facing documents only.  Design
and implementation plans retain historical context and are not public evidence
artifacts.
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "TESTING_RESULTS.md",
    ROOT / "ISSUES.md",
    ROOT / "docs" / "PAPER_CLAIMS_STATUS.md",
    ROOT / "docs" / "PAPER_COMPARISON.md",
    ROOT / "docs" / "REPLICATION_CHECKLIST.md",
    ROOT / "docs" / "PORTABILITY.md",
    ROOT / "docs" / "BENCHMARK_PROTOCOL.md",
    ROOT / "docs" / "RESULTS.md",
)
QUALIFIED_ARTIFACT_ID = re.compile(
    r"qualified(?:-[a-z0-9]+)?/\d{8}T\d{6}Z-[a-z0-9._-]+", re.IGNORECASE
)
THIRTY_TWO_K_SUCCESS = re.compile(
    r"(?:\b(?:successful|passed)\s+(?:\w+\s+){0,2}32k\b|"
    r"\bqualified\s+32k\b|\b32k\s+(?:(?:run|evaluation)\s+)?"
    r"(?:success|passed|succeeded|qualified)\b|\b32k\s+(?:has|is|was)"
    r"(?:\s+\w+){0,3}\s+(?:passed|qualified|successful)\b)",
    re.IGNORECASE,
)


def read_public_docs() -> str:
    """Return one normalized corpus of release-facing Markdown documents."""
    return "\n".join(
        path.read_text(encoding="utf-8") if path.exists() else ""
        for path in PUBLIC_DOCUMENTS
    )


def normalize_claim_text(value: str) -> str:
    """Normalize claim typography before matching stale public statements."""
    return " ".join(value.replace("×", "x").split()).lower()


def has_qualified_artifact_id(value: str) -> bool:
    """Accept only a concrete qualified-run artifact identifier."""
    return QUALIFIED_ARTIFACT_ID.search(value) is not None


def is_32k_success_claim(value: str) -> bool:
    """Recognize a positive 32K qualification statement, not a skipped leg."""
    lowered = value.lower()
    if any(marker in lowered for marker in ("not qualified", "skipped", "not run")):
        return False
    return THIRTY_TWO_K_SUCCESS.search(value) is not None


def test_docs_do_not_contain_unqualified_legacy_memory_claims() -> None:
    corpus = normalize_claim_text(read_public_docs())

    for stale_claim in (
        "5.2x compression",
        "4.5x",
        "status: met_for_4k_8k_16k_32k_t6",
        "scripts/run_turboquant.py",
        "gate is closed through 32k",
        "validated-qwen 32k niah",
    ):
        assert stale_claim not in corpus


def test_qualified_artifact_reference_rejects_generic_label() -> None:
    assert not has_qualified_artifact_id("a qualified artifact")
    assert has_qualified_artifact_id("qualified-v4/20260718T053419Z-unknown")


def test_claim_normalization_handles_unicode_multiplication_and_whitespace() -> None:
    assert normalize_claim_text("5.2×   compression") == "5.2x compression"


def test_32k_success_detection_covers_natural_language() -> None:
    assert is_32k_success_claim("32K is now qualified")
    assert is_32k_success_claim("A successful 32K run")
    assert not is_32k_success_claim("32K: skipped")
    assert not is_32k_success_claim("32K is not qualified")


def test_any_future_32k_success_claim_must_name_a_qualified_artifact() -> None:
    for path in PUBLIC_DOCUMENTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            lowered = line.lower()
            if "no 32k success" in lowered:
                continue
            if is_32k_success_claim(line):
                assert has_qualified_artifact_id(line), (
                    f"32K success claim in {path.name} must name a qualified artifact: {line}"
                )


def test_readme_links_direct_profile_and_benchmark_protocol() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "[direct profile](configs/profiles/qwen25-7b-direct-v1.json)" in readme
    assert "[benchmark protocol](docs/BENCHMARK_PROTOCOL.md)" in readme


def test_results_record_the_non_qualified_evidence_outcome() -> None:
    results_path = ROOT / "docs" / "RESULTS.md"
    assert results_path.exists()
    results = results_path.read_text(encoding="utf-8")

    assert "20260718T053419Z-unknown" in results
    assert "Complete 4K-8K matrix: baseline 12/12; direct profile 0/12" in results
    assert "Exact 8K subset: baseline 6/6; direct profile 0/6" in results
    assert "32K: skipped" in results
    assert "not retrieval-safe" in results
