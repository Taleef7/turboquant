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
    r"qualified(?:-[a-z0-9]+)*/\d{8}T\d{6}Z-[a-z0-9._-]+", re.IGNORECASE
)
THIRTY_TWO_K_SUCCESS = re.compile(
    r"(?:\b(?:successful(?:ly)?|passed|succeeded|qualified|validated)"
    r"\s+(?:\w+\s+){0,3}32k\b|\b32k\s+(?:\w+\s+){0,4}"
    r"(?:success(?:ful(?:ly)?)?|passed|succeeded|qualified|validated|completed)\b|"
    r"\b(?:retrieval[- ]safe|validated\s+niah)\s+(?:through|at)\s+32k\b|"
    r"\bretrieval\s+gate\s+(?:is\s+)?met\s+(?:through|at)\s+32k\b|"
    r"\b32k\s+(?:retrieval\s+)?gate\s+(?:is\s+)?met\b|"
    r"\b(?:profile\s+)?qualifies\s+(?:through|at)\s+32k\b|"
    r"\b32k\s+clears\s+(?:the\s+)?retrieval\s+gate\b)",
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
    if any(
        marker in lowered
        for marker in (
            "not qualified",
            "not retrieval-safe",
            "skipped",
            "not run",
            "failed",
            "fails",
            "not met",
            "does not qualify",
            "did not qualify",
            "doesn't qualify",
            "does not clear",
            "no 32k success",
        )
    ):
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
    assert has_qualified_artifact_id(
        "qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0"
    )


def test_claim_normalization_handles_unicode_multiplication_and_whitespace() -> None:
    assert normalize_claim_text("5.2×   compression") == "5.2x compression"


def test_32k_success_detection_covers_natural_language() -> None:
    assert is_32k_success_claim("32K is now qualified")
    assert is_32k_success_claim("A successful 32K run")
    assert is_32k_success_claim("32K gate passed")
    assert is_32k_success_claim("Retrieval-safe through 32K")
    assert is_32k_success_claim("32K retrieval validation succeeded")
    assert is_32k_success_claim("Validated NIAH through 32K")
    assert is_32k_success_claim("32K completed successfully")
    assert is_32k_success_claim("The retrieval gate is met through 32K")
    assert is_32k_success_claim("32K gate is met")
    assert is_32k_success_claim("The profile qualifies at 32K")
    assert is_32k_success_claim("32K clears the retrieval gate")
    assert not is_32k_success_claim("32K: skipped")
    assert not is_32k_success_claim("32K is not qualified")
    assert not is_32k_success_claim("The retrieval gate is not met through 32K")
    assert not is_32k_success_claim("The profile does not qualify at 32K")
    assert not is_32k_success_claim("32K fails the retrieval gate")


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


def test_results_record_the_capacity_limited_final_evidence_outcome() -> None:
    results_path = ROOT / "docs" / "RESULTS.md"
    assert results_path.exists()
    results = results_path.read_text(encoding="utf-8")

    assert "qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0" in results
    assert "5c9d4c34b0f0c31d015e2166ae7ad401858b202d" in results
    assert "Structural validation:** valid" in results
    assert "12/12" in results
    assert "exact baseline/direct output-token-ID agreement" in results
    assert "233,447,424" in results
    assert "1.29 tok/s" in results
    assert "33.99 tok/s" in results
    assert "12,809,961,472" in results
    assert "10,405,439,078" in results
    assert "8k_capacity_admission_failed" in results
    assert "No 32K quality or systems child was launched." in results

    # The predecessor stays visible only as retired invalid history.
    assert "20260718T053419Z-unknown" in results
    assert "7,751/7,754" in results
    assert "retired invalid artifact" in results
