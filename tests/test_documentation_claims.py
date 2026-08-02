"""Regression checks for public TurboQuant Forge claims.

These checks intentionally examine the release-facing documents only.  Design
and implementation plans retain historical context and are not public evidence
artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "REPRODUCTION.md",
    ROOT / "docs" / "CLAIMS.md",
    ROOT / "TESTING_RESULTS.md",
    ROOT / "ISSUES.md",
    ROOT / "docs" / "PAPER_CLAIMS_STATUS.md",
    ROOT / "docs" / "PAPER_COMPARISON.md",
    ROOT / "docs" / "REPLICATION_CHECKLIST.md",
    ROOT / "docs" / "PORTABILITY.md",
    ROOT / "docs" / "BENCHMARK_PROTOCOL.md",
    ROOT / "docs" / "RESULTS.md",
    ROOT / "docs" / "GILBRETH_HANDOFF.md",
    ROOT / "UPDATED_PLAN.md",
    ROOT / "scripts" / "README.md",
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


def public_claim_corpus() -> str:
    """Return normalized text for all public plans and script documentation."""
    return normalize_claim_text(
        "\n".join(
            path.read_text(encoding="utf-8") if path.exists() else ""
            for path in PUBLIC_DOCUMENTS
        )
    )


def read_public_docs() -> str:
    """Return one normalized corpus of public claim-bearing documentation."""
    return public_claim_corpus()


def load_ci_steps() -> tuple[SimpleNamespace, ...]:
    """Read the simple `run` steps in the checked-in CPU workflow without PyYAML."""
    lines = (
        (ROOT / ".github" / "workflows" / "ci.yml")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    steps: list[SimpleNamespace] = []
    index = 0
    while index < len(lines):
        match = re.match(r"^        run:\s*(.*)$", lines[index])
        if match is None:
            index += 1
            continue
        run = match.group(1)
        if run == "|":
            body: list[str] = []
            index += 1
            while index < len(lines) and lines[index].startswith("          "):
                body.append(lines[index].removeprefix("          "))
                index += 1
            run = "\n".join(body)
        else:
            index += 1
        steps.append(SimpleNamespace(run=run))
    return tuple(steps)


def is_pytest_invocation(run: str) -> bool:
    """Recognize the CPU test command, not installation of pytest itself."""
    return "python -m pytest" in run


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
    corpus = public_claim_corpus()

    for stale_claim in (
        "5.2x compression",
        "4.5x",
        "status: met_for_4k_8k_16k_32k_t6",
        "scripts/run_turboquant.py",
        "gate is closed through 32k",
        "validated-qwen 32k niah",
    ):
        assert stale_claim not in corpus


def test_all_public_plans_and_script_docs_use_current_claims() -> None:
    corpus = public_claim_corpus()

    assert "5.2x" not in corpus
    assert "full through 16k" not in corpus
    assert ">=80% of baseline" not in corpus


def test_pytest_invocation_detection_excludes_dependency_installation() -> None:
    assert not is_pytest_invocation("python -m pip install pytest ruff")
    assert is_pytest_invocation('python -m pytest tests -m "not gpu" -q')


def test_ci_installs_locked_runtime_before_pytest() -> None:
    steps = load_ci_steps()
    install_index = next(
        index
        for index, step in enumerate(steps)
        if "requirements/runtime.lock" in getattr(step, "run", "")
    )
    test_index = next(
        index
        for index, step in enumerate(steps)
        if is_pytest_invocation(getattr(step, "run", ""))
    )

    assert install_index < test_index


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
                assert has_qualified_artifact_id(
                    line
                ), f"32K success claim in {path.name} must name a qualified artifact: {line}"


def test_readme_links_direct_profile_and_benchmark_protocol() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "[direct profile](configs/profiles/qwen25-7b-direct-v1.json)" in readme
    assert "[benchmark protocol](docs/BENCHMARK_PROTOCOL.md)" in readme


def test_legacy_retrieval_profile_cannot_claim_current_32k_support() -> None:
    profile = json.loads(
        (ROOT / "configs" / "retrieval_profile.json").read_text(encoding="utf-8")
    )
    assert profile["name"].startswith("legacy-")
    assert profile["quality_gate"]["status"] == "legacy_unqualified"
    assert "not 32K-qualified" in profile["description"]


def test_reproduction_report_binds_the_canonical_codec_artifact() -> None:
    report = (ROOT / "docs" / "REPRODUCTION.md").read_text(encoding="utf-8")
    claims = (ROOT / "docs" / "CLAIMS.md").read_text(encoding="utf-8")

    for document in (report, claims):
        assert "codec-local-v1" in document
        assert "871cbc9058d4befe4b0f5e35b687d1b5f525cb2c" in document
        assert "1,408" in document
        assert "3,858" in document

    assert "108/108" in report
    assert "588/588" in report
    assert "sidecar:artifacts/codec-local-v1/claim-summary.json" in claims
    assert "claims.json" in claims


def test_public_codec_docs_reject_equivalence_and_performance_overclaims() -> None:
    corpus = public_claim_corpus()

    for forbidden in (
        "paper equivalent",
        "paper-scale reproduction",
        "pareto win",
        "32k qualified",
    ):
        assert forbidden not in corpus


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
