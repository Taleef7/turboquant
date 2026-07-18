"""Contract tests for self-validating benchmark artifacts."""

from __future__ import annotations

from scripts.benchmark_schema import (
    QualityMatrix,
    calculate_paired_bootstrap,
    validate_quality_artifact,
)


def _binding() -> dict[str, str]:
    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a" * 40,
        "model_revision_sha256": "m" * 64,
        "model_manifest_sha256": "h" * 64,
        "profile_sha256": "p" * 64,
        "codebook_sha256": "c" * 64,
        "code_sha256": "d" * 64,
        "runner_sha256": "r" * 64,
    }


def _row(context: int, depth: float, seed: int) -> dict:
    return {
        "context_target": context,
        "context_actual": context,
        "depth": depth,
        "trial_seed": seed,
        "status": "completed",
        "needle_present_after_tokenization": True,
        "oom": False,
        "exception": None,
        "baseline_correct": True,
        "profile_correct": True,
    }


def complete_matrix() -> dict:
    expected = {"contexts": [1024, 4096], "depths": [0.1], "seeds": [0, 1]}
    rows = [
        _row(context, depth, seed)
        for context in expected["contexts"]
        for depth in expected["depths"]
        for seed in expected["seeds"]
    ]
    return {
        "artifact_type": "quality_matrix",
        "schema_version": 1,
        "binding": _binding(),
        "metadata": {
            "timestamp_utc": "2026-07-18T00:00:00+00:00",
            "git_commit": "a" * 40,
        },
        "status": "completed",
        "expected_matrix": expected,
        "rows": rows,
    }


def test_artifact_rejects_omitted_matrix_row():
    matrix = complete_matrix()
    matrix["rows"].pop()

    result = validate_quality_artifact(matrix)

    assert result.valid is False
    assert "expected row count" in result.errors[0]


def test_quality_matrix_is_typed_and_valid_when_complete():
    matrix = QualityMatrix.from_dict(complete_matrix())

    assert matrix.expected_row_count == 4
    assert validate_quality_artifact(matrix.to_dict()).valid is True


def test_artifact_rejects_non_completed_token_loss_and_runtime_failure():
    matrix = complete_matrix()
    matrix["rows"][0]["status"] = "error"
    matrix["rows"][1]["needle_present_after_tokenization"] = False
    matrix["rows"][2]["oom"] = True
    matrix["rows"][3]["exception"] = "CUDA failure"

    result = validate_quality_artifact(matrix)

    assert result.valid is False
    assert any("status" in error for error in result.errors)
    assert any("tokenization" in error for error in result.errors)
    assert any("OOM" in error for error in result.errors)
    assert any("exception" in error for error in result.errors)


def test_paired_bootstrap_is_deterministic_and_uses_declared_seed():
    matrix = complete_matrix()
    matrix["rows"][1]["profile_correct"] = False
    matrix["rows"][3]["profile_correct"] = False

    first = calculate_paired_bootstrap(matrix["rows"], seed=1729, iterations=4000)
    second = calculate_paired_bootstrap(matrix["rows"], seed=1729, iterations=4000)

    assert first == second
    assert first.seed == 1729
    assert first.iterations == 4000
    assert first.upper_95_pp >= first.delta_pp


def test_quality_requires_top_level_status_and_metadata():
    matrix = complete_matrix()
    matrix.pop("status")
    matrix.pop("metadata")

    result = validate_quality_artifact(matrix)

    assert result.valid is False
    assert any("quality status" in error for error in result.errors)
    assert any("metadata" in error for error in result.errors)


def test_quality_artifact_rejects_materially_underfilled_requested_context():
    matrix = complete_matrix()
    matrix["rows"][0]["context_actual"] = 900

    result = validate_quality_artifact(matrix)

    assert result.valid is False
    assert any("context fidelity" in error for error in result.errors)


def test_quality_artifact_requires_actual_context_length_for_every_row():
    matrix = complete_matrix()
    matrix["rows"][0].pop("context_actual")

    result = validate_quality_artifact(matrix)

    assert result.valid is False
    assert any("context_actual" in error for error in result.errors)
