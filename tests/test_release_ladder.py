"""Release-ladder decisions must be explicit and mechanically enforceable."""

from scripts.reproduce_release import estimate_qualified_capacity, qualified_ladder_decision


def test_qualified_ladder_skips_32k_when_preflight_fails():
    decision = qualified_ladder_decision(
        preflight_valid=False, eight_k_retrieval_safe=True
    )

    assert decision["status"] == "skipped"
    assert decision["attempt_32k"] is False
    assert decision["reason"] == "preflight_invalid"


def test_qualified_ladder_skips_32k_when_8k_retrieval_gate_fails():
    decision = qualified_ladder_decision(
        preflight_valid=True, eight_k_retrieval_safe=False
    )

    assert decision["status"] == "skipped"
    assert decision["attempt_32k"] is False
    assert decision["reason"] == "8k_retrieval_gate_failed"


def test_qualified_ladder_skips_32k_when_8k_systems_are_not_clean():
    decision = qualified_ladder_decision(
        preflight_valid=True, eight_k_retrieval_safe=True, eight_k_systems_valid=False
    )

    assert decision["attempt_32k"] is False
    assert decision["reason"] == "8k_systems_invalid_or_failed"


def test_qualified_ladder_distinguishes_failed_8k_capacity_admission():
    decision = qualified_ladder_decision(
        preflight_valid=True,
        eight_k_retrieval_safe=True,
        eight_k_systems_valid=True,
        eight_k_capacity_admitted=False,
    )

    assert decision["status"] == "skipped"
    assert decision["attempt_32k"] is False
    assert decision["reason"] == "8k_capacity_admission_failed"


def test_qualified_ladder_attempts_32k_only_after_all_required_gates_pass():
    decision = qualified_ladder_decision(
        preflight_valid=True,
        eight_k_retrieval_safe=True,
        eight_k_systems_valid=True,
        eight_k_capacity_admitted=True,
    )

    assert decision["status"] == "approved"
    assert decision["attempt_32k"] is True


def test_capacity_preflight_projects_32k_from_1k_measured_peaks_and_free_vram():
    artifact = {
        "binding": {"model_sha256": "m"},
        "metadata": {"gpu_total_vram_bytes": 12_000, "gpu_free_vram_bytes": 8_000},
        "trials": [{"max_memory_reserved": 4_000, "actual_store_bytes": 500}],
    }

    estimate = estimate_qualified_capacity(artifact, target_context=32_768, source_context=1_024)

    assert estimate["projected_32k_bytes"] > 4_000
    assert estimate["source_binding"] == artifact["binding"]
    assert estimate["passes"] is False
    assert "formula" in estimate


def test_capacity_uses_measured_8k_reserved_peak_when_available():
    artifact = {"binding": {"model_sha256": "m"}, "metadata": {"gpu_total_vram_bytes": 20_000, "gpu_free_vram_bytes": 15_000}, "trials": [{"max_memory_reserved": 10_000, "actual_store_bytes": 1_000}]}
    estimate = estimate_qualified_capacity(artifact, target_context=32_768, source_context=8_192)
    assert estimate["source_context"] == 8192
    assert estimate["projected_32k_bytes"] >= 10_000
