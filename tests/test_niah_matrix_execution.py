"""NIAH expected cells exist before any GPU execution begins."""

from scripts.test_long_context import build_expected_niah_rows, execute_expected_niah_rows


def test_expected_niah_rows_are_prebuilt_and_preserved_after_early_error():
    expected = build_expected_niah_rows([1024], [0.1, 0.9], [0, 1])

    def execute(row):
        if row["trial_seed"] == 0:
            raise RuntimeError("fixture failure")
        return {"baseline_correct": True, "profile_correct": True, "needle_present_after_tokenization": True}

    rows = execute_expected_niah_rows(expected, execute)

    assert len(rows) == 4
    assert sum(row["status"] == "error" for row in rows) == 2
    assert sum(row["status"] == "completed" for row in rows) == 2
    assert all({"context_target", "depth", "trial_seed"} <= row.keys() for row in rows)
