"""Unit tests for reproduction runner helpers."""

import json

from reproduce_release import build_niah_command, extract_delta_pp


def test_build_niah_command_quick_uses_8k_profile():
    cmd = build_niah_command("quick", ".tmp/test_quick")
    assert "--max-context" in cmd
    assert cmd[cmd.index("--max-context") + 1] == "8192"
    assert cmd[cmd.index("--trials") + 1] == "2"


def test_build_niah_command_full_uses_16k_profile():
    cmd = build_niah_command("full", ".tmp/test_full")
    assert cmd[cmd.index("--max-context") + 1] == "16384"
    assert cmd[cmd.index("--buffer-size") + 1] == "12288"
    assert cmd[cmd.index("--trials") + 1] == "6"


def test_extract_delta_pp_from_rows(tmp_path):
    p = tmp_path / "rows.json"
    rows = [
        {
            "needle_present_after_tokenization": True,
            "baseline_correct": True,
            "turboquant_correct": True,
        },
        {
            "needle_present_after_tokenization": True,
            "baseline_correct": True,
            "turboquant_correct": False,
        },
        {
            "needle_present_after_tokenization": False,
            "baseline_correct": False,
            "turboquant_correct": False,
        },
    ]
    p.write_text(json.dumps(rows), encoding="utf-8")
    # Eligible rows: 2, baseline=100%, turbo=50% => delta=50pp
    assert extract_delta_pp(str(p)) == 50.0
