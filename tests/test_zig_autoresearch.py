from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "zig_autoresearch.py"
SPEC = importlib.util.spec_from_file_location("zig_autoresearch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_key_value_output_extracts_numeric_metrics():
    stdout = """
Cached decode benchmark
elapsed_s: 43.155244
cached_decode_tok_per_s: 0.04634430985953873
checksum: 86.02596092224121
note: benchmark
"""

    metrics = MODULE.parse_key_value_output(stdout)

    assert metrics["elapsed_s"] == 43.155244
    assert metrics["cached_decode_tok_per_s"] == 0.04634430985953873
    assert metrics["checksum"] == 86.02596092224121
    assert metrics["note"] == "benchmark"


def test_default_round_messages_are_stable():
    round_path = Path(__file__).resolve().parents[1] / "scripts" / "zig_round.py"
    round_spec = importlib.util.spec_from_file_location("zig_round", round_path)
    assert round_spec is not None and round_spec.loader is not None
    round_module = importlib.util.module_from_spec(round_spec)
    sys.modules[round_spec.name] = round_module
    round_spec.loader.exec_module(round_module)

    assert round_module.default_code_message("r007", "simd q6 dot") == "r007: simd q6 dot"
    assert (
        round_module.default_results_message("r007", "simd q6 dot", "abc1234")
        == "r007 results: simd q6 dot [code abc1234]"
    )


def test_round_status_parser_keeps_leading_space_paths():
    round_path = Path(__file__).resolve().parents[1] / "scripts" / "zig_round.py"
    round_spec = importlib.util.spec_from_file_location("zig_round", round_path)
    assert round_spec is not None and round_spec.loader is not None
    round_module = importlib.util.module_from_spec(round_spec)
    sys.modules[round_spec.name] = round_module
    round_spec.loader.exec_module(round_module)

    completed = __import__("subprocess").CompletedProcess(
        args=["git", "status", "--porcelain=v1"],
        returncode=0,
        stdout=" M experiments/results.csv\n?? experiments/run.json\n",
    )

    original = round_module.subprocess.check_output
    round_module.subprocess.check_output = lambda *args, **kwargs: completed.stdout
    try:
        entries = round_module.git_status()
    finally:
        round_module.subprocess.check_output = original

    assert entries[0].path == "experiments/results.csv"
    assert entries[0].is_experiment is True
    assert entries[1].path == "experiments/run.json"
    assert entries[1].is_experiment is True


def test_build_suite_specs_respects_profiles():
    assert [spec.name for spec in MODULE.build_suite_specs("model.gguf", 42, "micro")] == [
        "logits_matvec",
        "blk0_qkv_projection",
    ]
    assert [spec.name for spec in MODULE.build_suite_specs("model.gguf", 42, "decode")] == [
        "full_token_pass",
        "cached_decode",
    ]
    assert [spec.name for spec in MODULE.build_suite_specs("model.gguf", 42, "full")] == [
        "logits_matvec",
        "blk0_qkv_projection",
        "full_token_pass",
        "cached_decode",
    ]


def test_build_suite_specs_passes_metal_decode_flag_to_decode_suites():
    specs = MODULE.build_suite_specs("model.gguf", 42, "decode", metal_decode=True)

    assert specs[0].name == "full_token_pass"
    assert specs[1].name == "cached_decode"
    assert "--metal-decode" in specs[0].command
    assert "--metal-decode" in specs[1].command


def test_zig_build_command_uses_release_flags():
    assert MODULE.zig_build_command("Debug") == ["zig", "build"]
    assert MODULE.zig_build_command("ReleaseFast") == ["zig", "build", "--release=fast"]
    assert MODULE.zig_build_command("ReleaseSafe") == ["zig", "build", "--release=safe"]
    assert MODULE.zig_build_command("ReleaseSmall") == ["zig", "build", "--release=small"]


def test_render_summary_and_svg_include_latest_metrics():
    rows = [
        {
            "timestamp_utc": "2026-04-21T12:00:00+00:00",
            "run_id": "run-a",
            "profile": "full",
            "label": "baseline",
            "notes": "baseline",
            "suite": "cached_decode",
            "suite_title": "Cached Decode",
            "metric_key": "cached_decode_tok_per_s",
            "metric_value": 0.04,
            "metric_unit": "tok/s",
            "git_branch": "codex/zig",
            "git_short_commit": "aaaa111",
            "git_dirty": False,
            "git_subject": "baseline",
        },
        {
            "timestamp_utc": "2026-04-21T12:00:00+00:00",
            "run_id": "run-a",
            "profile": "full",
            "label": "baseline",
            "notes": "baseline",
            "suite": "full_token_pass",
            "suite_title": "Fresh Full Token Pass",
            "metric_key": "fresh_token_tok_per_s",
            "metric_value": 0.05,
            "metric_unit": "tok/s",
            "git_branch": "codex/zig",
            "git_short_commit": "aaaa111",
            "git_dirty": False,
            "git_subject": "baseline",
        },
        {
            "timestamp_utc": "2026-04-21T13:00:00+00:00",
            "run_id": "run-b",
            "profile": "full",
            "label": "simd q6",
            "notes": "simd q6",
            "suite": "cached_decode",
            "suite_title": "Cached Decode",
            "metric_key": "cached_decode_tok_per_s",
            "metric_value": 0.06,
            "metric_unit": "tok/s",
            "git_branch": "codex/zig",
            "git_short_commit": "bbbb222",
            "git_dirty": False,
            "git_subject": "simd q6",
        },
        {
            "timestamp_utc": "2026-04-21T13:00:00+00:00",
            "run_id": "run-b",
            "profile": "full",
            "label": "simd q6",
            "notes": "simd q6",
            "suite": "full_token_pass",
            "suite_title": "Fresh Full Token Pass",
            "metric_key": "fresh_token_tok_per_s",
            "metric_value": 0.07,
            "metric_unit": "tok/s",
            "git_branch": "codex/zig",
            "git_short_commit": "bbbb222",
            "git_dirty": False,
            "git_subject": "simd q6",
        },
    ]

    summary = MODULE.render_summary(rows)
    svg = MODULE.render_svg(rows)

    assert "Latest Run" in summary
    assert "simd q6" in summary
    assert "+0.02000" in summary
    assert "Cached Decode" in svg
    assert "Pure-Zig Optimization History" in svg
