from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTORESEARCH_PATH = REPO_ROOT / "scripts" / "zig_autoresearch.py"
ROUND_PATH = REPO_ROOT / "scripts" / "zig_round.py"
TRACE_DIR = REPO_ROOT / "experiments" / "traces"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module("zig_autoresearch", AUTORESEARCH_PATH)
ROUND_MODULE = load_module("zig_round", ROUND_PATH)


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
    assert ROUND_MODULE.default_code_message("r007", "simd q6 dot") == "r007: simd q6 dot"
    assert (
        ROUND_MODULE.default_results_message("r007", "simd q6 dot", "abc1234")
        == "r007 results: simd q6 dot [code abc1234]"
    )


def test_round_status_parser_keeps_leading_space_paths():
    completed = __import__("subprocess").CompletedProcess(
        args=["git", "status", "--porcelain=v1"],
        returncode=0,
        stdout=" M experiments/results.csv\n?? experiments/run.json\n",
    )

    original = ROUND_MODULE.subprocess.check_output
    ROUND_MODULE.subprocess.check_output = lambda *args, **kwargs: completed.stdout
    try:
        entries = ROUND_MODULE.git_status()
    finally:
        ROUND_MODULE.subprocess.check_output = original

    assert entries[0].path == "experiments/results.csv"
    assert entries[0].is_experiment is True
    assert entries[1].path == "experiments/run.json"
    assert entries[1].is_experiment is True


def test_build_suite_specs_respects_profiles():
    assert [spec.name for spec in MODULE.build_suite_specs("model.gguf", 42, "tracked")] == [
        "metal_add_one",
        "metal_qkv_projection",
        "metal_logits_projection",
        "cached_decode_trace",
    ]
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


def test_tracked_profile_forces_metal_decode_on_decode_suites():
    specs = MODULE.build_suite_specs("model.gguf", 42, "tracked")
    decode_specs = [spec for spec in specs if spec.name in {"cached_decode_trace"}]

    assert len(decode_specs) == 1
    assert all("--metal-decode" in spec.command for spec in decode_specs)
    assert all("--token-seq-file" in spec.command for spec in decode_specs)


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
            "run_outcome": "accepted",
            "run_backend": "metal+metal-cache",
            "run_coverage": "metal-kernels+decode",
            "suite_backend": "metal-cache",
            "suite_coverage": "cached-decode",
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
            "run_outcome": "accepted",
            "run_backend": "metal+metal-cache",
            "run_coverage": "metal-kernels+decode",
            "suite_backend": "metal-cache",
            "suite_coverage": "fresh-token",
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
            "run_outcome": "rejected",
            "run_backend": "metal+metal-cache",
            "run_coverage": "metal-kernels+decode",
            "suite_backend": "metal-cache",
            "suite_coverage": "cached-decode",
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
            "run_outcome": "rejected",
            "run_backend": "metal+metal-cache",
            "run_coverage": "metal-kernels+decode",
            "suite_backend": "metal-cache",
            "suite_coverage": "fresh-token",
        },
    ]

    summary = MODULE.render_summary(rows)
    svg = MODULE.render_svg(rows)

    assert "Latest Run" in summary
    assert "Outcome: `rejected`" in summary
    assert "Backend: `metal+metal-cache`" in summary
    assert "Coverage: `metal-kernels+decode`" in summary
    assert "simd q6" in summary
    assert "+0.02000" in summary
    assert "| Cached Decode | 0.04000 tok/s | `aaaa111` |" in summary
    assert "Cached Decode" in svg
    assert "Pure-Zig Optimization History" in svg


def make_result_row(
    *,
    timestamp_utc: str,
    run_id: str,
    profile: str,
    suite: str,
    metric_value: float,
    git_short_commit: str,
    run_outcome: str = "",
    run_backend: str = "metal+metal-cache",
    run_coverage: str = "metal-kernels+decode",
    metrics_json: dict[str, object] | None = None,
) -> dict[str, object]:
    suite_titles = {
        "metal_add_one": "Metal Add-One",
        "metal_qkv_projection": "Metal QKV Projection",
        "metal_logits_projection": "Metal Logits Projection",
        "cached_decode_trace": "Tracked Cached Decode",
        "full_token_pass": "Fresh Full Token Pass",
        "cached_decode": "Cached Decode",
    }
    metric_keys = {
        "metal_add_one": "metal_elements_per_s",
        "metal_qkv_projection": "metal_projection_passes_per_s",
        "metal_logits_projection": "metal_projection_passes_per_s",
        "cached_decode_trace": "steady_decode_tok_per_s",
        "full_token_pass": "fresh_token_tok_per_s",
        "cached_decode": "cached_decode_tok_per_s",
    }
    metric_units = {
        "metal_add_one": "elements/s",
        "metal_qkv_projection": "projection/s",
        "metal_logits_projection": "projection/s",
        "cached_decode_trace": "tok/s",
        "full_token_pass": "tok/s",
        "cached_decode": "tok/s",
    }
    suite_backends = {
        "metal_add_one": "metal",
        "metal_qkv_projection": "metal",
        "metal_logits_projection": "metal",
        "cached_decode_trace": "metal-cache",
        "full_token_pass": "metal-cache",
        "cached_decode": "metal-cache",
    }
    suite_coverages = {
        "metal_add_one": "metal-kernel",
        "metal_qkv_projection": "metal-kernel",
        "metal_logits_projection": "metal-kernel",
        "cached_decode_trace": "tracked-decode",
        "full_token_pass": "fresh-token",
        "cached_decode": "cached-decode",
    }
    return {
        "timestamp_utc": timestamp_utc,
        "run_id": run_id,
        "profile": profile,
        "round_id": "r200",
        "label": run_id,
        "notes": run_id,
        "suite": suite,
        "suite_title": suite_titles[suite],
        "metric_key": metric_keys[suite],
        "metric_value": metric_value,
        "metric_unit": metric_units[suite],
        "git_branch": "codex/zig-baseline-qwen36",
        "git_commit": f"{git_short_commit}000000000000000000000000000000000000000",
        "git_short_commit": git_short_commit,
        "git_dirty": False,
        "git_subject": run_id,
        "decode_backend": "metal-cache",
        "run_backend": run_backend,
        "run_coverage": run_coverage,
        "suite_backend": suite_backends[suite],
        "suite_coverage": suite_coverages[suite],
        "run_outcome": run_outcome,
        "metrics_json": metrics_json or {},
        "gate_baseline_run_id": "",
        "gate_max_regression_pct": "",
        "gate_regressions_json": [],
        "decision_trace": "",
    }


def test_update_run_rows_persists_outcome_and_trace_metadata(tmp_path: Path):
    results_csv = tmp_path / "results.csv"
    MODULE.append_rows(
        results_csv,
        [
            make_result_row(
                timestamp_utc="2026-04-21T17:14:45+00:00",
                run_id="run-a",
                profile="tracked",
                suite="cached_decode_trace",
                metric_value=1.05,
                git_short_commit="aaaa111",
            ),
        ],
    )

    updated = MODULE.update_run_rows(
        results_csv,
        "run-a",
        {
            "run_outcome": "accepted",
            "gate_baseline_run_id": "baseline-1",
            "gate_max_regression_pct": 5.0,
            "gate_regressions_json": [{"suite": "cached_decode_trace", "delta_pct": -1.5}],
            "decision_trace": tmp_path / "trace.json",
        },
    )

    assert updated == 1
    rows = MODULE.load_result_rows(results_csv)
    assert all(row["run_outcome"] == "accepted" for row in rows)
    assert all(row["gate_baseline_run_id"] == "baseline-1" for row in rows)
    assert rows[0]["gate_max_regression_pct"] == 5.0
    assert rows[0]["gate_regressions_json"][0]["suite"] == "cached_decode_trace"
    assert rows[0]["decision_trace"] == str(tmp_path / "trace.json")


def test_round_evaluate_run_accepts_improvements(tmp_path: Path):
    results_csv = tmp_path / "results.csv"
    MODULE.append_rows(
        results_csv,
        [
            make_result_row(
                timestamp_utc="2026-04-21T17:14:45+00:00",
                run_id="baseline",
                profile="tracked",
                suite="cached_decode_trace",
                metric_value=1.05,
                git_short_commit="aaaa111",
                run_outcome="accepted",
                metrics_json={
                    "argmax_mismatch_count": 0,
                    "max_topk_logit_delta": 0.0,
                    "mean_topk_logit_delta": 0.0,
                    "metal_project_hit_rate": 70.0,
                    "cpu_fallback_calls": 2200,
                    "first_decode_token_ms": 9000,
                },
            ),
            make_result_row(
                timestamp_utc="2026-04-21T18:14:45+00:00",
                run_id="candidate",
                profile="tracked",
                suite="cached_decode_trace",
                metric_value=1.08,
                git_short_commit="bbbb222",
                metrics_json={
                    "argmax_mismatch_count": 0,
                    "max_topk_logit_delta": 0.0,
                    "mean_topk_logit_delta": 0.0,
                    "metal_project_hit_rate": 82.0,
                    "cpu_fallback_calls": 1800,
                    "first_decode_token_ms": 8900,
                },
            ),
        ],
    )

    decision = ROUND_MODULE.evaluate_run(results_csv, "candidate", 5.0)

    assert decision.outcome == MODULE.RUN_OUTCOME_ACCEPTED
    assert decision.baseline_run_id == "baseline"
    assert len(decision.comparisons) == 1
    assert decision.regressions == []


def test_round_evaluate_run_rejects_large_regressions(tmp_path: Path):
    results_csv = tmp_path / "results.csv"
    MODULE.append_rows(
        results_csv,
        [
            make_result_row(
                timestamp_utc="2026-04-21T17:14:45+00:00",
                run_id="baseline",
                profile="tracked",
                suite="cached_decode_trace",
                metric_value=1.05,
                git_short_commit="aaaa111",
                run_outcome="accepted",
                metrics_json={
                    "argmax_mismatch_count": 0,
                    "max_topk_logit_delta": 0.0,
                    "mean_topk_logit_delta": 0.0,
                    "metal_project_hit_rate": 70.0,
                    "cpu_fallback_calls": 2200,
                    "first_decode_token_ms": 9000,
                },
            ),
            make_result_row(
                timestamp_utc="2026-04-21T18:14:45+00:00",
                run_id="candidate",
                profile="tracked",
                suite="cached_decode_trace",
                metric_value=0.97,
                git_short_commit="bbbb222",
                metrics_json={
                    "argmax_mismatch_count": 0,
                    "max_topk_logit_delta": 0.0,
                    "mean_topk_logit_delta": 0.0,
                    "metal_project_hit_rate": 68.0,
                    "cpu_fallback_calls": 2300,
                    "first_decode_token_ms": 9200,
                },
            ),
        ],
    )

    decision = ROUND_MODULE.evaluate_run(results_csv, "candidate", 5.0)

    assert decision.outcome == MODULE.RUN_OUTCOME_REJECTED
    assert decision.baseline_run_id == "baseline"
    assert [check.suite for check in decision.regressions] == ["cached_decode_trace"]


def test_annotate_run_writes_trace_and_updates_run_artifact(tmp_path: Path):
    results_csv = tmp_path / "results.csv"
    summary_md = tmp_path / "summary.md"
    plot_svg = tmp_path / "results.svg"
    run_artifact = tmp_path / "candidate.json"
    trace_path = tmp_path / "traces" / "candidate.json"

    MODULE.append_rows(
        results_csv,
        [
            make_result_row(
                timestamp_utc="2026-04-21T18:14:45+00:00",
                run_id="candidate",
                profile="tracked",
                suite="cached_decode_trace",
                metric_value=1.08,
                git_short_commit="bbbb222",
            ),
        ],
    )
    MODULE.write_run_artifact(
        run_artifact,
        {
            "run_id": "candidate",
            "profile": "tracked",
            "steps": [],
        },
    )

    decision = ROUND_MODULE.GateDecision(
        outcome="accepted",
        baseline_run_id="baseline",
        run_backend="metal+metal-cache",
        run_coverage="metal-kernels+decode",
        comparisons=[
            ROUND_MODULE.RegressionCheck(
                suite="cached_decode_trace",
                suite_title="Tracked Cached Decode",
                baseline_run_id="baseline",
                current_value=1.08,
                baseline_value=1.05,
                delta_pct=2.857142857142857,
            )
        ],
        regressions=[],
    )
    args = SimpleNamespace(
        round_id="r200",
        profile="tracked",
        notes="fixture",
        max_regression_pct=5.0,
        skip_regression_gate=False,
    )

    ROUND_MODULE.annotate_run(
        results_csv=results_csv,
        summary_md=summary_md,
        plot_svg=plot_svg,
        run_artifact=run_artifact,
        trace_path=trace_path,
        args=args,
        branch="codex/zig-baseline-qwen36",
        code_commit="bbbb222000000000000000000000000000000000",
        run_id="candidate",
        decision=decision,
    )

    rows = MODULE.load_result_rows(results_csv)
    assert all(row["run_outcome"] == "accepted" for row in rows)
    assert rows[0]["decision_trace"] == str(trace_path)
    assert json.loads(trace_path.read_text())["decision"] == "accepted"
    artifact = json.loads(run_artifact.read_text())
    assert artifact["run_outcome"] == "accepted"
    assert artifact["gate"]["baseline_run_id"] == "baseline"
    assert summary_md.exists()
    assert plot_svg.exists()


def test_trace_fixtures_cover_accept_and_reject_schema():
    accepted = json.loads((TRACE_DIR / "accepted-tracked-round.json").read_text())
    rejected = json.loads((TRACE_DIR / "rejected-tracked-round.json").read_text())

    assert set(accepted) == set(rejected)
    assert accepted["decision"] == "accepted"
    assert accepted["regressions"] == []
    assert rejected["decision"] == "rejected"
    assert rejected["regressions"][0]["suite"] == "cached_decode_trace"
