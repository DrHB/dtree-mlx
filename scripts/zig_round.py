#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTORESEARCH = REPO_ROOT / "scripts" / "zig_autoresearch.py"
TRACE_DIR = REPO_ROOT / "experiments" / "traces"


def load_autoresearch_module() -> Any:
    spec = importlib.util.spec_from_file_location("zig_autoresearch_runtime", AUTORESEARCH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load {AUTORESEARCH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUTORESEARCH_MODULE = load_autoresearch_module()


@dataclass(frozen=True)
class StatusEntry:
    code: str
    path: str

    @property
    def is_experiment(self) -> bool:
        return self.path == "experiments" or self.path.startswith("experiments/")


@dataclass(frozen=True)
class RegressionCheck:
    suite: str
    suite_title: str
    baseline_run_id: str
    current_value: float
    baseline_value: float
    delta_pct: float


@dataclass(frozen=True)
class GateDecision:
    outcome: str
    baseline_run_id: str
    run_backend: str
    run_coverage: str
    comparisons: list[RegressionCheck]
    regressions: list[RegressionCheck]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Commit a code round, run Zig eval on that exact commit, then commit and push the results.",
    )
    parser.add_argument("--round-id", required=True, help="Round identifier, for example r001.")
    parser.add_argument("--notes", required=True, help="Short note describing the experiment.")
    parser.add_argument("--profile", choices=AUTORESEARCH_MODULE.PROFILE_CHOICES, default="tracked")
    parser.add_argument("--metal-decode", action="store_true")
    parser.add_argument("--model", default="models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf")
    parser.add_argument("--token-id", type=int, default=42)
    parser.add_argument(
        "--optimize",
        choices=["Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"],
        default="ReleaseFast",
    )
    parser.add_argument("--code-message", default="", help="Commit message for the code change.")
    parser.add_argument("--results-message", default="", help="Commit message for the benchmark artifacts.")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="", help="Branch to push. Defaults to the current branch.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--allow-dirty-experiments", action="store_true")
    parser.add_argument("--allow-no-code-change", action="store_true")
    parser.add_argument("--max-regression-pct", type=float, default=5.0)
    parser.add_argument("--skip-regression-gate", action="store_true")
    parser.add_argument("--traces-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.traces_dir = args.traces_dir.resolve()

    status = git_status()
    code_changes = [entry for entry in status if not entry.is_experiment]
    experiment_changes = [entry for entry in status if entry.is_experiment]

    if experiment_changes and not args.allow_dirty_experiments:
        raise SystemExit(
            "refusing to start with dirty experiment artifacts; commit or clean experiments/ first"
        )

    current_branch = git_output(["branch", "--show-current"])
    branch = args.branch or current_branch
    if not branch:
        raise SystemExit("unable to determine current branch")

    code_message = args.code_message or default_code_message(args.round_id, args.notes)
    code_commit = git_output(["rev-parse", "HEAD"])

    if code_changes:
        if args.dry_run:
            print(f"dry_run_code_commit_message: {code_message}")
        else:
            run_checked(["git", "add", "-A"], "git add code changes")
            run_checked(["git", "commit", "-m", code_message], "git commit code changes")
            code_commit = git_output(["rev-parse", "HEAD"])
    elif not args.allow_no_code_change and not args.dry_run:
        print("no code changes detected; benchmarking current HEAD")

    short_code_commit = git_output(["rev-parse", "--short", code_commit])
    results_message = args.results_message or default_results_message(
        args.round_id,
        args.notes,
        short_code_commit,
    )

    autoresearch_cmd = [
        "python3",
        str(AUTORESEARCH),
        "--profile",
        args.profile,
        "--round-id",
        args.round_id,
        "--notes",
        args.notes,
        "--label",
        args.round_id,
        "--model",
        args.model,
        "--token-id",
        str(args.token_id),
        "--optimize",
        args.optimize,
    ]
    if args.metal_decode:
        autoresearch_cmd.append("--metal-decode")
    if args.skip_build:
        autoresearch_cmd.append("--skip-build")

    if args.dry_run:
        print(f"dry_run_eval_command: {' '.join(autoresearch_cmd)}")
        print(f"dry_run_results_commit_message: {results_message}")
        print(f"dry_run_max_regression_pct: {args.max_regression_pct}")
        print(f"dry_run_trace_dir: {args.traces_dir}")
        print(f"dry_run_push_target: {args.remote} {branch}")
        return

    eval_completed = run_checked(autoresearch_cmd, "zig_autoresearch")
    eval_meta = parse_eval_metadata(eval_completed.stdout)
    run_id = required_eval_value(eval_meta, "run_id")
    results_csv = Path(required_eval_value(eval_meta, "results_csv")).resolve()
    summary_md = Path(eval_meta.get("summary_md") or AUTORESEARCH_MODULE.SUMMARY_MD).resolve()
    plot_svg = Path(eval_meta.get("plot_svg") or AUTORESEARCH_MODULE.PLOT_SVG).resolve()
    run_artifact = Path(required_eval_value(eval_meta, "run_artifact")).resolve()

    decision = GateDecision(
        outcome=AUTORESEARCH_MODULE.RUN_OUTCOME_ACCEPTED,
        baseline_run_id="",
        run_backend="",
        run_coverage="",
        comparisons=[],
        regressions=[],
    )
    if not args.skip_regression_gate:
        decision = evaluate_run(results_csv, run_id, args.max_regression_pct)

    trace_path = args.traces_dir / f"{run_id}.json"
    annotate_run(
        results_csv=results_csv,
        summary_md=summary_md,
        plot_svg=plot_svg,
        run_artifact=run_artifact,
        trace_path=trace_path,
        args=args,
        branch=branch,
        code_commit=code_commit,
        run_id=run_id,
        decision=decision,
    )

    post_eval_status = git_status()
    stray_changes = [entry for entry in post_eval_status if not entry.is_experiment]
    if stray_changes:
        paths = ", ".join(entry.path for entry in stray_changes)
        raise SystemExit(f"eval produced unexpected non-experiment changes: {paths}")

    experiment_paths = [entry.path for entry in post_eval_status if entry.is_experiment]
    if not experiment_paths:
        raise SystemExit("no experiment artifacts were produced")

    print(f"run_id: {run_id}")
    print(f"run_outcome: {decision.outcome}")
    if decision.baseline_run_id:
        print(f"baseline_run_id: {decision.baseline_run_id}")
    if trace_path:
        print(f"decision_trace: {trace_path}")

    if decision.outcome == AUTORESEARCH_MODULE.RUN_OUTCOME_REJECTED:
        raise SystemExit(
            reject_message(
                round_id=args.round_id,
                run_id=run_id,
                max_regression_pct=args.max_regression_pct,
                regressions=decision.regressions,
            )
        )

    run_checked(["git", "add", "-A", "--", "experiments"], "git add experiment artifacts")
    run_checked(["git", "commit", "-m", results_message], "git commit experiment artifacts")
    results_commit = git_output(["rev-parse", "HEAD"])

    run_checked(["git", "push", args.remote, branch], "git push")

    print(f"round_id: {args.round_id}")
    print(f"code_commit: {code_commit}")
    print(f"code_short_commit: {short_code_commit}")
    print(f"results_commit: {results_commit}")
    print(f"pushed_branch: {branch}")


def parse_eval_metadata(stdout: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in stdout.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            data[key] = value
    return data


def required_eval_value(data: dict[str, str], key: str) -> str:
    value = data.get(key, "").strip()
    if not value:
        raise SystemExit(f"zig_autoresearch did not emit `{key}`")
    return value


def evaluate_run(results_csv: Path, run_id: str, max_regression_pct: float) -> GateDecision:
    rows = AUTORESEARCH_MODULE.load_result_rows(results_csv)
    ordered_rows = sorted(rows, key=lambda row: str(row.get("timestamp_utc", "")))
    runs = AUTORESEARCH_MODULE.collect_runs(ordered_rows)
    current_run = next((run for run in runs if run["run_id"] == run_id), None)
    if current_run is None:
        raise SystemExit(f"unable to find run `{run_id}` in {results_csv}")

    current_rows = current_run["suites"]
    reference_row = next(iter(current_rows.values()), {})
    current_profile = str(reference_row.get("profile", "") or "")
    current_backend = str(current_run.get("run_backend", "") or "")
    current_coverage = str(current_run.get("run_coverage", "") or "")
    current_branch = str(current_run.get("git_branch", "") or "")

    baseline_run = find_baseline_run(
        runs=runs,
        current_run_id=run_id,
        profile=current_profile,
        run_backend=current_backend,
        run_coverage=current_coverage,
        git_branch=current_branch,
    )
    if baseline_run is None:
        return GateDecision(
            outcome=AUTORESEARCH_MODULE.RUN_OUTCOME_ACCEPTED,
            baseline_run_id="",
            run_backend=current_backend,
            run_coverage=current_coverage,
            comparisons=[],
            regressions=[],
        )

    if current_profile == "tracked":
        return evaluate_tracked_run(current_run, baseline_run)

    comparisons: list[RegressionCheck] = []
    regressions: list[RegressionCheck] = []
    for suite_name in AUTORESEARCH_MODULE.SUITE_ORDER:
        current_row = current_rows.get(suite_name)
        baseline_row = baseline_run["suites"].get(suite_name)
        if not current_row or not baseline_row:
            continue
        baseline_value = float(baseline_row["metric_value"])
        if baseline_value == 0:
            continue
        current_value = float(current_row["metric_value"])
        delta_pct = (current_value - baseline_value) / baseline_value * 100.0
        check = RegressionCheck(
            suite=suite_name,
            suite_title=str(current_row.get("suite_title", suite_name)),
            baseline_run_id=str(baseline_run["run_id"]),
            current_value=current_value,
            baseline_value=baseline_value,
            delta_pct=delta_pct,
        )
        comparisons.append(check)
        if delta_pct < -max_regression_pct:
            regressions.append(check)

    outcome = (
        AUTORESEARCH_MODULE.RUN_OUTCOME_REJECTED
        if regressions
        else AUTORESEARCH_MODULE.RUN_OUTCOME_ACCEPTED
    )
    return GateDecision(
        outcome=outcome,
        baseline_run_id=str(baseline_run["run_id"]),
        run_backend=current_backend,
        run_coverage=current_coverage,
        comparisons=comparisons,
        regressions=regressions,
    )


def evaluate_tracked_run(current_run: dict[str, Any], baseline_run: dict[str, Any]) -> GateDecision:
    current_row = current_run["suites"].get("cached_decode_trace")
    baseline_row = baseline_run["suites"].get("cached_decode_trace")
    if not current_row or not baseline_row:
        raise SystemExit("tracked profile requires `cached_decode_trace` in both current and baseline runs")

    current_value = float(current_row["metric_value"])
    baseline_value = float(baseline_row["metric_value"])
    delta_pct = 0.0 if baseline_value == 0 else (current_value - baseline_value) / baseline_value * 100.0
    comparison = RegressionCheck(
        suite="cached_decode_trace",
        suite_title=str(current_row.get("suite_title", "Tracked Cached Decode")),
        baseline_run_id=str(baseline_run["run_id"]),
        current_value=current_value,
        baseline_value=baseline_value,
        delta_pct=delta_pct,
    )

    metrics = current_row.get("metrics_json") or {}
    baseline_metrics = baseline_row.get("metrics_json") or {}
    mismatch_count = int(metrics.get("argmax_mismatch_count", 0) or 0)
    max_logit_delta = float(metrics.get("max_topk_logit_delta", 0.0) or 0.0)
    mean_logit_delta = float(metrics.get("mean_topk_logit_delta", 0.0) or 0.0)
    current_hit_rate = float(metrics.get("metal_project_hit_rate", 0.0) or 0.0)
    baseline_hit_rate = float(baseline_metrics.get("metal_project_hit_rate", 0.0) or 0.0)
    current_fallbacks = float(metrics.get("cpu_fallback_calls", 0.0) or 0.0)
    baseline_fallbacks = float(baseline_metrics.get("cpu_fallback_calls", 0.0) or 0.0)
    current_first_ms = float(metrics.get("first_decode_token_ms", 0.0) or 0.0)
    baseline_first_ms = float(baseline_metrics.get("first_decode_token_ms", 0.0) or 0.0)

    if mismatch_count > 0 or max_logit_delta > 0.05 or mean_logit_delta > 0.01:
        return GateDecision(
            outcome=AUTORESEARCH_MODULE.RUN_OUTCOME_REJECTED,
            baseline_run_id=str(baseline_run["run_id"]),
            run_backend=str(current_run.get("run_backend", "") or ""),
            run_coverage=str(current_run.get("run_coverage", "") or ""),
            comparisons=[comparison],
            regressions=[comparison],
        )

    improved = current_value >= baseline_value * 1.01
    within_band = current_value >= baseline_value * 0.97
    hit_rate_improved = current_hit_rate >= baseline_hit_rate + 10.0
    fallback_improved = baseline_fallbacks > 0 and current_fallbacks <= baseline_fallbacks * 0.9
    first_token_ok = baseline_first_ms == 0 or current_first_ms <= baseline_first_ms * 1.05

    accepted = improved or (within_band and (hit_rate_improved or fallback_improved) and first_token_ok)
    return GateDecision(
        outcome=AUTORESEARCH_MODULE.RUN_OUTCOME_ACCEPTED if accepted else AUTORESEARCH_MODULE.RUN_OUTCOME_REJECTED,
        baseline_run_id=str(baseline_run["run_id"]),
        run_backend=str(current_run.get("run_backend", "") or ""),
        run_coverage=str(current_run.get("run_coverage", "") or ""),
        comparisons=[comparison],
        regressions=[] if accepted else [comparison],
    )


def find_baseline_run(
    runs: list[dict[str, Any]],
    current_run_id: str,
    profile: str,
    run_backend: str,
    run_coverage: str,
    git_branch: str,
) -> dict[str, Any] | None:
    for run in reversed(runs):
        if run["run_id"] == current_run_id:
            continue
        if str(run.get("git_branch", "") or "") != git_branch:
            continue
        if str(run.get("run_outcome", "") or "").strip() == AUTORESEARCH_MODULE.RUN_OUTCOME_REJECTED:
            continue
        suites = run.get("suites", {})
        if not suites:
            continue
        reference_row = next(iter(suites.values()))
        if str(reference_row.get("profile", "") or "") != profile:
            continue
        if str(run.get("run_backend", "") or "") != run_backend:
            continue
        if str(run.get("run_coverage", "") or "") != run_coverage:
            continue
        return run
    return None


def annotate_run(
    *,
    results_csv: Path,
    summary_md: Path,
    plot_svg: Path,
    run_artifact: Path,
    trace_path: Path,
    args: argparse.Namespace,
    branch: str,
    code_commit: str,
    run_id: str,
    decision: GateDecision,
) -> None:
    regressions = [comparison_to_dict(check) for check in decision.regressions]
    comparisons = [comparison_to_dict(check) for check in decision.comparisons]
    trace = {
        "trace_version": 1,
        "round_id": args.round_id,
        "run_id": run_id,
        "profile": args.profile,
        "notes": args.notes,
        "branch": branch,
        "code_commit": code_commit,
        "decision": decision.outcome,
        "run_backend": decision.run_backend,
        "run_coverage": decision.run_coverage,
        "baseline_run_id": decision.baseline_run_id,
        "max_regression_pct": args.max_regression_pct,
        "gate_skipped": args.skip_regression_gate,
        "comparisons": comparisons,
        "regressions": regressions,
    }

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")

    updated = AUTORESEARCH_MODULE.update_run_rows(
        results_csv,
        run_id,
        {
            "run_outcome": decision.outcome,
            "run_backend": decision.run_backend,
            "run_coverage": decision.run_coverage,
            "gate_baseline_run_id": decision.baseline_run_id,
            "gate_max_regression_pct": "" if args.skip_regression_gate else args.max_regression_pct,
            "gate_regressions_json": regressions,
            "decision_trace": trace_path,
        },
    )
    if updated == 0:
        raise SystemExit(f"unable to annotate run `{run_id}` in {results_csv}")

    AUTORESEARCH_MODULE.update_run_artifact(
        run_artifact,
        {
            "run_outcome": decision.outcome,
            "run_backend": decision.run_backend,
            "run_coverage": decision.run_coverage,
            "decision_trace": str(trace_path),
            "gate": {
                "baseline_run_id": decision.baseline_run_id,
                "max_regression_pct": args.max_regression_pct,
                "gate_skipped": args.skip_regression_gate,
                "comparisons": comparisons,
                "regressions": regressions,
            },
        },
    )
    AUTORESEARCH_MODULE.rebuild_reports(results_csv, summary_md, plot_svg)


def comparison_to_dict(check: RegressionCheck) -> dict[str, Any]:
    return {
        "suite": check.suite,
        "suite_title": check.suite_title,
        "baseline_run_id": check.baseline_run_id,
        "current_value": check.current_value,
        "baseline_value": check.baseline_value,
        "delta_pct": check.delta_pct,
    }


def reject_message(
    *,
    round_id: str,
    run_id: str,
    max_regression_pct: float,
    regressions: list[RegressionCheck],
) -> str:
    lines = [
        f"round `{round_id}` rejected for run `{run_id}`; regressions exceeded {max_regression_pct:.2f}%",
    ]
    for check in regressions:
        lines.append(
            "{suite}: {current:.5f} vs {baseline:.5f} ({delta:+.2f}%)".format(
                suite=check.suite,
                current=check.current_value,
                baseline=check.baseline_value,
                delta=check.delta_pct,
            )
        )
    return "\n".join(lines)


def git_status() -> list[StatusEntry]:
    try:
        lines = subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"git status --porcelain=v1 failed with exit code {exc.returncode}") from exc
    entries: list[StatusEntry] = []
    for line in lines:
        if not line:
            continue
        code = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        entries.append(StatusEntry(code=code, path=path))
    return entries


def default_code_message(round_id: str, notes: str) -> str:
    return f"{round_id}: {notes}"


def default_results_message(round_id: str, notes: str, short_code_commit: str) -> str:
    return f"{round_id} results: {notes} [code {short_code_commit}]"


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"git {' '.join(args)} failed with exit code {exc.returncode}") from exc


def run_checked(command: list[str], step_name: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{step_name} failed with exit code {completed.returncode}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


if __name__ == "__main__":
    main()
