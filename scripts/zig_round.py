#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTORESEARCH = REPO_ROOT / "scripts" / "zig_autoresearch.py"


@dataclass(frozen=True)
class StatusEntry:
    code: str
    path: str

    @property
    def is_experiment(self) -> bool:
        return self.path == "experiments" or self.path.startswith("experiments/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Commit a code round, run Zig eval on that exact commit, then commit and push the results.",
    )
    parser.add_argument("--round-id", required=True, help="Round identifier, for example r001.")
    parser.add_argument("--notes", required=True, help="Short note describing the experiment.")
    parser.add_argument("--profile", choices=["full", "decode", "micro", "metal"], default="metal")
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

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
        print(f"dry_run_push_target: {args.remote} {branch}")
        return

    run_checked(autoresearch_cmd, "zig_autoresearch")

    post_eval_status = git_status()
    stray_changes = [entry for entry in post_eval_status if not entry.is_experiment]
    if stray_changes:
        paths = ", ".join(entry.path for entry in stray_changes)
        raise SystemExit(f"eval produced unexpected non-experiment changes: {paths}")

    experiment_paths = [entry.path for entry in post_eval_status if entry.is_experiment]
    if not experiment_paths:
        raise SystemExit("no experiment artifacts were produced")

    run_checked(["git", "add", "-A", "--", "experiments"], "git add experiment artifacts")
    run_checked(["git", "commit", "-m", results_message], "git commit experiment artifacts")
    results_commit = git_output(["rev-parse", "HEAD"])

    run_checked(["git", "push", args.remote, branch], "git push")

    print(f"round_id: {args.round_id}")
    print(f"code_commit: {code_commit}")
    print(f"code_short_commit: {short_code_commit}")
    print(f"results_commit: {results_commit}")
    print(f"pushed_branch: {branch}")


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


def run_checked(command: list[str], step_name: str) -> None:
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


if __name__ == "__main__":
    main()
