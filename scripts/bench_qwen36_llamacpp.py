#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from huggingface_hub import hf_hub_download

from dtree_mlx.benchmark_cli import load_and_process_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_REPO = "bartowski/Qwen_Qwen3.6-27B-GGUF"
DEFAULT_TARGET_FILE = "Qwen_Qwen3.6-27B-Q4_K_M.gguf"
DEFAULT_DRAFT_REPO = "ggml-org/Qwen3-1.7B-GGUF"
DEFAULT_DRAFT_FILE = "Qwen3-1.7B-Q4_K_M.gguf"
DEFAULT_TARGET_PATH = REPO_ROOT / "models" / "qwen3.6-27b-q4km" / DEFAULT_TARGET_FILE
DEFAULT_DRAFT_PATH = REPO_ROOT / "models" / "qwen3-1.7b-q4km" / DEFAULT_DRAFT_FILE
DEFAULT_RESULT_DIR = REPO_ROOT / "benchmarks"
DEFAULT_PROMPT = (
    "Explain quicksort in concise but complete terms. Include its average and "
    "worst-case time complexity."
)
DATASET_FORMATTERS = {
    "gsm8k": lambda row: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(
        **row
    ),
    "humaneval": lambda row: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(
        **row
    ),
    "math500": lambda row: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(
        **row
    ),
    "mbpp": lambda row: row["prompt"],
}


@dataclass
class RunMetrics:
    mode: str
    command: list[str]
    generated_tokens: int | None
    prompt_tokens: int | None
    prompt_tps: float | None
    generation_tps: float | None
    total_tps: float | None
    total_time_s: float
    peak_memory_gb: float | None
    accepted_tokens: int | None
    drafted_tokens: int | None
    acceptance_rate: float | None
    raw_tail: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3.6-27B Q4_K_M with llama.cpp on Apple Silicon."
    )
    parser.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--target-file", default=DEFAULT_TARGET_FILE)
    parser.add_argument("--draft-repo", default=DEFAULT_DRAFT_REPO)
    parser.add_argument("--draft-file", default=DEFAULT_DRAFT_FILE)
    parser.add_argument("--target-path", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--draft-path", type=Path, default=DEFAULT_DRAFT_PATH)
    parser.add_argument("--download", action="store_true", help="Download missing GGUFs from Hugging Face.")
    parser.add_argument(
        "--llama-cli",
        default=None,
        help="Deprecated executable override. Prefer --llama-completion and --llama-speculative.",
    )
    parser.add_argument("--llama-completion", default=shutil.which("llama-completion") or "llama-completion")
    parser.add_argument("--llama-speculative", default=shutil.which("llama-speculative") or "llama-speculative")
    parser.add_argument("--dataset", choices=sorted(DATASET_FORMATTERS), default="gsm8k")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--warmup-prompts", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--ctx-size-draft", type=int, default=4096)
    parser.add_argument("--n-predict", "--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-layers", default="99")
    parser.add_argument("--gpu-layers-draft", default="99")
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--draft-max", type=int, default=12)
    parser.add_argument("--draft-min", type=int, default=3)
    parser.add_argument("--draft-p-min", type=float, default=0.6)
    parser.add_argument("--runs", type=int, default=None, help="Deprecated alias for --num-prompts.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Extra warmup repeats in addition to warmup prompts.")
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="0 lets llama.cpp choose.")
    parser.add_argument(
        "--spec-replace",
        nargs=2,
        action="append",
        default=[],
        metavar=("TARGET_TOKEN", "DRAFT_TOKEN"),
        help=(
            "Pass a llama.cpp speculative tokenizer bridge replacement. "
            "May be repeated; forwarded as --spec-replace TARGET_TOKEN DRAFT_TOKEN."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--skip-speculative", action="store_true")
    return parser.parse_args()


def resolve_prompt_counts(args: argparse.Namespace) -> None:
    explicit_prompt = args.prompt is not None or args.prompt_file is not None
    if args.num_prompts is None and args.runs is not None:
        args.num_prompts = args.runs
    if args.num_prompts is None:
        args.num_prompts = 1 if explicit_prompt else 8
    if args.warmup_prompts is None:
        args.warmup_prompts = 0 if explicit_prompt else 2


def load_prompt_sweep(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    total_prompts = args.num_prompts + args.warmup_prompts
    if total_prompts <= 0:
        raise SystemExit("--num-prompts plus --warmup-prompts must be positive.")
    if args.prompt is not None and args.prompt_file is not None:
        raise SystemExit("Use either --prompt or --prompt-file, not both.")

    if args.prompt is not None:
        prompts = [args.prompt] * total_prompts
    elif args.prompt_file is not None:
        prompts = [args.prompt_file.read_text()] * total_prompts
    else:
        dataset = load_and_process_dataset(args.dataset)
        formatter = DATASET_FORMATTERS[args.dataset]
        prompts = []
        for idx in range(total_prompts):
            row = dataset[idx % len(dataset)]
            prompts.append(row["turns"][0] if "turns" in row else formatter(row))
    return prompts[: args.warmup_prompts], prompts[args.warmup_prompts :]


def ensure_model(path: Path, repo: str, filename: str, *, download: bool) -> Path:
    if path.exists():
        return path
    if not download:
        raise SystemExit(
            f"Missing model file: {path}\n"
            f"Re-run with --download or place {filename!r} there."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo, filename=filename, local_dir=path.parent)
    return Path(downloaded)


def resolve_llama_bins(args: argparse.Namespace) -> None:
    if args.llama_cli is not None:
        args.llama_completion = args.llama_cli
        args.llama_speculative = args.llama_cli


def base_command(
    args: argparse.Namespace,
    *,
    executable: str,
    n_predict: int,
    prompt: str,
    completion: bool,
) -> list[str]:
    cmd = [
        executable,
        "-m",
        str(args.target_path),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(args.ctx_size),
        "-ngl",
        str(args.gpu_layers),
        "-fa",
        "on",
        "-ctk",
        args.cache_type_k,
        "-ctv",
        args.cache_type_v,
        "--temp",
        "0",
        "--seed",
        str(args.seed),
    ]
    if completion:
        cmd.extend(
            [
                "--no-display-prompt",
                "--simple-io",
                "--conversation",
                "--single-turn",
                "--jinja",
                "--no-warmup",
            ]
        )
    if args.threads > 0:
        cmd.extend(["-t", str(args.threads), "-tb", str(args.threads)])
    return cmd


def command_for_mode(args: argparse.Namespace, mode: str, n_predict: int, prompt: str) -> list[str]:
    if mode == "vanilla":
        return base_command(
            args,
            executable=args.llama_completion,
            n_predict=n_predict,
            prompt=prompt,
            completion=True,
        )

    if mode == "speculative":
        cmd = base_command(
            args,
            executable=args.llama_speculative,
            n_predict=n_predict,
            prompt=prompt,
            completion=False,
        )
        cmd.extend(
            [
                "-md",
                str(args.draft_path),
                "-ngld",
                str(args.gpu_layers_draft),
                "-cd",
                str(args.ctx_size_draft),
                "--draft-max",
                str(args.draft_max),
                "--draft-min",
                str(args.draft_min),
                "--draft-p-min",
                str(args.draft_p_min),
            ]
        )
        for target_token, draft_token in args.spec_replace:
            cmd.extend(["--spec-replace", target_token, draft_token])
        return cmd

    raise ValueError(f"Unknown mode={mode!r}")


def read_tail(path: Path, line_count: int) -> list[str]:
    proc = subprocess.run(
        ["tail", "-n", str(line_count), str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout.splitlines()
    tail: deque[str] = deque(maxlen=line_count)
    with path.open(errors="replace") as handle:
        for line in handle:
            tail.append(line.rstrip("\n"))
    return list(tail)


def run_command(cmd: list[str]) -> tuple[Path, float]:
    start = time.perf_counter()
    wrapped_cmd = ["/usr/bin/time", "-l", *cmd] if sys.platform == "darwin" and Path("/usr/bin/time").exists() else cmd
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", errors="replace", delete=False) as output_file:
        output_path = Path(output_file.name)
        proc = subprocess.run(
            wrapped_cmd,
            text=True,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        tail = "\n".join(read_tail(output_path, 120))
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}:\n"
            f"{' '.join(cmd)}\n\n{tail}"
        )
    return output_path, elapsed


def parse_llama_output(mode: str, cmd: list[str], output_path: Path, elapsed: float) -> RunMetrics:
    generated_tokens = None
    prompt_tokens = None
    prompt_tps = None
    generation_tps = None
    total_tps = None
    total_time_ms = None
    peak_memory_gb = None
    accepted_tokens = None
    drafted_tokens = None
    acceptance_rate = None
    raw_tail: deque[str] = deque(maxlen=80)

    with output_path.open(errors="replace") as handle:
        lines = (line.rstrip("\n") for line in handle)
        for line in lines:
            raw_tail.append(line)
            memory_match = re.search(r"^\s*(\d+)\s+maximum resident set size", line)
            if memory_match:
                maxrss = int(memory_match.group(1))
                if sys.platform == "darwin":
                    peak_memory_gb = maxrss / (1024**3)
                else:
                    peak_memory_gb = maxrss / (1024**2)

            prompt_match = re.search(
                r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens?.*?([\d.]+)\s*tokens per second",
                line,
            )
            if prompt_match:
                prompt_tokens = int(prompt_match.group(1))
                prompt_tps = float(prompt_match.group(2))
            eval_match = re.search(
                r"\beval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*runs?.*?([\d.]+|inf)\s*tokens per second",
                line,
            )
            if eval_match:
                generated_tokens = int(eval_match.group(1))
                generation_tps = None if eval_match.group(2) == "inf" else float(eval_match.group(2))
            total_match = re.search(
                r"total time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens?.*?([\d.]+)\s*tokens per second",
                line,
            )
            if total_match:
                total_tps = float(total_match.group(2))
            total_time_match = re.search(
                r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens?",
                line,
            )
            if total_time_match:
                total_time_ms = float(total_time_match.group(1))

            accepted_match = re.search(
                r"accepted\s*=\s*(\d+).*?drafted\s*=\s*(\d+).*?accept(?:ance)?(?: rate)?\s*=\s*([\d.]+)",
                line,
                re.IGNORECASE,
            )
            if accepted_match:
                accepted_tokens = int(accepted_match.group(1))
                drafted_tokens = int(accepted_match.group(2))
                acceptance_rate = float(accepted_match.group(3))

            draft_match = re.search(
                r"drafted\s+(\d+).*?accepted\s+(\d+)",
                line,
                re.IGNORECASE,
            )
            if draft_match:
                drafted_tokens = int(draft_match.group(1))
                accepted_tokens = int(draft_match.group(2))

    if acceptance_rate is None and accepted_tokens is not None and drafted_tokens:
        acceptance_rate = accepted_tokens / max(drafted_tokens, 1)
    if total_tps is None and generated_tokens is not None and total_time_ms:
        total_tps = generated_tokens / (total_time_ms / 1000.0)

    return RunMetrics(
        mode=mode,
        command=cmd,
        generated_tokens=generated_tokens,
        prompt_tokens=prompt_tokens,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        total_tps=total_tps,
        total_time_s=elapsed,
        peak_memory_gb=peak_memory_gb,
        accepted_tokens=accepted_tokens,
        drafted_tokens=drafted_tokens,
        acceptance_rate=acceptance_rate,
        raw_tail=list(raw_tail),
    )


def summarize(runs: list[RunMetrics]) -> dict[str, Any]:
    def avg(values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        return mean(present) if present else None

    return {
        "runs": len(runs),
        "generation_tps_mean": avg([run.generation_tps for run in runs]),
        "generation_tps_max": max(
            [run.generation_tps for run in runs if run.generation_tps is not None],
            default=None,
        ),
        "prompt_tps_mean": avg([run.prompt_tps for run in runs]),
        "total_tps_mean": avg([run.total_tps for run in runs]),
        "wall_time_s_mean": mean(run.total_time_s for run in runs),
        "peak_memory_gb_max": max(
            [run.peak_memory_gb for run in runs if run.peak_memory_gb is not None],
            default=None,
        ),
        "acceptance_rate_mean": avg([run.acceptance_rate for run in runs]),
        "accepted_tokens_total": sum(run.accepted_tokens or 0 for run in runs),
        "drafted_tokens_total": sum(run.drafted_tokens or 0 for run in runs),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen3.6-27B llama.cpp M2 Benchmark",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(payload["config"], indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| Mode | Mean gen TPS | Peak gen TPS | Mean total TPS | Peak memory GB | Acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in payload["summary"].items():
        acceptance = summary.get("acceptance_rate_mean")
        lines.append(
            "| {mode} | {gen} | {peak} | {total} | {memory} | {accept} |".format(
                mode=mode,
                gen=format_float(summary.get("generation_tps_mean")),
                peak=format_float(summary.get("generation_tps_max")),
                total=format_float(summary.get("total_tps_mean")),
                memory=format_float(summary.get("peak_memory_gb_max")),
                accept="" if acceptance is None else f"{acceptance:.1%}",
            )
        )
    lines.extend(["", "## Commands", ""])
    for mode, runs in payload["runs"].items():
        if not runs:
            continue
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(runs[0]["command"]))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    resolve_llama_bins(args)
    resolve_prompt_counts(args)
    warmup_prompts, benchmark_prompts = load_prompt_sweep(args)
    args.target_path = ensure_model(
        args.target_path,
        args.target_repo,
        args.target_file,
        download=args.download,
    )
    args.draft_path = ensure_model(
        args.draft_path,
        args.draft_repo,
        args.draft_file,
        download=args.download,
    )

    modes = []
    if not args.skip_vanilla:
        modes.append("vanilla")
    if not args.skip_speculative:
        modes.append("speculative")

    for mode in modes:
        for warmup_idx in range(args.warmup_runs):
            prompt = warmup_prompts[warmup_idx % len(warmup_prompts)] if warmup_prompts else DEFAULT_PROMPT
            cmd = command_for_mode(args, mode, args.warmup_tokens, prompt)
            print(f"[warmup {mode} legacy {warmup_idx + 1}/{args.warmup_runs}] {' '.join(cmd)}", flush=True)
            output_path, _ = run_command(cmd)
            output_path.unlink(missing_ok=True)
        for prompt_idx, prompt in enumerate(warmup_prompts):
            cmd = command_for_mode(args, mode, args.warmup_tokens, prompt)
            print(f"[warmup {mode} prompt {prompt_idx + 1}/{len(warmup_prompts)}] {' '.join(cmd)}", flush=True)
            output_path, _ = run_command(cmd)
            output_path.unlink(missing_ok=True)

    run_payload: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for mode in modes:
        for run_idx, prompt in enumerate(benchmark_prompts):
            cmd = command_for_mode(args, mode, args.n_predict, prompt)
            print(f"[run {mode} {run_idx + 1}/{len(benchmark_prompts)}] {' '.join(cmd)}", flush=True)
            output_path, elapsed = run_command(cmd)
            metrics = parse_llama_output(mode, cmd, output_path, elapsed)
            output_path.unlink(missing_ok=True)
            run_payload[mode].append(asdict(metrics))
            print(
                f"[result {mode} {run_idx + 1}/{len(benchmark_prompts)}] "
                f"gen_tps={format_float(metrics.generation_tps)} "
                f"total_tps={format_float(metrics.total_tps)} "
                f"acceptance={'' if metrics.acceptance_rate is None else f'{metrics.acceptance_rate:.1%}'}",
                flush=True,
            )

    payload = {
        "config": {
            "target_repo": args.target_repo,
            "target_file": args.target_file,
            "target_path": str(args.target_path),
            "draft_repo": args.draft_repo,
            "draft_file": args.draft_file,
            "draft_path": str(args.draft_path),
            "llama_cli": args.llama_cli,
            "llama_completion": args.llama_completion,
            "llama_speculative": args.llama_speculative,
            "ctx_size": args.ctx_size,
            "ctx_size_draft": args.ctx_size_draft,
            "n_predict": args.n_predict,
            "gpu_layers": args.gpu_layers,
            "gpu_layers_draft": args.gpu_layers_draft,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "draft_max": args.draft_max,
            "draft_min": args.draft_min,
            "draft_p_min": args.draft_p_min,
            "spec_replace": args.spec_replace,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "warmup_tokens": args.warmup_tokens,
            "seed": args.seed,
            "dataset": args.dataset,
            "num_prompts": args.num_prompts,
            "warmup_prompts": args.warmup_prompts,
            "prompt": args.prompt,
            "prompt_file": None if args.prompt_file is None else str(args.prompt_file),
        },
        "summary": {
            mode: summarize([RunMetrics(**run) for run in runs])
            for mode, runs in run_payload.items()
        },
        "runs": run_payload,
    }

    output_json = args.output_json or DEFAULT_RESULT_DIR / "qwen36_llamacpp_m2.json"
    output_md = args.output_md or DEFAULT_RESULT_DIR / "qwen36_llamacpp_m2.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(output_md, payload)
    print(f"[wrote] {output_json}", flush=True)
    print(f"[wrote] {output_md}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
