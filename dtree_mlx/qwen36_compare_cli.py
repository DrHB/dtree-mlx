#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import mlx.core as mx
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    disable_progress_bars,
)
from mlx_lm import load as mlx_lm_load

from .adapters import load_target_model
from .api import DFlashGenerator
from .ar_spec import (
    ARSpecResult,
    dtree_generate_ar,
    linear_spec_generate_ar,
    vanilla_generate_target,
)
from .benchmark_cli import load_prompts, split_warmup_and_benchmark_prompts
from .history import DEFAULT_HISTORY_PATH, append_rows, prompt_sha256, run_metadata


DEFAULT_TARGET_MODEL = "mlx-community/Qwen3.6-27B-4bit"
DEFAULT_AR_DRAFT_MODEL = "mlx-community/Qwen3-1.7B-4bit"
DEFAULT_DFLASH_DRAFT_MODEL = "z-lab/Qwen3.6-27B-DFlash"
DEFAULT_METHODS = [
    "vanilla_qwen36",
    "linear_spec_qwen17",
    "dtree_qwen17",
    "dflash_zlab",
    "dtree_zlab",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen3.6-27B vanilla, Qwen3-1.7B speculative decoding, "
            "and real z-lab DFlash/DTree rows on the same prompts."
        )
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--ar-draft-model", default=DEFAULT_AR_DRAFT_MODEL)
    parser.add_argument("--dflash-draft-model", default=DEFAULT_DFLASH_DRAFT_MODEL)
    parser.add_argument(
        "--require-dflash-draft",
        action="store_true",
        default=True,
        help="Require access to --dflash-draft-model before any benchmark runs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
        help="Subset of rows to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"],
        default="gsm8k",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--warmup-prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--draft-max", type=int, default=12)
    parser.add_argument("--draft-min", type=int, default=3)
    parser.add_argument("--draft-p-min", type=float, default=0.6)
    parser.add_argument("--tree-budget", type=int, default=24)
    parser.add_argument("--tree-branch-topk", type=int, default=4)
    parser.add_argument(
        "--verify-mode",
        choices=["parallel-replay", "parallel-greedy-argmax"],
        default="parallel-greedy-argmax",
        help="Verifier mode for z-lab DFlash/DTree rows.",
    )
    parser.add_argument(
        "--draft-attention-mask",
        choices=["auto", "none", "causal"],
        default="auto",
        help="Draft attention mask for the z-lab DFlash drafter.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Append aggregate rows to this CSV file.",
    )
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--experiment-tag", type=str, default="")
    return parser.parse_args()


def resolve_or_check_model(path_or_repo: str, *, role: str) -> str:
    path = Path(path_or_repo)
    if path.exists():
        config = path / "config.json"
        if not config.exists():
            raise SystemExit(f"{role} local path is missing config.json: {path}")
        return str(path)

    try:
        hf_hub_download(path_or_repo, "config.json")
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as exc:
        reason = str(exc).splitlines()[0]
        raise SystemExit(
            f"Cannot access {role} model {path_or_repo!r}: {reason}\n"
            "Run `huggingface-cli login`, set `HF_TOKEN`, or pass a local model path."
        ) from exc
    return path_or_repo


def preflight_access(args: argparse.Namespace) -> None:
    resolve_or_check_model(args.target_model, role="target")
    if any(method in args.methods for method in ("linear_spec_qwen17", "dtree_qwen17")):
        resolve_or_check_model(args.ar_draft_model, role="AR draft")
    if any(method in args.methods for method in ("dflash_zlab", "dtree_zlab")):
        if not args.require_dflash_draft:
            raise SystemExit("The Qwen3.6 z-lab DFlash row is hard-required for this benchmark.")
        resolve_or_check_model(args.dflash_draft_model, role="z-lab DFlash draft")


def aggregate_metrics(method: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError(f"No results for method={method!r}.")

    total_prompt_tokens = sum(int(result["num_input_tokens"]) for result in results)
    total_output_tokens = sum(int(result["num_output_tokens"]) for result in results)
    total_prefill_time = sum(float(result["prefill_time_s"]) for result in results)
    total_decode_time = sum(float(result["decode_time_s"]) for result in results)
    total_time = sum(float(result["total_time_s"]) for result in results)
    acceptance = [
        float(result.get("avg_acceptance_length", 0.0))
        for result in results
        if result.get("avg_acceptance_length") is not None
    ]

    aggregate: dict[str, Any] = {
        "method": method,
        "prompt_count": len(results),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "aggregate_prompt_tps": total_prompt_tokens / max(total_prefill_time, 1e-9),
        "aggregate_generation_tps": total_output_tokens / max(total_decode_time, 1e-9),
        "end_to_end_tps": total_output_tokens / max(total_time, 1e-9),
        "mean_generation_tps": mean(float(result["generation_tps"]) for result in results),
        "mean_acceptance_length": mean(acceptance) if acceptance else 0.0,
        "peak_memory_gb_max": max(float(result["peak_memory_gb"]) for result in results),
        "bridge_rejects": sum(int(result.get("bridge_rejects", 0)) for result in results),
        "draft_expansions": sum(int(result.get("draft_expansions", 0)) for result in results),
        "proposed_target_tokens": sum(int(result.get("proposed_target_tokens", 0)) for result in results),
    }
    verified_values = [
        float(result["avg_verified_tree_nodes"])
        for result in results
        if "avg_verified_tree_nodes" in result
    ]
    if verified_values:
        aggregate["mean_verified_tree_nodes"] = mean(verified_values)
    built_values = [
        float(result["avg_built_tree_nodes"])
        for result in results
        if "avg_built_tree_nodes" in result
    ]
    if built_values:
        aggregate["mean_built_tree_nodes"] = mean(built_values)
    return aggregate


def run_ar_methods(
    args: argparse.Namespace,
    warmup_prompts: list[str],
    benchmark_prompts: list[str],
    log,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    methods = [method for method in args.methods if method in {"vanilla_qwen36", "linear_spec_qwen17", "dtree_qwen17"}]
    if not methods:
        return {}, {}

    log(f"[load target] {args.target_model}")
    target = load_target_model(args.target_model)
    layer_ids = [0]

    draft_model = None
    draft_tokenizer = None
    if any(method in methods for method in ("linear_spec_qwen17", "dtree_qwen17")):
        log(f"[load AR draft] {args.ar_draft_model}")
        draft_model, draft_tokenizer = mlx_lm_load(args.ar_draft_model)

    results: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    outputs: dict[str, list[str]] = {method: [] for method in methods}

    def run_one(method: str, prompt: str, reset_peak_memory: bool) -> ARSpecResult:
        if method == "vanilla_qwen36":
            return vanilla_generate_target(
                target,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                layer_ids=layer_ids,
                reset_peak_memory=reset_peak_memory,
            )
        if method == "linear_spec_qwen17":
            if draft_model is None or draft_tokenizer is None:
                raise RuntimeError("AR draft model was not loaded.")
            return linear_spec_generate_ar(
                target,
                draft_model,
                draft_tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                layer_ids=layer_ids,
                draft_max=args.draft_max,
                draft_min=args.draft_min,
                draft_p_min=args.draft_p_min,
                reset_peak_memory=reset_peak_memory,
            )
        if method == "dtree_qwen17":
            if draft_model is None or draft_tokenizer is None:
                raise RuntimeError("AR draft model was not loaded.")
            return dtree_generate_ar(
                target,
                draft_model,
                draft_tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                layer_ids=layer_ids,
                draft_max=args.draft_max,
                draft_min=args.draft_min,
                draft_p_min=args.draft_p_min,
                tree_budget=args.tree_budget,
                branch_topk=args.tree_branch_topk,
                reset_peak_memory=reset_peak_memory,
            )
        raise ValueError(f"Unknown AR method: {method}")

    for index, prompt in enumerate(warmup_prompts, start=1):
        for method in methods:
            result = run_one(method, prompt, reset_peak_memory=False)
            log(
                f"[warmup {method} {index}/{len(warmup_prompts)}] "
                f"gen_tps={result.metrics['generation_tps']:.2f} "
                f"accept={result.metrics.get('avg_acceptance_length', 0.0):.2f}"
            )
            mx.clear_cache()

    for index, prompt in enumerate(benchmark_prompts, start=1):
        for method in methods:
            result = run_one(method, prompt, reset_peak_memory=True)
            results[method].append(result.metrics)
            outputs[method].append(result.text)
            log(
                f"[run {method} {index}/{len(benchmark_prompts)}] "
                f"prompt={result.metrics['num_input_tokens']} "
                f"out={result.metrics['num_output_tokens']} "
                f"gen_tps={result.metrics['generation_tps']:.2f} "
                f"accept={result.metrics.get('avg_acceptance_length', 0.0):.2f} "
                f"bridge_rejects={result.metrics.get('bridge_rejects', 0)}"
            )
            if args.print_output:
                log(result.text)
            mx.clear_cache()

    del target
    del draft_model
    del draft_tokenizer
    gc.collect()
    mx.clear_cache()
    return results, outputs


def run_zlab_methods(
    args: argparse.Namespace,
    warmup_prompts: list[str],
    benchmark_prompts: list[str],
    log,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    methods = [method for method in args.methods if method in {"dflash_zlab", "dtree_zlab"}]
    if not methods:
        return {}, {}

    log(f"[load z-lab target] {args.target_model}")
    log(f"[load z-lab DFlash draft] {args.dflash_draft_model}")
    runner = DFlashGenerator(
        target_model=args.target_model,
        draft_model=args.dflash_draft_model,
        draft_attention_mask=args.draft_attention_mask,
        seed=args.seed,
    )
    if "dtree_zlab" in methods and not runner.target.supports_tree_verification():
        raise SystemExit(
            f"dtree_zlab is not implemented for target family={runner.target.adapter.family!r}."
        )

    results: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    outputs: dict[str, list[str]] = {method: [] for method in methods}

    mode_for_method = {
        "dflash_zlab": "dflash",
        "dtree_zlab": "dtree",
    }

    for index, prompt in enumerate(warmup_prompts, start=1):
        for method in methods:
            result = runner.generate(
                prompt_text=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                speculative_tokens=args.draft_max,
                decode_mode=mode_for_method[method],
                tree_budget=args.tree_budget,
                verify_mode=args.verify_mode,
                reset_peak_memory=False,
            )
            log(
                f"[warmup {method} {index}/{len(warmup_prompts)}] "
                f"gen_tps={result.metrics['generation_tps']:.2f} "
                f"accept={result.metrics.get('avg_acceptance_length', 0.0):.2f}"
            )
            mx.clear_cache()

    for index, prompt in enumerate(benchmark_prompts, start=1):
        for method in methods:
            result = runner.generate(
                prompt_text=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                speculative_tokens=args.draft_max,
                decode_mode=mode_for_method[method],
                tree_budget=args.tree_budget,
                verify_mode=args.verify_mode,
                reset_peak_memory=True,
            )
            metrics = dict(result.metrics)
            metrics["decode_mode"] = method
            results[method].append(metrics)
            outputs[method].append(result.text)
            log(
                f"[run {method} {index}/{len(benchmark_prompts)}] "
                f"prompt={metrics['num_input_tokens']} out={metrics['num_output_tokens']} "
                f"gen_tps={metrics['generation_tps']:.2f} "
                f"accept={metrics.get('avg_acceptance_length', 0.0):.2f}"
            )
            if args.print_output:
                log(result.text)
            mx.clear_cache()

    del runner
    gc.collect()
    mx.clear_cache()
    return results, outputs


def build_payload(
    args: argparse.Namespace,
    method_results: dict[str, list[dict[str, Any]]],
    benchmark_prompts: list[str],
) -> dict[str, Any]:
    aggregates = {
        method: aggregate_metrics(method, results)
        for method, results in method_results.items()
        if results
    }
    return {
        "config": {
            "target_model": args.target_model,
            "ar_draft_model": args.ar_draft_model,
            "dflash_draft_model": args.dflash_draft_model,
            "dataset": args.dataset,
            "num_prompts": args.num_prompts,
            "warmup_prompts": args.warmup_prompts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "draft_max": args.draft_max,
            "draft_min": args.draft_min,
            "draft_p_min": args.draft_p_min,
            "tree_budget": args.tree_budget,
            "methods": args.methods,
        },
        "prompt_sha256": [prompt_sha256(prompt) for prompt in benchmark_prompts],
        "aggregate": aggregates,
        "per_prompt": method_results,
    }


def print_table(aggregates: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 96)
    print(
        f"{'method':<22} {'gen TPS':>10} {'e2e TPS':>10} {'accept':>8} "
        f"{'peak GB':>8} {'verified':>9} {'bridge':>8} {'draft exp':>10}"
    )
    print("-" * 96)
    for method in DEFAULT_METHODS:
        metrics = aggregates.get(method)
        if not metrics:
            continue
        print(
            f"{method:<22} "
            f"{metrics['aggregate_generation_tps']:>10.2f} "
            f"{metrics['end_to_end_tps']:>10.2f} "
            f"{metrics['mean_acceptance_length']:>8.2f} "
            f"{metrics['peak_memory_gb_max']:>8.2f} "
            f"{metrics.get('mean_verified_tree_nodes', 0.0):>9.2f} "
            f"{metrics.get('bridge_rejects', 0):>8} "
            f"{metrics.get('draft_expansions', 0):>10}"
        )
    print("=" * 96)


def maybe_write_history(
    args: argparse.Namespace,
    payload: dict[str, Any],
    benchmark_prompts: list[str],
    log,
) -> None:
    history_path = args.history_file or DEFAULT_HISTORY_PATH
    record_history = (args.history or args.history_file is not None) and not args.no_history
    if not record_history:
        return

    meta = run_metadata("dtree-mlx-qwen36-compare", experiment_tag=args.experiment_tag)
    rows = []
    prompt_hash = prompt_sha256(benchmark_prompts[0]) if len(benchmark_prompts) == 1 else ""
    for method, aggregate in payload["aggregate"].items():
        rows.append(
            {
                **meta,
                "record_type": "aggregate",
                "method": method,
                "target_model": args.target_model,
                "ar_draft_model": args.ar_draft_model,
                "dflash_draft_model": args.dflash_draft_model,
                "dataset": args.dataset,
                "prompt_source": (
                    "prompt"
                    if args.prompt is not None
                    else "prompt_file"
                    if args.prompt_file is not None
                    else "dataset"
                ),
                "prompt_file": args.prompt_file,
                "prompt_sha256": prompt_hash,
                "num_prompts_requested": args.num_prompts,
                "warmup_prompts": args.warmup_prompts,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "draft_max": args.draft_max,
                "draft_min": args.draft_min,
                "draft_p_min": args.draft_p_min,
                "tree_budget": args.tree_budget,
                **aggregate,
            }
        )
    append_rows(history_path, rows)
    log(f"[history] appended {len(rows)} row(s) to {history_path}")


def main() -> None:
    args = parse_args()
    if args.prompt is not None and args.prompt_file is not None:
        raise SystemExit("Use either --prompt or --prompt-file, not both.")
    if args.temperature >= 1e-5 and any(
        method in args.methods for method in ("linear_spec_qwen17", "dflash_zlab", "dtree_zlab")
    ):
        raise SystemExit("This Qwen3.6 comparison currently requires --temperature 0.")

    if args.json:
        disable_progress_bars()
    log = (lambda *items: None) if args.json else print
    random.seed(args.seed)

    preflight_access(args)
    prompts = load_prompts(args)
    warmup_prompts, benchmark_prompts = split_warmup_and_benchmark_prompts(
        prompts,
        args.warmup_prompts,
    )

    method_results: dict[str, list[dict[str, Any]]] = {}
    method_outputs: dict[str, list[str]] = {}

    ar_results, ar_outputs = run_ar_methods(args, warmup_prompts, benchmark_prompts, log)
    method_results.update(ar_results)
    method_outputs.update(ar_outputs)

    zlab_results, zlab_outputs = run_zlab_methods(args, warmup_prompts, benchmark_prompts, log)
    method_results.update(zlab_results)
    method_outputs.update(zlab_outputs)

    payload = build_payload(args, method_results, benchmark_prompts)
    payload["outputs"] = method_outputs if args.print_output else {}
    maybe_write_history(args, payload, benchmark_prompts, log)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print_table(payload["aggregate"])
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
