#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from huggingface_hub.utils import disable_progress_bars

from .api import DEFAULT_DRAFT_MODEL, DEFAULT_TARGET_MODEL, DFlashGenerator
from .benchmark_cli import load_prompts, split_warmup_and_benchmark_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash vs DTree on the same prompts."
    )
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help="MLX target model repo or local path.",
    )
    parser.add_argument(
        "--draft-model",
        default=DEFAULT_DRAFT_MODEL,
        help="Hugging Face repo or local path for the DFlash draft weights.",
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"],
        default="gsm8k",
        help="Dataset to benchmark when no explicit prompt is provided.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Run a single benchmark prompt.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Read a single benchmark prompt from a file.")
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to benchmark.")
    parser.add_argument("--warmup-prompts", type=int, default=0, help="Warmup prompts per mode.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle prompts before warmup/benchmark.")
    parser.add_argument("--speculative-tokens", type=int, default=None)
    parser.add_argument("--tree-budget", type=int, default=None)
    parser.add_argument(
        "--verify-mode",
        choices=[
            "stream",
            "chunked",
            "parallel-replay",
            "parallel-lazy-logits",
            "parallel-greedy-argmax",
        ],
        default="parallel-replay",
        help="Verifier strategy to use for the DFlash baseline.",
    )
    parser.add_argument("--verify-chunk-size", type=int, default=4)
    parser.add_argument("--target-quant-bits", type=int, default=None)
    parser.add_argument("--target-quant-group-size", type=int, default=64)
    parser.add_argument("--draft-quant-bits", type=int, default=None)
    parser.add_argument("--draft-quant-group-size", type=int, default=64)
    parser.add_argument(
        "--draft-attention-mask",
        choices=["auto", "none", "causal"],
        default="auto",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--print-output", action="store_true", help="Print decoded text for measured prompts.")
    return parser.parse_args()


def summarize_mode(results: list[dict[str, float]]) -> dict[str, float]:
    if not results:
        raise ValueError("No benchmark results to summarize.")
    total_prompt_tokens = sum(result["num_input_tokens"] for result in results)
    total_output_tokens = sum(result["num_output_tokens"] for result in results)
    total_prefill_time = sum(result["prefill_time_s"] for result in results)
    total_decode_time = sum(result["decode_time_s"] for result in results)
    total_time = sum(result["total_time_s"] for result in results)
    return {
        "prompt_count": len(results),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "aggregate_prompt_tps": total_prompt_tokens / max(total_prefill_time, 1e-9),
        "aggregate_generation_tps": total_output_tokens / max(total_decode_time, 1e-9),
        "end_to_end_tps": total_output_tokens / max(total_time, 1e-9),
        "mean_generation_tps": mean(result["generation_tps"] for result in results),
        "mean_acceptance_length": mean(result["avg_acceptance_length"] for result in results),
        "peak_memory_gb_max": max(result["peak_memory_gb"] for result in results),
    }


def benchmark_mode_order(prompt_index: int) -> tuple[str, str]:
    if prompt_index % 2 == 1:
        return ("dflash", "dtree")
    return ("dtree", "dflash")


def main() -> None:
    args = parse_args()
    if args.prompt is not None and args.prompt_file is not None:
        raise SystemExit("Use either --prompt or --prompt-file, not both.")

    random.seed(args.seed)
    if args.json:
        disable_progress_bars()
    log = (lambda *items: None) if args.json else print

    prompts = load_prompts(args)
    warmup_prompts, benchmark_prompts = split_warmup_and_benchmark_prompts(
        prompts,
        args.warmup_prompts,
    )

    runner = DFlashGenerator(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_attention_mask=args.draft_attention_mask,
        target_quant_bits=args.target_quant_bits,
        target_quant_group_size=args.target_quant_group_size,
        draft_quant_bits=args.draft_quant_bits,
        draft_quant_group_size=args.draft_quant_group_size,
        seed=args.seed,
    )

    mode_results: dict[str, list[dict[str, float]]] = {"dflash": [], "dtree": []}
    for warmup_idx, prompt in enumerate(warmup_prompts, start=1):
        mode_order = benchmark_mode_order(warmup_idx)
        log(f"[warmup prompt {warmup_idx}/{len(warmup_prompts)}] order={','.join(mode_order)}")
        for decode_mode in mode_order:
            warm_result = runner.generate(
                prompt_text=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                speculative_tokens=args.speculative_tokens,
                decode_mode=decode_mode,
                tree_budget=args.tree_budget,
                verify_mode=args.verify_mode,
                verify_chunk_size=args.verify_chunk_size,
                reset_peak_memory=False,
            )
            log(
                f"[warmup {decode_mode} {warmup_idx}/{len(warmup_prompts)}] "
                f"gen_tps={warm_result.metrics['generation_tps']:.2f} "
                f"accept={warm_result.metrics['avg_acceptance_length']:.2f}"
            )

    for index, prompt in enumerate(benchmark_prompts, start=1):
        mode_order = benchmark_mode_order(index)
        log(f"[prompt {index}/{len(benchmark_prompts)}] order={','.join(mode_order)}")
        for decode_mode in mode_order:
            result = runner.generate(
                prompt_text=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                speculative_tokens=args.speculative_tokens,
                decode_mode=decode_mode,
                tree_budget=args.tree_budget,
                verify_mode=args.verify_mode,
                verify_chunk_size=args.verify_chunk_size,
            )
            metrics = result.metrics
            mode_results[decode_mode].append(metrics)
            log(
                f"[run {decode_mode} {index}/{len(benchmark_prompts)}] "
                f"prompt={metrics['num_input_tokens']} out={metrics['num_output_tokens']} "
                f"gen_tps={metrics['generation_tps']:.2f} "
                f"accept={metrics['avg_acceptance_length']:.2f}"
            )
            if decode_mode == "dtree":
                log(f"  tree_budget={metrics['tree_budget']} tree_nodes={metrics['avg_verified_tree_nodes']:.2f}")
            if args.print_output:
                log(result.text)

    summary = {
        mode: summarize_mode(results) for mode, results in mode_results.items()
    }
    summary["speedup"] = {
        "generation_tps": summary["dtree"]["aggregate_generation_tps"]
        / max(summary["dflash"]["aggregate_generation_tps"], 1e-9),
        "end_to_end_tps": summary["dtree"]["end_to_end_tps"]
        / max(summary["dflash"]["end_to_end_tps"], 1e-9),
        "acceptance_length": summary["dtree"]["mean_acceptance_length"]
        / max(summary["dflash"]["mean_acceptance_length"], 1e-9),
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print("\n" + "=" * 60)
    for mode in ("dflash", "dtree"):
        metrics = summary[mode]
        print(f"{mode.upper()}:")
        print(f"  prompts:                {metrics['prompt_count']}")
        print(f"  generation TPS:         {metrics['aggregate_generation_tps']:.2f}")
        print(f"  end-to-end TPS:         {metrics['end_to_end_tps']:.2f}")
        print(f"  mean acceptance length: {metrics['mean_acceptance_length']:.2f}")
        print(f"  peak memory max:        {metrics['peak_memory_gb_max']:.2f} GB")
    print("Speedup:")
    print(f"  generation TPS:         {summary['speedup']['generation_tps']:.2f}x")
    print(f"  end-to-end TPS:         {summary['speedup']['end_to_end_tps']:.2f}x")
    print(f"  acceptance length:      {summary['speedup']['acceptance_length']:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
