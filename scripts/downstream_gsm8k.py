"""Plain vs DFlash vs DTree on gsm8k — sanity-check correctness via
downstream task accuracy. No bit-equality assumptions."""
from __future__ import annotations

import argparse
import re
from argparse import Namespace

from datasets import load_dataset
from tqdm import tqdm

from dtree_mlx import DFlashGenerator
from dtree_mlx.benchmark_cli import run_one_prompt

BOX_RE = re.compile(r"\\boxed\{([^{}]*)\}")
GT_RE = re.compile(r"####\s*([\-0-9,\.]+)")


def extract_boxed(text: str) -> str | None:
    matches = BOX_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip().replace(",", "").rstrip(".")


def extract_gt(answer_text: str) -> str | None:
    m = GT_RE.search(answer_text)
    return m.group(1).strip().replace(",", "") if m else None


def numeric_eq(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--speculative-tokens", type=int, default=16)
    ap.add_argument("--tree-budget", type=int, default=24)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = ds.select(range(args.num_prompts))

    runner = DFlashGenerator()
    tokenizer = runner.target.tokenizer
    model = runner.target.model

    plain_ns = Namespace(
        max_new_tokens=args.max_new_tokens, temperature=0.0,
        top_p=1.0, top_k=0, min_p=0.0, min_tokens_to_keep=1,
    )

    plain_correct = 0
    dflash_correct = 0
    dtree_correct = 0

    pbar = tqdm(total=len(rows), desc="gsm8k", ncols=100)
    for i, row in enumerate(rows):
        question = row["question"]
        gt = extract_gt(row["answer"])
        user_prompt = (
            f"{question}\nPlease reason step by step, and put your final "
            "answer within \\boxed{}."
        )
        prompt_tokens = runner.encode_prompt(user_prompt)

        plain_res = run_one_prompt(model, tokenizer, prompt_tokens.tolist(), plain_ns)
        dflash_res = runner.generate_from_tokens(
            prompt_tokens, max_new_tokens=args.max_new_tokens, temperature=0.0,
            decode_mode="dflash", speculative_tokens=args.speculative_tokens,
        )
        dtree_res = runner.generate_from_tokens(
            prompt_tokens, max_new_tokens=args.max_new_tokens, temperature=0.0,
            decode_mode="dtree", speculative_tokens=args.speculative_tokens,
            tree_budget=args.tree_budget,
        )

        p_ans = extract_boxed(plain_res.output_text)
        d_ans = extract_boxed(dflash_res.text)
        t_ans = extract_boxed(dtree_res.text)

        plain_correct += int(numeric_eq(p_ans, gt))
        dflash_correct += int(numeric_eq(d_ans, gt))
        dtree_correct += int(numeric_eq(t_ans, gt))

        pbar.update(1)
        pbar.set_postfix(
            P=f"{plain_correct}/{i+1}",
            F=f"{dflash_correct}/{i+1}",
            T=f"{dtree_correct}/{i+1}",
        )
    pbar.close()

    n = len(rows)
    print("=" * 70)
    print(f"N = {n}   max_new_tokens={args.max_new_tokens}  "
          f"speculative={args.speculative_tokens}  tree_budget={args.tree_budget}")
    print(f"plain  accuracy: {plain_correct}/{n} = {plain_correct/n:.1%}")
    print(f"dflash accuracy: {dflash_correct}/{n} = {dflash_correct/n:.1%}")
    print(f"dtree  accuracy: {dtree_correct}/{n} = {dtree_correct/n:.1%}")


if __name__ == "__main__":
    main()
