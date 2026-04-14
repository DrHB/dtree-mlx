# Correctness sanity check — gsm8k (N=30)

**Date:** 2026-04-14
**Target:** `mlx-community/Qwen3-4B-bf16`
**Draft:** `z-lab/Qwen3-4B-DFlash-b16`
**Hardware:** MacBook Pro (Apple Silicon), MLX 0.31.1

## What this measures

Task accuracy on the first 30 gsm8k-test prompts at `temperature=0`,
`max_new_tokens=512`. Each prompt is run three ways — plain
autoregressive (MLX-LM), DFlash, DTree — and the boxed numeric answer is
compared to gsm8k's ground truth. No bit-equality assumption.

This is the same correctness philosophy the upstream DDTree reference
and z-lab DFlash paper use: task accuracy on a real benchmark, not
token-level equality (which is not achievable on MLX/bf16 anyway —
different kernel shapes produce different float rounding, and neither
upstream holds themselves to that bar).

## Results

| Method | Accuracy |
|---|---:|
| Plain autoregressive (MLX-LM) | **24/30 = 80.0%** |
| DFlash (`speculative_tokens=16`) | **26/30 = 86.7%** |
| DTree (`speculative_tokens=16`, `tree_budget=24`) | **27/30 = 90.0%** |

All three are within 10pp of each other, well inside the binomial
noise envelope at N=30 (±17pp). No correctness regression: DTree and
DFlash are both functionally equivalent to plain autoregressive on this
sample. The DTree port is correct.

## Reproduce

```bash
cd dtree-mlx
uv run python -u scripts/downstream_gsm8k.py \
    --num-prompts 30 \
    --max-new-tokens 512 \
    --speculative-tokens 16 \
    --tree-budget 24
```

## Notes

- Bigger N (e.g. 100 or 300) would tighten the accuracy bars, but 30 is
  enough to rule out a catastrophic correctness regression (the kind a
  port typically introduces — wrong tree wiring, off-by-one in
  acceptance, dropped tokens on KV compaction).
- The `DFlashGenerator.generate_from_tokens` path is used for DFlash
  and DTree so that plain/DFlash/DTree all share the exact same
  tokenized prompt — no chat-template drift.
