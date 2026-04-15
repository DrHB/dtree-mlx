# Optimization Notes

Date: 2026-04-15

This file keeps the short version of what we measured on the local Apple M2 Max.

## Setup

- hardware: Apple M2 Max, 32 GB
- target: `mlx-community/Qwen3-4B-bf16`
- draft: `z-lab/Qwen3-4B-DFlash-b16`
- decoding: `temperature=0`

## Current Baseline

Representative local sweep: 8 gsm8k prompts, 2 warmup prompts, `max_new_tokens=512`.

| Method | Gen TPS | End-to-end TPS | Mean accept |
|---|---:|---:|---:|
| Plain MLX-LM | 36.42 | 34.98 | — |
| DFlash (`parallel-greedy-argmax`) | 51.87 | 50.31 | 5.71 |
| DTree (`tree_budget=24`) | 56.68 | 54.83 | 7.17 |

Best fixed local DTree setting so far:

- `--speculative-tokens 16`
- `--tree-budget 24`

Important constraint:

- the current draft is `b16`
- runtime clamps `--speculative-tokens` above `16`

## What Helped

| Change | Result |
|---|---|
| `--verify-mode parallel-greedy-argmax` for DFlash | Much faster than `parallel-replay` on Qwen3 at `temperature=0` |
| `--tree-budget 24` for DTree | Best fixed tree budget we found locally |
| `--target-quant-bits 4 --target-quant-group-size 64` | Helped DTree locally, but hurt DFlash |

Target quantization summary:

Matched 8-prompt sweep:

| Setting | DFlash E2E TPS | DTree E2E TPS |
|---|---:|---:|
| bf16 target | 50.31 | 54.83 |
| 4-bit target, group size 64 | 47.36 | 56.09 |

Small gsm8k sanity slice (`N=12`):

| Setting | Plain | DFlash | DTree |
|---|---:|---:|---:|
| bf16 target | 9/12 | 10/12 | 11/12 |
| 4-bit target, group size 64 | 11/12 | 12/12 | 11/12 |

Takeaway:

- `4-bit` target quantization is promising for DTree
- it is not a universal default
- keep it opt-in for now

## What Did Not Help

These were tested and did not improve the current 4B setup:

- adaptive tree budget
- hybrid DFlash/DTree routing
- simple depth-penalized tree scoring
- a DTree-side greedy verifier rewrite
- avoiding eager hidden-state concatenation
- larger requested `--speculative-tokens` values above `16` with the current `b16` draft
- a broader MLX verifier fusion attempt

## Bottleneck

DTree is still verifier-bound.

Local profile summary:

- verifier time is about `80%` of DTree decode time
- tree build is about `1-2%`
- bookkeeping is small
- inside verify, tree forward is the dominant share

Why DTree is not much faster today:

- DFlash verifies `16` positions and accepts about `5.9`
- DTree verifies `25` nodes and accepts about `6.5-7.2`

So DTree gets more acceptance, but not enough more acceptance to dominate the extra verifier work.

## Next Ideas

If we keep pushing on the current 4B setup, the remaining worthwhile directions are:

1. Improve tree quality, not just runtime:
   - better tree scoring
   - better acceptance per verified node
2. Improve the exact verifier path:
   - more MLX-side optimization
   - less verifier work per accepted token
3. Improve the draft:
   - better draft quality
   - or a larger-block draft than the current `b16`

## Qwen3.5 Notes

Local Janet prompt check, `temperature=0`, `max_new_tokens=128`, 1 warmup run:

| Model | Method | Gen TPS | End-to-end TPS | Mean accept |
|---|---|---:|---:|---:|
| Qwen3.5-4B | Plain MLX-LM | 37.05 | 33.46 | — |
| Qwen3.5-4B | DFlash | 48.27 | 45.30 | 5.12 |
| Qwen3.5-9B | Plain MLX-LM | 18.97 | 17.40 | — |
| Qwen3.5-9B | DFlash | 20.27 | 19.38 | 4.03 |

Reference fork checked on the same prompt:

| Repo | Model | Gen TPS | End-to-end TPS | Tokens/cycle |
|---|---|---:|---:|---:|
| `bstnxbt/dflash-mlx` | Qwen3.5-4B | 56.58 | 41.96 | 6.10 |
| `bstnxbt/dflash-mlx` | Qwen3.5-9B | 32.18 | 30.01 | 5.57 |

Takeaways:

- On our current code, `qwen3_5` does better with `draft_attention_mask=none` than `causal`.
- The current repo keeps two imported Qwen3.5 ideas from `bstnxbt/dflash-mlx`:
  - target-side hybrid rollback / split-attention hooks
  - a context-only draft cache
- On short prompts, 4B improves cleanly while 9B only improves a little.
- On a long synthetic 9B prompt (`~220` repeated reasoning sentences, `max_new_tokens=64`), current local results were:
  - our repo, `draft.cache_mode=kv`: `18.00` gen TPS, `4.75` e2e TPS, `4.12` mean accept
  - our repo, `draft.cache_mode=context-only`: `19.24` gen TPS, `4.79` e2e TPS, `4.40` mean accept
  - `bstnxbt/dflash-mlx`: `17.46` gen TPS, `4.84` e2e TPS, `3.76` tokens/cycle
- So the imported hybrid-target path helps more on long prefixes than on the short Janet prompt.

Qwen3.5 DTree update:

- The useful change was not another tree-policy tweak. It was a lazy exact verifier for `qwen3_5`.
- Instead of verifying the whole tree and then following one branch, the lazy path only verifies the branch the target actually takes.
- That makes verified-node count track accepted-path length instead of fixed `tree_budget + 1`.
- The older full-tree verifier is still reachable with `DTREE_QWEN35_TREE_MODE=full_tree`.

Short Janet checks:

- bf16, `--speculative-tokens 8 --tree-budget 8`:
  - DFlash: `48.27` gen TPS, `45.30` e2e TPS, accept `5.12`
  - DTree lazy: `29.39` gen TPS, `28.19` e2e TPS, accept `6.33`
- So bf16 Qwen3.5-4B DTree is much better than the old ~`17` e2e path, but still behind DFlash on that short prompt.

Local q4 setting that held up:

- `--target-quant-bits 4 --target-quant-group-size 64`
- `--speculative-tokens 16`
- `--tree-budget 24`

Matched 8-prompt gsm8k slice, Qwen3.5-4B, `max_new_tokens=256`:

| Method | Gen TPS | End-to-end TPS | Mean accept |
|---|---:|---:|---:|
| DFlash | 47.39 | 45.07 | 5.81 |
| DTree lazy | 50.55 | 48.31 | 6.95 |

Small correctness sanity slice on the same q4 setting (`N=12`, `max_new_tokens=512`):

| Method | Accuracy |
|---|---:|
| Plain MLX-LM | 11/12 |
| DFlash | 12/12 |
| DTree lazy | 11/12 |

Takeaway:

- This is the first local Qwen3.5-4B DTree setting that beats DFlash on a broader speed slice.
- It does not yet beat DFlash on the small correctness slice.
- The short greedy q4 check still matched DFlash token-for-token for `24` generated tokens, so the lazy path looks exact in the setting we checked.
