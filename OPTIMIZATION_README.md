# Optimization Notes

Date: 2026-04-14

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

Experimental DTree status:

- The repo now has a correctness-first `qwen3_5` DTree path that keeps the real caches untouched during tree verify and commits only the accepted path.
- The tree path now reuses the compiled per-layer Qwen3.5 verifier kernels from the DFlash path. That helps a little, but it does not change the overall economics by itself.
- On a short greedy Qwen3.5-4B check (`24` generated tokens), DTree matched DFlash token-for-token.

Short Janet checks with `temperature=0` and `--speculative-tokens 8 --tree-budget 8`:

| Model | Method | Gen TPS | End-to-end TPS | Mean accept |
|---|---|---:|---:|---:|
| Qwen3.5-4B | DFlash | 24.64 | 21.54 | 2.75 |
| Qwen3.5-4B | DTree | 13.38 | 12.85 | 3.78 |
| Qwen3.5-9B (`max_new_tokens=16`) | DFlash | 18.22 | 6.15 | 4.20 |
| Qwen3.5-9B (`max_new_tokens=16`) | DTree | 7.29 | 6.82 | 3.20 |

Conclusion:

- `qwen3_5` DTree is functional and exact on the short 4B check.
- It is not optimized yet, especially on 4B.

Useful local setting found later:

- For Qwen3.5-4B on the short Janet check, bf16 DTree still trails DFlash even after the compiled-kernel reuse.
- But `q4_g64` changes that:
  - DFlash, `--speculative-tokens 16`: `36.55` gen TPS, `33.12` e2e TPS, accept `4.25`
  - DTree, `--speculative-tokens 16 --tree-budget 2`: `38.76` gen TPS, `34.90` e2e TPS, accept `2.36`
- On that same quantized target, DTree still matched DFlash token-for-token on a short greedy 4B check.

Validation after the short-prompt crossover:

- The short Janet win did not hold on a broader local sweep.
- 8-prompt gsm8k sweep, Qwen3.5-4B, `q4_g64`, `max_new_tokens=512`, `spec=16`, `tree_budget=2`:
  - DFlash: `44.20` gen TPS, `42.92` e2e TPS, mean accept `5.90`
  - DTree: `37.55` gen TPS, `36.69` e2e TPS, mean accept `2.76`
- 30-prompt gsm8k correctness slice on the same setting:
  - Plain: `20/30`
  - DFlash: `20/30`
  - DTree: `21/30`

So the current Qwen3.5-4B `q4` DTree setting is not a speed win on a broader sweep, even though it still looks slightly better on this small accuracy slice.
