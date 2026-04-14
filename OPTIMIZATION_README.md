# Optimization Notes

Date: 2026-04-14

This document captures the current performance findings for `dtree-mlx` on the local Apple M2 Max machine and lists the most plausible optimization directions for both DFlash and DTree.

## Current Local Numbers

Machine:
- Apple M2 Max
- 32 GB RAM

Prompt used for the single-prompt checks below:
- Janet gsm8k-style word problem from the repo README

Settings:
- `temperature=0`
- `max_new_tokens=512`
- `speculative_tokens=16`
- `tree_budget=24` for DTree
- 1 warmup run

Measured results on this machine:

| Method | Gen TPS | End-to-end TPS | Notes |
|---|---:|---:|---|
| Vanilla MLX-LM | 35.13 | 33.41 | Plain autoregressive baseline |
| DFlash (`parallel-replay`) | 33.16 | 31.98 | Exact but slow verifier path |
| DFlash (`parallel-greedy-argmax`) | 61.71 | 59.33 | Exact, faster Qwen3 path at `temperature=0` |
| DTree | 62.16 | 59.73 | Exact tree verification |

Important context:
- The upstream `dflash-mlx` headline benchmark was recorded on an M4 Max, not this M2 Max.
- The old README table in this repo mixed stale measurements with an omitted DFlash verifier mode.
- DTree is not currently "broken" on this machine. It is already roughly tied with the fast DFlash baseline.

Prompt-set sweep on three gsm8k-style prompts:
- Best tested DFlash setting in the sweep: `speculative_tokens=16`
- Important correction: the current `z-lab/Qwen3-4B-DFlash-b16` draft has `block_size=16`, and both DFlash and DTree clamp `speculative_tokens` to `draft.block_size`, so any earlier sweep entry above `16` was effectively still running `16`.
- Best tested fixed DTree setting in the sweep was therefore effectively `speculative_tokens=16`, `tree_budget=24`
- Repeated A/B at requested `speculative_tokens=20`:
  - `tree_budget=18`: `55.40 gen_tps / 52.81 e2e_tps`, accept `6.15`
  - `tree_budget=24`: `60.11 gen_tps / 57.03 e2e_tps`, accept `6.83`

## What We Verified

### 1. The old README numbers were misleading

For the Janet prompt:
- The DFlash row only matched when using `--verify-mode parallel-greedy-argmax`.
- The default exact `parallel-replay` path was much slower.
- The README has been corrected.

### 2. The upstream M4 numbers are not the right baseline for this M2

Spot checks using the local upstream `dflash-mlx` repo showed:
- Plain MLX-LM on this M2 is around `38 tok/s` at 512 on the upstream built-in prompt.
- Upstream DFlash on this M2 is well below the published M4 headline values.

Conclusion:
- There is no evidence that the local machine is misconfigured.
- A substantial part of the gap is simply M2 Max vs M4 Max hardware.

### 3. DTree is verifier-bound, not tree-build-bound

From `scripts/profile_dtree.py` on this machine:
- DTree `verify_time_s` is roughly `83%` of decode time.
- Tree build is about `1.6%`.
- Bookkeeping is about `0.6%`.

Conclusion:
- The dominant cost is still target-side verification.
- Tree-building Python/NumPy overhead is real but not the main reason DTree is not much faster.

### 4. The custom tree forward is not obviously the hot regression

From `scripts/micro_verify.py`:
- Standard forward and tree-style forward were basically the same cost at the tested shape.

Conclusion:
- The tree-specific attention path itself is not the obvious problem.
- The issue is the economics of the verifier work, not a catastrophic implementation bug.

### 5. One candidate optimization was tested and ruled out

Tried:
- Avoiding eager concatenation of all verifier hidden states before selecting accepted indices.

Result:
- No meaningful change in Janet-prompt throughput.

Conclusion:
- Hidden-state concatenation is not the main bottleneck at current sizes.

### 6. Adaptive and hybrid tree policies were tested and did not help

Tested:
- `tree_budget_mode=adaptive`
- `tree_budget_mode=hybrid` with two policies:
  - linear fallback on any reduced budget
  - linear fallback only on the strongest-confidence rounds

Representative results from `scripts/profile_dtree.py`:
- Fixed `tree_budget=24`: about `58.9 gen_tps / 55.9 e2e_tps`, accept `6.83`
- Adaptive cap `24`: about `50.3 gen_tps / 48.1 e2e_tps`, accept `5.69`
- Hybrid cap `24` after tightening the gate: about `51.0 gen_tps / 48.6 e2e_tps`, accept `6.19`

Conclusion:
- The confidence heuristic cut too much acceptance.
- Routing easy rounds to linear verification does not recover enough performance.
- Tree-routing policy is not the best next lever.

### 7. Sweeping speculative horizon surfaced a block-size cap

From a fixed-budget sweep:
- DFlash improved up to about `speculative_tokens=16` and then flattened.
- DTree with `tree_budget=24` was weak at `speculative_tokens=8` and `12`, then appeared to plateau once `speculative_tokens >= 16`.
- That apparent plateau was explained by the draft cap: `speculative_tokens > 16` was being clamped to `16`.

Conclusion:
- There is no larger-horizon win available with the current `b16` draft.
- Any real horizon gain now requires a different draft checkpoint with `block_size > 16`.

### 8. A simple DTree greedy verifier path did not move the needle

Tested:
- wiring `parallel-greedy-argmax` into the actual tree posterior-selection path

Repeated fixed-DTree comparison at requested `speculative_tokens=20`, `tree_budget=24`:
- `parallel-replay`: `60.80 gen_tps / 57.70 e2e_tps`
- `parallel-greedy-argmax`: `60.74 gen_tps / 57.13 e2e_tps`

Conclusion:
- The tree verifier is not spending meaningful time in the final posterior token selection step.
- The expensive part is earlier in the exact tree forward / hidden-state verification work.

### 9. A shallower, breadth-biased tree score looked worse

Ad hoc experiment:
- monkeypatched the tree builder to score candidates as `path_log_prob - depth_penalty * depth`
- tested small positive penalties on the current best fixed setting

Observed trend before the run was stopped by Metal OOM from repeated model reloads:
- baseline penalty `0.00`: about `62.4 gen_tps / 59.2 e2e_tps`, accept `6.83`
- penalty `0.15`: about `60.1 gen_tps / 57.1 e2e_tps`, accept `6.63`
- penalty `0.30`: about `56.4 gen_tps / 53.7 e2e_tps`, accept `6.58`

Conclusion:
- simple depth penalization reduces acceptance and throughput on this prompt set
- the current best-first path score is not obviously leaving an easy win on the table

### 10. Target quantization is mode-dependent, not a free win

Local code now supports optional target quantization via:
- `--target-quant-bits`
- `--target-quant-group-size`

Matched 8-prompt gsm8k sweep on this M2 Max with `verify_mode=parallel-greedy-argmax`:
- bf16 target:
  - DFlash: `51.87 gen_tps / 50.31 e2e_tps`, accept `5.71`
  - DTree: `56.68 gen_tps / 54.83 e2e_tps`, accept `7.17`
- `4-bit`, `group_size=64` target:
  - DFlash: `48.96 gen_tps / 47.36 e2e_tps`, accept `5.56`
  - DTree: `58.43 gen_tps / 56.09 e2e_tps`, accept `6.88`

Short 3-prompt group-size spot check:
- bf16 target:
  - DFlash: `52.69 e2e_tps`
  - DTree: `54.46 e2e_tps`
- `4-bit`, `group_size=32`:
  - DFlash: `46.43 e2e_tps`
  - DTree: `56.15 e2e_tps`
- `4-bit`, `group_size=64`:
  - DFlash: `46.18 e2e_tps`
  - DTree: `58.64 e2e_tps`
- `4-bit`, `group_size=128`:
  - DFlash: `48.05 e2e_tps`
  - DTree: `54.38 e2e_tps`

Small downstream gsm8k sanity slice (`N=12`, `max_new_tokens=512`):
- bf16 target:
  - plain: `9/12`
  - DFlash: `10/12`
  - DTree: `11/12`
- `4-bit`, `group_size=64` target:
  - plain: `11/12`
  - DFlash: `12/12`
  - DTree: `11/12`

Conclusion:
- `4-bit`, `group_size=64` is the first runtime change that consistently helped DTree on local sweeps.
- The same setting hurt DFlash throughput on the same prompts.
- The small gsm8k slice did not show a correctness collapse, but it is far too small to treat as definitive.
- This should remain an opt-in DTree-focused flag, not a new default.

## Core Performance Problem

DTree currently verifies a larger batch than DFlash, but does not gain enough extra acceptance to dominate.

Approximate comparison on the Janet prompt:
- DFlash verifies `16` positions and accepts about `5.90`
- DTree verifies `25` nodes and accepts about `6.53`

That means:
- DTree gets only about `10%` more accepted tokens per step
- but pays for about `56%` more verifier positions

So the real problem is:
- not enough additional acceptance for the amount of extra exact target work

## Priority Optimization Ideas

## A. Better Tree Construction Policy

Current tree construction is best-first over draft log-probability.

That may not maximize accepted tokens per verifier FLOP.

Ideas:
- penalize breadth or depth explicitly
- maximize estimated accepted-prefix utility rather than raw path probability
- reserve more nodes near the early uncertain positions instead of spending them deep in the tree

Goal:
- more acceptance from the same `tree_budget`

## B. Real Fused LM-Head / Top-1 Path

Possible work:
- expose a real fused `lm_head_argmax()` path in the Qwen3 adapter instead of full logits followed by `mx.argmax`
- verify whether MLX can avoid materializing the full vocab projection for top-1 selection

Expected payoff:
- potentially meaningful, because verifier time dominates and the simple tree-side argmax rewrite was neutral

Risk:
- only helps if the LM-head / output projection is a real share of verifier cost

## C. Larger-Granularity MLX Compilation / Fusion

Current tree code uses compiled per-layer tree blocks.

Possible next step:
- compile or fuse more of the end-to-end verifier path
- reduce graph fragmentation across layer steps and post-processing

Potential targets:
- verifier forward + output extraction
- tree token compilation + mask setup

This is more invasive, but may pay off if MLX scheduling overhead is still significant.

## D. DTree Lazy-Logit / Early-Exit Verification

DFlash already has alternate verifier strategies such as lazy logits and greedy argmax.

DTree currently verifies the full tree and computes full logits for all nodes.

Potential idea:
- compute only what is needed to walk the accepted path
- or compute logits in chunks / stages for tree nodes

This is harder than the DFlash case because tree acceptance depends on multiple node outputs, but it is still one of the few directions that directly attacks the dominant cost center.

## E. Improve the Draft / Acceptance Rate

If DTree cannot get much more than `6.5` accepted tokens on average from `25` verified nodes, runtime optimization alone will not transform the result.

Possible directions:
- improve the draft model
- tune the draft attention mask mode
- tune speculative horizon for the actual prompt distribution
- train or fine-tune a tree-aware draft objective

This is higher effort, but may be the only way to move the acceptance economics materially.

## F. Adaptive / Hybrid Policies Revisited Later

These were tested once and lost clearly, but they may be worth revisiting only if:
- the tree construction policy changes
- the confidence signal gets much better
- or a cheaper linear verifier path becomes available

For now they should not be the main focus.

## Secondary Ideas

## H. Budget/Horizon Sweep as a First-Class Tool

Right now the best settings are found manually.

Useful addition:
- add a CLI sweep for `speculative_tokens` and `tree_budget`
- report gen/e2e TPS, acceptance, and average verified nodes

This will make regressions and sweet spots obvious.

## I. Prompt-Bucketed Benchmarking

Some prompts make DFlash look much faster than others.

Useful additions:
- benchmark prompts bucketed by acceptance / entropy / output length
- report prompt classes where DTree helps most

This may show that DTree is strong on a subset of workloads even if the overall mean is modest.

## J. Benchmark Hygiene

Keep the following stable when comparing methods:
- explicit warmup
- fixed verifier mode
- fixed prompt
- isolated runs rather than competing parallel model loads
- clear hardware labeling with absolute machine names

This repo already had one README-number mismatch caused by hidden benchmark assumptions. Avoid repeating that.

## Suggested Next Work Order

1. Improve tree construction so the same verified-node budget buys more acceptance.
2. Investigate a real fused MLX top-1 / LM-head path for Qwen3.
3. Explore larger-granularity compilation or fusion of the verifier path.
4. Consider tree-aware lazy-logit / early-exit verification ideas.
5. Revisit adaptive or hybrid routing only after one of the above changes lands.

## Things That Are Probably Not Worth Prioritizing

- Micro-optimizing tree build in Python first
- Micro-optimizing bookkeeping first
- Changing README benchmark prompts again without code changes
- Treating the M4 Max upstream numbers as the target for this M2 Max

Those are not the main blockers right now.
