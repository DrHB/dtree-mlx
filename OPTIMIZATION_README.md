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

## A. Adaptive Tree Budget

Best near-term idea.

Instead of always verifying `1 + tree_budget` nodes:
- use a smaller tree on easy/high-confidence rounds
- use a larger tree only on uncertain rounds

Why it matters:
- many rounds likely do not need 25 verified nodes
- this directly attacks wasted verifier work

Possible implementation:
- add `--tree-budget-mode adaptive`
- use draft top-k margin / entropy / cumulative mass to choose budget per round
- log the realized average budget

Success condition:
- same or better acceptance with lower average verified nodes

## B. Confidence-Gated Hybrid DFlash/DTree

Use DFlash behavior on easy rounds and DTree only when the draft looks uncertain.

Why it matters:
- DFlash's fast verifier path is already strong on Qwen3
- DTree should only pay its extra cost when tree branching is likely to help

Possible implementation:
- add a policy that starts from the draft logits
- choose linear verify when top-1 is decisive
- choose tree verify when the top tokens are close

This is likely more useful than always running tree verification.

## C. Better Tree Construction Policy

Current tree construction is best-first over draft log-probability.

That may not maximize accepted tokens per verifier FLOP.

Ideas:
- penalize breadth or depth explicitly
- maximize estimated accepted-prefix utility rather than raw path probability
- reserve more nodes near the early uncertain positions instead of spending them deep in the tree

Goal:
- more acceptance from the same `tree_budget`

## D. Faster DTree Verifier Output Path

DFlash benefits a lot from `parallel-greedy-argmax` on Qwen3 at `temperature=0`.
DTree still computes full LM-head logits and then argmaxes them.

Possible work:
- add a DTree greedy verifier path for `temperature=0`
- expose a real fused `lm_head_argmax()` path in the Qwen3 adapter instead of full logits followed by `mx.argmax`

Expected payoff:
- potentially meaningful, because verifier time dominates

Risk:
- only helps if the LM-head materialization is a real share of verifier cost

## E. Larger-Granularity MLX Compilation / Fusion

Current tree code uses compiled per-layer tree blocks.

Possible next step:
- compile or fuse more of the end-to-end verifier path
- reduce graph fragmentation across layer steps and post-processing

Potential targets:
- verifier forward + output extraction
- tree token compilation + mask setup

This is more invasive, but may pay off if MLX scheduling overhead is still significant.

## F. DTree Lazy-Logit / Early-Exit Verification

DFlash already has alternate verifier strategies such as lazy logits and greedy argmax.

DTree currently verifies the full tree and computes full logits for all nodes.

Potential idea:
- compute only what is needed to walk the accepted path
- or compute logits in chunks / stages for tree nodes

This is harder than the DFlash case because tree acceptance depends on multiple node outputs, but it is still one of the few directions that directly attacks the dominant cost center.

## G. Improve the Draft / Acceptance Rate

If DTree cannot get much more than `6.5` accepted tokens on average from `25` verified nodes, runtime optimization alone will not transform the result.

Possible directions:
- improve the draft model
- tune the draft attention mask mode
- tune speculative horizon for the actual prompt distribution
- train or fine-tune a tree-aware draft objective

This is higher effort, but may be the only way to move the acceptance economics materially.

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

1. Implement adaptive tree budget.
2. Add a hybrid DFlash/DTree policy based on draft confidence.
3. Add a DTree greedy verifier path for `temperature=0`.
4. Build a small sweep tool to compare fixed vs adaptive budgets on a prompt set.
5. Only after that, revisit lower-level MLX fusion work.

## Things That Are Probably Not Worth Prioritizing

- Micro-optimizing tree build in Python first
- Micro-optimizing bookkeeping first
- Changing README benchmark prompts again without code changes
- Treating the M4 Max upstream numbers as the target for this M2 Max

Those are not the main blockers right now.
