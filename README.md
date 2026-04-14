# dtree-mlx

Head-to-head benchmark of **DFlash** (block-diffusion speculative decoding) vs **DTree** (block-diffusion tree speculative decoding) on Apple Silicon, using MLX and Qwen3-4B.

Both methods share the same target model and the same DFlash draft; they differ only in how the draft's proposals are verified:

- **DFlash** — linear chain verification (one speculative path per step).
- **DTree** — tree verification over multiple speculative paths per step, built from the draft's top-k logits via best-first search over log-probability.

This repo measures what each method delivers on the same hardware, same prompt set, same checkpoints.

## Speed

Single-prompt head-to-head on Apple M2 Max (32 GB), Qwen3-4B-bf16 target + z-lab Qwen3-4B-DFlash draft. `temperature=0`, `max_new_tokens=512`, `speculative_tokens=16`, `tree_budget=24`, 1 warmup pass. DFlash below uses exact `verify_mode=parallel-greedy-argmax` on Qwen3, which is materially faster than the default exact `parallel-replay` verifier at `temperature=0`. Prompt is a gsm8k-style word problem. For single-prompt numbers, use the single-mode CLI below so warmup is explicit and mode-order bias is avoided.

| Method | Gen TPS | End-to-end TPS | Mean accept length | vs Vanilla |
|---|---:|---:|---:|---:|
| Vanilla (MLX-LM)   | 35.13 | 33.41 | — | 1.00× |
| DFlash (`parallel-greedy-argmax`) | 61.71 | 59.33 | 5.90 | 1.78× |
| DTree              | 62.16 | 59.73 | 6.53 | **1.79×** |

Both spec-decode methods comfortably beat vanilla. On this prompt, DTree still accepts ~10% more tokens per step than DFlash (6.53 vs 5.90), and with the current Qwen3 verifier paths both methods land at roughly the same throughput on this M2 Max. The important caveat is that the DFlash baseline is sensitive to verifier mode: exact `parallel-greedy-argmax` is much faster than exact `parallel-replay` on Qwen3 at `temperature=0`. The remaining optimization target for DTree is still verifier cost: tree verification dominates DTree decode time, especially once `tree_budget` grows past 24.

Reproduce:

```bash
PROMPT="Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Please reason step by step, and put your final answer within \\boxed{}."

# DFlash
uv run dtree-mlx --prompt "$PROMPT" --decode-mode dflash \
    --max-new-tokens 512 --speculative-tokens 16 \
    --verify-mode parallel-greedy-argmax --warmup-runs 1

# DTree
uv run dtree-mlx --prompt "$PROMPT" --decode-mode dtree \
    --max-new-tokens 512 --speculative-tokens 16 \
    --tree-budget 24 --warmup-runs 1

# Vanilla MLX-LM baseline
uv run dtree-mlx-bench --model mlx-community/Qwen3-4B-bf16 \
    --prompt "$PROMPT" --max-new-tokens 512 --num-prompts 1 --warmup-prompts 1
```

## Correctness sanity check

Three-way accuracy on the first 30 gsm8k-test prompts at `temperature=0`, `max_new_tokens=512`:

| Method | Accuracy |
|---|---:|
| Plain autoregressive (MLX-LM) | 24/30 = 80.0% |
| DFlash (`speculative_tokens=16`) | 26/30 = 86.7% |
| DTree (`speculative_tokens=16`, `tree_budget=24`) | 27/30 = 90.0% |

All three within binomial noise (±17pp at N=30). No regression in DTree relative to DFlash or plain. We do **not** assert bit-level token equality because MLX bf16 kernels are shape-dependent and neither upstream (z-lab DFlash, Ringel DDTree) holds themselves to that bar either — both validate via downstream task accuracy. See [`benchmarks/correctness_gsm8k_30.md`](benchmarks/correctness_gsm8k_30.md).

## Install

Clone and install deps via `uv` (recommended — handles the virtualenv automatically):

```bash
git clone https://github.com/DrHB/dtree-mlx.git
cd dtree-mlx
uv sync
```

`uv sync` reads `pyproject.toml` and installs everything into `./.venv`. You don't need to activate anything — `uv run <cmd>` picks up the venv automatically.

## Download the models

Both checkpoints live on Hugging Face. First run auto-downloads them, but you can pre-fetch them to avoid hiccups on slow networks:

```bash
# target model (~10 GB bf16)
hf download mlx-community/Qwen3-4B-bf16

# DFlash draft (~1 GB)
hf download z-lab/Qwen3-4B-DFlash-b16
```

## Run

### Head-to-head benchmark (primary use case)

```bash
uv run dtree-mlx-compare \
    --dataset gsm8k \
    --num-prompts 8 \
    --warmup-prompts 2 \
    --max-new-tokens 256 \
    --speculative-tokens 16 \
    --tree-budget 24
```

`dtree-mlx-compare` alternates the DFlash/DTree measurement order across prompts to reduce second-run bias. For a single prompt, prefer the per-mode `dtree-mlx` commands above. If you want the faster exact DFlash baseline on Qwen3 at `temperature=0`, pass `--verify-mode parallel-greedy-argmax`.

Supported datasets: `gsm8k`, `humaneval`, `math500`, `mbpp`, `mt-bench`.

### Single-prompt generation

```bash
# DFlash (baseline)
uv run dtree-mlx --prompt "Explain quicksort" --decode-mode dflash

# DTree (new)
uv run dtree-mlx --prompt "Explain quicksort" --decode-mode dtree --tree-budget 24
```

### Plain autoregressive baseline (no speculation)

```bash
uv run dtree-mlx-bench \
    --model mlx-community/Qwen3-4B-bf16 \
    --dataset gsm8k \
    --num-prompts 8
```

### Correctness sanity check (reruns the gsm8k accuracy table)

```bash
uv run python -u scripts/downstream_gsm8k.py \
    --num-prompts 30 \
    --max-new-tokens 512 \
    --speculative-tokens 16 \
    --tree-budget 24
```

## Tests

```bash
uv run pytest tests/ -v
```

Smoke tests cover: imports, CLI entrypoints, DTree tree-build invariants, determinism of both decode modes.

## Credits

This project is built on top of several open-source projects. Please cite them if you use this work.

### DFlash
The DFlash algorithm, draft-model architecture, and original implementation are from **z-lab**:
- Paper: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- Upstream (non-MLX) reference: https://github.com/z-lab/dflash (MIT)
- Pretrained drafts: https://huggingface.co/collections/z-lab/dflash

### dflash-mlx
This repository is derived from **dflash-mlx** by Arya Manjaramkar, which first brought DFlash to Apple Silicon:
- https://github.com/Aryagm/dflash-mlx (MIT, (c) 2026 Arya Manjaramkar)

The MLX adapter for Qwen3, the parallel-replay verifier, and the DFlash draft loader all originate there.

### DDTree
The tree-verification algorithm and the heap-based tree construction are ported from **DDTree** by Liran Ringel:
- https://github.com/liranringel/ddtree (MIT, (c) 2026 Liran Ringel)
- Writeup: https://liranringel.github.io/ddtree/DDTree.pdf

Our port brings it from the original non-MLX reference to MLX, and restricts it to Qwen3 for now. Any bugs introduced during the port are ours, not the upstream's.

### mlx / mlx-lm
Model loading, KV cache, and the Qwen3 transformer blocks are from Apple's MLX and mlx-lm projects:
- https://github.com/ml-explore/mlx (MIT)
- https://github.com/ml-explore/mlx-lm (MIT)

## License

MIT. See `LICENSE` and `NOTICE` for attribution to upstream projects.
