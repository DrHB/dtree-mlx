# dtree-mlx

`dtree-mlx` is a small MLX benchmark repo for comparing two exact speculative decoding methods on Apple Silicon:

- `dflash`: linear block verification
- `dtree`: tree verification over multiple draft branches

Today the repo is focused on one pair:

- target: `mlx-community/Qwen3-4B-bf16`
- draft: `z-lab/Qwen3-4B-DFlash-b16`

There is also experimental `qwen3_5` support:

- `mlx-community/Qwen3.5-4B-bf16` + `z-lab/Qwen3.5-4B-DFlash`
- `mlx-community/Qwen3.5-9B-bf16` + `z-lab/Qwen3.5-9B-DFlash`

## Local Result

Representative local sweep on an Apple M2 Max (32 GB), `temperature=0`, `max_new_tokens=512`, 8 gsm8k prompts, 2 warmup prompts:

| Method | Gen TPS | End-to-end TPS | Mean accept | vs plain |
|---|---:|---:|---:|---:|
| Plain MLX-LM | 36.42 | 34.98 | — | 1.00x |
| DFlash | 51.87 | 50.31 | 5.71 | 1.44x |
| DTree | 56.68 | 54.83 | 7.17 | 1.57x |

Notes:

- The fast exact DFlash baseline on Qwen3 uses `--verify-mode parallel-greedy-argmax`.
- The current draft is `b16`, so `--speculative-tokens` clamps at `16`.
- Best fixed DTree setting so far is `--speculative-tokens 16 --tree-budget 24`.
- Optional target quantization (`--target-quant-bits 4 --target-quant-group-size 64`) helped DTree locally, but hurt DFlash. It stays opt-in.

More detail is in [OPTIMIZATION_README.md](OPTIMIZATION_README.md).

## Qwen3.5-4B DTree

Local q4 sweep, Apple M2 Max, 8 gsm8k prompts, `temperature=0`, `max_new_tokens=256`:

| Method | Gen TPS | End-to-end TPS | Mean accept |
|---|---:|---:|---:|
| DFlash (`spec=16`) | 47.39 | 45.07 | 5.81 |
| DTree lazy (`spec=16`, `tree_budget=24`) | 50.55 | 48.31 | 6.95 |

Notes:

- The better default for `qwen3_5` is `--draft-attention-mask none`.
- The current `qwen3_5` path uses two imported ideas from `bstnxbt/dflash-mlx`: target-side hybrid rollback hooks and a context-only draft cache.
- The lazy verifier is the first local Qwen3.5-4B DTree path that beats DFlash on a broader speed slice.
- On a short greedy q4 check, DTree still matched DFlash token-for-token for 24 generated tokens.
- Small gsm8k correctness slice on the same q4 setting (`N=12`): plain `11/12`, DFlash `12/12`, DTree `11/12`.
- bf16 Qwen3.5-4B DTree is better than before, but still behind DFlash on the short Janet prompt: about `28.19` e2e TPS with `--speculative-tokens 8 --tree-budget 8`.
- Set `DTREE_QWEN35_TREE_MODE=full_tree` to compare against the older full-tree verifier.

## Reproduce

Head-to-head local sweep:

```bash
uv run dtree-mlx-compare \
    --dataset gsm8k \
    --num-prompts 8 \
    --warmup-prompts 2 \
    --max-new-tokens 512 \
    --speculative-tokens 16 \
    --tree-budget 24 \
    --verify-mode parallel-greedy-argmax
```

Single prompt:

```bash
PROMPT="Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Please reason step by step, and put your final answer within \\boxed{}."

uv run dtree-mlx --prompt "$PROMPT" --decode-mode dflash \
    --max-new-tokens 512 --speculative-tokens 16 \
    --verify-mode parallel-greedy-argmax --warmup-runs 1

uv run dtree-mlx --prompt "$PROMPT" --decode-mode dtree \
    --max-new-tokens 512 --speculative-tokens 16 \
    --tree-budget 24 --warmup-runs 1
```

Plain baseline:

```bash
uv run dtree-mlx-bench \
    --model mlx-community/Qwen3-4B-bf16 \
    --dataset gsm8k \
    --num-prompts 8 \
    --warmup-prompts 2 \
    --max-new-tokens 512 \
    --no-history
```

## Correctness

First 30 gsm8k test prompts, `temperature=0`, `max_new_tokens=512`:

| Method | Accuracy |
|---|---:|
| Plain MLX-LM | 24/30 = 80.0% |
| DFlash | 26/30 = 86.7% |
| DTree | 27/30 = 90.0% |

This repo does not claim token-by-token equality on MLX bf16. The useful check is downstream task accuracy. See [benchmarks/correctness_gsm8k_30.md](benchmarks/correctness_gsm8k_30.md).

## Install

```bash
git clone https://github.com/DrHB/dtree-mlx.git
cd dtree-mlx
uv sync
```

Optional prefetch:

```bash
hf download mlx-community/Qwen3-4B-bf16
hf download z-lab/Qwen3-4B-DFlash-b16
```

## Commands

Single prompt:

```bash
uv run dtree-mlx --prompt "Explain quicksort" --decode-mode dflash
uv run dtree-mlx --prompt "Explain quicksort" --decode-mode dtree --tree-budget 24

uv run dtree-mlx \
    --target-model mlx-community/Qwen3.5-4B-bf16 \
    --draft-model z-lab/Qwen3.5-4B-DFlash \
    --decode-mode dflash

uv run dtree-mlx \
    --target-model mlx-community/Qwen3.5-9B-bf16 \
    --draft-model z-lab/Qwen3.5-9B-DFlash \
    --decode-mode dflash

uv run dtree-mlx \
    --target-model mlx-community/Qwen3.5-4B-bf16 \
    --draft-model z-lab/Qwen3.5-4B-DFlash \
    --decode-mode dtree \
    --target-quant-bits 4 \
    --target-quant-group-size 64 \
    --speculative-tokens 16 \
    --tree-budget 24
```

Benchmark datasets:

```bash
uv run dtree-mlx-compare \
    --dataset gsm8k \
    --num-prompts 8 \
    --warmup-prompts 2 \
    --max-new-tokens 512 \
    --speculative-tokens 16 \
    --tree-budget 24 \
    --verify-mode parallel-greedy-argmax
```

Supported datasets: `gsm8k`, `humaneval`, `math500`, `mbpp`, `mt-bench`.

Correctness sanity check:

```bash
uv run python -u scripts/downstream_gsm8k.py \
    --num-prompts 30 \
    --max-new-tokens 512 \
    --speculative-tokens 16 \
    --tree-budget 24
```

Tests:

```bash
uv run pytest tests/ -v
```

## Notes

- `dtree-mlx-compare` alternates DFlash/DTree order across prompts.
- `parallel-replay` is still available, but it is not the fast Qwen3 baseline.
- `qwen3_5` defaults to `draft_attention_mask=none`.
- `qwen3_5` DTree defaults to the lazy exact verifier; set `DTREE_QWEN35_TREE_MODE=full_tree` to force the older full-tree verifier.
- Upstream `dflash-mlx` M4 Max numbers are kept in [benchmarks/qwen3-results.md](benchmarks/qwen3-results.md) for reference only. They are not local `dtree-mlx` numbers.

## Credits

- DFlash paper and draft checkpoints: https://github.com/z-lab/dflash
- Original MLX DFlash port: https://github.com/Aryagm/dflash-mlx
- Qwen3.5 hybrid-attention hooks and kernels adapted from: https://github.com/bstnxbt/dflash-mlx
- DDTree reference implementation: https://github.com/liranringel/ddtree
- MLX / mlx-lm: https://github.com/ml-explore/mlx and https://github.com/ml-explore/mlx-lm

## License

MIT. See `LICENSE` and `NOTICE`.
