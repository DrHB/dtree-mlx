# Correctness sanity check: gsm8k (N=30)

Date: 2026-04-14

Setup:

- target: `mlx-community/Qwen3-4B-bf16`
- draft: `z-lab/Qwen3-4B-DFlash-b16`
- `temperature=0`
- `max_new_tokens=512`
- `speculative_tokens=16`
- `tree_budget=24`

Result on the first 30 gsm8k test prompts:

| Method | Accuracy |
|---|---:|
| Plain MLX-LM | 24/30 = 80.0% |
| DFlash | 26/30 = 86.7% |
| DTree | 27/30 = 90.0% |

This check is task-level, not token-level. On MLX bf16, exact token equality is not a useful bar because kernel shapes can change rounding.

## Reproduce

```bash
uv run python -u scripts/downstream_gsm8k.py \
    --num-prompts 30 \
    --max-new-tokens 512 \
    --speculative-tokens 16 \
    --tree-budget 24
```
