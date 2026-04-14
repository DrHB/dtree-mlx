# Upstream dflash-mlx Reference Results

These numbers are from the upstream `dflash-mlx` project on an M4 Max. They are kept here as context only.

They are not local `dtree-mlx` results and should not be compared directly to the M2 Max numbers in this repo.

## Source setup

- hardware: MacBook Pro M4 Max, 36 GB
- prompt: built-in functional-equation prompt
- sampling: greedy, `temperature=0`
- verifier: exact `parallel-replay`
- draft: `z-lab/Qwen3-4B-DFlash-b16`
- warmup: 1 warmup run, 128 max new tokens

## BF16 long generation

| Max new tokens | Plain MLX-LM tok/s | dflash-mlx tok/s | Speedup | Avg accept |
|---:|---:|---:|---:|---:|
| 512 | 42.3 | 133.1 | 3.1x | 8.81 |
| 1024 | 42.0 | 144.6 | 3.4x | 9.66 |
| 2048 | 41.3 | 174.4 | 4.2x | 11.97 |
| 4028 | 40.6 | 186.4 | 4.6x | 13.55 |

## Runtime comparison at 4028 tokens

| Target | Runtime | Model | tok/s |
|---|---|---|---:|
| BF16 | llama.cpp | `Qwen3-4B-BF16.gguf` | 41.1 |
| BF16 | MLX-LM | `mlx-community/Qwen3-4B-bf16` | 40.6 |
| BF16 | dflash-mlx | `mlx-community/Qwen3-4B-bf16` | 186.4 |
| 4-bit / Q4_K_M | llama.cpp | `Qwen3-4B-Q4_K_M.gguf` | 97.8 |
| 4-bit | MLX-LM | `mlx-community/Qwen3-4B-4bit` | 110.5 |
| 4-bit | dflash-mlx | `mlx-community/Qwen3-4B-4bit` | 159.2 |
