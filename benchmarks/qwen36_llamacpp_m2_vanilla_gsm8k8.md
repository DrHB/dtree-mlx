# Qwen3.6-27B llama.cpp M2 Benchmark

## Config

```json
{
  "cache_type_k": "q8_0",
  "cache_type_v": "q8_0",
  "ctx_size": 8192,
  "ctx_size_draft": 4096,
  "dataset": "gsm8k",
  "draft_file": "Qwen3-1.7B-Q4_K_M.gguf",
  "draft_max": 12,
  "draft_min": 3,
  "draft_p_min": 0.6,
  "draft_path": "/Users/drhb/Documents/tmp/dtree-mlx/models/qwen3-1.7b-q4km/Qwen3-1.7B-Q4_K_M.gguf",
  "draft_repo": "ggml-org/Qwen3-1.7B-GGUF",
  "gpu_layers": "99",
  "gpu_layers_draft": "99",
  "llama_cli": null,
  "llama_completion": "/opt/homebrew/bin/llama-completion",
  "llama_speculative": "/opt/homebrew/bin/llama-speculative",
  "n_predict": 256,
  "num_prompts": 8,
  "prompt": null,
  "prompt_file": null,
  "runs": null,
  "seed": 0,
  "spec_replace": [],
  "target_file": "Qwen_Qwen3.6-27B-Q4_K_M.gguf",
  "target_path": "/Users/drhb/Documents/tmp/dtree-mlx/models/qwen3.6-27b-q4km/Qwen_Qwen3.6-27B-Q4_K_M.gguf",
  "target_repo": "bartowski/Qwen_Qwen3.6-27B-GGUF",
  "warmup_prompts": 2,
  "warmup_runs": 0,
  "warmup_tokens": 32
}
```

## Summary

| Mode | Mean gen TPS | Peak gen TPS | Mean total TPS | Peak memory GB | Acceptance |
|---|---:|---:|---:|---:|---:|
| vanilla | 7.79 | 8.18 | 7.51 | 16.90 |  |

## Commands

### vanilla

```bash
/opt/homebrew/bin/llama-completion -m /Users/drhb/Documents/tmp/dtree-mlx/models/qwen3.6-27b-q4km/Qwen_Qwen3.6-27B-Q4_K_M.gguf -p Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
Please reason step by step, and put your final answer within \boxed{}. -n 256 -c 8192 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --temp 0 --seed 0 --no-display-prompt --simple-io --conversation --single-turn --jinja --no-warmup
```

