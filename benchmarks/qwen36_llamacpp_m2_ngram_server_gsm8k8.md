# Qwen3.6-27B llama-server ngram-mod M2 Benchmark

## Summary

| Mode | Mean gen TPS | Peak gen TPS | Mean E2E TPS |
|---|---:|---:|---:|
| server_ngram_mod_checkpoints_chat | 7.83 | 8.37 | 7.49 |

## Config

```json
{
  "cache_type_k": "q8_0",
  "cache_type_v": "q8_0",
  "ctx_checkpoints": 128,
  "ctx_size": 8192,
  "dataset": "gsm8k",
  "draft_max": 48,
  "draft_min": 12,
  "llama_server_version": "b8890-8bccdbbff",
  "max_new_tokens": 256,
  "mode": "server_ngram_mod_checkpoints_chat_prompt",
  "num_prompts": 8,
  "prompt_format": "manual_qwen_chat_template",
  "server": "http://127.0.0.1:18080/completion",
  "spec_ngram_size_n": 24,
  "spec_type": "ngram-mod",
  "target": "models/qwen3.6-27b-q4km/Qwen_Qwen3.6-27B-Q4_K_M.gguf",
  "warmup_prompts": 2
}
```
