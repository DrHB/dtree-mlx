# Qwen3.6-27B M2 llama.cpp Benchmark

Date: 2026-04-24

Host: Apple M2 Max, llama.cpp b8890, Metal backend.

## Result

| Row | Backend | Mean gen TPS | Peak gen TPS | Mean E2E TPS | Peak memory | Status |
|---|---|---:|---:|---:|---:|---|
| vanilla_qwen36 | llama-completion | 7.79 | 8.18 | 7.51 | 16.90 GB | passed |
| ngram_mod_checkpoint | llama-server | 7.83 | 8.37 | 7.49 | ~17 GB Metal model residency | passed |
| qwen17_draft_spec | llama-speculative | n/a | n/a | n/a | n/a | blocked by tokenizer/special-token mismatch |

## Setup

Target:

```text
bartowski/Qwen_Qwen3.6-27B-GGUF
Qwen_Qwen3.6-27B-Q4_K_M.gguf
```

Requested draft:

```text
ggml-org/Qwen3-1.7B-GGUF
Qwen3-1.7B-Q4_K_M.gguf
```

Sweep:

```text
dataset = gsm8k
warmup_prompts = 2
num_prompts = 8
max_new_tokens = 256
ctx = 8192
cache = q8_0 K/V
temperature = 0
```

## Important Finding

The screenshot's Qwen3-1.7B draft setup does not run on upstream llama.cpp b8890 with this Qwen3.6-27B GGUF. The target tokenizer has `n_vocab = 248320`; the Qwen3-1.7B draft tokenizer has `n_vocab = 151936`. `llama-speculative` exits before decode with:

```text
main: draft model special tokens must match target model to use speculation
```

Passing same-string `--spec-replace` pairs for the special tokens did not bypass this check. This means the tweet likely used a patched runtime or a tokenizer-compatible model artifact, not stock Homebrew llama.cpp with these public GGUFs.

## N-gram Speculative Row

After upgrading llama.cpp from b8680 to b8890, Qwen3.6 recurrent-state speculative decoding can initialize with context checkpoints:

```text
--spec-type ngram-mod --spec-ngram-size-n 24 --draft-min 12 --draft-max 48 -ctxcp 128
```

On GSM8K this produced only marginal acceptance and no real speedup. The useful number is therefore the vanilla M2 baseline: about `7.8 tok/s` generation for Qwen3.6-27B Q4_K_M at 8K context on this machine.

Artifacts:

- `benchmarks/qwen36_llamacpp_m2_vanilla_gsm8k8.json`
- `benchmarks/qwen36_llamacpp_m2_vanilla_gsm8k8.md`
- `benchmarks/qwen36_llamacpp_m2_ngram_server_gsm8k8.json`
- `benchmarks/qwen36_llamacpp_m2_ngram_server_gsm8k8.md`
