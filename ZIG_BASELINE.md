# Zig Baseline Plan

This branch now has two layers:

- default binary: pure-Zig GGUF inspection, tensor mmap, and row decoding
- optional bootstrap binary: temporary `llama.cpp` runner for short inference checks

## Why not port `qwen3.6` safetensors directly yet?

As of April 21, 2026, the official Hugging Face Qwen 3.6 release is
`Qwen/Qwen3.6-35B-A3B`, published on April 15, 2026. Its public model card
describes a hybrid layout:

- MoE routing
- Gated DeltaNet linear-attention blocks
- gated attention blocks
- MTP training hooks

That is not a small "rewrite inference" task. It is a fresh runtime project.

## Baseline choice

The first practical Zig target in this repo is:

- model family: `Qwen3.6-35B-A3B`
- artifact: `batiai/Qwen3.6-35B-A3B-GGUF`
- default file: `Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf`

This keeps the branch focused on eliminating dependencies in the right order:

1. pure-Zig file and metadata handling
2. pure-Zig tensor loading and kernels
3. pure-Zig tokenizer and prompt path
4. only then speculative decoding

Today the pure-Zig binary already covers:

- GGUF header, metadata, tensor descriptors, and alignment
- memory-mapped access to the raw tensor payload
- native row decoding for `f32`, `q4_K`, and `q6_K`
- native `f32`, `q4_K`, and `q6_K` row-dot / matvec kernels
- one-token embedding -> RMSNorm -> projection subgraphs
- full fresh-token forward over the 40-layer Qwen3.6 GGUF with:
  recurrent DeltaNet blocks, one-token full-attention blocks, routed MoE FFN,
  shared expert, and final logits-head scan
- repeated-token cached decode with:
  recurrent state updates, recurrent conv history, full-attention KV cache, and
  IMRoPE text-position handling for the Qwen35MoE full-attention layers

Current measured native baseline on this machine:

- `0.0483` fresh-token passes/s
- command: `zig build run -- --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf --full-token-pass --token-id 42 --bench --bench-iters 2 --bench-warmup 1`
- `0.0463` cached decode tok/s
- command: `zig build run -- --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf --cached-decode --token-id 42 --bench --bench-iters 2 --bench-warmup 1`

The cached path currently uses a repeated token id as input so the model-state
machinery is real while tokenizer/chat-template work remains separate.

## Optimization Tracking

There is now an autoresearch-style experiment loop for the pure-Zig branch.

Artifacts:

- `experiments/results.csv`
- `experiments/summary.md`
- `experiments/results.svg`
- `program.md`

Canonical full run:

```bash
python3 scripts/zig_autoresearch.py --profile full --notes "baseline before optimization"
```

If you want clean commit-to-metric tracking, commit first and then run the
benchmark command so the CSV row and plot point are attached to that commit.

## Milestones

1. Pure-Zig GGUF core
   Parse GGUF header, metadata, tensor descriptors, alignment, and file layout
   without `llama.cpp` or `ggml`.

2. Pure-Zig tensor backend
   Map GGUF weights directly and decode the standard tensor formats used by the
   current Qwen3.6 baseline artifact.

3. Pure-Zig tokenizer and prompt layer
   Replace external chat-template and tokenization logic.

4. Pure-Zig inference baseline
   Load the target weights and run standard autoregressive decode.

5. DFlash
   Add speculative draft and verifier plumbing after the plain path is stable.

6. DTree
   Only after a DFlash-capable draft/target interface exists in Zig.

## Commands

Download the default model:

```bash
./scripts/download_qwen36.sh
```

Build the pure-Zig binary:

```bash
zig build
```

Inspect the model in pure Zig:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --all-kv
```

Inspect a real quantized tensor row in pure Zig:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --tensor output.weight \
  --row-index 0 \
  --value-limit 8
```

Benchmark a full pure-Zig matvec sweep over the logits head:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --tensor output.weight \
  --matvec \
  --bench \
  --bench-rows 248320 \
  --bench-iters 1 \
  --bench-warmup 0
```

Benchmark the first block attention projection from a real token embedding:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --token-id 42 \
  --norm-tensor blk.0.attn_norm.weight \
  --project-tensor blk.0.attn_qkv.weight \
  --bench \
  --bench-rows 8192 \
  --bench-iters 8 \
  --bench-warmup 1
```

Run the full fresh-token pass and inspect the top logits:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --full-token-pass \
  --token-id 42 \
  --value-limit 8
```

Benchmark the full fresh-token pass:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --full-token-pass \
  --token-id 42 \
  --bench \
  --bench-iters 2 \
  --bench-warmup 1
```

Run repeated-token cached decode:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --cached-decode \
  --token-id 42 \
  --decode-steps 2 \
  --value-limit 8
```

Benchmark repeated-token cached decode:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --cached-decode \
  --token-id 42 \
  --bench \
  --bench-iters 2 \
  --bench-warmup 1
```

Build the old bootstrap runner only if you explicitly want the temporary
`llama.cpp` path:

```bash
zig build -Dllama-bootstrap=true
```
