# Zig Baseline

This branch is the pure-Zig Qwen 3.6 port only.

Removed on purpose from this branch:

- MLX runtime code
- DFlash code
- DTree code
- Python model-serving code
- temporary `llama.cpp` bootstrap support

## Current Coverage

The current Zig runtime already does all of this on the real GGUF:

- parse GGUF header, metadata, tensor descriptors, and alignment
- mmap tensor payloads directly
- decode `f32`, `q4_K`, and `q6_K`
- run quantized row-dot / matvec kernels
- run the fresh-token one-step full forward path
- run cached decode with recurrent state and KV cache

The branch is still tokenizer-free. Current decode benchmarking uses a repeated
token id so the model-state path is real while the text frontend stays out of
the way until the tokenizer lands.

The branch now also has a native Zig Metal bootstrap on macOS. It is not
full-model inference yet, but it proves device discovery, shader compilation,
pipeline creation, buffer upload, dispatch, and result validation from Zig.

## Current Baseline

Latest tracked best in `experiments/results.csv`:

- logits head matvec: `15.475` matvec/s
- block 0 QKV projection: `138.39` projection/s
- fresh full-token pass: `0.64510` tok/s
- cached decode: `0.59878` tok/s

## Commands

Download the default model:

```bash
./scripts/download_qwen36.sh
```

Build:

```bash
zig build
```

Run the native Metal bootstrap:

```bash
zig build metal-bootstrap
```

Run the first inference-shaped Metal projection benchmark:

```bash
./zig-out/bin/dtree-mlx-metal-bootstrap \
  --bench \
  --bench-kind matvec \
  --bench-iters 20 \
  --bench-warmup 5 \
  --rows 8192 \
  --cols 2048
```

Inspect GGUF metadata:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --all-kv
```

Inspect one tensor row:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --tensor output.weight \
  --row-index 0 \
  --value-limit 8
```

Benchmark logits-head matvec:

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

Benchmark block 0 QKV projection:

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

Benchmark the fresh-token full pass:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --full-token-pass \
  --token-id 42 \
  --bench \
  --bench-iters 2 \
  --bench-warmup 1
```

Benchmark cached decode:

```bash
zig build run -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --cached-decode \
  --token-id 42 \
  --bench \
  --bench-iters 2 \
  --bench-warmup 1
```

Rebuild experiment reports without rerunning the benchmarks:

```bash
python3 scripts/zig_autoresearch.py --reports-only
```

Run one fully tracked optimization round:

```bash
python3 scripts/zig_round.py --round-id r001 --notes "describe the change"
```

## Next Optimization Order

1. `cached_decode`
2. `full_token_pass`
3. `output.weight` logits scan
4. block-local projection kernels
5. threading, SIMD, and memory-traffic reductions
