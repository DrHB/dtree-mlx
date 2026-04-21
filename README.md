# dtree-mlx

This branch is now intentionally narrow: it is a pure-Zig Qwen 3.6 inference
port workspace.

The old MLX / DFlash / DTree code has been removed from this branch so the repo
only tracks:

- the pure-Zig runtime in `src/`
- the Zig build in `build.zig`
- the model download helper in `scripts/download_qwen36.sh`
- the benchmark / experiment loop in `scripts/zig_autoresearch.py`
- generated experiment history in `experiments/`

## Scope

Current target:

- model family: `Qwen3.6-35B-A3B`
- artifact: GGUF
- default file:
  `models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf`

Current runtime scope:

- GGUF metadata parsing
- mmap tensor access
- native `f32`, `q4_K`, and `q6_K` decode
- native row-dot / matvec kernels
- full fresh-token forward in pure Zig
- cached autoregressive decode in pure Zig

Current non-goals for this branch:

- MLX integration
- Python model runtime
- DFlash
- DTree
- any `llama.cpp` bootstrap path

## Quickstart

Download the default model:

```bash
./scripts/download_qwen36.sh
```

Build the pure-Zig binary:

```bash
zig build --release=fast
```

Run the current full-token baseline:

```bash
zig build run --release=fast -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --full-token-pass \
  --token-id 42 \
  --bench \
  --bench-iters 2 \
  --bench-warmup 1
```

Run the current cached decode baseline:

```bash
zig build run --release=fast -- \
  --model models/qwen3.6-35b-a3b-q4km/Qwen-Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --cached-decode \
  --token-id 42 \
  --bench \
  --bench-iters 2 \
  --bench-warmup 1
```

## Current Baseline

Recorded pure-Zig baseline on this branch:

- `logits_matvec`: `0.18680` matvec/s
- `blk0_qkv_projection`: `6.2632` projection/s
- `full_token_pass`: `0.04881` tok/s
- `cached_decode`: `0.04749` tok/s

Those results are tracked in:

- [experiments/results.csv](experiments/results.csv)
- [experiments/summary.md](experiments/summary.md)
- [experiments/results.svg](experiments/results.svg)

## Experiment Loop

This branch uses a small autoresearch-style loop around the Zig binary.

Canonical tracked round:

```bash
python3 scripts/zig_round.py --round-id r001 --profile micro --notes "baseline before simd"
```

That command:

- commits the code change
- runs eval against that exact commit
- commits the generated experiment artifacts
- pushes the branch

Primitive eval-only run:

```bash
python3 scripts/zig_autoresearch.py --profile full --notes "baseline before simd"
```

Faster slices:

```bash
python3 scripts/zig_autoresearch.py --profile decode --notes "decode-only check"
python3 scripts/zig_autoresearch.py --profile micro --notes "kernel-only check"
```

Regenerate the markdown summary and SVG plot from the CSV:

```bash
python3 scripts/zig_autoresearch.py --reports-only
```

See [program.md](program.md) for the exact
rules of the optimization loop.

## Layout

- [build.zig](build.zig): Zig build entrypoint
- [src/pure_main.zig](src/pure_main.zig): CLI entrypoint
- [src/gguf.zig](src/gguf.zig): GGUF parser
- [src/gguf_store.zig](src/gguf_store.zig): mmap tensor store
- [src/quant.zig](src/quant.zig): quant decode + dot kernels
- [src/ops.zig](src/ops.zig): shared math ops
- [src/single_token.zig](src/single_token.zig): fresh-token forward
- [src/cached_decode.zig](src/cached_decode.zig): cached decode
- [scripts/zig_autoresearch.py](scripts/zig_autoresearch.py): benchmark harness
- [scripts/zig_round.py](scripts/zig_round.py): tracked experiment round runner
- [ZIG_BASELINE.md](ZIG_BASELINE.md): port notes and commands
