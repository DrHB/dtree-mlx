# Zig Optimization Program

This branch is restricted to the pure-Zig Qwen 3.6 port.

Stay inside this scope:

- `build.zig`
- `src/`
- `scripts/download_qwen36.sh`
- `scripts/zig_autoresearch.py`
- `scripts/zig_round.py`
- `experiments/`
- `README.md`
- `ZIG_BASELINE.md`

Do not reintroduce:

- MLX runtime code
- DFlash code
- DTree code
- Python model runtime code
- `llama.cpp` bootstrap paths

Research is allowed to:

- use literature search
- use internet search
- inspect external papers, repos, and implementation notes for optimization ideas

Research is not allowed to:

- install non-Zig libraries or toolchains
- add non-native runtime dependencies
- solve performance by delegating core compute to external runtimes

The only acceptable dependency additions are Zig-related build or tooling pieces
that directly support the native Zig runtime.

Everything performance-critical must stay native.

## Optimization Loop

Canonical tracked round:

```bash
python3 scripts/zig_round.py --round-id r001 --profile metal --notes "describe the Metal change"
```

This round runner does the mechanical bookkeeping:

1. commit the current code change
2. run eval against that exact commit
3. commit the experiment artifacts
4. push the branch

Primitive benchmark run:

```bash
python3 scripts/zig_autoresearch.py --profile full --notes "describe the change"
```

Faster benchmark slices:

```bash
python3 scripts/zig_autoresearch.py --profile decode --notes "decode-only check"
python3 scripts/zig_autoresearch.py --profile micro --notes "kernel-only check"
python3 scripts/zig_autoresearch.py --profile metal --notes "Metal-only check"
```

Artifacts:

- `experiments/results.csv`
- `experiments/summary.md`
- `experiments/results.svg`

Rules:

1. Make one meaningful change at a time.
2. Prefer using `scripts/zig_round.py` so code commit and eval artifacts stay paired.
3. Record the change with `--notes` if the commit subject is not enough.
4. Do not hand-edit `experiments/results.csv`.
5. Regenerate reports with:

```bash
python3 scripts/zig_autoresearch.py --reports-only
```

## Metric Priority

Optimize in this order:

1. `cached_decode`
2. `full_token_pass`
3. `logits_matvec`
4. `blk0_qkv_projection`

Primary objective:

- beat the current branch baseline, not just change code

## Current Best

As of 2026-04-21 on this branch:

- `cached_decode`: `0.59878 tok/s` at `c4e45fa` (`r005`)
- `full_token_pass`: `0.64510 tok/s` at `c4e45fa` (`r005`)

These numbers are still far from the earlier bootstrap reference of about
`36.78 tok/s` on the same class of model using an external highly optimized
runtime. That gap is large enough that CPU-only tuning is not expected to close
it.

## Current Direction

CPU optimization is no longer the active track.

The branch should now focus on:

1. pure-native Zig Metal bring-up on macOS
2. Metal benchmarking through the existing experiment loop
3. inference-relevant Metal kernels

Guidance:

- Do not spend new rounds on CPU-only tuning.
- Keep the CPU path as a correctness/reference path only.
- The path to tens of tok/s is expected to come from the Metal track.
