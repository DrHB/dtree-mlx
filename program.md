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
python3 scripts/zig_round.py --round-id r001 --notes "describe the change"
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
