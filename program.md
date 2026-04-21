# Zig Optimization Program

This branch is restricted to the pure-Zig Qwen 3.6 port.

Stay inside this scope:

- `build.zig`
- `src/`
- `scripts/download_qwen36.sh`
- `scripts/zig_autoresearch.py`
- `experiments/`
- `README.md`
- `ZIG_BASELINE.md`

Do not reintroduce:

- MLX runtime code
- DFlash code
- DTree code
- Python model runtime code
- `llama.cpp` bootstrap paths

## Optimization Loop

Canonical benchmark run:

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
2. Prefer benchmarking from a clean git tree.
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
