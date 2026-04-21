# Zig Optimization Program

This repo now has a dedicated pure-Zig optimization loop.

The canonical artifacts are:

- `experiments/results.csv`
- `experiments/summary.md`
- `experiments/results.svg`

The canonical runner is:

```bash
python3 scripts/zig_autoresearch.py --profile full --notes "baseline before SIMD"
```

Use it like this:

1. Make a meaningful change.
2. Commit it if you want clean commit-to-metric tracking.
3. Run the canonical benchmark profile.
4. Commit the updated experiment artifacts with the code change.

The runner always:

- builds the current Zig binary unless `--skip-build` is passed
- runs the selected benchmark profile
- appends one CSV row per suite
- records git branch, commit, dirty state, subject, and your notes
- writes a per-run raw JSON artifact under `experiments/runs/`
- regenerates `experiments/summary.md`
- regenerates `experiments/results.svg`

Profiles:

- `full`: logits-head matvec, block-0 QKV projection, fresh full-token pass, cached decode
- `decode`: fresh full-token pass and cached decode only
- `micro`: logits-head matvec and block-0 QKV projection only

Current performance work should optimize these in order:

1. `cached_decode`
2. `full_token_pass`
3. `logits_matvec`
4. `blk0_qkv_projection`

Rules for the optimization loop:

- Do not hand-edit `experiments/results.csv`.
- Use `--notes` to capture what changed if the git subject is too generic.
- Prefer benchmarking from a clean tree so the chart maps cleanly to commits.
- If you only need to rebuild the summary and plot from the CSV, run:

```bash
python3 scripts/zig_autoresearch.py --reports-only
```
