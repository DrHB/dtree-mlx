# Zig Experiment Summary

This file is generated from `experiments/results.csv`.

## Latest Run

- Run: `20260421T134736.065066+0000-edd131e-full-dirty`
- Time: `2026-04-21T13:47:36.065066+00:00`
- Commit: `edd131e` on `codex/zig-baseline-qwen36`
- Dirty tree: `True`
- Subject: Record lazy linear-step experiment
- Label: baseline before optimization
- Notes: baseline before optimization

## Latest Metrics

| Suite | Metric | Value | Delta vs previous same suite |
|---|---|---:|---:|
| Logits Head Matvec | `full_tensor_matvecs_per_s` | 0.18680 matvec/s | n/a |
| Block 0 QKV Projection | `full_projection_passes_per_s` | 6.2632 projection/s | n/a |
| Fresh Full Token Pass | `fresh_token_tok_per_s` | 0.04881 tok/s | n/a |
| Cached Decode | `cached_decode_tok_per_s` | 0.04749 tok/s | n/a |

## Best So Far

| Suite | Best | Commit | Time | Label |
|---|---:|---|---|---|
| Logits Head Matvec | 0.18680 matvec/s | `edd131e` | `2026-04-21T13:47:36.065066+00:00` | baseline before optimization |
| Block 0 QKV Projection | 6.2632 projection/s | `edd131e` | `2026-04-21T13:47:36.065066+00:00` | baseline before optimization |
| Fresh Full Token Pass | 0.04881 tok/s | `edd131e` | `2026-04-21T13:47:36.065066+00:00` | baseline before optimization |
| Cached Decode | 0.04749 tok/s | `edd131e` | `2026-04-21T13:47:36.065066+00:00` | baseline before optimization |

## Recent Runs

| Run | Commit | Label | Cached tok/s | Fresh tok/s | QKV proj/s | Logits matvec/s |
|---|---|---|---:|---:|---:|---:|
| `20260421T134736.065066+0000-edd131e-full-dirty` | `edd131e` | baseline before optimization | 0.04749 | 0.04881 | 6.2632 | 0.18680 |
