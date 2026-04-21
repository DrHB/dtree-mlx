# Zig Experiment Summary

This file is generated from `experiments/results.csv`.

## Latest Run

- Run: `20260421T142002.785445Z-11a5352-micro-dirty`
- Time: `2026-04-21T14:20:02.785445+00:00`
- Commit: `11a5352` on `codex/zig-baseline-qwen36`
- Dirty tree: `True`
- Subject: Add tracked Zig experiment round runner
- Label: releasefast harness sanity
- Notes: releasefast harness sanity

## Latest Metrics

| Suite | Metric | Value | Delta vs previous same suite |
|---|---|---:|---:|
| Logits Head Matvec | `full_tensor_matvecs_per_s` | 2.2272 matvec/s | +0.05924 |
| Block 0 QKV Projection | `full_projection_passes_per_s` | 71.721 projection/s | +0.19941 |

## Best So Far

| Suite | Best | Commit | Time | Label |
|---|---:|---|---|---|
| Logits Head Matvec | 2.2272 matvec/s | `11a5352` | `2026-04-21T14:20:02.785445+00:00` | releasefast harness sanity |
| Block 0 QKV Projection | 71.721 projection/s | `11a5352` | `2026-04-21T14:20:02.785445+00:00` | releasefast harness sanity |
| Fresh Full Token Pass | 0.42981 tok/s | `11a5352` | `2026-04-21T14:17:30.141807+00:00` | releasefast check |
| Cached Decode | 0.40603 tok/s | `11a5352` | `2026-04-21T14:17:30.141807+00:00` | releasefast check |

## Recent Runs

| Run | Commit | Label | Cached tok/s | Fresh tok/s | QKV proj/s | Logits matvec/s |
|---|---|---|---:|---:|---:|---:|
| `20260421T142002.785445Z-11a5352-micro-dirty` | `11a5352` | releasefast harness sanity |  |  | 71.721 | 2.2272 |
| `20260421T141730.141807Z-11a5352-decode-dirty` | `11a5352` | releasefast check | 0.40603 | 0.42981 |  |  |
| `20260421T141730.141804Z-11a5352-micro-dirty` | `11a5352` | releasefast check |  |  | 71.522 | 2.1679 |
| `20260421T141434.727674Z-11a5352-micro` | `11a5352` | pre-round sanity |  |  | 6.3511 | 0.18395 |
| `20260421T141434.727661Z-11a5352-decode` | `11a5352` | pre-round sanity | 0.04486 | 0.04898 |  |  |
| `20260421T134736.065066+0000-edd131e-full-dirty` | `edd131e` | baseline before optimization | 0.04749 | 0.04881 | 6.2632 | 0.18680 |
