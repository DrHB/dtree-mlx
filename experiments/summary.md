# Zig Experiment Summary

This file is generated from `experiments/results.csv`.

## Latest Run

- Run: `20260421T144028.003287Z-f8c07b8-decode`
- Time: `2026-04-21T14:40:28.003287+00:00`
- Commit: `f8c07b8` on `codex/zig-baseline-qwen36`
- Dirty tree: `False`
- Subject: r004: parallel engine row projections
- Label: r004
- Notes: parallel engine row projections

## Latest Metrics

| Suite | Metric | Value | Delta vs previous same suite |
|---|---|---:|---:|
| Fresh Full Token Pass | `fresh_token_tok_per_s` | 0.57953 tok/s | +0.06544 |
| Cached Decode | `cached_decode_tok_per_s` | 0.54272 tok/s | +0.05668 |

## Best So Far

| Suite | Best | Commit | Time | Label |
|---|---:|---|---|---|
| Logits Head Matvec | 15.475 matvec/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Block 0 QKV Projection | 138.39 projection/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Fresh Full Token Pass | 0.57953 tok/s | `f8c07b8` | `2026-04-21T14:40:28.003287+00:00` | r004 |
| Cached Decode | 0.54272 tok/s | `f8c07b8` | `2026-04-21T14:40:28.003287+00:00` | r004 |

## Recent Runs

| Run | Commit | Label | Cached tok/s | Fresh tok/s | QKV proj/s | Logits matvec/s |
|---|---|---|---:|---:|---:|---:|
| `20260421T144028.003287Z-f8c07b8-decode` | `f8c07b8` | r004 | 0.54272 | 0.57953 |  |  |
| `20260421T143848.472187Z-573105b-full` | `573105b` | r003 | 0.48605 | 0.51409 | 138.39 | 15.475 |
| `20260421T142203.570145Z-333098c-decode` | `333098c` | r002 | 0.33362 | 0.42725 |  |  |
| `20260421T142012.280840Z-d17380f-decode` | `d17380f` | r001 | 0.40690 | 0.43082 |  |  |
| `20260421T142002.785445Z-11a5352-micro-dirty` | `11a5352` | releasefast harness sanity |  |  | 71.721 | 2.2272 |
| `20260421T141730.141807Z-11a5352-decode-dirty` | `11a5352` | releasefast check | 0.40603 | 0.42981 |  |  |
| `20260421T141730.141804Z-11a5352-micro-dirty` | `11a5352` | releasefast check |  |  | 71.522 | 2.1679 |
| `20260421T141434.727674Z-11a5352-micro` | `11a5352` | pre-round sanity |  |  | 6.3511 | 0.18395 |
| `20260421T141434.727661Z-11a5352-decode` | `11a5352` | pre-round sanity | 0.04486 | 0.04898 |  |  |
| `20260421T134736.065066+0000-edd131e-full-dirty` | `edd131e` | baseline before optimization | 0.04749 | 0.04881 | 6.2632 | 0.18680 |
