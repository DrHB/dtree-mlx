# Zig Experiment Summary

This file is generated from `experiments/results.csv`.

## Latest Run

- Run: `20260421T142012.280840Z-d17380f-decode`
- Time: `2026-04-21T14:20:12.280840+00:00`
- Commit: `d17380f` on `codex/zig-baseline-qwen36`
- Dirty tree: `False`
- Subject: r001: force releasefast benchmark path
- Label: r001
- Notes: force releasefast benchmark path

## Latest Metrics

| Suite | Metric | Value | Delta vs previous same suite |
|---|---|---:|---:|
| Fresh Full Token Pass | `fresh_token_tok_per_s` | 0.43082 tok/s | +0.00102 |
| Cached Decode | `cached_decode_tok_per_s` | 0.40690 tok/s | +0.00087 |

## Best So Far

| Suite | Best | Commit | Time | Label |
|---|---:|---|---|---|
| Logits Head Matvec | 2.2272 matvec/s | `11a5352` | `2026-04-21T14:20:02.785445+00:00` | releasefast harness sanity |
| Block 0 QKV Projection | 71.721 projection/s | `11a5352` | `2026-04-21T14:20:02.785445+00:00` | releasefast harness sanity |
| Fresh Full Token Pass | 0.43082 tok/s | `d17380f` | `2026-04-21T14:20:12.280840+00:00` | r001 |
| Cached Decode | 0.40690 tok/s | `d17380f` | `2026-04-21T14:20:12.280840+00:00` | r001 |

## Recent Runs

| Run | Commit | Label | Cached tok/s | Fresh tok/s | QKV proj/s | Logits matvec/s |
|---|---|---|---:|---:|---:|---:|
| `20260421T142012.280840Z-d17380f-decode` | `d17380f` | r001 | 0.40690 | 0.43082 |  |  |
| `20260421T142002.785445Z-11a5352-micro-dirty` | `11a5352` | releasefast harness sanity |  |  | 71.721 | 2.2272 |
| `20260421T141730.141807Z-11a5352-decode-dirty` | `11a5352` | releasefast check | 0.40603 | 0.42981 |  |  |
| `20260421T141730.141804Z-11a5352-micro-dirty` | `11a5352` | releasefast check |  |  | 71.522 | 2.1679 |
| `20260421T141434.727674Z-11a5352-micro` | `11a5352` | pre-round sanity |  |  | 6.3511 | 0.18395 |
| `20260421T141434.727661Z-11a5352-decode` | `11a5352` | pre-round sanity | 0.04486 | 0.04898 |  |  |
| `20260421T134736.065066+0000-edd131e-full-dirty` | `edd131e` | baseline before optimization | 0.04749 | 0.04881 | 6.2632 | 0.18680 |
