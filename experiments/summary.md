# Zig Experiment Summary

This file is generated from `experiments/results.csv`.

## Latest Run

- Run: `20260421T150706.488718Z-2bed129-metal`
- Time: `2026-04-21T15:07:06.488718+00:00`
- Commit: `2bed129` on `codex/zig-baseline-qwen36`
- Dirty tree: `False`
- Subject: r006: add metal benchmark harness
- Label: r006
- Notes: add metal benchmark harness

## Latest Metrics

| Suite | Metric | Value | Delta vs previous same suite |
|---|---|---:|---:|
| Metal Add-One | `metal_elements_per_s` | 4642798317.47 elements/s | n/a |

## Best So Far

| Suite | Best | Commit | Time | Label |
|---|---:|---|---|---|
| Metal Add-One | 4642798317.47 elements/s | `2bed129` | `2026-04-21T15:07:06.488718+00:00` | r006 |
| Logits Head Matvec | 15.475 matvec/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Block 0 QKV Projection | 138.39 projection/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Fresh Full Token Pass | 0.64510 tok/s | `c4e45fa` | `2026-04-21T14:41:57.327290+00:00` | r005 |
| Cached Decode | 0.59878 tok/s | `c4e45fa` | `2026-04-21T14:41:57.327290+00:00` | r005 |

## Recent Runs

| Run | Commit | Label | Metal elems/s | Cached tok/s | Fresh tok/s | QKV proj/s | Logits matvec/s |
|---|---|---|---:|---:|---:|---:|---:|
| `20260421T150706.488718Z-2bed129-metal` | `2bed129` | r006 | 4642798317.47 |  |  |  |  |
| `20260421T144157.327290Z-c4e45fa-decode` | `c4e45fa` | r005 |  | 0.59878 | 0.64510 |  |  |
| `20260421T144028.003287Z-f8c07b8-decode` | `f8c07b8` | r004 |  | 0.54272 | 0.57953 |  |  |
| `20260421T143848.472187Z-573105b-full` | `573105b` | r003 |  | 0.48605 | 0.51409 | 138.39 | 15.475 |
| `20260421T142203.570145Z-333098c-decode` | `333098c` | r002 |  | 0.33362 | 0.42725 |  |  |
| `20260421T142012.280840Z-d17380f-decode` | `d17380f` | r001 |  | 0.40690 | 0.43082 |  |  |
| `20260421T142002.785445Z-11a5352-micro-dirty` | `11a5352` | releasefast harness sanity |  |  |  | 71.721 | 2.2272 |
| `20260421T141730.141807Z-11a5352-decode-dirty` | `11a5352` | releasefast check |  | 0.40603 | 0.42981 |  |  |
| `20260421T141730.141804Z-11a5352-micro-dirty` | `11a5352` | releasefast check |  |  |  | 71.522 | 2.1679 |
| `20260421T141434.727674Z-11a5352-micro` | `11a5352` | pre-round sanity |  |  |  | 6.3511 | 0.18395 |
