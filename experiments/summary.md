# Zig Experiment Summary

This file is generated from `experiments/results.csv`.

## Latest Run

- Run: `20260421T153351.012228Z-1d19296-metal`
- Time: `2026-04-21T15:33:51.012228+00:00`
- Commit: `1d19296` on `codex/zig-baseline-qwen36`
- Dirty tree: `False`
- Subject: r009: tile metal qkv projection across four rows
- Label: r009
- Notes: tile metal qkv projection across four rows

## Latest Metrics

| Suite | Metric | Value | Delta vs previous same suite |
|---|---|---:|---:|
| Metal Add-One | `metal_elements_per_s` | 4191788926.64 elements/s | +1663271926.40 |
| Metal QKV Projection | `metal_projection_passes_per_s` | 1682.37 projection/s | +69.596 |

## Best So Far

| Suite | Best | Commit | Time | Label |
|---|---:|---|---|---|
| Metal Add-One | 4642798317.47 elements/s | `2bed129` | `2026-04-21T15:07:06.488718+00:00` | r006 |
| Metal QKV Projection | 1682.37 projection/s | `1d19296` | `2026-04-21T15:33:51.012228+00:00` | r009 |
| Logits Head Matvec | 15.475 matvec/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Block 0 QKV Projection | 138.39 projection/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Fresh Full Token Pass | 0.64510 tok/s | `c4e45fa` | `2026-04-21T14:41:57.327290+00:00` | r005 |
| Cached Decode | 0.59878 tok/s | `c4e45fa` | `2026-04-21T14:41:57.327290+00:00` | r005 |

## Recent Runs

| Run | Commit | Label | Metal elems/s | Metal proj/s | Cached tok/s | Fresh tok/s | QKV proj/s | Logits matvec/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `20260421T153351.012228Z-1d19296-metal` | `1d19296` | r009 | 4191788926.64 | 1682.37 |  |  |  |  |
| `20260421T153148.630891Z-41d964b-metal` | `41d964b` | r008 | 2528517000.24 | 1612.77 |  |  |  |  |
| `20260421T151827.907314Z-4b6c733-metal` | `4b6c733` | r007 | 2342401429.69 | 1282.30 |  |  |  |  |
| `20260421T150706.488718Z-2bed129-metal` | `2bed129` | r006 | 4642798317.47 |  |  |  |  |  |
| `20260421T144157.327290Z-c4e45fa-decode` | `c4e45fa` | r005 |  |  | 0.59878 | 0.64510 |  |  |
| `20260421T144028.003287Z-f8c07b8-decode` | `f8c07b8` | r004 |  |  | 0.54272 | 0.57953 |  |  |
| `20260421T143848.472187Z-573105b-full` | `573105b` | r003 |  |  | 0.48605 | 0.51409 | 138.39 | 15.475 |
| `20260421T142203.570145Z-333098c-decode` | `333098c` | r002 |  |  | 0.33362 | 0.42725 |  |  |
| `20260421T142012.280840Z-d17380f-decode` | `d17380f` | r001 |  |  | 0.40690 | 0.43082 |  |  |
| `20260421T142002.785445Z-11a5352-micro-dirty` | `11a5352` | releasefast harness sanity |  |  |  |  | 71.721 | 2.2272 |
