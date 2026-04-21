# Zig Experiment Summary

This file is generated from `experiments/results.csv`.

## Latest Run

- Run: `20260421T161635.177790Z-b5e359b-decode`
- Time: `2026-04-21T16:16:35.177790+00:00`
- Commit: `b5e359b` on `codex/zig-baseline-qwen36`
- Dirty tree: `False`
- Subject: r013: expand metal cache to decode projections
- Label: r013
- Notes: expand metal cache to decode projections

## Latest Metrics

| Suite | Metric | Value | Delta vs previous same suite |
|---|---|---:|---:|
| Fresh Full Token Pass | `fresh_token_tok_per_s` | 0.84203 tok/s | +0.15587 |
| Cached Decode | `cached_decode_tok_per_s` | 0.78079 tok/s | +0.13712 |

## Best So Far

| Suite | Best | Commit | Time | Label |
|---|---:|---|---|---|
| Metal Add-One | 4642798317.47 elements/s | `2bed129` | `2026-04-21T15:07:06.488718+00:00` | r006 |
| Metal QKV Projection | 2262.96 projection/s | `4c0566e` | `2026-04-21T15:39:43.916627+00:00` | r011 |
| Metal Logits Projection | 187.30 projection/s | `4c0566e` | `2026-04-21T15:39:43.916627+00:00` | r011 |
| Logits Head Matvec | 15.475 matvec/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Block 0 QKV Projection | 138.39 projection/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Fresh Full Token Pass | 0.84203 tok/s | `b5e359b` | `2026-04-21T16:16:35.177790+00:00` | r013 |
| Cached Decode | 0.78079 tok/s | `b5e359b` | `2026-04-21T16:16:35.177790+00:00` | r013 |

## Recent Runs

| Run | Commit | Label | Metal elems/s | Metal qkv/s | Metal logits/s | Cached tok/s | Fresh tok/s | QKV proj/s | Logits matvec/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `20260421T161635.177790Z-b5e359b-decode` | `b5e359b` | r013 |  |  |  | 0.78079 | 0.84203 |  |  |
| `20260421T160847.061448Z-0b22f76-decode` | `0b22f76` | r012 |  |  |  | 0.64367 | 0.68616 |  |  |
| `20260421T153943.916627Z-4c0566e-metal` | `4c0566e` | r011 | 3008826398.85 | 2262.96 | 187.30 |  |  |  |  |
| `20260421T153450.306930Z-a074a09-metal` | `a074a09` | r010 | 2997215949.69 | 1965.80 | 183.59 |  |  |  |  |
| `20260421T153351.012228Z-1d19296-metal` | `1d19296` | r009 | 4191788926.64 | 1682.37 |  |  |  |  |  |
| `20260421T153148.630891Z-41d964b-metal` | `41d964b` | r008 | 2528517000.24 | 1612.77 |  |  |  |  |  |
| `20260421T151827.907314Z-4b6c733-metal` | `4b6c733` | r007 | 2342401429.69 | 1282.30 |  |  |  |  |  |
| `20260421T150706.488718Z-2bed129-metal` | `2bed129` | r006 | 4642798317.47 |  |  |  |  |  |  |
| `20260421T144157.327290Z-c4e45fa-decode` | `c4e45fa` | r005 |  |  |  | 0.59878 | 0.64510 |  |  |
| `20260421T144028.003287Z-f8c07b8-decode` | `f8c07b8` | r004 |  |  |  | 0.54272 | 0.57953 |  |  |
