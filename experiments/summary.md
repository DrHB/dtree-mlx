# Zig Experiment Summary

This file is generated from `experiments/results.csv`.

## Latest Run

- Run: `20260421T210053.127388Z-036de75-tracked`
- Time: `2026-04-21T21:00:53.127388+00:00`
- Commit: `036de75` on `codex/zig-baseline-qwen36`
- Dirty tree: `False`
- Subject: r016: zero-copy q4k q6k metal projections
- Outcome: `accepted`
- Backend: `metal+metal-cache`
- Coverage: `metal-kernels+decode`
- Label: r016
- Notes: zero-copy q4k q6k metal projections

## Latest Metrics

| Suite | Backend | Coverage | Metric | Value | Delta vs previous same suite |
|---|---|---|---|---:|---:|
| Metal Add-One | `metal` | `metal-kernel` | `metal_elements_per_s` | 2837824086.60 elements/s | +367389406.43 |
| Metal QKV Projection | `metal` | `metal-kernel` | `metal_projection_passes_per_s` | 1365.00 projection/s | +179.11 |
| Metal Logits Projection | `metal` | `metal-kernel` | `metal_projection_passes_per_s` | 149.87 projection/s | +60.288 |
| Tracked Cached Decode | `metal-cache` | `tracked-decode` | `steady_decode_tok_per_s` | 0.93101 tok/s | +0.82517 |

## Best So Far

| Suite | Best | Commit | Time | Label |
|---|---:|---|---|---|
| Metal Add-One | 4642798317.47 elements/s | `2bed129` | `2026-04-21T15:07:06.488718+00:00` | r006 |
| Metal QKV Projection | 2262.96 projection/s | `4c0566e` | `2026-04-21T15:39:43.916627+00:00` | r011 |
| Metal Logits Projection | 187.30 projection/s | `4c0566e` | `2026-04-21T15:39:43.916627+00:00` | r011 |
| Tracked Cached Decode | 0.93101 tok/s | `036de75` | `2026-04-21T21:00:53.127388+00:00` | r016 |
| Logits Head Matvec | 15.475 matvec/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Block 0 QKV Projection | 138.39 projection/s | `573105b` | `2026-04-21T14:38:48.472187+00:00` | r003 |
| Fresh Full Token Pass | 1.1535 tok/s | `2d4b1f0` | `2026-04-21T17:14:45.237004+00:00` | r014 |
| Cached Decode | 1.0547 tok/s | `2d4b1f0` | `2026-04-21T17:14:45.237004+00:00` | r014 |

## Recent Runs

| Run | Outcome | Backend | Coverage | Commit | Label | Metal elems/s | Metal qkv/s | Metal logits/s | Tracked tok/s | Cached tok/s | Fresh tok/s | QKV proj/s | Logits matvec/s |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260421T210053.127388Z-036de75-tracked` | `accepted` | `metal+metal-cache` | `metal-kernels+decode` | `036de75` | r016 | 2837824086.60 | 1365.00 | 149.87 | 0.93101 | 0.93101 |  |  |  |
| `20260421T203639.102422Z-7931839-tracked` | `accepted` | `metal+metal-cache` | `metal-kernels+decode` | `7931839` | r015 | 2470434680.17 | 1185.89 | 89.581 | 0.10584 | 0.10584 |  |  |  |
| `20260421T171445.237004Z-2d4b1f0-decode` | `unreviewed` | `metal-cache` | `decode` | `2d4b1f0` | r014 |  |  |  |  | 1.0547 | 1.1535 |  |  |
| `20260421T161635.177790Z-b5e359b-decode` | `unreviewed` | `metal-cache` | `decode` | `b5e359b` | r013 |  |  |  |  | 0.78079 | 0.84203 |  |  |
| `20260421T160847.061448Z-0b22f76-decode` | `unreviewed` | `metal-cache` | `decode` | `0b22f76` | r012 |  |  |  |  | 0.64367 | 0.68616 |  |  |
| `20260421T153943.916627Z-4c0566e-metal` | `unreviewed` | `metal` | `metal-kernels` | `4c0566e` | r011 | 3008826398.85 | 2262.96 | 187.30 |  |  |  |  |  |
| `20260421T153450.306930Z-a074a09-metal` | `unreviewed` | `metal` | `metal-kernels` | `a074a09` | r010 | 2997215949.69 | 1965.80 | 183.59 |  |  |  |  |  |
| `20260421T153351.012228Z-1d19296-metal` | `unreviewed` | `metal` | `metal-kernels` | `1d19296` | r009 | 4191788926.64 | 1682.37 |  |  |  |  |  |  |
| `20260421T153148.630891Z-41d964b-metal` | `unreviewed` | `metal` | `metal-kernels` | `41d964b` | r008 | 2528517000.24 | 1612.77 |  |  |  |  |  |  |
| `20260421T151827.907314Z-4b6c733-metal` | `unreviewed` | `metal` | `metal-kernels` | `4b6c733` | r007 | 2342401429.69 | 1282.30 |  |  |  |  |  |  |
