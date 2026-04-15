# Optimization Tracker

| Field | Value |
|---|---|
| Date | 2026-04-15 |
| Hardware | Apple M2 Max, 32 GB |
| Main Qwen3 target | `mlx-community/Qwen3-4B-bf16` |
| Main Qwen3 draft | `z-lab/Qwen3-4B-DFlash-b16` |
| Main Qwen3.5 targets | `mlx-community/Qwen3.5-4B-bf16`, `mlx-community/Qwen3.5-9B-bf16` |
| Main Qwen3.5 drafts | `z-lab/Qwen3.5-4B-DFlash`, `z-lab/Qwen3.5-9B-DFlash` |
| Decode setting | `temperature=0` |
| Hard constraint | Current Qwen3 draft is `b16`, so runtime clamps `--speculative-tokens` to `16` |
| Primary score | `end_to_end_tps / matched_dflash_end_to_end_tps` when a head-to-head DFlash baseline exists |
| Commit policy | Historical rows may point to the commit that recorded or landed the result; new experiments should get their own commit |

| Track | Setting | Gen TPS | End-to-end TPS | Mean accept | Score | Decision |
|---|---|---:|---:|---:|---:|---|
| Qwen3 baseline | 8 gsm8k prompts, 2 warmups, plain | 36.42 | 34.98 | — | 0.70x | Reference |
| Qwen3 baseline | 8 gsm8k prompts, 2 warmups, DFlash `parallel-greedy-argmax` | 51.87 | 50.31 | 5.71 | 1.00x | Keep |
| Qwen3 baseline | 8 gsm8k prompts, 2 warmups, DTree `spec=16 tree_budget=24` | 56.68 | 54.83 | 7.17 | 1.09x | Keep |
| Qwen3 q4 | 8 gsm8k prompts, DFlash `q4_g64` | — | 47.36 | — | 1.00x | Opt-in only |
| Qwen3 q4 | 8 gsm8k prompts, DTree `q4_g64` | — | 56.09 | — | 1.18x | Opt-in only |
| Qwen3.5-4B | Janet, DFlash | 48.27 | 45.30 | 5.12 | 1.00x | Reference |
| Qwen3.5-4B | Janet, DTree lazy bf16 `spec=8 tree_budget=8` | 29.39 | 28.19 | 6.33 | 0.62x | Better, still behind DFlash |
| Qwen3.5-4B | 8 gsm8k prompts, DFlash `q4_g64 spec=16` | 47.39 | 45.07 | 5.81 | 1.00x | Reference |
| Qwen3.5-4B | 8 gsm8k prompts, DTree lazy `q4_g64 spec=16 tree_budget=24` | 50.55 | 48.31 | 6.95 | 1.07x | Keep |
| Qwen3.5-9B | Janet, DFlash | 20.27 | 19.38 | 4.03 | 1.00x | Keep |

| Commit | Area | Experiment | Change | Result | Score | Decision |
|---|---|---|---|---|---|---|
| `74db97d` | Plumbing | Benchmark warmup and ordering | Fixed prompt warmup handling and alternating compare order | Removed misleading single-prompt numbers | — | Keep |
| `5943b2d` | Qwen3 DFlash | Fast exact verifier | `parallel-greedy-argmax` instead of `parallel-replay` | Large speedup on Qwen3 at `temperature=0` | 1.00x | Keep |
| `068f6b9` | Qwen3 DTree | Fixed budget sweep | Swept fixed budgets under `b16` clamp | `tree_budget=24` was best local fixed point | 1.09x | Keep |
| `895f9ef` | Qwen3 target | Target quantization | `--target-quant-bits 4 --target-quant-group-size 64` | Helped DTree, hurt DFlash | 1.18x | Keep opt-in |
| `895f9ef` | Qwen3 target | 8-bit target quantization | `--target-quant-bits 8` | Worse than bf16 / q4 path | <1.00x | Drop |
| `local-only` | Qwen3 DTree | Adaptive tree budget | Dynamic budget by confidence | Slower than fixed budget | <1.00x | Drop |
| `local-only` | Qwen3 DTree | Hybrid DFlash/DTree routing | Switched between linear and tree verify | Slower than fixed modes | <1.00x | Drop |
| `local-only` | Qwen3 DTree | Depth-penalized scoring | Penalized deeper branches | No broad win | ~1.00x | Drop |
| `local-only` | Qwen3 DTree | Wider candidate pools | Increased branch candidate pool | No broad win | ~1.00x | Drop |
| `local-only` | Qwen3 DTree | Single-fork tree | Narrower tree shape | Worse | <1.00x | Drop |
| `a765a56` | Qwen3 DTree | DTree greedy verifier | Honored `parallel-greedy-argmax` in tree path | Exact, but not faster | <1.00x | Drop |
| `local-only` | Qwen3 runtime | Hidden-state concat cleanup | Avoided some eager concatenation | Flat | ~1.00x | Drop |
| `local-only` | Qwen3 runtime | Broader MLX verifier fusion | More aggressive compiled verifier path | Flat to worse | <=1.00x | Drop |
| `e6ea8b4` | Qwen3.5 support | Qwen3.5 adapter port | Added `qwen3_5` DFlash support | Functional 4B and 9B path | 1.00x | Keep |
| `e6ea8b4` | Qwen3.5 DFlash | Draft attention mask | Compared `none` vs `causal` | `none` is better | >1.00x | Keep |
| `f34a01f` | Qwen3.5 DFlash | Imported hybrid hooks | Ported rollback hooks, split attention, context-only draft cache from `bstnxbt/dflash-mlx` | Helped, especially on long prefixes | >1.00x on longer prefixes | Keep |
| `eb0d95c` | Qwen3.5 DFlash | Draft KV precompute | Precomputed context K/V and query-only draft attention | Correct, basically flat | ~1.00x | Keep as cleanup |
| `local-only` | Qwen3.5 DFlash | Fused all-layer draft K/V | Tried more aggressive fused draft projection | Worse | <1.00x | Drop |
| `f34a01f` | Qwen3.5 DFlash | 9B target-side hook import | Imported 9B hybrid-target tricks | Short-prompt gain modest, long-prefix path better | >1.00x on longer prefixes | Keep |
| `31c348e` | Qwen3.5 DTree | Initial correctness-first tree | Full-tree exact verifier over hybrid caches | Exact, very slow | ~0.38x | Replaced |
| `a3f43f0` | Qwen3.5 DTree | Full-attention tree batching | Ran full-attention tree layers over whole tree block | Real win, but not enough alone | ~0.47x | Keep in full-tree path |
| `local-only` | Qwen3.5 DTree | Compiled recurrent breadth path | Compiled breadth-by-depth recurrent tree layer | Flat to worse | <=1.00x | Drop |
| `local-only` | Qwen3.5 DTree | Sequential exact tree | Walked real cache state branch-by-branch | Modest gain, memory-heavy | ~0.79x on Janet q4 | Drop |
| `e286574` | Qwen3.5 DTree | Lazy exact verifier | Verified only the branch the target follows | First real 4B DTree win on broader speed slice | 1.07x | Keep default |
| `2d02291` | Qwen3.5 DTree | Wider lazy candidate pool | Let `tree_budget=24` search a wider draft top-k pool (`candidate_topk=48`) before heap pruning | 3-prompt and 8-prompt q4 sweeps kept the same acceptance (`7.06` and `6.85`), so the small TPS drift was noise rather than a real tree gain | ~1.00x, same acceptance | Drop |
| `87276b4` | Qwen3.5 DTree | Sibling bonus scoring | Added a shallow-depth bonus for sibling expansions to force more early breadth | Quick q4 screen at `0.5` and `1.0` kept acceptance essentially flat (`7.06` / `7.05`) and only moved TPS by noise-level amounts | ~1.02x short, no acceptance gain | Drop |
| `dc60b2f` | Qwen3.5 DTree | Chunked lazy spine verifier | Verified a short best-child spine, rolled back on divergence, then continued from the matched sibling | `lazy_chunk_size=4` looked good on 3 prompts (`1.08x`) but on the 8-prompt q4 slice it only moved DTree from `46.45` to `46.55` e2e TPS | 1.05x broad, not meaningful vs chunk1 | Drop |
| `7ded69c` | Qwen3.5 DTree | Compiled lazy tree step | Routed each lazy verifier hop through a compiled one-step Qwen3.5 path instead of per-hop `forward_dflash()` plumbing | On q4 `spec=16 tree_budget=24`, the 3-prompt slice regressed (`51.26 -> 50.43` e2e TPS) and the 8-prompt slice only moved DTree from `47.68` to `47.94` e2e TPS | 1.06x broad, ~flat vs prior lazy | Drop |
| `14bcc2c` | Qwen3.5 DTree | Rank-aware tree scoring | Added an online SEQUOIA-style rank-acceptance prior to bias sibling selection beyond raw draft joint probability | Quick q4 screen with `DTREE_TREE_SCORE_MODE=rank_accept` dropped DTree to `50.19` e2e TPS versus `50.43` on the current lazy baseline, with no acceptance gain | 1.01x short, worse than baseline | Drop |
| `e286574` | Qwen3.5 DTree | Full-tree fallback | `DTREE_QWEN35_TREE_MODE=full_tree` | Useful for comparison only | 0.62x | Keep fallback |
| `d61ba98` | Qwen3.5 DTree | Small-tree q4 setting | `q4_g64 spec=16 tree_budget=2` | Won short Janet, lost broader sweep | 1.05x short / 0.86x broad | Drop |
| `e286574` | Qwen3.5 DTree | Lazy-budget sweep | Lazy path with `spec=16` and larger budgets | `tree_budget=24` best local point; bigger trees fell back | 1.07x | Keep `24` |
| `e286574` | Qwen3.5 DTree | q4 correctness sanity | `N=12`, `q4_g64 spec=16 tree_budget=24` | Plain `11/12`, DFlash `12/12`, DTree lazy `11/12` | 1.07x speed / 0.92x acc | Speed win, not accuracy win |
| `e286574` | Qwen3.5 DTree | Short greedy exactness | 24-token q4 greedy check vs DFlash | Token-for-token match | Exactness confidence | Keep confidence |
| `e286574` | Qwen3.5 DTree | Tree verified-node accounting | Lazy path counts verified nodes on followed path | Verified nodes now track acceptance length instead of fixed `tree_budget + 1` | Lower verifier work | Keep |

| Validation | Setting | Result | Score | Decision |
|---|---|---|---|---|
| Qwen3 q4 sanity | `N=12`, `max_new_tokens=512` | bf16: plain `9/12`, DFlash `10/12`, DTree `11/12`; q4: plain `11/12`, DFlash `12/12`, DTree `11/12` | Mixed | q4 stays opt-in |
| Qwen3.5-4B speed | 8 gsm8k prompts, `q4_g64 spec=16 tree_budget=24` | DFlash `45.07` e2e TPS, DTree lazy `48.31` e2e TPS | 1.07x | First clean 4B DTree speed win |
| Qwen3.5-4B correctness | `N=12`, `q4_g64 spec=16 tree_budget=24` | DFlash `12/12`, DTree lazy `11/12` | 0.92x | Needs larger validation |

| Open lead | Why it still matters | Status |
|---|---|---|
| Larger Qwen3.5-4B correctness sweep | Current lazy q4 DTree wins on speed but not on the small accuracy slice | Next useful validation |
| Better lazy-tree scoring | Lazy verification makes bigger trees cheaper; better branch ranking may raise acceptance more | Open |
| Faster Qwen3.5 top-1 verifier | Could still reduce lazy-path cost further | Open |
| Better draft than current `b16` | Acceptance is still limited by draft quality and block size | Open |
