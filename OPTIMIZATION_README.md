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

| Track | Setting | Gen TPS | End-to-end TPS | Mean accept | Decision |
|---|---|---:|---:|---:|---|
| Qwen3 baseline | 8 gsm8k prompts, 2 warmups, plain | 36.42 | 34.98 | — | Reference |
| Qwen3 baseline | 8 gsm8k prompts, 2 warmups, DFlash `parallel-greedy-argmax` | 51.87 | 50.31 | 5.71 | Keep |
| Qwen3 baseline | 8 gsm8k prompts, 2 warmups, DTree `spec=16 tree_budget=24` | 56.68 | 54.83 | 7.17 | Keep |
| Qwen3 q4 | 8 gsm8k prompts, DFlash `q4_g64` | — | 47.36 | — | Opt-in only |
| Qwen3 q4 | 8 gsm8k prompts, DTree `q4_g64` | — | 56.09 | — | Opt-in only |
| Qwen3.5-4B | Janet, DFlash | 48.27 | 45.30 | 5.12 | Reference |
| Qwen3.5-4B | Janet, DTree lazy bf16 `spec=8 tree_budget=8` | 29.39 | 28.19 | 6.33 | Better, still behind DFlash |
| Qwen3.5-4B | 8 gsm8k prompts, DFlash `q4_g64 spec=16` | 47.39 | 45.07 | 5.81 | Reference |
| Qwen3.5-4B | 8 gsm8k prompts, DTree lazy `q4_g64 spec=16 tree_budget=24` | 50.55 | 48.31 | 6.95 | Keep |
| Qwen3.5-9B | Janet, DFlash | 20.27 | 19.38 | 4.03 | Keep |

| Area | Experiment | Change | Result | Decision |
|---|---|---|---|---|
| Plumbing | Benchmark warmup and ordering | Fixed prompt warmup handling and alternating compare order | Removed misleading single-prompt numbers | Keep |
| Qwen3 DFlash | Fast exact verifier | `parallel-greedy-argmax` instead of `parallel-replay` | Large speedup on Qwen3 at `temperature=0` | Keep |
| Qwen3 DTree | Fixed budget sweep | Swept fixed budgets under `b16` clamp | `tree_budget=24` was best local fixed point | Keep |
| Qwen3 target | Target quantization | `--target-quant-bits 4 --target-quant-group-size 64` | Helped DTree, hurt DFlash | Keep opt-in |
| Qwen3 target | 8-bit target quantization | `--target-quant-bits 8` | Worse than bf16 / q4 path | Drop |
| Qwen3 DTree | Adaptive tree budget | Dynamic budget by confidence | Slower than fixed budget | Drop |
| Qwen3 DTree | Hybrid DFlash/DTree routing | Switched between linear and tree verify | Slower than fixed modes | Drop |
| Qwen3 DTree | Depth-penalized scoring | Penalized deeper branches | No broad win | Drop |
| Qwen3 DTree | Wider candidate pools | Increased branch candidate pool | No broad win | Drop |
| Qwen3 DTree | Single-fork tree | Narrower tree shape | Worse | Drop |
| Qwen3 DTree | DTree greedy verifier | Honored `parallel-greedy-argmax` in tree path | Exact, but not faster | Drop |
| Qwen3 runtime | Hidden-state concat cleanup | Avoided some eager concatenation | Flat | Drop |
| Qwen3 runtime | Broader MLX verifier fusion | More aggressive compiled verifier path | Flat to worse | Drop |
| Qwen3.5 support | Qwen3.5 adapter port | Added `qwen3_5` DFlash support | Functional 4B and 9B path | Keep |
| Qwen3.5 DFlash | Draft attention mask | Compared `none` vs `causal` | `none` is better | Keep |
| Qwen3.5 DFlash | Imported hybrid hooks | Ported rollback hooks, split attention, context-only draft cache from `bstnxbt/dflash-mlx` | Helped, especially on long prefixes | Keep |
| Qwen3.5 DFlash | Draft KV precompute | Precomputed context K/V and query-only draft attention | Correct, basically flat | Keep as cleanup |
| Qwen3.5 DFlash | Fused all-layer draft K/V | Tried more aggressive fused draft projection | Worse | Drop |
| Qwen3.5 DFlash | 9B target-side hook import | Imported 9B hybrid-target tricks | Short-prompt gain modest, long-prefix path better | Keep |
| Qwen3.5 DTree | Initial correctness-first tree | Full-tree exact verifier over hybrid caches | Exact, very slow | Replaced |
| Qwen3.5 DTree | Full-attention tree batching | Ran full-attention tree layers over whole tree block | Real win, but not enough alone | Keep in full-tree path |
| Qwen3.5 DTree | Compiled recurrent breadth path | Compiled breadth-by-depth recurrent tree layer | Flat to worse | Drop |
| Qwen3.5 DTree | Sequential exact tree | Walked real cache state branch-by-branch | Modest gain, memory-heavy | Drop |
| Qwen3.5 DTree | Lazy exact verifier | Verified only the branch the target follows | First real 4B DTree win on broader speed slice | Keep default |
| Qwen3.5 DTree | Full-tree fallback | `DTREE_QWEN35_TREE_MODE=full_tree` | Useful for comparison only | Keep fallback |
| Qwen3.5 DTree | Small-tree q4 setting | `q4_g64 spec=16 tree_budget=2` | Won short Janet, lost broader sweep | Drop |
| Qwen3.5 DTree | Lazy-budget sweep | Lazy path with `spec=16` and larger budgets | `tree_budget=24` best local point; bigger trees fell back | Keep `24` |
| Qwen3.5 DTree | q4 correctness sanity | `N=12`, `q4_g64 spec=16 tree_budget=24` | Plain `11/12`, DFlash `12/12`, DTree lazy `11/12` | Speed win, not accuracy win |
| Qwen3.5 DTree | Short greedy exactness | 24-token q4 greedy check vs DFlash | Token-for-token match | Keep confidence |
| Qwen3.5 DTree | Tree verified-node accounting | Lazy path counts verified nodes on followed path | Verified nodes now track acceptance length instead of fixed `tree_budget + 1` | Keep |

| Validation | Setting | Result | Decision |
|---|---|---|---|
| Qwen3 q4 sanity | `N=12`, `max_new_tokens=512` | bf16: plain `9/12`, DFlash `10/12`, DTree `11/12`; q4: plain `11/12`, DFlash `12/12`, DTree `11/12` | q4 stays opt-in |
| Qwen3.5-4B speed | 8 gsm8k prompts, `q4_g64 spec=16 tree_budget=24` | DFlash `45.07` e2e TPS, DTree lazy `48.31` e2e TPS | First clean 4B DTree speed win |
| Qwen3.5-4B correctness | `N=12`, `q4_g64 spec=16 tree_budget=24` | DFlash `12/12`, DTree lazy `11/12` | Needs larger validation |

| Open lead | Why it still matters | Status |
|---|---|---|
| Larger Qwen3.5-4B correctness sweep | Current lazy q4 DTree wins on speed but not on the small accuracy slice | Next useful validation |
| Better lazy-tree scoring | Lazy verification makes bigger trees cheaper; better branch ranking may raise acceptance more | Open |
| Faster Qwen3.5 top-1 verifier | Could still reduce lazy-path cost further | Open |
| Better draft than current `b16` | Acceptance is still limited by draft quality and block size | Open |
