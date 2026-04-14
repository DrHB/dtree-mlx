"""Measure verify forward cost vs seq_len at fixed past=256."""
from __future__ import annotations
import time
import mlx.core as mx
from dtree_mlx.api import DFlashGenerator

PAST = 256
SEQ_LENS = [1, 2, 4, 8, 16, 24, 32, 48, 64]
REPEATS = 30
PROMPT = "Natalia sold clips to 48 of her friends in April. How many clips did Natalia sell altogether?"

gen = DFlashGenerator()
toks = gen.encode_prompt(PROMPT).tolist()
while len(toks) < PAST + max(SEQ_LENS) + 1:
    toks = toks + toks
prompt_ids = mx.array(toks[:PAST], dtype=mx.uint32)[None]

def bench(seq_len: int, use_tree: bool) -> float:
    cache = gen.target.make_cache()
    gen.target.forward_with_hidden_states(prompt_ids, cache, gen.draft.target_layer_ids)
    mx.eval(cache[0].keys)
    decode_ids = mx.array([toks[PAST]] * seq_len, dtype=mx.uint32)[None]
    if use_tree:
        pos = mx.array(list(range(PAST, PAST + seq_len)), dtype=mx.uint32)[None]
        past_mask = mx.ones((seq_len, PAST), dtype=mx.bool_)
        tri = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        mask = mx.concatenate([past_mask, tri], axis=1)
    # warmup
    if use_tree:
        out, _ = gen.target.forward_tree_with_hidden_states(decode_ids, cache, gen.draft.target_layer_ids, position_ids=pos, attention_mask=mask)
    else:
        out, _ = gen.target.forward_with_hidden_states(decode_ids, cache, gen.draft.target_layer_ids)
    mx.eval(out)
    gen.target.compact_kv_caches(cache, past_length=PAST, keep_current_indices=[])

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        if use_tree:
            out, _ = gen.target.forward_tree_with_hidden_states(decode_ids, cache, gen.draft.target_layer_ids, position_ids=pos, attention_mask=mask)
        else:
            out, _ = gen.target.forward_with_hidden_states(decode_ids, cache, gen.draft.target_layer_ids)
        mx.eval(out)
        gen.target.compact_kv_caches(cache, past_length=PAST, keep_current_indices=[])
    return (time.perf_counter() - t0) * 1000 / REPEATS

print(f"{'seq':>4} {'std_ms':>8} {'tree_ms':>8}")
for s in SEQ_LENS:
    a = bench(s, False)
    b = bench(s, True)
    print(f"{s:>4} {a:>8.2f} {b:>8.2f}")
