"""Micro-benchmark: where does verify-forward time actually go?

Compares:
  A) Standard Qwen3 forward (scalar offset RoPE) at seq_len=33, past=256
  B) Tree forward (per-token RoPE reshape path)     at seq_len=33, past=256

If A << B, RoPE reshape is the bottleneck. If A ~= B, the cost is intrinsic
to running 33-token forward with kv-cache of 256 and we need another lever.
"""
from __future__ import annotations
import time
import mlx.core as mx
from dtree_mlx.api import DFlashGenerator

PROMPT = "Natalia sold clips to 48 of her friends in April. How many clips did Natalia sell altogether?"
SEQ_LEN = 33  # matches tree_budget=32 + root
PAST = 256
REPEATS = 50

gen = DFlashGenerator()
model = gen.target.model

# Build a synthetic PAST-length token sequence by repeating a real prompt's encoding
toks = gen.encode_prompt(PROMPT).tolist()
while len(toks) < PAST + SEQ_LEN + 1:
    toks = toks + toks
toks = toks[: PAST + SEQ_LEN + 1]

def make_cache():
    return gen.target.make_cache()

prompt_ids = mx.array(toks[:PAST], dtype=mx.uint32)[None]

decode_ids = mx.array([toks[PAST]] * SEQ_LEN, dtype=mx.uint32)[None]
position_ids = mx.array(list(range(PAST, PAST + SEQ_LEN)), dtype=mx.uint32)[None]
# Tree attention mask: causal over the 33 tokens + full over past
past_mask = mx.ones((SEQ_LEN, PAST), dtype=mx.bool_)
tri = mx.tril(mx.ones((SEQ_LEN, SEQ_LEN), dtype=mx.bool_))
attention_mask = mx.concatenate([past_mask, tri], axis=1)

# A) standard forward (no position_ids, no mask) — cache auto-advances
def run_a():
    local_cache = make_cache()
    # re-prefill so cache has PAST tokens
    gen.target.forward_with_hidden_states(prompt_ids, local_cache, gen.draft.target_layer_ids)
    mx.eval(local_cache[0].keys)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        out, _ = gen.target.forward_with_hidden_states(decode_ids, local_cache, gen.draft.target_layer_ids)
        mx.eval(out)
        # trim back so we measure same state each iter
        gen.target.compact_kv_caches(local_cache, past_length=PAST, keep_current_indices=[])
    return time.perf_counter() - t0

# B) tree forward (per-token rope path)
def run_b():
    local_cache = make_cache()
    gen.target.forward_with_hidden_states(prompt_ids, local_cache, gen.draft.target_layer_ids)
    mx.eval(local_cache[0].keys)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        out, _ = gen.target.forward_tree_with_hidden_states(
            decode_ids, local_cache, gen.draft.target_layer_ids,
            position_ids=position_ids, attention_mask=attention_mask,
        )
        mx.eval(out)
        gen.target.compact_kv_caches(local_cache, past_length=PAST, keep_current_indices=[])
    return time.perf_counter() - t0

# warmup both
run_a(); run_b()

ta = run_a()
tb = run_b()
print(f"\nseq_len={SEQ_LEN} past={PAST} repeats={REPEATS}")
print(f"A (standard forward, scalar-offset RoPE): {ta*1000/REPEATS:.2f} ms/call")
print(f"B (tree forward, per-token RoPE path):    {tb*1000/REPEATS:.2f} ms/call")
print(f"B / A: {tb/ta:.2f}x")
