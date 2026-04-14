"""Diagnose DFlash-vs-DTree token divergence on 'def fibonacci(n):'.

Compares three paths at temperature=0:
  1. Pure greedy (mlx_lm.generate, no speculation)
  2. DFlash (chain speculation)
  3. DTree  (tree speculation, budget=24)

If pure greedy == DFlash != DTree: DTree bug
If pure greedy == DTree != DFlash: DFlash bug
If pure greedy != DFlash != DTree: FP non-determinism (kernel shape sensitivity)
"""
from __future__ import annotations
import mlx.core as mx
from dtree_mlx.api import DFlashGenerator

PROMPT = "def fibonacci(n):"
MAX_NEW = 64

gen = DFlashGenerator()

# Pure greedy via a single-token loop on the target model (no draft involvement).
# This is the ground truth that both spec methods should match.
tokenizer = gen.target.tokenizer
model = gen.target.model
prompt_tokens = gen.encode_prompt(PROMPT)

def pure_greedy():
    cache = gen.target.make_cache()
    logits, _ = gen.target.forward_with_hidden_states(prompt_tokens[None], cache, gen.draft.target_layer_ids)
    mx.eval(logits)
    next_tok = int(mx.argmax(logits[0, -1, :]).item())
    out = [next_tok]
    for _ in range(MAX_NEW - 1):
        ids = mx.array([next_tok], dtype=mx.uint32)[None]
        logits, _ = gen.target.forward_with_hidden_states(ids, cache, gen.draft.target_layer_ids)
        mx.eval(logits)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        out.append(next_tok)
    return out

greedy_tokens = pure_greedy()

# Warmup
gen.generate(PROMPT, max_new_tokens=8, temperature=0.0, decode_mode="dflash", speculative_tokens=16)
gen.generate(PROMPT, max_new_tokens=8, temperature=0.0, decode_mode="dtree", speculative_tokens=16, tree_budget=24)

r_dflash = gen.generate(PROMPT, max_new_tokens=MAX_NEW, temperature=0.0, decode_mode="dflash", speculative_tokens=16)
r_dtree  = gen.generate(PROMPT, max_new_tokens=MAX_NEW, temperature=0.0, decode_mode="dtree",  speculative_tokens=16, tree_budget=24)

dflash = list(r_dflash.generated_tokens)
dtree = list(r_dtree.generated_tokens)
greedy = greedy_tokens[:len(dflash)]

def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))

print(f"greedy len={len(greedy)}  dflash len={len(dflash)}  dtree len={len(dtree)}")
print(f"greedy vs dflash: first diff at index {first_diff(greedy, dflash)}")
print(f"greedy vs dtree : first diff at index {first_diff(greedy, dtree)}")
print(f"dflash vs dtree : first diff at index {first_diff(dflash, dtree)}")

k = first_diff(greedy, dflash)
if k < len(greedy):
    print(f"\nAt step {k}:")
    print(f"  greedy: {greedy[k]} = {tokenizer.decode([greedy[k]])!r}")
    print(f"  dflash: {dflash[k] if k < len(dflash) else '?'} = {tokenizer.decode([dflash[k]]) if k<len(dflash) else '?'!r}")
    print(f"  dtree : {dtree[k] if k < len(dtree) else '?'} = {tokenizer.decode([dtree[k]]) if k<len(dtree) else '?'!r}")

k2 = first_diff(greedy, dtree)
if k2 < len(greedy) and k2 != k:
    print(f"\nAt step {k2}:")
    print(f"  greedy: {greedy[k2]} = {tokenizer.decode([greedy[k2]])!r}")
    print(f"  dflash: {dflash[k2] if k2 < len(dflash) else '?'} = {tokenizer.decode([dflash[k2]]) if k2<len(dflash) else '?'!r}")
    print(f"  dtree : {dtree[k2] if k2 < len(dtree) else '?'} = {tokenizer.decode([dtree[k2]]) if k2<len(dtree) else '?'!r}")

print("\n--- decoded texts ---")
print(f"greedy: {tokenizer.decode(greedy)}")
print(f"dflash: {tokenizer.decode(dflash)}")
print(f"dtree : {tokenizer.decode(dtree)}")
