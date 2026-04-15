"""Profile the DTree verifier breakdown on the current best fixed setting."""
from __future__ import annotations

from dtree_mlx.api import DFlashGenerator

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
]
MAX_NEW_TOKENS = 256
SPECULATIVE = 16
TREE_BUDGET = 24

TOP_LEVEL_KEYS = [
    "draft_time_s",
    "tree_build_time_s",
    "tree_compile_time_s",
    "verify_time_s",
    "bookkeeping_time_s",
    "unattributed_decode_time_s",
]
VERIFY_KEYS = [
    "verify_tree_forward_time_s",
    "verify_tree_logits_time_s",
    "verify_tree_argmax_time_s",
    "verify_tree_sample_time_s",
]
BOOKKEEPING_KEYS = [
    "bookkeeping_follow_tree_time_s",
    "bookkeeping_cache_compact_time_s",
    "bookkeeping_hidden_select_time_s",
    "bookkeeping_output_commit_time_s",
]


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


gen = DFlashGenerator()
gen.generate(
    PROMPTS[0],
    max_new_tokens=64,
    decode_mode="dtree",
    speculative_tokens=SPECULATIVE,
    tree_budget=TREE_BUDGET,
)

totals = {key: 0.0 for key in TOP_LEVEL_KEYS + VERIFY_KEYS + BOOKKEEPING_KEYS}
decode_times: list[float] = []
gen_tps: list[float] = []
e2e_tps: list[float] = []
accept: list[float] = []
tree_nodes: list[float] = []

for prompt in PROMPTS:
    result = gen.generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        decode_mode="dtree",
        speculative_tokens=SPECULATIVE,
        tree_budget=TREE_BUDGET,
        profile=True,
    )
    metrics = result.metrics
    profile = metrics["profile"]
    for key in totals:
        totals[key] += profile.get(key, 0.0)
    decode_times.append(metrics["decode_time_s"])
    gen_tps.append(metrics["generation_tps"])
    e2e_tps.append(metrics["end_to_end_tps"])
    accept.append(metrics["avg_acceptance_length"])
    tree_nodes.append(metrics["avg_verified_tree_nodes"])

decode_total = sum(decode_times)
verify_total = totals["verify_time_s"] or 1.0
book_total = totals["bookkeeping_time_s"] or 1.0

print(
    f"DTree verifier breakdown: spec={SPECULATIVE} tree_budget={TREE_BUDGET} "
    f"prompts={len(PROMPTS)}"
)
print(
    f"mean gen_tps={mean(gen_tps):.2f} e2e_tps={mean(e2e_tps):.2f} "
    f"accept={mean(accept):.2f} tree_nodes={mean(tree_nodes):.2f}"
)
print()
print("Top-level decode share")
for key in TOP_LEVEL_KEYS:
    print(f"  {key:32s} {100.0 * totals[key] / max(decode_total, 1e-9):6.1f}%")
print()
print("Verifier share")
for key in VERIFY_KEYS:
    print(
        f"  {key:32s} {100.0 * totals[key] / verify_total:6.1f}% "
        f"({100.0 * totals[key] / max(decode_total, 1e-9):5.1f}% decode)"
    )
print()
print("Bookkeeping share")
for key in BOOKKEEPING_KEYS:
    print(
        f"  {key:32s} {100.0 * totals[key] / book_total:6.1f}% "
        f"({100.0 * totals[key] / max(decode_total, 1e-9):5.2f}% decode)"
    )
