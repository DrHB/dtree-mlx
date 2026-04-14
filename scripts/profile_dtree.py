"""Profile DTree stage timings across tree budgets."""
from __future__ import annotations

from dtree_mlx.api import DFlashGenerator

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
]
MAX_NEW_TOKENS = 256
SPECULATIVE = 16
BUDGETS = [8, 16, 24, 32, 48]

gen = DFlashGenerator()
# warmup both modes once
gen.generate(PROMPTS[0], max_new_tokens=64, decode_mode="dflash", speculative_tokens=SPECULATIVE)
gen.generate(PROMPTS[0], max_new_tokens=64, decode_mode="dtree", speculative_tokens=SPECULATIVE, tree_budget=16)

# DFlash baseline
dflash_e2e = []
dflash_gen = []
for p in PROMPTS:
    r = gen.generate(p, max_new_tokens=MAX_NEW_TOKENS, decode_mode="dflash", speculative_tokens=SPECULATIVE, profile=True)
    dflash_e2e.append(r.metrics["end_to_end_tps"])
    dflash_gen.append(r.metrics["generation_tps"])

print(f"\nDFLASH baseline: gen_tps={sum(dflash_gen)/len(dflash_gen):.2f} e2e_tps={sum(dflash_e2e)/len(dflash_e2e):.2f}\n")

# DTree sweep
print(f"{'budget':>7} {'gen_tps':>8} {'e2e_tps':>8} {'accept':>7} {'draft':>7} {'build':>7} {'cmpl':>7} {'verify':>7} {'book':>7} {'unatr':>7}")
for b in BUDGETS:
    totals = {k: 0.0 for k in ["draft_time_s","tree_build_time_s","tree_compile_time_s","verify_time_s","bookkeeping_time_s","unattributed_decode_time_s"]}
    steps = 0
    e2e = []
    gtps = []
    accept = []
    for p in PROMPTS:
        r = gen.generate(p, max_new_tokens=MAX_NEW_TOKENS, decode_mode="dtree", speculative_tokens=SPECULATIVE, tree_budget=b, profile=True)
        prof = r.metrics["profile"]
        for k in totals:
            totals[k] += prof[k]
        steps += prof["steps"]
        e2e.append(r.metrics["end_to_end_tps"])
        gtps.append(r.metrics["generation_tps"])
        accept.append(r.metrics["avg_acceptance_length"])
    total = sum(totals.values()) or 1.0
    pct = {k: 100.0 * v / total for k, v in totals.items()}
    print(f"{b:>7} {sum(gtps)/len(gtps):>8.2f} {sum(e2e)/len(e2e):>8.2f} {sum(accept)/len(accept):>7.2f} "
          f"{pct['draft_time_s']:>6.1f}% {pct['tree_build_time_s']:>6.1f}% {pct['tree_compile_time_s']:>6.1f}% "
          f"{pct['verify_time_s']:>6.1f}% {pct['bookkeeping_time_s']:>6.1f}% {pct['unattributed_decode_time_s']:>6.1f}%")
