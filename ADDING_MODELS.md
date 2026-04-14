# Adding Models

`dtree-mlx` only supports target models that have both:

- a matching DFlash draft checkpoint
- an MLX adapter that exposes the verifier internals

Today the repo is centered on Qwen3-4B. Adding a new pair means wiring the target adapter first, then validating exact greedy decoding.

## 1. Check the pair

Pick a target and matching draft from the upstream DFlash list:

```text
target model + z-lab/<target>-DFlash
```

Inspect the target config:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
import json

path = snapshot_download("mlx-community/Qwen3-4B-bf16")
config = json.load(open(f"{path}/config.json"))
print(config["model_type"])
print(config.get("model_file"))
PY
```

If the `model_type` already maps cleanly onto an existing adapter, start there.

## 2. Add an adapter

Add a target adapter in [dtree_mlx/adapters.py](dtree_mlx/adapters.py) and register it in `ADAPTERS`.

The adapter needs to provide prompt building, stop tokens, cache handling, LM head access, verifier forward paths, and cache rewind / rollback helpers.

The key method is the one that returns verifier logits plus the hidden states required by the draft checkpoint.

## 3. Decide whether you need a custom model file

You usually do not need a custom MLX model fork for a normal decoder-only transformer with a standard KV cache.

You may need one if the target has cache state that cannot be rolled back generically, for example:

- hybrid attention plus linear attention
- recurrent state
- architecture-specific execution paths that MLX-LM does not expose cleanly

## 4. Validate exact greedy decoding

Before listing a new pair as supported, run a plain-vs-DFlash exact greedy smoke test on the same prompt:

```bash
uv run python - <<'PY'
from argparse import Namespace

from dtree_mlx import DFlashGenerator
from dtree_mlx.benchmark_cli import run_one_prompt

target = "<target>"
draft = "<draft>"
prompt = "Write a quicksort in Python."

runner = DFlashGenerator(target_model=target, draft_model=draft)
prompt_tokens = runner.encode_prompt(prompt)

dflash = runner.generate_from_tokens(
    prompt_tokens,
    max_new_tokens=64,
    temperature=0.0,
    verify_mode="parallel-replay",
)

plain_args = Namespace(
    max_new_tokens=64,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    min_p=0.0,
    min_tokens_to_keep=1,
)
plain = run_one_prompt(
    runner.target.model,
    runner.target.tokenizer,
    prompt_tokens.tolist(),
    plain_args,
)

if dflash.text != plain.output_text:
    raise SystemExit("FAIL: DFlash output differs from plain greedy target output.")
if dflash.metrics["avg_acceptance_length"] <= 0:
    raise SystemExit("FAIL: DFlash did not accept any draft tokens.")

print("PASS: exact greedy smoke test matched plain target output.")
PY
```

## 5. Update the docs

Only after the exact smoke test works:

- add the pair to the README
- add benchmark numbers
- note any target-specific caveats
