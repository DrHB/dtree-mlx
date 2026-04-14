from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from dtree_mlx.benchmark_cli import load_prompts, split_warmup_and_benchmark_prompts
from dtree_mlx.compare_cli import benchmark_mode_order


def test_load_prompts_repeats_explicit_prompt_for_warmup_and_measurement():
    args = Namespace(
        prompt="Explicit prompt",
        prompt_file=None,
        num_prompts=2,
        warmup_prompts=1,
        dataset="gsm8k",
        shuffle=False,
        seed=0,
    )

    prompts = load_prompts(args)

    assert prompts == ["Explicit prompt", "Explicit prompt", "Explicit prompt"]


def test_load_prompts_repeats_prompt_file_for_warmup_and_measurement(tmp_path: Path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("From file")
    args = Namespace(
        prompt=None,
        prompt_file=prompt_file,
        num_prompts=1,
        warmup_prompts=2,
        dataset="gsm8k",
        shuffle=False,
        seed=0,
    )

    prompts = load_prompts(args)

    assert prompts == ["From file", "From file", "From file"]


def test_split_warmup_and_benchmark_prompts_preserves_tail():
    warmup, benchmark = split_warmup_and_benchmark_prompts(
        ["p1", "p2", "p3"],
        warmup_prompts=1,
    )

    assert warmup == ["p1"]
    assert benchmark == ["p2", "p3"]


def test_split_warmup_and_benchmark_prompts_rejects_all_warmup():
    with pytest.raises(ValueError, match="No prompts left to benchmark after warmup."):
        split_warmup_and_benchmark_prompts(["only"], warmup_prompts=1)


def test_compare_cli_alternates_mode_order():
    assert benchmark_mode_order(1) == ("dflash", "dtree")
    assert benchmark_mode_order(2) == ("dtree", "dflash")
    assert benchmark_mode_order(3) == ("dflash", "dtree")
