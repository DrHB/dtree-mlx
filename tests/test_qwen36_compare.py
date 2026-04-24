from __future__ import annotations

import sys
from pathlib import Path


class FakeTokenizer:
    eos_token_ids = [0]

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, tokens, skip_special_tokens: bool = False):
        del skip_special_tokens
        return "".join(chr(int(token)) for token in tokens)


def test_qwen36_compare_cli_defaults():
    from dtree_mlx import qwen36_compare_cli

    saved_argv = sys.argv
    try:
        sys.argv = [qwen36_compare_cli.__name__]
        args = qwen36_compare_cli.parse_args()
    finally:
        sys.argv = saved_argv

    assert args.target_model == "mlx-community/Qwen3.6-27B-4bit"
    assert args.ar_draft_model == "mlx-community/Qwen3-1.7B-4bit"
    assert args.dflash_draft_model == "z-lab/Qwen3.6-27B-DFlash"
    assert args.require_dflash_draft is True
    assert args.draft_max == 12
    assert args.draft_min == 3
    assert args.draft_p_min == 0.6
    assert args.tree_budget == 24
    assert args.methods == qwen36_compare_cli.DEFAULT_METHODS


def test_qwen36_compare_preflight_accepts_local_config(tmp_path: Path):
    from dtree_mlx.qwen36_compare_cli import resolve_or_check_model

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    assert resolve_or_check_model(str(model_dir), role="target") == str(model_dir)


def test_bridge_accepts_exact_extension():
    from dtree_mlx.ar_spec import bridge_draft_suffix_to_target

    tokenizer = FakeTokenizer()
    prefix_text = "hello"
    prefix_tokens = tokenizer.encode(prefix_text)
    result = bridge_draft_suffix_to_target(
        tokenizer,
        prefix_tokens,
        prefix_text,
        " world",
    )

    assert result.rejected is False
    assert result.target_suffix == tokenizer.encode(" world")


def test_bridge_rejects_non_prefix_extension():
    from dtree_mlx.ar_spec import bridge_draft_suffix_to_target

    tokenizer = FakeTokenizer()
    result = bridge_draft_suffix_to_target(
        tokenizer,
        [ord("x")],
        "hello",
        " world",
    )

    assert result.rejected is True
    assert result.target_suffix == []


def test_aggregate_metrics_sums_bridge_and_expansion_counts():
    from dtree_mlx.qwen36_compare_cli import aggregate_metrics

    aggregate = aggregate_metrics(
        "linear_spec_qwen17",
        [
            {
                "num_input_tokens": 10,
                "num_output_tokens": 5,
                "prefill_time_s": 1.0,
                "decode_time_s": 2.0,
                "total_time_s": 3.0,
                "generation_tps": 2.5,
                "avg_acceptance_length": 2.0,
                "peak_memory_gb": 1.5,
                "bridge_rejects": 1,
                "draft_expansions": 4,
                "proposed_target_tokens": 8,
            },
            {
                "num_input_tokens": 10,
                "num_output_tokens": 5,
                "prefill_time_s": 1.0,
                "decode_time_s": 1.0,
                "total_time_s": 2.0,
                "generation_tps": 5.0,
                "avg_acceptance_length": 3.0,
                "peak_memory_gb": 2.0,
                "bridge_rejects": 2,
                "draft_expansions": 6,
                "proposed_target_tokens": 9,
            },
        ],
    )

    assert aggregate["bridge_rejects"] == 3
    assert aggregate["draft_expansions"] == 10
    assert aggregate["proposed_target_tokens"] == 17
    assert aggregate["mean_acceptance_length"] == 2.5
