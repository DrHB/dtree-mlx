"""Smoke test: the package imports and public surface is wired correctly.

Intentionally does not load any model weights. A contributor can run this
without an Apple Silicon GPU and without a ~12 GB model download.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def test_public_api_importable():
    import dtree_mlx

    expected = {
        "DFlashGenerator",
        "DFlashResult",
        "DFlashDraftModel",
        "LoadedTargetModel",
        "adapter_for_model_type",
        "dflash_generate",
        "dtree_generate",
        "load_draft_model",
        "load_target_model",
        "longest_prefix_match",
        "sample_tokens",
    }
    assert expected.issubset(set(dtree_mlx.__all__))
    for name in expected:
        assert hasattr(dtree_mlx, name), f"missing export: {name}"


def test_generator_is_callable_class():
    from dtree_mlx import DFlashGenerator

    assert callable(DFlashGenerator)


def test_cli_entrypoints_importable():
    from dtree_mlx import benchmark_cli, cli, compare_cli

    for module in (cli, benchmark_cli, compare_cli):
        assert callable(module.main)


def test_cli_verify_mode_choices_exclude_unsafe_modes():
    from dtree_mlx import cli

    saved_argv = sys.argv
    try:
        sys.argv = [cli.__name__]
        args = cli.parse_args()
        assert args.verify_mode == "parallel-replay"
    finally:
        sys.argv = saved_argv


def test_cli_decode_mode_defaults_to_dflash():
    from dtree_mlx import cli

    saved_argv = sys.argv
    try:
        sys.argv = [cli.__name__]
        args = cli.parse_args()
        assert args.decode_mode == "dflash"
    finally:
        sys.argv = saved_argv


def test_cli_parses_target_quantization_flags():
    from dtree_mlx import cli

    saved_argv = sys.argv
    try:
        sys.argv = [
            cli.__name__,
            "--target-quant-bits",
            "4",
            "--target-quant-group-size",
            "64",
        ]
        args = cli.parse_args()
        assert args.target_quant_bits == 4
        assert args.target_quant_group_size == 64
    finally:
        sys.argv = saved_argv


def test_compare_cli_parses_target_quantization_flags():
    from dtree_mlx import compare_cli

    saved_argv = sys.argv
    try:
        sys.argv = [
            compare_cli.__name__,
            "--target-quant-bits",
            "4",
            "--target-quant-group-size",
            "64",
        ]
        args = compare_cli.parse_args()
        assert args.target_quant_bits == 4
        assert args.target_quant_group_size == 64
    finally:
        sys.argv = saved_argv


def test_qwen35_adapter_is_registered():
    from dtree_mlx import adapter_for_model_type

    assert adapter_for_model_type("qwen3_5") is not None


def test_generator_auto_mask_uses_target_family(monkeypatch):
    from dtree_mlx import api

    def build_generator(family: str) -> str:
        fake_target = SimpleNamespace(
            adapter=SimpleNamespace(family=family),
            model=object(),
            resolved_model_path=Path("/tmp/target"),
        )
        fake_draft = SimpleNamespace(
            attention_mask_mode=None,
            block_size=16,
            target_layer_ids=[],
        )
        monkeypatch.setattr(api, "load_target_model", lambda target_model: fake_target)
        monkeypatch.setattr(
            api,
            "load_draft_model",
            lambda draft_model: (fake_draft, Path("/tmp/draft")),
        )
        monkeypatch.setattr(
            api,
            "maybe_quantize_draft_model",
            lambda draft, bits, group_size: {},
        )
        generator = api.DFlashGenerator()
        return generator.draft_attention_mask

    assert build_generator("qwen3") == "none"
    assert build_generator("qwen3_5") == "none"
