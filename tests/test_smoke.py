"""Smoke test: the package imports and public surface is wired correctly.

Intentionally does not load any model weights. A contributor can run this
without an Apple Silicon GPU and without a ~12 GB model download.
"""
from __future__ import annotations

import sys


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
