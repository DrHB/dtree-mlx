from __future__ import annotations

import pytest
import mlx.core as mx

from dtree_mlx.draft import DFlashDraftModel, DraftArgs
from dtree_mlx.runtime import trim_draft_cache


def build_test_draft() -> DFlashDraftModel:
    args = DraftArgs(
        model_type="qwen3",
        hidden_size=8,
        num_hidden_layers=2,
        intermediate_size=16,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        head_dim=4,
        tie_word_embeddings=False,
        block_size=4,
        dflash_config={
            "target_layer_ids": [0, 1],
            "mask_token_id": 0,
        },
    )
    draft = DFlashDraftModel(args)
    mx.eval(draft.parameters())
    return draft


@pytest.mark.parametrize("cache_mode", ["kv", "context-only"])
def test_draft_precompute_context_matches_uncached_path(cache_mode: str):
    draft = build_test_draft()
    draft.cache_mode = cache_mode

    block_size = draft.block_size
    context_len = 2
    target_width = len(draft.target_layer_ids) * draft.args.hidden_size

    noise_embedding = (
        mx.arange(block_size * draft.args.hidden_size, dtype=mx.float32)
        .reshape(1, block_size, draft.args.hidden_size)
        / 100.0
    )
    target_hidden = (
        mx.arange(context_len * target_width, dtype=mx.float32)
        .reshape(1, context_len, target_width)
        / 100.0
    )

    uncached_output = draft(noise_embedding=noise_embedding, target_hidden=target_hidden)
    cache = draft.make_cache()
    cached_output = draft(
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        cache=cache,
    )
    mx.eval(uncached_output, cached_output)

    assert bool(
        mx.allclose(uncached_output, cached_output, atol=1e-5, rtol=1e-5).item()
    )

    trim_draft_cache(cache, block_size)
    for layer_cache in cache:
        assert int(layer_cache.offset) == context_len
