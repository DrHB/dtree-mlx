from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention


def _normalize_position_ids(position_ids: mx.array, batch_size: int, seq_len: int) -> mx.array:
    if position_ids.ndim == 1:
        position_ids = position_ids[None]
    if position_ids.shape != (batch_size, seq_len):
        raise ValueError(
            "position_ids must have shape "
            f"({batch_size}, {seq_len}), got {tuple(position_ids.shape)}"
        )
    return position_ids.astype(mx.uint32)


def apply_rope_position_ids(rope: Any, tensor: mx.array, position_ids: mx.array) -> mx.array:
    batch_size, num_heads, seq_len, head_dim = tensor.shape
    position_ids = _normalize_position_ids(position_ids, batch_size, seq_len)
    flattened = tensor.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, num_heads, 1, head_dim)
    rotated = rope(flattened, offset=position_ids.reshape(-1))
    return rotated.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)


def forward_attention_with_position_ids(
    attn: Any,
    hidden_states: mx.array,
    mask: mx.array | None,
    cache: Any,
    position_ids: mx.array,
) -> mx.array:
    batch_size, seq_len, _ = hidden_states.shape

    queries = attn.q_proj(hidden_states)
    keys = attn.k_proj(hidden_states)
    values = attn.v_proj(hidden_states)

    queries = attn.q_norm(queries.reshape(batch_size, seq_len, attn.n_heads, -1)).transpose(
        0, 2, 1, 3
    )
    keys = attn.k_norm(keys.reshape(batch_size, seq_len, attn.n_kv_heads, -1)).transpose(
        0, 2, 1, 3
    )
    values = values.reshape(batch_size, seq_len, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    queries = apply_rope_position_ids(attn.rope, queries, position_ids)
    keys = apply_rope_position_ids(attn.rope, keys, position_ids)

    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)

    output = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=cache,
        scale=attn.scale,
        mask=mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    return attn.o_proj(output)


def forward_transformer_block_with_position_ids(
    layer: Any,
    hidden_states: mx.array,
    mask: mx.array | None,
    cache: Any,
    position_ids: mx.array,
) -> mx.array:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = forward_attention_with_position_ids(
        attn=layer.self_attn,
        hidden_states=hidden_states,
        mask=mask,
        cache=cache,
        position_ids=position_ids,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states
