from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

COMPILED_TREE_BLOCK_FNS: dict[int, Any] = {}


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


def get_compiled_tree_block_fn(layer: Any):
    key = id(layer)
    compiled = COMPILED_TREE_BLOCK_FNS.get(key)
    if compiled is not None:
        return compiled

    attn = layer.self_attn

    @mx.compile
    def compiled_tree_block(
        hidden_states: mx.array,
        old_keys: mx.array,
        old_values: mx.array,
        position_ids: mx.array,
        mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        residual = hidden_states
        inputs = layer.input_layernorm(hidden_states)
        batch_size, seq_len, _ = inputs.shape

        queries = attn.q_proj(inputs)
        new_keys = attn.k_proj(inputs)
        new_values = attn.v_proj(inputs)

        queries = attn.q_norm(
            queries.reshape(batch_size, seq_len, attn.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        new_keys = attn.k_norm(
            new_keys.reshape(batch_size, seq_len, attn.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        new_values = new_values.reshape(
            batch_size, seq_len, attn.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)

        queries = apply_rope_position_ids(attn.rope, queries, position_ids)
        new_keys = apply_rope_position_ids(attn.rope, new_keys, position_ids)

        keys = mx.concatenate([old_keys, new_keys], axis=2)
        values = mx.concatenate([old_values, new_values], axis=2)
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None,
            scale=attn.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        hidden_states = residual + attn.o_proj(output)

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)
        return hidden_states, new_keys, new_values

    COMPILED_TREE_BLOCK_FNS[key] = compiled_tree_block
    return compiled_tree_block


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
    if cache is not None and cache.keys is not None and hidden_states.shape[1] > 1:
        compiled = get_compiled_tree_block_fn(layer)
        old_keys = cache.keys[..., : cache.offset, :]
        old_values = cache.values[..., : cache.offset, :]
        hidden_states, new_keys, new_values = compiled(
            hidden_states,
            old_keys,
            old_values,
            position_ids,
            mask,
        )
        cache.update_and_fetch(new_keys, new_values)
        return hidden_states

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
