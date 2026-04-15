from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

from .custom_qwen35_model import (
    get_compiled_full_attention_verify_fn,
    get_compiled_linear_verify_fn,
)


@dataclass
class Qwen35PendingTreeState:
    full_keys: dict[int, list[mx.array]]
    full_values: dict[int, list[mx.array]]
    linear_conv_states: dict[int, list[mx.array]]
    linear_states: dict[int, list[mx.array]]


def build_node_lineages(parents: list[int]) -> list[list[int]]:
    if not parents:
        return []
    if parents[0] != -1:
        raise ValueError(f"tree root parent must be -1, got {parents[0]}")

    lineages: list[list[int]] = [[]]
    for node_index in range(1, len(parents)):
        parent_index = int(parents[node_index])
        if parent_index < 0 or parent_index >= node_index:
            raise ValueError(
                f"tree parent index must precede child, got parents[{node_index}]={parent_index}"
            )
        lineages.append([*lineages[parent_index], parent_index])
    return lineages


def _target_text_model(target_model: Any) -> Any:
    if hasattr(target_model, "language_model") and hasattr(target_model.language_model, "model"):
        return target_model.language_model.model
    if hasattr(target_model, "model"):
        return target_model.model
    raise AttributeError(f"Unsupported Qwen3.5 target model: {type(target_model)!r}")


def _full_attention_prefix_chunks(
    layer_cache: Any,
    pending_values: list[mx.array],
    lineage: list[int],
    *,
    value_kind: str,
) -> list[mx.array]:
    chunks: list[mx.array] = []
    base_value = getattr(layer_cache, value_kind, None)
    base_offset = int(getattr(layer_cache, "offset", 0) or 0)
    if base_value is not None and base_offset > 0:
        chunks.append(base_value[..., :base_offset, :])
    for ancestor_index in lineage:
        chunks.append(pending_values[ancestor_index])
    return chunks


def _initial_linear_conv_state(layer: Any, layer_cache: Any, dtype: mx.Dtype) -> mx.array:
    conv_state = layer_cache[0]
    if conv_state is not None:
        return conv_state
    linear = layer.linear_attn
    return mx.zeros((1, linear.conv_kernel_size - 1, linear.conv_dim), dtype=dtype)


def _initial_linear_state(layer: Any, layer_cache: Any, dtype: mx.Dtype) -> mx.array:
    state = layer_cache[1]
    if state is not None:
        return state
    linear = layer.linear_attn
    return mx.zeros(
        (1, linear.num_v_heads, linear.head_v_dim, linear.head_k_dim),
        dtype=dtype,
    )


def forward_full_attention_tree_token(
    layer: Any,
    hidden_states: mx.array,
    prefix_key_chunks: list[mx.array],
    prefix_value_chunks: list[mx.array],
    *,
    position_id: int,
) -> tuple[mx.array, mx.array, mx.array]:
    attn = layer.self_attn
    old_keys = mx.concatenate(prefix_key_chunks, axis=2) if prefix_key_chunks else None
    old_values = mx.concatenate(prefix_value_chunks, axis=2) if prefix_value_chunks else None
    prefix_len = 0 if old_keys is None else int(old_keys.shape[2])
    if old_keys is not None and old_values is not None:
        compiled = get_compiled_full_attention_verify_fn(layer)
        return compiled(
            hidden_states,
            old_keys,
            old_values,
            prefix_len,
        )

    residual = hidden_states
    inputs = layer.input_layernorm(hidden_states)
    batch_size, seq_len, _ = inputs.shape

    q_proj_output = attn.q_proj(inputs)
    queries, gate = mx.split(
        q_proj_output.reshape(batch_size, seq_len, attn.num_attention_heads, -1),
        2,
        axis=-1,
    )
    gate = gate.reshape(batch_size, seq_len, -1)

    new_keys = attn.k_proj(inputs)
    new_values = attn.v_proj(inputs)

    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    new_keys = attn.k_norm(
        new_keys.reshape(batch_size, seq_len, attn.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    new_values = new_values.reshape(
        batch_size,
        seq_len,
        attn.num_key_value_heads,
        -1,
    ).transpose(0, 2, 1, 3)

    queries = attn.rope(queries, offset=position_id)
    new_keys = attn.rope(new_keys, offset=position_id)

    all_keys = mx.concatenate([*prefix_key_chunks, new_keys], axis=2) if prefix_key_chunks else new_keys
    all_values = (
        mx.concatenate([*prefix_value_chunks, new_values], axis=2)
        if prefix_value_chunks
        else new_values
    )
    output = scaled_dot_product_attention(
        queries,
        all_keys,
        all_values,
        cache=None,
        scale=attn.scale,
        mask=None,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    output = attn.o_proj(output * mx.sigmoid(gate))

    hidden_states = residual + output
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)
    return hidden_states, new_keys, new_values


def forward_qwen35_tree_with_hidden_states(
    target_model: Any,
    inputs: mx.array,
    cache: list[Any],
    layer_ids: list[int],
    parents: list[int],
    position_ids: mx.array,
) -> tuple[mx.array, mx.array, Qwen35PendingTreeState]:
    # Hybrid recurrent layers need one state per branch, so Qwen3.5 tree
    # verification walks the tree node-by-node and commits only the accepted path.
    text_model = _target_text_model(target_model)
    if inputs.shape[0] != 1:
        raise ValueError(f"Qwen3.5 tree verification expects batch=1, got {inputs.shape[0]}")
    if int(inputs.shape[1]) != len(parents):
        raise ValueError(
            f"tree parent list length must match verify length, got {len(parents)} vs {int(inputs.shape[1])}"
        )

    lineages = build_node_lineages(parents)
    positions = [int(position) for position in position_ids[0].tolist()]
    embedded = text_model.embed_tokens(inputs)
    target_layer_ids = set(layer_ids)

    pending = Qwen35PendingTreeState(
        full_keys={
            layer_idx: []
            for layer_idx, layer in enumerate(text_model.layers)
            if not getattr(layer, "is_linear", False)
        },
        full_values={
            layer_idx: []
            for layer_idx, layer in enumerate(text_model.layers)
            if not getattr(layer, "is_linear", False)
        },
        linear_conv_states={
            layer_idx: []
            for layer_idx, layer in enumerate(text_model.layers)
            if getattr(layer, "is_linear", False)
        },
        linear_states={
            layer_idx: []
            for layer_idx, layer in enumerate(text_model.layers)
            if getattr(layer, "is_linear", False)
        },
    )

    norm_hidden_states: list[mx.array] = []
    verifier_hidden_states: list[mx.array] = []

    for node_index in range(len(parents)):
        hidden_states = embedded[:, node_index : node_index + 1, :]
        selected_hidden_states: list[mx.array] = []
        parent_index = int(parents[node_index])
        lineage = lineages[node_index]

        for layer_idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
            if getattr(layer, "is_linear", False):
                if parent_index >= 0:
                    initial_conv_state = pending.linear_conv_states[layer_idx][parent_index]
                    initial_state = pending.linear_states[layer_idx][parent_index]
                else:
                    initial_conv_state = _initial_linear_conv_state(
                        layer,
                        layer_cache,
                        hidden_states.dtype,
                    )
                    initial_state = _initial_linear_state(
                        layer,
                        layer_cache,
                        hidden_states.dtype,
                    )
                (
                    hidden_states,
                    new_conv_state,
                    new_state,
                    _qkv,
                    _keys,
                    _values,
                    _g,
                    _beta,
                ) = get_compiled_linear_verify_fn(layer)(
                    hidden_states,
                    initial_conv_state,
                    initial_state,
                )
                pending.linear_conv_states[layer_idx].append(new_conv_state)
                pending.linear_states[layer_idx].append(new_state)
            else:
                prefix_key_chunks = _full_attention_prefix_chunks(
                    layer_cache,
                    pending.full_keys[layer_idx],
                    lineage,
                    value_kind="keys",
                )
                prefix_value_chunks = _full_attention_prefix_chunks(
                    layer_cache,
                    pending.full_values[layer_idx],
                    lineage,
                    value_kind="values",
                )
                hidden_states, new_keys, new_values = forward_full_attention_tree_token(
                    layer,
                    hidden_states,
                    prefix_key_chunks,
                    prefix_value_chunks,
                    position_id=positions[node_index],
                )
                pending.full_keys[layer_idx].append(new_keys)
                pending.full_values[layer_idx].append(new_values)

            if layer_idx in target_layer_ids:
                selected_hidden_states.append(hidden_states)

        norm_hidden_states.append(text_model.norm(hidden_states))
        verifier_hidden_states.append(mx.concatenate(selected_hidden_states, axis=-1))

    return (
        mx.concatenate(norm_hidden_states, axis=1),
        mx.concatenate(verifier_hidden_states, axis=1),
        pending,
    )


def commit_qwen35_tree_path(
    cache: list[Any],
    pending: Qwen35PendingTreeState,
    accepted_indices: list[int],
) -> None:
    if not accepted_indices:
        return

    deepest_index = accepted_indices[-1]
    for layer_idx, layer_cache in enumerate(cache):
        if layer_idx in pending.full_keys:
            new_keys = mx.concatenate(
                [pending.full_keys[layer_idx][index] for index in accepted_indices],
                axis=2,
            )
            new_values = mx.concatenate(
                [pending.full_values[layer_idx][index] for index in accepted_indices],
                axis=2,
            )
            layer_cache.update_and_fetch(new_keys, new_values)
            continue

        if layer_idx not in pending.linear_states:
            continue
        layer_cache[0] = pending.linear_conv_states[layer_idx][deepest_index]
        layer_cache[1] = pending.linear_states[layer_idx][deepest_index]
        if hasattr(layer_cache, "left_padding"):
            layer_cache.left_padding = None
        if hasattr(layer_cache, "lengths"):
            layer_cache.lengths = None
