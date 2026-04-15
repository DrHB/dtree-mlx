from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .custom_qwen35_model import (
    get_compiled_linear_verify_fn,
)
from .qwen35_target import qwen35_full_attention_output


@dataclass
class Qwen35PendingTreeState:
    full_keys: dict[int, mx.array]
    full_values: dict[int, mx.array]
    linear_conv_states: dict[int, list[mx.array]]
    linear_states: dict[int, list[mx.array]]


FULL_ATTENTION_TREE_BLOCK_FNS: dict[int, Any] = {}


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


def get_compiled_full_attention_tree_block_fn(layer: Any):
    key = id(layer)
    compiled = FULL_ATTENTION_TREE_BLOCK_FNS.get(key)
    if compiled is not None:
        return compiled

    attn = layer.self_attn

    @mx.compile
    def compiled_full_attention_tree_block(
        hidden_states: mx.array,
        old_keys: mx.array,
        old_values: mx.array,
        position_ids: mx.array,
        mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
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

        queries = apply_rope_position_ids(attn.rope, queries, position_ids)
        new_keys = apply_rope_position_ids(attn.rope, new_keys, position_ids)

        keys = mx.concatenate([old_keys, new_keys], axis=2)
        values = mx.concatenate([old_values, new_values], axis=2)
        output = qwen35_full_attention_output(
            attn=attn,
            queries=queries,
            keys=keys,
            values=values,
            mask=mask,
            cache=None,
            cached_prefix_len=int(old_keys.shape[2]),
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = attn.o_proj(output * mx.sigmoid(gate))

        hidden_states = residual + output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)
        return hidden_states, new_keys, new_values

    FULL_ATTENTION_TREE_BLOCK_FNS[key] = compiled_full_attention_tree_block
    return compiled_full_attention_tree_block


def forward_full_attention_tree_block(
    layer: Any,
    hidden_states: mx.array,
    attention_mask: mx.array,
    layer_cache: Any,
    *,
    position_ids: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    attn = layer.self_attn
    old_keys = getattr(layer_cache, "keys", None)
    old_values = getattr(layer_cache, "values", None)
    offset = int(getattr(layer_cache, "offset", 0) or 0)
    if old_keys is not None and old_values is not None and offset > 0:
        compiled = get_compiled_full_attention_tree_block_fn(layer)
        return compiled(
            hidden_states,
            old_keys[..., :offset, :],
            old_values[..., :offset, :],
            position_ids,
            attention_mask,
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

    queries = apply_rope_position_ids(attn.rope, queries, position_ids)
    new_keys = apply_rope_position_ids(attn.rope, new_keys, position_ids)

    prefix_len = 0
    all_keys = new_keys
    all_values = new_values
    if old_keys is not None and old_values is not None and offset > 0:
        prefix_len = offset
        all_keys = mx.concatenate([old_keys[..., :offset, :], new_keys], axis=2)
        all_values = mx.concatenate([old_values[..., :offset, :], new_values], axis=2)
    output = qwen35_full_attention_output(
        attn=attn,
        queries=queries,
        keys=all_keys,
        values=all_values,
        mask=attention_mask,
        cache=None,
        cached_prefix_len=prefix_len,
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
    attention_mask: mx.array,
) -> tuple[mx.array, mx.array, Qwen35PendingTreeState]:
    text_model = _target_text_model(target_model)
    if inputs.shape[0] != 1:
        raise ValueError(f"Qwen3.5 tree verification expects batch=1, got {inputs.shape[0]}")
    if int(inputs.shape[1]) != len(parents):
        raise ValueError(
            f"tree parent list length must match verify length, got {len(parents)} vs {int(inputs.shape[1])}"
        )

    embedded = text_model.embed_tokens(inputs)
    target_layer_ids = set(layer_ids)
    node_count = len(parents)

    pending = Qwen35PendingTreeState(
        full_keys={},
        full_values={},
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

    node_hidden_states = [embedded[:, index : index + 1, :] for index in range(node_count)]
    selected_hidden_states: list[mx.array] = []
    tree_hidden_states: mx.array | None = None

    for layer_idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
        if getattr(layer, "is_linear", False):
            if tree_hidden_states is not None:
                node_hidden_states = [
                    tree_hidden_states[:, index : index + 1, :]
                    for index in range(node_count)
                ]
                tree_hidden_states = None

            next_node_hidden_states: list[mx.array] = []
            for node_index in range(node_count):
                hidden_states = node_hidden_states[node_index]
                parent_index = int(parents[node_index])
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
                next_node_hidden_states.append(hidden_states)

            node_hidden_states = next_node_hidden_states
            if layer_idx in target_layer_ids:
                selected_hidden_states.append(mx.concatenate(node_hidden_states, axis=1))
            continue

        if tree_hidden_states is None:
            tree_hidden_states = mx.concatenate(node_hidden_states, axis=1)

        tree_hidden_states, new_keys, new_values = forward_full_attention_tree_block(
            layer,
            tree_hidden_states,
            attention_mask,
            layer_cache,
            position_ids=position_ids,
        )
        pending.full_keys[layer_idx] = new_keys
        pending.full_values[layer_idx] = new_values

        if layer_idx in target_layer_ids:
            selected_hidden_states.append(tree_hidden_states)

    if tree_hidden_states is None:
        tree_hidden_states = mx.concatenate(node_hidden_states, axis=1)

    verifier_hidden = (
        mx.concatenate(selected_hidden_states, axis=-1)
        if selected_hidden_states
        else tree_hidden_states[:, :, :0]
    )
    return (
        text_model.norm(tree_hidden_states),
        verifier_hidden,
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
    accepted_index_array = mx.array(accepted_indices, dtype=mx.uint32)
    for layer_idx, layer_cache in enumerate(cache):
        if layer_idx in pending.full_keys:
            new_keys = mx.take(pending.full_keys[layer_idx], accepted_index_array, axis=2)
            new_values = mx.take(pending.full_values[layer_idx], accepted_index_array, axis=2)
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
