from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from mlx_lm.models.gated_delta import compute_g, gated_delta_update

from .custom_qwen35_model import (
    get_compiled_linear_verify_fn,
)
from .qwen35_target import qwen35_full_attention_output
from .qwen35_tree_kernels import tree_conv1d_kernel, tree_gated_delta_kernel


@dataclass
class Qwen35PendingTreeState:
    full_keys: dict[int, mx.array]
    full_values: dict[int, mx.array]
    linear_conv_states: dict[int, mx.array]
    linear_states: dict[int, mx.array]


FULL_ATTENTION_TREE_BLOCK_FNS: dict[int, Any] = {}
_QWEN35_TREE_LINEAR_MODE = os.environ.get("DTREE_QWEN35_TREE_LINEAR_MODE", "batched").lower()
_QWEN35_TREE_KERNEL_ENABLED = os.environ.get("DTREE_QWEN35_TREE_KERNEL", "1").lower() not in (
    "",
    "0",
    "false",
)
_QWEN35_TREE_CONV_KERNEL_ENABLED = os.environ.get("DTREE_QWEN35_TREE_CONV_KERNEL", "1").lower() not in (
    "",
    "0",
    "false",
)


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


def _tree_depth_groups(parents: list[int]) -> list[list[int]]:
    if not parents:
        return []
    if parents[0] != -1:
        raise ValueError(f"tree root parent must be -1, got {parents[0]}")

    depths = [0] * len(parents)
    groups: list[list[int]] = [[0]]
    for node_index in range(1, len(parents)):
        parent_index = int(parents[node_index])
        if parent_index < 0 or parent_index >= node_index:
            raise ValueError(
                f"tree parent index must precede child, got parents[{node_index}]={parent_index}"
            )
        depth = depths[parent_index] + 1
        depths[node_index] = depth
        while len(groups) <= depth:
            groups.append([])
        groups[depth].append(node_index)
    return groups


def _forward_linear_tree_serial(
    layer: Any,
    tree_hidden_states: mx.array,
    layer_cache: Any,
    parents: list[int],
) -> tuple[mx.array, mx.array, mx.array]:
    node_count = int(tree_hidden_states.shape[1])
    node_hidden_states = [tree_hidden_states[:, index : index + 1, :] for index in range(node_count)]
    next_node_hidden_states: list[mx.array] = []
    node_conv_states: list[mx.array] = []
    node_states: list[mx.array] = []

    for node_index in range(node_count):
        hidden_states = node_hidden_states[node_index]
        parent_index = int(parents[node_index])
        if parent_index >= 0:
            initial_conv_state = node_conv_states[parent_index]
            initial_state = node_states[parent_index]
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
        next_node_hidden_states.append(hidden_states)
        node_conv_states.append(new_conv_state)
        node_states.append(new_state)

    return (
        mx.concatenate(next_node_hidden_states, axis=1),
        mx.concatenate(node_states, axis=0),
        mx.concatenate(node_conv_states, axis=0),
    )


def _forward_linear_tree_batched_kernel(
    linear_attn: Any,
    *,
    qkv: mx.array,
    z: mx.array,
    a: mx.array,
    b: mx.array,
    base_state: mx.array,
    base_conv_state: mx.array,
    parents: list[int],
    depth_groups: list[list[int]],
    conv_weight: mx.array,
    input_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array, mx.array] | None:
    if not _QWEN35_TREE_KERNEL_ENABLED or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if linear_attn.head_k_dim % 32 != 0:
        return None

    batch_size, tree_size, _ = qkv.shape
    keep = int(linear_attn.conv_kernel_size) - 1
    parents_mx = mx.array(parents, dtype=mx.int32)

    conv_result = None
    if _QWEN35_TREE_CONV_KERNEL_ENABLED:
        conv_result = tree_conv1d_kernel(
            qkv,
            base_conv_state,
            conv_weight,
            parents_mx,
        )

    if conv_result is not None:
        conv_out, node_conv_states_full = conv_result
        q_all, k_all, v_all = [
            tensor.reshape(batch_size, tree_size, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [linear_attn.key_dim, 2 * linear_attn.key_dim], -1),
                [
                    linear_attn.num_k_heads,
                    linear_attn.num_k_heads,
                    linear_attn.num_v_heads,
                ],
                [
                    linear_attn.head_k_dim,
                    linear_attn.head_k_dim,
                    linear_attn.head_v_dim,
                ],
                strict=True,
            )
        ]
        node_conv_states = node_conv_states_full[0]
    else:
        q_parts: list[mx.array | None] = [None] * tree_size
        k_parts: list[mx.array | None] = [None] * tree_size
        v_parts: list[mx.array | None] = [None] * tree_size
        node_conv_state_parts: list[mx.array | None] = [None] * tree_size

        for indices in depth_groups:
            if not indices:
                continue

            index_array = mx.array(indices, dtype=mx.int32)
            group_size = len(indices)
            parent_conv_states = []
            for tree_index in indices:
                parent_index = int(parents[tree_index])
                if parent_index < 0:
                    parent_conv_states.append(base_conv_state)
                else:
                    parent_conv_state = node_conv_state_parts[parent_index]
                    if parent_conv_state is None:
                        return None
                    parent_conv_states.append(parent_conv_state)

            conv_state = mx.concatenate(parent_conv_states, axis=0)
            qkv_step = mx.take(qkv, index_array, axis=1).reshape(
                group_size, 1, linear_attn.conv_dim
            )
            conv_input = mx.concatenate([conv_state, qkv_step], axis=1)
            new_conv_state = (
                mx.contiguous(conv_input[:, -keep:, :])
                if keep > 0
                else mx.zeros((group_size, 0, linear_attn.conv_dim), dtype=input_dtype)
            )
            conv_out = nn.silu(
                (conv_input * conv_weight[None, :, :]).sum(axis=1)[:, None, :]
            )
            q, k, v = [
                tensor.reshape(group_size, 1, heads, dim)
                for tensor, heads, dim in zip(
                    mx.split(conv_out, [linear_attn.key_dim, 2 * linear_attn.key_dim], -1),
                    [
                        linear_attn.num_k_heads,
                        linear_attn.num_k_heads,
                        linear_attn.num_v_heads,
                    ],
                    [
                        linear_attn.head_k_dim,
                        linear_attn.head_k_dim,
                        linear_attn.head_v_dim,
                    ],
                    strict=True,
                )
            ]
            for group_pos, tree_index in enumerate(indices):
                q_parts[tree_index] = q[group_pos : group_pos + 1]
                k_parts[tree_index] = k[group_pos : group_pos + 1]
                v_parts[tree_index] = v[group_pos : group_pos + 1]
                node_conv_state_parts[tree_index] = new_conv_state[group_pos : group_pos + 1]

        if (
            any(part is None for part in q_parts)
            or any(part is None for part in k_parts)
            or any(part is None for part in v_parts)
            or any(state is None for state in node_conv_state_parts)
        ):
            return None

        q_all = mx.concatenate(q_parts, axis=1)  # type: ignore[arg-type]
        k_all = mx.concatenate(k_parts, axis=1)  # type: ignore[arg-type]
        v_all = mx.concatenate(v_parts, axis=1)  # type: ignore[arg-type]
        node_conv_states = mx.concatenate(node_conv_state_parts, axis=0)  # type: ignore[arg-type]

    inv_scale = k_all.shape[-1] ** -0.5
    q_all = (inv_scale**2) * mx.fast.rms_norm(q_all, None, 1e-6)
    k_all = inv_scale * mx.fast.rms_norm(k_all, None, 1e-6)
    g = compute_g(linear_attn.A_log, a, linear_attn.dt_bias)
    beta = mx.sigmoid(b)
    kernel_result = tree_gated_delta_kernel(
        q_all,
        k_all,
        v_all,
        g,
        beta,
        base_state,
        parents_mx,
    )
    if kernel_result is None:
        return None

    raw_out, node_states_full = kernel_result
    out = linear_attn.norm(raw_out, z)
    out = linear_attn.out_proj(out.reshape(batch_size, tree_size, -1))
    if linear_attn.sharding_group is not None:
        out = mx.distributed.all_sum(out, group=linear_attn.sharding_group)
    return out, node_states_full[0], node_conv_states


def _forward_linear_tree_batched(
    layer: Any,
    tree_hidden_states: mx.array,
    layer_cache: Any,
    parents: list[int],
    depth_groups: list[list[int]],
) -> tuple[mx.array, mx.array, mx.array]:
    linear_attn = layer.linear_attn
    residual = tree_hidden_states
    inputs = layer.input_layernorm(tree_hidden_states)
    batch_size, tree_size, _ = inputs.shape
    if batch_size != 1:
        raise ValueError("Qwen3.5 tree verification expects batch=1 for recurrent layers.")

    if linear_attn.sharding_group is not None:
        inputs = sum_gradients(linear_attn.sharding_group)(inputs)

    qkv = linear_attn.in_proj_qkv(inputs)
    z = linear_attn.in_proj_z(inputs).reshape(
        batch_size,
        tree_size,
        linear_attn.num_v_heads,
        linear_attn.head_v_dim,
    )
    b = linear_attn.in_proj_b(inputs)
    a = linear_attn.in_proj_a(inputs)

    base_conv_state = _initial_linear_conv_state(layer, layer_cache, inputs.dtype)
    base_state = _initial_linear_state(layer, layer_cache, inputs.dtype)
    conv_weight = linear_attn.conv1d.weight[:, :, 0].T

    kernel_result = _forward_linear_tree_batched_kernel(
        linear_attn,
        qkv=qkv,
        z=z,
        a=a,
        b=b,
        base_state=base_state,
        base_conv_state=base_conv_state,
        parents=parents,
        depth_groups=depth_groups,
        conv_weight=conv_weight,
        input_dtype=inputs.dtype,
    )
    if kernel_result is not None:
        out, node_states, node_conv_states = kernel_result
    else:
        raw_outputs: list[mx.array | None] = [None] * tree_size
        node_state_parts: list[mx.array | None] = [None] * tree_size
        node_conv_state_parts: list[mx.array | None] = [None] * tree_size
        keep = int(linear_attn.conv_kernel_size) - 1

        for indices in depth_groups:
            if not indices:
                continue

            index_array = mx.array(indices, dtype=mx.int32)
            group_size = len(indices)
            parent_states = []
            parent_conv_states = []
            for tree_index in indices:
                parent_index = int(parents[tree_index])
                if parent_index < 0:
                    parent_states.append(base_state)
                    parent_conv_states.append(base_conv_state)
                else:
                    parent_state = node_state_parts[parent_index]
                    parent_conv_state = node_conv_state_parts[parent_index]
                    if parent_state is None or parent_conv_state is None:
                        raise ValueError("parent state missing during tree-aware verify")
                    parent_states.append(parent_state)
                    parent_conv_states.append(parent_conv_state)

            state_in = mx.concatenate(parent_states, axis=0)
            conv_state = mx.concatenate(parent_conv_states, axis=0)
            qkv_step = mx.take(qkv, index_array, axis=1).reshape(
                group_size, 1, linear_attn.conv_dim
            )
            conv_input = mx.concatenate([conv_state, qkv_step], axis=1)
            new_conv_state = (
                mx.contiguous(conv_input[:, -keep:, :])
                if keep > 0
                else mx.zeros((group_size, 0, linear_attn.conv_dim), dtype=inputs.dtype)
            )
            conv_out = nn.silu(
                (conv_input * conv_weight[None, :, :]).sum(axis=1)[:, None, :]
            )
            q, k, v = [
                tensor.reshape(group_size, 1, heads, dim)
                for tensor, heads, dim in zip(
                    mx.split(conv_out, [linear_attn.key_dim, 2 * linear_attn.key_dim], -1),
                    [
                        linear_attn.num_k_heads,
                        linear_attn.num_k_heads,
                        linear_attn.num_v_heads,
                    ],
                    [
                        linear_attn.head_k_dim,
                        linear_attn.head_k_dim,
                        linear_attn.head_v_dim,
                    ],
                    strict=True,
                )
            ]

            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            a_step = mx.take(a, index_array, axis=1).reshape(
                group_size, 1, linear_attn.num_v_heads
            )
            b_step = mx.take(b, index_array, axis=1).reshape(
                group_size, 1, linear_attn.num_v_heads
            )
            out_step, state_out = gated_delta_update(
                q,
                k,
                v,
                a_step,
                b_step,
                linear_attn.A_log,
                linear_attn.dt_bias,
                state_in,
                None,
                use_kernel=not linear_attn.training,
            )

            for group_pos, tree_index in enumerate(indices):
                raw_outputs[tree_index] = out_step[group_pos : group_pos + 1]
                node_state_parts[tree_index] = state_out[group_pos : group_pos + 1]
                node_conv_state_parts[tree_index] = new_conv_state[group_pos : group_pos + 1]

        if (
            any(output is None for output in raw_outputs)
            or any(state is None for state in node_state_parts)
            or any(state is None for state in node_conv_state_parts)
        ):
            raise ValueError("tree-aware linear verify did not produce every node output")

        raw_out = mx.concatenate(raw_outputs, axis=1)  # type: ignore[arg-type]
        out = linear_attn.norm(raw_out, z)
        out = linear_attn.out_proj(out.reshape(batch_size, tree_size, -1))
        if linear_attn.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=linear_attn.sharding_group)
        node_states = mx.concatenate(node_state_parts, axis=0)  # type: ignore[arg-type]
        node_conv_states = mx.concatenate(node_conv_state_parts, axis=0)  # type: ignore[arg-type]

    hidden_states = residual + out
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)
    return hidden_states, node_states, node_conv_states


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
    depth_groups = _tree_depth_groups(parents)

    pending = Qwen35PendingTreeState(
        full_keys={},
        full_values={},
        linear_conv_states={},
        linear_states={},
    )

    selected_hidden_states: list[mx.array] = []
    tree_hidden_states = embedded

    for layer_idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
        if getattr(layer, "is_linear", False):
            if _QWEN35_TREE_LINEAR_MODE == "serial":
                tree_hidden_states, node_states, node_conv_states = _forward_linear_tree_serial(
                    layer,
                    tree_hidden_states,
                    layer_cache,
                    parents,
                )
            else:
                tree_hidden_states, node_states, node_conv_states = _forward_linear_tree_batched(
                    layer,
                    tree_hidden_states,
                    layer_cache,
                    parents,
                    depth_groups,
                )
            pending.linear_conv_states[layer_idx] = node_conv_states
            pending.linear_states[layer_idx] = node_states
            if layer_idx in target_layer_ids:
                selected_hidden_states.append(tree_hidden_states)
            continue

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
        layer_cache[0] = pending.linear_conv_states[layer_idx][deepest_index : deepest_index + 1]
        layer_cache[1] = pending.linear_states[layer_idx][deepest_index : deepest_index + 1]
        if hasattr(layer_cache, "left_padding"):
            layer_cache.left_padding = None
        if hasattr(layer_cache, "lengths"):
            layer_cache.lengths = None
