from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from dtree_mlx.qwen35_tree_kernels import tree_conv1d_kernel, tree_gated_delta_kernel


def _require_tree_kernel(result):
    if result is None:
        pytest.skip("tree kernel unavailable on this device/runtime")
    return result


def test_tree_conv1d_kernel_matches_reference():
    qkv = mx.array(
        np.random.default_rng(0).normal(size=(1, 5, 8)).astype(np.float32)
    )
    base_conv_state = mx.array(
        np.random.default_rng(1).normal(size=(1, 3, 8)).astype(np.float32)
    )
    conv_weight = mx.array(
        np.random.default_rng(2).normal(size=(4, 8)).astype(np.float32)
    )
    parents = mx.array([-1, 0, 0, 1, 2], dtype=mx.int32)

    result = _require_tree_kernel(
        tree_conv1d_kernel(qkv, base_conv_state, conv_weight, parents)
    )
    conv_out, conv_states = result
    mx.eval(conv_out, conv_states)

    qkv_np = np.array(qkv)
    base_conv_state_np = np.array(base_conv_state)
    conv_weight_np = np.array(conv_weight)
    parents_np = np.array(parents)

    ref_conv_out = np.zeros_like(qkv_np)
    ref_conv_states = np.zeros((1, 5, 3, 8), dtype=np.float32)
    for t in range(5):
        parent = int(parents_np[t])
        parent_state = (
            base_conv_state_np[:, :, :]
            if parent < 0
            else ref_conv_states[:, parent, :, :]
        )
        conv_input = np.concatenate([parent_state, qkv_np[:, t : t + 1, :]], axis=1)
        acc = (conv_input * conv_weight_np[None, :, :]).sum(axis=1)
        ref_conv_out[:, t, :] = acc / (1.0 + np.exp(-acc))
        ref_conv_states[:, t, :, :] = conv_input[:, 1:, :]

    np.testing.assert_allclose(np.array(conv_out), ref_conv_out, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(conv_states), ref_conv_states, atol=1e-5, rtol=1e-5)


def test_tree_gated_delta_kernel_matches_reference():
    rng = np.random.default_rng(3)
    q = mx.array(rng.normal(size=(1, 5, 1, 32)).astype(np.float32))
    k = mx.array(rng.normal(size=(1, 5, 1, 32)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, 5, 1, 8)).astype(np.float32))
    g = mx.array(rng.normal(size=(1, 5, 1)).astype(np.float32))
    beta = mx.array(rng.normal(size=(1, 5, 1)).astype(np.float32))
    state = mx.array(rng.normal(size=(1, 1, 8, 32)).astype(np.float32))
    parents = mx.array([-1, 0, 0, 1, 2], dtype=mx.int32)

    result = _require_tree_kernel(
        tree_gated_delta_kernel(q, k, v, g, beta, state, parents)
    )
    out, states = result
    mx.eval(out, states)

    q_np = np.array(q)
    k_np = np.array(k)
    v_np = np.array(v)
    g_np = np.array(g)
    beta_np = np.array(beta)
    state_np = np.array(state)
    parents_np = np.array(parents)

    ref_out = np.zeros((1, 5, 1, 8), dtype=np.float32)
    ref_states = np.zeros((1, 5, 1, 8, 32), dtype=np.float32)
    for t in range(5):
        parent = int(parents_np[t])
        parent_state = state_np[:, :, :, :] if parent < 0 else ref_states[:, parent, :, :, :]
        step_state = parent_state * g_np[:, t : t + 1, :, None, None]
        kv_mem = np.sum(step_state * k_np[:, t : t + 1, :, None, :], axis=-1)
        delta = (v_np[:, t : t + 1, :, :] - kv_mem) * beta_np[:, t : t + 1, :, None]
        step_state = step_state + delta[..., None] * k_np[:, t : t + 1, :, None, :]
        ref_out[:, t : t + 1, :, :] = np.sum(
            step_state * q_np[:, t : t + 1, :, None, :],
            axis=-1,
        )
        ref_states[:, t : t + 1, :, :, :] = step_state

    np.testing.assert_allclose(np.array(out), ref_out, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(states), ref_states, atol=1e-5, rtol=1e-5)
