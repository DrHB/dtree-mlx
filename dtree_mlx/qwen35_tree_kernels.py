from __future__ import annotations

from typing import Optional

import mlx.core as mx


def _make_tree_gated_delta_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        for (int t = 0; t < T; ++t) {
          auto parent_idx = parents[t];

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto q_t = q + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto k_t = k + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto v_t = v + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_t[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_t[dv_idx] - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_t[s_idx] * delta;
            out += state[i] * q_t[s_idx];
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(static_cast<InT>(state[i]));
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
        }
    """

    return mx.fast.metal_kernel(
        name="dtree_qwen35_tree_gated_delta",
        input_names=["q", "k", "v", "g", "beta", "state_in", "parents", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_tree_conv1d_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto c_idx = thread_position_in_grid.x;
        auto b_idx = thread_position_in_grid.y;

        if (c_idx >= ConvDim) {
          return;
        }

        for (int t = 0; t < T; ++t) {
          auto parent_idx = parents[t];

          float acc = 0.0f;
          for (int k = 0; k < Keep; ++k) {
            float x;
            if (parent_idx < 0) {
              x = static_cast<float>(
                base_conv_state[(b_idx * Keep + k) * ConvDim + c_idx]
              );
            } else {
              x = static_cast<float>(
                conv_states[
                  (((b_idx * T + parent_idx) * Keep + k) * ConvDim) + c_idx
                ]
              );
            }
            auto w = static_cast<float>(conv_weight[k * ConvDim + c_idx]);
            acc += x * w;
          }

          auto qkv_t = qkv + (b_idx * T + t) * ConvDim;
          acc += static_cast<float>(qkv_t[c_idx])
            * static_cast<float>(conv_weight[Keep * ConvDim + c_idx]);

          auto silu = acc / (1.0f + exp(-acc));
          conv_out[(b_idx * T + t) * ConvDim + c_idx] =
            static_cast<InT>(silu);

          for (int k = 0; k < Keep; ++k) {
            InT value;
            if (k + 1 < Keep) {
              if (parent_idx < 0) {
                value = base_conv_state[(b_idx * Keep + k + 1) * ConvDim + c_idx];
              } else {
                value = conv_states[
                  (((b_idx * T + parent_idx) * Keep + k + 1) * ConvDim) + c_idx
                ];
              }
            } else {
              value = qkv_t[c_idx];
            }
            conv_states[
              (((b_idx * T + t) * Keep + k) * ConvDim) + c_idx
            ] = value;
          }
        }
    """

    return mx.fast.metal_kernel(
        name="dtree_qwen35_tree_conv1d",
        input_names=["qkv", "base_conv_state", "conv_weight", "parents", "T"],
        output_names=["conv_out", "conv_states"],
        source=source,
    )


_TREE_GATED_DELTA_KERNEL = _make_tree_gated_delta_kernel()
_TREE_CONV1D_KERNEL = _make_tree_conv1d_kernel()


def tree_conv1d_kernel(
    qkv: mx.array,
    base_conv_state: mx.array,
    conv_weight: mx.array,
    parents: mx.array,
) -> Optional[tuple[mx.array, mx.array]]:
    if _TREE_CONV1D_KERNEL is None:
        return None
    if int(base_conv_state.shape[1]) <= 0:
        return None

    batch_size, tree_size, conv_dim = qkv.shape
    keep = int(base_conv_state.shape[1])
    if int(conv_weight.shape[0]) != keep + 1:
        return None
    if int(conv_weight.shape[1]) != conv_dim:
        return None

    input_dtype = qkv.dtype
    return _TREE_CONV1D_KERNEL(
        inputs=[qkv, base_conv_state, conv_weight, parents, tree_size],
        template=[
            ("InT", input_dtype),
            ("Keep", keep),
            ("ConvDim", conv_dim),
        ],
        grid=(conv_dim, batch_size, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch_size, tree_size, conv_dim), (batch_size, tree_size, keep, conv_dim)],
        output_dtypes=[input_dtype, input_dtype],
    )


def tree_gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parents: mx.array,
) -> Optional[tuple[mx.array, mx.array]]:
    if _TREE_GATED_DELTA_KERNEL is None:
        return None
    batch_size, tree_size, num_k_heads, head_k_dim = k.shape
    num_v_heads, head_v_dim = v.shape[2:]
    if head_k_dim % 32 != 0:
        return None

    input_dtype = q.dtype
    state_dtype = state.dtype
    return _TREE_GATED_DELTA_KERNEL(
        inputs=[q, k, v, g, beta, state, parents, tree_size],
        template=[
            ("InT", input_dtype),
            ("StT", state_dtype),
            ("Dk", head_k_dim),
            ("Dv", head_v_dim),
            ("Hk", num_k_heads),
            ("Hv", num_v_heads),
        ],
        grid=(32, head_v_dim, batch_size * num_v_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch_size, tree_size, num_v_heads, head_v_dim), (batch_size, tree_size, num_v_heads, head_v_dim, head_k_dim)],
        output_dtypes=[input_dtype, state_dtype],
    )
