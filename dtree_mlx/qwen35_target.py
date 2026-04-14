# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Adapted from https://github.com/bstnxbt/dflash-mlx

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import gated_delta as gated_delta_mod
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import _BaseCache

from .qwen35_kernels import (
    batched_sdpa_2pass_exact,
    gated_delta_kernel_with_tape,
    tape_replay_kernel,
)


class RecurrentRollbackCache(_BaseCache):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        instance._armed = False
        instance._tape = None
        instance._tape_k = None
        instance._tape_g = None
        instance._tape_qkv = None
        instance._snapshot = None
        return instance

    def __init__(self, size: int, *, conv_kernel_size: int = 4):
        self.cache = [None] * size
        self.conv_kernel_size = int(conv_kernel_size)

    def __getitem__(self, idx: int):
        return self.cache[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self.cache[idx] = value

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value) -> None:
        self.cache = value

    def filter(self, batch_indices):
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other):
        def cat(lhs, rhs):
            if lhs is None:
                return rhs
            if rhs is None:
                return lhs
            return mx.concatenate([lhs, rhs])

        self.cache = [cat(lhs, rhs) for lhs, rhs in zip(self.cache, other.cache, strict=True)]

    def extract(self, idx):
        cache = RecurrentRollbackCache(len(self.cache), conv_kernel_size=self.conv_kernel_size)
        cache.cache = [c[idx : idx + 1] if c is not None else None for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = None if lengths is None else mx.array(lengths)

    def finalize(self):
        self.lengths = None
        self.left_padding = None

    def advance(self, n: int):
        if self.lengths is not None:
            self.lengths -= n
        if self.left_padding is not None:
            self.left_padding -= n

    def make_mask(self, n: int):
        if self.left_padding is not None:
            positions = mx.arange(n)
            return positions >= self.left_padding[:, None]
        if self.lengths is not None:
            positions = mx.arange(n)
            return positions < self.lengths[:, None]
        return None

    def empty(self):
        return self.cache[0] is None

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)

    def checkpoint(self) -> None:
        self._snapshot = list(self.cache)

    def arm_rollback(self, prefix_len: int = 0) -> None:
        del prefix_len
        self._armed = True
        self._tape = None
        self._tape_k = None
        self._tape_g = None
        self._tape_qkv = None
        self.checkpoint()

    def clear_rollback_state(self) -> None:
        self._armed = False
        self._tape = None
        self._tape_k = None
        self._tape_g = None
        self._tape_qkv = None
        self._snapshot = None

    def record_tape(
        self,
        *,
        tape: mx.array,
        k: mx.array,
        g: mx.array,
        qkv: mx.array,
    ) -> None:
        self._tape = mx.contiguous(tape)
        self._tape_k = mx.contiguous(k)
        self._tape_g = mx.contiguous(g)
        self._tape_qkv = mx.contiguous(qkv)

    def _rebuild_conv_state(self, accepted_steps: int) -> mx.array | None:
        if self._tape_qkv is None:
            return self.cache[0]
        keep = self.conv_kernel_size - 1
        if keep <= 0:
            return None
        conv_state = self._snapshot[0] if self._snapshot is not None else None
        if conv_state is None:
            prefix = mx.zeros(
                (self._tape_qkv.shape[0], keep, self._tape_qkv.shape[-1]),
                dtype=self._tape_qkv.dtype,
            )
        else:
            prefix = conv_state
        conv_input = mx.concatenate([prefix, self._tape_qkv], axis=1)
        start = accepted_steps
        end = min(start + keep, int(conv_input.shape[1]))
        return mx.contiguous(conv_input[:, start:end, :])

    def rollback(self, acceptance_length: int) -> None:
        if self._snapshot is None:
            return
        self.cache = list(self._snapshot)
        if (
            self._tape is not None
            and self._tape_k is not None
            and self._tape_g is not None
            and self.cache[1] is not None
        ):
            accepted_steps = int(acceptance_length) + 1
            state = tape_replay_kernel(
                self._tape[:, :accepted_steps],
                self._tape_k[:, :accepted_steps],
                self._tape_g[:, :accepted_steps],
                self.cache[1],
                None,
            )
            self.cache[1] = state
            self.cache[0] = self._rebuild_conv_state(accepted_steps)
        self.clear_rollback_state()


def _target_text_wrapper(target_model: Any) -> Any:
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    return target_model


def _target_text_model(target_model: Any) -> Any:
    wrapper = _target_text_wrapper(target_model)
    if hasattr(wrapper, "model"):
        return wrapper.model
    raise AttributeError(f"Unsupported target text model: {type(wrapper)!r}")


def _attention_num_heads(attn: Any) -> int:
    for attr in ("num_attention_heads", "n_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing attention head count attribute")


def _attention_num_kv_heads(attn: Any) -> int:
    for attr in ("num_key_value_heads", "n_kv_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing KV head count attribute")


def _attention_has_gated_q_proj(attn: Any) -> bool:
    q_proj = getattr(attn, "q_proj", None)
    q_norm = getattr(attn, "q_norm", None)
    q_proj_weight = getattr(q_proj, "weight", None)
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_proj_weight is None or q_norm_weight is None:
        return False
    try:
        num_attention_heads = _attention_num_heads(attn)
    except AttributeError:
        return False
    expected_out_dim = 2 * num_attention_heads * int(q_norm_weight.shape[0])
    return int(q_proj_weight.shape[0]) == expected_out_dim


class _ExactSmallProjPad(nn.Module):
    def __init__(self, linear: nn.Module, *, pad_m: int = 16):
        super().__init__()
        self.linear = linear
        self.pad_m = int(pad_m)
        self._dflash_exact_small_proj_wrapped = True

    @property
    def weight(self) -> mx.array:
        return self.linear.weight

    @weight.setter
    def weight(self, value: mx.array) -> None:
        self.linear.weight = value

    @property
    def bias(self):
        return getattr(self.linear, "bias", None)

    @bias.setter
    def bias(self, value) -> None:
        self.linear.bias = value

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3 and x.shape[1] < self.pad_m:
            batch_size, seq_len, hidden_dim = x.shape
            pad = mx.zeros((batch_size, self.pad_m - seq_len, hidden_dim), dtype=x.dtype)
            out = self.linear(mx.concatenate([x, pad], axis=1))
            return out[:, :seq_len, :]
        return self.linear(x)


def _install_exact_small_proj_hooks(linear_attn: Any) -> None:
    for attr_name in ("in_proj_b", "in_proj_a"):
        proj = getattr(linear_attn, attr_name, None)
        if proj is None or getattr(proj, "_dflash_exact_small_proj_wrapped", False):
            continue
        setattr(linear_attn, attr_name, _ExactSmallProjPad(proj))


def _install_speculative_linear_cache_hook(linear_attn: Any) -> None:
    cls = type(linear_attn)
    if getattr(cls, "_dflash_speculative_call_installed", False):
        return

    original_call = cls.__call__

    def speculative_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not isinstance(cache, RecurrentRollbackCache) or not getattr(cache, "_armed", False):
            return original_call(self, inputs, mask=mask, cache=cache)

        from mlx.nn.layers.distributed import sum_gradients

        batch_size, seq_len, _ = inputs.shape
        sharding_group = getattr(self, "sharding_group", None)

        if sharding_group is not None:
            inputs = sum_gradients(sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z_proj = self.in_proj_z(inputs)
        z = z_proj.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (batch_size, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(batch_size, seq_len, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]

        state = cache[1]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        g = gated_delta_mod.compute_g(self.A_log, a, self.dt_bias)
        beta = mx.sigmoid(b)

        if state is None:
            _, _, _, d_k = q.shape
            h_v, d_v = v.shape[-2:]
            state = mx.zeros((batch_size, h_v, d_v, d_k), dtype=q.dtype)
        state_in = state

        if (
            mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and not self.training
        ):
            out, state, innovation_tape = gated_delta_kernel_with_tape(
                q,
                k,
                v,
                g,
                beta,
                state,
                mask,
            )
            cache.record_tape(
                tape=innovation_tape,
                k=k,
                g=g,
                qkv=qkv,
            )
        else:
            out, state = gated_delta_mod.gated_delta_ops(q, k, v, g, beta, state, mask)
            decay = g[..., None, :] if g.ndim == 4 else g[..., None, None]
            decayed_state = state_in[:, None, ...] * decay
            kv_mem = (decayed_state * k[..., None, :]).sum(axis=-1)
            innovation_tape = (v - kv_mem) * beta[..., None]
            cache.record_tape(
                tape=innovation_tape,
                k=k,
                g=g,
                qkv=qkv,
            )

        cache[1] = state
        out = self.norm(out, z)
        out_flat = out.reshape(batch_size, seq_len, -1)
        out = self.out_proj(out_flat)

        if sharding_group is not None:
            out = mx.distributed.all_sum(out, group=sharding_group)

        return out

    cls.__call__ = speculative_call
    cls._dflash_speculative_call_installed = True


def _split_sdpa_mask(
    mask: Optional[Any],
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> Optional[Any]:
    if mask is None or mask == "causal":
        return mask
    return mask[..., query_start:query_end, :key_end]


def _split_sdpa_output(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any],
    chunk_size: int,
    cached_prefix_len: int,
) -> mx.array:
    query_len = int(queries.shape[2])
    if query_len <= chunk_size:
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        )

    outputs: list[mx.array] = []
    for start in range(0, query_len, chunk_size):
        end = min(start + chunk_size, query_len)
        key_end = cached_prefix_len + end
        chunk_mask = _split_sdpa_mask(mask, query_start=start, query_end=end, key_end=key_end)
        outputs.append(
            scaled_dot_product_attention(
                queries[:, :, start:end, :],
                keys[:, :, :key_end, :],
                values[:, :, :key_end, :],
                cache=cache,
                scale=scale,
                mask=chunk_mask,
            )
        )
    return mx.concatenate(outputs, axis=2)


def _install_split_full_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_split_full_attention_installed", False):
        return

    original_call = cls.__call__

    def split_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not getattr(self, "_dflash_split_sdpa_enabled", False):
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        batch_size, seq_len, _ = x.shape
        q_proj_output = self.q_proj(x)
        num_attention_heads = _attention_num_heads(self)
        num_key_value_heads = _attention_num_kv_heads(self)
        queries, gate = mx.split(
            q_proj_output.reshape(batch_size, seq_len, num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(batch_size, seq_len, num_key_value_heads, -1)).transpose(
            0,
            2,
            1,
            3,
        )
        values = values.reshape(batch_size, seq_len, num_key_value_heads, -1).transpose(
            0,
            2,
            1,
            3,
        )

        cached_prefix_len = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0
        if cache is not None:
            queries = self.rope(queries, offset=cached_prefix_len)
            keys = self.rope(keys, offset=cached_prefix_len)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        total_kv_len = int(keys.shape[2])
        exact_prefix_threshold = int(
            getattr(
                self,
                "_dflash_split_sdpa_exact_kv_threshold",
                1024,
            )
        )
        chunk_size = int(getattr(self, "_dflash_split_sdpa_chunk_size", 8))
        should_split = (
            cache is not None
            and cached_prefix_len >= exact_prefix_threshold
            and (mask is None or mask == "causal" or isinstance(mask, mx.array))
        )
        should_use_batched_2pass = (
            should_split
            and int(queries.shape[2]) == 16
            and queries.dtype in (mx.bfloat16, mx.float16)
            and int(queries.shape[-1]) in (128, 256)
            and int(values.shape[-1]) in (128, 256)
        )
        if should_use_batched_2pass:
            output = batched_sdpa_2pass_exact(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask if isinstance(mask, mx.array) else None,
            )
            if output is None:
                output = _split_sdpa_output(
                    queries=queries,
                    keys=keys,
                    values=values,
                    scale=self.scale,
                    mask=mask,
                    cache=cache,
                    chunk_size=chunk_size,
                    cached_prefix_len=cached_prefix_len,
                )
        elif should_split:
            output = _split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask,
                cache=cache,
                chunk_size=chunk_size,
                cached_prefix_len=cached_prefix_len,
            )
        else:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        gated_output = output * mx.sigmoid(gate)
        return self.o_proj(gated_output)

    cls.__call__ = split_call
    cls._dflash_split_full_attention_installed = True


def _install_target_hooks(target_model: Any) -> None:
    text_model = _target_text_model(target_model)
    if getattr(text_model, "_dflash_speculative_hooks_installed", False):
        return
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            _install_exact_small_proj_hooks(layer.linear_attn)
            _install_speculative_linear_cache_hook(layer.linear_attn)
        elif not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            _install_split_full_attention_hook(layer.self_attn)
    text_model._dflash_speculative_hooks_installed = True


def configure_full_attention_split(
    target_model: Any,
    *,
    enabled: bool,
    chunk_size: int = 8,
) -> None:
    text_model = _target_text_model(target_model)
    _install_target_hooks(target_model)
    for layer in text_model.layers:
        if not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            layer.self_attn._dflash_split_sdpa_enabled = enabled
            layer.self_attn._dflash_split_sdpa_chunk_size = int(chunk_size)
            layer.self_attn._dflash_split_sdpa_exact_kv_threshold = 1024


def make_target_cache(
    target_model: Any,
    *,
    enable_speculative_linear_cache: bool,
) -> list[Any]:
    text_model = _target_text_model(target_model)
    _install_target_hooks(target_model)
    configure_full_attention_split(target_model, enabled=True)

    caches: list[Any] = []
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            if enable_speculative_linear_cache:
                conv_kernel_size = int(getattr(layer.linear_attn, "conv_kernel_size", 4))
                caches.append(
                    RecurrentRollbackCache(size=2, conv_kernel_size=conv_kernel_size)
                )
            else:
                caches.append(cache_mod.ArraysCache(size=2))
        else:
            caches.append(cache_mod.KVCache())
    return caches
