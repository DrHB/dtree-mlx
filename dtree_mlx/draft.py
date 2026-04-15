from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models import cache as cache_lib
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.utils import quantize_model


def resolve_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path
    return Path(snapshot_download(path_or_repo))


@dataclass
class DraftArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_scaling: dict | None = None
    block_size: int = 16
    dflash_config: dict | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DraftArgs":
        keys = {
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "rms_norm_eps",
            "vocab_size",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
            "head_dim",
            "tie_word_embeddings",
            "attention_bias",
            "attention_dropout",
            "rope_scaling",
            "block_size",
            "dflash_config",
        }
        return cls(**{key: config[key] for key in keys if key in config})


class ContextOnlyDraftKVCache:
    def __init__(self, sink_size: int = 64, window_size: int = 1024):
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def append_context(
        self,
        context_keys: mx.array,
        context_values: mx.array,
        num_positions: int,
    ) -> None:
        if context_keys is None or context_values is None or int(num_positions) <= 0:
            return
        if self.keys is None:
            self.keys = context_keys
            self.values = context_values
        else:
            self.keys = mx.concatenate([self.keys, context_keys], axis=2)
            self.values = mx.concatenate([self.values, context_values], axis=2)
        self.offset += int(num_positions)
        self._apply_window()

    def _apply_window(self) -> None:
        if self.keys is None or self.values is None:
            return
        cache_len = int(self.keys.shape[2])
        max_len = self.sink_size + self.window_size
        if cache_len <= max_len:
            return
        sink_keys = self.keys[:, :, : self.sink_size, :]
        sink_values = self.values[:, :, : self.sink_size, :]
        window_keys = self.keys[:, :, -self.window_size :, :]
        window_values = self.values[:, :, -self.window_size :, :]
        self.keys = mx.concatenate([sink_keys, window_keys], axis=2)
        self.values = mx.concatenate([sink_values, window_values], axis=2)

    def fetch(self) -> tuple[mx.array | None, mx.array | None]:
        return self.keys, self.values

    def trim(self, num_tokens: int) -> None:
        del num_tokens


def _is_quantized_linear(module: Any) -> bool:
    quantized_cls = getattr(nn, "QuantizedLinear", None)
    return quantized_cls is not None and isinstance(module, quantized_cls)


class DFlashAttention(nn.Module):
    def __init__(self, args: DraftArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.mask_mode = "none"

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def _project_queries(self, hidden_states: mx.array) -> mx.array:
        batch_size, query_len, _ = hidden_states.shape
        queries = self.q_proj(hidden_states)
        return self.q_norm(
            queries.reshape(batch_size, query_len, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)

    def _project_keys_values(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        batch_size, seq_len, _ = hidden_states.shape
        keys = self.k_proj(hidden_states)
        keys = self.k_norm(
            keys.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = self.v_proj(hidden_states).reshape(
            batch_size,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)
        return keys, values

    def precompute_context_kv(
        self,
        target_hidden: mx.array,
        cache: cache_lib.KVCache | ContextOnlyDraftKVCache | None,
    ) -> None:
        if cache is None:
            return
        context_len = int(target_hidden.shape[1])
        if context_len <= 0:
            return
        context_keys, context_values = self._project_keys_values(target_hidden)
        context_keys = self.rope(context_keys, offset=int(cache.offset))
        if isinstance(cache, ContextOnlyDraftKVCache):
            cache.append_context(context_keys, context_values, context_len)
        else:
            cache.update_and_fetch(context_keys, context_values)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array | None = None,
        cache: cache_lib.KVCache | ContextOnlyDraftKVCache | None = None,
    ) -> mx.array:
        batch_size, query_len, _ = hidden_states.shape
        if cache is not None and target_hidden is not None:
            self.precompute_context_kv(target_hidden, cache)
            target_hidden = None

        queries = self._project_queries(hidden_states)
        noise_keys, noise_values = self._project_keys_values(hidden_states)

        if cache is not None:
            cache_offset = int(cache.offset)
            if isinstance(cache, ContextOnlyDraftKVCache):
                queries = self.rope(queries, offset=cache_offset)
                noise_keys = self.rope(noise_keys, offset=cache_offset)
                cached_keys, cached_values = cache.fetch()
                if cached_keys is None or cached_values is None:
                    raise ValueError("Context-only draft cache is missing precomputed context.")
                if hasattr(mx.fast, "dflash_cross_attention"):
                    output = mx.fast.dflash_cross_attention(
                        queries,
                        cached_keys,
                        cached_values,
                        noise_keys,
                        noise_values,
                        scale=self.scale,
                    )
                    output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_len, -1)
                    return self.o_proj(output)
                keys = mx.concatenate([cached_keys, noise_keys], axis=-2)
                values = mx.concatenate([cached_values, noise_values], axis=-2)
            else:
                queries = self.rope(queries, offset=cache_offset)
                noise_keys = self.rope(noise_keys, offset=cache_offset)
                keys, values = cache.update_and_fetch(noise_keys, noise_values)
        else:
            if target_hidden is None:
                raise ValueError("Draft attention requires target_hidden when cache is disabled.")
            context_len = int(target_hidden.shape[1])
            context_keys, context_values = self._project_keys_values(target_hidden)
            queries = self.rope(queries, offset=context_len)
            context_keys = self.rope(context_keys, offset=0)
            noise_keys = self.rope(noise_keys, offset=context_len)
            if hasattr(mx.fast, "dflash_cross_attention"):
                output = mx.fast.dflash_cross_attention(
                    queries,
                    context_keys,
                    context_values,
                    noise_keys,
                    noise_values,
                    scale=self.scale,
                )
                output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_len, -1)
                return self.o_proj(output)
            keys = mx.concatenate([context_keys, noise_keys], axis=-2)
            values = mx.concatenate([context_values, noise_values], axis=-2)

        mask = None
        if not isinstance(cache, ContextOnlyDraftKVCache):
            mask = "causal" if self.mask_mode == "causal" and query_len > 1 else None
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None if isinstance(cache, ContextOnlyDraftKVCache) else cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_len, -1)
        return self.o_proj(output)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, args: DraftArgs):
        super().__init__()
        self.self_attn = DFlashAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array | None = None,
        cache: cache_lib.KVCache | ContextOnlyDraftKVCache | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(mx.fast, "dflash_gated_mlp") and not _is_quantized_linear(
            self.mlp.gate_proj
        ):
            hidden_states = mx.fast.dflash_gated_mlp(
                hidden_states,
                self.mlp.gate_proj.weight,
                self.mlp.up_proj.weight,
                self.mlp.down_proj.weight,
            )
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, args: DraftArgs):
        super().__init__()
        self.args = args
        self.layers = [DFlashDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.target_layer_ids = list(args.dflash_config["target_layer_ids"])
        self.fc = nn.Linear(
            len(self.target_layer_ids) * args.hidden_size,
            args.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.block_size = args.block_size
        self.mask_token_id = int(args.dflash_config["mask_token_id"])
        self.attention_mask_mode = "none"
        self.cache_mode = "kv"

    def make_cache(self) -> list[cache_lib.KVCache | ContextOnlyDraftKVCache]:
        if self.cache_mode == "context-only":
            return [ContextOnlyDraftKVCache() for _ in self.layers]
        return [cache_lib.KVCache() for _ in self.layers]

    def prepare_target_hidden(self, target_hidden: mx.array) -> mx.array:
        return self.hidden_norm(self.fc(target_hidden))

    def precompute_context_kv(
        self,
        target_hidden: mx.array,
        cache: list[cache_lib.KVCache | ContextOnlyDraftKVCache] | None,
    ) -> None:
        if cache is None:
            return

        for layer, layer_cache in zip(self.layers, cache):
            if layer_cache is None:
                continue
            layer.self_attn.precompute_context_kv(target_hidden, layer_cache)

    def __call__(
        self,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: list[cache_lib.KVCache | ContextOnlyDraftKVCache] | None = None,
    ) -> mx.array:
        hidden_states = noise_embedding
        target_hidden = self.prepare_target_hidden(target_hidden)
        if cache is None:
            cache = [None] * len(self.layers)
        else:
            self.precompute_context_kv(target_hidden, cache)
        for layer, layer_cache in zip(self.layers, cache):
            layer.self_attn.mask_mode = self.attention_mask_mode
            layer_target_hidden = None if layer_cache is not None else target_hidden
            hidden_states = layer(hidden_states, layer_target_hidden, cache=layer_cache)
        return self.norm(hidden_states)


def load_draft_model(path_or_repo: str) -> tuple[DFlashDraftModel, Path]:
    model_path = resolve_model_path(path_or_repo)
    config = json.loads((model_path / "config.json").read_text())
    draft = DFlashDraftModel(DraftArgs.from_dict(config))

    weights: list[tuple[str, mx.array]] = []
    for weight_file in sorted(model_path.glob("model*.safetensors")):
        weights.extend(mx.load(str(weight_file)).items())
    if not weights:
        raise FileNotFoundError(f"No draft weights found in {model_path}")
    draft.load_weights(weights)
    mx.eval(draft.parameters())
    return draft, model_path


def maybe_quantize_draft_model(
    draft: DFlashDraftModel,
    bits: int | None,
    group_size: int,
) -> dict[str, Any] | None:
    if bits is None:
        return None
    _, quantized_config = quantize_model(
        model=draft,
        config={},
        group_size=group_size,
        bits=bits,
    )
    mx.eval(draft.parameters())
    return quantized_config.get("quantization")
