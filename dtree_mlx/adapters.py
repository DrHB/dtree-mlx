from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models import cache as cache_lib
from mlx_lm.models import qwen3

from .model_prep import prepare_custom_model
from .qwen35_tree import commit_qwen35_tree_path, forward_qwen35_tree_with_hidden_states
from .qwen35_target import make_target_cache as make_qwen35_target_cache
from .qwen3_tree import forward_transformer_block_with_position_ids


def resolve_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path
    return Path(snapshot_download(path_or_repo))


class MLXTargetAdapter:
    family: str = "unknown"

    def resolve_target_model_path(self, path_or_repo: str) -> Path:
        return resolve_model_path(path_or_repo)

    def build_prompt(self, tokenizer, prompt_text: str) -> mx.array:
        raise NotImplementedError

    def stop_token_ids(self, tokenizer) -> set[int]:
        raise NotImplementedError

    def make_cache(
        self,
        model,
        speculative_linear_cache: bool = False,
    ) -> list[Any]:
        del speculative_linear_cache
        return model.make_cache()

    def embed_tokens(self, model, tokens: mx.array) -> mx.array:
        raise NotImplementedError

    def lm_head_logits(self, model, hidden_states: mx.array) -> mx.array:
        raise NotImplementedError

    def lm_head_argmax(self, model, hidden_states: mx.array) -> mx.array:
        # Hook for greedy verifier experiments; architecture-specific adapters
        # can replace this with a fused top-1 LM-head kernel.
        logits = self.lm_head_logits(model, hidden_states)
        return mx.argmax(logits, axis=-1).astype(mx.uint32)

    def forward_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        raise NotImplementedError

    def forward_verifier_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        raise NotImplementedError(
            f"{self.family} does not expose verifier states before the LM head."
        )

    def forward_tree_step_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        logits, target_hidden = self.forward_with_hidden_states(
            model,
            inputs,
            cache,
            layer_ids,
            return_rollback_records=False,
        )
        return logits, target_hidden

    def forward_accept_all_block(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        logits, target_hidden = self.forward_with_hidden_states(
            model,
            inputs,
            cache,
            layer_ids,
            return_rollback_records=False,
        )
        return logits[:, -1:, :], target_hidden

    def supports_tree_verification(self) -> bool:
        return False

    def forward_tree_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        parents: list[int],
        position_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        raise NotImplementedError(
            f"{self.family} does not expose explicit tree verification."
        )

    def snapshot_linear_caches(
        self,
        model,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        raise NotImplementedError

    def restore_linear_caches(
        self,
        model,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        raise NotImplementedError

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        raise NotImplementedError

    def rollback_linear_caches(
        self,
        model,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        raise NotImplementedError

    def cache_summary(self, cache: list[Any]) -> str:
        raise NotImplementedError

    def compact_kv_caches(
        self,
        model,
        cache: list[Any],
        past_length: int,
        keep_current_indices: list[int],
    ) -> None:
        raise NotImplementedError(
            f"{self.family} does not implement DTree cache compaction."
        )


class Qwen35TargetAdapter(MLXTargetAdapter):
    family = "qwen3_5"

    def __init__(self) -> None:
        self._pending_tree_state = None

    def resolve_target_model_path(self, path_or_repo: str) -> Path:
        model_path = resolve_model_path(path_or_repo)
        config = json.loads((model_path / "config.json").read_text())
        if (
            config.get("model_type") == "qwen3_5"
            and config.get("model_file") != "custom_qwen35_dflash_model.py"
        ):
            source_id = path_or_repo if not Path(path_or_repo).exists() else str(model_path)
            return prepare_custom_model(source_id)
        return model_path

    def build_prompt(self, tokenizer, prompt_text: str) -> mx.array:
        messages = [{"role": "user", "content": prompt_text}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return mx.array(tokens, dtype=mx.uint32)

    def stop_token_ids(self, tokenizer) -> set[int]:
        eos_token_ids = tokenizer.eos_token_ids
        if isinstance(eos_token_ids, int):
            return {eos_token_ids}
        return set(eos_token_ids)

    def make_cache(
        self,
        model,
        speculative_linear_cache: bool = False,
    ) -> list[Any]:
        return make_qwen35_target_cache(
            model,
            enable_speculative_linear_cache=speculative_linear_cache,
        )

    def embed_tokens(self, model, tokens: mx.array) -> mx.array:
        return model.language_model.model.embed_tokens(tokens)

    def lm_head_logits(self, model, hidden_states: mx.array) -> mx.array:
        language_model = model.language_model
        text_model = language_model.model
        if language_model.args.tie_word_embeddings:
            return text_model.embed_tokens.as_linear(hidden_states)
        return language_model.lm_head(hidden_states)

    def forward_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        if not hasattr(model, "forward_dflash"):
            raise NotImplementedError(
                "Qwen3.5 support requires the custom MLX model wrapper."
            )
        return model.forward_dflash(
            inputs=inputs,
            cache=cache,
            layer_ids=layer_ids,
            return_rollback_records=return_rollback_records,
        )

    def forward_verifier_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        if hasattr(model, "language_model") and hasattr(
            model.language_model.model,
            "forward_dflash",
        ):
            return model.language_model.model.forward_dflash(
                inputs=inputs,
                cache=cache,
                layer_ids=layer_ids,
                return_rollback_records=True,
            )
        raise NotImplementedError(
            "Qwen3.5 lazy-logit verification requires the custom MLX model fork."
        )

    def forward_tree_step_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        if hasattr(model, "language_model") and hasattr(
            model.language_model.model,
            "forward_tree_step",
        ):
            return model.language_model.model.forward_tree_step(
                inputs=inputs,
                cache=cache,
                layer_ids=layer_ids,
            )
        return super().forward_tree_step_with_hidden_states(
            model,
            inputs,
            cache,
            layer_ids,
        )

    def forward_accept_all_block(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        if hasattr(model, "language_model") and hasattr(
            model.language_model.model,
            "forward_dflash",
        ):
            norm_hidden_states, target_hidden = model.language_model.model.forward_dflash(
                inputs=inputs,
                cache=cache,
                layer_ids=layer_ids,
                return_rollback_records=False,
            )
            return self.lm_head_logits(model, norm_hidden_states[:, -1:, :]), target_hidden
        return super().forward_accept_all_block(model, inputs, cache, layer_ids)

    def snapshot_linear_caches(
        self,
        model,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        if hasattr(model, "snapshot_linear_caches"):
            return model.snapshot_linear_caches(cache)
        raise NotImplementedError(
            "Qwen3.5 linear-cache snapshots require the custom MLX model wrapper."
        )

    def restore_linear_caches(
        self,
        model,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        if hasattr(model, "restore_linear_caches"):
            model.restore_linear_caches(cache, snapshots)
            return
        raise NotImplementedError(
            "Qwen3.5 linear-cache restore requires the custom MLX model wrapper."
        )

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        for layer_cache in cache:
            if isinstance(layer_cache, cache_lib.KVCache):
                layer_cache.trim(num_tokens)

    def rollback_linear_caches(
        self,
        model,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        if hasattr(model, "rollback_linear_caches"):
            model.rollback_linear_caches(cache, rollback_records, accepted_inputs)
            return
        raise NotImplementedError(
            "Qwen3.5 linear-cache rollback requires the custom MLX model wrapper."
        )

    def cache_summary(self, cache: list[Any]) -> str:
        parts: list[str] = []
        for idx, layer_cache in enumerate(cache):
            if isinstance(layer_cache, cache_lib.KVCache):
                parts.append(f"{idx}:kv={layer_cache.offset}")
            else:
                recurrent = None
                try:
                    state = layer_cache[1]
                    recurrent = None if state is None else tuple(state.shape)
                except Exception:
                    pass
                parts.append(f"{idx}:ssm={recurrent}")
        return " ".join(parts)

    def supports_tree_verification(self) -> bool:
        return True

    def forward_tree_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        parents: list[int],
        position_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        norm_hidden_states, target_hidden, pending = forward_qwen35_tree_with_hidden_states(
            model,
            inputs,
            cache,
            layer_ids,
            parents,
            position_ids,
            attention_mask,
        )
        self._pending_tree_state = pending
        return norm_hidden_states, target_hidden

    def compact_kv_caches(
        self,
        model,
        cache: list[Any],
        past_length: int,
        keep_current_indices: list[int],
    ) -> None:
        del model, past_length
        if self._pending_tree_state is None:
            raise RuntimeError("Qwen3.5 tree compaction requires pending tree state.")
        commit_qwen35_tree_path(
            cache,
            self._pending_tree_state,
            keep_current_indices,
        )
        self._pending_tree_state = None


class Qwen3TargetAdapter(MLXTargetAdapter):
    family = "qwen3"

    def build_prompt(self, tokenizer, prompt_text: str) -> mx.array:
        messages = [{"role": "user", "content": prompt_text}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return mx.array(tokens, dtype=mx.uint32)

    def stop_token_ids(self, tokenizer) -> set[int]:
        eos_token_ids = tokenizer.eos_token_ids
        if isinstance(eos_token_ids, int):
            return {eos_token_ids}
        return set(eos_token_ids)

    def make_cache(
        self,
        model,
        speculative_linear_cache: bool = False,
    ) -> list[Any]:
        del speculative_linear_cache
        return [cache_lib.KVCache() for _ in model.layers]

    def embed_tokens(self, model, tokens: mx.array) -> mx.array:
        return model.model.embed_tokens(tokens)

    def lm_head_logits(self, model, hidden_states: mx.array) -> mx.array:
        if model.args.tie_word_embeddings:
            return model.model.embed_tokens.as_linear(hidden_states)
        return model.lm_head(hidden_states)

    def _forward_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        text_model = model.model
        hidden_states = text_model.embed_tokens(inputs)
        mask = qwen3.create_attention_mask(hidden_states, cache[0])

        selected_hidden_states: list[mx.array] = []
        target_layer_ids = set(layer_ids)
        for idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            if idx in target_layer_ids:
                selected_hidden_states.append(hidden_states)

        norm_hidden_states = text_model.norm(hidden_states)
        target_hidden = mx.concatenate(selected_hidden_states, axis=-1)
        return norm_hidden_states, target_hidden

    def forward_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        norm_hidden_states, target_hidden = self._forward_states(
            model,
            inputs,
            cache,
            layer_ids,
        )
        logits = self.lm_head_logits(model, norm_hidden_states)
        if return_rollback_records:
            return logits, target_hidden, {}
        return logits, target_hidden

    def forward_verifier_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        norm_hidden_states, target_hidden = self._forward_states(
            model,
            inputs,
            cache,
            layer_ids,
        )
        return norm_hidden_states, target_hidden, {}

    def forward_accept_all_block(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        norm_hidden_states, target_hidden = self._forward_states(
            model,
            inputs,
            cache,
            layer_ids,
        )
        return self.lm_head_logits(model, norm_hidden_states[:, -1:, :]), target_hidden

    def supports_tree_verification(self) -> bool:
        return True

    def forward_tree_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        parents: list[int],
        position_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        del parents
        text_model = model.model
        hidden_states = text_model.embed_tokens(inputs)

        selected_hidden_states: list[mx.array] = []
        target_layer_ids = set(layer_ids)
        for idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
            hidden_states = forward_transformer_block_with_position_ids(
                layer=layer,
                hidden_states=hidden_states,
                mask=attention_mask,
                cache=layer_cache,
                position_ids=position_ids,
            )
            if idx in target_layer_ids:
                selected_hidden_states.append(hidden_states)

        norm_hidden_states = text_model.norm(hidden_states)
        target_hidden = mx.concatenate(selected_hidden_states, axis=-1)
        return norm_hidden_states, target_hidden

    def snapshot_linear_caches(
        self,
        model,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        return {}

    def restore_linear_caches(
        self,
        model,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        return None

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        for layer_cache in cache:
            if isinstance(layer_cache, cache_lib.KVCache):
                layer_cache.trim(num_tokens)

    def rollback_linear_caches(
        self,
        model,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        return None

    def cache_summary(self, cache: list[Any]) -> str:
        return " ".join(
            f"{idx}:kv={layer_cache.offset}"
            for idx, layer_cache in enumerate(cache)
            if isinstance(layer_cache, cache_lib.KVCache)
        )

    def compact_kv_caches(
        self,
        model,
        cache: list[Any],
        past_length: int,
        keep_current_indices: list[int],
    ) -> None:
        keep_count = len(keep_current_indices)
        keep_array = mx.array(keep_current_indices, dtype=mx.uint32) if keep_count else None

        for layer_cache in cache:
            if not isinstance(layer_cache, cache_lib.KVCache):
                continue
            current_length = layer_cache.offset - past_length
            if current_length <= 0:
                continue
            if keep_count == 0:
                layer_cache.offset = past_length
                continue
            if keep_count == current_length:
                continue
            kept_keys = mx.take(
                layer_cache.keys[..., past_length : layer_cache.offset, :],
                keep_array,
                axis=2,
            )
            kept_values = mx.take(
                layer_cache.values[..., past_length : layer_cache.offset, :],
                keep_array,
                axis=2,
            )
            layer_cache.keys[..., past_length : past_length + keep_count, :] = kept_keys
            layer_cache.values[..., past_length : past_length + keep_count, :] = kept_values
            layer_cache.offset = past_length + keep_count


ADAPTERS: dict[str, type[MLXTargetAdapter]] = {
    "qwen3": Qwen3TargetAdapter,
    "qwen3_5": Qwen35TargetAdapter,
}


def adapter_for_model_type(model_type: str) -> type[MLXTargetAdapter] | None:
    return ADAPTERS.get(model_type)


@dataclass
class LoadedTargetModel:
    requested_model: str
    resolved_model_path: Path
    model: Any
    tokenizer: Any
    adapter: MLXTargetAdapter

    def build_prompt(self, prompt_text: str) -> mx.array:
        return self.adapter.build_prompt(self.tokenizer, prompt_text)

    def stop_token_ids(self) -> set[int]:
        return self.adapter.stop_token_ids(self.tokenizer)

    def make_cache(
        self,
        speculative_linear_cache: bool = False,
    ) -> list[Any]:
        return self.adapter.make_cache(
            self.model,
            speculative_linear_cache=speculative_linear_cache,
        )

    def embed_tokens(self, tokens: mx.array) -> mx.array:
        return self.adapter.embed_tokens(self.model, tokens)

    def lm_head_logits(self, hidden_states: mx.array) -> mx.array:
        return self.adapter.lm_head_logits(self.model, hidden_states)

    def lm_head_argmax(self, hidden_states: mx.array) -> mx.array:
        return self.adapter.lm_head_argmax(self.model, hidden_states)

    def forward_with_hidden_states(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        return self.adapter.forward_with_hidden_states(
            self.model,
            inputs,
            cache,
            layer_ids,
            return_rollback_records=return_rollback_records,
        )

    def forward_verifier_states(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        return self.adapter.forward_verifier_states(
            self.model,
            inputs,
            cache,
            layer_ids,
        )

    def forward_tree_step_with_hidden_states(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        return self.adapter.forward_tree_step_with_hidden_states(
            self.model,
            inputs,
            cache,
            layer_ids,
        )

    def forward_accept_all_block(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        return self.adapter.forward_accept_all_block(
            self.model,
            inputs,
            cache,
            layer_ids,
        )

    def supports_tree_verification(self) -> bool:
        return self.adapter.supports_tree_verification()

    def forward_tree_with_hidden_states(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        parents: list[int],
        position_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return self.adapter.forward_tree_with_hidden_states(
            self.model,
            inputs,
            cache,
            layer_ids,
            parents,
            position_ids,
            attention_mask,
        )

    def snapshot_linear_caches(
        self,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        return self.adapter.snapshot_linear_caches(self.model, cache)

    def restore_linear_caches(
        self,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        self.adapter.restore_linear_caches(self.model, cache, snapshots)

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        self.adapter.rewind_kv_caches(cache, num_tokens)

    def rollback_linear_caches(
        self,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        self.adapter.rollback_linear_caches(
            self.model,
            cache,
            rollback_records,
            accepted_inputs,
        )

    def cache_summary(self, cache: list[Any]) -> str:
        return self.adapter.cache_summary(cache)

    def compact_kv_caches(
        self,
        cache: list[Any],
        past_length: int,
        keep_current_indices: list[int],
    ) -> None:
        self.adapter.compact_kv_caches(
            self.model,
            cache,
            past_length,
            keep_current_indices,
        )


def load_target_model(path_or_repo: str) -> LoadedTargetModel:
    base_path = resolve_model_path(path_or_repo)
    config = json.loads((base_path / "config.json").read_text())
    model_type = config.get("model_type")
    adapter_cls = adapter_for_model_type(model_type)
    if adapter_cls is None:
        registered = ", ".join(sorted(ADAPTERS))
        raise NotImplementedError(
            f"Unsupported MLX DFlash target model_type={model_type!r} for "
            f"{path_or_repo!r}. A matching DFlash draft checkpoint is not enough; "
            "the target family also needs an MLX adapter for hidden-state "
            "extraction and exact cache rollback. Current adapters: "
            f"{registered}. See ADDING_MODELS.md."
        )

    adapter = adapter_cls()
    resolved_model_path = adapter.resolve_target_model_path(path_or_repo)
    model, tokenizer = load(str(resolved_model_path))
    return LoadedTargetModel(
        requested_model=path_or_repo,
        resolved_model_path=resolved_model_path,
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
    )
