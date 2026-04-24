from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from .adapters import LoadedTargetModel
from .runtime import (
    arm_target_rollback_with_prefix,
    generated_token_count,
    peak_memory_gb,
    sample_tokens,
    stop_position,
    verify_block_parallel_greedy_argmax,
)


@dataclass
class ARSpecResult:
    text: str
    output_tokens: list[int]
    generated_tokens: list[int]
    metrics: dict[str, Any]


@dataclass
class BridgeResult:
    target_suffix: list[int]
    rejected: bool


@dataclass
class DraftTokenCandidate:
    token_id: int
    logprob: float
    probability: float


@dataclass
class GreedyDraftProposal:
    token_ids: list[int]
    text: str
    expansions: int


@dataclass
class ARTreeBuild:
    tree_tokens: list[int]
    child_maps: list[dict[int, int]]
    bridge_rejects: int
    draft_expansions: int
    duplicate_paths: int


class ARDraftState:
    def __init__(self, model: Any, tokenizer: Any, context_text: str):
        self.model = model
        self.tokenizer = tokenizer
        self.context_text = ""
        self.tokens: list[int] = []
        self.cache: list[Any] = []
        self.logits: mx.array | None = None
        self.cache_rebuilds = 0
        self.incremental_tokens = 0
        self.rebuild(context_text)

    def rebuild(self, context_text: str) -> None:
        tokens = encode_text(self.tokenizer, context_text)
        if not tokens:
            raise ValueError("AR draft context cannot be empty.")
        cache = make_ar_cache(self.model)
        logits = self.model(mx.array(tokens, dtype=mx.uint32)[None], cache=cache)
        mx.eval(logits)
        self.context_text = context_text
        self.tokens = tokens
        self.cache = cache
        self.logits = logits
        self.cache_rebuilds += 1

    def advance_to(self, context_text: str) -> None:
        if context_text == self.context_text:
            return
        if not context_text.startswith(self.context_text):
            self.rebuild(context_text)
            return

        next_tokens = encode_text(self.tokenizer, context_text)
        if len(next_tokens) < len(self.tokens) or next_tokens[: len(self.tokens)] != self.tokens:
            self.rebuild(context_text)
            return

        suffix = next_tokens[len(self.tokens) :]
        if suffix:
            logits = self.model(mx.array(suffix, dtype=mx.uint32)[None], cache=self.cache)
            mx.eval(logits)
            self.logits = logits
            self.incremental_tokens += len(suffix)
        self.tokens = next_tokens
        self.context_text = context_text

    def propose_greedy(
        self,
        *,
        draft_max: int,
        draft_min: int,
        draft_p_min: float,
    ) -> GreedyDraftProposal:
        if self.logits is None:
            raise RuntimeError("AR draft state is missing logits.")

        base_logits = self.logits
        suffix: list[int] = []
        expansions = 0
        eos_ids = _token_id_set(getattr(self.tokenizer, "eos_token_ids", None))
        eos_ids.update(_token_id_set(getattr(self.tokenizer, "eos_token_id", None)))

        logits = base_logits
        for _ in range(max(1, draft_max)):
            candidates = _topk_candidates(logits[0, -1, :], 1)
            candidate = candidates[0]
            if len(suffix) >= draft_min and candidate.probability < draft_p_min:
                break
            suffix.append(candidate.token_id)
            expansions += 1
            if candidate.token_id in eos_ids:
                break
            logits = self.model(mx.array([[candidate.token_id]], dtype=mx.uint32), cache=self.cache)
            mx.eval(logits)

        if suffix:
            trim_prompt_cache(self.cache, len(suffix))
        self.logits = base_logits
        return GreedyDraftProposal(
            token_ids=suffix,
            text=decode_tokens(self.tokenizer, suffix),
            expansions=expansions,
        )


def render_chat_prompt(tokenizer: Any, prompt_text: str) -> str:
    if getattr(tokenizer, "has_chat_template", False) or getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt_text


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def decode_tokens(tokenizer: Any, tokens: list[int]) -> str:
    if not tokens:
        return ""
    return tokenizer.decode(tokens, skip_special_tokens=False)


def generated_text_for_target(
    target: LoadedTargetModel,
    output_tokens: list[int],
    prompt_len: int,
) -> str:
    return decode_tokens(target.tokenizer, output_tokens[prompt_len:])


def bridge_draft_suffix_to_target(
    target_tokenizer: Any,
    target_prefix_tokens: list[int],
    target_prefix_text: str,
    draft_suffix_text: str,
) -> BridgeResult:
    if not draft_suffix_text:
        return BridgeResult(target_suffix=[], rejected=True)

    encoded = encode_text(target_tokenizer, target_prefix_text + draft_suffix_text)
    prefix_len = len(target_prefix_tokens)
    if len(encoded) < prefix_len or encoded[:prefix_len] != target_prefix_tokens:
        return BridgeResult(target_suffix=[], rejected=True)
    return BridgeResult(target_suffix=encoded[prefix_len:], rejected=False)


def _topk_candidates(logits: mx.array, topk: int) -> list[DraftTokenCandidate]:
    topk = max(1, min(int(topk), int(logits.shape[-1])))
    logits = logits.astype(mx.float32)
    partition = mx.argpartition(logits, kth=int(logits.shape[-1]) - topk, axis=-1)[-topk:]
    top_logits = mx.take(logits, partition)
    order = mx.argsort(-top_logits)
    top_token_ids = mx.take(partition, order)
    top_logits = mx.take(top_logits, order)
    log_probs = top_logits - mx.logsumexp(logits, axis=-1)
    probs = mx.exp(log_probs)
    mx.eval(top_token_ids, log_probs, probs)
    return [
        DraftTokenCandidate(
            token_id=int(token_id),
            logprob=float(logprob),
            probability=float(probability),
        )
        for token_id, logprob, probability in zip(
            top_token_ids.tolist(),
            log_probs.tolist(),
            probs.tolist(),
        )
    ]


def draft_next_candidates(
    draft_model: Any,
    context_tokens: list[int],
    topk: int,
) -> list[DraftTokenCandidate]:
    if not context_tokens:
        raise ValueError("AR draft requires a non-empty context.")
    cache = make_ar_cache(draft_model)
    logits = draft_model(mx.array(context_tokens, dtype=mx.uint32)[None], cache=cache)
    mx.eval(logits)
    return _topk_candidates(logits[0, -1, :], topk)


def draft_greedy_suffix(
    draft_model: Any,
    draft_tokenizer: Any,
    draft_context_text: str,
    *,
    draft_max: int,
    draft_min: int,
    draft_p_min: float,
) -> tuple[list[int], str, int]:
    context_tokens = encode_text(draft_tokenizer, draft_context_text)
    cache = make_ar_cache(draft_model)
    logits = draft_model(mx.array(context_tokens, dtype=mx.uint32)[None], cache=cache)
    mx.eval(logits)

    eos_ids = _token_id_set(getattr(draft_tokenizer, "eos_token_ids", None))
    eos_ids.update(_token_id_set(getattr(draft_tokenizer, "eos_token_id", None)))

    suffix: list[int] = []
    expansions = 1
    for _ in range(max(1, draft_max)):
        candidates = _topk_candidates(logits[0, -1, :], 1)
        candidate = candidates[0]
        if len(suffix) >= draft_min and candidate.probability < draft_p_min:
            break
        suffix.append(candidate.token_id)
        if candidate.token_id in eos_ids:
            break
        logits = draft_model(mx.array([[candidate.token_id]], dtype=mx.uint32), cache=cache)
        mx.eval(logits)
        expansions += 1

    return suffix, decode_tokens(draft_tokenizer, suffix), expansions


def _token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def make_ar_cache(model: Any) -> list[Any]:
    make_cache = getattr(model, "make_cache", None)
    if callable(make_cache):
        return make_cache()
    return make_prompt_cache(model)


def _insert_target_suffix(
    tree_tokens: list[int],
    child_maps: list[dict[int, int]],
    suffix: list[int],
    max_nodes: int,
) -> bool:
    if not suffix:
        return False

    current_index = 0
    inserted = False
    for token_id in suffix:
        children = child_maps[current_index]
        next_index = children.get(token_id)
        if next_index is None:
            if len(tree_tokens) >= max_nodes:
                break
            next_index = len(tree_tokens)
            tree_tokens.append(token_id)
            child_maps.append({})
            children[token_id] = next_index
            inserted = True
        current_index = next_index
    return inserted


def build_ar_target_tree(
    draft_model: Any,
    draft_tokenizer: Any,
    target_tokenizer: Any,
    *,
    root_token_id: int,
    target_prefix_tokens: list[int],
    target_prefix_text: str,
    draft_context_text: str,
    tree_budget: int,
    draft_max: int,
    draft_min: int,
    draft_p_min: float,
    branch_topk: int = 4,
) -> ARTreeBuild:
    max_nodes = max(1, int(tree_budget) + 1)
    tree_tokens = [int(root_token_id)]
    child_maps: list[dict[int, int]] = [{}]
    bridge_rejects = 0
    draft_expansions = 0
    duplicate_paths = 0

    context_tokens = encode_text(draft_tokenizer, draft_context_text)
    heap: list[tuple[float, tuple[int, ...], float]] = [(0.0, tuple(), 0.0)]
    seen_prefixes: set[tuple[int, ...]] = {tuple()}
    eos_ids = _token_id_set(getattr(draft_tokenizer, "eos_token_ids", None))
    eos_ids.update(_token_id_set(getattr(draft_tokenizer, "eos_token_id", None)))

    while heap and len(tree_tokens) < max_nodes:
        _, prefix, logprob = heapq.heappop(heap)
        if len(prefix) >= draft_max or (prefix and prefix[-1] in eos_ids):
            continue

        candidates = draft_next_candidates(
            draft_model,
            [*context_tokens, *prefix],
            topk=branch_topk,
        )
        draft_expansions += 1

        for candidate in candidates:
            if len(prefix) >= draft_min and candidate.probability < draft_p_min:
                continue
            next_prefix = (*prefix, candidate.token_id)
            if next_prefix in seen_prefixes:
                continue
            seen_prefixes.add(next_prefix)

            suffix_text = decode_tokens(draft_tokenizer, list(next_prefix))
            bridge = bridge_draft_suffix_to_target(
                target_tokenizer,
                target_prefix_tokens,
                target_prefix_text,
                suffix_text,
            )
            if bridge.rejected or not bridge.target_suffix:
                bridge_rejects += 1
            elif not _insert_target_suffix(
                tree_tokens,
                child_maps,
                bridge.target_suffix,
                max_nodes,
            ):
                duplicate_paths += 1

            if len(tree_tokens) >= max_nodes:
                break
            heapq.heappush(
                heap,
                (-(logprob + candidate.logprob), next_prefix, logprob + candidate.logprob),
            )

    return ARTreeBuild(
        tree_tokens=tree_tokens,
        child_maps=child_maps,
        bridge_rejects=bridge_rejects,
        draft_expansions=draft_expansions,
        duplicate_paths=duplicate_paths,
    )


def lazy_follow_target_tree(
    target: LoadedTargetModel,
    target_cache: list[Any],
    layer_ids: list[int],
    tree_tokens: list[int],
    child_maps: list[dict[int, int]],
    temperature: float,
) -> tuple[list[int], int]:
    accepted_indices: list[int] = []
    current_index = 0
    while True:
        logits, _ = target.forward_with_hidden_states(
            mx.array([[tree_tokens[current_index]]], dtype=mx.uint32),
            target_cache,
            layer_ids,
        )
        posterior = sample_tokens(logits[:, -1, :], temperature)
        mx.eval(posterior)
        posterior_token = int(posterior.item())
        accepted_indices.append(current_index)
        next_index = child_maps[current_index].get(posterior_token)
        if next_index is None:
            return accepted_indices, posterior_token
        current_index = next_index


def _target_prompt_tokens(target: LoadedTargetModel, prompt_text: str) -> tuple[str, list[int]]:
    prompt = render_chat_prompt(target.tokenizer, prompt_text)
    return prompt, encode_text(target.tokenizer, prompt)


def vanilla_generate_target(
    target: LoadedTargetModel,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    layer_ids: list[int],
    reset_peak_memory: bool = True,
) -> ARSpecResult:
    if reset_peak_memory:
        mx.reset_peak_memory()
    prompt_rendered, prompt_token_list = _target_prompt_tokens(target, prompt_text)
    del prompt_rendered
    prompt_tokens = mx.array(prompt_token_list, dtype=mx.uint32)
    prompt_len = int(prompt_tokens.shape[0])
    total_max_tokens = prompt_len + int(max_new_tokens)
    target_cache = target.make_cache()

    sync_start = time.perf_counter()
    logits, _ = target.forward_with_hidden_states(
        prompt_tokens[None],
        target_cache,
        layer_ids,
    )
    first_token = int(sample_tokens(logits[:, -1, :], temperature).item())
    mx.eval(logits)
    prefill_time = time.perf_counter() - sync_start

    output_tokens = prompt_token_list + [first_token]
    start = prompt_len
    decode_start = time.perf_counter()
    steps = 0
    stop_ids = target.stop_token_ids()

    while start < total_max_tokens:
        current_token = output_tokens[start]
        logits, _ = target.forward_with_hidden_states(
            mx.array([[current_token]], dtype=mx.uint32),
            target_cache,
            layer_ids,
        )
        posterior = sample_tokens(logits[:, -1, :], temperature)
        mx.eval(posterior)
        output_tokens = output_tokens[: start + 1]
        output_tokens.append(int(posterior.item()))
        start += 1
        steps += 1

        stop_idx = stop_position(output_tokens, prompt_len, stop_ids)
        if stop_idx is not None:
            output_tokens = output_tokens[: stop_idx + 1]
            break
        if len(output_tokens) > total_max_tokens:
            output_tokens = output_tokens[:total_max_tokens]
            break

    decode_time = time.perf_counter() - decode_start
    output_tokens = output_tokens[:total_max_tokens]
    return _build_result(
        target,
        output_tokens,
        prompt_len,
        prefill_time,
        decode_time,
        {
            "decode_mode": "vanilla_qwen36",
            "avg_acceptance_length": 1.0,
            "acceptance_lengths": [1] * steps,
            "speculative_tokens": 1,
        },
    )


def linear_spec_generate_ar(
    target: LoadedTargetModel,
    draft_model: Any,
    draft_tokenizer: Any,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    layer_ids: list[int],
    draft_max: int,
    draft_min: int,
    draft_p_min: float,
    reset_peak_memory: bool = True,
) -> ARSpecResult:
    if temperature >= 1e-5:
        raise ValueError("linear_spec_qwen17 currently supports temperature=0 only.")
    if reset_peak_memory:
        mx.reset_peak_memory()

    target_prompt_text, prompt_token_list = _target_prompt_tokens(target, prompt_text)
    draft_prompt_text = render_chat_prompt(draft_tokenizer, prompt_text)
    prompt_tokens = mx.array(prompt_token_list, dtype=mx.uint32)
    prompt_len = int(prompt_tokens.shape[0])
    total_max_tokens = prompt_len + int(max_new_tokens)
    target_cache = target.make_cache(speculative_linear_cache=target.adapter.family == "qwen3_5")

    sync_start = time.perf_counter()
    logits, _ = target.forward_with_hidden_states(
        prompt_tokens[None],
        target_cache,
        layer_ids,
    )
    first_token = int(sample_tokens(logits[:, -1, :], temperature).item())
    mx.eval(logits)
    prefill_time = time.perf_counter() - sync_start

    output_tokens = prompt_token_list + [first_token]
    start = prompt_len
    acceptance_lengths: list[int] = []
    bridge_rejects = 0
    draft_expansions = 0
    proposed_target_tokens = 0
    stop_ids = target.stop_token_ids()
    initial_generated_text = generated_text_for_target(target, output_tokens, prompt_len)
    draft_state = ARDraftState(
        draft_model,
        draft_tokenizer,
        draft_prompt_text + initial_generated_text,
    )

    decode_start = time.perf_counter()
    while start < total_max_tokens:
        generated_text = generated_text_for_target(target, output_tokens, prompt_len)
        target_prefix_text = target_prompt_text + generated_text
        draft_context_text = draft_prompt_text + generated_text
        draft_state.advance_to(draft_context_text)
        proposal = draft_state.propose_greedy(
            draft_max=draft_max,
            draft_min=draft_min,
            draft_p_min=draft_p_min,
        )
        draft_expansions += proposal.expansions

        bridge = bridge_draft_suffix_to_target(
            target.tokenizer,
            output_tokens,
            target_prefix_text,
            proposal.text,
        )
        if bridge.rejected:
            bridge_rejects += 1
        target_suffix = bridge.target_suffix[: max(0, int(draft_max))]
        proposed_target_tokens += len(target_suffix)
        block_tokens = [output_tokens[start], *target_suffix]
        draft_block_size = len(block_tokens)

        arm_target_rollback_with_prefix(target_cache, prefix_len=start)
        accepted_inputs, posterior_token, _ = verify_block_parallel_greedy_argmax(
            target=target,
            target_cache=target_cache,
            block_tokens=block_tokens,
            draft_block_size=draft_block_size,
            temperature=temperature,
            layer_ids=layer_ids,
        )
        acceptance_lengths.append(accepted_inputs)

        output_tokens = output_tokens[:start]
        output_tokens.extend(block_tokens[:accepted_inputs])
        output_tokens.append(posterior_token)
        start += accepted_inputs

        stop_idx = stop_position(output_tokens, prompt_len, stop_ids)
        if stop_idx is not None:
            output_tokens = output_tokens[: stop_idx + 1]
            break
        if len(output_tokens) > total_max_tokens:
            output_tokens = output_tokens[:total_max_tokens]
            break

    decode_time = time.perf_counter() - decode_start
    output_tokens = output_tokens[:total_max_tokens]
    return _build_result(
        target,
        output_tokens,
        prompt_len,
        prefill_time,
        decode_time,
        {
            "decode_mode": "linear_spec_qwen17",
            "avg_acceptance_length": sum(acceptance_lengths) / max(len(acceptance_lengths), 1),
            "acceptance_lengths": acceptance_lengths,
            "speculative_tokens": int(draft_max),
            "bridge_rejects": bridge_rejects,
            "draft_expansions": draft_expansions,
            "draft_cache_rebuilds": draft_state.cache_rebuilds,
            "draft_incremental_tokens": draft_state.incremental_tokens,
            "proposed_target_tokens": proposed_target_tokens,
        },
    )


def dtree_generate_ar(
    target: LoadedTargetModel,
    draft_model: Any,
    draft_tokenizer: Any,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    layer_ids: list[int],
    draft_max: int,
    draft_min: int,
    draft_p_min: float,
    tree_budget: int,
    branch_topk: int = 4,
    reset_peak_memory: bool = True,
) -> ARSpecResult:
    if reset_peak_memory:
        mx.reset_peak_memory()

    target_prompt_text, prompt_token_list = _target_prompt_tokens(target, prompt_text)
    draft_prompt_text = render_chat_prompt(draft_tokenizer, prompt_text)
    prompt_tokens = mx.array(prompt_token_list, dtype=mx.uint32)
    prompt_len = int(prompt_tokens.shape[0])
    total_max_tokens = prompt_len + int(max_new_tokens)
    target_cache = target.make_cache()

    sync_start = time.perf_counter()
    logits, _ = target.forward_with_hidden_states(
        prompt_tokens[None],
        target_cache,
        layer_ids,
    )
    first_token = int(sample_tokens(logits[:, -1, :], temperature).item())
    mx.eval(logits)
    prefill_time = time.perf_counter() - sync_start

    output_tokens = prompt_token_list + [first_token]
    start = prompt_len
    acceptance_lengths: list[int] = []
    verified_tree_nodes: list[int] = []
    built_tree_nodes: list[int] = []
    bridge_rejects = 0
    draft_expansions = 0
    duplicate_paths = 0
    stop_ids = target.stop_token_ids()

    decode_start = time.perf_counter()
    while start < total_max_tokens:
        generated_text = generated_text_for_target(target, output_tokens, prompt_len)
        target_prefix_text = target_prompt_text + generated_text
        draft_context_text = draft_prompt_text + generated_text
        tree = build_ar_target_tree(
            draft_model,
            draft_tokenizer,
            target.tokenizer,
            root_token_id=output_tokens[start],
            target_prefix_tokens=output_tokens,
            target_prefix_text=target_prefix_text,
            draft_context_text=draft_context_text,
            tree_budget=tree_budget,
            draft_max=draft_max,
            draft_min=draft_min,
            draft_p_min=draft_p_min,
            branch_topk=branch_topk,
        )
        bridge_rejects += tree.bridge_rejects
        draft_expansions += tree.draft_expansions
        duplicate_paths += tree.duplicate_paths
        built_tree_nodes.append(len(tree.tree_tokens))

        accepted_indices, posterior_token = lazy_follow_target_tree(
            target,
            target_cache,
            layer_ids,
            tree.tree_tokens,
            tree.child_maps,
            temperature,
        )
        accepted_inputs = len(accepted_indices)
        accepted_token_ids = [tree.tree_tokens[index] for index in accepted_indices]
        acceptance_lengths.append(accepted_inputs)
        verified_tree_nodes.append(accepted_inputs)

        output_tokens = output_tokens[:start]
        output_tokens.extend(accepted_token_ids)
        output_tokens.append(posterior_token)
        start += accepted_inputs

        stop_idx = stop_position(output_tokens, prompt_len, stop_ids)
        if stop_idx is not None:
            output_tokens = output_tokens[: stop_idx + 1]
            break
        if len(output_tokens) > total_max_tokens:
            output_tokens = output_tokens[:total_max_tokens]
            break

    decode_time = time.perf_counter() - decode_start
    output_tokens = output_tokens[:total_max_tokens]
    return _build_result(
        target,
        output_tokens,
        prompt_len,
        prefill_time,
        decode_time,
        {
            "decode_mode": "dtree_qwen17",
            "avg_acceptance_length": sum(acceptance_lengths) / max(len(acceptance_lengths), 1),
            "acceptance_lengths": acceptance_lengths,
            "speculative_tokens": int(draft_max),
            "tree_budget": int(tree_budget),
            "avg_verified_tree_nodes": sum(verified_tree_nodes) / max(len(verified_tree_nodes), 1),
            "verified_tree_nodes": verified_tree_nodes,
            "avg_built_tree_nodes": sum(built_tree_nodes) / max(len(built_tree_nodes), 1),
            "built_tree_nodes": built_tree_nodes,
            "bridge_rejects": bridge_rejects,
            "draft_expansions": draft_expansions,
            "duplicate_tree_paths": duplicate_paths,
        },
    )


def _build_result(
    target: LoadedTargetModel,
    output_tokens: list[int],
    prompt_len: int,
    prefill_time: float,
    decode_time: float,
    extra_metrics: dict[str, Any],
) -> ARSpecResult:
    generated_tokens = generated_token_count(output_tokens, prompt_len)
    total_time = prefill_time + decode_time
    metrics = {
        "num_input_tokens": prompt_len,
        "num_output_tokens": generated_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "prompt_tps": prompt_len / max(prefill_time, 1e-9),
        "generation_tps": generated_tokens / max(decode_time, 1e-9),
        "end_to_end_tps": generated_tokens / max(total_time, 1e-9),
        "peak_memory_gb": peak_memory_gb(),
        "target_cache_summary": "",
        **extra_metrics,
    }
    generated = output_tokens[prompt_len:]
    text = target.tokenizer.decode(generated, skip_special_tokens=False)
    return ARSpecResult(
        text=text,
        output_tokens=output_tokens,
        generated_tokens=generated,
        metrics=metrics,
    )
