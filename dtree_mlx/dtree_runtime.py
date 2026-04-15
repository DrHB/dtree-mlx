from __future__ import annotations

import heapq
import os
import time
from typing import Any

import mlx.core as mx
import numpy as np

from .adapters import LoadedTargetModel
from .draft import DFlashDraftModel
from .runtime import (
    add_profile_elapsed,
    generated_token_count,
    peak_memory_gb,
    profile_start,
    sample_tokens,
    stop_position,
    trim_draft_cache,
)


DTREE_STAGE_ORDER = (
    "draft_time_s",
    "tree_build_time_s",
    "tree_compile_time_s",
    "verify_time_s",
    "bookkeeping_time_s",
)
DTREE_TREE_BUILD_STAGE_ORDER = (
    "tree_build_copy_time_s",
    "tree_build_heap_time_s",
    "tree_build_visibility_time_s",
)
DTREE_VERIFY_DETAIL_STAGE_ORDER = (
    "verify_tree_forward_time_s",
    "verify_tree_logits_time_s",
    "verify_tree_argmax_time_s",
    "verify_tree_sample_time_s",
)
DTREE_BOOKKEEPING_DETAIL_STAGE_ORDER = (
    "bookkeeping_follow_tree_time_s",
    "bookkeeping_cache_compact_time_s",
    "bookkeeping_hidden_select_time_s",
    "bookkeeping_output_commit_time_s",
)


def empty_dtree_profile() -> dict[str, float]:
    return {
        **{name: 0.0 for name in DTREE_STAGE_ORDER},
        **{name: 0.0 for name in DTREE_TREE_BUILD_STAGE_ORDER},
        **{name: 0.0 for name in DTREE_VERIFY_DETAIL_STAGE_ORDER},
        **{name: 0.0 for name in DTREE_BOOKKEEPING_DETAIL_STAGE_ORDER},
    }


def cache_compaction_eval_tensors(
    cache: list[Any],
    past_length: int,
    keep_count: int,
) -> list[mx.array]:
    if keep_count <= 0:
        return []

    tensors: list[mx.array] = []
    for layer_cache in cache:
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        offset = getattr(layer_cache, "offset", None)
        if keys is None or values is None or offset is None:
            continue
        end = min(past_length + keep_count, int(offset))
        if end <= past_length:
            continue
        tensors.append(keys[..., past_length:end, :])
        tensors.append(values[..., past_length:end, :])
    return tensors


def build_dtree_tree(
    draft_logits: mx.array,
    budget: int,
    candidate_topk: int | None = None,
) -> tuple[list[int], list[int], list[int], list[dict[int, int]], mx.array, dict[str, float]]:
    subtimes = {name: 0.0 for name in DTREE_TREE_BUILD_STAGE_ORDER}

    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = mx.array([[True]], dtype=mx.bool_)
        return [], [], [-1], [dict()], visibility, subtimes

    requested_topk = budget if candidate_topk is None else max(budget, int(candidate_topk))
    topk = min(requested_topk, int(draft_logits.shape[-1]))
    depth_limit = int(draft_logits.shape[0])

    copy_start = time.perf_counter()
    logits = draft_logits.astype(mx.float32)
    partition = mx.argpartition(logits, kth=int(logits.shape[-1]) - topk, axis=-1)[
        :, -topk:
    ]
    top_logits = mx.take_along_axis(logits, partition, axis=-1)
    order = mx.argsort(-top_logits, axis=-1)
    top_token_ids = mx.take_along_axis(partition, order, axis=-1)
    top_logits = mx.take_along_axis(top_logits, order, axis=-1)
    top_log_probs = top_logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(top_token_ids, top_log_probs)
    top_token_ids_np = np.array(top_token_ids, dtype=np.int64)
    top_log_probs_np = np.array(top_log_probs, dtype=np.float32)
    subtimes["tree_build_copy_time_s"] = time.perf_counter() - copy_start

    heap_start = time.perf_counter()
    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids: list[int] = []
    node_depths: list[int] = []
    parents = [-1]
    child_maps: list[dict[int, int]] = [dict()]

    while heap and len(node_token_ids) < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = len(node_token_ids) + 1
        node_token_ids.append(token_id)
        node_depths.append(depth)
        parents.append(parent_index)
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs_np[depth - 1, rank]) + float(
                top_log_probs_np[depth - 1, rank + 1]
            )
            heapq.heappush(
                heap,
                (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw),
            )

    subtimes["tree_build_heap_time_s"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + len(node_token_ids)
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = parents[index]
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    subtimes["tree_build_visibility_time_s"] = time.perf_counter() - visibility_start

    visibility = mx.array(visibility_np, dtype=mx.bool_)
    return node_token_ids, node_depths, parents, child_maps, visibility, subtimes


def compile_dtree_tree(
    root_token_id: int,
    start: int,
    node_token_ids: list[int],
    node_depths: list[int],
    visibility: mx.array,
    past_length: int,
    verify_input_ids_buffer: mx.array,
    verify_position_ids_buffer: mx.array,
    attention_mask_buffer: mx.array,
) -> tuple[mx.array, list[int], mx.array, mx.array]:
    tree_tokens = [root_token_id, *node_token_ids]
    positions = [start, *[start + depth for depth in node_depths]]

    current_length = len(tree_tokens)
    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_position_ids = verify_position_ids_buffer[:, :current_length]

    verify_input_ids[0, 0] = root_token_id
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_input_ids[0, 1:current_length] = mx.array(node_token_ids, dtype=mx.uint32)
        verify_position_ids[0, 1:current_length] = mx.array(positions[1:], dtype=mx.uint32)

    attention_mask = attention_mask_buffer[:current_length, : past_length + current_length]
    attention_mask[:, past_length : past_length + current_length] = visibility
    return verify_input_ids, tree_tokens, verify_position_ids, attention_mask


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior_tokens: list[int],
) -> tuple[list[int], int]:
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def lazy_follow_tree_exact(
    target: LoadedTargetModel,
    target_cache: list[Any],
    layer_ids: list[int],
    tree_tokens: list[int],
    child_maps: list[dict[int, int]],
    temperature: float,
    verify_mode: str,
    profile_times: dict[str, float] | None,
) -> tuple[list[int], int, mx.array]:
    if target.adapter.family != "qwen3_5":
        raise NotImplementedError("lazy_follow_tree_exact is only implemented for qwen3_5.")

    target_model = target.model
    if hasattr(target_model, "language_model") and hasattr(target_model.language_model, "model"):
        text_model = target_model.language_model.model
    elif hasattr(target_model, "model"):
        text_model = target_model.model
    else:
        raise AttributeError(f"Unsupported qwen3_5 target model: {type(target_model)!r}")

    accepted_indices: list[int] = []
    hidden_chunks: list[mx.array] = []
    current_index = 0

    while True:
        verify_forward_start = profile_start(profile_times)
        input_ids = mx.array([[tree_tokens[current_index]]], dtype=mx.uint32)
        norm_hidden_states, verifier_hidden = text_model.forward_dflash(
            input_ids,
            target_cache,
            layer_ids,
        )
        if profile_times is not None:
            mx.eval(norm_hidden_states, verifier_hidden)
        add_profile_elapsed(
            profile_times,
            "verify_tree_forward_time_s",
            verify_forward_start,
        )

        accepted_indices.append(current_index)
        hidden_chunks.append(verifier_hidden)

        if verify_mode == "parallel-greedy-argmax":
            verify_argmax_start = profile_start(profile_times)
            posterior = target.lm_head_argmax(norm_hidden_states)
            if profile_times is not None:
                mx.eval(posterior)
            add_profile_elapsed(
                profile_times,
                "verify_tree_argmax_time_s",
                verify_argmax_start,
            )
        else:
            verify_logits_start = profile_start(profile_times)
            verifier_logits = target.lm_head_logits(norm_hidden_states)
            if profile_times is not None:
                mx.eval(verifier_logits)
            add_profile_elapsed(
                profile_times,
                "verify_tree_logits_time_s",
                verify_logits_start,
            )

            verify_sample_start = profile_start(profile_times)
            posterior = sample_tokens(verifier_logits, temperature)
            if profile_times is not None:
                mx.eval(posterior)
            add_profile_elapsed(
                profile_times,
                "verify_tree_sample_time_s",
                verify_sample_start,
            )

        posterior_token = int(posterior[0, 0].item())
        next_index = child_maps[current_index].get(posterior_token)
        if next_index is None:
            return accepted_indices, posterior_token, mx.concatenate(hidden_chunks, axis=1)
        current_index = next_index


def dtree_generate(
    target: LoadedTargetModel,
    draft: DFlashDraftModel,
    prompt_tokens: mx.array,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: set[int],
    layer_ids: list[int],
    speculative_tokens: int | None,
    tree_budget: int | None = None,
    verify_mode: str = "parallel-replay",
    profile: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    if not target.supports_tree_verification():
        raise NotImplementedError(
            f"DTree is not implemented for target adapter family={target.adapter.family!r}."
        )
    if verify_mode not in {"parallel-replay", "parallel-greedy-argmax"}:
        raise ValueError(
            "DTree only supports verify_mode='parallel-replay' or "
            "'parallel-greedy-argmax'."
        )
    if verify_mode == "parallel-greedy-argmax" and temperature >= 1e-5:
        raise ValueError("DTree parallel-greedy-argmax only supports temperature=0.")

    target_cache = target.make_cache()
    draft_cache = draft.make_cache()
    profile_times = empty_dtree_profile() if profile else None
    total_max_tokens = int(prompt_tokens.shape[0]) + max_new_tokens
    prompt_len = int(prompt_tokens.shape[0])

    if speculative_tokens is None:
        block_size = draft.block_size
    else:
        block_size = max(1, min(speculative_tokens, draft.block_size))

    if block_size <= 1:
        raise ValueError("DTree requires speculative_tokens >= 2.")

    effective_tree_budget = max(tree_budget if tree_budget is not None else block_size - 1, 0)
    max_tree_nodes = 1 + effective_tree_budget
    verify_input_ids_buffer = mx.zeros((1, max_tree_nodes), dtype=mx.uint32)
    verify_position_ids_buffer = mx.zeros((1, max_tree_nodes), dtype=mx.uint32)
    attention_mask_buffer = mx.ones(
        (max_tree_nodes, total_max_tokens + max_tree_nodes),
        dtype=mx.bool_,
    )

    sync_start = time.perf_counter()
    logits, target_hidden = target.forward_with_hidden_states(
        prompt_tokens[None],
        target_cache,
        layer_ids,
    )
    first_token = int(sample_tokens(logits[:, -1, :], temperature).item())
    mx.eval(logits, target_hidden)
    prefill_time = time.perf_counter() - sync_start

    output_tokens = prompt_tokens.tolist() + [first_token]
    start = prompt_len
    acceptance_lengths: list[int] = []
    verified_tree_nodes: list[int] = []
    qwen35_tree_mode = os.environ.get("DTREE_QWEN35_TREE_MODE")
    tree_candidate_topk = os.environ.get("DTREE_TREE_CANDIDATE_TOPK")
    candidate_topk = (
        int(tree_candidate_topk)
        if tree_candidate_topk is not None and tree_candidate_topk.strip()
        else None
    )
    # Qwen3.5 hybrid caches are expensive to materialize for every speculative
    # branch. The default tree path verifies only the branch the target follows.
    use_lazy_qwen35_tree = (
        target.adapter.family == "qwen3_5"
        and qwen35_tree_mode != "full_tree"
    )

    decode_start = time.perf_counter()
    while start < total_max_tokens:
        draft_start = profile_start(profile_times)
        block_tokens = [output_tokens[start]] + [draft.mask_token_id] * (block_size - 1)
        block_input = mx.array(block_tokens, dtype=mx.uint32)[None]
        noise_embedding = target.embed_tokens(block_input)

        draft_hidden = draft(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = target.lm_head_logits(draft_hidden[:, 1:, :])
        mx.eval(draft_logits)
        trim_draft_cache(draft_cache, block_size)
        add_profile_elapsed(profile_times, "draft_time_s", draft_start)

        tree_build_start = profile_start(profile_times)
        if candidate_topk is not None:
            (
                node_token_ids,
                node_depths,
                parents,
                child_maps,
                visibility,
                tree_build_subtimes,
            ) = build_dtree_tree(
                draft_logits[0],
                effective_tree_budget,
                candidate_topk=candidate_topk,
            )
        else:
            (
                node_token_ids,
                node_depths,
                parents,
                child_maps,
                visibility,
                tree_build_subtimes,
            ) = build_dtree_tree(
                draft_logits[0],
                effective_tree_budget,
            )
        add_profile_elapsed(profile_times, "tree_build_time_s", tree_build_start)
        if profile_times is not None:
            for key, value in tree_build_subtimes.items():
                profile_times[key] += value

        tree_compile_start = profile_start(profile_times)
        tree_tokens = [block_tokens[0], *node_token_ids]
        if not use_lazy_qwen35_tree:
            verify_input_ids, tree_tokens, verify_position_ids, attention_mask = compile_dtree_tree(
                root_token_id=block_tokens[0],
                start=start,
                node_token_ids=node_token_ids,
                node_depths=node_depths,
                visibility=visibility,
                past_length=start,
                verify_input_ids_buffer=verify_input_ids_buffer,
                verify_position_ids_buffer=verify_position_ids_buffer,
                attention_mask_buffer=attention_mask_buffer,
            )
        add_profile_elapsed(profile_times, "tree_compile_time_s", tree_compile_start)

        verify_start = profile_start(profile_times)
        if use_lazy_qwen35_tree:
            accepted_indices, posterior_token, verifier_hidden = lazy_follow_tree_exact(
                target=target,
                target_cache=target_cache,
                layer_ids=layer_ids,
                tree_tokens=tree_tokens,
                child_maps=child_maps,
                temperature=temperature,
                verify_mode=verify_mode,
                profile_times=profile_times,
            )
        else:
            if profile_times is not None:
                verify_forward_start = profile_start(profile_times)
                norm_hidden_states, verifier_hidden = target.forward_tree_with_hidden_states(
                    verify_input_ids,
                    target_cache,
                    layer_ids,
                    parents=parents,
                    position_ids=verify_position_ids,
                    attention_mask=attention_mask,
                )
                mx.eval(norm_hidden_states, verifier_hidden)
                add_profile_elapsed(
                    profile_times,
                    "verify_tree_forward_time_s",
                    verify_forward_start,
                )

                if verify_mode == "parallel-greedy-argmax":
                    verify_argmax_start = profile_start(profile_times)
                    posterior = target.lm_head_argmax(norm_hidden_states)
                    mx.eval(posterior)
                    add_profile_elapsed(
                        profile_times,
                        "verify_tree_argmax_time_s",
                        verify_argmax_start,
                    )
                else:
                    verify_logits_start = profile_start(profile_times)
                    verifier_logits = target.lm_head_logits(norm_hidden_states)
                    mx.eval(verifier_logits)
                    add_profile_elapsed(
                        profile_times,
                        "verify_tree_logits_time_s",
                        verify_logits_start,
                    )

                    verify_sample_start = profile_start(profile_times)
                    posterior = sample_tokens(verifier_logits, temperature)
                    mx.eval(posterior)
                    add_profile_elapsed(
                        profile_times,
                        "verify_tree_sample_time_s",
                        verify_sample_start,
                    )
            else:
                norm_hidden_states, verifier_hidden = target.forward_tree_with_hidden_states(
                    verify_input_ids,
                    target_cache,
                    layer_ids,
                    parents=parents,
                    position_ids=verify_position_ids,
                    attention_mask=attention_mask,
                )
                if verify_mode == "parallel-greedy-argmax":
                    posterior = target.lm_head_argmax(norm_hidden_states)
                else:
                    posterior = sample_tokens(target.lm_head_logits(norm_hidden_states), temperature)
                mx.eval(posterior, verifier_hidden)
            posterior_tokens = posterior[0].tolist()
            accepted_indices, posterior_token = follow_verified_tree(child_maps, posterior_tokens)
        add_profile_elapsed(profile_times, "verify_time_s", verify_start)

        bookkeeping_start = profile_start(profile_times)
        follow_tree_start = profile_start(profile_times)
        accepted_inputs = len(accepted_indices)
        accepted_token_ids = [tree_tokens[index] for index in accepted_indices]
        add_profile_elapsed(
            profile_times,
            "bookkeeping_follow_tree_time_s",
            follow_tree_start,
        )

        compact_start = profile_start(profile_times)
        if not use_lazy_qwen35_tree:
            target.compact_kv_caches(
                target_cache,
                past_length=start,
                keep_current_indices=accepted_indices,
            )
            if profile_times is not None:
                compact_tensors = cache_compaction_eval_tensors(
                    target_cache,
                    past_length=start,
                    keep_count=accepted_inputs,
                )
                if compact_tensors:
                    mx.eval(*compact_tensors)
        add_profile_elapsed(
            profile_times,
            "bookkeeping_cache_compact_time_s",
            compact_start,
        )

        hidden_select_start = profile_start(profile_times)
        if use_lazy_qwen35_tree:
            target_hidden = verifier_hidden
        else:
            accepted_index_array = mx.array(accepted_indices, dtype=mx.uint32)
            target_hidden = mx.take(verifier_hidden, accepted_index_array, axis=1)
        if profile_times is not None:
            mx.eval(target_hidden)
        add_profile_elapsed(
            profile_times,
            "bookkeeping_hidden_select_time_s",
            hidden_select_start,
        )

        output_commit_start = profile_start(profile_times)
        output_tokens = output_tokens[:start]
        output_tokens.extend(accepted_token_ids)
        output_tokens.append(posterior_token)
        start += accepted_inputs

        acceptance_lengths.append(accepted_inputs)
        verified_tree_nodes.append(accepted_inputs if use_lazy_qwen35_tree else len(tree_tokens))
        add_profile_elapsed(
            profile_times,
            "bookkeeping_output_commit_time_s",
            output_commit_start,
        )
        add_profile_elapsed(profile_times, "bookkeeping_time_s", bookkeeping_start)

        stop_idx = stop_position(output_tokens, prompt_len, stop_token_ids)
        if stop_idx is not None:
            output_tokens = output_tokens[: stop_idx + 1]
            break

        if len(output_tokens) > total_max_tokens:
            output_tokens = output_tokens[:total_max_tokens]
            break

    decode_time = time.perf_counter() - decode_start
    output_tokens = output_tokens[:total_max_tokens]
    generated_tokens = generated_token_count(output_tokens, prompt_len)
    total_time = prefill_time + decode_time

    metrics = {
        "decode_mode": "dtree",
        "num_input_tokens": prompt_len,
        "num_output_tokens": generated_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "prompt_tps": prompt_len / max(prefill_time, 1e-9),
        "generation_tps": generated_tokens / max(decode_time, 1e-9),
        "end_to_end_tps": generated_tokens / max(total_time, 1e-9),
        "avg_acceptance_length": sum(acceptance_lengths) / max(len(acceptance_lengths), 1),
        "acceptance_lengths": acceptance_lengths,
        "avg_verified_tree_nodes": sum(verified_tree_nodes) / max(len(verified_tree_nodes), 1),
        "verified_tree_nodes": verified_tree_nodes,
        "peak_memory_gb": peak_memory_gb(),
        "target_cache_summary": target.cache_summary(target_cache),
        "speculative_tokens": block_size,
        "tree_budget": effective_tree_budget,
        "tree_candidate_topk": candidate_topk if candidate_topk is not None else effective_tree_budget,
        "verify_mode": verify_mode,
    }
    if profile_times is not None:
        profiled_time = sum(profile_times[name] for name in DTREE_STAGE_ORDER)
        metrics["profile"] = {
            **profile_times,
            "unattributed_decode_time_s": decode_time - profiled_time,
            "steps": len(acceptance_lengths),
        }
    return output_tokens, metrics
