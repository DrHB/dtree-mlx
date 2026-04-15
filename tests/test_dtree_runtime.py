from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from dtree_mlx.dtree_runtime import (
    build_dtree_tree,
    follow_verified_tree,
    lazy_follow_tree_exact,
)
from dtree_mlx.qwen35_tree import build_node_lineages


def test_build_dtree_tree_returns_prefix_closed_visibility():
    draft_logits = mx.array(
        [
            [5.0, 4.0, 1.0],
            [6.0, 2.0, 1.0],
            [7.0, 3.0, 1.0],
        ],
        dtype=mx.float32,
    )

    node_token_ids, node_depths, parents, child_maps, visibility, _ = build_dtree_tree(
        draft_logits=draft_logits,
        budget=4,
    )

    assert node_token_ids
    assert len(node_token_ids) == len(node_depths)
    assert parents[0] == -1
    assert visibility.shape[0] == 1 + len(node_token_ids)
    assert bool(visibility[0, 0].item()) is True
    for node_index in range(1, len(parents)):
        parent_index = parents[node_index]
        assert parent_index < node_index
        assert child_maps[parent_index][node_token_ids[node_index - 1]] == node_index
        assert bool(visibility[node_index, parent_index].item()) is True
        assert bool(visibility[node_index, node_index].item()) is True


def test_build_dtree_tree_sibling_bonus_can_prefer_early_breadth():
    draft_logits = mx.array(
        [
            [5.0, 4.7, 1.0],
            [6.0, 1.0, 0.0],
        ],
        dtype=mx.float32,
    )

    node_token_ids, node_depths, *_ = build_dtree_tree(
        draft_logits=draft_logits,
        budget=2,
        sibling_bonus=0.0,
    )
    assert node_depths == [1, 2]

    node_token_ids_bonus, node_depths_bonus, *_ = build_dtree_tree(
        draft_logits=draft_logits,
        budget=2,
        sibling_bonus=0.5,
    )
    assert node_depths_bonus == [1, 1]
    assert node_token_ids_bonus[1] != node_token_ids[1]


def test_follow_verified_tree_walks_matching_children():
    child_maps = [
        {11: 1, 12: 2},
        {21: 3},
        {},
        {},
    ]
    accepted, next_token = follow_verified_tree(child_maps, [11, 21, 99, 0])
    assert accepted == [0, 1, 3]
    assert next_token == 0


def test_build_node_lineages_returns_root_to_parent_paths():
    parents = [-1, 0, 0, 1, 3, 2]
    assert build_node_lineages(parents) == [
        [],
        [0],
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 2],
    ]


def test_lazy_follow_tree_exact_walks_only_the_taken_branch():
    class FakeTextModel:
        def forward_dflash(self, inputs, cache, layer_ids):
            del cache, layer_ids
            token = int(inputs[0, 0].item())
            norm_hidden = mx.array([[[token]]], dtype=mx.float32)
            verifier_hidden = mx.array([[[token + 1000]]], dtype=mx.float32)
            return norm_hidden, verifier_hidden

    class FakeTarget:
        def __init__(self):
            self.adapter = SimpleNamespace(family="qwen3_5")
            self.model = SimpleNamespace(language_model=SimpleNamespace(model=FakeTextModel()))

        def lm_head_argmax(self, hidden_states):
            token = int(hidden_states[0, 0, 0].item())
            mapping = {10: 100, 11: 300, 21: 999}
            return mx.array([[mapping[token]]], dtype=mx.uint32)

    child_maps = [
        {100: 1, 200: 2},
        {300: 3},
        {},
        {},
    ]

    accepted, next_token, verifier_hidden = lazy_follow_tree_exact(
        target=FakeTarget(),
        target_cache=[],
        layer_ids=[],
        tree_tokens=[10, 11, 12, 21],
        child_maps=child_maps,
        temperature=0.0,
        verify_mode="parallel-greedy-argmax",
        profile_times=None,
    )

    assert accepted == [0, 1, 3]
    assert next_token == 999
    assert verifier_hidden.tolist() == [[[1010.0], [1011.0], [1021.0]]]
