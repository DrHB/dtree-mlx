from __future__ import annotations

import mlx.core as mx

from dtree_mlx.dtree_runtime import build_dtree_tree, follow_verified_tree
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
