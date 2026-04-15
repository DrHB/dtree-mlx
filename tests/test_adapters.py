from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from dtree_mlx.adapters import _quantized_lm_head_argmax


def _fake_quantized_module(weight: mx.array, *, group_size: int = 64, bits: int = 4):
    q_weight, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    return SimpleNamespace(
        weight=q_weight,
        scales=scales,
        biases=biases,
        bits=bits,
        group_size=group_size,
    )


def test_quantized_lm_head_argmax_matches_full_quantized_matmul(monkeypatch):
    monkeypatch.setenv("DTREE_QWEN35_ARGMAX_MAX_TOKENS", "8")
    monkeypatch.setenv("DTREE_QWEN35_ARGMAX_CHUNK_ROWS", "64")

    vocab = 128
    hidden = 64
    weight = mx.random.normal((vocab, hidden), dtype=mx.float32)
    module = _fake_quantized_module(weight)
    hidden_states = mx.random.normal((1, 4, hidden), dtype=mx.float32)

    reference = mx.argmax(
        mx.quantized_matmul(
            hidden_states.reshape(-1, hidden),
            module.weight,
            scales=module.scales,
            biases=module.biases,
            transpose=True,
            group_size=module.group_size,
            bits=module.bits,
        ).reshape(1, 4, vocab),
        axis=-1,
    ).astype(mx.uint32)
    candidate = _quantized_lm_head_argmax(module, hidden_states)
    mx.eval(reference, candidate)

    assert candidate is not None
    assert candidate.tolist() == reference.tolist()


def test_quantized_lm_head_argmax_preserves_first_index_on_cross_chunk_tie(monkeypatch):
    monkeypatch.setenv("DTREE_QWEN35_ARGMAX_MAX_TOKENS", "8")
    monkeypatch.setenv("DTREE_QWEN35_ARGMAX_CHUNK_ROWS", "64")

    vocab = 128
    hidden = 64
    weight_np = np.zeros((vocab, hidden), dtype=np.float32)
    weight_np[0] = 1.0
    weight_np[70] = 1.0
    weight = mx.array(weight_np)
    module = _fake_quantized_module(weight)
    hidden_states = mx.ones((1, 1, hidden), dtype=mx.float32)

    reference = mx.argmax(
        mx.quantized_matmul(
            hidden_states.reshape(-1, hidden),
            module.weight,
            scales=module.scales,
            biases=module.biases,
            transpose=True,
            group_size=module.group_size,
            bits=module.bits,
        ).reshape(1, 1, vocab),
        axis=-1,
    ).astype(mx.uint32)
    candidate = _quantized_lm_head_argmax(module, hidden_states)
    mx.eval(reference, candidate)

    assert candidate is not None
    assert int(reference[0, 0].item()) == 0
    assert candidate.tolist() == reference.tolist()
