"""Tests for decoding behaviour."""

from __future__ import annotations

import torch

from common.eval.generate import generate
from common.model.config import GPTConfig
from common.model.gpt import GPT

SMALL = GPTConfig(
    vocab_size=97, block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.0
)


def _model() -> GPT:
    torch.manual_seed(0)
    return GPT(SMALL).eval()


def test_prompt_is_preserved_and_length_is_exact() -> None:
    model, idx = _model(), torch.randint(0, 97, (3, 8))
    out = generate(model, idx, max_new_tokens=12, temperature=1.0)
    assert out.shape == (3, 20)
    assert torch.equal(out[:, :8], idx)


def test_greedy_decoding_is_deterministic() -> None:
    """Temperature 0 must take the argmax, so two runs agree exactly."""
    model, idx = _model(), torch.randint(0, 97, (2, 8))
    assert torch.equal(
        generate(model, idx, 10, temperature=0.0),
        generate(model, idx, 10, temperature=0.0),
    )


def test_greedy_matches_the_argmax_of_the_first_step() -> None:
    model, idx = _model(), torch.randint(0, 97, (2, 8))
    expected = model(idx).logits[:, -1, :].argmax(dim=-1)
    assert torch.equal(generate(model, idx, 1, temperature=0.0)[:, -1], expected)


def test_sampling_is_not_deterministic() -> None:
    """At temperature 1 two runs should differ, or nothing is being sampled."""
    model, idx = _model(), torch.randint(0, 97, (4, 8))
    torch.manual_seed(1)
    first = generate(model, idx, 16, temperature=1.0)
    torch.manual_seed(2)
    second = generate(model, idx, 16, temperature=1.0)
    assert not torch.equal(first, second)


def _confident_model() -> GPT:
    """A model overfitted onto a short cycle, so its logits are actually peaked.

    Temperature rescales logits, so it can only matter once the distribution is
    not already flat. An untrained model outputs near-uniform logits and
    temperature has almost no effect on it -- which makes an untrained model
    the wrong instrument for testing this property.
    """
    torch.manual_seed(5)
    model = GPT(SMALL)
    cycle = torch.arange(5).repeat(4, 4)[:, :SMALL.block_size]  # (4, 16), period 5
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(200):
        loss = model(cycle[:, :-1], cycle[:, 1:]).loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return model.eval()


def test_higher_temperature_gives_more_distinct_tokens() -> None:
    """Flattening a peaked distribution must widen the set of tokens produced."""
    model = _confident_model()
    idx = torch.arange(5).repeat(8, 2)[:, :8]

    torch.manual_seed(3)
    cold = generate(model, idx, 40, temperature=0.3)[:, 8:]
    torch.manual_seed(3)
    hot = generate(model, idx, 40, temperature=1.5)[:, 8:]

    assert len(cold.unique()) <= 6  # the learned cycle, give or take
    assert len(hot.unique()) > len(cold.unique())


def test_top_k_restricts_the_candidate_set() -> None:
    """With k=1 sampling degenerates to greedy."""
    model, idx = _model(), torch.randint(0, 97, (2, 8))
    torch.manual_seed(4)
    assert torch.equal(
        generate(model, idx, 8, temperature=1.0, top_k=1),
        generate(model, idx, 8, temperature=0.0),
    )


def test_generation_continues_past_the_context_window() -> None:
    """Beyond block_size the model must be fed a cropped context, not crash."""
    model = _model()
    out = generate(model, torch.randint(0, 97, (1, SMALL.block_size)), 10)
    assert out.size(1) == SMALL.block_size + 10
