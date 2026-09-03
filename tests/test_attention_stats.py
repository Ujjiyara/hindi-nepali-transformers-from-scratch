"""Tests for the attention statistics.

Each metric is checked against an attention pattern built by hand, where the
right answer is known exactly: a head that always looks one token back must
report distance 1 and entropy 0, and a head that averages over everything
available must report normalised entropy 1.
"""

from __future__ import annotations

import math

import pytest
import torch

from common.eval.attention import _per_head_statistics, collect_statistics
from common.model.config import GPTConfig
from common.model.gpt import GPT

T = 8


def _uniform_causal() -> torch.Tensor:
    """Row i spreads mass evenly over keys 0..i -- maximally diffuse."""
    mask = torch.tril(torch.ones(T, T))
    return (mask / mask.sum(dim=-1, keepdim=True)).view(1, 1, T, T)


def _attends_to(offset: int) -> torch.Tensor:
    """Row i puts all its mass on key i-offset, falling back to itself."""
    probs = torch.zeros(T, T)
    for i in range(T):
        probs[i, max(i - offset, 0)] = 1.0
    return probs.view(1, 1, T, T)


def test_uniform_attention_saturates_normalised_entropy() -> None:
    """Averaging over everything available is normalised entropy 1 by definition."""
    entropy, normalised, _ = _per_head_statistics(_uniform_causal())
    assert normalised.item() == pytest.approx(1.0, abs=1e-5)
    # Raw entropy is the mean of log(i+1) over query positions 1..T-1.
    expected = sum(math.log(i + 1) for i in range(1, T)) / (T - 1)
    assert entropy.item() == pytest.approx(expected, abs=1e-5)


def test_previous_token_head_has_distance_one_and_no_entropy() -> None:
    """The canonical local head: sharp, and looking exactly one step back."""
    entropy, normalised, distance = _per_head_statistics(_attends_to(1))
    assert distance.item() == pytest.approx(1.0)
    assert entropy.item() == pytest.approx(0.0, abs=1e-6)
    assert normalised.item() == pytest.approx(0.0, abs=1e-6)


def test_self_attending_head_has_zero_distance() -> None:
    """A head that only looks at the current token looks back no distance."""
    _, _, distance = _per_head_statistics(_attends_to(0))
    assert distance.item() == pytest.approx(0.0)


def test_longer_range_head_reports_a_larger_distance() -> None:
    """Distance must increase with how far back the head actually looks."""
    _, _, near = _per_head_statistics(_attends_to(1))
    _, _, far = _per_head_statistics(_attends_to(4))
    assert far.item() > near.item()
    # Early queries clamp to position 0, so the mean is below the nominal 4.
    assert 1.0 < far.item() <= 4.0


def test_query_zero_is_excluded() -> None:
    """Position 0 attends only to itself and would bias every average down."""
    _, normalised, _ = _per_head_statistics(_uniform_causal())
    # Including query 0 would contribute a normalised entropy of 0/0.
    assert math.isfinite(normalised.item())


def test_untrained_model_attends_diffusely() -> None:
    """An untrained model has near-uniform logits, so heads should average."""
    cfg = GPTConfig(vocab_size=97, block_size=32, n_layer=3, n_head=4,
                    d_model=64, dropout=0.0)
    torch.manual_seed(0)
    batches = [(torch.randint(0, 97, (4, 32)), None) for _ in range(2)]
    stats = collect_statistics(GPT(cfg), batches, torch.device("cpu"))

    assert stats.entropy.shape == (3, 4)
    assert stats.distance.shape == (3, 4)
    assert stats.sequences == 8
    assert stats.normalised_entropy.min() > 0.95
    assert len(stats.summary_rows()) == 12
    assert set(stats.as_dict()) >= {"entropy", "distance", "n_layer", "n_head"}


def test_no_batches_is_an_error() -> None:
    cfg = GPTConfig(vocab_size=97, block_size=16, n_layer=1, n_head=2, d_model=32)
    with pytest.raises(ValueError, match="no batches"):
        collect_statistics(GPT(cfg), [], torch.device("cpu"))
