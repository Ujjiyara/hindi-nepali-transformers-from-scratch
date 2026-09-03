"""Correctness tests for the hand-written causal self-attention.

Attention that is subtly wrong still trains and still produces a falling loss
curve, so "it converged" is not evidence. These tests pin down the two things
that cannot be seen from a loss curve: that the heads are split along the
feature axis rather than the time axis, and that no information flows backwards
in time.

Everything runs on CPU in float32 with dropout disabled. Dropout would make two
forward passes over the same input differ for reasons unrelated to masking, and
on CPU the reduction order is deterministic, so the causality tests can demand
bitwise equality rather than a tolerance.
"""

from __future__ import annotations

import math

import pytest
import torch

from common.model.attention import CausalSelfAttention
from common.model.config import GPTConfig

TEST_CONFIG = GPTConfig(
    vocab_size=997, d_model=48, n_head=4, block_size=32, dropout=0.0, n_layer=1
)


@pytest.fixture
def attn() -> CausalSelfAttention:
    """A small attention module in eval mode, with a fixed seed."""
    torch.manual_seed(0)
    return CausalSelfAttention(TEST_CONFIG).eval()


def _reference_forward(attn: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """Attention computed one head at a time, with no reshape tricks.

    This is the independent definition of "head *i* owns feature columns
    ``[i*d_head : (i+1)*d_head]``". It loops over heads and slices columns
    explicitly, so it agrees with the module under test only if the module's
    view/transpose pair carves heads out of the feature axis. Splitting along
    time instead -- the classic bug -- disagrees here immediately.
    """
    B, T, C = x.shape
    d_head = attn.d_head
    qkv = attn.qkv(x)
    q, k, v = qkv[..., :C], qkv[..., C : 2 * C], qkv[..., 2 * C :]

    keep = torch.tril(torch.ones(T, T, dtype=torch.bool))
    heads = []
    for i in range(attn.n_head):
        cols = slice(i * d_head, (i + 1) * d_head)
        scores = q[..., cols] @ k[..., cols].transpose(1, 2) / math.sqrt(d_head)
        scores = scores.masked_fill(~keep, float("-inf"))
        heads.append(torch.softmax(scores, dim=-1) @ v[..., cols])

    return attn.proj(torch.cat(heads, dim=-1))


def test_output_shape_matches_input(attn: CausalSelfAttention) -> None:
    """Attention is shape-preserving, which is what makes it stackable."""
    x = torch.randn(3, 20, TEST_CONFIG.d_model)
    assert attn(x).shape == x.shape


def test_probabilities_are_a_causal_distribution(attn: CausalSelfAttention) -> None:
    """Rows sum to one, and no query attends to a later key."""
    _, probs = attn(torch.randn(2, 20, TEST_CONFIG.d_model), return_attention=True)

    assert probs.shape == (2, TEST_CONFIG.n_head, 20, 20)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, TEST_CONFIG.n_head, 20))
    # triu(1) is everything strictly above the diagonal: the future.
    assert (probs.triu(1) == 0).all()


def test_matches_a_per_head_reference(attn: CausalSelfAttention) -> None:
    """The batched reshape agrees with explicit per-head column slicing."""
    x = torch.randn(2, 20, TEST_CONFIG.d_model)
    assert torch.allclose(attn(x), _reference_forward(attn, x), atol=1e-6)


def test_future_tokens_cannot_change_the_present(attn: CausalSelfAttention) -> None:
    """The check the brief asks for: perturb token t+1, watch position t.

    Masked positions contribute exactly 0.0 to the weighted sum, so the outputs
    at positions <= t are built from identical values summed in an identical
    order. Bitwise equality is the correct assertion, not a tolerance.
    """
    torch.manual_seed(1)
    T, t = 20, 9
    x = torch.randn(1, T, TEST_CONFIG.d_model)

    perturbed = x.clone()
    perturbed[:, t + 1 :, :] = torch.randn(1, T - t - 1, TEST_CONFIG.d_model)

    before, after = attn(x), attn(perturbed)

    assert torch.equal(before[:, : t + 1], after[:, : t + 1])
    # The change must actually have landed, or the test proves nothing.
    assert not torch.equal(before[:, t + 1 :], after[:, t + 1 :])


def test_no_gradient_path_from_the_future(attn: CausalSelfAttention) -> None:
    """The stronger statement: no path exists at all, not just that one input
    did not travel down it.

    Differentiating output ``t`` with respect to the whole input must leave
    every position after ``t`` with exactly zero gradient.
    """
    torch.manual_seed(2)
    T, t = 20, 9
    x = torch.randn(1, T, TEST_CONFIG.d_model, requires_grad=True)

    attn(x)[0, t].sum().backward()

    assert (x.grad[0, t + 1 :] == 0).all()
    assert (x.grad[0, : t + 1] != 0).any()


def test_sequence_longer_than_block_size_is_rejected(attn: CausalSelfAttention) -> None:
    """Better a clear error than a silently mis-sliced mask."""
    with pytest.raises(ValueError, match="exceeds block_size"):
        attn(torch.randn(1, TEST_CONFIG.block_size + 1, TEST_CONFIG.d_model))
