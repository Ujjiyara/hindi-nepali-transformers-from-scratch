"""End-to-end tests for the assembled language model.

The attention tests pin down the masking; these pin down the things that only
appear once embeddings, the block stack and the output head are wired together:
that the parameter budget is what the config claims, that the model starts out
predicting a uniform distribution, that causality survives the full depth, and
that the thing can actually learn.
"""

from __future__ import annotations

import math

import pytest
import torch

from common.model.config import GPTConfig
from common.model.gpt import GPT

SMALL = GPTConfig(
    vocab_size=257, block_size=32, n_layer=3, n_head=4, d_model=64, dropout=0.0
)


@pytest.fixture
def model() -> GPT:
    """A small model in eval mode, deterministically initialised."""
    torch.manual_seed(0)
    return GPT(SMALL).eval()


def test_parameter_count_matches_the_config() -> None:
    """The number reported in the config is the number actually built.

    This is what makes ``GPTConfig.parameter_count`` usable as a report table:
    a geometry can be costed before training, and the claim is checked here
    against the assembled model rather than trusted.
    """
    cfg = GPTConfig(vocab_size=16000)
    assert GPT(cfg).num_parameters() == cfg.parameter_count()["total"]


def test_budget_geometry_is_about_25m() -> None:
    """The chosen Model H / Model L geometry sits inside the brief's budget."""
    total = GPTConfig(vocab_size=16000).parameter_count()["total"]
    assert 23e6 < total < 26e6


def test_tied_head_shares_storage_with_the_embedding() -> None:
    """Tying must alias one matrix, not copy it into two."""
    model = GPT(GPTConfig(vocab_size=257, tie_embeddings=True))
    assert model.lm_head.weight is model.token_embedding.weight

    untied = GPT(GPTConfig(vocab_size=257, tie_embeddings=False))
    assert untied.lm_head.weight is not untied.token_embedding.weight
    assert untied.num_parameters() - model.num_parameters() == 257 * 384


def test_initial_loss_is_uniform_over_the_vocabulary(model: GPT) -> None:
    """An untrained model should be maximally uncertain: loss near ln(V).

    A much larger value means the output head is badly scaled; a much smaller
    one means information is leaking from the targets.
    """
    idx = torch.randint(0, SMALL.vocab_size, (4, 16))
    loss = model(idx, targets=torch.randint(0, SMALL.vocab_size, (4, 16))).loss
    assert loss.item() == pytest.approx(math.log(SMALL.vocab_size), abs=0.3)


def test_future_tokens_cannot_change_earlier_logits(model: GPT) -> None:
    """Causality holds through the whole stack, not just one attention layer."""
    torch.manual_seed(1)
    T, t = 24, 11
    idx = torch.randint(0, SMALL.vocab_size, (1, T))

    changed = idx.clone()
    changed[:, t + 1 :] = (changed[:, t + 1 :] + 1) % SMALL.vocab_size

    before = model(idx).logits
    after = model(changed).logits

    assert torch.equal(before[:, : t + 1], after[:, : t + 1])
    assert not torch.equal(before[:, t + 1 :], after[:, t + 1 :])


def test_no_gradient_reaches_position_t_from_later_tokens(model: GPT) -> None:
    """The stronger form, measured at the embedding output.

    Token ids are discrete so they carry no gradient; the equivalent statement
    is that the embedding rows for positions after ``t`` receive none.
    """
    torch.manual_seed(2)
    T, t = 24, 11
    idx = torch.randint(0, SMALL.vocab_size, (1, T))

    captured = {}

    def hook(_module, _args, output):
        output.retain_grad()
        captured["emb"] = output
        return output

    handle = model.token_embedding.register_forward_hook(hook)
    try:
        model(idx).logits[0, t].sum().backward()
    finally:
        handle.remove()

    grad = captured["emb"].grad[0]
    assert (grad[t + 1 :] == 0).all()
    assert (grad[: t + 1] != 0).any()


def test_sequence_longer_than_block_size_is_rejected(model: GPT) -> None:
    """Learned absolute positions make block_size a hard ceiling."""
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, SMALL.vocab_size, (1, SMALL.block_size + 1)))


def test_disabling_positional_embeddings_removes_the_table() -> None:
    """The bonus ablation is a config flag, not a separate code path."""
    ablated = GPT(SMALL.with_(use_positional=False))
    assert ablated.position_embedding is None
    assert (
        GPT(SMALL).num_parameters() - ablated.num_parameters()
        == SMALL.block_size * SMALL.d_model
    )


def test_model_can_overfit_a_single_batch() -> None:
    """If a fixed batch cannot be memorised, the backward path is severed.

    Cheapest possible check that gradients actually flow everywhere they
    should; it catches a detached tensor or a frozen parameter long before a
    real training run would.
    """
    torch.manual_seed(3)
    model = GPT(SMALL).train()
    idx = torch.randint(0, SMALL.vocab_size, (2, 16))
    targets = torch.randint(0, SMALL.vocab_size, (2, 16))

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = model(idx, targets).loss.item()
    for _ in range(200):
        loss = model(idx, targets).loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    assert first > 5.0
    assert loss.item() < 0.1
