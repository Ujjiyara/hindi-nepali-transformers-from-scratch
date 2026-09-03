"""Tests for perplexity and bits-per-byte.

These check the arithmetic against values that can be worked out by hand, so a
sign error or a misplaced log base cannot hide behind a plausible-looking
number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from common.data.loader import TokenArray
from common.eval.lm import score_split
from common.model.config import GPTConfig
from common.model.gpt import GPT

SMALL = GPTConfig(
    vocab_size=97, block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.0
)


@pytest.fixture
def tokens(tmp_path) -> TokenArray:
    rng = np.random.default_rng(0)
    path = tmp_path / "val.bin"
    rng.integers(0, 97, size=4_000, dtype=np.uint16).tofile(path)
    return TokenArray(path)


def _stats(tokens: TokenArray, bytes_per_token: float = 3.0) -> dict:
    return {"tokens": len(tokens),
            "utf8_bytes": int(len(tokens) * bytes_per_token)}


def test_untrained_model_scores_near_uniform(tokens: TokenArray) -> None:
    """An untrained model's perplexity should sit near the vocabulary size."""
    torch.manual_seed(0)
    metrics = score_split(
        GPT(SMALL), tokens, _stats(tokens), "validation", 4, SMALL.block_size,
        torch.device("cpu"),
    )
    assert metrics.cross_entropy == pytest.approx(math.log(97), abs=0.35)
    assert metrics.perplexity == pytest.approx(97, rel=0.4)


def test_perplexity_is_the_exponential_of_cross_entropy(tokens: TokenArray) -> None:
    torch.manual_seed(0)
    m = score_split(GPT(SMALL), tokens, _stats(tokens), "validation", 4,
                    SMALL.block_size, torch.device("cpu"))
    assert m.perplexity == pytest.approx(math.exp(m.cross_entropy))
    assert m.cross_entropy == pytest.approx(m.total_nll_nats / m.tokens_scored)


def test_bits_per_byte_converts_nats_to_bits_over_bytes(tokens: TokenArray) -> None:
    """BPB = total nats / (ln2 * bytes) -- checked against the definition."""
    torch.manual_seed(0)
    stats = _stats(tokens)
    m = score_split(GPT(SMALL), tokens, stats, "validation", 4,
                    SMALL.block_size, torch.device("cpu"))
    assert m.bits_per_byte == pytest.approx(
        m.total_nll_nats / (math.log(2) * m.bytes_scored)
    )
    # Fewer bytes per token means more bits attributed to each byte.
    denser = score_split(GPT(SMALL), tokens, _stats(tokens, 1.5), "validation",
                         4, SMALL.block_size, torch.device("cpu"))
    assert denser.bits_per_byte == pytest.approx(2 * m.bits_per_byte, rel=1e-3)


def test_bytes_are_scaled_to_the_portion_actually_scored(tokens: TokenArray) -> None:
    """A dropped trailing window must not be charged bytes it never covered."""
    stats = _stats(tokens)
    torch.manual_seed(0)
    m = score_split(GPT(SMALL), tokens, stats, "validation", 4,
                    SMALL.block_size, torch.device("cpu"))
    assert m.bytes_scored <= stats["utf8_bytes"]
    assert m.bytes_scored == pytest.approx(
        stats["utf8_bytes"] * m.tokens_scored / stats["tokens"], rel=1e-6
    )


def test_a_trained_model_beats_an_untrained_one(tmp_path) -> None:
    """Sanity: learning real structure must lower both perplexity and BPB.

    The token stream here is a repeating cycle, so there is something to learn
    that generalises from one window to the next. On uniformly random ids --
    the fixture the other tests use -- training on a slice cannot improve the
    rest of the split, because there is no structure to find.
    """
    path = tmp_path / "cycle.bin"
    np.tile(np.arange(20, dtype=np.uint16), 200).tofile(path)
    tokens = TokenArray(path)
    stats = _stats(tokens)

    torch.manual_seed(0)
    model = GPT(SMALL)
    before = score_split(model, tokens, stats, "validation", 4,
                         SMALL.block_size, torch.device("cpu"))

    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    window = torch.from_numpy(
        np.asarray(tokens.tokens[: 17 * 8]).astype(np.int64)
    ).view(8, 17)
    for _ in range(150):
        loss = model(window[:, :-1], window[:, 1:]).loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    after = score_split(model, tokens, stats, "validation", 4,
                        SMALL.block_size, torch.device("cpu"))
    assert after.cross_entropy < before.cross_entropy
    assert after.bits_per_byte < before.bits_per_byte
    # A 20-token cycle is fully predictable, so perplexity should approach 1.
    assert after.perplexity < 2.0
