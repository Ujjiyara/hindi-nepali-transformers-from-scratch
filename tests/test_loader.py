"""Tests for batch construction.

The property that matters for the brief is reproducibility: a run that resumes
must see the same data it would have seen had it never stopped. That is what
makes a resumed loss curve continuous rather than merely plausible.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from common.data.loader import TokenArray, evaluation_windows, training_batch


@pytest.fixture
def tokens(tmp_path) -> TokenArray:
    """A small synthetic token array: ids 0..9999 in order."""
    path = tmp_path / "toy.bin"
    np.arange(10_000, dtype=np.uint16).tofile(path)
    return TokenArray(path)


def test_targets_are_inputs_shifted_by_one(tokens: TokenArray) -> None:
    """y[t] must be the token following x[t] -- the whole training signal."""
    x, y = training_batch(tokens, batch_size=4, block_size=16, seed=7, step=0)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])
    # The array counts upwards, so the shift is directly visible.
    assert torch.equal(y, x + 1)


def test_batches_are_a_pure_function_of_the_step(tokens: TokenArray) -> None:
    """Same (seed, step) gives the same batch; a later step gives a new one."""
    first = training_batch(tokens, 4, 16, seed=7, step=42)
    again = training_batch(tokens, 4, 16, seed=7, step=42)
    later = training_batch(tokens, 4, 16, seed=7, step=43)
    other_seed = training_batch(tokens, 4, 16, seed=8, step=42)

    assert torch.equal(first[0], again[0])
    assert not torch.equal(first[0], later[0])
    assert not torch.equal(first[0], other_seed[0])


def test_resumed_run_sees_the_same_stream(tokens: TokenArray) -> None:
    """Steps 0..9 uninterrupted equal steps 0..4 then a resume at 5."""
    straight = [training_batch(tokens, 2, 8, 11, s)[0] for s in range(10)]
    resumed = [training_batch(tokens, 2, 8, 11, s)[0] for s in range(5)]
    resumed += [training_batch(tokens, 2, 8, 11, s)[0] for s in range(5, 10)]
    assert all(torch.equal(a, b) for a, b in zip(straight, resumed))


def test_windows_never_run_past_the_end(tokens: TokenArray) -> None:
    """Every sampled window must be fully inside the array."""
    x, y = training_batch(tokens, batch_size=64, block_size=32, seed=1, step=0)
    assert int(y.max()) <= 9999


def test_evaluation_covers_the_split_once(tokens: TokenArray) -> None:
    """Deterministic, non-overlapping, and near-complete coverage."""
    block = 25
    scored = torch.cat(
        [y.reshape(-1) for _, y in evaluation_windows(tokens, 4, block)]
    )
    # Every token except the first is predicted exactly once, up to the
    # trailing partial window that is dropped.
    assert len(scored) == len(torch.unique(scored))
    assert len(scored) >= len(tokens) - block - 1

    twice = torch.cat(
        [y.reshape(-1) for _, y in evaluation_windows(tokens, 4, block)]
    )
    assert torch.equal(scored, twice)


def test_max_batches_limits_evaluation(tokens: TokenArray) -> None:
    """The periodic validation check reads only part of the split."""
    assert len(list(evaluation_windows(tokens, 4, 16, max_batches=3))) == 3


def test_missing_file_is_reported_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="run common.data.run pack"):
        TokenArray(tmp_path / "absent.bin")
