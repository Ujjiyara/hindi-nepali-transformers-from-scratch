"""Tests for the learning-rate curve."""

from __future__ import annotations

import pytest

from common.train.schedule import ScheduleConfig, learning_rate

CFG = ScheduleConfig(max_lr=6e-4, min_lr=6e-5, warmup_steps=100, max_steps=1000)


def test_warmup_rises_to_the_peak() -> None:
    """Linear from near zero to max_lr, reaching it at the end of warmup."""
    assert learning_rate(0, CFG) == pytest.approx(6e-4 / 100)
    assert learning_rate(99, CFG) == pytest.approx(6e-4)
    rates = [learning_rate(s, CFG) for s in range(100)]
    assert all(b > a for a, b in zip(rates, rates[1:]))


def test_cosine_decays_to_the_floor() -> None:
    """Monotone down after warmup, arriving exactly at min_lr."""
    rates = [learning_rate(s, CFG) for s in range(100, 1000)]
    assert all(b < a for a, b in zip(rates, rates[1:]))
    assert learning_rate(1000, CFG) == pytest.approx(6e-5)


def test_rate_is_held_beyond_the_planned_end() -> None:
    """An over-running job keeps learning slowly rather than stopping dead."""
    assert learning_rate(5000, CFG) == pytest.approx(6e-5)


def test_rate_depends_only_on_the_step() -> None:
    """This is what lets the step alone stand in for scheduler state."""
    assert learning_rate(555, CFG) == learning_rate(555, CFG)


def test_impossible_curves_are_rejected() -> None:
    with pytest.raises(ValueError, match="shorter than max_steps"):
        ScheduleConfig(warmup_steps=1000, max_steps=1000)
    with pytest.raises(ValueError, match="min_lr must not exceed"):
        ScheduleConfig(max_lr=1e-5, min_lr=1e-4)
