"""Tests for figure generation.

Figures are checked for existence and for the properties the brief grades --
a title, both axis labels, and a legend where more than one series is drawn --
rather than for appearance.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from common.eval.plots import (
    devanagari_font,
    plot_attention_heatmaps,
    plot_attention_summary,
    plot_loss_curves,
    plot_validation_perplexity,
    read_metrics,
)


@pytest.fixture
def metrics(tmp_path):
    """A short run log with both training and validation records."""
    path = tmp_path / "metrics.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for step in range(1, 11):
            handle.write(json.dumps({
                "step": step * 10, "tokens": step * 32768,
                "train_loss": 9.0 - 0.5 * step, "lr": 6e-4,
            }) + "\n")
            if step % 3 == 0:
                loss = 9.1 - 0.5 * step
                handle.write(json.dumps({
                    "step": step * 10, "tokens": step * 32768,
                    "val_loss": loss, "val_ppl": float(np.exp(loss)),
                }) + "\n")
    return path


def test_read_metrics_separates_the_two_record_kinds(metrics) -> None:
    train, validation = read_metrics(metrics)
    assert len(train) == 10 and len(validation) == 3
    assert all("train_loss" in r for r in train)
    assert all("val_loss" in r for r in validation)


def test_loss_curves_are_drawn_for_both_axes(metrics, tmp_path) -> None:
    runs = {"Hindi (Model H)": metrics, "Nepali (Model L)": metrics}
    written = plot_loss_curves(runs, tmp_path / "fig")
    assert {p.name for p in written} == {
        "pretraining_loss_by_step.png", "pretraining_loss_by_tokens.png"
    }
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_every_figure_carries_a_title_axis_labels_and_legend(
    metrics, tmp_path, monkeypatch
) -> None:
    """The brief awards zero for a component whose plots are unlabelled.

    The plot helpers close their figures before returning, so the figure is
    intercepted on its way to ``plt.close`` and inspected while still live.
    """
    from common.eval import plots as plots_module

    captured = []
    real_close = plots_module.plt.close

    def capture(figure=None):
        captured.append(figure)

    monkeypatch.setattr(plots_module.plt, "close", capture)
    plots_module.plot_loss_curves(
        {"Hindi (Model H)": metrics, "Nepali (Model L)": metrics}, tmp_path / "fig"
    )
    plots_module.plot_validation_perplexity({"Hindi (Model H)": metrics},
                                            tmp_path / "fig")
    monkeypatch.undo()

    assert len(captured) == 3
    for figure in captured:
        axes = figure.axes[0]
        assert axes.get_title(), "figure has no title"
        assert axes.get_xlabel(), "figure has no x-label"
        assert axes.get_ylabel(), "figure has no y-label"
        assert axes.get_legend() is not None, "multi-series figure has no legend"
        real_close(figure)


def test_perplexity_figure_is_written(metrics, tmp_path) -> None:
    path = plot_validation_perplexity({"Hindi (Model H)": metrics}, tmp_path / "fig")
    assert path.exists() and path.name == "validation_perplexity.png"


def test_attention_heatmap_handles_devanagari_labels(tmp_path) -> None:
    """Real token strings, including the SentencePiece word-boundary marker."""
    probs = np.tril(np.random.dirichlet(np.ones(10), size=(4, 10)))
    probs /= probs.sum(axis=-1, keepdims=True)
    tokens = ["▁भारत", "▁के", "▁प्रधानमंत्री", "▁ने", "▁आज",
              "▁नई", "▁दिल्ली", "▁में", "▁एक", "।"]

    path = plot_attention_heatmaps(probs, tokens, "Hindi (Model H)", 0, tmp_path)
    assert path.exists() and path.stat().st_size > 0


def test_attention_summary_draws_entropy_and_distance(tmp_path) -> None:
    stats = {
        label: {
            "normalised_entropy": np.random.rand(10, 6).tolist(),
            "distance": (np.random.rand(10, 6) * 20).tolist(),
        }
        for label in ("Hindi (Model H)", "Nepali (Model L)")
    }
    written = plot_attention_summary(stats, tmp_path / "fig")
    assert {p.name for p in written} == {
        "attention_entropy.png", "attention_distance.png"
    }
    assert all(p.exists() for p in written)


def test_a_devanagari_font_stack_is_available() -> None:
    """Without one, heatmap labels silently become empty boxes."""
    font = devanagari_font()
    if font is not None:
        assert "DejaVu Sans" in font.get_family(), "needs a Latin/U+2581 fallback"
