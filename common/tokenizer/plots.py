"""Plotting for the tokenizer sweep.

Kept separate from the metric computation so that figures can be regenerated
without re-running any measurement. Every figure carries a title, both axis
labels and a legend, as the project brief requires.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a headless run
import matplotlib.pyplot as plt  # noqa: E402

_MARKERS = ["o", "s", "^", "D", "v"]


def plot_sweep(sweep_by_lang: dict[str, list[dict]], out_dir: Path) -> list[Path]:
    """Draw the vocabulary-size sweep figures.

    ``sweep_by_lang`` maps a language label to its list of metric dicts, so both
    models appear on the same axes and the resource gap is directly visible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    panels = [
        ("fertility", "Fertility (tokens per word)", "tokenizer_fertility.png",
         "Tokenizer fertility vs. vocabulary size"),
        ("chars_per_token", "Characters per token", "tokenizer_chars_per_token.png",
         "Characters per token vs. vocabulary size"),
        ("vocab_utilisation", "Fraction of vocabulary seen in held-out text",
         "tokenizer_vocab_utilisation.png",
         "Vocabulary utilisation vs. vocabulary size"),
    ]

    for key, ylabel, filename, title in panels:
        figure, axes = plt.subplots(figsize=(7.5, 5))
        for index, (lang, rows) in enumerate(sorted(sweep_by_lang.items())):
            rows = sorted(rows, key=lambda r: r["vocab_size"])
            axes.plot(
                [r["vocab_size"] for r in rows],
                [r[key] for r in rows],
                marker=_MARKERS[index % len(_MARKERS)],
                linewidth=2,
                label=lang,
            )
        axes.set_title(title)
        axes.set_xlabel("Vocabulary size (subword units)")
        axes.set_ylabel(ylabel)
        axes.set_xscale("log", base=2)
        axes.grid(True, alpha=0.3, linestyle="--")
        axes.legend(title="Model")
        figure.tight_layout()

        path = out_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    return written


def plot_token_frequency(
    metrics_by_lang: dict[str, dict], out_dir: Path, top_k: int = 25
) -> Path:
    """Bar chart of the most frequent tokens per language."""
    out_dir.mkdir(parents=True, exist_ok=True)
    languages = sorted(metrics_by_lang)
    figure, axes_list = plt.subplots(
        1, len(languages), figsize=(7 * len(languages), 6), squeeze=False
    )

    for column, lang in enumerate(languages):
        axes = axes_list[0][column]
        top = metrics_by_lang[lang]["top_tokens"][:top_k]
        labels = [piece.replace("▁", "_") for piece, _ in top][::-1]
        values = [count for _, count in top][::-1]
        axes.barh(range(len(values)), values, label=lang)
        axes.set_yticks(range(len(labels)))
        axes.set_yticklabels(labels, fontsize=8)
        axes.set_title(f"Most frequent tokens - {lang}")
        axes.set_xlabel("Occurrences in held-out text")
        axes.set_ylabel("Token (_ marks a word boundary)")
        axes.legend(loc="lower right")

    figure.tight_layout()
    path = out_dir / "tokenizer_top_tokens.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def load_sweep(path: Path) -> list[dict]:
    """Read one sweep.json and return its per-vocabulary-size rows."""
    return json.loads(path.read_text(encoding="utf-8"))["sweep"]
