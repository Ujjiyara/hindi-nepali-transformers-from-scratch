"""Figures for Phase 2: loss curves, attention heatmaps, and head summaries.

Kept separate from every module that computes a number, so figures can be
redrawn from stored results without re-running a model. Every figure carries a
title, both axis labels and a legend where more than one series is drawn, which
the brief requires on pain of zero marks for the component.

**Devanagari labels.** Matplotlib's default family (DejaVu Sans) has no
Devanagari coverage and silently renders every glyph as an empty box, so
attention heatmaps labelled with real tokens would be unreadable. A Devanagari
font is located once at import and used for tick labels on those figures; if
none is installed, the heatmaps fall back to numeric positions rather than
drawing boxes and pretending to be labelled.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a headless Kaggle or CI run
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

log = logging.getLogger(__name__)

_MARKERS = ["o", "s", "^", "D", "v"]
_DEVANAGARI_CANDIDATES = (
    "Noto Sans Devanagari", "Noto Serif Devanagari", "Nirmala UI",
    "Lohit Devanagari", "Mangal", "FreeSans",
)


def devanagari_font() -> fm.FontProperties | None:
    """A font stack that can draw Devanagari, or None if none is installed.

    A *list* of families, not one: Noto Sans Devanagari covers Devanagari but
    has no Latin glyphs and, more importantly here, no U+2581 -- the block
    character SentencePiece uses to mark a word boundary, which therefore
    appears on almost every token label. Matplotlib falls back through the list
    per glyph, so Devanagari is drawn by the first family and everything else
    by the second.
    """
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _DEVANAGARI_CANDIDATES:
        if name in available:
            return fm.FontProperties(family=[name, "DejaVu Sans"])
    log.warning("no Devanagari font found; heatmaps will use numeric labels")
    return None


def read_metrics(path: Path) -> tuple[list[dict], list[dict]]:
    """Split a run's ``metrics.jsonl`` into its training and validation records."""
    train, validation = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        (train if "train_loss" in record else validation).append(record)
    return train, validation


def plot_loss_curves(runs: dict[str, Path], out_dir: Path) -> list[Path]:
    """Draw training and validation loss for every model on shared axes.

    Args:
        runs: Label (e.g. ``"Hindi (Model H)"``) to that run's metrics.jsonl.
        out_dir: Where the PNGs go.

    Returns:
        Paths written.

    Two x-axes are drawn because they answer different questions. Against
    *steps* the two models are compared at equal optimiser work, which is the
    fair comparison of the architectures. Against *tokens seen* they are
    compared at equal data, which is the comparison the resource gap is about
    -- Model L simply runs out of corpus earlier.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for xkey, xlabel, filename in (
        ("step", "Optimiser step", "pretraining_loss_by_step.png"),
        ("tokens", "Training tokens seen", "pretraining_loss_by_tokens.png"),
    ):
        figure, axes = plt.subplots(figsize=(8, 5))
        for index, (label, path) in enumerate(sorted(runs.items())):
            train, validation = read_metrics(path)
            colour = f"C{index}"
            if train:
                axes.plot([r[xkey] for r in train], [r["train_loss"] for r in train],
                          color=colour, alpha=0.35, linewidth=1,
                          label=f"{label} — training")
            if validation:
                axes.plot([r[xkey] for r in validation],
                          [r["val_loss"] for r in validation],
                          color=colour, marker=_MARKERS[index % len(_MARKERS)],
                          markersize=4, linewidth=2,
                          label=f"{label} — validation")

        axes.set_title("Pretraining loss")
        axes.set_xlabel(xlabel)
        axes.set_ylabel("Cross-entropy (nats per token)")
        axes.grid(True, alpha=0.3, linestyle="--")
        axes.legend()
        figure.tight_layout()
        path = out_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    return written


def plot_validation_perplexity(runs: dict[str, Path], out_dir: Path) -> Path:
    """Validation perplexity against tokens seen, both models together."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 5))

    for index, (label, path) in enumerate(sorted(runs.items())):
        _, validation = read_metrics(path)
        if not validation:
            continue
        axes.plot([r["tokens"] for r in validation],
                  [r["val_ppl"] for r in validation],
                  marker=_MARKERS[index % len(_MARKERS)], markersize=4,
                  linewidth=2, label=label)

    axes.set_title("Validation perplexity during pretraining")
    axes.set_xlabel("Training tokens seen")
    axes.set_ylabel("Perplexity (per token)")
    axes.set_yscale("log")
    axes.grid(True, alpha=0.3, linestyle="--", which="both")
    axes.legend()
    figure.tight_layout()
    path = out_dir / "validation_perplexity.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_attention_heatmaps(
    probs: np.ndarray,
    tokens: list[str],
    label: str,
    layer: int,
    out_dir: Path,
    max_heads: int = 4,
    max_positions: int = 24,
) -> Path:
    """Draw one layer's heads as query-by-key heatmaps for a real sentence.

    Args:
        probs: ``(n_head, T, T)`` attention probabilities for one sequence.
        tokens: Decoded token strings, used as tick labels.
        label: Model label for the title.
        layer: Which layer these came from.
        out_dir: Destination directory.
        max_heads: Heads to draw side by side.
        max_positions: Truncate to the first N positions; a 512x512 grid is
            unreadable at figure size and the structure is visible early.

    Returns:
        The path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    font = devanagari_font()

    n_head = min(max_heads, probs.shape[0])
    size = min(max_positions, probs.shape[-1], len(tokens))
    labels = tokens[:size] if font else [str(i) for i in range(size)]

    figure, axes_row = plt.subplots(1, n_head, figsize=(4.2 * n_head, 4.6),
                                    squeeze=False)
    image = None
    for head in range(n_head):
        axes = axes_row[0][head]
        image = axes.imshow(probs[head, :size, :size], cmap="viridis",
                            vmin=0.0, aspect="auto")
        axes.set_title(f"Head {head}")
        axes.set_xlabel("Key position (attended to)")
        if head == 0:
            axes.set_ylabel("Query position (attending from)")
        axes.set_xticks(range(size))
        axes.set_yticks(range(size))
        axes.set_xticklabels(labels, rotation=90, fontsize=6,
                             fontproperties=font)
        axes.set_yticklabels(labels, fontsize=6, fontproperties=font)

    figure.suptitle(f"{label} — attention weights, layer {layer}")
    figure.colorbar(image, ax=axes_row[0], label="Attention weight",
                    fraction=0.025)
    path = out_dir / f"attention_heatmap_{label.split()[0].lower()}_layer{layer}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_attention_summary(stats: dict[str, dict], out_dir: Path) -> list[Path]:
    """Per-head entropy and distance, by layer, for every model.

    Args:
        stats: Model label to that model's ``AttentionStatistics.as_dict()``.
        out_dir: Destination directory.

    Returns:
        Paths written.

    Each head is a point at its layer index, jittered slightly so overlapping
    heads stay visible. This is the figure that shows head specialisation: a
    layer whose heads are spread widely in distance has differentiated into
    local and long-range heads, while a tight cluster near zero has not.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for key, ylabel, filename, title in (
        ("normalised_entropy", "Normalised attention entropy (1 = uniform)",
         "attention_entropy.png", "Attention entropy per head"),
        ("distance", "Mean attention distance (token positions)",
         "attention_distance.png", "Attention distance per head"),
    ):
        figure, axes = plt.subplots(figsize=(8, 5))
        for index, (label, payload) in enumerate(sorted(stats.items())):
            values = np.asarray(payload[key])
            n_layer, n_head = values.shape
            offset = (index - 0.5) * 0.18
            for head in range(n_head):
                axes.scatter(
                    np.arange(n_layer) + offset, values[:, head],
                    color=f"C{index}", alpha=0.65, s=26,
                    marker=_MARKERS[index % len(_MARKERS)],
                    label=label if head == 0 else None,
                )
            axes.plot(np.arange(n_layer) + offset, values.mean(axis=1),
                      color=f"C{index}", linewidth=2, alpha=0.9,
                      label=f"{label} — layer mean")

        axes.set_title(title)
        axes.set_xlabel("Layer (0 = first block)")
        axes.set_ylabel(ylabel)
        axes.grid(True, alpha=0.3, linestyle="--")
        axes.legend(fontsize=8)
        figure.tight_layout()
        path = out_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    return written
