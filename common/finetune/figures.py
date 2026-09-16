"""Figures for Phase 3, drawn from stored results only.

Every figure here reads JSON (and one small ``.npz`` of heatmap matrices) that
the finetune, evaluation and attention stages wrote under
``<Language>/runs/finetune/``, so all of them can be redrawn without a
checkpoint or a model. Every figure has a title, axis labels and a legend or
colour bar, as the brief requires.

Run from the repository root::

    python -m common.finetune.figures \
        --run Hindi/runs/finetune --run Nepali/runs/finetune \
        --out report/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.eval.plots import devanagari_font, read_metrics  # noqa: E402

log = logging.getLogger("phase3-figures")

TASKS = ("numeric_pair", "numeric_triple", "chain_superlative", "chain_relation")
TASK_LABELS = {
    "numeric_pair": "Numeric pair",
    "numeric_triple": "Numeric triple",
    "chain_superlative": "Chain superlative",
    "chain_relation": "Chain relation (yes/no)",
}
METRICS = (("exact_match", "Exact match"), ("prefix_match", "Prefix match"),
           ("choice_accuracy", "Choice accuracy"))


class Run:
    """One language's Phase 3 outputs, loaded once."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        eval_dir = directory / "eval"
        self.pretrained = json.loads((eval_dir / "pretrained.json").read_text(encoding="utf-8"))
        self.finetuned = json.loads((eval_dir / "finetuned.json").read_text(encoding="utf-8"))
        attention_path = eval_dir / "attention.json"
        self.attention = (json.loads(attention_path.read_text(encoding="utf-8"))
                          if attention_path.exists() else None)
        heatmap_path = eval_dir / "attention.npz"
        self.heatmaps = dict(np.load(heatmap_path)) if heatmap_path.exists() else None
        # "Hindi (Model H), finetuned" -> "Hindi (Model H)"
        self.label = self.finetuned["label"].split(",")[0]
        self.short = self.label.split()[0].lower()


def plot_finetune_curves(runs: list[Run], out_dir: Path) -> Path:
    """Answer-token loss and teacher-forced validation exact match per step."""
    figure, (loss_axes, exact_axes) = plt.subplots(1, 2, figsize=(13, 4.8))
    for index, run in enumerate(runs):
        train, validation = read_metrics(run.directory / "metrics.jsonl")
        colour = f"C{index}"
        loss_axes.plot([r["step"] for r in train], [r["train_loss"] for r in train],
                       color=colour, alpha=0.35, linewidth=1,
                       label=f"{run.label} — training")
        loss_axes.plot([r["step"] for r in validation], [r["val_loss"] for r in validation],
                       color=colour, marker="o", markersize=4, linewidth=2,
                       label=f"{run.label} — validation")
        exact_axes.plot([r["step"] for r in validation],
                        [100 * r["val_exact_match"] for r in validation],
                        color=colour, marker="o", markersize=4, linewidth=2, label=run.label)

    loss_axes.set_title("Reasoning finetune: answer-token loss")
    loss_axes.set_xlabel("Optimiser step (0 = pretrained model)")
    loss_axes.set_ylabel("Cross-entropy on answer tokens (nats)")
    loss_axes.set_yscale("log")
    loss_axes.grid(True, alpha=0.3, linestyle="--", which="both")
    loss_axes.legend(fontsize=8)

    exact_axes.set_title("Validation exact match (teacher-forced)")
    exact_axes.set_xlabel("Optimiser step (0 = pretrained model)")
    exact_axes.set_ylabel("Examples with every answer token correct (%)")
    exact_axes.set_ylim(0, 100)
    exact_axes.grid(True, alpha=0.3, linestyle="--")
    exact_axes.legend(fontsize=8)

    figure.tight_layout()
    path = out_dir / "finetune_loss.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_accuracy_overview(runs: list[Run], out_dir: Path, split: str = "test") -> Path:
    """Pretrained vs finetuned on all three metrics, per language."""
    figure, axes_row = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 4.8),
                                    squeeze=False)
    width = 0.38
    for axes, run in zip(axes_row[0], runs):
        x = np.arange(len(METRICS))
        for offset, (stage, results, colour) in enumerate((
                ("Pretrained", run.pretrained, "C7"), ("Finetuned", run.finetuned, "C2"))):
            overall = results["splits"][split]["overall"]
            values = [100 * overall[key] for key, _ in METRICS]
            bars = axes.bar(x + (offset - 0.5) * width, values, width, color=colour,
                            label=stage)
            axes.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        chance = 100 * run.finetuned["splits"][split]["overall"]["chance"]
        axes.axhline(chance, color="C3", linestyle="--", linewidth=1.2,
                     label=f"Chance over choices ({chance:.1f}%)")
        axes.set_xticks(x)
        axes.set_xticklabels([name for _, name in METRICS])
        axes.set_ylim(0, 110)
        axes.set_title(f"{run.label} — reasoning test set")
        axes.set_xlabel("Metric")
        axes.set_ylabel("Accuracy (%)")
        axes.grid(True, axis="y", alpha=0.3, linestyle="--")
        axes.legend(fontsize=8, loc="upper left")

    figure.tight_layout()
    path = out_dir / "reasoning_accuracy.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_accuracy_by_task(runs: list[Run], out_dir: Path, split: str = "test") -> Path:
    """Per task family: pretrained choice accuracy, finetuned choice and exact."""
    figure, axes_row = plt.subplots(1, len(runs), figsize=(7 * len(runs), 5),
                                    squeeze=False)
    width = 0.27
    series = (
        ("Pretrained — choice accuracy", lambda r: r.pretrained, "choice_accuracy", "C7"),
        ("Finetuned — choice accuracy", lambda r: r.finetuned, "choice_accuracy", "C0"),
        ("Finetuned — exact match", lambda r: r.finetuned, "exact_match", "C2"),
    )
    for axes, run in zip(axes_row[0], runs):
        x = np.arange(len(TASKS))
        for offset, (name, source, key, colour) in enumerate(series):
            by_task = source(run)["splits"][split]["by_task"]
            values = [100 * by_task[t][key] for t in TASKS]
            bars = axes.bar(x + (offset - 1) * width, values, width, label=name, color=colour)
            axes.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
        chance = [100 * run.finetuned["splits"][split]["by_task"][t]["chance"] for t in TASKS]
        axes.scatter(x, chance, marker="_", s=900, color="C3", zorder=3,
                     label="Chance over choices")
        axes.set_xticks(x)
        axes.set_xticklabels([TASK_LABELS[t] for t in TASKS], rotation=12)
        axes.set_ylim(0, 110)
        axes.set_title(f"{run.label} — accuracy by task family")
        axes.set_xlabel("Task family")
        axes.set_ylabel("Accuracy (%)")
        axes.grid(True, axis="y", alpha=0.3, linestyle="--")
        axes.legend(fontsize=8, loc="upper left")

    figure.tight_layout()
    path = out_dir / "reasoning_accuracy_by_task.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_accuracy_by_hops(runs: list[Run], out_dir: Path) -> Path:
    """Chain tasks: accuracy against the number of comparisons to chain.

    ``test`` supplies chains of three and four entities and ``test_long``
    chains of five, which no training example had, so the rightmost point is
    a length-generalisation measurement.
    """
    figure, axes_row = plt.subplots(1, 2, figsize=(13, 4.8))
    for axes, task in zip(axes_row, ("chain_superlative", "chain_relation")):
        for index, run in enumerate(runs):
            for stage, results, style, key in (
                    ("finetuned exact match", run.finetuned, "-", "exact_match"),
                    ("pretrained choice accuracy", run.pretrained, ":", "choice_accuracy")):
                # The two splits are kept as separate series: test_long
                # relation questions also span 1-3 hops, but over longer
                # chains, so merging them would overwrite the test points.
                for split, marker, suffix in (("test", "o", "3–4 entity chains"),
                                              ("test_long", "D", "5 entity chains")):
                    points = {}
                    for bucket, summary in results["splits"][split]["by_task_hops"].items():
                        name, hops = bucket.split("/")
                        if name == task:
                            points[int(hops)] = 100 * summary[key]
                    hops = sorted(points)
                    axes.plot(hops, [points[h] for h in hops], linestyle=style,
                              marker=marker, color=f"C{index}",
                              alpha=1.0 if split == "test" else 0.6,
                              label=f"{run.label} — {stage} ({suffix})")
        axes.set_title(f"{TASK_LABELS[task]}: accuracy by hops")
        axes.set_xticks(range(1, 5))
        axes.set_xlabel("Comparisons separating the entities (hops); 5-entity chains are an unseen length")
        axes.set_ylabel("Accuracy (%)")
        axes.set_ylim(0, 105)
        axes.grid(True, alpha=0.3, linestyle="--")
        axes.legend(fontsize=7)

    figure.tight_layout()
    path = out_dir / "reasoning_accuracy_by_hops.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_heatmap_comparison(run: Run, layer: int, out_dir: Path) -> Path:
    """Every head of one layer, pretrained row above finetuned row, same prompt."""
    tokens = run.attention["heatmap_example"]["tokens"]
    font = devanagari_font()
    labels = tokens if font else [str(i) for i in range(len(tokens))]
    n_head = run.heatmaps["pretrained"].shape[1]

    figure, grid = plt.subplots(2, n_head, figsize=(3.4 * n_head, 8.6), squeeze=False,
                                gridspec_kw={"hspace": 0.12, "wspace": 0.06})
    image = None
    for row, stage in enumerate(("pretrained", "finetuned")):
        for head in range(n_head):
            axes = grid[row][head]
            image = axes.imshow(run.heatmaps[stage][layer, head], cmap="viridis",
                                vmin=0.0, vmax=1.0, aspect="auto")
            axes.set_title(f"{stage.capitalize()} — head {head}", fontsize=9)
            axes.set_xticks(range(len(labels)))
            axes.set_yticks(range(len(labels)))
            # Token labels on the outer edges only; every panel shares them.
            if row == 1:
                axes.set_xticklabels(labels, rotation=90, fontsize=5, fontproperties=font)
                axes.set_xlabel("Key position (attended to)", fontsize=8)
            else:
                axes.set_xticklabels([])
            if head == 0:
                axes.set_yticklabels(labels, fontsize=5, fontproperties=font)
                axes.set_ylabel("Query position (attending from)", fontsize=8)
            else:
                axes.set_yticklabels([])

    figure.suptitle(f"{run.label} — attention on a reasoning prompt, layer {layer}: "
                    "pretrained (top) vs finetuned (bottom)")
    figure.colorbar(image, ax=grid, label="Attention weight", fraction=0.02)
    path = out_dir / f"attention_finetune_{run.short}_layer{layer}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_head_change(runs: list[Run], out_dir: Path) -> Path:
    """Per-head change (finetuned minus pretrained) on reasoning prompts."""
    figure, grid = plt.subplots(len(runs), 2, figsize=(11, 3.6 * len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        stages = run.attention["stages"]
        for column, (key, name, unit) in enumerate((
                ("distance", "mean attention distance", "tokens"),
                ("normalised_entropy", "normalised entropy", "")
        )):
            before = np.asarray(stages["pretrained"]["reasoning_prompts"][key])
            after = np.asarray(stages["finetuned"]["reasoning_prompts"][key])
            delta = after - before
            bound = float(np.abs(delta).max()) or 1.0
            axes = grid[row][column]
            image = axes.imshow(delta, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
            axes.set_title(f"{run.label}: change in {name}")
            axes.set_xlabel("Head")
            axes.set_ylabel("Layer (0 = first block)")
            axes.set_xticks(range(delta.shape[1]))
            axes.set_yticks(range(delta.shape[0]))
            label = f"Finetuned − pretrained ({unit})" if unit else "Finetuned − pretrained"
            figure.colorbar(image, ax=axes, label=label)

    figure.suptitle("Per-head attention change after reasoning finetuning "
                    "(reasoning test prompts)")
    figure.tight_layout()
    path = out_dir / "attention_finetune_head_change.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_layer_profiles(runs: list[Run], out_dir: Path) -> Path:
    """Layer-mean distance and entropy, before and after, on prompts and corpus text."""
    panels = (
        ("reasoning_prompts", "distance", "Mean attention distance (tokens)",
         "Reasoning prompts: attention distance"),
        ("reasoning_prompts", "normalised_entropy", "Normalised entropy (1 = uniform)",
         "Reasoning prompts: attention entropy"),
        ("corpus_text", "distance", "Mean attention distance (tokens)",
         "Corpus text: attention distance"),
        ("corpus_text", "normalised_entropy", "Normalised entropy (1 = uniform)",
         "Corpus text: attention entropy"),
    )
    figure, grid = plt.subplots(2, 2, figsize=(12, 8.5))
    for axes, (source, key, ylabel, title) in zip(grid.flat, panels):
        for index, run in enumerate(runs):
            for stage, style, marker in (("pretrained", ":", "s"), ("finetuned", "-", "o")):
                stats = run.attention["stages"][stage].get(source)
                if stats is None:
                    continue
                values = np.asarray(stats[key]).mean(axis=1)
                axes.plot(range(len(values)), values, linestyle=style, marker=marker,
                          markersize=4, color=f"C{index}", label=f"{run.label} — {stage}")
        axes.set_title(title)
        axes.set_xlabel("Layer (0 = first block)")
        axes.set_ylabel(ylabel)
        axes.grid(True, alpha=0.3, linestyle="--")
        axes.legend(fontsize=7)
    figure.suptitle("Layer-mean attention statistics before and after finetuning")
    figure.tight_layout()
    path = out_dir / "attention_finetune_layer_profiles.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_answer_position(runs: list[Run], out_dir: Path) -> Path:
    """Where the answer position looks: entity mass and gold-entity share by layer."""
    panels = (
        ("entity", "Attention on entity-name tokens (share of row)",
         "Answer position → all entity mentions"),
        ("gold_share", "Correct entity's share of entity attention",
         "Answer position → the correct entity"),
        ("question", "Attention on question tokens (share of row)",
         "Answer position → question sentence"),
        ("first", "Attention on the first token (share of row)",
         "Answer position → first token (sink)"),
    )
    figure, grid = plt.subplots(2, 2, figsize=(12, 8.5))
    for axes, (key, ylabel, title) in zip(grid.flat, panels):
        for index, run in enumerate(runs):
            for stage, style, marker in (("pretrained", ":", "s"), ("finetuned", "-", "o")):
                values = run.attention["stages"][stage]["answer_position"][key]
                layer_values = np.asarray(values).mean(axis=1)
                axes.plot(range(len(layer_values)), layer_values, linestyle=style,
                          marker=marker, markersize=4, color=f"C{index}",
                          label=f"{run.label} — {stage}")
        axes.set_title(title)
        axes.set_xlabel("Layer (0 = first block)")
        axes.set_ylabel(ylabel)
        axes.grid(True, alpha=0.3, linestyle="--")
        axes.legend(fontsize=7)
    figure.suptitle("Attention from the final prompt token (where the answer is predicted), "
                    "mean over heads")
    figure.tight_layout()
    path = out_dir / "attention_finetune_answer_position.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Draw the Phase 3 figures.")
    parser.add_argument("--run", action="append", required=True, type=Path,
                        help="a <Language>/runs/finetune directory; repeat per language")
    parser.add_argument("--out", type=Path, default=Path("report/figures"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    runs = [Run(directory) for directory in args.run]
    written = [
        plot_finetune_curves(runs, args.out),
        plot_accuracy_overview(runs, args.out),
        plot_accuracy_by_task(runs, args.out),
        plot_accuracy_by_hops(runs, args.out),
    ]
    with_attention = [run for run in runs if run.attention is not None]
    if with_attention:
        for run in with_attention:
            n_layer = run.heatmaps["pretrained"].shape[0]
            for layer in (0, n_layer - 1):
                written.append(plot_heatmap_comparison(run, layer, args.out))
        written += [plot_head_change(with_attention, args.out),
                    plot_layer_profiles(with_attention, args.out),
                    plot_answer_position(with_attention, args.out)]
    for path in written:
        log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
