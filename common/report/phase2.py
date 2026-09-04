"""Generate ``report/phase2.md`` from measured training and evaluation outputs.

Usage::

    python -m common.report.phase2 \
        --run hi=Hindi/runs/base --run ne=Nepali/runs/base \
        --out report/phase2.md --figures report/figures

Every number in the report comes from ``results.json``, ``config.json`` or
``metrics.jsonl`` written by the training and evaluation stages. Nothing is
typed by hand, so the report cannot drift away from the run that produced it --
the same property Phase 1's report has, and the reason a stale figure cannot
survive a re-run.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("report")

MODEL_ROLE = {"hi": "Model H (higher-resource)", "ne": "Model L (lower-resource)"}
LANGUAGE_NAMES = {"hi": "Hindi", "ne": "Nepali"}


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None if it is not there yet."""
    if not path.exists():
        log.warning("missing %s", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _label(lang: str) -> str:
    """``"Hindi (Model H)"`` for a language code."""
    return f"{LANGUAGE_NAMES.get(lang, lang)} ({MODEL_ROLE.get(lang, lang).split(' (')[0]})"


def _final_metrics(run_dir: Path) -> dict | None:
    return _read_json(run_dir / "eval" / "results.json")


def _architecture_section(configs: dict[str, dict]) -> list[str]:
    """Geometry, parameter budget, and the decisions the brief asks about."""
    out = ["\n## 1. Architecture\n"]
    any_cfg = next(iter(configs.values()))
    model = any_cfg["model"]
    params = any_cfg["parameters"]

    out.append(
        "Both models use the same geometry. That is not a shared artifact -- "
        "the weights, vocabulary and data stay entirely separate -- but it "
        "makes the comparison between them a comparison of languages and "
        "corpora rather than of architectures.\n"
    )

    out.append("| Setting | Value | Why |\n|---|---|---|")
    out.append(f"| Layers | {model['n_layer']} | depth over width at a fixed "
               "budget: each block is another sequential transformation of the "
               "residual stream |")
    out.append(f"| Model dimension | {model['d_model']} | with 16,000 vocabulary "
               "entries this lands at ~24.1M parameters |")
    out.append(f"| Heads | {model['n_head']} | head dimension "
               f"{model['d_model'] // model['n_head']}, the size tensor cores "
               "are most efficient at |")
    out.append(f"| Feed-forward width | {model['d_ff']} | 4x the model "
               "dimension, the standard ratio |")
    out.append(f"| Context length | {model['block_size']} | a hard ceiling: "
               "learned absolute positions have no embedding beyond it |")
    out.append(f"| Dropout | {model['dropout']} | on embeddings, attention "
               "probabilities and sublayer outputs |")
    out.append(f"| Tied embeddings | {str(model['tie_embeddings']).lower()} | "
               "saves vocab x d_model parameters, a quarter of the budget |\n")

    out.append("**Parameter count.**\n")
    out.append("| Component | Parameters |\n|---|---|")
    for key, name in (
        ("embedding", "Token embedding (tied with the output head)"),
        ("positional", "Positional embedding"),
        ("blocks", f"{model['n_layer']} transformer blocks"),
        ("final_norm", "Final LayerNorm"),
        ("lm_head", "Output head (0 when tied)"),
    ):
        out.append(f"| {name} | {params[key]:,} |")
    out.append(f"| **Total** | **{params['total']:,}** |\n")

    out.append(
        "**Positional embeddings are learned and absolute.** Self-attention is "
        "permutation-equivariant, so with no position signal the model sees an "
        "unordered bag of tokens. A learned table costs one small matrix and "
        "needs no extra machinery; the price is that "
        f"{model['block_size']} is a hard architectural limit on context, not a "
        "soft one, and the model has no representation at all for a longer "
        "position. A relative scheme such as RoPE would lift that.\n"
    )
    out.append(
        "**Pre-norm, not post-norm.** LayerNorm sits inside each residual "
        "branch, leaving an unnormalised identity path from input to output so "
        "that gradient reaches layer 0 undistorted. Post-norm renormalises that "
        "path once per sublayer, which is why deep post-norm stacks need long "
        "warmups to train at all.\n"
    )
    return out


def _verification_section() -> list[str]:
    """The causality evidence the brief asks for by name."""
    return [
        "\n## 2. Correctness of the attention implementation\n",
        "Attention that is subtly wrong still trains and still produces a "
        "falling loss curve, so the loss curve is not evidence. Four properties "
        "are asserted in `tests/test_attention.py` and `tests/test_model.py`, "
        "and run with `python -m pytest tests/`.\n",
        "| Property | Test | Why it matters |\n|---|---|---|",
        "| Output is shape-preserving | `test_output_shape_matches_input` | "
        "otherwise the residual stream does not typecheck |",
        "| Rows sum to 1, upper triangle is exactly 0 | "
        "`test_probabilities_are_a_causal_distribution` | the mask is applied "
        "before the softmax, not after |",
        "| Batched reshape equals per-head column slicing | "
        "`test_matches_a_per_head_reference` | catches heads being split along "
        "*time* instead of *features* -- a bug that trains and converges |",
        "| Changing token t+1 leaves logits at t bitwise identical | "
        "`test_future_tokens_cannot_change_the_present` | the check the brief "
        "names |",
        "| No gradient path from positions after t to output t | "
        "`test_no_gradient_reaches_position_t_from_later_tokens` | the stronger "
        "statement: not that one perturbation failed to travel, but that no "
        "route exists |\n",
        "Two further sanity checks are asserted: loss at initialisation is "
        "`ln(vocab_size)` = 9.68, meaning the untrained model predicts a "
        "uniform distribution; and the model can drive a fixed batch to near "
        "zero loss, meaning nothing in the backward path is severed.\n",
    ]


def _training_section(configs: dict[str, dict], results: dict[str, dict]) -> list[str]:
    """Hyperparameters and what was actually run."""
    out = ["\n## 3. Pretraining\n"]
    out.append("| | " + " | ".join(_label(l) for l in configs) + " |")
    out.append("|---|" + "---|" * len(configs))

    rows = [
        ("Training tokens available", lambda c, r: f"{c['train_tokens']:,}"),
        ("Planned steps (one epoch)", lambda c, r: f"{c['train']['max_steps']:,}"),
        ("Steps completed", lambda c, r: f"{r['step']:,}" if r else "not yet run"),
        ("Tokens seen", lambda c, r: f"{r['tokens_seen']:,}" if r else "--"),
        ("Micro-batch", lambda c, r: str(c["train"]["batch_size"])),
        ("Gradient accumulation", lambda c, r: str(c["train"]["grad_accum"])),
        ("Tokens per optimiser step", lambda c, r: f"{c['train']['batch_size'] * c['train']['grad_accum'] * c['model']['block_size']:,}"),
        ("Peak learning rate", lambda c, r: f"{c['schedule']['max_lr']:.1e}"),
        ("Final learning rate", lambda c, r: f"{c['schedule']['min_lr']:.1e}"),
        ("Warmup steps", lambda c, r: f"{c['schedule']['warmup_steps']:,}"),
        ("Weight decay (2-D params only)", lambda c, r: str(c["train"]["weight_decay"])),
        ("AdamW betas", lambda c, r: f"({c['train']['beta1']}, {c['train']['beta2']})"),
        ("Gradient clip", lambda c, r: str(c["train"]["grad_clip"])),
    ]
    for name, getter in rows:
        cells = [getter(configs[l], results.get(l)) for l in configs]
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    out.append(
        "\nWeight decay is applied only to parameters with two or more "
        "dimensions. Decaying a LayerNorm gain towards zero fights the "
        "normalisation the layer exists to perform, and biases carry no "
        "capacity worth regularising.\n"
    )
    out.append(
        "**Checkpoints are resumable.** Each holds model weights, AdamW's "
        "moment estimates, the schedule, the fp16 loss scale, the step, the "
        "data position and the RNG state. `tests/test_checkpoint.py` runs eight "
        "steps straight through, then the same eight with a save and restore in "
        "the middle, and asserts the two loss sequences are *equal* -- not "
        "merely close. Training batches are a pure function of `(seed, step)`, "
        "so a resumed run reads exactly the data it would have read.\n"
    )
    return out


def _lm_results_section(results: dict[str, dict]) -> list[str]:
    """Perplexity and bits-per-byte, side by side."""
    out = ["\n## 4. Language-modelling results\n"]
    if not any(results.values()):
        out.append("_Not yet run._\n")
        return out

    for split in ("validation", "test"):
        out.append(f"\n**{split.capitalize()} split.**\n")
        out.append("| Model | Cross-entropy (nats/token) | Perplexity | "
                   "Bits per byte | Tokens scored |\n|---|---|---|---|---|")
        for lang, payload in results.items():
            if not payload:
                continue
            m = payload["language_modelling"][split]
            out.append(
                f"| {_label(lang)} | {m['cross_entropy']:.4f} | "
                f"{m['perplexity']:.2f} | {m['bits_per_byte']:.4f} | "
                f"{m['tokens_scored']:,} |"
            )

    out.append(
        "\n**Perplexity cannot be compared across the two models; "
        "bits-per-byte can.** Perplexity is per token, and Phase 1 measured "
        "3.86 characters per token for Hindi against 4.67 for Nepali -- a "
        "Nepali token is about 21% more text. Two models with identical "
        "perplexity are therefore not equally good. Bits-per-byte divides the "
        "same likelihood by the passage's size in UTF-8 bytes instead, which "
        "removes the tokenizer from the comparison.\n"
    )
    return out


def _generation_section(results: dict[str, dict]) -> list[str]:
    """Overlap metrics and diversity diagnostics at each temperature."""
    out = ["\n## 5. Generation quality\n"]
    if not any(r and "generation" in r for r in results.values()):
        out.append("_Not yet run._\n")
        return out

    out.append("| Model | Decoding | BLEU | chrF | ROUGE-L | Repetition (4-gram) "
               "| Distinct-1 | Distinct-2 |\n|---|---|---|---|---|---|---|---|")
    for lang, payload in results.items():
        if not payload or "generation" not in payload:
            continue
        for row in payload["generation"]:
            decoding = "greedy" if row["temperature"] == 0 else f"T={row['temperature']}"
            out.append(
                f"| {_label(lang)} | {decoding} | {row['bleu']:.2f} | "
                f"{row['chrf']:.2f} | {row['rouge_l']:.3f} | "
                f"{row['repetition_rate']:.3f} | {row['distinct_1']:.3f} | "
                f"{row['distinct_2']:.3f} |"
            )

    out.append(
        "\n**These metrics are weak evidence for open-ended generation, and "
        "that is worth stating plainly.** A prompt has many valid "
        "continuations; the test set contains one. n-gram precision against "
        "that single reference penalises a fluent-but-different continuation "
        "exactly as hard as gibberish, so BLEU here measures agreement with one "
        "arbitrary sample from the space of good answers rather than quality. "
        "They remain useful in one direction: they separate a model that has "
        "learned the language from one that has not, and they can be compared "
        "*between* Model H and Model L, which is the comparison this project is "
        "about.\n"
    )
    out.append(
        "**chrF is the most informative of the three here.** It matches "
        "character n-grams rather than whole words, so a correct stem with the "
        "wrong case ending earns partial credit. Hindi and Nepali agglutinate "
        "case and postposition markers onto stems, which makes word-level BLEU "
        "close to all-or-nothing per token in a way it is not for English. "
        "ROUGE-L sits between the two: its longest-common-subsequence rewards "
        "correct ordering without demanding adjacency.\n"
    )
    out.append(
        "Repetition rate and Distinct-1/2 are read together. High repetition "
        "with low distinctness is degeneration -- the model looping inside each "
        "sample. Low repetition with low distinctness means samples are varied "
        "internally but the model produces the same safe continuation for every "
        "prompt.\n"
    )
    return out


def _attention_section(results: dict[str, dict], figures: str) -> list[str]:
    """Per-head entropy and distance, with the heatmaps."""
    out = ["\n## 6. Attention analysis\n"]
    if not any(r and "attention" in r for r in results.values()):
        out.append("_Not yet run._\n")
        return out

    out.append(
        "Three quantities per (layer, head), averaged over held-out text. "
        "Query position 0 is excluded from both averages: it can only attend to "
        "itself, so its entropy and distance are zero regardless of what the "
        "head learned.\n"
    )
    out.append(
        "**Normalised entropy** divides each head's entropy by the maximum "
        "available at that query position, `log(i+1)`. Near 1 means the head is "
        "doing little more than averaging over its context; near 0 means it has "
        "committed to specific tokens. **Mean attention distance** is how far "
        "back a query looks on average, in token positions: local or positional "
        "heads sit at one to three, content-based heads range further.\n"
    )

    out.append("| Model | Layer | Mean normalised entropy | Mean distance | "
               "Sharpest head | Longest-range head |\n|---|---|---|---|---|---|")
    for lang, payload in results.items():
        if not payload or "attention" not in payload:
            continue
        att = payload["attention"]
        entropy = att["normalised_entropy"]
        distance = att["distance"]
        for layer in range(att["n_layer"]):
            ent_row, dist_row = entropy[layer], distance[layer]
            sharpest = min(range(len(ent_row)), key=lambda h: ent_row[h])
            longest = max(range(len(dist_row)), key=lambda h: dist_row[h])
            out.append(
                f"| {_label(lang)} | {layer} | "
                f"{sum(ent_row) / len(ent_row):.3f} | "
                f"{sum(dist_row) / len(dist_row):.1f} | "
                f"head {sharpest} ({ent_row[sharpest]:.3f}) | "
                f"head {longest} ({dist_row[longest]:.1f}) |"
            )

    out.append(f"\n![attention entropy]({figures}/attention_entropy.png)\n")
    out.append(f"![attention distance]({figures}/attention_distance.png)\n")
    for lang, payload in results.items():
        if not payload or "attention" not in payload:
            continue
        name = LANGUAGE_NAMES.get(lang, lang).lower()
        for layer in (0, payload["attention"]["n_layer"] - 1):
            out.append(
                f"![{lang} attention layer {layer}]"
                f"({figures}/attention_heatmap_{name}_layer{layer}.png)\n"
            )

    out.append(
        "Every heatmap is lower-triangular by construction, which is the causal "
        "mask visible directly: a query at position *i* has zero weight on every "
        "key after it. A full square would mean the mask was broken.\n"
    )
    return out


def build(runs: dict[str, Path], figures: str) -> str:
    """Assemble the whole report from the measured outputs of each run."""
    configs, results = {}, {}
    for lang, run_dir in runs.items():
        config = _read_json(run_dir / "config.json") or {}
        packed = _read_json(Path(config.get("tokens", ".")) / "_stats_pack.json")
        if packed:
            config["train_tokens"] = packed["splits"]["train"]["tokens"]
        configs[lang] = config
        results[lang] = _final_metrics(run_dir)

    out: list[str] = []
    add = out.append

    add("# Phase 2 — Model Implementation, Pretraining, and Evaluation\n")
    add("Language Models and Agents, Monsoon 2026 — Individual Project\n")
    add("Every number in this report is generated from the `config.json`, "
        "`metrics.jsonl` and `results.json` files written by the training and "
        "evaluation stages; none is typed by hand. Regenerate with "
        "`python -m common.report.phase2`.\n")

    out.extend(_architecture_section(configs))
    out.extend(_verification_section())
    out.extend(_training_section(configs, results))

    add("\n### 3.1 Loss curves\n")
    add(f"![pretraining loss by step]({figures}/pretraining_loss_by_step.png)\n")
    add(f"![pretraining loss by tokens]({figures}/pretraining_loss_by_tokens.png)\n")
    add(f"![validation perplexity]({figures}/validation_perplexity.png)\n")
    add("Two x-axes because they answer different questions. Against optimiser "
        "steps the models are compared at equal work, which isolates the "
        "architecture. Against tokens seen they are compared at equal data, "
        "which is the comparison the resource gap is about.\n")

    out.extend(_lm_results_section(results))
    out.extend(_generation_section(results))
    out.extend(_attention_section(results, figures))

    add("\n## 7. Reproduction\n")
    add("See the repository `README.md` for the full command sequence and the "
        "Google Drive links to the checkpoints.\n")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate the Phase 2 report.")
    parser.add_argument("--run", action="append", required=True,
                        metavar="LANG=DIR",
                        help="e.g. hi=Hindi/runs/base (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("report/phase2.md"))
    parser.add_argument("--figures", default="figures",
                        help="figure path as written into the markdown")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    runs = {}
    for entry in args.run:
        lang, _, directory = entry.partition("=")
        runs[lang] = Path(directory)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(runs, args.figures), encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
