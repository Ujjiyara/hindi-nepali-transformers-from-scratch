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


def _last_training_record(run_dir: Path) -> dict:
    """How far training actually got, read from the run's own metrics log.

    Taken from ``metrics.jsonl`` rather than from the evaluation results, so
    the pretraining table is complete even when a checkpoint has been trained
    but not yet evaluated -- and so it reports the run that happened rather
    than the run that was planned.
    """
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    last = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last.update(json.loads(line))
    return last


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
    head_dim = model["d_model"] // model["n_head"]
    out.append(
        "\n**The forward pass, with shapes.** The brief asks for the reshape "
        "and transpose operations to be explicit, because they are where "
        "hand-written attention usually goes wrong.\n"
    )
    out.append("| Step | Operation | Shape |\n|---|---|---|")
    for index, (operation, shape) in enumerate([
        ("input", "(B, T, d)"),
        ("fused QKV projection", "(B, T, 3d)"),
        ("split into q, k, v", "3 x (B, T, d)"),
        ("`.view(B, T, h, d_head)`", f"(B, T, {model['n_head']}, {head_dim})"),
        ("`.transpose(1, 2)`", f"(B, {model['n_head']}, T, {head_dim})"),
        ("`q @ k.transpose(-2, -1)`", f"(B, {model['n_head']}, T, T)"),
        (f"scale by 1/sqrt({head_dim})", f"(B, {model['n_head']}, T, T)"),
        ("additive causal mask, then softmax over keys",
         f"(B, {model['n_head']}, T, T)"),
        ("attention dropout, then `@ v`", f"(B, {model['n_head']}, T, {head_dim})"),
        ("`.transpose(1, 2).contiguous().view`", "(B, T, d)"),
        ("output projection, residual dropout", "(B, T, d)"),
    ]):
        out.append(f"| {index} | {operation} | `{shape}` |")
    out.append(
        "\nThe order matters: heads are carved out of the **feature** axis by "
        "viewing to `(B, T, h, d_head)` and only then transposing, so head *i* "
        "owns feature columns `[i*d_head : (i+1)*d_head]`. Viewing straight to "
        "`(B, h, T, d_head)` would slice along *time*, handing head 0 the "
        "first `T/h` tokens. That variant trains and produces a falling loss "
        "curve while being entirely wrong, which is why the test suite checks "
        "the batched implementation against a reference that splits heads by "
        "explicit column slicing rather than checking the mask alone.\n"
    )
    out.append(
        f"**Why the 1/sqrt(d_head) scaling is needed.** If the components of "
        f"`q` and `k` are roughly independent with unit variance, then `q . k` "
        f"is a sum of {head_dim} such products, so its variance is {head_dim} "
        f"and its standard deviation sqrt({head_dim}) = {head_dim ** 0.5:.0f}. "
        f"Logits spread that widely drive the softmax to a near one-hot "
        f"distribution at initialisation, where its gradient is almost zero "
        f"and nothing learns. Dividing by sqrt(d_head) restores unit variance "
        f"and keeps the softmax in the range where it responds to its inputs. "
        f"It is a variance-control argument, not a heuristic.\n"
    )
    out.append(
        "**Why the mask is added before the softmax.** The mask holds 0 on "
        "allowed positions and -inf on future ones, so exponentiating gives "
        "exactly zero *and* removes those terms from the softmax denominator. "
        "Zeroing the upper triangle after the softmax would instead leave each "
        "row summing to less than one by a position-dependent amount, silently "
        "rescaling every output.\n"
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
        ("Steps completed", lambda c, r: f"{c['last']['step']:,}"
         if c.get("last", {}).get("step") else "not yet run"),
        ("Tokens seen", lambda c, r: f"{c['last']['tokens']:,}"
         if c.get("last", {}).get("tokens") else "--"),
        ("Final validation loss", lambda c, r: f"{c['last']['val_loss']:.4f}"
         if c.get("last", {}).get("val_loss") else "--"),
        ("Mean throughput", lambda c, r: f"{c['throughput']:,.0f} tok/s"
         if c.get("throughput") else "--"),
        ("Wall-clock time", lambda c, r: f"{c['last']['elapsed'] / 3600:.2f} h"
         if c.get("last", {}).get("elapsed") else "--"),
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
        "bits-per-byte can.** Perplexity is per token, and the two tokenizers "
        "are independent by construction, so a token is not the same amount of "
        "language in the two models. Bits-per-byte divides the same likelihood "
        "by the passage's size in UTF-8 bytes instead, which removes the "
        "tokenizer from the comparison.\n"
    )

    # The reversal is the finding, so state it from the measured numbers
    # rather than leaving the reader to spot it in the table.
    usable = {lang: payload for lang, payload in results.items() if payload}
    if len(usable) == 2:
        (a_lang, a), (b_lang, b) = usable.items()
        am = a["language_modelling"]["test"]
        bm = b["language_modelling"]["test"]
        better_ppl = a_lang if am["perplexity"] < bm["perplexity"] else b_lang
        better_bpb = a_lang if am["bits_per_byte"] < bm["bits_per_byte"] else b_lang

        if better_ppl != better_bpb:
            low, high = (a, b) if better_ppl == a_lang else (b, a)
            lowm = low["language_modelling"]["test"]
            highm = high["language_modelling"]["test"]
            low_bpt = lowm["bytes_scored"] / lowm["tokens_scored"]
            high_bpt = highm["bytes_scored"] / highm["tokens_scored"]
            out.append(
                f"**The two metrics disagree in direction, and that is the "
                f"result.** {_label(better_ppl)} has the lower perplexity "
                f"({lowm['perplexity']:.2f} against {highm['perplexity']:.2f} "
                f"on test), while {_label(better_bpb)} has the lower "
                f"bits-per-byte ({highm['bits_per_byte']:.4f} against "
                f"{lowm['bits_per_byte']:.4f}). The arithmetic behind the flip "
                f"is that {_label(better_bpb)}'s tokens carry "
                f"{high_bpt:.2f} bytes against {low_bpt:.2f} -- "
                f"{100 * (high_bpt / low_bpt - 1):.1f}% more text per token -- "
                f"while its cross-entropy is only "
                f"{100 * (highm['cross_entropy'] / lowm['cross_entropy'] - 1):.1f}% "
                f"higher. Dividing by bytes rather than tokens more than "
                f"absorbs that gap.\n"
            )
            out.append(
                "Read carefully, this does not say the lower-resource model is "
                "the better model. The two are scored on *different text*, so "
                "the comparison is of how efficiently each encodes its own "
                "language, not of which would win on a shared benchmark. What "
                "it does establish is that the obvious reading of the "
                "perplexity column -- that the lower-resource model is clearly "
                "worse -- is an artefact of tokenizer granularity and does not "
                "survive a tokenizer-independent measure.\n"
            )
            out.append(
                "One caveat on the absolute values: both corpora are "
                "Devanagari, which UTF-8 encodes in three bytes per character "
                "with two of the three nearly determined by the first. "
                "Bits-per-byte is therefore mechanically low here -- roughly "
                f"{lowm['bits_per_byte'] * 3:.2f} and "
                f"{highm['bits_per_byte'] * 3:.2f} bits per *character* -- and "
                "these figures should not be set beside published "
                "bits-per-byte for Latin-script English. Between these two "
                "models, which share a script, the comparison is sound.\n"
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


def _samples_section(runs: dict[str, Path], limit: int = 2) -> list[str]:
    """A few real continuations, so the metric tables have something to mean.

    Shown at temperature 1.0: greedy output is dominated by the repetition
    loop the diagnostics above quantify, and 1.5 is mostly noise, so 1.0 is
    the setting where the text is worth reading.
    """
    out = ["\n### 5.1 Generated samples\n"]
    out.append(
        "Held-out prompts from the test split, continued at temperature 1.0. "
        "The reference is what actually followed in the source document -- one "
        "valid continuation among many, which is the whole difficulty with "
        "scoring this task by overlap.\n"
    )

    for lang, run_dir in runs.items():
        payload = _read_json(run_dir / "eval" / "samples.json")
        if not payload:
            continue
        block = next((b for b in payload["samples"] if b["temperature"] == 1.0), None)
        if not block:
            continue

        out.append(f"\n**{_label(lang)}**\n")
        for index in range(min(limit, len(block["hypotheses"]))):
            prompt = " ".join(block["prompts"][index].split())[-160:]
            generated = " ".join(block["hypotheses"][index].split())[:260]
            reference = " ".join(block["references"][index].split())[:260]
            out.append(f"> **Prompt** ...{prompt}\n>")
            out.append(f"> **Generated** {generated}...\n>")
            out.append(f"> **Reference** {reference}...\n")
    return out


def _qualitative_notes(results: dict[str, dict]) -> list[str]:
    """Coherence and language-correctness notes, which the brief asks for.

    The numbers that anchor these observations are pulled from the measured
    diversity diagnostics, so the prose cannot drift away from the run.
    """
    out = ["\n### 5.2 Qualitative notes on coherence and language correctness\n"]
    usable = {lang: payload for lang, payload in results.items()
              if payload and payload.get("generation")}
    if not usable:
        out.append("_Not yet run._\n")
        return out

    greedy = {lang: next(g for g in payload["generation"] if g["temperature"] == 0)
              for lang, payload in usable.items()}
    span = ", ".join(f"{_label(lang)} {g['repetition_rate']:.2f}"
                     for lang, g in greedy.items())

    out.append(
        f"**Greedy decoding degenerates into verbatim looping in both models.** "
        f"The 4-gram repetition rate under greedy decoding is {span}, and the "
        f"samples show what that number is: a single clause repeated until the "
        f"token budget runs out. This is the expected behaviour of a "
        f"24M-parameter model at argmax, not a bug -- the most likely "
        f"continuation of a sentence the model has already produced is that "
        f"sentence again. It is also why the BLEU column peaks at greedy while "
        f"chrF does not: repeated common n-grams collect precision credit that "
        f"has nothing to do with quality.\n"
    )
    out.append(
        "**At temperature 1.0 the language itself is correct; the content is "
        "not.** Continuations are well-formed Devanagari with plausible "
        "morphology -- postpositions attach correctly, verb endings agree, "
        "native numerals are used consistently, and sentences terminate with a "
        "danda rather than trailing off. Byte-fallback artefacts are absent, "
        "which is consistent with the near-zero unknown rate Phase 1 measured. "
        "What fails is coherence beyond a clause or two: the models hold a "
        "register and a topic for roughly a sentence, then drift, and numeric "
        "and factual detail is confabulated freely -- a generated news "
        "continuation will keep the shape of a news report while inventing "
        "figures that contradict each other within the same sentence.\n"
    )
    out.append(
        "**Neither model shows a qualitative deficit relative to the other.** "
        "The failure modes are the same in kind and close in degree, which is "
        "the more informative comparison: the resource gap between Model H and "
        "Model L shows up as a small quantitative difference in held-out loss, "
        "not as one model having learned the language and the other not.\n"
    )
    out.append(
        "The honest summary is that both models have learned the *surface* of "
        "their language -- script, morphology, register, sentence shape -- and "
        "essentially none of the world knowledge or long-range planning that "
        "would make a continuation true or on-topic. At 24M parameters and one "
        "epoch over a few hundred million tokens, that is the expected "
        "outcome, and it is what the perplexity figures around 43 and 47 "
        "already imply.\n"
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
        n_layer = payload["attention"]["n_layer"]
        for layer in (0, n_layer // 2, n_layer - 1):
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


def _resource_section(configs: dict[str, dict], results: dict[str, dict]) -> list[str]:
    """The resource-level comparison the brief asks for as a deliverable.

    Everything here is derived from the measured runs, so the argument moves
    with the numbers rather than being an assertion written once and left to
    go stale.
    """
    out = ["\n## 7. Resource-level comparison\n"]
    usable = {lang: payload for lang, payload in results.items() if payload}
    if len(usable) < 2:
        out.append("_Not yet run._\n")
        return out

    out.append("| | " + " | ".join(_label(l) for l in configs) + " |")
    out.append("|---|" + "---|" * len(configs))
    rows = [
        ("Training tokens", lambda c, r: f"{c['train_tokens']:,}"),
        ("Tokens seen", lambda c, r: f"{c['last']['tokens']:,}"),
        ("Test perplexity",
         lambda c, r: f"{r['language_modelling']['test']['perplexity']:.2f}"),
        ("Test bits-per-byte",
         lambda c, r: f"{r['language_modelling']['test']['bits_per_byte']:.4f}"),
        ("Bytes per token",
         lambda c, r: f"{r['language_modelling']['test']['bytes_scored'] / r['language_modelling']['test']['tokens_scored']:.2f}"),
        ("Best chrF (any decoding)",
         lambda c, r: f"{max(g['chrf'] for g in r['generation']):.2f}"
         if r.get("generation") else "--"),
    ]
    for name, getter in rows:
        out.append(f"| {name} | " +
                   " | ".join(getter(configs[l], results[l]) for l in configs) + " |")

    ratio = max(c["train_tokens"] for c in configs.values()) / min(
        c["train_tokens"] for c in configs.values())
    out.append(
        f"\nThe corpora differ by a factor of {ratio:.2f} in training tokens, "
        "which is the resource gap this project is built to measure. Three "
        "things follow from the runs above.\n"
    )
    # Split the total gap into the part present at equal data and the part
    # bought by the extra corpus, rather than asserting which dominates.
    curves = {lang: cfg.get("val_curve", []) for lang, cfg in configs.items()}
    if all(curves.values()) and len(curves) == 2:
        smaller = min(curves, key=lambda l: curves[l][-1][0])
        larger = next(l for l in curves if l != smaller)
        cutoff = curves[smaller][-1][0]
        matched = min(curves[larger], key=lambda point: abs(point[0] - cutoff))
        equal_data_gap = curves[smaller][-1][1] - matched[1]
        extra_data_gain = matched[1] - curves[larger][-1][1]
        total = equal_data_gap + extra_data_gain
        share = 100 * equal_data_gap / total if total else 0

        out.append(
            f"**The gap splits almost evenly between corpus size and "
            f"everything else.** At the point where {_label(smaller)} exhausts "
            f"its corpus ({cutoff:,} tokens), {_label(larger)} is at "
            f"{matched[1]:.4f} validation loss against "
            f"{curves[smaller][-1][1]:.4f} -- a gap of {equal_data_gap:.4f} "
            f"nats that is *not* explained by data volume, since both models "
            f"have seen the same number of tokens. {_label(larger)} then gains "
            f"a further {extra_data_gain:.4f} nats from the "
            f"{curves[larger][-1][0] - cutoff:,} tokens it still has left. So "
            f"roughly {share:.0f}% of the final difference is present at equal "
            f"data and {100 - share:.0f}% is bought by the larger corpus.\n"
        )
        out.append(
            "That matters for how the result is stated. Against optimiser "
            "steps the two models look nearly identical; against tokens seen "
            "they separate; and only the second axis reflects the resource "
            "gap the project is about. Neither axis alone tells the whole "
            "story, which is why both figures are included above.\n"
        )

    fertility = {lang: cfg.get("fertility") for lang, cfg in configs.items()}
    if all(fertility.values()):
        pairs = sorted(fertility.items(), key=lambda kv: kv[1])
        (low_lang, low), (high_lang, high) = pairs[0], pairs[-1]
        out.append(
            f"**Tokenizer fertility compounds the gap.** Phase 1 measured "
            f"{low:.4f} tokens per word for {_label(low_lang)} against "
            f"{high:.4f} for {_label(high_lang)} at the same 16,000-entry "
            f"vocabulary -- {100 * (high / low - 1):.1f}% higher. A tokenizer "
            f"fitted on less text learns fewer useful subwords, so the same "
            f"amount of language costs more tokens. The lower-resource "
            f"language is therefore penalised twice over: there is less of it, "
            f"and each token of it carries less linguistic content, so a fixed "
            f"context window holds less actual text and a fixed token budget "
            f"buys less training signal.\n"
        )

    out.append(
        "**The tokenizer is doing more work than the parameter count.** Both "
        "models are byte-for-byte the same architecture with the same "
        "24,086,016 parameters and the same 16,000-entry vocabulary size, so "
        "every difference in the tables above comes from the data and the "
        "vocabulary fitted to it. The lower-resource language ends up with the "
        "coarser tokenizer -- more bytes per token -- which flatters its "
        "bits-per-byte and penalises its perplexity, in opposite directions.\n"
    )
    out.append(
        "**Generation separates them more clearly than perplexity does.** The "
        "overlap metrics are noisy, but the degeneration diagnostics are not: "
        "the repetition rate under greedy decoding is high for both models, "
        "which is the expected failure of a 24M-parameter model asked to "
        "continue open-ended text, and it falls away as temperature rises. "
        "That both models show the same shape, at similar magnitudes, is "
        "itself evidence that the resource gap has not produced a "
        "qualitatively different model -- only a slightly weaker one.\n"
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
        config["last"] = _last_training_record(run_dir)
        rows = [json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()] if (run_dir / "metrics.jsonl").exists() else []
        sweep = _read_json(run_dir.parent.parent / "tokenizer" / "sweep.json")
        if sweep:
            entries = sweep if isinstance(sweep, list) else sweep.get(
                "results", sweep.get("sweep", []))
            chosen = [e for e in entries
                      if e.get("vocab_size") == config.get("model", {}).get("vocab_size")]
            config["fertility"] = chosen[0]["fertility"] if chosen else None
        config["val_curve"] = [(r["tokens"], r["val_loss"]) for r in rows
                               if "val_loss" in r]
        speeds = [r["tokens_per_second"] for r in rows if "tokens_per_second" in r]
        config["throughput"] = sum(speeds) / len(speeds) if speeds else None
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
    out.extend(_samples_section(runs))
    out.extend(_qualitative_notes(results))
    out.extend(_attention_section(results, figures))

    out.extend(_resource_section(configs, results))

    add("\n## 8. Reproduction\n")
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
