"""Generate the LaTeX tables for the consolidated final report.

Usage::

    python -m common.report.final --lang hi=Hindi --lang ne=Nepali \
        --out report/final/tables

Writes one ``.tex`` file per table into ``--out``. Each file holds only a
``tabular`` body; the caption, label and placement live in
``report/final/report.tex``, so the prose around a table can be edited without
touching this script.

Like the Phase 1 and Phase 2 reports, every number comes from a file a pipeline
stage wrote -- ``_stats_pipeline.json``, ``_token_counts.json``, ``sweep.json``,
``runs/base/{config.json,metrics.jsonl,eval/results.json}`` and
``runs/finetune/{dataset_stats.json,eval/*.json,eval/*_predictions.jsonl}`` --
so a table cannot drift from the run that produced it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from common.report.phase1 import SOURCE_NOTES

log = logging.getLogger("report")

MODEL = {"hi": "Model H", "ne": "Model L"}
LANGUAGE = {"hi": "Hindi", "ne": "Nepali"}
TASKS = ("numeric_pair", "numeric_triple", "chain_superlative", "chain_relation")
TASK_NAMES = {
    "numeric_pair": "Numeric pair",
    "numeric_triple": "Numeric triple",
    "chain_superlative": "Chain superlative",
    "chain_relation": "Chain relation (yes/no)",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _json(path: Path) -> dict:
    """Read a JSON file, failing loudly: a table built on a missing file is wrong."""
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file into a list of records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _pct(value: float, digits: int = 1) -> str:
    """A fraction as a LaTeX percentage, e.g. ``0.295`` -> ``29.5\\%``."""
    return f"{100 * value:.{digits}f}\\%"


def _tabular(spec: str, header: list[str], rows: list[str]) -> str:
    """A booktabs tabular from a column spec, header lines and body rows."""
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", *header, "\\midrule", *rows,
             "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _last_record(rows: list[dict]) -> dict:
    """The latest value of every key in a metrics log."""
    last: dict = {}
    for row in rows:
        last.update(row)
    return last


# --------------------------------------------------------------------------
# Phase 1: corpora and tokenizers
# --------------------------------------------------------------------------

def corpus_table(roots: dict[str, Path]) -> str:
    """Corpus scale, cleaning and the manual share, Model H beside Model L."""
    data = {}
    for lang, root in roots.items():
        pipeline = _json(root / "data" / "processed" / "_stats_pipeline.json")
        counts = _json(root / "data" / "processed" / "_token_counts.json")
        methods = [SOURCE_NOTES.get(name, ("", "unlabelled"))[1]
                   for name in pipeline["per_source"]]
        data[lang] = (pipeline, counts, methods)

    def row(label: str, fn) -> str:
        return f"{label} & " + " & ".join(fn(*data[lang]) for lang in roots) + r" \\"

    rows = [
        row("Sources (manual / downloaded)",
            lambda p, c, m: f"{m.count('manual')} / {m.count('downloaded')}"),
        row("Documents loaded", lambda p, c, m: f"{p['totals']['docs_loaded']:,}"),
        row("Removed by quality filters", lambda p, c, m:
            f"{p['totals']['docs_loaded'] - p['totals']['docs_normalised']:,}"),
        row("Exact / near duplicates removed", lambda p, c, m:
            f"{p['totals']['dropped_exact_duplicate']:,} / "
            f"{p['totals']['dropped_near_duplicate']:,}"),
        row("Final documents", lambda p, c, m: f"{p['totals']['docs_after_dedup']:,}"),
        row("Characters", lambda p, c, m: f"{p['corpus']['characters']:,}"),
        row("Train / val / test tokens", lambda p, c, m: " / ".join(
            f"{c['splits'][s]['tokens'] / 1e6:.1f}M" for s in ("train", "validation", "test"))),
        row("Manual share of training tokens",
            lambda p, c, m: _pct(c["splits"]["train"]["manual_token_fraction"])),
        row("Characters per token", lambda p, c, m: f"{c['corpus']['chars_per_token']:.2f}"),
        row("Training tokens vs.\\ 500M target",
            lambda p, c, m: f"{c['splits']['train']['tokens'] / 5e8:.2f}$\\times$"),
    ]
    header = [" & " + " & ".join(MODEL[l] for l in roots) + r" \\"]
    return _tabular("l" + "r" * len(roots), header, rows)


def tokenizer_table(roots: dict[str, Path], chosen: int = 16000) -> str:
    """The vocabulary sweep for both languages, with the chosen size in bold."""
    sweeps = {lang: {r["vocab_size"]: r for r in _json(root / "tokenizer" / "sweep.json")["sweep"]}
              for lang, root in roots.items()}
    sizes = sorted(next(iter(sweeps.values())))
    langs = list(roots)

    header = [
        "& \\multicolumn{2}{c}{Fertility} & \\multicolumn{2}{c}{Chars/token} "
        "& \\multicolumn{2}{c}{Byte fallback} & \\multicolumn{2}{c}{Vocab used} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "Vocab & " + " & ".join(["H", "L"] * 4) + r" \\",
    ]
    rows = []
    for size in sizes:
        cells = []
        for key, fmt in (("fertility", "{:.3f}"), ("chars_per_token", "{:.2f}"),
                         ("byte_fallback_rate", "{:.4f}"), ("vocab_utilisation", None)):
            for lang in langs:
                value = sweeps[lang][size][key]
                cells.append(_pct(value) if fmt is None else fmt.format(value))
        label = f"{size:,}"
        if size == chosen:
            label = f"\\textbf{{{label}}}"
            cells = [f"\\textbf{{{c}}}" for c in cells]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    return _tabular("l" + "r" * 8, header, rows)


# --------------------------------------------------------------------------
# Phase 2: pretraining and evaluation
# --------------------------------------------------------------------------

def pretraining_table(roots: dict[str, Path]) -> str:
    """What each model saw during pretraining and how long it took."""
    info = {}
    for lang, root in roots.items():
        run = root / "runs" / "base"
        config = _json(run / "config.json")
        metrics = _jsonl(run / "metrics.jsonl")
        packed = _json(root / "data" / "tokens" / "_stats_pack.json")
        info[lang] = {
            "config": config,
            "last": _last_record(metrics),
            "packed": packed["splits"]["train"]["tokens"],
            "throughput": np.mean([m["tokens_per_second"] for m in metrics
                                   if "tokens_per_second" in m]),
        }

    def row(label: str, fn) -> str:
        return f"{label} & " + " & ".join(fn(info[lang]) for lang in roots) + r" \\"

    rows = [
        row("Tokens in packed train array", lambda i: f"{i['packed']:,}"),
        row("Optimiser steps (one epoch)", lambda i: f"{i['last']['step']:,}"),
        row("Tokens seen", lambda i: f"{i['last']['tokens']:,}"),
        row("Warmup steps", lambda i: f"{i['config']['schedule']['warmup_steps']}"),
        row("Mean throughput (tok/s)", lambda i: f"{i['throughput']:,.0f}"),
        row("Wall-clock time", lambda i: f"{i['last']['elapsed'] / 3600:.2f} h"),
        row("Final val.\\ loss (nats/token)", lambda i: f"{i['last']['val_loss']:.4f}"),
    ]
    header = [" & " + " & ".join(MODEL[l] for l in roots) + r" \\"]
    return _tabular("l" + "r" * len(roots), header, rows)


def lm_table(roots: dict[str, Path]) -> str:
    """Cross-entropy, perplexity and bits-per-byte on both held-out splits."""
    rows = []
    for lang, root in roots.items():
        lm = _json(root / "runs" / "base" / "eval" / "results.json")["language_modelling"]
        cells = []
        for split in ("validation", "test"):
            s = lm[split]
            cells += [f"{s['cross_entropy']:.3f}", f"{s['perplexity']:.2f}",
                      f"{s['bits_per_byte']:.3f}"]
        test = lm["test"]
        cells.append(f"{test['bytes_scored'] / test['tokens_scored']:.2f}")
        rows.append(f"{LANGUAGE[lang]} ({MODEL[lang]}) & " + " & ".join(cells) + r" \\")
    header = [
        "& \\multicolumn{3}{c}{Validation} & \\multicolumn{3}{c}{Test} & \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        r"Model & CE & PPL & BPB & CE & PPL & BPB & Bytes/token \\",
    ]
    return _tabular("l" + "r" * 7, header, rows)


def generation_table(roots: dict[str, Path]) -> str:
    """Overlap and diversity metrics for every decoding setting."""
    rows = []
    for index, (lang, root) in enumerate(roots.items()):
        if index:
            rows.append("\\midrule")
        generation = _json(root / "runs" / "base" / "eval" / "results.json")["generation"]
        for position, g in enumerate(generation):
            name = MODEL[lang] if position == 0 else ""
            decoding = "greedy" if g["temperature"] == 0 else f"$T={g['temperature']:.1f}$"
            rows.append(
                f"{name} & {decoding} & {g['bleu']:.2f} & {g['chrf']:.2f} & "
                f"{g['rouge_l']:.3f} & {g['repetition_rate']:.3f} & "
                f"{g['distinct_1']:.3f} & {g['distinct_2']:.3f} \\\\"
            )
    header = [r"Model & Decoding & BLEU-4 & chrF & ROUGE-L & Rep-4 & Dist-1 & Dist-2 \\"]
    return _tabular("llrrrrrr", header, rows)


def heads_table(roots: dict[str, Path]) -> str:
    """How many of the 60 heads are local, sharp, diffuse or sink-like.

    Thresholds are the ones fixed in the Phase 2 report: local = mean distance
    below 5 positions, sharp = normalised entropy below 0.4, diffuse = above
    0.7, sink-like = sharp and mean distance above 100.
    """
    rows = []
    for lang, root in roots.items():
        att = _json(root / "runs" / "base" / "eval" / "results.json")["attention"]
        entropy = np.array(att["normalised_entropy"])
        distance = np.array(att["distance"])
        layer, head = np.unravel_index(entropy.argmin(), entropy.shape)
        rows.append(
            f"{LANGUAGE[lang]} ({MODEL[lang]}) & {entropy.size} & {int((distance < 5).sum())} & "
            f"{int((entropy < 0.4).sum())} & {int((entropy > 0.7).sum())} & "
            f"{int(((entropy < 0.4) & (distance > 100)).sum())} & "
            f"L{layer} H{head} ({entropy[layer, head]:.3f}, {distance[layer, head]:.1f}) \\\\"
        )
    header = [r"Model & Heads & Local & Sharp & Diffuse & Sink-like & "
              r"Sharpest head (entropy, distance) \\"]
    return _tabular("lrrrrrl", header, rows)


# --------------------------------------------------------------------------
# Phase 3: reasoning finetune
# --------------------------------------------------------------------------

def reasoning_data_table(roots: dict[str, Path]) -> str:
    """Split sizes, disjoint name pools and prompt lengths for both languages."""
    stats = {lang: _json(root / "runs" / "finetune" / "dataset_stats.json")
             for lang, root in roots.items()}
    for lang, s in stats.items():
        leaks = {k: v for k, v in s["leakage"].items() if v not in (0, {}) and
                 not (isinstance(v, dict) and not any(v.values()))}
        if leaks:
            log.warning("%s: leakage counters are not zero: %s", lang, leaks)

    rows = []
    for split in ("train", "validation", "test", "test_long"):
        first = next(iter(stats.values()))["splits"][split]
        hops = sorted(int(h) for h in first["by_hops"])
        names = []
        for lang in roots:
            pool = stats[lang]["entities"].get("test" if split == "test_long" else split)
            names.append(f"{pool['people']}+{pool['objects']}")
        tokens = [f"{stats[lang]['splits'][split]['tokens_mean']:.1f}" for lang in roots]
        rows.append(
            f"\\texttt{{{split.replace('_', chr(92) + '_')}}} & {first['examples']:,} & "
            f"{first['templates']} & {hops[0]}--{hops[-1]} & " + " & ".join(names) + " & "
            + " & ".join(tokens) + r" \\"
        )
    header = [
        "& & & & \\multicolumn{2}{c}{Names (people+objects)} & "
        "\\multicolumn{2}{c}{Mean tokens} \\\\",
        "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
        r"Split & Examples & Templates & Hops & H & L & H & L \\",
    ]
    return _tabular("lrrrrrrr", header, rows)


def reasoning_results_table(roots: dict[str, Path]) -> str:
    """Pretrained vs finetuned accuracy by task family, against chance."""
    evals = {lang: {stage: _json(root / "runs" / "finetune" / "eval" / f"{stage}.json")
                    for stage in ("pretrained", "finetuned")}
             for lang, root in roots.items()}

    def cells(lang: str, pick) -> list[str]:
        pre, post = pick(evals[lang]["pretrained"]), pick(evals[lang]["finetuned"])
        return [f"{100 * post['chance']:.1f}", f"{100 * pre['choice_accuracy']:.1f}",
                f"\\textbf{{{100 * post['exact_match']:.1f}}}",
                f"{100 * post['choice_accuracy']:.1f}"]

    rows = []
    for task in TASKS:
        row = [TASK_NAMES[task]]
        for lang in roots:
            row += cells(lang, lambda e: e["splits"]["test"]["by_task"][task])
        rows.append(" & ".join(row) + r" \\")
    rows.append("\\midrule")
    for split, label in (("test", "All \\texttt{test}"), ("test_long", "All \\texttt{test\\_long}")):
        row = [label]
        for lang in roots:
            row += cells(lang, lambda e: e["splits"][split]["overall"])
        rows.append(" & ".join(row) + r" \\")
    row = ["\\texttt{test\\_long}, 4 hops"]
    for lang in roots:
        row += cells(lang, lambda e: e["splits"]["test_long"]["by_hops"]["4"])
    rows.append(" & ".join(row) + r" \\")

    header = [
        "& \\multicolumn{4}{c}{Hindi (Model H)} & \\multicolumn{4}{c}{Nepali (Model L)} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "& & Pre & \\multicolumn{2}{c}{Finetuned} & & Pre & \\multicolumn{2}{c}{Finetuned} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){8-9}",
        r"Task (\texttt{test} unless noted) & Chance & choice & exact & choice & Chance & choice & exact & choice \\",
    ]
    return _tabular("l" + "r" * 8, header, rows)


def ablations_table(roots: dict[str, Path]) -> str:
    """The short finetuning runs that led to the final configuration."""
    runs = []
    for lang, root in roots.items():
        base = root / "runs" / "finetune_ablations"
        if base.exists():
            runs += [(lang, d) for d in sorted(base.iterdir()) if (d / "config.json").exists()]
        runs.append((lang, root / "runs" / "finetune"))

    rows = []
    for lang, run in runs:
        config = _json(run / "config.json")
        evals = [m for m in _jsonl(run / "metrics.jsonl") if "val_exact_match" in m]
        train, schedule = config["train"], config["schedule"]
        best = max(evals, key=lambda m: m["val_exact_match"])
        name = "final" if run.name == "finetune" else run.name
        rows.append(
            f"{MODEL[lang][-1]} & \\texttt{{{name.replace('_', chr(92) + '_')}}} & "
            f"{'frozen' if train.get('freeze_embeddings') else 'trained'} & "
            f"{train['batch_size'] * train['grad_accum']} & {schedule['max_lr']:.0e} & "
            f"{evals[-1]['step']} & {best['val_exact_match']:.3f} (step {best['step']}) & "
            f"{evals[-1]['val_exact_match']:.3f} \\\\"
        )
    header = [r"& Run & Embedding & Batch & Peak LR & Steps & Best val.\ exact & Last val.\ exact \\"]
    return _tabular("lllrrrrr", header, rows)


def _bias_rates(preds: list[dict], records: dict[str, dict], answer_key: str) -> dict[str, float]:
    """How often one checkpoint's answers follow each shortcut, on ``test``.

    ``answer_key`` is ``"prediction"`` (greedy output) or ``"choice_prediction"``
    (highest-likelihood candidate). The pretrained model never produces the
    answer format, so its biases are read from the choice it scores highest;
    the finetuned model's from what it actually generates. Passing ``"answer"``
    gives the rate in the gold labels, the baseline each bias is read against.
    """
    relation = [p for p in preds if p["task"] == "chain_relation"]
    pair = [p for p in preds if p["task"] == "numeric_pair" and p["question"] != "equal"]
    superlative = [p for p in preds if p["task"] in ("numeric_triple", "chain_superlative")]
    chains = [p for p in preds if p["task"] == "chain_superlative"]

    # choices are [yes, no] for relations and [a, b, both-equal] for pairs.
    no, equal = relation[0]["choices"][1], pair[0]["choices"][2]
    return {
        "relation_no": np.mean([p[answer_key] == no for p in relation]),
        "pair_equal": np.mean([p[answer_key] == equal for p in pair]),
        # ``entities`` is the order of first mention in the prompt.
        "last_mentioned": np.mean([p[answer_key] == records[p["id"]]["entities"][-1]
                                   for p in superlative]),
        # Inner chain entities appear in two comparison sentences, the two ends
        # in one; the extreme asked for is always an end, never an inner one.
        "inner_entity": np.mean([p[answer_key] in records[p["id"]]["chain"][1:-1]
                                 for p in chains]),
    }


def error_rows(roots: dict[str, Path]) -> dict[str, dict[str, tuple[float, ...]]]:
    """Systematic biases on ``test``: ``(pretrained, finetuned, gold)`` per bias.

    A large gap between a model's rate and the gold rate means the model is
    following a shortcut rather than making the comparison.
    """
    out: dict[str, dict[str, tuple[float, ...]]] = {}
    for lang, root in roots.items():
        records = {r["id"]: r for r in _jsonl(root / "data" / "reasoning" / "test.jsonl")}
        stages = {}
        for stage in ("pretrained", "finetuned"):
            path = root / "runs" / "finetune" / "eval" / f"{stage}_predictions.jsonl"
            stages[stage] = [p for p in _jsonl(path) if p["split"] == "test"]

        pre = _bias_rates(stages["pretrained"], records, "choice_prediction")
        post = _bias_rates(stages["finetuned"], records, "prediction")
        gold = _bias_rates(stages["finetuned"], records, "answer")
        rows = {key: (pre[key], post[key], gold[key]) for key in pre}
        rows["off_format"] = tuple(
            np.mean([p["prediction"] not in p["choices"] for p in stages[stage]])
            for stage in ("pretrained", "finetuned")) + (0.0,)
        out[lang] = rows
    return out


def errors_table(roots: dict[str, Path]) -> str:
    """Pretrained, finetuned and gold rates for each bias found in the predictions."""
    stats = error_rows(roots)
    labels = {
        "off_format": "Greedy output is not a candidate answer",
        "relation_no": "Chain relation: ``no''",
        "pair_equal": "Numeric pair (greater/smaller): ``both equal''",
        "last_mentioned": "Superlatives: the last-mentioned entity",
        "inner_entity": "Chain superlative: an inner chain entity",
    }
    rows = []
    for key, label in labels.items():
        cells = []
        for lang in roots:
            pre, post, gold = stats[lang][key]
            cells += [_pct(pre, 0), _pct(post, 0), "--" if key == "off_format" else _pct(gold, 0)]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    header = [
        "& \\multicolumn{3}{c}{Hindi (Model H)} & \\multicolumn{3}{c}{Nepali (Model L)} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        r"Share of \texttt{test} answers that are\ldots & Pre & FT & Gold & Pre & FT & Gold \\",
    ]
    return _tabular("lrrrrrr", header, rows)


def examples_table(roots: dict[str, Path], per_language: int = 2) -> str:
    """One success and one failure per language, chosen deterministically.

    The first correct and the first incorrect three-entity chain superlative in
    the prediction file, so the examples are reproducible rather than picked.
    """
    rows = []
    for index, (lang, root) in enumerate(roots.items()):
        if index:
            rows.append("\\midrule")
        preds = [p for p in _jsonl(root / "runs" / "finetune" / "eval" / "finetuned_predictions.jsonl")
                 if p["split"] == "test" and p["task"] == "chain_superlative" and p["hops"] == 2]
        chosen = [next(p for p in preds if p["exact"]), next(p for p in preds if not p["exact"])]
        for p in chosen[:per_language]:
            mark = "\\checkmark" if p["exact"] else "$\\times$"
            rows.append(f"{MODEL[lang][-1]} & {p['prompt']} & {p['answer']} & "
                        f"{p['prediction']} {mark} \\\\")
    header = [r"& Prompt & Gold & Finetuned output \\"]
    lines = ["\\begin{tabularx}{\\linewidth}{lXll}", "\\toprule", *header, "\\midrule",
             *rows, "\\bottomrule", "\\end{tabularx}"]
    return "\n".join(lines) + "\n"


def attention_shift_table(roots: dict[str, Path]) -> str:
    """Pretrained vs finetuned attention statistics, averaged over heads."""
    rows = []
    for index, (lang, root) in enumerate(roots.items()):
        if index:
            rows.append("\\midrule")
        att = _json(root / "runs" / "finetune" / "eval" / "attention.json")
        for stage in ("pretrained", "finetuned"):
            s = att["stages"][stage]
            reasoning, text, answer = s["reasoning_prompts"], s["corpus_text"], s["answer_position"]
            entity = np.array(answer["entity"])
            share = np.array(answer["gold_share"])
            name = MODEL[lang] if stage == "pretrained" else ""
            rows.append(
                f"{name} & {stage} & "
                f"{np.mean(reasoning['normalised_entropy']):.3f} & {np.mean(reasoning['distance']):.2f} & "
                f"{np.mean(text['normalised_entropy']):.3f} & {np.mean(text['distance']):.1f} & "
                f"{entity.mean():.3f} & {entity[-1].mean():.3f} & "
                f"{share.mean():.3f} & {share[-1].mean():.3f} \\\\"
            )
    header = [
        "& & \\multicolumn{2}{c}{Reasoning prompts} & \\multicolumn{2}{c}{Corpus text} "
        "& \\multicolumn{2}{c}{Answer pos.\\ $\\to$ entities} "
        "& \\multicolumn{2}{c}{Gold share of entity attn.} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
        r"Model & Checkpoint & Entropy & Dist. & Entropy & Dist. & All layers & Layer 9 & All layers & Layer 9 \\",
    ]
    return _tabular("llrrrrrrrr", header, rows)


TABLES = {
    "corpus": corpus_table,
    "tokenizer": tokenizer_table,
    "pretraining": pretraining_table,
    "lm": lm_table,
    "generation": generation_table,
    "heads": heads_table,
    "reasoning_data": reasoning_data_table,
    "reasoning_results": reasoning_results_table,
    "ablations": ablations_table,
    "errors": errors_table,
    "examples": examples_table,
    "attention_shift": attention_shift_table,
}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate the final report's LaTeX tables.")
    parser.add_argument("--lang", action="append", required=True, metavar="CODE=DIR",
                        help="e.g. hi=Hindi (repeatable, Model H first)")
    parser.add_argument("--out", type=Path, default=Path("report/final/tables"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    roots = {}
    for entry in args.lang:
        code, _, directory = entry.partition("=")
        roots[code] = Path(directory)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, build in TABLES.items():
        path = args.out / f"{name}.tex"
        path.write_text(build(roots), encoding="utf-8")
        log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
