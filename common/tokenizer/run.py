"""Train and evaluate tokenizers, and sweep vocabulary size.

Run from the repository root::

    # sweep vocabulary sizes for one language
    python -m common.tokenizer.run sweep --data-root Nepali/data --lang ne \
        --out-dir Nepali/tokenizer --vocab-sizes 8000,16000,32000

    # draw the comparison figures once both languages have been swept
    python -m common.tokenizer.run plot \
        --sweep ne=Nepali/tokenizer/sweep.json hi=Hindi/tokenizer/sweep.json \
        --out-dir report/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

from common.scrape.io_utils import atomic_write_bytes, utc_now
from common.tokenizer.corpus import export_lines, iter_eval_text
from common.tokenizer.evaluate import evaluate, tokenization_examples
from common.tokenizer.train import TokenizerConfig, train_tokenizer

log = logging.getLogger("tokenizer")

DEFAULT_VOCAB_SIZES = [8000, 16000, 32000]


def cmd_sweep(args: argparse.Namespace) -> int:
    data_root: Path = args.data_root
    train_split = data_root / "processed" / "train.jsonl.gz"
    eval_split = data_root / "processed" / "validation.jsonl.gz"

    for path in (train_split, eval_split):
        if not path.exists():
            log.error("missing %s -- run common.pipeline.run first", path)
            return 1

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # One shared plain-text export feeds every vocabulary size in the sweep,
    # so the only variable across runs is vocab_size.
    with tempfile.TemporaryDirectory() as tmp:
        text_path = Path(tmp) / "train.txt"
        lines, chars = export_lines(train_split, text_path, max_lines=args.max_lines)
        log.info("training text: %s lines, %s chars", f"{lines:,}", f"{chars:,}")

        held_out = list(iter_eval_text(eval_split, max_docs=args.eval_docs))
        log.info("held-out documents: %d", len(held_out))

        results = []
        for vocab_size in args.vocab_sizes:
            cfg = TokenizerConfig(
                vocab_size=vocab_size,
                model_type=args.model_type,
                character_coverage=args.character_coverage,
            )
            prefix = out_dir / f"{args.lang}_{args.model_type}_{vocab_size}"
            model_path = train_tokenizer(text_path, prefix, cfg)

            metrics = evaluate(str(model_path), held_out)
            row = metrics.as_dict()
            row["model_path"] = str(model_path)
            row["model_type"] = args.model_type
            results.append(row)

            log.info(
                "vocab=%-6d fertility=%.3f chars/token=%.2f "
                "unk=%.2e byte_fallback=%.2e utilisation=%.1f%%",
                vocab_size, metrics.fertility, metrics.chars_per_token,
                metrics.unknown_rate, metrics.byte_fallback_rate,
                100 * metrics.vocab_utilisation,
            )

    # Qualitative examples come from the best (lowest-fertility) tokenizer.
    best = min(results, key=lambda r: r["fertility"])
    samples = [text.split("\n")[0][:200] for text in held_out[:5] if text.strip()]
    examples = tokenization_examples(best["model_path"], samples)

    report = {
        "lang": args.lang,
        "generated_at": utc_now(),
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
        "train_split": str(train_split),
        "eval_split": str(eval_split),
        "sweep": results,
        "best_by_fertility": best["vocab_size"],
        "examples": examples,
    }
    out_path = out_dir / "sweep.json"
    atomic_write_bytes(
        out_path, json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
    )

    log.info("=" * 70)
    log.info("lowest fertility at vocab=%d -> %s", best["vocab_size"], out_path)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from common.tokenizer.plots import plot_sweep, plot_token_frequency

    sweeps: dict[str, list[dict]] = {}
    best_metrics: dict[str, dict] = {}
    for entry in args.sweep:
        label, _, path = entry.partition("=")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sweeps[label] = data["sweep"]
        best_metrics[label] = min(data["sweep"], key=lambda r: r["fertility"])

    written = plot_sweep(sweeps, args.out_dir)
    written.append(plot_token_frequency(best_metrics, args.out_dir))
    for path in written:
        log.info("wrote %s", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokenizer training and evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="train several vocab sizes and measure each")
    sweep.add_argument("--data-root", required=True, type=Path)
    sweep.add_argument("--lang", required=True)
    sweep.add_argument("--out-dir", required=True, type=Path)
    sweep.add_argument(
        "--vocab-sizes",
        type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_VOCAB_SIZES,
    )
    sweep.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    sweep.add_argument("--character-coverage", type=float, default=0.9995)
    # Unset by default: export the whole training split and let SentencePiece
    # subsample it uniformly via input_sentence_size. A value here caps the
    # export by sampling (see export_lines), which is for quick test runs.
    sweep.add_argument("--max-lines", type=int, default=None)
    sweep.add_argument("--eval-docs", type=int, default=2000)
    sweep.set_defaults(func=cmd_sweep)

    plot = sub.add_parser("plot", help="draw comparison figures from sweep.json files")
    plot.add_argument("--sweep", nargs="+", required=True, metavar="LABEL=PATH")
    plot.add_argument("--out-dir", required=True, type=Path)
    plot.set_defaults(func=cmd_plot)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
