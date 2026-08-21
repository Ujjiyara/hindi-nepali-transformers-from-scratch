"""Exact token counts for a built corpus, using that language's chosen tokenizer.

The pipeline reports token counts as ``characters / 3.5``, because it runs
before any tokenizer exists -- the tokenizer is trained on the splits the
pipeline writes, so at that point there is nothing better to divide by. That
placeholder is not close enough to publish: the trained tokenizers measure
3.96 characters per token for Hindi and 4.71 for Nepali, so the estimate
overstates Hindi by 13% and Nepali by 35%.

The error is worst for the lower-resource language, which is exactly where it
does the most damage -- the token count is what decides how many epochs Model L
needs in Phase 2, so an inflated figure would quietly make the training budget
look healthier than it is.

Once the tokenizer exists the real count is just a matter of encoding the
corpus, which this does, one split at a time and in parallel over documents::

    python -m common.tokenizer.count --data-root Nepali/data --lang ne

Results are written to ``processed/_token_counts.json`` and the report prefers
them over the pipeline's estimate whenever they are present.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import sentencepiece as spm

from common.scrape.io_utils import atomic_write_bytes, read_jsonl, utc_now

log = logging.getLogger("count")

# Documents per task handed to a worker. Large enough that the cost of pickling
# a batch is small against the cost of encoding it.
BATCH_DOCS = 500

_MODEL: spm.SentencePieceProcessor | None = None


def _init_worker(model_path: str) -> None:
    """Load the tokenizer once per worker rather than once per batch."""
    global _MODEL
    _MODEL = spm.SentencePieceProcessor()
    _MODEL.load(model_path)


def _count_batch(batch: list[tuple[str, str]]) -> dict[str, tuple[int, int, int]]:
    """Per collection method, (tokens, characters, documents) for one batch.

    Broken down by ``collection_method`` because the project's floor is stated
    in *tokens* -- at least 20% of training tokens manually collected -- and a
    character share is not a safe proxy for it. Manual and downloaded text
    tokenise at different rates, so the two fractions differ, and the one that
    has to be met is the token one.
    """
    assert _MODEL is not None
    out: dict[str, tuple[int, int, int]] = {}
    texts = [text for text, _ in batch]
    pieces = _MODEL.encode(texts)
    for (text, method), piece in zip(batch, pieces):
        tokens, chars, docs = out.get(method, (0, 0, 0))
        out[method] = (tokens + len(piece), chars + len(text), docs + 1)
    return out


def _batches(path: Path, size: int = BATCH_DOCS):
    batch: list[tuple[str, str]] = []
    for record in read_jsonl(path):
        batch.append((record.get("text", ""), record.get("collection_method", "unknown")))
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_split(path: Path, model_path: Path, workers: int) -> dict:
    """Encode one split and return exact totals, split by collection method."""
    by_method: dict[str, list[int]] = {}
    tokens = chars = docs = 0
    started = time.time()
    with mp.Pool(workers, initializer=_init_worker, initargs=(str(model_path),)) as pool:
        for result in pool.imap_unordered(_count_batch, _batches(path), chunksize=1):
            for method, (t, c, d) in result.items():
                row = by_method.setdefault(method, [0, 0, 0])
                row[0] += t
                row[1] += c
                row[2] += d
                tokens += t
                chars += c
                docs += d
            if docs % 100_000 < BATCH_DOCS:
                log.info("  %s: %s docs, %s tokens (%.1f min)",
                         path.name, f"{docs:,}", f"{tokens:,}", (time.time() - started) / 60)

    manual = by_method.get("manual", [0, 0, 0])
    return {
        "documents": docs,
        "characters": chars,
        "tokens": tokens,
        "manual_tokens": manual[0],
        "manual_characters": manual[1],
        "manual_token_fraction": round(manual[0] / tokens, 4) if tokens else 0.0,
        "manual_char_fraction": round(manual[1] / chars, 4) if chars else 0.0,
        "by_method": {m: {"tokens": v[0], "characters": v[1], "documents": v[2]}
                      for m, v in sorted(by_method.items())},
    }


def chosen_model(config_path: Path) -> tuple[Path, int]:
    """The tokenizer this language selected: (model path, vocab size)."""
    model_path = vocab_size = None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("model_path:"):
            model_path = Path(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("vocab_size:"):
            vocab_size = int(stripped.split(":", 1)[1].strip())
    if model_path is None or vocab_size is None:
        raise SystemExit(f"{config_path} must set both model_path and vocab_size")
    return model_path, vocab_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count corpus tokens exactly.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--config", type=Path, default=None,
                        help="tokenizer config (default <data-root>/../configs/tokenizer.yaml)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    config_path = args.config or args.data_root.parent / "configs" / "tokenizer.yaml"
    model_path, vocab_size = chosen_model(config_path)
    if not model_path.exists():
        log.error("tokenizer model %s not found -- run the sweep first", model_path)
        return 1

    splits = {}
    for name in ("train", "validation", "test"):
        path = args.data_root / "processed" / f"{name}.jsonl.gz"
        if not path.exists():
            log.error("missing %s -- run common.pipeline.run first", path)
            return 1
        splits[name] = count_split(path, model_path, args.workers)
        row = splits[name]
        log.info("%-11s %10s docs  %15s tokens  manual %.1f%% of tokens (%.1f%% of chars)",
                 name, f"{row['documents']:,}", f"{row['tokens']:,}",
                 100 * row["manual_token_fraction"], 100 * row["manual_char_fraction"])

    total_tokens = sum(r["tokens"] for r in splits.values())
    total_chars = sum(r["characters"] for r in splits.values())
    total_manual_tokens = sum(r["manual_tokens"] for r in splits.values())
    total_manual_chars = sum(r["manual_characters"] for r in splits.values())
    report = {
        "lang": args.lang,
        "generated_at": utc_now(),
        "tokenizer": {"model": str(model_path), "vocab_size": vocab_size},
        "splits": splits,
        "corpus": {
            "documents": sum(r["documents"] for r in splits.values()),
            "characters": total_chars,
            "tokens": total_tokens,
            "chars_per_token": round(total_chars / max(total_tokens, 1), 4),
            "manual_tokens": total_manual_tokens,
            "manual_token_fraction": round(total_manual_tokens / max(total_tokens, 1), 4),
            "manual_char_fraction": round(total_manual_chars / max(total_chars, 1), 4),
        },
    }
    out_path = args.data_root / "processed" / "_token_counts.json"
    atomic_write_bytes(out_path, json.dumps(report, indent=2).encode("utf-8"))

    log.info("=" * 72)
    log.info("CORPUS %s: %s tokens over %s characters (%.3f chars/token) with vocab %s",
             args.lang, f"{total_tokens:,}", f"{total_chars:,}",
             report["corpus"]["chars_per_token"], f"{vocab_size:,}")
    log.info("counts -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
