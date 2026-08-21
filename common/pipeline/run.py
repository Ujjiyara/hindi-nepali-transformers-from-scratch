"""Build a language's final corpus: normalise -> dedup -> split.

Reads every ``*.jsonl.gz`` under the language's ``interim/`` tree -- Wikipedia
dumps, scraped sites and downloaded corpora alike, since all three emit the same
schema -- and writes ``processed/{train,validation,test}.jsonl.gz`` plus a
statistics file.

**Why this is a two-pass design.** The obvious implementation loads every
document into a list and deduplicates that. The Hindi corpus with CC-100 is
several gigabytes of text, so that list does not fit in memory. Instead:

  pass 1  stream the interim files, normalise, quality-filter, write the
          survivors to a temporary file, and keep only a fingerprint per
          document (a hash, a 128-value MinHash signature, its length, and
          whether it was manually collected) -- a few hundred bytes each
          instead of a few kilobytes.
  dedup   run entirely on those fingerprints.
  pass 2  stream the temporary file again and write out only the survivors.

Peak memory is therefore set by the number of documents, not by the size of the
corpus.

Run from the repository root::

    python -m common.pipeline.run --data-root Nepali/data --lang ne
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from common.pipeline.dedup import (
    DocFeatures,
    MinHasher,
    deduplicate_features,
    shingles,
    text_hash,
)
from common.pipeline.normalize import NormalizeConfig, normalize_text, quality_verdict
from common.pipeline.split import SplitConfig, assign_split
from common.scrape.io_utils import atomic_write_bytes, read_jsonl, utc_now

log = logging.getLogger("pipeline")

# Rough characters-per-token for Devanagari, used only for progress reporting.
# The authoritative token counts come from the trained tokenizer.
CHARS_PER_TOKEN_ESTIMATE = 3.5


def interim_files(data_root: Path, exclude: list[str] | None = None) -> list[Path]:
    """All interim shards, minus any whose path contains an excluded substring."""
    paths = sorted((data_root / "interim").rglob("*.jsonl.gz"))
    if exclude:
        paths = [p for p in paths if not any(token in str(p) for token in exclude)]
    return paths


def read_jsonl_tolerant(path: Path):
    """Stream a shard, surviving a truncated tail.

    An ingester may still be writing its newest shard when a pipeline run
    starts. A half-written gzip member raises partway through; the documents
    read up to that point are valid, so the error is logged and the file is
    abandoned rather than failing the whole run.
    """
    try:
        yield from read_jsonl(path)
    except (EOFError, OSError, json.JSONDecodeError) as exc:
        log.warning("%s: truncated or unreadable (%s) -- using what was read",
                    path.name, type(exc).__name__)


def main(argv: list[str] | None = None) -> int:
    """Build one language's corpus: normalise, deduplicate, split, report."""
    parser = argparse.ArgumentParser(description="Normalise, dedup and split a corpus.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--no-dedup", action="store_true", help="debugging only")
    parser.add_argument(
        "--min-manual-fraction",
        type=float,
        default=0.20,
        help="fail if manually collected text is a smaller share of the "
             "corpus than this (by characters, measured after dedup)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="skip interim files whose path contains any of these substrings",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    norm_cfg = NormalizeConfig(min_chars=args.min_chars)
    split_cfg = SplitConfig(
        val_fraction=args.val_fraction, test_fraction=args.test_fraction
    )
    boilerplate = norm_cfg.compiled_boilerplate()
    hasher = MinHasher()

    stats: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)

    out_dir = args.data_root / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.data_root / "interim" / "_normalised.tmp.jsonl.gz"

    sources = interim_files(args.data_root, args.exclude)
    if not sources:
        log.error("no interim files under %s", args.data_root / "interim")
        return 1
    for path in sources:
        log.info("input: %s", path.relative_to(args.data_root))

    # --- pass 1: normalise, filter, fingerprint ---------------------------
    started = time.time()
    features: list[DocFeatures] = []

    with gzip.open(scratch, "wt", encoding="utf-8") as sink:
        for path in sources:
            for record in read_jsonl_tolerant(path):
                source = record.get("source_name", "unknown")
                stats["docs_loaded"] += 1
                per_source[source]["in"] += 1

                text = normalize_text(record.get("text", ""), norm_cfg, boilerplate)
                reason = quality_verdict(text, norm_cfg)
                if reason is not None:
                    stats[f"dropped_{reason}"] += 1
                    per_source[source][f"dropped_{reason}"] += 1
                    continue

                record["text"] = text
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")

                features.append(
                    DocFeatures(
                        exact_hash=text_hash(text),
                        signature=hasher.signature(shingles(text)),
                        is_manual=record.get("collection_method") == "manual",
                        n_chars=len(text),
                    )
                )
                stats["docs_normalised"] += 1

                if stats["docs_normalised"] % 50000 == 0:
                    log.info(
                        "  normalised %s docs (%.1f min)",
                        f"{stats['docs_normalised']:,}", (time.time() - started) / 60,
                    )

    log.info(
        "pass 1 done: %s docs kept of %s (%.1f min)",
        f"{stats['docs_normalised']:,}", f"{stats['docs_loaded']:,}",
        (time.time() - started) / 60,
    )

    # --- dedup on fingerprints only ---------------------------------------
    if args.no_dedup:
        keep = set(range(len(features)))
    else:
        result = deduplicate_features(features)
        stats["dropped_exact_duplicate"] = result.exact_dropped
        stats["dropped_near_duplicate"] = result.near_dropped
        stats["near_duplicate_clusters"] = result.clusters
        keep = set(result.keep_indices)
    del features

    stats["docs_after_dedup"] = len(keep)

    # --- pass 2: write the splits -----------------------------------------
    handles = {
        name: gzip.open(out_dir / f"{name}.jsonl.gz", "wt", encoding="utf-8")
        for name in ("train", "validation", "test")
    }
    summary = {
        name: {"documents": 0, "characters": 0, "manual_documents": 0, "manual_characters": 0}
        for name in handles
    }

    try:
        for index, record in enumerate(read_jsonl(scratch)):
            if index not in keep:
                continue
            split = assign_split(record["id"], split_cfg)
            record["split"] = split
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")

            row = summary[split]
            row["documents"] += 1
            row["characters"] += len(record["text"])
            if record.get("collection_method") == "manual":
                row["manual_documents"] += 1
                row["manual_characters"] += len(record["text"])
    finally:
        for handle in handles.values():
            handle.close()
        if scratch.exists():
            os.remove(scratch)

    for name, row in summary.items():
        chars = row["characters"]
        row["est_tokens"] = int(chars / CHARS_PER_TOKEN_ESTIMATE)
        row["manual_char_fraction"] = round(row["manual_characters"] / chars, 4) if chars else 0.0
        log.info(
            "%-11s %8d docs  %15s chars  manual %.1f%%",
            name, row["documents"], f"{chars:,}", 100 * row["manual_char_fraction"],
        )

    # --- statistics report -------------------------------------------------
    total_chars = sum(r["characters"] for r in summary.values())
    manual_chars = sum(r["manual_characters"] for r in summary.values())
    report = {
        "lang": args.lang,
        "generated_at": utc_now(),
        "config": {"normalize": norm_cfg.__dict__, "split": split_cfg.__dict__},
        "totals": dict(sorted(stats.items())),
        "splits": summary,
        "corpus": {
            "documents": sum(r["documents"] for r in summary.values()),
            "characters": total_chars,
            "est_tokens": int(total_chars / CHARS_PER_TOKEN_ESTIMATE),
            "manual_characters": manual_chars,
            "manual_char_fraction": round(manual_chars / total_chars, 4) if total_chars else 0.0,
        },
        "per_source": {k: dict(sorted(v.items())) for k, v in sorted(per_source.items())},
    }
    stats_path = out_dir / "_stats_pipeline.json"
    atomic_write_bytes(
        stats_path, json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
    )

    log.info("=" * 72)
    log.info(
        "CORPUS %s: %s docs, %s chars (~%s tokens), manual %.1f%%  [%.1f min]",
        args.lang,
        f"{report['corpus']['documents']:,}",
        f"{total_chars:,}",
        f"{report['corpus']['est_tokens']:,}",
        100 * report["corpus"]["manual_char_fraction"],
        (time.time() - started) / 60,
    )
    log.info("stats -> %s", stats_path)

    # The manual share is a hard requirement on the corpus, and it is not
    # knowable until after deduplication: dedup decides which copy of a
    # document survives, so downloaded text can displace manual text and move
    # this number after every input is already fixed. Checking it here means a
    # corpus that breaks the rule can never be handed quietly to the tokenizer.
    # Outputs and statistics are still written, so the shortfall can be
    # measured and corrected rather than merely reported.
    manual_fraction = report["corpus"]["manual_char_fraction"]
    if manual_fraction < args.min_manual_fraction:
        deficit = args.min_manual_fraction * total_chars - manual_chars
        downloaded_budget = manual_chars / args.min_manual_fraction - manual_chars
        log.error(
            "manual share is %.1f%%, below the required %.1f%%. Either collect "
            "~%s more manual characters, or cut downloaded input to ~%s "
            "characters (currently %s) -- e.g. --exclude some downloaded shards.",
            100 * manual_fraction,
            100 * args.min_manual_fraction,
            f"{int(deficit):,}",
            f"{int(downloaded_budget):,}",
            f"{total_chars - manual_chars:,}",
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
