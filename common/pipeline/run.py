"""Build a language's final corpus: normalise -> dedup -> split.

Reads every ``*.jsonl.gz`` under the language's ``interim/`` tree (Wikipedia
dumps and scraped sites alike, since both emit the same schema) and writes
``processed/{train,validation,test}.jsonl.gz`` plus a statistics file.

Run from the repository root::

    python -m common.pipeline.run --data-root Nepali/data --lang ne
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from common.pipeline.dedup import deduplicate
from common.pipeline.normalize import NormalizeConfig, normalize_text, quality_verdict
from common.pipeline.split import SplitConfig, assign_split
from common.scrape.io_utils import atomic_write_bytes, read_jsonl, utc_now, write_jsonl

log = logging.getLogger("pipeline")

# Rough characters-per-token for Devanagari, used only for progress reporting.
# The authoritative token counts come from the trained tokenizer.
CHARS_PER_TOKEN_ESTIMATE = 3.5


def load_interim(data_root: Path) -> list[dict]:
    """Load every interim document for one language."""
    records: list[dict] = []
    for path in sorted((data_root / "interim").rglob("*.jsonl.gz")):
        count = 0
        for record in read_jsonl(path):
            records.append(record)
            count += 1
        log.info("loaded %-45s %8d docs", str(path.relative_to(data_root)), count)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalise, dedup and split a corpus.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--no-dedup", action="store_true", help="skip dedup (debugging only)")
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

    stats: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)

    # --- load + normalise -------------------------------------------------
    raw_records = load_interim(args.data_root)
    stats["docs_loaded"] = len(raw_records)
    log.info("loaded %d documents total", len(raw_records))

    kept: list[dict] = []
    for record in raw_records:
        source = record.get("source_name", "unknown")
        per_source[source]["in"] += 1

        text = normalize_text(record.get("text", ""), norm_cfg, boilerplate)
        reason = quality_verdict(text, norm_cfg)
        if reason is not None:
            stats[f"dropped_{reason}"] += 1
            per_source[source][f"dropped_{reason}"] += 1
            continue

        record["text"] = text
        kept.append(record)
        stats["docs_normalised"] += 1

    del raw_records
    log.info("after normalise/filter: %d documents", len(kept))

    # --- dedup ------------------------------------------------------------
    if args.no_dedup:
        deduped = kept
    else:
        result = deduplicate(kept)
        stats["dropped_exact_duplicate"] = result.exact_dropped
        stats["dropped_near_duplicate"] = result.near_dropped
        stats["near_duplicate_clusters"] = result.clusters
        deduped = [kept[i] for i in result.keep_indices]
    del kept

    stats["docs_after_dedup"] = len(deduped)
    log.info("after dedup: %d documents", len(deduped))

    # --- split ------------------------------------------------------------
    buckets: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for record in deduped:
        record["split"] = assign_split(record["id"], split_cfg)
        buckets[record["split"]].append(record)

    out_dir = args.data_root / "processed"
    summary: dict[str, dict] = {}
    for name, items in buckets.items():
        path = out_dir / f"{name}.jsonl.gz"
        write_jsonl(path, items)
        chars = sum(len(item["text"]) for item in items)
        manual = sum(1 for item in items if item.get("collection_method") == "manual")
        manual_chars = sum(
            len(item["text"]) for item in items if item.get("collection_method") == "manual"
        )
        summary[name] = {
            "documents": len(items),
            "characters": chars,
            "est_tokens": int(chars / CHARS_PER_TOKEN_ESTIMATE),
            "manual_documents": manual,
            "manual_characters": manual_chars,
            "manual_char_fraction": round(manual_chars / chars, 4) if chars else 0.0,
        }
        log.info(
            "%-11s %7d docs  %13s chars  manual %.1f%%",
            name, len(items), f"{chars:,}", 100 * summary[name]["manual_char_fraction"],
        )

    # --- statistics report -------------------------------------------------
    report = {
        "lang": args.lang,
        "generated_at": utc_now(),
        "config": {
            "normalize": norm_cfg.__dict__,
            "split": split_cfg.__dict__,
        },
        "totals": dict(sorted(stats.items())),
        "splits": summary,
        "per_source": {k: dict(sorted(v.items())) for k, v in sorted(per_source.items())},
    }
    stats_path = out_dir / "_stats_pipeline.json"
    atomic_write_bytes(
        stats_path, json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
    )

    total_chars = sum(s["characters"] for s in summary.values())
    manual_chars = sum(s["manual_characters"] for s in summary.values())
    log.info("=" * 70)
    log.info(
        "CORPUS %s: %d docs, %s chars (~%s tokens), manual %.1f%%",
        args.lang,
        sum(s["documents"] for s in summary.values()),
        f"{total_chars:,}",
        f"{int(total_chars / CHARS_PER_TOKEN_ESTIMATE):,}",
        100 * manual_chars / total_chars if total_chars else 0,
    )
    log.info("stats -> %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
