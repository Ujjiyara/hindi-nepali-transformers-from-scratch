"""Ingest AI4Bharat's Sangraha corpus into the project's document schema.

Sangraha (CC-BY-4.0) is a 251B-token Indic corpus. What makes it worth adding
alongside CC-100 is the *verified* subset: scraped from human-verified sites,
plus text OCR'd out of Indic-language PDFs and transcribed from video and
podcasts. CC-100 is raw Common Crawl, so those last two are registers the
corpus otherwise has none of.

Only ``verified`` and ``unverified`` are used here. The ``synthetic`` subset is
deliberately excluded: it is English Wikipedia machine-translated into Indic
languages, plus romanised transliterations of that. Training a Nepali model on
translationese would undermine the thing being measured, and the romanised half
would fail the Devanagari filter anyway. It is also the bulk of Sangraha's
Nepali (10.6B of 12.9B tokens), so taking the subsets on token count alone
would have got this badly wrong.

Data is read shard by shard, and each shard is deleted once consumed unless
``--keep-parquet`` is passed, so peak disk stays near one shard rather than the
17GB the full Nepali verified split occupies::

    python -m common.downloaded.sangraha --lang ne --subset verified \\
        --out Nepali/data/interim/downloaded/sangraha_ne.jsonl.gz \\
        --target-chars 300000000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path

from common.scrape.io_utils import Stats, devanagari_ratio, utc_now

log = logging.getLogger("sangraha")

REPO_ID = "ai4bharat/sangraha"
MIN_DEVANAGARI_RATIO = 0.60

# Sangraha's own directory name per language, which is not the ISO code the
# rest of this project uses.
LANG_DIRS = {"ne": "nep", "hi": "hin"}


def shard_names(subset: str, lang_dir: str) -> list[str]:
    """Shard paths for one language, smallest first.

    Smallest first is deliberate: Nepali's verified split is one 8.6GB shard
    and 24 of roughly 340MB. Since only a few hundred million characters are
    ever needed, starting small means one ordinary shard satisfies the target
    and the outlier is never fetched.
    """
    from huggingface_hub import HfApi

    prefix = f"{subset}/{lang_dir}/"
    info = HfApi().repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    shards = [
        (sibling.size or 0, sibling.rfilename)
        for sibling in info.siblings
        if sibling.rfilename.startswith(prefix) and sibling.rfilename.endswith(".parquet")
    ]
    if not shards:
        raise SystemExit(f"no parquet shards under {prefix} in {REPO_ID}")
    return [name for _, name in sorted(shards)]


def iter_documents(subset: str, lang_dir: str, cache_dir: Path, keep: bool):
    """Yield (text, type) from Sangraha shards, deleting each once consumed."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for name in shard_names(subset, lang_dir):
        log.info("fetching %s", name)
        path = Path(
            hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=str(cache_dir))
        )
        try:
            parquet = pq.ParquetFile(path)
            columns = set(parquet.schema_arrow.names)
            has_type = "type" in columns
            wanted = ["text", "type"] if has_type else ["text"]
            for batch in parquet.iter_batches(batch_size=10_000, columns=wanted):
                texts = batch.column("text").to_pylist()
                kinds = batch.column("type").to_pylist() if has_type else [""] * len(texts)
                for text, kind in zip(texts, kinds):
                    if text:
                        yield text, (kind or "")
        finally:
            if not keep:
                path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Sangraha into the project schema.")
    parser.add_argument("--lang", required=True, help="project language code, e.g. ne")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subset", default="verified", choices=["verified", "unverified"])
    parser.add_argument("--target-chars", type=int, required=True)
    parser.add_argument("--shard-chars", type=int, default=50_000_000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--source", default=None, help="source_name (default sangraha_<lang>)")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--keep-parquet", action="store_true",
                        help="do not delete each parquet shard after reading it")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    lang_dir = LANG_DIRS.get(args.lang)
    if lang_dir is None:
        log.error("no Sangraha directory known for language %r", args.lang)
        return 1

    source = args.source or f"sangraha_{args.lang}"
    cache_dir = args.cache_dir or args.out.parent.parent.parent / "raw" / "sangraha"
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    stats = Stats("downloaded", source)
    stem = args.out.name.replace(".jsonl.gz", "")
    shard_index = kept = chars = shard_chars = 0

    def open_shard(index: int):
        path = args.out.parent / f"{stem}.part{index:04d}.jsonl.gz"
        return path, gzip.open(path, "wt", encoding="utf-8")

    shard_path, handle = open_shard(shard_index)
    try:
        for text, kind in iter_documents(args.subset, lang_dir, cache_dir, args.keep_parquet):
            stats.bump("docs_seen")
            text = text.strip()
            if len(text) < args.min_chars:
                stats.bump("dropped_too_short")
                continue
            if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
                stats.bump("dropped_wrong_script")
                continue

            identifier = hashlib.sha256(
                f"{source}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}".encode()
            ).hexdigest()
            handle.write(json.dumps({
                "id": identifier,
                "text": text,
                "source_name": source,
                "source_url": f"https://huggingface.co/datasets/{REPO_ID}",
                # NOT manual: a pre-assembled public corpus, like CC-100.
                "collection_method": "downloaded",
                "lang": args.lang,
                "fetched_at": utc_now(),
                "meta": {"n_chars": len(text), "sangraha_subset": args.subset,
                         "sangraha_type": kind},
            }, ensure_ascii=False) + "\n")

            kept += 1
            chars += len(text)
            shard_chars += len(text)
            stats.bump("kept")
            stats.bump(f"type_{kind or 'unknown'}")

            if shard_chars >= args.shard_chars:
                handle.close()
                log.info("sealed %s (%s chars so far)", shard_path.name, f"{chars:,}")
                shard_index += 1
                shard_chars = 0
                shard_path, handle = open_shard(shard_index)

            if chars >= args.target_chars:
                log.info("reached target of %s chars", f"{args.target_chars:,}")
                break

            if kept % 50_000 == 0:
                log.info("%s docs, %sM chars, %.1f%% of target",
                         f"{kept:,}", f"{chars // 1_000_000}", 100 * chars / args.target_chars)
    finally:
        handle.close()

    stats.counts["chars_kept"] = chars
    stats.save(args.out.parent / f"_stats_downloaded_{source}.json")
    log.info("%s: %s docs, %s chars -> %s.part*.jsonl.gz",
             source, f"{kept:,}", f"{chars:,}", stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
