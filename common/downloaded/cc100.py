"""Streaming ingester for the CC-100 monolingual corpora.

CC-100 ships one language per ``.xz`` file, documents separated by blank lines.
The Hindi file is 2.4 GB compressed and roughly 15 GB expanded -- more than the
free disk on this machine and far more text than the corpus target needs.

So the file is never stored. It is streamed from the URL, decompressed
incrementally, and cut off as soon as the character target is met; the only
thing written is the gzipped JSONL of documents actually kept. That keeps peak
disk to the output size and makes the ingest finish in minutes instead of
downloading 2.4 GB to discard most of it.

Usage::

    python -m common.downloaded.cc100 \
        --url https://data.statmt.org/cc-100/ne.txt.xz \
        --out Nepali/data/interim/downloaded/cc100_ne.jsonl.gz \
        --lang ne --target-chars 400000000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import lzma
import os
import time
from pathlib import Path

import requests

from common.scrape.io_utils import Stats, devanagari_ratio, utc_now

log = logging.getLogger("cc100")

CHUNK_BYTES = 1 << 20  # 1 MiB of compressed data per read
MIN_DEVANAGARI_RATIO = 0.60


def stream_documents(url: str, user_agent: str, min_chars: int):
    """Yield documents from a remote ``.xz`` corpus without storing the archive.

    CC-100 separates documents with a blank line. Decompression is incremental,
    so memory stays flat regardless of archive size, and the generator can be
    abandoned early -- closing the response mid-stream stops the transfer.
    """
    decompressor = lzma.LZMADecompressor()
    pending = ""
    buffer: list[str] = []

    with requests.get(
        url, stream=True, timeout=(10, 120), headers={"User-Agent": user_agent}
    ) as response:
        response.raise_for_status()
        for compressed in response.iter_content(chunk_size=CHUNK_BYTES):
            if not compressed:
                continue
            raw = decompressor.decompress(compressed)
            if not raw:
                continue

            pending += raw.decode("utf-8", errors="replace")
            *lines, pending = pending.split("\n")

            for line in lines:
                line = line.strip()
                if line:
                    buffer.append(line)
                elif buffer:
                    text = "\n".join(buffer)
                    buffer = []
                    if len(text) >= min_chars:
                        yield text

    if buffer:
        text = "\n".join(buffer)
        if len(text) >= min_chars:
            yield text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stream-ingest a CC-100 corpus.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--source", default=None, help="source_name (default cc100_<lang>)")
    parser.add_argument(
        "--target-chars",
        type=int,
        required=True,
        help="stop once this many characters have been kept",
    )
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument(
        "--user-agent",
        default="IIIT-LMA-corpus-research/1.0 (sakshamgoyal.sgm27@gmail.com)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    source = args.source or f"cc100_{args.lang}"
    stats = Stats("downloaded", source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out.with_suffix(args.out.suffix + ".tmp")

    kept = 0
    chars = 0
    started = time.time()

    with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
        for text in stream_documents(args.url, args.user_agent, args.min_chars):
            stats.bump("docs_seen")

            if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
                stats.bump("dropped_wrong_script")
                continue

            identifier = hashlib.sha256(
                f"{source}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}".encode()
            ).hexdigest()
            handle.write(
                json.dumps(
                    {
                        "id": identifier,
                        "text": text,
                        "source_name": source,
                        "source_url": args.url,
                        # NOT manual: this is a pre-assembled public corpus.
                        "collection_method": "downloaded",
                        "lang": args.lang,
                        "fetched_at": utc_now(),
                        "meta": {"n_chars": len(text)},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
            chars += len(text)
            stats.bump("kept")

            if kept % 50000 == 0:
                rate = chars / max(time.time() - started, 1e-6) / 1e6
                log.info(
                    "%s docs, %.0fM chars (%.1f MB/s of text), %.0f%% of target",
                    f"{kept:,}", chars / 1e6, rate, 100 * chars / args.target_chars,
                )

            if chars >= args.target_chars:
                log.info("reached target of %s chars", f"{args.target_chars:,}")
                break

    os.replace(tmp_path, args.out)
    stats.bump("chars_kept", chars)
    stats.save(args.out.parent / f"_stats_downloaded_{source}.json")

    log.info(
        "%s: %s docs, %s chars (~%s tokens) in %.1f min -> %s",
        source, f"{kept:,}", f"{chars:,}", f"{int(chars / 3.5):,}",
        (time.time() - started) / 60, args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
