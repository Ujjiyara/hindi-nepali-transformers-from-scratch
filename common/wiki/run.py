"""Extract a Wikimedia dump into the project's document schema.

Run from the repository root::

    python -m common.wiki.run \
        --dump Nepali/data/raw/wikipedia/newiki-latest-pages-articles.xml.bz2 \
        --out  Nepali/data/interim/wikipedia/newiki.jsonl.gz \
        --lang ne --site ne.wikipedia.org --source newiki --workers 4

``--limit`` processes only the first N articles, for a quick sanity check
before committing to a full pass.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
from pathlib import Path

from common.scrape.io_utils import Stats, devanagari_ratio, doc_id, utc_now
from common.wiki.dump import article_url, iter_stripped

log = logging.getLogger("wiki")

# Articles shorter than this after stripping are stubs or disambiguation
# pages: a title, an infobox and a sentence. They add little signal and a lot
# of near-duplicate boilerplate.
MIN_ARTICLE_CHARS = 300
MIN_DEVANAGARI_RATIO = 0.60

# Wikimedia project implied by the tail of a source name such as "hiwikibooks".
# Order matters: the longer names must be tested before the "wiki" that is a
# prefix of them all.
_PROJECT_SUFFIXES = (
    "wikisource",
    "wikibooks",
    "wikiquote",
    "wikiversity",
    "wikinews",
    "wiktionary",
    "wikivoyage",
    "wiki",
)
# "hiwiki" is Wikipedia; every other project names itself in its host.
_PROJECT_HOSTS = {"wiki": "wikipedia"}


def check_site_matches_source(source: str, site: str) -> str | None:
    """Return a complaint if --site looks wrong for --source, else None.

    ``--site`` only ever appears inside the ``source_url`` recorded on each
    document, so getting it wrong is invisible: the run succeeds, the counts
    are right, and every document quietly claims to come from a wiki it did
    not come from. Three Hindi dumps were extracted with hi.wikisource.org
    before this check existed, which also made documents with the same title
    in different wikis collide on document id, since the id is a hash of that
    URL.
    """
    for suffix in _PROJECT_SUFFIXES:
        if source.endswith(suffix):
            expected = _PROJECT_HOSTS.get(suffix, suffix)
            if expected not in site:
                return (
                    f"--source {source!r} is a {expected} dump but --site is "
                    f"{site!r}, which is not a {expected} host. The site is "
                    f"recorded in every document's source_url, so this would "
                    f"misattribute the whole dump. Pass --allow-site-mismatch "
                    f"if this really is intended."
                )
            return None
    return None


def main(argv: list[str] | None = None) -> int:
    """Extract one Wikimedia dump into the project's document schema."""
    parser = argparse.ArgumentParser(description="Extract a Wikimedia XML dump to JSONL.")
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lang", required=True, help="ISO code recorded on each document")
    parser.add_argument("--site", required=True, help="e.g. ne.wikipedia.org")
    parser.add_argument("--source", required=True, help="source_name, e.g. newiki")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=MIN_ARTICLE_CHARS)
    parser.add_argument(
        "--namespaces",
        default="0",
        help="comma-separated dump namespaces. Default '0' (articles). "
        "Wikisource needs '250': its ns 0 holds only transclusion shells.",
    )
    parser.add_argument(
        "--allow-site-mismatch",
        action="store_true",
        help="skip the check that --site matches the project named by --source",
    )
    args = parser.parse_args(argv)
    namespaces = frozenset(n.strip() for n in args.namespaces.split(",") if n.strip())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.allow_site_mismatch:
        complaint = check_site_matches_source(args.source, args.site)
        if complaint:
            log.error("%s", complaint)
            return 1

    stats = Stats("wikipedia", args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out.with_suffix(args.out.suffix + ".tmp")

    started = time.time()
    kept = 0
    total_chars = 0

    with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
        for page_id, title, text in iter_stripped(
            args.dump, workers=args.workers, namespaces=namespaces
        ):
            stats.bump("pages_in")

            if stats.counts["pages_in"] % 20000 == 0:
                rate = stats.counts["pages_in"] / max(time.time() - started, 1e-6)
                log.info(
                    "%d pages read, %d kept, %.0f pages/s",
                    stats.counts["pages_in"], kept, rate,
                )

            if len(text) < args.min_chars:
                stats.bump("dropped_too_short")
            elif devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
                stats.bump("dropped_wrong_script")
            else:
                url = article_url(args.site, title)
                record = {
                    "id": doc_id(url),
                    "text": text,
                    "source_name": args.source,
                    "source_url": url,
                    # Parsed and cleaned by us from a raw dump, so this counts
                    # toward the >=20% manual requirement.
                    "collection_method": "manual",
                    "lang": args.lang,
                    "fetched_at": utc_now(),
                    "meta": {
                        "wiki_page_id": page_id,
                        "title": title,
                        "n_chars": len(text),
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                total_chars += len(text)
                stats.bump("kept")

            if args.limit is not None and stats.counts["pages_in"] >= args.limit:
                break

    os.replace(tmp_path, args.out)

    stats.bump("chars_kept", total_chars)
    stats.save(args.out.parent / f"_stats_wikipedia_{args.source}.json")

    elapsed = time.time() - started
    log.info(stats.summary())
    log.info(
        "%s: %d articles, %s chars (~%s tokens at 3.5 chars/token) in %.1f min -> %s",
        args.source, kept, f"{total_chars:,}",
        f"{int(total_chars / 3.5):,}", elapsed / 60, args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
