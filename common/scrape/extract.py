"""Stage 3 -- HTML to clean text.

This stage decides corpus quality. A naive ``soup.get_text()`` keeps navigation
menus, "related articles" blocks, cookie banners and comment sections. That
boilerplate is highly repetitive, so it survives deduplication and ends up
being memorised by the language model.

We therefore use ``trafilatura``, which is purpose-built for article-body
extraction, with a BeautifulSoup fallback. Both are general-purpose libraries,
not pretrained models or corpora, so they are within the project's constraints.

Output records use the project-wide document schema::

    {id, text, source_name, source_url, collection_method, lang, fetched_at, meta}

``collection_method`` is always ``"manual"`` here: these pages were selected,
fetched and cleaned by us, as opposed to a pre-assembled public corpus. That
field is what produces the manual-vs-downloaded token split the report needs,
so it must never be dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path

from common.scrape.config import ScrapeConfig, SiteConfig
from common.scrape.io_utils import (
    Stats,
    devanagari_ratio,
    read_gzip_bytes,
    read_jsonl,
    write_jsonl,
)

log = logging.getLogger(__name__)

# Below this fraction of Devanagari letters a page is not usable Hindi/Nepali
# prose. Catches error pages, English-only pages, and -- importantly for Nepali
# -- legacy Preeti/Kantipur font text, which is ASCII bytes that merely *render*
# as Devanagari and would otherwise enter the corpus as Latin gibberish.
MIN_DEVANAGARI_RATIO = 0.60

try:
    import trafilatura
except ImportError:  # pragma: no cover - dependency check
    trafilatura = None


def _extract_with_trafilatura(html: bytes) -> str | None:
    """Article-body extraction with boilerplate removal."""
    if trafilatura is None:
        return None
    return trafilatura.extract(
        html,
        include_comments=False,   # comment sections are not article prose
        include_tables=False,     # tables tokenise badly and add little text
        favor_precision=True,     # prefer dropping doubtful blocks over admitting junk
        no_fallback=False,
    )


def _extract_with_soup(html: bytes) -> str | None:
    """Fallback: strip scripts/styles/nav and take the remaining text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


class Extractor:
    """Turns one site's raw HTML into clean text records."""

    def __init__(self, cfg: ScrapeConfig, site: SiteConfig) -> None:
        self.cfg = cfg
        self.site = site
        self.stats = Stats("extract", site.name)
        self.site_dir = cfg.raw_dir / site.name

    def _raw_path(self, identifier: str) -> Path:
        return self.site_dir / identifier[:2] / f"{identifier}.html.gz"

    def _record(self, entry: dict, text: str) -> dict:
        """Build a document in the project schema."""
        return {
            "id": entry["id"],
            "text": text,
            "source_name": entry["source_name"],
            "source_url": entry.get("final_url") or entry["url"],
            "collection_method": "manual",
            "lang": entry.get("lang", self.cfg.lang),
            "fetched_at": entry.get("fetched_at"),
            "meta": {
                "n_chars": len(text),
                "devanagari_ratio": round(devanagari_ratio(text), 4),
            },
        }

    def run(self) -> int:
        """Extract every fetched page for this site. Returns records written."""
        manifest_path = self.site_dir / "_manifest.jsonl"
        if not manifest_path.exists():
            log.warning("[%s] no manifest at %s -- run fetch first", self.site.name, manifest_path)
            return 0

        records = []
        # Tolerant: extraction is deliberately re-runnable against a fetch that
        # is still in progress, and the fetcher may be midway through writing
        # the manifest's last line.
        for entry in read_jsonl(manifest_path, skip_bad_lines=True):
            self.stats.bump("pages_in")
            path = self._raw_path(entry["id"])
            if not path.exists():
                self.stats.bump("raw_missing")
                continue

            html = read_gzip_bytes(path)
            text = _extract_with_trafilatura(html)
            if not text:
                text = _extract_with_soup(html)
                if text:
                    self.stats.bump("used_soup_fallback")

            if not text:
                self.stats.bump("dropped_no_text")
                continue
            if len(text) < self.cfg.min_chars:
                self.stats.bump("dropped_too_short")
                continue

            ratio = devanagari_ratio(text)
            if ratio < MIN_DEVANAGARI_RATIO:
                # Logged separately because a whole source failing this way
                # usually means legacy-font encoding, not a scraper bug.
                self.stats.bump("dropped_wrong_script")
                continue

            records.append(self._record(entry, text))
            self.stats.bump("kept")
            self.stats.bump("chars_kept", len(text))

        out_path = self.cfg.interim_dir / f"{self.site.name}.jsonl.gz"
        written = write_jsonl(out_path, records)
        self.stats.save(self.cfg.interim_dir / f"_stats_extract_{self.site.name}.json")

        log.info("%s -> %s", self.stats.summary(), out_path)
        return written


def extract_site(cfg: ScrapeConfig, site: SiteConfig) -> int:
    """Convenience wrapper around :class:`Extractor`."""
    return Extractor(cfg, site).run()
