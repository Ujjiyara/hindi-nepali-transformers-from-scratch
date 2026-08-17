"""Corpus scraper.

Three independent stages, each reading and writing files so that any stage can
be re-run without repeating the previous one:

    discover : seed sites  -> urls/<site>.jsonl      (which pages exist)
    fetch    : urls        -> raw/<site>/*.html.gz   (bytes, never modified)
    extract  : raw         -> interim/<site>.jsonl   (clean text, project schema)

Fetching costs hours; extraction costs minutes. Keeping the raw bytes on disk
means a bug in the extractor is a two-minute re-run instead of an overnight
re-scrape.
"""

from common.scrape.config import ScrapeConfig, SiteConfig

__all__ = ["ScrapeConfig", "SiteConfig"]
