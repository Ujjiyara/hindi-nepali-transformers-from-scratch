"""Stage 2 -- fetching.

Downloads the URLs found by the discover stage and stores the **raw response
bytes**, unmodified, under ``raw/scrape/<site>/``. Raw bytes are treated as
immutable: extraction is re-runnable from them, so a bug in the extractor never
costs another overnight scrape.

Concurrency model: one worker thread per *site*, sites in parallel. Each thread
owns its site and sleeps politely between its own requests, so several sites
are fetched at once while every individual site still sees a well-behaved
single client. No shared rate limiter and no locks are needed.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from common.scrape.config import ScrapeConfig, SiteConfig
from common.scrape.io_utils import (
    JsonlAppender,
    SeenSet,
    Stats,
    doc_id,
    read_jsonl,
    utc_now,
    write_gzip_bytes,
)
from common.scrape.robots import RobotsPolicy

log = logging.getLogger(__name__)

# Status codes worth trying again: transient server-side or rate-limit issues.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Permanent failures: recorded once and never retried.
PERMANENT_STATUS = {400, 401, 403, 404, 410, 451}


class Fetcher:
    """Downloads one site's URLs, politely and resumably."""

    def __init__(self, cfg: ScrapeConfig, site: SiteConfig, robots: RobotsPolicy) -> None:
        self.cfg = cfg
        self.site = site
        self.robots = robots
        self.stats = Stats("fetch", site.name)

        # A Session reuses the underlying TCP connection across requests, which
        # is several times faster than a fresh connection per URL.
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent

        self.site_dir = cfg.raw_dir / site.name
        # robots.txt Crawl-delay wins when it is stricter than our own setting.
        declared = self.robots.crawl_delay(site.base_url)
        self.delay = max(site.delay_seconds, declared or 0.0)
        self._last_request_at = 0.0

    def _wait_turn(self) -> None:
        """Space requests ``delay`` seconds apart, measured start-to-start.

        Sleeping the full delay *after* each response adds the server's own
        response time on top of the interval, so a 1.5s delay against a host
        that answers in 1.4s becomes a request every 2.9s -- roughly half the
        collection rate, for no gain in politeness. A Crawl-delay is a minimum
        interval between requests, so waiting only the remainder honours it
        exactly while nearly doubling throughput.
        """
        remaining = self.delay - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _raw_path(self, identifier: str) -> Path:
        """Shard raw files by id prefix so no single directory holds 100k files."""
        return self.site_dir / identifier[:2] / f"{identifier}.html.gz"

    def _get(self, url: str) -> requests.Response | None:
        """GET with retry and exponential backoff. Returns None on failure."""
        for attempt in range(1, self.cfg.max_retries + 1):
            # Every actual request is spaced, retries included -- a retry is
            # another hit on a host that has just told us it is struggling.
            self._wait_turn()
            try:
                response = self.session.get(
                    url, timeout=self.cfg.request_timeout, allow_redirects=True
                )
            except requests.RequestException as exc:
                self.stats.bump("network_error")
                log.debug("network error %s (attempt %d): %s", url, attempt, exc)
            else:
                status = response.status_code
                if status == 200:
                    return response
                if status in PERMANENT_STATUS:
                    self.stats.bump(f"http_{status}")
                    return None
                if status not in RETRYABLE_STATUS:
                    self.stats.bump(f"http_{status}")
                    return None

                self.stats.bump(f"retry_http_{status}")
                # Honour Retry-After when the server sends it.
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(float(retry_after), 60.0))
                    continue

            # Exponential backoff with jitter, so retries do not synchronise.
            if attempt < self.cfg.max_retries:
                time.sleep((2 ** (attempt - 1)) + random.uniform(0, 0.5))

        self.stats.bump("give_up")
        return None

    def run(self, limit: int | None = None) -> int:
        """Fetch this site's URLs. Returns the number of new pages saved."""
        urls_path = self.cfg.urls_dir / f"{self.site.name}.jsonl"
        if not urls_path.exists():
            log.warning("[%s] no URL file at %s -- run discover first", self.site.name, urls_path)
            return 0

        manifest_path = self.site_dir / "_manifest.jsonl"
        seen_path = self.site_dir / "_seen_urls.txt"
        saved = 0

        with SeenSet(seen_path) as seen, JsonlAppender(manifest_path) as manifest:
            for record in read_jsonl(urls_path):
                if limit is not None and saved >= limit:
                    break

                url = record["url"]
                self.stats.bump("considered")

                if url in seen:
                    self.stats.bump("already_fetched")
                    continue
                if not self.robots.can_fetch(url):
                    self.stats.bump("blocked_by_robots")
                    seen.add(url)  # never reconsider
                    continue

                response = self._get(url)

                if response is None:
                    self.stats.bump("failed")
                    seen.add(url)
                    continue

                identifier = doc_id(url)
                # response.content is raw *bytes*. Deliberately not
                # response.text: requests guesses the encoding from headers,
                # and Devanagari pages frequently declare it wrongly or not at
                # all, producing mojibake. Encoding detection is left to the
                # extractor, which sees the HTML meta tags too.
                write_gzip_bytes(self._raw_path(identifier), response.content)

                manifest.write(
                    {
                        "id": identifier,
                        "url": url,
                        "final_url": response.url,
                        "source_name": self.site.name,
                        "lang": self.cfg.lang,
                        "status": response.status_code,
                        "content_type": response.headers.get("Content-Type", ""),
                        "bytes": len(response.content),
                        "fetched_at": utc_now(),
                    }
                )
                seen.add(url)
                saved += 1
                self.stats.bump("saved")

                if saved % 100 == 0:
                    log.info("[%s] saved %d pages", self.site.name, saved)

        self.stats.save(self.site_dir / "_stats_fetch.json")
        log.info(self.stats.summary())
        return saved


def fetch_all(cfg: ScrapeConfig, sites: list[SiteConfig], limit: int | None = None) -> int:
    """Fetch several sites concurrently: one thread per site."""
    robots = RobotsPolicy(cfg.user_agent, timeout=cfg.request_timeout)

    def work(site: SiteConfig) -> int:
        return Fetcher(cfg, site, robots).run(limit=limit)

    if len(sites) == 1:
        return work(sites[0])

    with ThreadPoolExecutor(max_workers=len(sites)) as pool:
        return sum(pool.map(work, sites))
