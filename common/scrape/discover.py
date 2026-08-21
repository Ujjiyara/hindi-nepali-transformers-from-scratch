"""Stage 1 -- URL discovery.

Finds which pages exist on a site without downloading article bodies. Three
strategies, tried in the order listed in the site config:

``sitemap``
    Read the site's XML sitemaps. Cheapest and most complete *when it exists* --
    no crawling, and it reaches deep archive pages a link-crawl would never
    find. Many Nepali news sites turn out not to publish article sitemaps at
    all, so this often yields nothing.

``rss``
    Parse the site's RSS/Atom feed. Always small (a few dozen recent items) but
    free and a reliable source of seed URLs.

``crawl``
    Breadth-first link following from seed pages, restricted to the same
    domain. Slowest and most fragile, but for a site without sitemaps it is the
    only way to reach volume.

Output: ``<data_root>/urls/<site>.jsonl``, one record per discovered page.
"""

from __future__ import annotations

import gzip
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse

import requests

from common.scrape.config import ScrapeConfig, SiteConfig
from common.scrape.io_utils import Stats, read_jsonl, utc_now, write_jsonl
from common.scrape.robots import RobotsPolicy

log = logging.getLogger(__name__)

# Sitemaps are namespaced XML. Forgetting the namespace makes every findall()
# return an empty list, which looks exactly like an empty sitemap -- the single
# most common way to lose an hour on this stage.
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

MAX_SITEMAP_DEPTH = 4

# Query parameters that identify a campaign rather than a document. Left in the
# URL they cause the same article to be fetched several times.
TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|ref|share)", re.IGNORECASE)

_HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#]+)["']""", re.IGNORECASE)

# Endings that are certainly not article HTML.
_ASSET_RE = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|pdf|mp3|mp4|zip|xml|rss)$", re.IGNORECASE
)


def _host(url: str) -> str:
    """Hostname with any leading ``www.`` removed, lowercased.

    The crawl restricts itself to one site by comparing hostnames, and many
    sites redirect between the bare domain and the ``www`` one. Comparing the
    raw netloc means a config written with the wrong variant matches nothing:
    the crawl fetches its seed, finds every link "off-site", and terminates
    after a handful of pages having discovered almost nothing -- a silent
    near-empty result rather than an error.
    """
    return urlparse(url).netloc.lower().removeprefix("www.")


def normalise_url(url: str) -> str:
    """Canonical form of a URL, so the same page is not fetched twice.

    Drops the fragment (never sent to the server anyway) and strips tracking
    query parameters, which vary per referrer but identify the same document.
    """
    url, _ = urldefrag(url.strip())
    parsed = urlparse(url)
    if parsed.query:
        kept = [
            part
            for part in parsed.query.split("&")
            if part and not TRACKING_PARAMS.match(part.split("=")[0])
        ]
        parsed = parsed._replace(query="&".join(kept))
    return parsed.geturl().rstrip("/")


def _looks_like_xml(data: bytes) -> bool:
    """True if ``data`` plausibly starts an XML document.

    Needed because several sites answer HTTP 200 with their homepage HTML for
    any unknown path (a "soft 404"). Without this check a missing sitemap is
    reported as malformed XML, which sends you debugging the wrong thing.
    """
    head = data[:200].lstrip()
    return head.startswith((b"<?xml", b"<urlset", b"<sitemapindex", b"<rss", b"<feed"))


class SiteDiscoverer:
    """Collects candidate article URLs for one site."""

    def __init__(self, cfg: ScrapeConfig, site: SiteConfig, robots: RobotsPolicy) -> None:
        self.cfg = cfg
        self.site = site
        self.robots = robots
        self.stats = Stats("discover", site.name)
        self.found: dict[str, None] = {}  # ordered set of accepted URLs

        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent

        self.domain = _host(site.base_url)
        # The politeness delay applies to EVERY request this class makes, not
        # just crawl requests. Sitemap indexes can fan out to hundreds of child
        # files; fetching those back-to-back looks exactly like an attack and
        # will get the scraper IP-blocked before it collects anything.
        self.delay = max(
            site.delay_seconds, self.robots.crawl_delay(site.base_url) or 0.0
        )

    # --- helpers ---------------------------------------------------------

    def _get(self, url: str) -> bytes | None:
        if not self.robots.can_fetch(url):
            self.stats.bump("blocked_by_robots")
            return None
        try:
            response = self.session.get(url, timeout=self.cfg.request_timeout)
        except requests.RequestException as exc:
            log.debug("fetch failed %s: %s", url, exc)
            self.stats.bump("fetch_error")
            time.sleep(self.delay)
            return None
        time.sleep(self.delay)
        if response.status_code != 200:
            self.stats.bump(f"http_{response.status_code}")
            return None

        data = response.content
        if data[:2] == b"\x1f\x8b":  # gzipped sitemap, detected by magic number
            try:
                data = gzip.decompress(data)
            except OSError:
                self.stats.bump("gunzip_error")
                return None
        return data

    def _accept(self, url: str) -> None:
        url = normalise_url(url)
        if not url.startswith("http"):
            return
        self.stats.bump("urls_seen")
        if not self.site.accepts(url):
            self.stats.bump("dropped_by_url_filter")
            return
        if url not in self.found:
            self.found[url] = None

    def _full(self) -> bool:
        return self.site.max_urls is not None and len(self.found) >= self.site.max_urls

    # --- strategy: sitemap -----------------------------------------------

    def _from_sitemaps(self) -> None:
        seeds = list(self.site.sitemap_urls) or self.robots.sitemaps(self.site.base_url)
        if not seeds:
            seeds = [urljoin(self.site.base_url, "/sitemap.xml")]

        queue = deque((url, 0) for url in seeds)
        visited: set[str] = set()

        while queue and not self._full():
            sitemap_url, depth = queue.popleft()
            if sitemap_url in visited or depth > MAX_SITEMAP_DEPTH:
                continue
            visited.add(sitemap_url)

            data = self._get(sitemap_url)
            if data is None:
                continue
            if not _looks_like_xml(data):
                log.warning("%s is not XML (soft 404?) -- no sitemap here", sitemap_url)
                self.stats.bump("sitemap_soft_404")
                continue

            try:
                root = ET.fromstring(data)
            except ET.ParseError as exc:
                log.warning("malformed sitemap %s: %s", sitemap_url, exc)
                self.stats.bump("sitemap_parse_error")
                continue

            self.stats.bump("sitemaps_read")

            for element in root.findall(".//sm:sitemap/sm:loc", SITEMAP_NS):
                if element.text:
                    self._queue_sitemap(queue, element.text.strip(), depth + 1)

            for element in root.findall(".//sm:url/sm:loc", SITEMAP_NS):
                if not element.text:
                    continue
                url = element.text.strip()
                # Some sites (setopati among them) build their sitemap index
                # with <url> entries instead of <sitemap> entries. Anything
                # ending in .xml is a child sitemap, not an article.
                if url.lower().endswith(".xml"):
                    self._queue_sitemap(queue, url, depth + 1)
                else:
                    self._accept(url)

    def _queue_sitemap(self, queue: deque, url: str, depth: int) -> None:
        """Queue a child sitemap unless it is excluded.

        Tag and author sitemaps are typically hundreds of files listing pages
        that contain no article prose. Fetching them wastes the request budget
        and risks rate limiting, so they are skipped before being requested.
        """
        if any(re.search(p, url) for p in self.site.sitemap_exclude):
            self.stats.bump("sitemap_skipped_by_filter")
            return
        queue.append((url, depth))

    # --- strategy: rss ----------------------------------------------------

    def _from_rss(self) -> None:
        feeds = list(self.site.rss_urls) or [urljoin(self.site.base_url, "/feed")]
        for feed_url in feeds:
            if self._full():
                return
            data = self._get(feed_url)
            if data is None or not _looks_like_xml(data):
                continue
            try:
                root = ET.fromstring(data)
            except ET.ParseError:
                self.stats.bump("rss_parse_error")
                continue

            self.stats.bump("feeds_read")
            # RSS 2.0: <item><link>text</link>. Atom: <entry><link href="..."/>.
            for element in root.iter():
                tag = element.tag.rsplit("}", 1)[-1]
                if tag != "link":
                    continue
                url = element.get("href") or (element.text or "")
                if url.strip():
                    self._accept(url.strip())

    # --- strategy: crawl --------------------------------------------------

    def _from_crawl(self) -> None:
        """Breadth-first link following, restricted to this domain.

        Every fetched page contributes its links both to the frontier and to the
        candidate set; the URL filters in the site config decide which of those
        are articles. Bounded by ``crawl_max_pages`` and ``crawl_max_depth`` so
        a run always terminates.
        """
        seeds = list(self.site.seeds) or [self.site.base_url]
        # URLs already discovered by cheaper strategies make excellent seeds:
        # category and tag listing pages link to many articles.
        seeds += list(self.found)[: self.site.crawl_seed_limit]

        queue = deque((normalise_url(u), 0) for u in seeds)
        visited: set[str] = set()
        fetched = 0

        while queue and fetched < self.site.crawl_max_pages and not self._full():
            url, depth = queue.popleft()
            if url in visited or depth > self.site.crawl_max_depth:
                continue
            visited.add(url)

            if _host(url) != self.domain or _ASSET_RE.search(url):
                continue

            data = self._get(url)  # _get applies the politeness delay
            fetched += 1
            if data is None:
                continue
            self.stats.bump("crawl_pages_fetched")

            html = data.decode("utf-8", errors="replace")
            for href in _HREF_RE.findall(html):
                child = normalise_url(urljoin(url, href))
                if not child.startswith("http"):
                    continue
                if _host(child) != self.domain or _ASSET_RE.search(child):
                    continue
                self._accept(child)
                if child not in visited:
                    queue.append((child, depth + 1))

            if fetched % 50 == 0:
                log.info(
                    "[%s] crawl: %d pages fetched, %d URLs found",
                    self.site.name, fetched, len(self.found),
                )

    # --- entry point ------------------------------------------------------

    def run(self) -> list[dict[str, object]]:
        """Run this site's strategies in order and return the URL records found."""
        strategies = {
            "sitemap": self._from_sitemaps,
            "rss": self._from_rss,
            "crawl": self._from_crawl,
        }
        for name in self.site.strategies:
            if self._full():
                break
            action = strategies.get(name)
            if action is None:
                log.warning("[%s] unknown strategy %r, skipping", self.site.name, name)
                continue
            before = len(self.found)
            try:
                action()
            except KeyboardInterrupt:
                # A crawl of a large news site can run for hours. Interrupting
                # it must not throw away what it has already found, so the URLs
                # collected so far fall through to the normal write below.
                log.warning("[%s] interrupted during %s -- keeping %d URLs found so far",
                            self.site.name, name, len(self.found))
                break
            log.info("[%s] strategy %-8s -> +%d URLs (total %d)",
                     self.site.name, name, len(self.found) - before, len(self.found))

        urls = list(self.found)
        if self.site.max_urls is not None:
            urls = urls[: self.site.max_urls]
        self.stats.bump("urls_kept", len(urls))

        discovered_at = utc_now()
        return [
            {
                "url": url,
                "source_name": self.site.name,
                "lang": self.cfg.lang,
                "discovered_at": discovered_at,
            }
            for url in urls
        ]


def discover_site(cfg: ScrapeConfig, site: SiteConfig, robots: RobotsPolicy) -> int:
    """Run discovery for one site and write ``urls/<site>.jsonl``. Returns URL count."""
    discoverer = SiteDiscoverer(cfg, site, robots)
    records = discoverer.run()

    out_path = cfg.urls_dir / f"{site.name}.jsonl"
    # Merge rather than overwrite. A later run with wider crawl limits explores
    # different parts of the site, so replacing the file would silently discard
    # URLs an earlier run found but the fetcher had not reached yet.
    if out_path.exists():
        merged = {record["url"]: record for record in read_jsonl(out_path)}
        before = len(merged)
        for record in records:
            merged.setdefault(record["url"], record)
        log.info("[%s] merged with %d previously discovered URLs -> %d total",
                 site.name, before, len(merged))
        records = list(merged.values())
    written = write_jsonl(out_path, records)
    discoverer.stats.save(cfg.urls_dir / f"_stats_discover_{site.name}.json")

    log.info("%s -> %s", discoverer.stats.summary(), out_path)
    return written
