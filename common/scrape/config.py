"""Configuration objects for the corpus scraper.

Every tunable value lives in a per-language YAML file (``<Lang>/configs/
scrape_sites.yaml``) rather than in the code, so that a run is reproducible
from the committed config alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# common/scrape/config.py -> common/scrape -> common -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SiteConfig:
    """One website to scrape.

    Attributes:
        name: Short identifier, used as a directory name and as ``source_name``
            in the emitted records. Must be filesystem safe.
        base_url: Site root, used to locate ``robots.txt``.
        sitemap_urls: Explicit sitemap locations. If empty, the sitemaps
            advertised in ``robots.txt`` are used, falling back to
            ``<base_url>/sitemap.xml``.
        url_include: Regexes; if non-empty a URL must match at least one to be
            kept. Use this to select article URLs and drop tag/category pages.
        url_exclude: Regexes; a URL matching any of these is always dropped.
        max_urls: Optional cap on discovered URLs for this site.
        delay_seconds: Politeness delay between requests to this site. The
            ``Crawl-delay`` from robots.txt overrides this when it is larger.
    """

    name: str
    base_url: str
    strategies: list[str] = field(default_factory=lambda: ["sitemap", "rss", "crawl"])
    sitemap_urls: list[str] = field(default_factory=list)
    sitemap_exclude: list[str] = field(
        default_factory=lambda: [r"/tags?[-/]", r"/authors?[-/]", r"/members?[-/]"]
    )
    rss_urls: list[str] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    url_include: list[str] = field(default_factory=list)
    url_exclude: list[str] = field(default_factory=list)
    max_urls: int | None = None
    delay_seconds: float = 1.5
    crawl_max_pages: int = 1500
    crawl_max_depth: int = 4
    crawl_seed_limit: int = 300

    def __post_init__(self) -> None:
        self._include = [re.compile(p) for p in self.url_include]
        self._exclude = [re.compile(p) for p in self.url_exclude]

    def accepts(self, url: str) -> bool:
        """Return True if ``url`` passes this site's include/exclude filters."""
        if any(p.search(url) for p in self._exclude):
            return False
        if self._include:
            return any(p.search(url) for p in self._include)
        return True


@dataclass
class ScrapeConfig:
    """Top-level scrape configuration for a single language.

    Attributes:
        lang: ISO code recorded on every emitted document (``hi`` / ``ne``).
        data_root: Language data directory, relative to the repository root
            (e.g. ``Nepali/data``).
        user_agent: Sent on every request. Identifies the project and a contact
            address so site owners can reach us rather than silently blocking.
        sites: Sites to scrape.
        request_timeout: ``(connect, read)`` seconds. Never unset: a request
            without a timeout can hang forever and stall the whole run.
        max_retries: Attempts for retryable failures (5xx, timeouts).
        min_chars: Extracted documents shorter than this are discarded as
            navigation stubs / error pages rather than real articles.
    """

    lang: str
    data_root: Path
    user_agent: str
    sites: list[SiteConfig] = field(default_factory=list)
    request_timeout: tuple[float, float] = (5.0, 30.0)
    max_retries: int = 3
    min_chars: int = 200

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScrapeConfig":
        """Load a config from YAML, resolving ``data_root`` against the repo root."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        sites = [SiteConfig(**site) for site in raw.pop("sites", [])]
        timeout = raw.pop("request_timeout", None)

        data_root = Path(raw.pop("data_root"))
        if not data_root.is_absolute():
            data_root = REPO_ROOT / data_root

        cfg = cls(data_root=data_root, sites=sites, **raw)
        if timeout is not None:
            cfg.request_timeout = (float(timeout[0]), float(timeout[1]))
        return cfg

    def site(self, name: str) -> SiteConfig:
        """Look up a single site by name, or raise KeyError."""
        for site in self.sites:
            if site.name == name:
                return site
        raise KeyError(f"no site named {name!r} in config for {self.lang}")

    # --- directory layout -------------------------------------------------
    # raw/ is immutable once written; everything else is derived and can be
    # deleted and regenerated at any time.

    @property
    def urls_dir(self) -> Path:
        return self.data_root / "urls"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw" / "scrape"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim" / "scrape"
