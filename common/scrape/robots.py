"""robots.txt handling.

Honouring robots.txt is both the ethical baseline for corpus collection and
practically useful: the file usually advertises the site's sitemaps, which
saves us from crawling to discover URLs.

Note on why we do not use ``RobotFileParser.read()``: that method fetches with
``urllib``, whose default User-Agent is ``Python-urllib/<version>``. Many news
sites block that UA with HTTP 403, and ``RobotFileParser`` interprets a 403 on
robots.txt as "everything is disallowed". The result is a site that looks
forbidden while its robots.txt is in fact permissive. We therefore fetch the
file ourselves with the project's real User-Agent and hand the text to
``parse()``.
"""

from __future__ import annotations

import logging
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

log = logging.getLogger(__name__)


class RobotsPolicy:
    """Per-domain robots.txt rules, fetched once with our own User-Agent and cached."""

    def __init__(self, user_agent: str, timeout: tuple[float, float] = (5.0, 30.0)) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        self._parsers: dict[str, RobotFileParser | None] = {}

    def _parser(self, url: str) -> RobotFileParser | None:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain in self._parsers:
            return self._parsers[domain]

        robots_url = urljoin(domain, "/robots.txt")
        parser: RobotFileParser | None = RobotFileParser()
        parser.set_url(robots_url)

        try:
            response = self._session.get(robots_url, timeout=self.timeout)
        except requests.RequestException as exc:
            log.warning("could not fetch %s: %s", robots_url, exc)
            parser = None
        else:
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            elif response.status_code in (401, 403):
                # Access to the rules themselves is refused: treat the whole
                # site as off-limits rather than guessing.
                parser.disallow_all = True
                log.warning("%s returned %s -- treating site as disallowed",
                            robots_url, response.status_code)
            elif 400 <= response.status_code < 500:
                # No robots.txt published. The convention is "allowed".
                parser.allow_all = True
            else:
                log.warning("%s returned HTTP %s", robots_url, response.status_code)
                parser = None

        self._parsers[domain] = parser
        return parser

    def can_fetch(self, url: str) -> bool:
        """Return True if robots.txt permits fetching ``url``."""
        parser = self._parser(url)
        if parser is None:
            return True  # no usable robots.txt -> conventional default
        return parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        """Return the site's declared Crawl-delay in seconds, if any."""
        parser = self._parser(url)
        if parser is None:
            return None
        try:
            delay = parser.crawl_delay(self.user_agent)
        except Exception:
            return None
        return float(delay) if delay is not None else None

    def sitemaps(self, base_url: str) -> list[str]:
        """Return sitemap URLs advertised in robots.txt (often better than guessing)."""
        parser = self._parser(base_url)
        if parser is None:
            return []
        try:
            found = parser.site_maps()
        except Exception:
            return []
        return list(found or [])
