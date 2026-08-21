"""Command-line entry point for the corpus scraper.

Run from the repository root, e.g.::

    python -m common.scrape.run triage  --config Nepali/configs/scrape_sites.yaml
    python -m common.scrape.run discover --config Nepali/configs/scrape_sites.yaml
    python -m common.scrape.run fetch    --config Nepali/configs/scrape_sites.yaml --limit 5
    python -m common.scrape.run extract  --config Nepali/configs/scrape_sites.yaml

``--site NAME`` restricts any subcommand to a single site.
"""

from __future__ import annotations

import argparse
import logging
import sys

import requests

from common.scrape.config import ScrapeConfig, SiteConfig
from common.scrape.discover import discover_site
from common.scrape.extract import extract_site
from common.scrape.fetch import fetch_all
from common.scrape.io_utils import devanagari_ratio, visible_text_from_html
from common.scrape.robots import RobotsPolicy

# Judged on tag-stripped text, not raw HTML. A server-rendered Devanagari
# article page scores well above this; a JavaScript shell scores near zero.
MIN_VISIBLE_DEVANAGARI_RATIO = 0.50
# Absolute floor: a page can be 90% Devanagari and still be a nearly empty
# shell with only a headline in it.
MIN_DEVANAGARI_CHARS = 500

log = logging.getLogger("scrape")


def _selected_sites(cfg: ScrapeConfig, name: str | None) -> list[SiteConfig]:
    return [cfg.site(name)] if name else list(cfg.sites)


def cmd_triage(cfg: ScrapeConfig, sites: list[SiteConfig]) -> int:
    """Check each site is reachable, robots-permitted, and serves real Devanagari.

    Sites that render their content with JavaScript return an almost empty HTML
    shell to a plain HTTP client; sites using legacy Preeti/Kantipur fonts return
    ASCII that only *looks* like Devanagari in a browser. Both are unusable, and
    both are far cheaper to detect now than after an overnight scrape.
    """
    robots = RobotsPolicy(cfg.user_agent, timeout=cfg.request_timeout)
    session = requests.Session()
    session.headers["User-Agent"] = cfg.user_agent
    usable = 0

    for site in sites:
        print(f"\n=== {site.name} ({site.base_url})")
        try:
            response = session.get(site.base_url, timeout=cfg.request_timeout)
        except requests.RequestException as exc:
            print(f"  UNREACHABLE: {exc}")
            continue

        html = response.content.decode("utf-8", errors="replace")
        visible = visible_text_from_html(html)
        ratio = devanagari_ratio(visible)
        n_deva = sum(1 for c in visible if 0x0900 <= ord(c) <= 0x097F)

        sitemaps = robots.sitemaps(site.base_url)
        allowed = robots.can_fetch(site.base_url)
        delay = robots.crawl_delay(site.base_url)

        print(f"  HTTP {response.status_code}   {len(response.content):,} bytes")
        print(f"  robots: can_fetch={allowed}  crawl_delay={delay}  sitemaps={len(sitemaps)}")
        for sitemap in sitemaps[:3]:
            print(f"    - {sitemap}")
        print(f"  visible text: {len(visible):,} chars   devanagari={n_deva:,} ({ratio:.2f})")

        if not allowed:
            print("  VERDICT: robots.txt disallows -- skip this site")
        elif ratio < MIN_VISIBLE_DEVANAGARI_RATIO or n_deva < MIN_DEVANAGARI_CHARS:
            print("  VERDICT: little Devanagari in rendered text -- "
                  "likely JS-rendered or legacy font. SKIP")
        else:
            if delay and delay >= 10:
                print(f"  NOTE: Crawl-delay={delay}s makes bulk collection very slow "
                      "(~{:,} pages/hour)".format(int(3600 / delay)))
            if not sitemaps and not site.sitemap_urls:
                print("  VERDICT: usable, but no sitemap advertised -- set sitemap_urls manually")
            else:
                print("  VERDICT: looks usable")
            usable += 1

    print(f"\n{usable}/{len(sites)} site(s) look usable.")
    return 0


def cmd_discover(cfg: ScrapeConfig, sites: list[SiteConfig]) -> int:
    """discover: find candidate article URLs for each selected site."""
    robots = RobotsPolicy(cfg.user_agent, timeout=cfg.request_timeout)
    total = sum(discover_site(cfg, site, robots) for site in sites)
    print(f"discovered {total} URL(s) -> {cfg.urls_dir}")
    return 0


def cmd_fetch(cfg: ScrapeConfig, sites: list[SiteConfig], limit: int | None) -> int:
    """fetch: download the discovered URLs, one thread per site."""
    total = fetch_all(cfg, sites, limit=limit)
    print(f"saved {total} new page(s) -> {cfg.raw_dir}")
    return 0


def cmd_extract(cfg: ScrapeConfig, sites: list[SiteConfig]) -> int:
    """extract: turn stored raw HTML into clean text records."""
    total = sum(extract_site(cfg, site) for site in sites)
    print(f"extracted {total} document(s) -> {cfg.interim_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the triage / discover / fetch / extract subcommands."""
    parser = argparse.ArgumentParser(description="Corpus scraper (discover / fetch / extract).")
    parser.add_argument("command", choices=["triage", "discover", "fetch", "extract"])
    parser.add_argument("--config", required=True, help="path to a language scrape config YAML")
    parser.add_argument("--site", default=None, help="restrict to a single site by name")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="fetch: stop after this many NEW pages per site (use a small value first)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = ScrapeConfig.from_yaml(args.config)
    sites = _selected_sites(cfg, args.site)
    if not sites:
        print("no sites configured", file=sys.stderr)
        return 1

    if args.command == "triage":
        return cmd_triage(cfg, sites)
    if args.command == "discover":
        return cmd_discover(cfg, sites)
    if args.command == "fetch":
        return cmd_fetch(cfg, sites, args.limit)
    return cmd_extract(cfg, sites)


if __name__ == "__main__":
    raise SystemExit(main())
