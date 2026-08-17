"""Streaming reader for Wikimedia ``pages-articles`` XML dumps.

The Hindi dump is 229 MB compressed and roughly 1.4 GB expanded, so the file is
never loaded into memory. ``iterparse`` yields elements as they are parsed and
each ``<page>`` is cleared immediately after use; without that clearing the
element tree accumulates and the process runs out of memory partway through.
"""

from __future__ import annotations

import bz2
import logging
import multiprocessing as mp
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, NamedTuple
from urllib.parse import quote

from common.wiki.wikitext import strip_wikitext

log = logging.getLogger(__name__)

# Only namespace 0 is article content. Everything else (Talk, User, Template,
# Category, Help, Portal, ...) is metadata or discussion, not prose.
ARTICLE_NAMESPACE = "0"


class WikiPage(NamedTuple):
    """One article as it appears in the dump, before wikitext stripping."""

    page_id: str
    title: str
    wikitext: str


def _localname(tag: str) -> str:
    """Strip the XML namespace, e.g. '{http://...}page' -> 'page'."""
    return tag.rsplit("}", 1)[-1]


def iter_pages(
    dump_path: Path, namespaces: frozenset[str] = frozenset({ARTICLE_NAMESPACE})
) -> Iterator[WikiPage]:
    """Yield pages from a ``*-pages-articles.xml.bz2`` dump.

    Skips redirect stubs (a redirect page has no body of its own, only a
    pointer to another title) and any page outside ``namespaces``.

    ``namespaces`` defaults to article space. Wikisource needs an override:
    its ns 0 entries are transclusion shells (``<pages index="book.pdf" ... />``)
    that contain no literal text, while the actual work is held one scanned page
    per entry in the proofread Page namespace -- ns 250 on Hindi Wikisource.
    """
    with bz2.open(dump_path, "rb") as handle:
        context = ET.iterparse(handle, events=("start", "end"))
        _, root = next(context)  # the <mediawiki> element

        for event, element in context:
            if event != "end" or _localname(element.tag) != "page":
                continue

            namespace = None
            title = None
            page_id = None
            wikitext = None
            is_redirect = False

            for child in element:
                name = _localname(child.tag)
                if name == "ns":
                    namespace = (child.text or "").strip()
                elif name == "title":
                    title = child.text
                elif name == "id" and page_id is None:
                    page_id = (child.text or "").strip()
                elif name == "redirect":
                    is_redirect = True
                elif name == "revision":
                    for revision_child in child:
                        if _localname(revision_child.tag) == "text":
                            wikitext = revision_child.text

            if namespace in namespaces and not is_redirect and title and wikitext:
                yield WikiPage(page_id or "", title, wikitext)

            # Free the page and any siblings the root is still holding on to.
            element.clear()
            root.clear()


def article_url(site: str, title: str) -> str:
    """Canonical article URL, e.g. https://ne.wikipedia.org/wiki/<title>."""
    return f"https://{site}/wiki/{quote(title.replace(' ', '_'), safe='')}"


def _strip_one(page: WikiPage) -> tuple[str, str, str]:
    """Worker function: (page_id, title, plain_text). Must be top-level for pickling."""
    return page.page_id, page.title, strip_wikitext(page.wikitext)


def iter_stripped(
    dump_path: Path,
    workers: int = 1,
    chunk_size: int = 200,
    namespaces: frozenset[str] = frozenset({ARTICLE_NAMESPACE}),
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(page_id, title, plain_text)``, optionally stripping in parallel.

    Decompression and XML parsing stay in the main process (they are sequential
    by nature); the CPU-bound wikitext stripping is what gets distributed.
    """
    pages = iter_pages(dump_path, namespaces=namespaces)
    if workers <= 1:
        for page in pages:
            yield _strip_one(page)
        return

    with mp.Pool(processes=workers) as pool:
        yield from pool.imap(_strip_one, pages, chunksize=chunk_size)
