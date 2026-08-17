"""Filesystem helpers shared by the scrape stages.

Three properties every stage needs and that these helpers provide:

* **atomic** writes, so a process killed mid-write leaves the previous file
  intact rather than a truncated one;
* **resumable** runs, via an on-disk set of already-processed URLs;
* **instrumented** runs, via a ``_stats.json`` sidecar per stage that feeds the
  cleaning table in the Phase 1 report.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

# Devanagari block, used for the script-ratio sanity check. Both Hindi and
# Nepali are written in this script, so a page with almost no Devanagari is
# either the wrong language, an error page, or legacy non-Unicode font text.
DEVANAGARI_START, DEVANAGARI_END = 0x0900, 0x097F


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (recorded on every document)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def doc_id(url: str) -> str:
    """Stable document id: SHA-256 of the URL.

    Deterministic across runs and machines, which is what makes the pipeline
    idempotent -- re-fetching a URL overwrites the same file instead of
    creating a duplicate record.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def devanagari_ratio(text: str) -> float:
    """Fraction of *letter* characters that are Devanagari.

    Digits, punctuation and whitespace are ignored so that the ratio is not
    diluted by numbers and markup. Returns 0.0 for text with no letters.
    """
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    hits = sum(1 for c in letters if DEVANAGARI_START <= ord(c) <= DEVANAGARI_END)
    return hits / len(letters)


_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)\b.*?</\1>", re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")


def visible_text_from_html(html: str) -> str:
    """Crude tag-strip, for *measurement* only (not for corpus extraction).

    Measuring a script ratio on raw HTML is meaningless: tags, CSS class names
    and inline JavaScript are all Latin, so even a fully Devanagari page scores
    only ~0.10. Stripping markup first is what makes the ratio interpretable.

    Real corpus extraction uses trafilatura (see :mod:`common.scrape.extract`),
    which also removes navigation and boilerplate; this helper does not.
    """
    without_code = _SCRIPT_STYLE_RE.sub(" ", html)
    return _TAG_RE.sub(" ", without_code)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (write to .tmp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)  # atomic on POSIX


def write_gzip_bytes(path: Path, data: bytes) -> None:
    """Gzip ``data`` and write it atomically. Raw HTML compresses ~5-10x."""
    atomic_write_bytes(path, gzip.compress(data))


def read_gzip_bytes(path: Path) -> bytes:
    """Read a file written by :func:`write_gzip_bytes`."""
    with gzip.open(path, "rb") as handle:
        return handle.read()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a .jsonl or .jsonl.gz file, skipping blank lines."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSON Lines (gzipped if the path ends in .gz).

    Returns the number of records written. Written atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open

    count = 0
    with opener(tmp, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


class JsonlAppender:
    """Append-only JSON Lines writer, flushed after every record.

    Used for the fetch manifest: flushing on every line is what lets a run
    survive Ctrl-C or a dropped connection without losing the record of what
    was already downloaded.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlAppender":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SeenSet:
    """Persistent set of already-processed URLs, for resumable runs.

    Backed by a plain newline-delimited file. Loaded into memory on startup and
    appended (and flushed) after each success, so an interrupted run resumes
    exactly where it stopped.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                self._seen = {line.strip() for line in handle if line.strip()}
        self._handle = open(path, "a", encoding="utf-8")

    def __contains__(self, url: str) -> bool:
        return url in self._seen

    def __len__(self) -> int:
        return len(self._seen)

    def add(self, url: str) -> None:
        if url in self._seen:
            return
        self._seen.add(url)
        self._handle.write(url + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "SeenSet":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Stats:
    """Counters for one stage, dumped to a ``_stats.json`` sidecar.

    Every drop is recorded with a reason. The Phase 1 report's cleaning table
    is generated from these files, and a source that silently yields nothing
    (for example a site serving legacy Preeti-font text) is diagnosed by
    reading the drop reasons rather than by guesswork.
    """

    def __init__(self, stage: str, label: str) -> None:
        self.stage = stage
        self.label = label
        self.started_at = utc_now()
        self.counts: Counter[str] = Counter()

    def bump(self, key: str, amount: int = 1) -> None:
        self.counts[key] += amount

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "label": self.label,
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "counts": dict(sorted(self.counts.items())),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            path,
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False).encode("utf-8"),
        )

    def summary(self) -> str:
        parts = [f"{key}={value}" for key, value in sorted(self.counts.items())]
        return f"[{self.stage}:{self.label}] " + "  ".join(parts)
