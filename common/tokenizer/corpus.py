"""Turning a JSONL split into the plain-text file SentencePiece trains on."""

from __future__ import annotations

import logging
from pathlib import Path

from common.scrape.io_utils import read_jsonl

log = logging.getLogger(__name__)


def export_lines(
    split_path: Path, out_path: Path, max_lines: int | None = None, min_line_chars: int = 10
) -> tuple[int, int]:
    """Write one text line per corpus line, for SentencePiece training input.

    Returns ``(lines_written, characters_written)``.

    SentencePiece expects one sentence (or short passage) per line. Documents
    are therefore exploded into their lines; very short lines are skipped
    because they are usually headings or list fragments rather than prose.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines_written = 0
    chars_written = 0

    with open(out_path, "w", encoding="utf-8") as handle:
        for record in read_jsonl(split_path):
            for line in record.get("text", "").splitlines():
                line = line.strip()
                if len(line) < min_line_chars:
                    continue
                handle.write(line + "\n")
                lines_written += 1
                chars_written += len(line)
                if max_lines is not None and lines_written >= max_lines:
                    log.info("reached max_lines=%d", max_lines)
                    return lines_written, chars_written

    return lines_written, chars_written


def iter_eval_text(split_path: Path, max_docs: int | None = None):
    """Yield held-out document texts, for tokenizer evaluation."""
    for index, record in enumerate(read_jsonl(split_path)):
        if max_docs is not None and index >= max_docs:
            return
        text = record.get("text", "")
        if text:
            yield text
