"""Turning a JSONL split into the plain-text file SentencePiece trains on."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from common.scrape.io_utils import read_jsonl

log = logging.getLogger(__name__)

# Sampling seed. Fixed so that a capped export is reproducible: the same split
# always yields the same training text, and therefore the same vocabulary.
SAMPLE_SEED = 20260818


def export_lines(
    split_path: Path,
    out_path: Path,
    max_lines: int | None = None,
    min_line_chars: int = 10,
) -> tuple[int, int]:
    """Write one text line per corpus line, for SentencePiece training input.

    Returns ``(lines_written, characters_written)``.

    SentencePiece expects one sentence (or short passage) per line. Documents
    are therefore exploded into their lines; very short lines are skipped
    because they are usually headings or list fragments rather than prose.

    ``max_lines`` **samples** rather than truncates. Truncating would take the
    lines in file order, and a split is written source by source -- so the
    first N lines of the Hindi corpus are pure CC-100, with no Wikipedia, news
    or Wikisource in them at all. A tokenizer fitted on that prefix would be a
    web-crawl tokenizer being measured against mixed held-out text. Reservoir
    sampling keeps the cap while leaving the source mix intact.

    Leaving ``max_lines`` unset exports everything, which is the normal case:
    SentencePiece then does its own uniform subsampling via
    ``input_sentence_size``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def source_lines():
        for record in read_jsonl(split_path):
            for line in record.get("text", "").splitlines():
                line = line.strip()
                if len(line) >= min_line_chars:
                    yield line

    if max_lines is None:
        lines_written = 0
        chars_written = 0
        with open(out_path, "w", encoding="utf-8") as handle:
            for line in source_lines():
                handle.write(line + "\n")
                lines_written += 1
                chars_written += len(line)
        return lines_written, chars_written

    # Reservoir sampling: one streaming pass, holding at most max_lines lines.
    rng = random.Random(SAMPLE_SEED)
    reservoir: list[str] = []
    seen = 0
    for line in source_lines():
        seen += 1
        if len(reservoir) < max_lines:
            reservoir.append(line)
        else:
            index = rng.randrange(seen)
            if index < max_lines:
                reservoir[index] = line

    if seen > max_lines:
        log.info("sampled %s lines of %s (max_lines)", f"{max_lines:,}", f"{seen:,}")

    with open(out_path, "w", encoding="utf-8") as handle:
        for line in reservoir:
            handle.write(line + "\n")
    return len(reservoir), sum(len(line) for line in reservoir)


def iter_eval_text(split_path: Path, max_docs: int | None = None):
    """Yield held-out document texts, for tokenizer evaluation."""
    for index, record in enumerate(read_jsonl(split_path)):
        if max_docs is not None and index >= max_docs:
            return
        text = record.get("text", "")
        if text:
            yield text
