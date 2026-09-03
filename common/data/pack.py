"""Turning a JSONL split into the flat token array training reads.

Tokenizing on the fly would re-encode the whole corpus every epoch and make
the data loader, not the GPU, the bottleneck. Instead each split is encoded
once into a single contiguous ``uint16`` array of token ids -- documents
concatenated, separated by the tokenizer's own end-of-sequence id -- and
written to a ``.bin`` file that training memory-maps.

**Why this shape.** A training example is then just a slice: pick an offset,
take ``block_size + 1`` ids, and use ``[:-1]`` as input and ``[1:]`` as target.
No padding is ever needed, so no compute is wasted on pad tokens and no
attention mask beyond the causal one is required. Memory-mapping means a 1GB
array costs no RAM; the operating system pages in the windows actually read.

**Why uint16.** Both vocabularies are 16,000 entries, comfortably inside
65,535, so ids fit in two bytes instead of eight. That is the difference
between a 1.05GB and a 4.2GB file for Hindi, which matters when the file has to
be uploaded to a Kaggle dataset and read from a session with limited disk.

**Sequences cross document boundaries.** A window may start in one article and
end in the next. That is deliberate and standard: the alternative -- padding
each document to ``block_size`` -- wastes a large fraction of every batch on
short documents. The EOS token between documents is the signal the model uses
to learn that a topic has ended.

The byte count recorded in the sidecar is what bits-per-byte is divided by
later, so it is measured here, on exactly the text that was encoded, rather
than re-derived downstream.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm

from common.scrape.io_utils import read_jsonl, utc_now

log = logging.getLogger(__name__)

# Documents are encoded in batches: SentencePiece parallelises a batch across
# threads, and one call per document would spend most of its time in overhead.
ENCODE_BATCH = 2000


def pack_split(
    split_path: Path,
    sp: spm.SentencePieceProcessor,
    out_path: Path,
    num_threads: int = 4,
    limit: int | None = None,
) -> dict:
    """Encode one split into a flat ``uint16`` token file.

    Args:
        split_path: A ``train/validation/test.jsonl.gz`` from Phase 1.
        sp: This language's tokenizer, already loaded.
        out_path: Destination ``.bin`` file.
        num_threads: Threads SentencePiece may use per batch.
        limit: Stop after this many documents. For smoke tests only.

    Returns:
        Statistics for the sidecar: document and token counts, UTF-8 byte
        count, and the token split by collection method (manual vs downloaded),
        which lets Phase 2 confirm it is training on the same corpus mix Phase 1
        reported.
    """
    eos_id = sp.eos_id()
    if eos_id < 0:
        raise ValueError("tokenizer has no eos id; cannot separate documents")
    if sp.get_piece_size() > np.iinfo(np.uint16).max:
        raise ValueError(
            f"vocabulary of {sp.get_piece_size()} does not fit in uint16"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    documents = tokens = utf8_bytes = 0
    tokens_by_method: Counter[str] = Counter()
    docs_by_method: Counter[str] = Counter()

    def flush(handle, batch: list[dict]) -> None:
        """Encode a batch of records and append their ids to the open file."""
        nonlocal documents, tokens, utf8_bytes
        encoded = sp.encode([r["text"] for r in batch], out_type=int,
                            num_threads=num_threads)
        flat: list[int] = []
        for record, ids in zip(batch, encoded):
            ids.append(eos_id)
            flat.extend(ids)
            method = record.get("collection_method", "unknown")
            tokens_by_method[method] += len(ids)
            docs_by_method[method] += 1
            utf8_bytes += len(record["text"].encode("utf-8"))
            documents += 1
        tokens += len(flat)
        handle.write(np.asarray(flat, dtype=np.uint16).tobytes())

    with open(tmp_path, "wb") as handle:
        batch: list[dict] = []
        for record in read_jsonl(split_path):
            batch.append(record)
            if len(batch) >= ENCODE_BATCH:
                flush(handle, batch)
                batch = []
                if limit and documents >= limit:
                    break
                if documents % 100_000 == 0:
                    log.info("%s: %d documents, %d tokens", split_path.name,
                             documents, tokens)
        if batch and not (limit and documents >= limit):
            flush(handle, batch)

    tmp_path.replace(out_path)
    log.info("wrote %s: %d documents, %d tokens", out_path, documents, tokens)

    return {
        "split": split_path.stem.replace(".jsonl", ""),
        "path": str(out_path),
        "documents": documents,
        "tokens": tokens,
        "utf8_bytes": utf8_bytes,
        "eos_id": eos_id,
        "dtype": "uint16",
        "tokens_by_method": dict(tokens_by_method),
        "documents_by_method": dict(docs_by_method),
    }


def write_sidecar(out_dir: Path, lang: str, tokenizer_path: Path,
                  vocab_size: int, splits: list[dict]) -> Path:
    """Record what was packed, next to the ``.bin`` files.

    The tokenizer path and vocabulary size are stored so that training can
    refuse to start if the model config, the tokenizer and the token files do
    not all describe the same language.
    """
    payload = {
        "lang": lang,
        "generated_at": utc_now(),
        "tokenizer": {"model": str(tokenizer_path), "vocab_size": vocab_size},
        "splits": {s["split"]: s for s in splits},
    }
    path = out_dir / "_stats_pack.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path
