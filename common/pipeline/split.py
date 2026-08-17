"""Stage: train / validation / test splitting.

Two properties matter and both are easy to get wrong.

**Split by document, never by line.** If sentences from one article land in
both train and test, the model has effectively seen the test set and held-out
perplexity is meaningless.

**Deterministic assignment.** The split is a pure function of the document id,
so re-running the pipeline, adding new sources, or rebuilding on another
machine leaves every existing document in the split it was already in. A
shuffle with a random seed does not give you that: adding one document
reshuffles everything and silently moves old test documents into train.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

_BUCKETS = 10_000


@dataclass
class SplitConfig:
    """Fractions must sum to 1.0. ``salt`` re-randomises the assignment."""

    val_fraction: float = 0.01
    test_fraction: float = 0.01
    salt: str = "lma-phase1"

    def __post_init__(self) -> None:
        if not 0 <= self.val_fraction < 1 or not 0 <= self.test_fraction < 1:
            raise ValueError("fractions must be in [0, 1)")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val + test must leave room for train")


def assign_split(doc_id: str, cfg: SplitConfig) -> str:
    """Return 'train', 'validation' or 'test' for a document id.

    The id is hashed with a salt and mapped to one of 10,000 buckets; the
    bucket index decides the split. Because it depends only on the id, the
    assignment is stable across runs and reproducible from the config alone.
    """
    digest = hashlib.sha256(f"{cfg.salt}:{doc_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % _BUCKETS

    test_cut = int(cfg.test_fraction * _BUCKETS)
    val_cut = test_cut + int(cfg.val_fraction * _BUCKETS)

    if bucket < test_cut:
        return "test"
    if bucket < val_cut:
        return "validation"
    return "train"
