"""Stage: deduplication.

Two levels, because they catch different things:

**Exact** -- identical text after normalisation. Cheap, catches syndicated
copies republished byte-for-byte and pages fetched twice under different URLs.

**Near-duplicate** -- MinHash signatures indexed with LSH. Catches the same
wire story rewritten with a different headline, boilerplate-heavy pages that
differ only in a date, and the large overlap between a scraped corpus and a
downloaded one built from Common Crawl.

Why near-duplicate removal matters for a language model: repeated text is
memorised rather than generalised, and duplicated documents leaking across the
train/test boundary make held-out perplexity look better than it is.

**The tie-break that protects the 20% rule.** Downloaded corpora (CC-100,
OSCAR, mC4) are built from Common Crawl, which crawled Wikipedia -- so a
manually parsed Wikipedia article and a downloaded copy of the same article are
near-duplicates. Whichever survives is otherwise arbitrary, and if the
downloaded copy wins, that document silently flips from ``manual`` to
``downloaded`` and the manual fraction erodes with no error raised. Cluster
representatives are therefore chosen manual-first.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Mersenne prime 2^61-1, the modulus for the hash family.
_MERSENNE = (1 << 61) - 1

DEFAULT_NUM_PERM = 128
DEFAULT_BANDS = 16  # 16 bands x 8 rows -> similarity threshold ~0.76
DEFAULT_SHINGLE_WORDS = 5


def text_hash(text: str) -> str:
    """Hash used for exact-duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(
    text: str, size: int = DEFAULT_SHINGLE_WORDS, max_shingles: int | None = 400
) -> set[int]:
    """Word-level n-gram fingerprints of a document.

    Word shingles are used rather than character shingles: they are far cheaper
    on long documents and, for Devanagari, they avoid the sensitivity of
    character n-grams to matra-level differences that carry little meaning.

    Each shingle is hashed to 32 bits so the MinHash arithmetic below cannot
    overflow uint64.
    """
    words = text.split()
    if len(words) < size:
        return {
            int.from_bytes(
                hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=4).digest(),
                "big",
            )
        } if words else set()

    # Long documents are subsampled with a fixed stride. Hashing every shingle
    # of a 20,000-word Wikisource page costs far more than it adds: a few
    # hundred evenly spaced shingles estimate Jaccard similarity just as well,
    # and the stride keeps the choice deterministic.
    total = len(words) - size + 1
    stride = 1
    if max_shingles is not None and total > max_shingles:
        stride = total // max_shingles + 1

    result = set()
    for index in range(0, total, stride):
        gram = " ".join(words[index : index + size]).encode("utf-8")
        result.add(int.from_bytes(hashlib.blake2b(gram, digest_size=4).digest(), "big"))
    return result


class MinHasher:
    """Random hash family producing MinHash signatures.

    ``a`` and ``b`` are kept below 2^31 and shingles below 2^32 so that
    ``a * h + b`` stays inside uint64 and no silent wraparound corrupts the
    permutation.
    """

    def __init__(self, num_perm: int = DEFAULT_NUM_PERM, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)

    def signature(self, shingle_set: set[int]) -> np.ndarray:
        if not shingle_set:
            return np.full(self.num_perm, np.uint64(_MERSENNE), dtype=np.uint64)
        values = np.fromiter(shingle_set, dtype=np.uint64, count=len(shingle_set))
        hashed = (self.a[:, None] * values[None, :] + self.b[:, None]) % np.uint64(_MERSENNE)
        return hashed.min(axis=1)


class UnionFind:
    """Disjoint-set forest, used to merge LSH candidate pairs into clusters."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]  # path halving
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass
class DedupResult:
    keep_indices: list[int]
    exact_dropped: int
    near_dropped: int
    clusters: int


@dataclass
class DocFeatures:
    """Everything dedup needs about a document, without holding its text.

    Text is the memory hog: a full Hindi corpus with CC-100 does not fit in
    RAM. Extracting these features in a streaming first pass lets dedup run on
    a few hundred bytes per document instead of a few kilobytes, and the text
    is re-read from disk only for the documents that survive.
    """

    exact_hash: str
    signature: np.ndarray
    is_manual: bool
    n_chars: int


def _pick(indices: list[int], features: list[DocFeatures]) -> int:
    """Cluster representative: manual first, then longest (see module docstring)."""
    return max(indices, key=lambda i: (features[i].is_manual, features[i].n_chars))


def deduplicate_features(
    features: list[DocFeatures], bands: int = DEFAULT_BANDS
) -> DedupResult:
    """Deduplicate from precomputed features rather than from records."""
    if not features:
        return DedupResult([], 0, 0, 0)

    num_perm = len(features[0].signature)
    if num_perm % bands:
        raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")
    rows = num_perm // bands

    # --- exact duplicates -------------------------------------------------
    by_hash: dict[str, list[int]] = {}
    for index, feature in enumerate(features):
        by_hash.setdefault(feature.exact_hash, []).append(index)

    survivors: list[int] = []
    exact_dropped = 0
    for group in by_hash.values():
        survivors.append(_pick(group, features) if len(group) > 1 else group[0])
        exact_dropped += len(group) - 1
    survivors.sort()
    del by_hash
    log.info("exact dedup: %d -> %d documents", len(features), len(survivors))

    # --- near duplicates via LSH -----------------------------------------
    signatures = np.stack([features[i].signature for i in survivors])
    union = UnionFind(len(survivors))
    for band in range(bands):
        buckets: dict[bytes, int] = {}
        block = np.ascontiguousarray(signatures[:, band * rows : (band + 1) * rows])
        for position in range(len(survivors)):
            key = block[position].tobytes()
            first = buckets.get(key)
            if first is None:
                buckets[key] = position
            else:
                union.union(first, position)
    del signatures

    clusters: dict[int, list[int]] = {}
    for position in range(len(survivors)):
        clusters.setdefault(union.find(position), []).append(position)

    keep: list[int] = []
    near_dropped = 0
    multi = 0
    for members in clusters.values():
        original = [survivors[p] for p in members]
        if len(original) > 1:
            multi += 1
            near_dropped += len(original) - 1
        keep.append(_pick(original, features))
    keep.sort()

    log.info(
        "near dedup: %d -> %d documents (%d clusters)", len(survivors), len(keep), multi
    )
    return DedupResult(keep, exact_dropped, near_dropped, multi)


def _representative(indices: list[int], records: list[dict]) -> int:
    """Pick which member of a duplicate cluster to keep.

    Manual documents win over downloaded ones (see the module docstring);
    length is the tie-break, on the assumption that the longer copy is the
    less truncated one.
    """
    return max(
        indices,
        key=lambda i: (
            records[i].get("collection_method") == "manual",
            len(records[i].get("text", "")),
        ),
    )


def deduplicate(
    records: list[dict],
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
    shingle_size: int = DEFAULT_SHINGLE_WORDS,
) -> DedupResult:
    """Remove exact and near-duplicate documents, preferring manual copies."""
    if num_perm % bands:
        raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")
    rows = num_perm // bands

    # --- exact duplicates -------------------------------------------------
    by_hash: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_hash.setdefault(text_hash(record["text"]), []).append(index)

    survivors = []
    exact_dropped = 0
    for group in by_hash.values():
        survivors.append(_representative(group, records) if len(group) > 1 else group[0])
        exact_dropped += len(group) - 1
    survivors.sort()

    log.info("exact dedup: %d -> %d documents", len(records), len(survivors))

    # --- near duplicates --------------------------------------------------
    hasher = MinHasher(num_perm=num_perm)
    signatures = np.empty((len(survivors), num_perm), dtype=np.uint64)
    for position, index in enumerate(survivors):
        signatures[position] = hasher.signature(
            shingles(records[index]["text"], shingle_size)
        )
        if position and position % 20000 == 0:
            log.info("  minhash %d/%d", position, len(survivors))

    union = UnionFind(len(survivors))
    for band in range(bands):
        buckets: dict[bytes, int] = {}
        block = signatures[:, band * rows : (band + 1) * rows]
        for position in range(len(survivors)):
            key = block[position].tobytes()
            first = buckets.get(key)
            if first is None:
                buckets[key] = position
            else:
                union.union(first, position)

    clusters: dict[int, list[int]] = {}
    for position in range(len(survivors)):
        clusters.setdefault(union.find(position), []).append(position)

    keep = []
    near_dropped = 0
    multi_clusters = 0
    for members in clusters.values():
        original = [survivors[position] for position in members]
        if len(original) > 1:
            multi_clusters += 1
            near_dropped += len(original) - 1
        keep.append(_representative(original, records))

    keep.sort()
    log.info("near dedup: %d -> %d documents (%d clusters)", len(survivors), len(keep), multi_clusters)

    return DedupResult(
        keep_indices=keep,
        exact_dropped=exact_dropped,
        near_dropped=near_dropped,
        clusters=multi_clusters,
    )
