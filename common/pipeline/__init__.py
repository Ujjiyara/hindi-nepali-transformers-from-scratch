"""Shared cleaning pipeline.

Both languages run the same code over their own data; nothing is shared
*between* languages except the implementation. Stages, in order::

    normalize : Unicode NFC, invisible characters, boilerplate, quality filters
    dedup     : exact hash, then MinHash/LSH near-duplicates
    split     : deterministic per-document train / validation / test

Each stage reads and writes JSON Lines and drops a ``_stats.json`` beside its
output, so the corpus tables in the Phase 1 report are generated from measured
counts rather than written by hand.
"""
