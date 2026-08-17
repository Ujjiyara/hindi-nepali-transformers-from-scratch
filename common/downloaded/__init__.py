"""Ingestion of pre-built public corpora.

These are the *downloaded* half of the dataset, as opposed to the Wikimedia
dumps and scraped pages we collect and clean ourselves. Every document emitted
here carries ``collection_method="downloaded"`` so the manual fraction required
by the project brief can be measured after deduplication.
"""
