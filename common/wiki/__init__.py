"""Wikimedia XML dump extraction.

Turns a ``*-pages-articles.xml.bz2`` dump into the same JSON Lines document
schema the scraper emits, so both feed one shared cleaning pipeline.

Dumps are the highest-value manual source in this project: one file, no rate
limits, no robots.txt, and parsing the wikitext ourselves is exactly the
"collect and clean it yourself" work the 20% manual requirement is about.
"""
