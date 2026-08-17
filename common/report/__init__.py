"""Report generation.

The Phase 1 report is generated from the ``_stats.json`` and ``sweep.json``
files the pipeline stages write, rather than typed by hand. Every number in the
report is therefore a measurement, and regenerating after a re-run cannot leave
a stale figure behind.
"""
