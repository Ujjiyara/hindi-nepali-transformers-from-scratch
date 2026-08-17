"""Tokenizer training and evaluation.

One tokenizer per language, trained from scratch on that language's TRAIN split
only. Vocabularies are never shared: Model H and Model L get independent
vocabularies, and the two may differ in size.

Vocabulary size is chosen empirically by sweeping candidates and comparing
fertility (tokens per word) on held-out text -- see :mod:`common.tokenizer.evaluate`.
"""
