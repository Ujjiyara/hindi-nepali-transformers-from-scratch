"""Tokenizer evaluation metrics, computed on held-out text.

The headline number is **fertility**: the average number of tokens produced per
whitespace-delimited word. Lower is better -- it means each token carries more
meaning, more real text fits in a fixed context window, and the model spends
fewer parameters describing the same content.

Fertility is the metric that exposes the resource gap between Model H and Model
L. A tokenizer fitted on less text learns fewer useful merges, so the same word
shatters into more pieces; the lower-resource model therefore sees less actual
language per context window even with an identical architecture.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import sentencepiece as spm

# SentencePiece writes byte-fallback pieces as <0xNN>.
_BYTE_PIECE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


@dataclass
class TokenizerMetrics:
    """Held-out measurements for one tokenizer."""

    vocab_size: int
    documents: int
    characters: int
    words: int
    tokens: int
    unknown_tokens: int
    byte_fallback_tokens: int
    unique_tokens_used: int
    top_tokens: list[tuple[str, int]] = field(default_factory=list)

    @property
    def fertility(self) -> float:
        """Tokens per whitespace-delimited word."""
        return self.tokens / self.words if self.words else 0.0

    @property
    def chars_per_token(self) -> float:
        """Average characters per token: the inverse view of fertility."""
        return self.characters / self.tokens if self.tokens else 0.0

    @property
    def unknown_rate(self) -> float:
        """Fraction of tokens that are <unk>; zero by construction with byte fallback."""
        return self.unknown_tokens / self.tokens if self.tokens else 0.0

    @property
    def byte_fallback_rate(self) -> float:
        """Fraction of tokens that are raw UTF-8 byte pieces rather than subwords."""
        return self.byte_fallback_tokens / self.tokens if self.tokens else 0.0

    @property
    def vocab_utilisation(self) -> float:
        """Fraction of the vocabulary that actually occurs in held-out text.

        A low value means the vocabulary is oversized for the corpus: many
        entries were learned from rare training contexts and will receive
        poorly estimated embeddings during pretraining.
        """
        return self.unique_tokens_used / self.vocab_size if self.vocab_size else 0.0

    def as_dict(self) -> dict:
        """The metrics as a plain dictionary, for sweep.json."""
        return {
            "vocab_size": self.vocab_size,
            "documents": self.documents,
            "characters": self.characters,
            "words": self.words,
            "tokens": self.tokens,
            "fertility": round(self.fertility, 4),
            "chars_per_token": round(self.chars_per_token, 4),
            "unknown_tokens": self.unknown_tokens,
            "unknown_rate": round(self.unknown_rate, 8),
            "byte_fallback_tokens": self.byte_fallback_tokens,
            "byte_fallback_rate": round(self.byte_fallback_rate, 8),
            "unique_tokens_used": self.unique_tokens_used,
            "vocab_utilisation": round(self.vocab_utilisation, 4),
            "top_tokens": self.top_tokens,
        }


def evaluate(
    model_path: str, texts: Iterable[str], top_k: int = 30
) -> TokenizerMetrics:
    """Measure a trained tokenizer on held-out documents."""
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    unk_id = processor.unk_id()

    documents = characters = words = tokens = 0
    unknown = byte_fallback = 0
    counts: Counter[str] = Counter()

    for text in texts:
        documents += 1
        characters += len(text)
        words += len(text.split())

        ids = processor.encode(text)
        pieces = processor.encode(text, out_type=str)
        tokens += len(ids)
        unknown += sum(1 for i in ids if i == unk_id)
        byte_fallback += sum(1 for p in pieces if _BYTE_PIECE_RE.match(p))
        counts.update(pieces)

    return TokenizerMetrics(
        vocab_size=processor.get_piece_size(),
        documents=documents,
        characters=characters,
        words=words,
        tokens=tokens,
        unknown_tokens=unknown,
        byte_fallback_tokens=byte_fallback,
        unique_tokens_used=len(counts),
        top_tokens=counts.most_common(top_k),
    )


def tokenization_examples(
    model_path: str, sentences: list[str], max_pieces: int = 40
) -> list[dict]:
    """Segment example sentences, for the qualitative table in the report."""
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    examples = []
    for sentence in sentences:
        pieces = processor.encode(sentence, out_type=str)
        examples.append(
            {
                "text": sentence,
                "n_tokens": len(pieces),
                "n_words": len(sentence.split()),
                "fertility": round(len(pieces) / max(len(sentence.split()), 1), 3),
                "pieces": pieces[:max_pieces],
            }
        )
    return examples
