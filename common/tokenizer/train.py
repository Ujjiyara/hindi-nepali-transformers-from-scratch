"""Training SentencePiece tokenizers from scratch.

Pretrained tokenizers are not permitted, and none is used: every model here is
fitted on this project's own text. The library only supplies the training
algorithm.

**Trained on the TRAIN split only.** Fitting the vocabulary on validation or
test text leaks held-out information into the model -- the tokenizer would be
built knowing exactly which subwords appear in the evaluation data, making
held-out perplexity optimistic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import sentencepiece as spm

log = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Settings for one tokenizer.

    Attributes:
        vocab_size: Target vocabulary size, the parameter being swept.
        model_type: ``unigram`` or ``bpe``. Unigram is the default: its
            probabilistic segmentation tends to give lower fertility than BPE's
            greedy merges on morphologically rich scripts.
        character_coverage: Fraction of characters given their own vocabulary
            entry. Indic scripts have a long tail of rare characters; at 1.0
            every stray foreign character permanently consumes a slot, so a
            little below 1.0 is better and the remainder is handled by byte
            fallback.
        byte_fallback: Decompose unseen characters into UTF-8 bytes instead of
            emitting <unk>. This makes the unknown-token rate zero by
            construction; the meaningful diagnostic becomes the byte-fallback
            rate reported by the evaluator.
    """

    vocab_size: int
    model_type: str = "unigram"
    character_coverage: float = 0.9995
    byte_fallback: bool = True
    input_sentence_size: int = 3_000_000
    shuffle_input_sentence: bool = True
    max_sentence_length: int = 16384
    num_threads: int = 4
    extra_symbols: list[str] = field(default_factory=list)

    def as_kwargs(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "model_type": self.model_type,
            "character_coverage": self.character_coverage,
            "byte_fallback": self.byte_fallback,
            "input_sentence_size": self.input_sentence_size,
            "shuffle_input_sentence": self.shuffle_input_sentence,
            "max_sentence_length": self.max_sentence_length,
            "num_threads": self.num_threads,
            # Explicit ids keep the special tokens stable across languages so
            # the Phase 2 model code does not need per-language handling.
            "pad_id": 0,
            "unk_id": 1,
            "bos_id": 2,
            "eos_id": 3,
            "user_defined_symbols": self.extra_symbols,
        }


def train_tokenizer(text_path: Path, model_prefix: Path, cfg: TokenizerConfig) -> Path:
    """Train one tokenizer; returns the path to the written ``.model`` file."""
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "training %s vocab=%d on %s", cfg.model_type, cfg.vocab_size, text_path.name
    )
    spm.SentencePieceTrainer.train(
        input=str(text_path), model_prefix=str(model_prefix), **cfg.as_kwargs()
    )
    return model_prefix.with_suffix(".model")
