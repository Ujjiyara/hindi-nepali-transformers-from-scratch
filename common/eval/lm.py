"""Intrinsic language-modelling metrics: cross-entropy, perplexity, bits-per-byte.

**Why perplexity alone cannot compare Model H with Model L.** Perplexity is
per *token*, and a token is not the same amount of language in the two models.
Phase 1 measured 3.86 characters per token for Hindi and 4.67 for Nepali, so a
Nepali token carries about 21% more text. Two models with identical perplexity
are therefore not equally good: the one whose tokens are larger is predicting
more language for the same score. Since the brief requires the two tokenizers
to be independent, this is not an artefact that can be tuned away.

**Bits-per-byte removes the tokenizer from the comparison.** The total negative
log-likelihood of a passage is a property of the model and the passage, not of
how the passage was chopped up; dividing it by the passage's size in UTF-8
bytes rather than in tokens gives a number the two models can be compared on::

    BPB = total_NLL_in_nats / (ln 2 * bytes)

Two details in that denominator, both stated here because they affect the
number and must be the same for both models:

1. The byte count is measured at packing time on exactly the text that was
   encoded, and recorded in ``_stats_pack.json``. It is scaled by the fraction
   of the split actually scored, because evaluation drops the final partial
   window.
2. The end-of-document token is scored like any other -- predicting where a
   document ends is real modelling -- but contributes no bytes. That inflates
   BPB by the EOS share of all tokens: 0.16% for Hindi, 0.24% for Nepali. It is
   left in rather than special-cased, since removing it would make the
   likelihood no longer that of the sequence the model was trained on.

Devanagari characters are three bytes each in UTF-8, so bits-per-byte here is
roughly a third of bits-per-character. That is a constant factor and does not
affect the comparison.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from common.data.loader import TokenArray, evaluation_windows
from common.model.gpt import GPT

log = logging.getLogger(__name__)


@dataclass
class LanguageModelMetrics:
    """Held-out language-modelling results for one model on one split.

    Attributes:
        split: Which split these came from.
        tokens_scored: Predictions actually made.
        bytes_scored: UTF-8 bytes those predictions cover.
        total_nll_nats: Summed negative log-likelihood, in nats.
        cross_entropy: Nats per token -- the training objective's own units.
        perplexity: ``exp(cross_entropy)``. Comparable only within a language.
        bits_per_byte: Comparable *across* the two languages.
    """

    split: str
    tokens_scored: int
    bytes_scored: int
    total_nll_nats: float
    cross_entropy: float
    perplexity: float
    bits_per_byte: float

    def as_dict(self) -> dict:
        """Plain dict, for the results JSON the report is generated from."""
        return asdict(self)


@torch.no_grad()
def score_split(
    model: GPT,
    tokens: TokenArray,
    split_stats: dict,
    split: str,
    batch_size: int,
    block_size: int,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.float32,
    max_batches: int | None = None,
) -> LanguageModelMetrics:
    """Score a whole split and return its metrics.

    Args:
        model: The model to evaluate. Put into eval mode here, so dropout is
            off and the numbers are deterministic.
        tokens: Packed token array for the split.
        split_stats: That split's entry from ``_stats_pack.json``, which
            supplies the UTF-8 byte count.
        split: Split name, recorded in the result.
        batch_size: Windows per forward pass. Logits are
            ``(B, T, vocab_size)`` and dominate memory here, so this is
            usually smaller than the training batch.
        block_size: Sequence length.
        device: Where to run.
        autocast_dtype: Mixed-precision dtype. The likelihood sum itself is
            accumulated in Python floats (doubles) regardless, because
            summing millions of fp16 values loses real precision.
        max_batches: Stop early. For smoke tests; the reported numbers run
            over the whole split.

    Returns:
        A :class:`LanguageModelMetrics`.
    """
    model.eval()
    total_nll = 0.0
    scored = 0

    for x, y in evaluation_windows(tokens, batch_size, block_size, max_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=device.type == "cuda"):
            logits = model(x).logits
        # reduction="sum" rather than the mean: the mean over a batch weights
        # every batch equally, and the final partial batch is smaller.
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
            reduction="sum",
        ).item()
        scored += y.numel()

    if scored == 0:
        raise ValueError(f"{tokens.path}: nothing to score")

    # The split's bytes, scaled by the fraction of it that was scored.
    fraction = scored / split_stats["tokens"]
    bytes_scored = int(round(split_stats["utf8_bytes"] * fraction))

    cross_entropy = total_nll / scored
    metrics = LanguageModelMetrics(
        split=split,
        tokens_scored=scored,
        bytes_scored=bytes_scored,
        total_nll_nats=total_nll,
        cross_entropy=cross_entropy,
        perplexity=math.exp(cross_entropy),
        bits_per_byte=total_nll / (math.log(2) * bytes_scored),
    )
    log.info(
        "%s: CE %.4f nats/token, PPL %.2f, BPB %.4f (%d tokens, %d bytes)",
        split, metrics.cross_entropy, metrics.perplexity,
        metrics.bits_per_byte, scored, bytes_scored,
    )
    return metrics
