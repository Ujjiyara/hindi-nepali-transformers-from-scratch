"""Overlap and diversity metrics for generated continuations.

**These metrics are weak evidence for open-ended generation, and the brief asks
that to be said explicitly.** A prompt has many valid continuations; the test
set contains one. n-gram precision against that single reference scores a
fluent-but-different continuation exactly as harshly as gibberish, so BLEU on
this task measures agreement with one arbitrary sample from the space of good
answers, not quality. The numbers are still reported, and still informative in
one direction: they can separate a model that has learned the language from one
that has not, and they can be compared *between* Model H and Model L, which is
the comparison this project is about.

Of the three, **chrF is the most meaningful here**. It matches character
n-grams rather than whole words, so a correctly inflected stem with the wrong
case ending still earns partial credit. Hindi and Nepali are morphologically
rich and agglutinate case and postposition markers onto stems, so word-level
BLEU is nearly all-or-nothing per token in a way it is not for English.

**ROUGE-L is implemented here rather than taken from ``rouge_score``.** That
library's tokenizer applies ``[^a-z0-9]+ -> " "`` to its input, which deletes
every Devanagari character and returns an empty token list, silently scoring
every pair as zero. BLEU and chrF come from ``sacrebleu``, whose ``13a``
tokenizer only inserts spaces around ASCII punctuation and leaves Devanagari
untouched -- verified in ``tests/test_text_metrics.py``.

Hindi and Nepali both delimit words with spaces, so whitespace tokenization is
appropriate. The danda (``।``) stays attached to its preceding word under 13a;
that is a small, equal effect on both models.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass

import sacrebleu

log = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Quality and diversity of one model's generations at one temperature.

    Attributes:
        temperature: Decoding temperature; 0.0 denotes greedy.
        samples: Number of continuations scored.
        bleu: Corpus BLEU-4 against the held-out continuations.
        chrf: chrF2 -- character n-gram F-score.
        rouge_l: Mean sentence-level ROUGE-L F1.
        repetition_rate: Mean per-sample fraction of repeated 4-grams. A model
            stuck in a loop approaches 1.
        distinct_1: Unique unigrams over all unigrams, pooled across samples.
        distinct_2: The same for bigrams. Low values with low repetition_rate
            mean the samples are individually varied but collectively same-y.
        mean_tokens: Mean whitespace-token length of a continuation.
    """

    temperature: float
    samples: int
    bleu: float
    chrf: float
    rouge_l: float
    repetition_rate: float
    distinct_1: float
    distinct_2: float
    mean_tokens: float

    def as_dict(self) -> dict:
        """Plain dict, for the results JSON the report reads."""
        return asdict(self)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Every contiguous n-gram of a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Rolling one row of the dynamic-programming table rather than the whole
    matrix: continuations are short, but this is called once per pair per
    temperature per model and there is no reason to hold O(n*m).
    """
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b, start=1):
            if token == other:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    """Sentence-level ROUGE-L F1 over whitespace tokens.

    The longest common subsequence rewards words appearing in the right order
    without requiring them to be adjacent, which is why it is less brittle than
    BLEU's contiguous n-gram matching on free-form text.
    """
    hyp, ref = hypothesis.split(), reference.split()
    if not hyp or not ref:
        return 0.0

    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0

    precision, recall = lcs / len(hyp), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def repetition_rate(texts: list[str], n: int = 4) -> float:
    """Mean per-sample fraction of n-grams that are not unique.

    Computed per sample and then averaged, rather than pooled: pooling would
    count the same phrase appearing in two different continuations as
    repetition, which is a property of the prompt set, not of degeneration
    within a sample.
    """
    rates = []
    for text in texts:
        grams = _ngrams(text.split(), n)
        if grams:
            rates.append(1.0 - len(set(grams)) / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


def distinct_n(texts: list[str], n: int) -> float:
    """Unique n-grams over total n-grams, pooled across all samples.

    Pooled deliberately: this is the standard definition, and it is the measure
    that catches a model producing the same safe continuation for every prompt.
    """
    total: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        total.update(_ngrams(text.split(), n))
    count = sum(total.values())
    return len(total) / count if count else 0.0


def score_generations(
    hypotheses: list[str], references: list[str], temperature: float
) -> GenerationMetrics:
    """Score a set of continuations against their held-out references.

    Args:
        hypotheses: Generated continuations.
        references: True continuations, in the same order.
        temperature: Recorded on the result; 0.0 means greedy.

    Returns:
        A :class:`GenerationMetrics`.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypotheses against {len(references)} references"
        )
    if not hypotheses:
        raise ValueError("nothing to score")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    rouge = sum(rouge_l(h, r) for h, r in zip(hypotheses, references))

    metrics = GenerationMetrics(
        temperature=temperature,
        samples=len(hypotheses),
        bleu=bleu.score,
        chrf=chrf.score,
        rouge_l=rouge / len(hypotheses),
        repetition_rate=repetition_rate(hypotheses),
        distinct_1=distinct_n(hypotheses, 1),
        distinct_2=distinct_n(hypotheses, 2),
        mean_tokens=sum(len(h.split()) for h in hypotheses) / len(hypotheses),
    )
    log.info(
        "T=%.1f  BLEU %.2f  chrF %.2f  ROUGE-L %.3f  rep4 %.3f  d1 %.3f  d2 %.3f",
        temperature, metrics.bleu, metrics.chrf, metrics.rouge_l,
        metrics.repetition_rate, metrics.distinct_1, metrics.distinct_2,
    )
    return metrics
