"""Tests for generation-quality metrics.

Two things are being pinned down. First that the arithmetic is right, checked
against values computable by hand. Second -- and the reason this file exists at
all -- that the metrics survive Devanagari, since the obvious library choice for
ROUGE-L silently returns zero on it.
"""

from __future__ import annotations

import pytest

from common.eval.text_metrics import (
    distinct_n,
    repetition_rate,
    rouge_l,
    score_generations,
)

HI = "यह एक परीक्षण वाक्य है"
NE = "यो एउटा परीक्षण वाक्य हो"


def test_rouge_l_is_one_for_an_exact_match() -> None:
    assert rouge_l(HI, HI) == pytest.approx(1.0)


def test_rouge_l_is_zero_when_nothing_overlaps() -> None:
    assert rouge_l("क ख ग", "घ ङ च") == 0.0
    assert rouge_l("", HI) == 0.0


def test_rouge_l_matches_a_hand_computed_value() -> None:
    """LCS of 'a b c d' and 'a c e' is 'a c', length 2.

    precision 2/4, recall 2/3, F1 = 2 * 0.5 * 0.667 / (0.5 + 0.667) = 0.5714.
    """
    assert rouge_l("a b c d", "a c e") == pytest.approx(0.5714, abs=1e-4)


def test_rouge_l_rewards_order_without_requiring_adjacency() -> None:
    """A gapped subsequence counts, but F1 still charges for the extra words.

    LCS of "क ख ग घ" and "क ग" is 2: recall is perfect, precision is 2/4, so
    F1 is 2/3. Reversing the hypothesis breaks the ordering and drops the LCS
    to 1, which is the property being tested -- ROUGE-L is order-sensitive but
    tolerates gaps.
    """
    assert rouge_l("क ख ग घ", "क ग") == pytest.approx(2 / 3)
    assert rouge_l("घ ग ख क", "क ग") == pytest.approx(1 / 3)


def test_repetition_rate_detects_a_loop() -> None:
    """A model stuck repeating a phrase approaches 1; varied text approaches 0."""
    looped = " ".join(["क ख ग घ"] * 8)
    assert repetition_rate([looped]) > 0.7
    assert repetition_rate(["क ख ग घ ङ च छ ज झ ञ ट ठ"]) == 0.0


def test_repetition_rate_is_per_sample_not_pooled() -> None:
    """Two identical samples each repeat nothing internally."""
    assert repetition_rate(["क ख ग घ ङ", "क ख ग घ ङ"]) == 0.0


def test_distinct_n_pools_across_samples() -> None:
    """Same continuation for every prompt must show up as low diversity."""
    varied = distinct_n(["क ख ग", "घ ङ च"], 1)
    identical = distinct_n(["क ख ग", "क ख ग"], 1)
    assert varied == pytest.approx(1.0)
    assert identical == pytest.approx(0.5)


def test_metrics_do_not_collapse_on_devanagari() -> None:
    """The whole point: an exact match must score full marks in both scripts.

    rouge_score's tokenizer would return an empty token list here and score 0.
    """
    for text in (HI, NE):
        metrics = score_generations([text], [text], temperature=0.0)
        assert metrics.bleu == pytest.approx(100.0)
        assert metrics.chrf == pytest.approx(100.0)
        assert metrics.rouge_l == pytest.approx(1.0)


def test_chrf_gives_partial_credit_where_word_bleu_gives_none() -> None:
    """The reason chrF is the metric to trust on these two languages.

    Every content word here is the correct stem with a different inflection.
    Word-level BLEU scores that as a total miss; chrF matches the shared
    character n-grams and awards partial credit. This is not a claim that chrF
    always exceeds BLEU -- on two unrelated sentences it is usually lower,
    because character n-grams overlap less than a stray shared function word.
    """
    reference = "परीक्षण वाक्य लिखा गया"
    inflected = "परीक्षण वाक्यों लिखे गए"
    metrics = score_generations([inflected], [reference], temperature=1.0)
    # One of four words matches exactly, so BLEU is low but not zero.
    assert metrics.bleu < 25.0
    assert metrics.chrf > 55.0
    assert metrics.chrf > 3 * metrics.bleu


def test_mismatched_input_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="against"):
        score_generations([HI, HI], [HI], temperature=1.0)
    with pytest.raises(ValueError, match="nothing to score"):
        score_generations([], [], temperature=1.0)
