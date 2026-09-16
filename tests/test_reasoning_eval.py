"""Tests for the reasoning evaluation.

Batched decoding over right-padded prompts is only valid if it gives exactly
what decoding each prompt alone gives, so that is checked directly, as is the
candidate scoring against a hand-computed log-probability.
"""

from __future__ import annotations

import random

import torch

from common.eval.generate import generate
from common.finetune.evaluate import choice_scores, greedy_answers, is_prefix_match
from common.model.config import GPTConfig
from common.model.gpt import GPT

CFG = GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_head=2, d_model=16)


class FakeSP:
    """Just enough of a SentencePiece processor: ids are written as digits."""

    def eos_id(self) -> int:
        return 3

    def pad_id(self) -> int:
        return 0

    def encode(self, text: str, out_type=int) -> list[int]:
        return [int(t) for t in text.split()]


def _model() -> GPT:
    torch.manual_seed(3)
    return GPT(CFG).eval()


def _prompts() -> list[list[int]]:
    rng = random.Random(0)
    return [[rng.randrange(4, 40) for _ in range(rng.randrange(3, 12))] for _ in range(7)]


def test_batched_greedy_equals_one_prompt_at_a_time() -> None:
    model, sp = _model(), FakeSP()
    prompts = _prompts()
    batched = greedy_answers(model, sp, prompts, torch.device("cpu"),
                             batch_size=4, max_new_tokens=6)
    for prompt, output in zip(prompts, batched):
        alone = generate(model, torch.tensor([prompt]), 6, temperature=0.0)[0]
        expected = []
        for token in alone[len(prompt):].tolist():
            if token == sp.eos_id():
                break
            expected.append(token)
        assert output == expected


def test_choice_score_is_the_summed_log_probability() -> None:
    model, sp = _model(), FakeSP()
    prompt = [5, 6, 7]
    (scores,) = choice_scores(model, sp, [prompt], [["8 9", "10"]], torch.device("cpu"))

    with torch.no_grad():
        log_probs = torch.log_softmax(model(torch.tensor([prompt + [8, 9]])).logits, -1)[0]
        single = torch.log_softmax(model(torch.tensor([prompt])).logits, -1)[0, 2, 10]
    expected = float(log_probs[2, 8] + log_probs[3, 9])
    assert abs(scores[0] - expected) < 1e-5
    assert abs(scores[1] - float(single)) < 1e-5


def test_prefix_match_requires_a_whole_word() -> None:
    assert is_prefix_match("राम", "राम")
    assert is_prefix_match("राम है।", "राम")
    assert not is_prefix_match("रामको", "राम")
    assert not is_prefix_match("श्याम", "राम")
