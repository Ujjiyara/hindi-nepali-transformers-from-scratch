"""Tests for the reasoning finetune: masking, data order, and resume.

The two properties that matter most are the ones that fail silently. If the
prompt is not masked, the finetune still trains and the loss still falls --
it just learns the wrong thing. If resume is not exact, an interrupted finetune
still finishes -- on a different trajectory. Both are checked for equality,
not plausibility.
"""

from __future__ import annotations

import json
import random

import pytest
import torch

from common.finetune.data import (
    EncodedExample,
    batch_indices,
    collate,
    training_batch,
)
from common.finetune.loop import evaluate_answers, finetune
from common.model.config import GPTConfig
from common.model.gpt import GPT
from common.train.checkpoint import TrainingState, save_checkpoint
from common.train.loop import MetricLog, TrainConfig
from common.train.schedule import ScheduleConfig

PAD = 0
SMALL = GPTConfig(vocab_size=64, block_size=24, n_layer=2, n_head=2, d_model=16,
                  dropout=0.1)


def _examples(count: int, seed: int = 0) -> list[EncodedExample]:
    """Random prompts of varying length with a short answer and EOS (id 3)."""
    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        prompt = [rng.randrange(4, 64) for _ in range(rng.randrange(5, 15))]
        answer = [rng.randrange(4, 64) for _ in range(rng.randrange(1, 3))]
        examples.append(EncodedExample(tuple(prompt + answer + [3]), len(prompt)))
    return examples


def test_only_answer_and_eos_targets_are_supervised() -> None:
    examples = _examples(6)
    x, y = collate(examples, PAD)
    for row, example in enumerate(examples):
        supervised = (y[row] != -1).nonzero().flatten().tolist()
        answer_length = len(example.ids) - example.prompt_length  # answer + EOS
        # The first supervised position predicts the first answer token from
        # the last prompt token.
        assert supervised == list(range(example.prompt_length - 1,
                                        example.prompt_length - 1 + answer_length))
        assert y[row, supervised].tolist() == list(example.ids[example.prompt_length:])
        assert x[row, : len(example.ids) - 1].tolist() == list(example.ids[:-1])


def test_padding_never_changes_the_real_positions() -> None:
    """Right padding plus a causal mask: batching must not alter any logit."""
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    examples = _examples(4)
    x, _ = collate(examples, PAD)
    batched = model(x).logits
    for row, example in enumerate(examples):
        alone, _ = collate([example], PAD)
        n = alone.size(1)
        torch.testing.assert_close(batched[row, :n], model(alone).logits[0])


def test_each_epoch_visits_every_example_once() -> None:
    n, batch = 10, 3
    stream = [i for step in range(10) for i in batch_indices(n, batch, 7, step)]
    for epoch in range(3):
        assert sorted(stream[epoch * n : (epoch + 1) * n]) == list(range(n))
    assert batch_indices(n, batch, 7, 4) == batch_indices(n, batch, 7, 4)


def test_teacher_forced_exact_match_is_one_for_a_memorised_answer() -> None:
    """A model whose logits always favour the right token scores 1.0."""
    examples = _examples(5)

    class Oracle(torch.nn.Module):
        training = False

        def forward(self, x):
            logits = torch.full((*x.shape, 64), -10.0)
            for row, example in enumerate(examples):
                for t, token in enumerate(example.ids[1:]):
                    logits[row, t, token] = 10.0
            return type("Out", (), {"logits": logits})()

        def eval(self):
            return self

        def train(self, mode=True):
            return self

    result = evaluate_answers(Oracle(), examples, 8, PAD, torch.device("cpu"),
                              torch.float32)
    assert result["exact_match"] == 1.0
    assert result["loss"] < 1e-3


@pytest.fixture
def pretrained(tmp_path):
    """A tiny 'pretrained' checkpoint for the finetune to start from."""
    torch.manual_seed(1)
    model = GPT(SMALL)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "pretrained" / "best.pt"
    save_checkpoint(path, model, optimizer, None, TrainingState(step=100), SMALL,
                    ScheduleConfig(max_steps=100, warmup_steps=1))
    return path


def _run(pretrained, directory, stop_at_step=None):
    cfg = TrainConfig(batch_size=4, grad_accum=2, max_steps=6, eval_interval=100,
                      eval_batches=1, checkpoint_interval=100, keep_checkpoints=2,
                      log_interval=1, seed=5, device="cpu", dtype="float32")
    schedule = ScheduleConfig(max_lr=1e-3, min_lr=1e-4, warmup_steps=2, max_steps=6)
    finetune(pretrained, cfg, schedule, _examples(20), _examples(8, seed=1), PAD,
             directory / "checkpoints", MetricLog(directory / "metrics.jsonl"),
             stop_at_step=stop_at_step)
    records = [json.loads(line) for line in
               (directory / "metrics.jsonl").read_text().splitlines()]
    return [r["train_loss"] for r in records if "train_loss" in r]


def test_finetune_starts_from_the_pretrained_weights(pretrained, tmp_path) -> None:
    """Step 0 of the log is the pretrained model, not a random one."""
    _run(pretrained, tmp_path / "run", stop_at_step=1)
    first = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[0])
    assert first["step"] == 0 and "val_loss" in first

    torch.manual_seed(1)
    reference = GPT(SMALL).eval()
    val = _examples(8, seed=1)
    expected = evaluate_answers(reference, val, 4, PAD, torch.device("cpu"),
                                torch.float32, max_batches=1)
    assert first["val_loss"] == pytest.approx(expected["loss"])


def test_interrupted_finetune_matches_an_uninterrupted_one(pretrained, tmp_path) -> None:
    straight = _run(pretrained, tmp_path / "straight")

    first = _run(pretrained, tmp_path / "resumed", stop_at_step=3)
    torch.manual_seed(999)  # the session dies; the new one starts elsewhere
    both = _run(pretrained, tmp_path / "resumed")

    assert len(first) == 3
    assert both == straight
