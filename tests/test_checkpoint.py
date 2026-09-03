"""Tests for checkpoint save and resume.

The brief requires that a run survive being killed. "It loaded without an
error" is not evidence of that -- a checkpoint that restores weights but drops
the optimizer moments, the data position or the RNG will load cleanly and then
train differently. So the test below runs eight steps straight through, runs the
same eight with a save/kill/restore in the middle, and demands the two loss
sequences be *equal*, not merely close.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from common.data.loader import TokenArray, training_batch
from common.model.config import GPTConfig
from common.model.gpt import GPT
from common.train.checkpoint import (
    TrainingState,
    find_latest,
    load_checkpoint,
    load_config,
    prune,
    save_checkpoint,
)
from common.train.schedule import ScheduleConfig, learning_rate

SMALL = GPTConfig(
    vocab_size=257, block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.1
)
SCHEDULE = ScheduleConfig(max_lr=1e-3, min_lr=1e-4, warmup_steps=2, max_steps=8)
SEED = 99


@pytest.fixture
def tokens(tmp_path) -> TokenArray:
    """A small synthetic token stream."""
    rng = np.random.default_rng(0)
    path = tmp_path / "toy.bin"
    rng.integers(0, 257, size=20_000, dtype=np.uint16).tofile(path)
    return TokenArray(path)


def _fresh() -> tuple[GPT, torch.optim.Optimizer]:
    """A model and optimizer built from an identical starting point."""
    torch.manual_seed(SEED)
    model = GPT(SMALL).train()
    return model, torch.optim.AdamW(model.parameters(), lr=SCHEDULE.max_lr)


def _steps(model, optimizer, tokens, first, last) -> list[float]:
    """Run micro-steps [first, last) and return the loss at each."""
    losses = []
    for step in range(first, last):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, SCHEDULE)
        x, y = training_batch(tokens, 2, SMALL.block_size, SEED, step)
        loss = model(x, y).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_resume_reproduces_an_uninterrupted_run(tmp_path, tokens) -> None:
    """Eight steps straight equal four, a crash, and four more."""
    model, optimizer = _fresh()
    torch.manual_seed(SEED)
    straight = _steps(model, optimizer, tokens, 0, 8)

    model, optimizer = _fresh()
    torch.manual_seed(SEED)
    first_half = _steps(model, optimizer, tokens, 0, 4)
    path = save_checkpoint(
        tmp_path / "step_000004.pt", model, optimizer, None,
        TrainingState(step=4, micro_step=4, tokens_seen=4 * 2 * 16, seed=SEED),
        SMALL, SCHEDULE,
    )

    # The session dies. Everything below starts from a different random state.
    del model, optimizer
    torch.manual_seed(12345)
    revived = GPT(SMALL).train()
    revived_opt = torch.optim.AdamW(revived.parameters(), lr=SCHEDULE.max_lr)
    state = load_checkpoint(path, revived, revived_opt)

    assert state.step == 4 and state.micro_step == 4
    second_half = _steps(revived, revived_opt, tokens, state.micro_step, 8)

    assert first_half == straight[:4]
    assert second_half == straight[4:]


def test_checkpoint_carries_everything_the_brief_requires(tmp_path, tokens) -> None:
    """Weights, optimizer state, scheduler state, step and config."""
    model, optimizer = _fresh()
    _steps(model, optimizer, tokens, 0, 2)
    path = save_checkpoint(
        tmp_path / "step_000002.pt", model, optimizer, None,
        TrainingState(step=2, micro_step=2), SMALL, SCHEDULE,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) >= {
        "model", "optimizer", "scheduler", "state", "model_config"
    }
    assert payload["optimizer"]["state"], "optimizer moments were not saved"
    assert payload["scheduler"]["max_lr"] == SCHEDULE.max_lr
    assert payload["state"]["step"] == 2


def test_geometry_can_be_recovered_without_the_config_file(tmp_path) -> None:
    """A checkpoint is self-describing, so evaluation needs only the file."""
    model, optimizer = _fresh()
    path = save_checkpoint(
        tmp_path / "step_000000.pt", model, optimizer, None,
        TrainingState(), SMALL, SCHEDULE,
    )
    recovered, state = load_config(path)
    assert recovered == SMALL
    assert state["step"] == 0
    assert GPT(recovered).num_parameters() == model.num_parameters()


def test_find_latest_picks_the_highest_step(tmp_path) -> None:
    """Resume must not be fooled by lexicographic ordering."""
    assert find_latest(tmp_path) is None
    for step in (5, 40, 300):
        (tmp_path / f"step_{step:06d}.pt").touch()
    (tmp_path / "best.pt").touch()
    assert find_latest(tmp_path).name == "step_000300.pt"


def test_prune_keeps_the_newest_and_spares_best(tmp_path) -> None:
    """Kaggle's working directory is capped; old checkpoints must go."""
    for step in range(6):
        (tmp_path / f"step_{step:06d}.pt").touch()
    (tmp_path / "best.pt").touch()

    prune(tmp_path, keep=2)

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["best.pt", "step_000004.pt", "step_000005.pt"]


def test_format_mismatch_is_refused(tmp_path) -> None:
    """Better a clear error than silently loading an incompatible file."""
    torch.save({"format_version": 99, "model": {}}, tmp_path / "old.pt")
    model, _ = _fresh()
    with pytest.raises(ValueError, match="checkpoint format 99"):
        load_checkpoint(tmp_path / "old.pt", model)
