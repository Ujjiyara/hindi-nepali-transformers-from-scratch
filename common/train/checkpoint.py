"""Saving and restoring a run, so an interrupted job continues rather than restarts.

The brief makes this mandatory: a checkpoint must carry model weights,
optimizer state, scheduler state, the current step and the configuration, and a
run that cannot resume loses marks. Free Kaggle sessions are killed routinely,
so this is also simply how the training actually gets done.

**What is stored, and why each piece is needed.**

- ``model``: the weights. Obvious, and on its own useless.
- ``optimizer``: AdamW's first and second moment estimates for every parameter.
  Dropping these restarts the moment estimates from zero, which behaves like an
  unwarmed run and puts a visible kink in the loss curve.
- ``scheduler``: the schedule's shape. The rate itself is a pure function of the
  step (see :mod:`common.train.schedule`), so the step plus this config *is* the
  scheduler state; storing the config makes that self-contained rather than
  dependent on the caller passing the same flags again.
- ``scaler``: fp16 loss-scaling state. On a T4 the loss is scaled to keep small
  gradients representable, and the scale is adapted over time; restarting it at
  the default causes a burst of skipped steps after every resume.
- ``state``: step, micro-step, tokens seen, best validation loss, and the data
  seed. The micro-step is what the loader turns back into an exact position in
  the data stream.
- ``rng``: torch's generator states, so dropout masks continue as they would
  have. Without this a resume is statistically equivalent but not identical,
  and the resume test below could then only check "close", not "equal".
- ``model_config``: the geometry, so a checkpoint can be rebuilt into a model
  without the config file that produced it.

Writes are atomic -- to a temporary file, then renamed -- because a session
killed midway through a 290MB write would otherwise leave a truncated file
where the newest checkpoint should be.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from common.model.config import GPTConfig
from common.train.schedule import ScheduleConfig

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
_STEP_PATTERN = re.compile(r"step_(\d+)\.pt$")


@dataclass
class TrainingState:
    """Where a run has got to.

    Attributes:
        step: Optimiser steps completed.
        micro_step: Forward/backward passes consumed. With gradient
            accumulation this is ``step * accumulation``, and it is the value
            the data loader turns into a position in the token stream.
        tokens_seen: Training tokens consumed, for the report's x-axis.
        best_val_loss: Lowest validation loss seen, for keeping ``best.pt``.
        seed: Run seed, which together with ``micro_step`` determines the data.
    """

    step: int = 0
    micro_step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    seed: int = 1337


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    state: TrainingState,
    model_cfg: GPTConfig,
    schedule_cfg: ScheduleConfig,
    extra: dict | None = None,
) -> Path:
    """Write a complete, resumable snapshot atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": asdict(schedule_cfg),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "state": asdict(state),
        "rng": {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    log.info("saved %s (step %d)", path, state.step)
    return path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> TrainingState:
    """Restore a snapshot in place and return where the run had got to.

    Args:
        path: Checkpoint file.
        model: Model to load weights into; must already have the right shape.
        optimizer: Optional. Omitted when only evaluating a checkpoint.
        scaler: Optional fp16 scaler.
        map_location: Where tensors should land.
        restore_rng: Restore generator states. Off when loading a checkpoint
            for evaluation, where continuing the training RNG is meaningless.

    Returns:
        The :class:`TrainingState` the run was at.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: checkpoint format {version}, expected {FORMAT_VERSION}"
        )

    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])

    if restore_rng and payload.get("rng"):
        torch.set_rng_state(payload["rng"]["cpu"].cpu())
        if torch.cuda.is_available() and payload["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(
                [s.cpu() for s in payload["rng"]["cuda"]]
            )

    state = TrainingState(**payload["state"])
    log.info("resumed %s at step %d (%d tokens seen)", path, state.step,
             state.tokens_seen)
    return state


def load_config(path: Path, map_location: str = "cpu") -> tuple[GPTConfig, dict]:
    """Read the geometry and bookkeeping out of a checkpoint without a model."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return GPTConfig(**payload["model_config"]), payload["state"]


def find_latest(directory: Path) -> Path | None:
    """Highest-numbered ``step_*.pt`` in a directory, or None if there is none.

    Resume works off this rather than a ``latest.pt`` copy: duplicating a 290MB
    file on every save is wasted I/O, and a symlink does not survive being
    packaged into a Kaggle dataset.
    """
    if not directory.exists():
        return None
    checkpoints = [
        (int(m.group(1)), p)
        for p in directory.glob("step_*.pt")
        if (m := _STEP_PATTERN.search(p.name))
    ]
    return max(checkpoints)[1] if checkpoints else None


def prune(directory: Path, keep: int) -> list[Path]:
    """Delete all but the ``keep`` newest step checkpoints.

    ``best.pt`` is named differently and is never touched. Kaggle's working
    directory is capped at about 20GB, so unbounded checkpoints fill it and
    kill the session.
    """
    checkpoints = sorted(
        (
            (int(m.group(1)), p)
            for p in directory.glob("step_*.pt")
            if (m := _STEP_PATTERN.search(p.name))
        ),
        reverse=True,
    )
    removed = []
    for _, path in checkpoints[keep:]:
        path.unlink()
        removed.append(path)
        log.info("pruned %s", path)
    return removed
