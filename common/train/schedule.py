"""Learning-rate schedule: linear warmup into a cosine decay.

**Why warmup.** AdamW divides each update by a running estimate of the gradient's
second moment. For the first few hundred steps that estimate is built from
almost no samples and is badly wrong, so a full-size learning rate then takes
large, poorly scaled steps and can destroy the initialisation before training
has begun. Ramping the rate up over the first ~1-2% of steps lets the moment
estimates settle first.

**Why cosine decay.** A high rate early explores; a low rate late lets the
model settle into a minimum instead of bouncing around it. Cosine spends
comparatively longer at high rates than a linear decay does and arrives at the
floor smoothly, with no discontinuity for the optimiser to react to.

The rate is a **pure function of the step**, which is why nothing schedule-shaped
has to be serialised for a resume to be exact: the step, plus this config, is
the scheduler state. It is written into every checkpoint for that reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleConfig:
    """Shape of the learning-rate curve.

    Attributes:
        max_lr: Peak rate, reached at the end of warmup.
        min_lr: Floor held from ``max_steps`` onwards. Roughly ``max_lr / 10``
            is conventional; a floor of exactly zero stops learning entirely if
            the run is extended past its planned length.
        warmup_steps: Optimiser steps spent ramping linearly from ~0 to
            ``max_lr``.
        max_steps: Step at which the cosine reaches ``min_lr``.
    """

    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    max_steps: int = 16_000

    def __post_init__(self) -> None:
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be shorter than max_steps")
        if self.min_lr > self.max_lr:
            raise ValueError("min_lr must not exceed max_lr")


def learning_rate(step: int, cfg: ScheduleConfig) -> float:
    """Rate for a given optimiser step.

    Args:
        step: Zero-based optimiser step (not micro-step).
        cfg: The curve.

    Returns:
        The learning rate to set on every parameter group before this step.
    """
    if step < cfg.warmup_steps:
        # step + 1 so the very first step is not taken at a rate of zero.
        return cfg.max_lr * (step + 1) / cfg.warmup_steps

    if step >= cfg.max_steps:
        return cfg.min_lr

    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))
