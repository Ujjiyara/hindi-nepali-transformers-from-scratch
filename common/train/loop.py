"""The pretraining loop: AdamW, gradient accumulation, mixed precision, resume.

Model H and Model L run this same code over their own data with their own
config. Nothing is shared between them -- not weights, not vocabulary, not the
token arrays -- and neither is ever initialised from the other.

**Gradient accumulation.** A 16GB T4 holds a micro-batch of about 16 sequences
of 512 tokens, which is 8k tokens -- too small a batch for a stable gradient
estimate. Running several micro-batches and stepping once on their summed
gradient gives the statistics of a large batch at the memory cost of a small
one. The loss of each micro-batch is divided by the accumulation count so that
the accumulated gradient is a mean and not a sum, which keeps the effective
learning rate independent of the accumulation setting.

**Mixed precision.** T4s (Turing) have fp16 tensor cores but no bfloat16, so
fp16 is used with a gradient scaler: gradients are computed on a scaled-up loss
to keep small values from flushing to zero in fp16, then unscaled before
clipping and the optimizer step. Newer cards get bf16, which has fp32's
exponent range and needs no scaler at all. ``dtype: auto`` picks whichever the
card supports.

**Weight decay on matrices only.** Decay is applied to parameters with two or
more dimensions -- the projections and embeddings -- and not to biases or
LayerNorm gains. Pulling a LayerNorm gain towards zero fights the very thing
the layer exists to do, and biases carry no capacity worth regularising.

**Gradient clipping** by global norm bounds how far any single step can move,
which matters most early on, when a rare batch can otherwise produce a gradient
large enough to undo the warmup.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from common.data.loader import TokenArray, evaluation_windows, training_batch
from common.model.config import GPTConfig
from common.model.gpt import GPT
from common.train.checkpoint import (
    TrainingState,
    find_latest,
    load_checkpoint,
    prune,
    save_checkpoint,
)
from common.train.schedule import ScheduleConfig, learning_rate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Everything about a pretraining run that is not the model's geometry.

    Attributes:
        batch_size: Sequences per micro-batch, bounded by GPU memory.
        grad_accum: Micro-batches summed into one optimiser step. The effective
            batch is ``batch_size * grad_accum * block_size`` tokens.
        max_steps: Optimiser steps to run. One epoch is
            ``training_tokens / (batch_size * grad_accum * block_size)``.
        weight_decay: Applied to 2-D parameters only.
        beta1, beta2: AdamW moments. 0.95 for the second moment rather than
            torch's 0.999 is the usual language-modelling choice: it forgets
            stale curvature faster on noisy gradients.
        grad_clip: Global-norm clip. 0 disables.
        eval_interval: Optimiser steps between validation checks.
        eval_batches: Validation batches per check. The full split is only
            scored at the end, by the evaluation stage.
        checkpoint_interval: Steps between checkpoints. Set so a lost session
            costs minutes, not hours.
        keep_checkpoints: Step checkpoints retained on disk, newest first.
        log_interval: Steps between progress lines.
        seed: Determines initialisation and the data stream.
        device: ``auto``, ``cuda``, ``cuda:1``, or ``cpu``.
        dtype: ``auto``, ``bfloat16``, ``float16`` or ``float32``.
    """

    batch_size: int = 16
    grad_accum: int = 4
    max_steps: int = 16_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_batches: int = 40
    checkpoint_interval: int = 1000
    keep_checkpoints: int = 2
    log_interval: int = 20
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"

    def tokens_per_step(self, block_size: int) -> int:
        """Training tokens consumed by one optimiser step."""
        return self.batch_size * self.grad_accum * block_size


def load_run_config(path: str | Path) -> tuple[TrainConfig, ScheduleConfig]:
    """Read a per-language ``train.yaml`` into its two config objects.

    The file holds run settings at the top level and the learning-rate curve
    under a ``schedule`` key. Unknown keys anywhere are an error: a typo would
    otherwise silently leave a hyperparameter at its default while the config
    file on the branch claims otherwise, which is exactly the kind of drift the
    brief's reproducibility requirement is about.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    schedule_raw = raw.pop("schedule", {}) or {}

    for name, values, cls in (
        ("schedule", schedule_raw, ScheduleConfig),
        ("run", raw, TrainConfig),
    ):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"{path}: unknown {name} keys {sorted(unknown)}")

    return TrainConfig(**raw), ScheduleConfig(**schedule_raw)


def resolve_device(requested: str) -> torch.device:
    """Turn ``auto`` into the best device present."""
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Pick an autocast dtype the hardware actually supports.

    Turing cards (the T4s on Kaggle) have fp16 tensor cores but no bfloat16, so
    ``auto`` must not simply prefer bf16 wherever CUDA is available.
    """
    if requested != "auto":
        return getattr(torch, requested)
    if device.type != "cuda":
        return torch.float32

    # Compute capability, not is_bf16_supported(). That helper returns True on
    # cards with no bfloat16 tensor cores -- Pascal (sm_60) and Turing (sm_75)
    # included -- because PyTorch can emulate the type in software. Taking it
    # at its word picks a dtype the hardware runs slowly, which on a T4 means
    # giving up the fp16 tensor cores that make the card worth using.
    # Bfloat16 arrives with Ampere, sm_80.
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def build_optimizer(
    model: torch.nn.Module, cfg: TrainConfig, lr: float
) -> torch.optim.Optimizer:
    """AdamW with decay on matrices and none on biases or norm gains."""
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if param.requires_grad:
            (decay if param.dim() >= 2 else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    log.info(
        "optimizer: %d decayed tensors (%d params), %d undecayed (%d params)",
        len(decay), sum(p.numel() for p in decay),
        len(no_decay), sum(p.numel() for p in no_decay),
    )

    # The fused kernel launches one CUDA op for the whole update instead of one
    # per tensor; on a small model that is a real fraction of step time.
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=(cfg.beta1, cfg.beta2),
                                 eps=1e-8, fused=True)
    except (RuntimeError, ValueError):
        return torch.optim.AdamW(groups, lr=lr, betas=(cfg.beta1, cfg.beta2),
                                 eps=1e-8)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    tokens: TokenArray,
    cfg: TrainConfig,
    block_size: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_batches: int | None,
) -> float:
    """Mean cross-entropy over deterministic held-out windows.

    Deterministic windows rather than random samples, so successive validation
    checks are comparable to each other and the curve reflects the model
    improving rather than the draw changing.
    """
    was_training = model.training
    model.eval()
    total, batches = 0.0, 0
    for x, y in evaluation_windows(tokens, cfg.batch_size, block_size, max_batches):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=device.type == "cuda"):
            total += model(x, y).loss.item()
        batches += 1
    model.train(was_training)
    return total / max(batches, 1)


class MetricLog:
    """Append-only JSONL record of a run, and the source for the loss curves.

    Kept in the repository rather than in the ignored data directory: the brief
    grades what is on the branch, and a loss curve stored only on Drive is not.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        """Write one record and flush, so a killed session keeps its history."""
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def train(
    model_cfg: GPTConfig,
    train_cfg: TrainConfig,
    schedule_cfg: ScheduleConfig,
    train_tokens: TokenArray,
    val_tokens: TokenArray,
    checkpoint_dir: Path,
    metric_log: MetricLog,
    resume: bool = True,
    stop_at_step: int | None = None,
) -> TrainingState:
    """Pretrain one model, resuming from ``checkpoint_dir`` if anything is there.

    Args:
        stop_at_step: End this invocation once the run reaches this step,
            leaving the schedule untouched. A Kaggle session is capped at
            twelve hours, so a long run is several invocations that each
            stop early and resume; the learning-rate curve must stay the
            curve the whole run was planned around, which is why this is
            separate from ``max_steps``.

    Returns:
        The final :class:`TrainingState`.
    """
    target = train_cfg.max_steps
    if stop_at_step is not None:
        target = min(target, stop_at_step)
    device = resolve_device(train_cfg.device)
    autocast_dtype = resolve_dtype(train_cfg.dtype, device)
    use_amp = device.type == "cuda"
    log.info("device %s, autocast dtype %s", device, autocast_dtype)

    torch.manual_seed(train_cfg.seed)
    model = GPT(model_cfg).to(device)
    log.info("model: %.2fM parameters", model.num_parameters() / 1e6)

    optimizer = build_optimizer(model, train_cfg, schedule_cfg.max_lr)
    # The scaler is only meaningful for fp16; bf16 has the range to go without.
    scaler = torch.amp.GradScaler(
        device.type, enabled=use_amp and autocast_dtype is torch.float16
    )

    state = TrainingState(seed=train_cfg.seed)
    if resume and (latest := find_latest(checkpoint_dir)) is not None:
        state = load_checkpoint(latest, model, optimizer, scaler, map_location=device)
    else:
        log.info("no checkpoint found; starting from scratch")

    tokens_per_step = train_cfg.tokens_per_step(model_cfg.block_size)
    model.train()
    started = time.time()
    window_started, window_tokens = started, 0

    while state.step < target:
        lr = learning_rate(state.step, schedule_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        # Kept on-device and summed as a tensor: calling .item() inside the
        # accumulation loop would force a GPU sync every micro-step.
        running = torch.zeros((), device=device)

        for _ in range(train_cfg.grad_accum):
            x, y = training_batch(
                train_tokens, train_cfg.batch_size, model_cfg.block_size,
                train_cfg.seed, state.micro_step,
            )
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=use_amp):
                loss = model(x, y).loss / train_cfg.grad_accum
            scaler.scale(loss).backward()
            running += loss.detach()
            state.micro_step += 1

        if train_cfg.grad_clip > 0:
            # Gradients must be unscaled before the norm is meaningful.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.grad_clip
            ).item()
        else:
            grad_norm = float("nan")

        scaler.step(optimizer)
        scaler.update()

        state.step += 1
        state.tokens_seen += tokens_per_step
        window_tokens += tokens_per_step
        train_loss = running.item()

        if state.step % train_cfg.log_interval == 0:
            now = time.time()
            throughput = window_tokens / max(now - window_started, 1e-6)
            remaining = (train_cfg.max_steps - state.step) * tokens_per_step
            log.info(
                "step %6d/%d  loss %.4f  lr %.2e  |g| %.2f  %.0f tok/s  eta %s",
                state.step, train_cfg.max_steps, train_loss, lr, grad_norm,
                throughput, _format_eta(remaining / max(throughput, 1e-6)),
            )
            metric_log.append({
                "step": state.step, "tokens": state.tokens_seen,
                "train_loss": train_loss, "lr": lr, "grad_norm": grad_norm,
                "tokens_per_second": throughput, "elapsed": now - started,
            })
            window_started, window_tokens = now, 0

        if state.step % train_cfg.eval_interval == 0 or state.step == target:
            val_loss = estimate_loss(
                model, val_tokens, train_cfg, model_cfg.block_size, device,
                autocast_dtype, train_cfg.eval_batches,
            )
            log.info("step %6d  val loss %.4f  val ppl %.2f", state.step,
                     val_loss, math.exp(val_loss))
            metric_log.append({
                "step": state.step, "tokens": state.tokens_seen,
                "val_loss": val_loss, "val_ppl": math.exp(val_loss),
                "elapsed": time.time() - started,
            })
            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer,
                                scaler, state, model_cfg, schedule_cfg)

        if state.step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir / f"step_{state.step:06d}.pt", model, optimizer,
                scaler, state, model_cfg, schedule_cfg,
            )
            prune(checkpoint_dir, train_cfg.keep_checkpoints)

    # Only if the loop did not already save at this exact step, which it does
    # whenever max_steps lands on a checkpoint interval. The real checkpoints
    # are ~290MB; writing one twice is a minute of wasted session time.
    if state.step % train_cfg.checkpoint_interval != 0:
        save_checkpoint(
            checkpoint_dir / f"step_{state.step:06d}.pt", model, optimizer,
            scaler, state, model_cfg, schedule_cfg,
        )
    log.info("%s at step %d/%d, %d tokens seen, best val loss %.4f",
             "finished" if state.step >= train_cfg.max_steps else "stopped early",
             state.step, train_cfg.max_steps, state.tokens_seen,
             state.best_val_loss)
    return state


def _format_eta(seconds: float) -> str:
    """Seconds as ``HHhMMm``, for progress lines."""
    if not math.isfinite(seconds):
        return "?"
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"
