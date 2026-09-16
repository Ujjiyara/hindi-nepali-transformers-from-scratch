"""The reasoning finetune loop.

Deliberately the pretraining loop with three things changed, so that everything
already tested there -- AdamW with decay on matrices only, warmup plus cosine,
gradient clipping, mixed precision, atomic resumable checkpoints -- carries
over unchanged:

1. **Initialisation.** A fresh run loads the weights of *this language's own*
   pretrained ``best.pt`` and nothing else: not its optimizer moments, which
   describe the pretraining loss surface, and not its step counter. The model
   geometry is read from that checkpoint too, so a finetune cannot be started
   from a checkpoint of a different shape or vocabulary.
2. **Data.** Batches are masked question/answer pairs from
   :mod:`common.finetune.data` rather than windows of the corpus.
3. **Validation.** Alongside the answer-token loss, a teacher-forced exact
   match is logged: the fraction of validation examples for which the argmax
   at *every* answer position (EOS included) is the right token. It is cheap --
   one forward pass, no decoding -- and tracks the quantity the finetune is
   for, where loss alone can keep falling on examples already answered right.

Checkpoints are written with :func:`common.train.checkpoint.save_checkpoint`,
so they are the same format as pretraining checkpoints -- weights, optimizer,
schedule, scaler, step, data position, RNG, geometry -- plus an ``extra``
record naming the stage and the checkpoint the finetune started from.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch

from common.finetune.data import EncodedExample, evaluation_batches, training_batch
from common.model.gpt import GPT
from common.train.checkpoint import (
    TrainingState,
    find_latest,
    load_checkpoint,
    load_config,
    prune,
    save_checkpoint,
)
from common.train.loop import (
    MetricLog,
    TrainConfig,
    _format_eta,
    build_optimizer,
    resolve_device,
    resolve_dtype,
)
from common.train.schedule import ScheduleConfig, learning_rate

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_answers(
    model: GPT,
    examples: list[EncodedExample],
    batch_size: int,
    pad_id: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict:
    """Answer-token loss and teacher-forced exact match over held-out examples.

    Returns:
        ``{"loss": mean cross-entropy per answer token, "exact_match": share
        of examples whose every answer token is the argmax, "examples": n}``.
    """
    was_training = model.training
    model.eval()
    loss_sum, token_count, exact, seen = 0.0, 0, 0, 0

    for x, y in evaluation_batches(examples, batch_size, pad_id, max_batches):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=device.type == "cuda"):
            logits = model(x).logits
        logits = logits.float()
        supervised = y != -1
        losses = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), y.clamp(min=0), reduction="none"
        )
        loss_sum += losses[supervised].sum().item()
        token_count += int(supervised.sum())
        correct = (logits.argmax(dim=-1) == y) | ~supervised
        exact += int(correct.all(dim=1).sum())
        seen += x.size(0)

    model.train(was_training)
    return {"loss": loss_sum / max(token_count, 1),
            "exact_match": exact / max(seen, 1), "examples": seen}


def finetune(
    init_checkpoint: Path,
    train_cfg: TrainConfig,
    schedule_cfg: ScheduleConfig,
    train_examples: list[EncodedExample],
    val_examples: list[EncodedExample],
    pad_id: int,
    checkpoint_dir: Path,
    metric_log: MetricLog,
    resume: bool = True,
    stop_at_step: int | None = None,
) -> TrainingState:
    """Finetune one pretrained model on its own language's reasoning data.

    Args:
        init_checkpoint: This language's pretrained checkpoint. Used for the
            weights of a fresh run; ignored when resuming a finetune.
        train_cfg: Run settings, in the same format as pretraining.
        schedule_cfg: Learning-rate curve for the finetune.
        train_examples, val_examples: Encoded reasoning splits.
        pad_id: This tokenizer's pad id.
        checkpoint_dir: Where finetune checkpoints go; kept apart from the
            pretraining checkpoints so resume never picks one of those up.
        metric_log: JSONL log for the finetuning curves.
        resume: Continue from the newest ``step_*.pt`` in ``checkpoint_dir``.
        stop_at_step: End this invocation early without changing the schedule.

    Returns:
        The final :class:`TrainingState`.
    """
    target = train_cfg.max_steps if stop_at_step is None else min(
        train_cfg.max_steps, stop_at_step)
    device = resolve_device(train_cfg.device)
    autocast_dtype = resolve_dtype(train_cfg.dtype, device)
    use_amp = device.type == "cuda"

    model_cfg, _ = load_config(init_checkpoint)
    torch.manual_seed(train_cfg.seed)
    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg, schedule_cfg.max_lr)
    scaler = torch.amp.GradScaler(
        device.type, enabled=use_amp and autocast_dtype is torch.float16
    )
    extra = {"stage": "reasoning_finetune", "init_checkpoint": str(init_checkpoint)}

    def evaluate() -> dict:
        return evaluate_answers(model, val_examples, train_cfg.batch_size, pad_id,
                                device, autocast_dtype, train_cfg.eval_batches)

    state = TrainingState(seed=train_cfg.seed)
    if resume and (latest := find_latest(checkpoint_dir)) is not None:
        state = load_checkpoint(latest, model, optimizer, scaler, map_location=device)
    else:
        # Weights only: pretraining's optimizer moments and step do not apply.
        pretrained = load_checkpoint(init_checkpoint, model, map_location=device,
                                     restore_rng=False)
        log.info("initialised from %s (pretrained to step %d)",
                 init_checkpoint, pretrained.step)
        torch.manual_seed(train_cfg.seed)
        # The pretrained model's score, as step 0 of the finetuning curve.
        before = evaluate()
        log.info("step 0 (pretrained)  val answer loss %.4f  val exact %.3f",
                 before["loss"], before["exact_match"])
        metric_log.append({"step": 0, "tokens": 0, "val_loss": before["loss"],
                           "val_exact_match": before["exact_match"],
                           "val_ppl": math.exp(before["loss"]), "elapsed": 0.0})

    model.train()
    started = time.time()
    window_started = started

    while state.step < target:
        lr = learning_rate(state.step, schedule_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running = torch.zeros((), device=device)
        step_tokens = 0

        for _ in range(train_cfg.grad_accum):
            x, y = training_batch(train_examples, train_cfg.batch_size,
                                  train_cfg.seed, state.micro_step, pad_id)
            step_tokens += int((x != pad_id).sum())
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=use_amp):
                loss = model(x, y).loss / train_cfg.grad_accum
            scaler.scale(loss).backward()
            running += loss.detach()
            state.micro_step += 1

        if train_cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.grad_clip).item()
        else:
            grad_norm = float("nan")
        scaler.step(optimizer)
        scaler.update()

        state.step += 1
        state.tokens_seen += step_tokens
        train_loss = running.item()

        if state.step % train_cfg.log_interval == 0:
            now = time.time()
            per_step = (now - window_started) / train_cfg.log_interval
            epoch = state.micro_step * train_cfg.batch_size / len(train_examples)
            log.info("step %5d/%d  loss %.4f  lr %.2e  |g| %.2f  epoch %.2f  %.2fs/step  eta %s",
                     state.step, train_cfg.max_steps, train_loss, lr, grad_norm, epoch,
                     per_step, _format_eta((train_cfg.max_steps - state.step) * per_step))
            metric_log.append({"step": state.step, "tokens": state.tokens_seen,
                               "train_loss": train_loss, "lr": lr,
                               "grad_norm": grad_norm, "epoch": epoch,
                               "elapsed": now - started})
            window_started = now

        if state.step % train_cfg.eval_interval == 0 or state.step == target:
            result = evaluate()
            log.info("step %5d  val answer loss %.4f  val exact %.3f",
                     state.step, result["loss"], result["exact_match"])
            metric_log.append({"step": state.step, "tokens": state.tokens_seen,
                               "val_loss": result["loss"],
                               "val_exact_match": result["exact_match"],
                               "val_ppl": math.exp(result["loss"]),
                               "elapsed": time.time() - started})
            if result["loss"] < state.best_val_loss:
                state.best_val_loss = result["loss"]
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scaler,
                                state, model_cfg, schedule_cfg, extra)

        if state.step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir / f"step_{state.step:06d}.pt", model,
                            optimizer, scaler, state, model_cfg, schedule_cfg, extra)
            prune(checkpoint_dir, train_cfg.keep_checkpoints)

    if state.step % train_cfg.checkpoint_interval != 0:
        save_checkpoint(checkpoint_dir / f"step_{state.step:06d}.pt", model,
                        optimizer, scaler, state, model_cfg, schedule_cfg, extra)
    log.info("%s at step %d/%d, best val answer loss %.4f",
             "finished" if state.step >= train_cfg.max_steps else "stopped early",
             state.step, train_cfg.max_steps, state.best_val_loss)
    return state
