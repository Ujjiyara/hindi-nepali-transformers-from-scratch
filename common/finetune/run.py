"""Finetune one pretrained model on its own language's reasoning data.

From the repository root, once per language::

    python -m common.finetune.run \
        --init-checkpoint Hindi/checkpoints/best.pt \
        --train-config Hindi/configs/finetune.yaml \
        --data Hindi/data/reasoning \
        --tokenizer Hindi/tokenizer/hi_unigram_16000.model \
        --checkpoint-dir Hindi/checkpoints/finetune \
        --run-dir Hindi/runs/finetune

Re-running the same command after an interruption resumes from the newest
checkpoint in ``--checkpoint-dir``. The tokenizer is the one the model was
pretrained with and is only read, never changed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import sentencepiece as spm
import torch

from common.finetune.data import encode_examples
from common.finetune.loop import FinetuneConfig, finetune
from common.reasoning.generate import read_jsonl
from common.train.checkpoint import load_config
from common.train.loop import MetricLog, load_run_config

log = logging.getLogger("finetune")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Reasoning finetune for one model.")
    parser.add_argument("--init-checkpoint", required=True, type=Path,
                        help="this language's pretrained best.pt")
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path,
                        help="directory holding train.jsonl and validation.jsonl")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads, when two runs share one machine")
    parser.add_argument("--stop-at-step", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(args.run_dir / "train.log", encoding="utf-8")],
    )
    if args.threads:
        torch.set_num_threads(args.threads)

    train_cfg, schedule_cfg = load_run_config(args.train_config, FinetuneConfig)
    if args.device:
        train_cfg = replace(train_cfg, device=args.device)

    model_cfg, pretrained_state = load_config(args.init_checkpoint)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if sp.get_piece_size() != model_cfg.vocab_size:
        raise ValueError(
            f"tokenizer has {sp.get_piece_size()} pieces but {args.init_checkpoint} "
            f"was pretrained with {model_cfg.vocab_size}"
        )

    train_records = read_jsonl(args.data / "train.jsonl")
    val_records = read_jsonl(args.data / "validation.jsonl")
    train_examples = encode_examples(train_records, sp, model_cfg.block_size)
    val_examples = encode_examples(val_records, sp, model_cfg.block_size)

    epochs = train_cfg.max_steps * train_cfg.batch_size * train_cfg.grad_accum / len(train_examples)
    log.info("%d train / %d validation examples; %d steps = %.2f epochs",
             len(train_examples), len(val_examples), train_cfg.max_steps, epochs)

    resolved = {
        "stage": "reasoning_finetune",
        "init_checkpoint": str(args.init_checkpoint),
        "init_checkpoint_step": pretrained_state["step"],
        "tokenizer": str(args.tokenizer),
        "data": str(args.data),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "epochs": epochs,
        "loss": "cross-entropy on answer tokens and EOS only; prompt targets masked",
        "model": model_cfg.to_dict(),
        "train": asdict(train_cfg),
        "schedule": asdict(schedule_cfg),
        "checkpoint_dir": str(args.checkpoint_dir),
    }
    (args.run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n",
                                              encoding="utf-8")

    state = finetune(
        init_checkpoint=args.init_checkpoint,
        train_cfg=train_cfg,
        schedule_cfg=schedule_cfg,
        train_examples=train_examples,
        val_examples=val_examples,
        pad_id=sp.pad_id(),
        checkpoint_dir=args.checkpoint_dir,
        metric_log=MetricLog(args.run_dir / "metrics.jsonl"),
        resume=not args.no_resume,
        stop_at_step=args.stop_at_step,
    )
    log.info("done: step %d, best val answer loss %.4f", state.step, state.best_val_loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
