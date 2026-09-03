"""Pretrain one model. Run once per language, never on both at once.

From the repository root::

    python -m common.train.run \
        --model-config Hindi/configs/model.yaml \
        --train-config Hindi/configs/train.yaml \
        --tokens Hindi/data/tokens \
        --checkpoint-dir Hindi/checkpoints \
        --run-dir Hindi/runs/hi-base

On a Kaggle session with two T4s the two models are trained side by side, one
per GPU, by launching this twice with ``--device cuda:0`` and ``--device
cuda:1``. They share nothing but the machine.

Re-running the same command after a session dies resumes from the newest
checkpoint in ``--checkpoint-dir``; pass ``--no-resume`` to start over.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

from common.data.loader import TokenArray
from common.model.config import GPTConfig
from common.train.checkpoint import find_latest
from common.train.loop import MetricLog, load_run_config, train

log = logging.getLogger("train")


def check_tokens_match_model(tokens_dir: Path, model_cfg: GPTConfig) -> None:
    """Refuse to train if the token files were built by a different tokenizer.

    The two models must stay independent, and the cheapest way for that to fail
    silently is a copy-paste in a command line that points Model H at Nepali
    tokens. The vocabulary sizes would still both be 16,000, so nothing else
    would notice.
    """
    sidecar = tokens_dir / "_stats_pack.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"{sidecar} not found -- run common.data.run pack first"
        )
    packed = json.loads(sidecar.read_text(encoding="utf-8"))
    packed_vocab = packed["tokenizer"]["vocab_size"]
    if packed_vocab != model_cfg.vocab_size:
        raise ValueError(
            f"token files were packed with a {packed_vocab}-entry vocabulary "
            f"but the model config asks for {model_cfg.vocab_size}"
        )
    log.info("tokens: %s, packed with %s", tokens_dir, packed["tokenizer"]["model"])


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Pretrain one monolingual model.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path,
                        help="directory holding train/validation .bin files")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="where metrics.jsonl and the resolved config go")
    parser.add_argument("--device", default=None,
                        help="override the config, e.g. cuda:1 or cpu")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="shorten the whole run; also reshapes the LR curve")
    parser.add_argument("--stop-at-step", type=int, default=None,
                        help="end this session at this step, schedule unchanged")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    model_cfg = GPTConfig.from_yaml(args.model_config)
    train_cfg, schedule_cfg = load_run_config(args.train_config)
    if args.device:
        train_cfg = replace(train_cfg, device=args.device)
    if args.max_steps is not None:
        train_cfg = replace(train_cfg, max_steps=args.max_steps)
        schedule_cfg = replace(schedule_cfg, max_steps=args.max_steps,
                               warmup_steps=min(schedule_cfg.warmup_steps,
                                                max(1, args.max_steps // 10)))

    check_tokens_match_model(args.tokens, model_cfg)

    # Hyperparameters are written next to the metrics, so a run on the branch
    # is reproducible from the branch alone -- the brief asks for exactly this.
    args.run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "model": model_cfg.to_dict(),
        "train": asdict(train_cfg),
        "schedule": asdict(schedule_cfg),
        "tokens": str(args.tokens),
        "checkpoint_dir": str(args.checkpoint_dir),
        "parameters": model_cfg.parameter_count(),
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )

    train_tokens = TokenArray(args.tokens / "train.bin")
    val_tokens = TokenArray(args.tokens / "validation.bin")
    epochs = (train_cfg.max_steps * train_cfg.tokens_per_step(model_cfg.block_size)
              / len(train_tokens))
    log.info("%d training tokens, %d planned steps, %.2f epochs",
             len(train_tokens), train_cfg.max_steps, epochs)

    if not args.no_resume and (latest := find_latest(args.checkpoint_dir)):
        log.info("will resume from %s", latest)

    state = train(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        schedule_cfg=schedule_cfg,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        checkpoint_dir=args.checkpoint_dir,
        metric_log=MetricLog(args.run_dir / "metrics.jsonl"),
        resume=not args.no_resume,
        stop_at_step=args.stop_at_step,
    )
    log.info("done: step %d, best val loss %.4f", state.step, state.best_val_loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
