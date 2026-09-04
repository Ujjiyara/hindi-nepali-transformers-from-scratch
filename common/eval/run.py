"""Evaluate one trained checkpoint and write every Phase 2 number it produces.

From the repository root, once per model::

    python -m common.eval.run \
        --checkpoint Hindi/checkpoints/best.pt \
        --tokens Hindi/data/tokens \
        --tokenizer Hindi/tokenizer/hi_unigram_16000.model \
        --processed Hindi/data/processed \
        --label "Hindi (Model H)" \
        --out-dir Hindi/runs/base/eval

Everything lands in ``<out-dir>/results.json`` and ``<out-dir>/samples.json``,
which are small enough to commit and are what the report is generated from --
so no figure or table in the report depends on a checkpoint being present.

The checkpoint carries its own geometry, so no model config file is needed
here: an evaluation cannot silently be run against the wrong architecture.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import sentencepiece as spm
import torch

from common.data.loader import TokenArray, evaluation_windows
from common.eval.attention import capture_attention, collect_statistics
from common.eval.generate import build_prompts, generate_for_prompts
from common.eval.lm import score_split
from common.eval.plots import plot_attention_heatmaps
from common.model.gpt import GPT
from common.train.checkpoint import load_checkpoint, load_config
from common.train.loop import resolve_device, resolve_dtype

log = logging.getLogger("eval")

# Greedy plus the three temperatures the brief names.
TEMPERATURES = (0.0, 0.5, 1.0, 1.5)


def load_model(checkpoint: Path, device: torch.device) -> tuple[GPT, dict]:
    """Rebuild a model from a checkpoint and return it with its training state."""
    cfg, state = load_config(checkpoint)
    model = GPT(cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device, restore_rng=False)
    log.info("loaded %s: %.2fM parameters, trained to step %d",
             checkpoint, model.num_parameters() / 1e6, state["step"])
    return model, state


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate one pretrained model.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path,
                        help="directory of packed .bin files")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--processed", required=True, type=Path,
                        help="directory holding test.jsonl.gz, for prompts")
    parser.add_argument("--label", required=True,
                        help='e.g. "Hindi (Model H)"')
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--figures", type=Path, default=Path("report/figures"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="logits are (B, T, vocab) and dominate memory here")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--continuation-tokens", type=int, default=96)
    parser.add_argument("--attention-batches", type=int, default=4)
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="cap batches when scoring; smoke tests only, "
                             "since reported numbers must cover the split")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-attention", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    device = resolve_device(args.device)
    autocast_dtype = resolve_dtype("auto", device)
    model, state = load_model(args.checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    if sp.get_piece_size() != model.cfg.vocab_size:
        raise ValueError(
            f"tokenizer has {sp.get_piece_size()} pieces but the checkpoint "
            f"was trained with {model.cfg.vocab_size}"
        )

    packed = json.loads((args.tokens / "_stats_pack.json").read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "step": state["step"],
        "tokens_seen": state["tokens_seen"],
        "parameters": model.num_parameters(),
        "model_config": model.cfg.to_dict(),
        "language_modelling": {},
    }

    # --- intrinsic metrics -------------------------------------------------
    for split in ("validation", "test"):
        metrics = score_split(
            model,
            TokenArray(args.tokens / f"{split}.bin"),
            packed["splits"][split],
            split,
            args.batch_size,
            model.cfg.block_size,
            device,
            autocast_dtype,
            max_batches=args.max_eval_batches,
        )
        entry = metrics.as_dict()
        # Guards against a capped smoke run being mistaken for a result.
        entry["complete_split"] = args.max_eval_batches is None
        results["language_modelling"][split] = entry

    # --- generation --------------------------------------------------------
    if not args.skip_generation:
        prompts = build_prompts(
            args.processed / "test.jsonl.gz", sp, args.prompts,
            args.prompt_tokens, args.continuation_tokens,
        )
        references = [p.reference_text for p in prompts]

        # Every continuation is kept, not just the few shown in the report, so
        # that the overlap metrics can be recomputed from this file without a
        # GPU. sacrebleu is not part of the Kaggle image and the run may have
        # no internet, and losing hours of generation to a missing text-metrics
        # package would be a poor trade.
        try:
            from common.eval.text_metrics import score_generations
        except ImportError as error:
            score_generations = None
            log.warning("%s: writing samples, scoring them elsewhere", error)

        generation_results, samples = [], []
        for temperature in TEMPERATURES:
            hypotheses = generate_for_prompts(
                model, prompts, sp, args.continuation_tokens, temperature,
                device, batch_size=args.batch_size,
            )
            if score_generations is not None:
                generation_results.append(
                    score_generations(hypotheses, references, temperature).as_dict()
                )
            samples.append({
                "temperature": temperature,
                "prompts": [p.prompt_text for p in prompts],
                "references": references,
                "hypotheses": hypotheses,
            })

        if generation_results:
            results["generation"] = generation_results
        (args.out_dir / "samples.json").write_text(
            json.dumps({"label": args.label, "samples": samples},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # --- attention ---------------------------------------------------------
    if not args.skip_attention:
        test_tokens = TokenArray(args.tokens / "test.bin")
        stats = collect_statistics(
            model,
            evaluation_windows(test_tokens, args.batch_size, model.cfg.block_size),
            device,
            max_batches=args.attention_batches,
        )
        results["attention"] = stats.as_dict()

        # Heatmaps for a real sentence, at an early and a late layer.
        prompt_ids = next(iter(
            evaluation_windows(test_tokens, 1, min(24, model.cfg.block_size))
        ))[0]
        pieces = [sp.id_to_piece(int(i)) for i in prompt_ids[0]]
        attentions = capture_attention(model, prompt_ids, device)
        for layer in (0, model.cfg.n_layer - 1):
            path = plot_attention_heatmaps(
                attentions[layer][0].numpy(), pieces, args.label, layer,
                args.figures,
            )
            log.info("wrote %s", path)

    (args.out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("wrote %s", args.out_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
