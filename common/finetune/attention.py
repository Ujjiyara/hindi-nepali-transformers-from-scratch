"""Pretrained versus finetuned attention, on comparative-reasoning prompts.

The brief asks for the Phase 2 attention toolkit to be rerun on the finetuned
checkpoints and compared with the pretrained ones. This module does that, on
the *same* inputs for both checkpoints, and adds one measurement specific to
the reasoning task.

**Reused from Phase 2** (:mod:`common.eval.attention`): per-head entropy,
normalised entropy and mean attention distance. They are computed twice per
checkpoint -- on reasoning test prompts, and on ordinary held-out corpus text --
so that a change on the prompts can be told apart from a change everywhere
(i.e. the finetune having simply rewritten the head).

**Answer-position attention.** The token the model reads last, the ":" after
"उत्तर", is where the answer's first token is predicted, so its attention row
shows what the model consults when it answers. For every layer and head the
mass of that row is split into:

``entity``
    tokens belonging to any mention of an entity named in the prompt;
``gold``
    tokens of mentions of the correct entity, for questions whose answer is an
    entity (the share of ``entity`` mass that lands on the right one is what a
    head that "looks up the answer" would raise);
``question``
    tokens of the question sentence itself;
``first``
    the first token, the attention sink found in Phase 2.

Mention spans are found in the prompt text with a word-start match (a name
may be followed by an attached Nepali postposition, and "राम" occurs inside
"किलोग्राम") and mapped to tokens through SentencePiece's byte offsets, so a
token counts as part of a mention exactly when its bytes overlap it.

Run once per language::

    python -m common.finetune.attention \
        --pretrained Hindi/checkpoints/best.pt \
        --finetuned Hindi/checkpoints/finetune/best.pt \
        --tokenizer Hindi/tokenizer/hi_unigram_16000.model \
        --data Hindi/data/reasoning \
        --tokens Hindi/data/tokens \
        --label "Hindi (Model H)" \
        --out Hindi/runs/finetune/eval/attention.json

Computation only; the figures are drawn by :mod:`common.finetune.figures`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from common.data.loader import TokenArray, evaluation_windows
from common.eval.attention import capture_attention, collect_statistics
from common.eval.run import load_model
from common.reasoning.generate import read_jsonl
from common.train.loop import resolve_device

log = logging.getLogger("attention-compare")


def mention_spans(text: str, names: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Byte spans of every mention of every name, longest names matched first."""
    alternatives = "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))
    spans: dict[str, list[tuple[int, int]]] = {n: [] for n in names}
    for match in re.finditer(rf"(?<![^\s,])(?:{alternatives})", text):
        start = len(text[: match.start()].encode("utf-8"))
        end = start + len(match.group(0).encode("utf-8"))
        spans[match.group(0)].append((start, end))
    return spans


def token_masks(record: dict, sp) -> tuple[list[int], dict[str, np.ndarray]]:
    """Token ids of a prompt and boolean masks over them for each region."""
    proto = sp.encode(record["prompt"], return_type="proto")
    pieces = list(proto.pieces)
    ids = [p.id for p in pieces]
    n = len(ids)

    def overlapping(spans) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        for i, piece in enumerate(pieces):
            mask[i] = any(piece.begin < end and start < piece.end for start, end in spans)
        return mask

    spans = mention_spans(record["prompt"], record["entities"])
    masks = {"entity": overlapping([s for v in spans.values() for s in v])}
    if record["answer"] in spans:
        masks["gold"] = overlapping(spans[record["answer"]])

    # The question is the text after the last sentence end mark.
    text = record["prompt"]
    boundary = text.rfind("।")
    question_start = len(text[: boundary + 1].encode("utf-8"))
    masks["question"] = np.array([p.begin >= question_start for p in pieces])
    first = np.zeros(n, dtype=bool)
    first[0] = True
    masks["first"] = first
    return ids, masks


@torch.no_grad()
def answer_position_attention(model, sp, records: list[dict], device) -> dict:
    """Mean attention mass from the final prompt token onto each region.

    Returns:
        ``{region: (n_layer, n_head) list}`` plus ``gold_share`` -- gold mass
        divided by entity mass -- and the example counts behind each.
    """
    model.eval()
    totals: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    share_total, share_count = None, 0

    for record in records:
        ids, masks = token_masks(record, sp)
        attentions = model(torch.tensor([ids], device=device), return_attention=True).attentions
        # (n_layer, n_head, T): each head's attention row at the last query.
        rows = torch.stack([a[0, :, -1, :] for a in attentions]).float().cpu().numpy()

        for region, mask in masks.items():
            mass = rows[:, :, mask].sum(axis=-1)
            totals[region] = totals.get(region, 0) + mass
            counts[region] = counts.get(region, 0) + 1

        if "gold" in masks:
            entity_mass = rows[:, :, masks["entity"]].sum(axis=-1)
            gold_mass = rows[:, :, masks["gold"]].sum(axis=-1)
            share = gold_mass / np.maximum(entity_mass, 1e-9)
            share_total = share if share_total is None else share_total + share
            share_count += 1

    result = {region: (totals[region] / counts[region]).tolist() for region in totals}
    result["gold_share"] = (share_total / share_count).tolist() if share_count else None
    result["counts"] = {**counts, "gold_share": share_count}
    return result


def pick_heatmap_example(records: list[dict]) -> dict:
    """A deterministic, readable prompt: the first three-entity chain superlative."""
    for record in records:
        if record["task"] == "chain_superlative" and record["hops"] == 2:
            return record
    return records[0]


def analyse(model, sp, records: list[dict], text_tokens: TokenArray | None,
            device, text_batches: int) -> dict:
    """Every attention measurement for one checkpoint."""
    prompts = ((torch.tensor([sp.encode(r["prompt"], out_type=int)]), None)
               for r in records)
    result = {
        "reasoning_prompts": collect_statistics(model, prompts, device).as_dict(),
        "answer_position": answer_position_attention(model, sp, records, device),
    }
    if text_tokens is not None:
        result["corpus_text"] = collect_statistics(
            model, evaluation_windows(text_tokens, 4, model.cfg.block_size), device,
            max_batches=text_batches,
        ).as_dict()
    return result


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Compare attention before/after finetuning.")
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--finetuned", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tokens", type=Path, default=None,
                        help="packed corpus tokens, for statistics on ordinary text")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=400)
    parser.add_argument("--text-batches", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    records = read_jsonl(args.data / f"{args.split}.jsonl")[: args.examples]
    text_tokens = TokenArray(args.tokens / "test.bin") if args.tokens else None
    example = pick_heatmap_example(records)
    example_ids = sp.encode(example["prompt"], out_type=int)

    results = {
        "label": args.label,
        "split": args.split,
        "examples": len(records),
        "heatmap_example": {
            "id": example["id"], "prompt": example["prompt"], "answer": example["answer"],
            "tokens": [sp.id_to_piece(i) for i in example_ids],
        },
        "stages": {},
    }
    heatmaps = {}
    for stage, checkpoint in (("pretrained", args.pretrained), ("finetuned", args.finetuned)):
        model, state = load_model(checkpoint, device)
        log.info("%s: analysing %s (step %d)", stage, checkpoint, state["step"])
        results["stages"][stage] = {"checkpoint": str(checkpoint), "step": state["step"],
                                    **analyse(model, sp, records, text_tokens, device,
                                              args.text_batches)}
        attentions = capture_attention(model, torch.tensor([example_ids]), device)
        heatmaps[stage] = np.stack([a[0].numpy() for a in attentions])  # (L, H, T, T)
        del model

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    # The heatmap matrices are small (L x H x T x T at T ~ 35) and let the
    # figures be redrawn without a checkpoint.
    np.savez_compressed(args.out.with_suffix(".npz"), **heatmaps)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
