"""Score a checkpoint on the synthetic reasoning test sets.

Run once for the pretrained and once for the finetuned checkpoint of each
language, on the same test files, so the two are compared on identical
examples::

    python -m common.finetune.evaluate \
        --checkpoint Hindi/checkpoints/finetune/best.pt \
        --tokenizer Hindi/tokenizer/hi_unigram_16000.model \
        --data Hindi/data/reasoning \
        --label "Hindi (Model H), finetuned" \
        --out Hindi/runs/finetune/eval/finetuned.json

**Three metrics, because one would mislead.**

``exact_match``
    Greedy decoding from the prompt until EOS (or a token cap); the decoded
    text must equal the gold answer exactly. This is the strict metric the
    brief names, and what a user of the model would experience.
``prefix_match``
    The decoded text *begins* with the gold answer as a whole word. A
    pretrained model never learned to stop after an answer, so it might name
    the right entity and then keep writing; exact match scores that as wrong,
    and this metric separates "wrong answer" from "right answer, no EOS".
``choice_accuracy``
    No generation at all: each candidate answer offered by the example (the
    entities in it, or yes/no, or "both equal") is scored by the total
    log-probability the model assigns to its tokens after the prompt, and the
    highest-scoring candidate is the prediction. This is the fair comparison
    for the *pretrained* model, which cannot be expected to produce the answer
    format but may still prefer the right entity. It is compared against the
    chance baseline (one over the number of candidates), not against zero.

Candidate scores sum token log-probabilities without EOS, so a candidate is
not penalised merely for being followed by more text in the pretraining
distribution; the length bias this leaves (multi-token "both equal" phrases
against one-token names) only arises in ``numeric_pair`` and is visible in the
per-task breakdown.

**Batched greedy decoding with right padding.** Prompts differ in length. Each
row keeps its own write position; every step runs the model over the whole
padded buffer and reads each row's logits at *its* last real position. Causal
masking means the padding to the right of a row cannot affect those logits, and
no row's tokens are ever moved to a position they would not have held when
decoded alone -- which left padding would do, against a learned absolute
positional table.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
import torch

from common.eval.run import load_model
from common.reasoning.generate import read_jsonl
from common.train.loop import resolve_device

log = logging.getLogger("reasoning-eval")

_WORD_BOUNDARY = set(" \t,।?.!:;")


@torch.no_grad()
def greedy_answers(model, sp, prompts: list[list[int]], device: torch.device,
                   batch_size: int = 64, max_new_tokens: int = 10) -> list[list[int]]:
    """Greedy continuation of each prompt, stopping at EOS.

    Returns:
        The generated ids for each prompt, EOS excluded.
    """
    model.eval()
    eos, pad = sp.eos_id(), sp.pad_id()
    outputs: list[list[int]] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        lengths = torch.tensor([len(p) for p in batch], device=device)
        width = int(lengths.max()) + max_new_tokens
        buffer = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        for row, prompt in enumerate(batch):
            buffer[row, : len(prompt)] = torch.tensor(prompt, device=device)

        generated: list[list[int]] = [[] for _ in batch]
        finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
        rows = torch.arange(len(batch), device=device)

        for _ in range(max_new_tokens):
            logits = model(buffer[:, : int(lengths.max())]).logits
            next_tokens = logits[rows, lengths - 1].argmax(dim=-1)
            for row in (~finished).nonzero().flatten().tolist():
                token = int(next_tokens[row])
                if token == eos:
                    finished[row] = True
                else:
                    generated[row].append(token)
            buffer[rows, lengths] = next_tokens
            lengths = lengths + 1
            if bool(finished.all()):
                break

        outputs.extend(generated)
    return outputs


@torch.no_grad()
def choice_scores(model, sp, prompts: list[list[int]], choices: list[list[str]],
                  device: torch.device, batch_size: int = 128) -> list[list[float]]:
    """Summed log-probability of every candidate answer after its prompt.

    Returns:
        For each example, one score per candidate, in the example's order.
    """
    model.eval()
    pad = sp.pad_id()
    # Flatten (example, candidate) pairs so candidates batch together.
    pairs = []
    for index, (prompt, options) in enumerate(zip(prompts, choices)):
        for option in options:
            pairs.append((index, prompt, sp.encode(option, out_type=int)))

    flat_scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        width = max(len(p) + len(c) for _, p, c in batch)
        x = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        for row, (_, prompt, candidate) in enumerate(batch):
            ids = prompt + candidate
            x[row, : len(ids)] = torch.tensor(ids, device=device)
        log_probs = torch.log_softmax(model(x).logits.float(), dim=-1)
        for row, (_, prompt, candidate) in enumerate(batch):
            # Token at position t is predicted by the logits at t - 1.
            positions = torch.arange(len(prompt) - 1, len(prompt) + len(candidate) - 1,
                                     device=device)
            targets = torch.tensor(candidate, device=device)
            flat_scores.append(float(log_probs[row, positions, targets].sum()))

    per_example: list[list[float]] = [[] for _ in prompts]
    for (index, _, _), score in zip(pairs, flat_scores):
        per_example[index].append(score)
    return per_example


def is_prefix_match(prediction: str, answer: str) -> bool:
    """The prediction starts with the whole answer, then a boundary or nothing."""
    if not prediction.startswith(answer):
        return False
    rest = prediction[len(answer):]
    return not rest or rest[0] in _WORD_BOUNDARY


def _summarise(rows: list[dict]) -> dict:
    """Mean of each metric, plus the chance baseline, over a set of rows."""
    n = len(rows)
    return {
        "examples": n,
        "exact_match": sum(r["exact"] for r in rows) / n,
        "prefix_match": sum(r["prefix"] for r in rows) / n,
        "choice_accuracy": sum(r["choice_correct"] for r in rows) / n,
        "chance": sum(1 / len(r["choices"]) for r in rows) / n,
    }


def evaluate_split(model, sp, records: list[dict], device: torch.device,
                   batch_size: int = 64) -> tuple[dict, list[dict]]:
    """Every metric for one split, overall and broken down.

    Returns:
        ``(summary, rows)`` -- the summary has ``overall`` and breakdowns by
        task, attribute, hops and task-by-hops; ``rows`` is one record per
        example with the prediction, for qualitative analysis.
    """
    prompts = [sp.encode(r["prompt"], out_type=int) for r in records]
    generated = greedy_answers(model, sp, prompts, device, batch_size)
    scores = choice_scores(model, sp, prompts, [r["choices"] for r in records],
                           device, batch_size * 2)

    rows = []
    for record, ids, candidate_scores in zip(records, generated, scores):
        prediction = sp.decode(ids).strip()
        best = max(range(len(candidate_scores)), key=candidate_scores.__getitem__)
        rows.append({
            "id": record["id"],
            "task": record["task"],
            "attribute": record["attribute"],
            "question": record["question"],
            "hops": record["hops"],
            "prompt": record["prompt"],
            "answer": record["answer"],
            "prediction": prediction,
            "exact": prediction == record["answer"],
            "prefix": is_prefix_match(prediction, record["answer"]),
            "choices": record["choices"],
            "choice_scores": candidate_scores,
            "choice_prediction": record["choices"][best],
            "choice_correct": record["choices"][best] == record["answer"],
        })

    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups["task"][row["task"]].append(row)
        groups["attribute"][row["attribute"]].append(row)
        groups["hops"][str(row["hops"])].append(row)
        groups["task_hops"][f"{row['task']}/{row['hops']}"].append(row)
        groups["question"][f"{row['task']}/{row['question']}"].append(row)

    summary = {"overall": _summarise(rows)}
    for name, buckets in groups.items():
        summary[f"by_{name}"] = {key: _summarise(value)
                                 for key, value in sorted(buckets.items())}
    return summary, rows


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate reasoning accuracy.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["test", "test_long"])
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path,
                        help="results JSON; per-example predictions go beside it")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.threads:
        torch.set_num_threads(args.threads)

    device = resolve_device(args.device)
    model, state = load_model(args.checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if sp.get_piece_size() != model.cfg.vocab_size:
        raise ValueError("tokenizer and checkpoint vocabularies differ")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    results = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "stage": payload.get("extra", {}).get("stage", "pretraining"),
        "step": state["step"],
        "splits": {},
    }
    del payload

    predictions = {}
    for split in args.splits:
        records = read_jsonl(args.data / f"{split}.jsonl")
        summary, rows = evaluate_split(model, sp, records, device, args.batch_size)
        results["splits"][split] = summary
        predictions[split] = rows
        overall = summary["overall"]
        log.info("%s %s: exact %.3f  prefix %.3f  choice %.3f  (chance %.3f, n=%d)",
                 args.label, split, overall["exact_match"], overall["prefix_match"],
                 overall["choice_accuracy"], overall["chance"], overall["examples"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    predictions_path = args.out.with_name(args.out.stem + "_predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for split, rows in predictions.items():
            for row in rows:
                handle.write(json.dumps({"split": split, **row}, ensure_ascii=False) + "\n")
    log.info("wrote %s and %s", args.out, predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
