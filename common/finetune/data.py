"""Turning reasoning examples into supervised batches.

Pretraining reads one flat token stream and predicts every token. Finetuning
on question/answer pairs is different in two ways, and both live here.

**Loss only on the answer.** Each example is encoded as::

    prompt tokens | answer tokens | EOS

and the target at every prompt position is set to -1, which the model's
cross-entropy ignores. The model is therefore trained to produce the answer
*given* the question, not to model the question text itself. Were the prompt
included, most of each example's loss would come from predicting names and
numbers that are random by construction -- unlearnable noise that would swamp
the one or two answer tokens the finetune exists to teach. The EOS target is
kept: it teaches the model to stop after the answer, which is what lets the
evaluation read an exact answer back out of free generation.

**The prompt and answer are encoded separately** and concatenated as ids,
rather than encoding the joined string. The evaluation feeds the model exactly
``encode(prompt)`` and scores ``encode(answer)`` after it, so training must see
the same boundary; encoding the joined text could merge the space before the
answer into a different piece and train on a tokenisation the model is never
shown at test time.

**Right padding.** Examples in a batch differ in length, so shorter ones are
padded at the end with the pad id and target -1. Because attention is causal,
padding placed *after* a sequence can never influence its real positions, and
positions stay aligned with the learned positional table -- which left padding
would break, by shifting every token to a position it never held in
pretraining.

**Order.** Batches are drawn from a fresh permutation each epoch, and the batch
at any micro-step is a pure function of ``(seed, micro_step)``, exactly as the
pretraining loader is, so a resumed finetune continues with the batch it would
have seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EncodedExample:
    """One example as token ids.

    Attributes:
        ids: ``prompt + answer + [eos]``.
        prompt_length: How many leading ids are prompt; the first supervised
            prediction is the one made *at* position ``prompt_length - 1``.
    """

    ids: tuple[int, ...]
    prompt_length: int


def encode_example(record: dict, sp) -> EncodedExample:
    """Encode one reasoning record with this language's tokenizer."""
    prompt = sp.encode(record["prompt"], out_type=int)
    answer = sp.encode(record["answer"], out_type=int)
    return EncodedExample(tuple(prompt + answer + [sp.eos_id()]), len(prompt))


def encode_examples(records: list[dict], sp, block_size: int) -> list[EncodedExample]:
    """Encode a split, refusing anything longer than the model's context."""
    encoded = [encode_example(r, sp) for r in records]
    longest = max(len(e.ids) for e in encoded)
    if longest > block_size + 1:
        raise ValueError(f"an example is {longest} tokens; block_size is {block_size}")
    return encoded


def collate(examples: list[EncodedExample], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch and mask every non-answer target.

    Returns:
        ``(x, y)``, each ``(B, T)`` int64, where ``T`` is the longest example
        minus one. ``y`` is -1 on prompt positions and on padding.
    """
    length = max(len(e.ids) for e in examples) - 1
    x = torch.full((len(examples), length), pad_id, dtype=torch.long)
    y = torch.full((len(examples), length), -1, dtype=torch.long)
    for row, example in enumerate(examples):
        ids = torch.tensor(example.ids, dtype=torch.long)
        n = len(ids) - 1
        x[row, :n] = ids[:-1]
        y[row, :n] = ids[1:]
        # Targets at positions < prompt_length - 1 are prompt tokens.
        y[row, : example.prompt_length - 1] = -1
    return x, y


def batch_indices(n_examples: int, batch_size: int, seed: int, micro_step: int) -> list[int]:
    """Example indices for a micro-step, reproducible from ``(seed, micro_step)``.

    The stream is the concatenation of per-epoch permutations; micro-step ``k``
    takes the ``batch_size`` entries starting at ``k * batch_size``, wrapping
    into the next epoch's permutation when an epoch runs out.
    """
    start = micro_step * batch_size
    indices = []
    for position in range(start, start + batch_size):
        epoch, offset = divmod(position, n_examples)
        permutation = np.random.default_rng([seed, epoch]).permutation(n_examples)
        indices.append(int(permutation[offset]))
    return indices


def training_batch(examples: list[EncodedExample], batch_size: int, seed: int,
                   micro_step: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The padded, masked batch for one micro-step."""
    chosen = [examples[i] for i in batch_indices(len(examples), batch_size, seed, micro_step)]
    return collate(chosen, pad_id)


def evaluation_batches(examples: list[EncodedExample], batch_size: int, pad_id: int,
                       max_batches: int | None = None):
    """Consecutive batches over a split, in order."""
    for number, start in enumerate(range(0, len(examples), batch_size)):
        if max_batches is not None and number >= max_batches:
            return
        yield collate(examples[start : start + batch_size], pad_id)
