"""Sampling continuations from a trained model, and the held-out prompts to test on.

**Decoding.** ``temperature`` divides the logits before the softmax. Below 1 it
sharpens the distribution towards the argmax, above 1 it flattens it towards
uniform, and 0 is exactly greedy decoding. The brief asks for greedy plus 0.5,
1.0 and 1.5, which spans the useful range: 0.5 is repetitive but locally fluent,
1.5 is diverse but incoherent, and the interesting question is where between
them a 24M model trained on a few hundred million tokens sits.

**Context cropping.** Learned absolute positional embeddings only exist for
``block_size`` positions, so once the generated sequence is longer than that,
the model is fed its last ``block_size`` tokens. Generation therefore never
attends further back than its training context -- a hard limit of this
positional scheme, and one of the things the brief asks to be explained.

**Prompts come from the test split**, which the tokenizer was not fitted on and
the model never saw. Each prompt is a fixed number of tokens taken from the
start of a held-out document, and the reference continuation is the tokens that
actually followed. Because every prompt has the same token length, they batch
into a single generation pass.

The reference is decoded from *token ids* rather than sliced out of the raw
text, so reference and hypothesis have passed through the same tokenizer
round-trip and any loss in that round-trip affects both equally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from common.model.gpt import GPT
from common.scrape.io_utils import read_jsonl

log = logging.getLogger(__name__)


@dataclass
class HeldOutPrompt:
    """One test-split prompt with the continuation that truly followed it.

    Attributes:
        doc_id: Source document, so a sample can be traced back.
        prompt_ids: Token ids fed to the model.
        prompt_text: Those ids decoded, for the report.
        reference_text: The true continuation, decoded from ids.
    """

    doc_id: str
    prompt_ids: list[int]
    prompt_text: str
    reference_text: str


def build_prompts(
    split_path,
    sp,
    n_prompts: int = 64,
    prompt_tokens: int = 64,
    continuation_tokens: int = 96,
    seed: int = 20260904,
) -> list[HeldOutPrompt]:
    """Take evenly spaced held-out documents and cut each into prompt + reference.

    Documents are taken at a fixed stride through the split rather than at
    random, so the selection is reproducible from this function alone and is
    spread across the sources rather than concentrated in whichever source was
    written first.

    Args:
        split_path: ``test.jsonl.gz``.
        sp: This language's loaded SentencePiece processor.
        n_prompts: How many prompts to build.
        prompt_tokens: Tokens of context given to the model.
        continuation_tokens: Tokens of reference continuation.
        seed: Reserved so the selection can be varied reproducibly.

    Returns:
        Up to ``n_prompts`` prompts. Documents too short to supply both a
        prompt and a full reference are skipped.
    """
    needed = prompt_tokens + continuation_tokens
    # A first pass counts documents so the stride can spread the selection
    # across the whole split rather than taking the first N.
    candidates = []
    for record in read_jsonl(split_path):
        if len(record.get("text", "")) >= needed * 2:
            candidates.append(record)

    if not candidates:
        raise ValueError(f"{split_path}: no documents long enough for prompting")

    stride = max(1, len(candidates) // n_prompts)
    prompts: list[HeldOutPrompt] = []
    for record in candidates[::stride]:
        ids = sp.encode(record["text"], out_type=int)
        if len(ids) < needed:
            continue
        prompt_ids = ids[:prompt_tokens]
        reference_ids = ids[prompt_tokens:needed]
        prompts.append(
            HeldOutPrompt(
                doc_id=record["id"],
                prompt_ids=prompt_ids,
                prompt_text=sp.decode(prompt_ids),
                reference_text=sp.decode(reference_ids),
            )
        )
        if len(prompts) == n_prompts:
            break

    log.info("%s: built %d prompts of %d tokens", split_path, len(prompts),
             prompt_tokens)
    return prompts


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Extend each sequence in ``idx`` by ``max_new_tokens`` tokens.

    Args:
        model: Trained model, used in eval mode.
        idx: Prompt token ids, ``(B, T)``.
        max_new_tokens: Tokens to append.
        temperature: 0 for greedy decoding, otherwise the softmax temperature.
        top_k: If set, sample only from the k most likely tokens.

    Returns:
        ``(B, T + max_new_tokens)`` including the prompt.
    """
    was_training = model.training
    model.eval()
    block_size = model.cfg.block_size

    for _ in range(max_new_tokens):
        # Only the last block_size tokens have positional embeddings.
        context = idx[:, -block_size:]
        logits = model(context).logits[:, -1, :]

        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

    model.train(was_training)
    return idx


@torch.no_grad()
def generate_for_prompts(
    model: GPT,
    prompts: list[HeldOutPrompt],
    sp,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    batch_size: int = 16,
    top_k: int | None = None,
    seed: int = 20260904,
) -> list[str]:
    """Decode one continuation per prompt at a single temperature.

    The generator is re-seeded per call so that a reported sample is
    reproducible from the checkpoint and the temperature alone.
    """
    torch.manual_seed(seed)
    outputs: list[str] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        idx = torch.tensor([p.prompt_ids for p in batch], dtype=torch.long,
                           device=device)
        completed = generate(model, idx, max_new_tokens, temperature, top_k)
        for row in completed[:, idx.size(1):].tolist():
            outputs.append(sp.decode(row))

    return outputs
