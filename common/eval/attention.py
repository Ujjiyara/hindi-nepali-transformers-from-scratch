"""Measuring what the hand-written attention heads actually learned.

Three quantities, all computed per (layer, head) and averaged over held-out
text. They exist because a loss curve says a model works but nothing about how.

**Attention entropy**, in nats, is ``-sum_j p_ij log p_ij`` averaged over query
positions. A head that always looks at one specific token has entropy near 0; a
head that averages uniformly over its context has entropy near ``log(i+1)`` at
query position ``i``. Because that ceiling grows with position, the raw value
is not comparable across positions, so a **normalised** entropy -- divided by
``log(i+1)``, the maximum available at that position -- is reported alongside
it. Normalised entropy near 1 means the head is doing little more than
averaging; near 0 means it has committed to specific tokens.

**Mean attention distance** is ``sum_j p_ij * (i - j)``: how far back a query
looks, on average, in token positions. This is what separates the two head
types the literature describes -- local or positional heads sit at a distance
of one to three tokens and often implement "attend to the previous token",
while content-based heads range much further and vary with what the text says.

**Query position 0 is excluded from both.** It can only attend to itself, so
its entropy is exactly 0 and its distance exactly 0 regardless of what the head
has learned; including it would drag every head's average towards zero by a
constant that depends only on sequence length.

Computation lives here and plotting lives in ``common/eval/plots.py``, so the
numbers that go in the report's tables are produced once and drawn separately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from common.model.gpt import GPT

log = logging.getLogger(__name__)


@dataclass
class AttentionStatistics:
    """Per-head summaries, each an ``(n_layer, n_head)`` array.

    Attributes:
        entropy: Mean attention entropy in nats.
        normalised_entropy: Entropy divided by the maximum available at each
            query position, in [0, 1].
        distance: Mean attention distance in token positions.
        sequences: How many held-out sequences these were averaged over.
    """

    entropy: np.ndarray
    normalised_entropy: np.ndarray
    distance: np.ndarray
    sequences: int

    def as_dict(self) -> dict:
        """Nested lists, for the results JSON the report reads."""
        return {
            "entropy": self.entropy.tolist(),
            "normalised_entropy": self.normalised_entropy.tolist(),
            "distance": self.distance.tolist(),
            "sequences": self.sequences,
            "n_layer": int(self.entropy.shape[0]),
            "n_head": int(self.entropy.shape[1]),
        }

    def summary_rows(self) -> list[dict]:
        """One row per (layer, head), for a report table."""
        rows = []
        for layer in range(self.entropy.shape[0]):
            for head in range(self.entropy.shape[1]):
                rows.append({
                    "layer": layer,
                    "head": head,
                    "entropy": float(self.entropy[layer, head]),
                    "normalised_entropy": float(
                        self.normalised_entropy[layer, head]
                    ),
                    "distance": float(self.distance[layer, head]),
                })
        return rows


def _per_head_statistics(probs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Entropy, normalised entropy and distance for one layer's probabilities.

    Args:
        probs: ``(B, n_head, T, T)`` post-softmax attention weights.

    Returns:
        Three ``(n_head,)`` tensors, averaged over batch and query positions
        1..T-1.
    """
    _, _, T, _ = probs.shape
    probs = probs.float()

    # 0 * log(0) is 0, but torch returns nan for it, so guard the log.
    plogp = torch.where(probs > 0, probs * probs.log(), torch.zeros_like(probs))
    entropy = -plogp.sum(dim=-1)  # (B, n_head, T)

    positions = torch.arange(T, device=probs.device)
    # Maximum entropy available at query i is log(i + 1) unmasked keys.
    ceiling = torch.log(positions.float() + 1).clamp(min=1e-12)
    normalised = entropy / ceiling

    # distance[i, j] = i - j, so a head attending to the previous token scores 1.
    offsets = (positions.view(-1, 1) - positions.view(1, -1)).float()
    distance = (probs * offsets).sum(dim=-1)  # (B, n_head, T)

    # Drop query 0: it can only attend to itself, so it carries no information
    # about the head and would bias every average downwards.
    return (
        entropy[:, :, 1:].mean(dim=(0, 2)),
        normalised[:, :, 1:].mean(dim=(0, 2)),
        distance[:, :, 1:].mean(dim=(0, 2)),
    )


@torch.no_grad()
def collect_statistics(
    model: GPT,
    batches,
    device: torch.device,
    max_batches: int | None = None,
) -> AttentionStatistics:
    """Average attention statistics over held-out batches.

    Args:
        model: Trained model; put in eval mode so dropout does not perturb the
            probability matrices being measured.
        batches: Iterable of ``(x, y)`` batches; only ``x`` is used.
        device: Where to run.
        max_batches: Stop after this many batches. Attention matrices are
            ``(B, n_head, T, T)`` per layer, so a handful of batches is both
            enough for a stable mean and as much as memory allows.

    Returns:
        An :class:`AttentionStatistics`.
    """
    model.eval()
    totals: list[list[torch.Tensor]] | None = None
    sequences = 0
    used = 0

    for x, _ in batches:
        x = x.to(device, non_blocking=True)
        attentions = model(x, return_attention=True).attentions

        stats = [_per_head_statistics(probs) for probs in attentions]
        if totals is None:
            totals = [[s.clone() * x.size(0) for s in layer] for layer in stats]
        else:
            for layer_total, layer_stat in zip(totals, stats):
                for accumulated, value in zip(layer_total, layer_stat):
                    accumulated += value * x.size(0)

        sequences += x.size(0)
        used += 1
        if max_batches is not None and used >= max_batches:
            break

    if totals is None:
        raise ValueError("no batches supplied")

    entropy = np.stack([(t[0] / sequences).cpu().numpy() for t in totals])
    normalised = np.stack([(t[1] / sequences).cpu().numpy() for t in totals])
    distance = np.stack([(t[2] / sequences).cpu().numpy() for t in totals])

    log.info(
        "attention over %d sequences: entropy %.3f-%.3f nats, distance %.1f-%.1f",
        sequences, entropy.min(), entropy.max(), distance.min(), distance.max(),
    )
    return AttentionStatistics(entropy, normalised, distance, sequences)


@torch.no_grad()
def capture_attention(
    model: GPT, idx: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Attention probabilities for one batch, for drawing heatmaps.

    Returns:
        ``n_layer`` tensors of shape ``(B, n_head, T, T)``, on the CPU.
    """
    model.eval()
    attentions = model(idx.to(device), return_attention=True).attentions
    return tuple(probs.float().cpu() for probs in attentions)
