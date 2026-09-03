"""Reading batches out of a packed token array.

**Training batches are a pure function of the step number.** Offsets for step
``s`` come from a generator seeded with ``(seed, s)``, so the batch at step
5000 is the same batch whether the run reached it in one go or resumed there
after a crash. The usual alternative -- draw from a long-lived RNG and
checkpoint its state -- also works, but it makes correct resume depend on
serialising and restoring that state exactly, and it silently degrades to
"replays the same data" if that is ever missed. Deriving the batch from the
step removes the failure mode instead of guarding against it, which matters
because Kaggle sessions are killed routinely and the brief marks resume.

**Evaluation is deterministic and exhaustive**, not sampled: consecutive
non-overlapping windows over the whole split, so held-out loss is a property
of the split rather than of which windows happened to be drawn. Stepping by
``block_size`` with windows of ``block_size + 1`` means every token except the
first is predicted exactly once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

log = logging.getLogger(__name__)


class TokenArray:
    """A memory-mapped flat array of token ids.

    Memory-mapping rather than loading: the Hindi training array is about a
    gigabyte, and only the windows actually sampled need to be resident.
    """

    def __init__(self, path: str | Path, dtype: str = "uint16") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- run common.data.run pack first"
            )
        self.tokens = np.memmap(self.path, dtype=np.dtype(dtype), mode="r")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def max_start(self, block_size: int) -> int:
        """Largest offset a window of ``block_size + 1`` ids can start at."""
        return len(self) - block_size - 1


def _to_tensors(windows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``(B, block_size + 1)`` id windows into inputs and targets.

    ``x[:, t]`` is the token at position ``t`` and ``y[:, t]`` the one that
    follows it, which is the shift the cross-entropy objective expects. Ids are
    widened to int64 because that is what ``nn.Embedding`` indexes with.
    """
    windows = torch.from_numpy(windows.astype(np.int64))
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def training_batch(
    tokens: TokenArray, batch_size: int, block_size: int, seed: int, step: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The batch for a given step: reproducible from ``(seed, step)`` alone.

    Args:
        tokens: The packed training split.
        batch_size: Sequences per batch (per micro-step, before accumulation).
        block_size: Sequence length.
        seed: Run seed, recorded in the checkpoint.
        step: Global micro-step index.

    Returns:
        ``(x, y)``, each ``(batch_size, block_size)`` int64.
    """
    high = tokens.max_start(block_size)
    if high <= 0:
        raise ValueError(
            f"{tokens.path} holds {len(tokens)} tokens, too few for "
            f"block_size {block_size}"
        )
    # A SeedSequence built from both values: independent draws per step, and
    # no state carried between calls.
    rng = np.random.default_rng([seed, step])
    starts = rng.integers(0, high, size=batch_size)
    windows = np.stack([tokens.tokens[s : s + block_size + 1] for s in starts])
    return _to_tensors(windows)


def evaluation_windows(
    tokens: TokenArray, batch_size: int, block_size: int, max_batches: int | None = None
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Iterate the whole split in consecutive non-overlapping windows.

    Args:
        tokens: The packed split to evaluate on.
        batch_size: Windows per yielded batch.
        block_size: Sequence length.
        max_batches: Stop early. Used for the periodic validation check during
            training; the final reported numbers run over everything.

    Yields:
        ``(x, y)`` batches. A trailing partial window is dropped, so callers
        that need a byte count must scale it by the fraction actually scored.
    """
    starts = range(0, tokens.max_start(block_size), block_size)
    batch: list[np.ndarray] = []
    emitted = 0

    for start in starts:
        batch.append(np.asarray(tokens.tokens[start : start + block_size + 1]))
        if len(batch) == batch_size:
            yield _to_tensors(np.stack(batch))
            batch = []
            emitted += 1
            if max_batches is not None and emitted >= max_batches:
                return

    if batch and (max_batches is None or emitted < max_batches):
        yield _to_tensors(np.stack(batch))
