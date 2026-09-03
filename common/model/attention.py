"""Multi-head causal self-attention, built from primitives.

The brief requires this to be student-implemented: no ``nn.Transformer*``, no
pre-built attention block. Only ``nn.Linear`` and ``nn.Dropout`` are used, so
every tensor operation in the forward pass is visible here.

**The shape journey.** This is the part worth reading closely, because the
reshapes are where hand-written attention usually goes wrong::

    x                          (B, T, d)
    qkv(x)                     (B, T, 3d)      one fused matmul, not three
    split                      3 x (B, T, d)
    .view(B, T, h, d_k)        (B, T, h, d_k)  heads carved out of the LAST axis
    .transpose(1, 2)           (B, h, T, d_k)  heads become a batch dimension
    q @ k^T                    (B, h, T, T)    scores[b, i, j]: query i on key j
    * 1/sqrt(d_k)              (B, h, T, T)
    mask, softmax(dim=-1)      (B, h, T, T)    each row sums to 1
    probs @ v                  (B, h, T, d_k)
    .transpose(1, 2)           (B, T, h, d_k)  exact inverse of the split
    .contiguous().view         (B, T, d)
    proj                       (B, T, d)       shape-preserving overall

The order ``view(B, T, h, d_k)`` *then* ``transpose`` matters. The tensor is
laid out as ``(B, T, d)``, so reshaping the trailing ``d`` into ``(h, d_k)``
gives head *i* the columns ``[i*d_k : (i+1)*d_k]``. Viewing straight to
``(B, h, T, d_k)`` would instead slice along *time*, handing head 0 the first
``T/h`` tokens. That runs, trains, and produces a plausible loss curve while
being entirely wrong, which is why ``test_attention.py`` checks the result
against a reference that splits the heads by explicit column slicing.

**Why 1/sqrt(d_k).** If ``q`` and ``k`` have roughly independent unit-variance
components then ``q . k`` is a sum of ``d_k`` such products, so its variance is
``d_k`` and its standard deviation ``sqrt(d_k)`` -- 8 at ``d_k = 64``. Logits
with that spread drive softmax to near one-hot, where its gradient is close to
zero and nothing learns. Dividing by ``sqrt(d_k)`` restores unit variance and
keeps the softmax in its responsive range at initialisation.

**Why the mask is applied before the softmax.** The brief defines an additive
mask ``M`` holding 0 on allowed positions and -inf on future ones; adding -inf
and then exponentiating gives exactly zero *and* removes the term from the
softmax denominator. ``masked_fill`` below is that same operation without
materialising the float matrix. Zeroing the upper triangle *after* the softmax
would instead leave each row summing to less than one by a position-dependent
amount, silently rescaling every output. A causal row is never fully masked --
position ``i`` can always see itself -- so the ``0/0`` NaN that -inf masking
can otherwise cause does not arise here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention over a ``(B, T, d_model)`` sequence.

    Shape-preserving: the output has the same shape as the input, which is what
    lets a block write ``x = x + attn(norm(x))``.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head
        self.scale = 1.0 / math.sqrt(cfg.d_head)

        # One projection producing Q, K and V together. Mathematically the same
        # as three d->d layers; a single matmul keeps the GPU busier.
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

        # Two distinct dropouts, as the brief asks: one on the attention
        # probabilities (severing individual query-key edges) and one on the
        # sublayer output before it re-enters the residual stream.
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # A buffer, not a parameter: it follows the module under .to(device)
        # but carries no gradient and is never handed to the optimiser.
        # persistent=False keeps a constant out of every checkpoint and lets
        # block_size change without breaking state_dict loading.
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)
            ).view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attend over ``x``.

        Args:
            x: Input activations, ``(B, T, d_model)``.
            return_attention: Also return the post-softmax, pre-dropout
                probability matrix ``(B, n_head, T, T)``. Used by the Phase 2
                attention analysis; off during training so nothing is retained.

        Returns:
            ``(B, T, d_model)``, or that plus the attention probabilities.
        """
        B, T, C = x.shape
        if T > self.causal_mask.size(-1):
            raise ValueError(
                f"sequence length {T} exceeds block_size "
                f"{self.causal_mask.size(-1)}: there is no mask for it"
            )

        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        probs = F.softmax(scores, dim=-1)

        y = self.attn_dropout(probs) @ v

        # Inverse of the head split: transpose back, then flatten the head axis
        # into the feature axis. .contiguous() is required because transpose
        # returns a non-contiguous view that .view() cannot reinterpret.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))

        if return_attention:
            return y, probs
        return y
