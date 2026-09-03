"""One transformer block: attention, feed-forward, residuals, normalisation.

**Pre-norm, not post-norm.** The block computes::

    x = x + attn(norm(x))
    x = x + mlp(norm(x))

so the LayerNorm sits *inside* each residual branch and the ``x +`` path runs
from input to output untouched. Gradient reaching the output therefore reaches
layer 0 undistorted, which is what lets a ten-layer stack train without warmup
tricks or careful residual scaling. Post-norm -- ``x = norm(x + attn(x))`` --
renormalises that path once per sublayer, so the signal a deep layer receives
depends on everything above it; it is trainable but fragile, and was the reason
the original transformer needed a long warmup. A final normalisation after the
last block, added by the model, compensates for the fact that a pre-norm stack
never normalises its own output.

**Why the feed-forward is here at all.** Attention mixes information *across*
positions but is, per position, close to linear. The MLP is where each position
gets its own non-linear computation, and it holds two thirds of the block's
parameters (``8d^2`` against ``4d^2``). Removing it leaves a stack that can
route information but barely transform it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.model.attention import CausalSelfAttention
from common.model.config import GPTConfig


class FeedForward(nn.Module):
    """Position-wise MLP: ``d_model -> d_ff -> d_model`` with GELU between.

    Applied independently and identically at every position -- it holds no
    notion of sequence, which is precisely the division of labour with
    attention.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model)`` in, ``(B, T, d_model)`` out."""
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """Attention sublayer then feed-forward sublayer, each pre-normed."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_mlp = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Transform ``(B, T, d_model)`` in place in the residual stream.

        Args:
            x: Input activations.
            return_attention: Also return this block's attention probabilities,
                ``(B, n_head, T, T)``, for the attention analysis.
        """
        if return_attention:
            delta, probs = self.attn(self.ln_attn(x), return_attention=True)
            x = x + delta
            x = x + self.mlp(self.ln_mlp(x))
            return x, probs

        x = x + self.attn(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x
