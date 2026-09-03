"""The decoder-only language model: embeddings, block stack, tied output head.

**Positional embeddings are learned and absolute.** Self-attention is
permutation-equivariant -- permute the input positions and the outputs permute
with them, unchanged -- so with no position signal the model sees a bag of
tokens and cannot represent word order at all. A learned table is chosen over
sinusoidal encodings because it costs one small matrix (``block_size * d``,
0.20M parameters here), needs no extra machinery, and this project never
extrapolates past its training length. The cost is exactly that: the table has
``block_size`` rows, so the model has no representation for position 512 and
the context length is a hard architectural ceiling rather than a soft one. A
relative scheme such as RoPE would lift that, at the price of complexity the
brief does not ask for.

Token and position embeddings are combined by addition, then dropped out
together, which is the standard construction: both live in the same
``d_model``-dimensional space and the residual stream carries their sum.

**The output head is tied to the input embedding.** One matrix serves both as
"what vector represents this token" and "how strongly does this hidden state
predict this token". That saves ``vocab_size * d_model`` = 6.1M parameters --
a quarter of the whole budget, which instead pays for transformer blocks -- and
it also acts as a regulariser, since every token's row now receives gradient
from both jobs.

**Initialisation.** Weights are normal(0, 0.02), and the projections that write
into the residual stream (attention output and MLP output) are additionally
scaled by ``1/sqrt(2 * n_layer)``. Each of the ``2 * n_layer`` sublayers adds
its result into the stream, so without that damping the stream's variance grows
with depth and the last blocks receive activations far outside the range their
LayerNorms were initialised for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model.block import Block
from common.model.config import GPTConfig


@dataclass
class GPTOutput:
    """What a forward pass returns.

    Attributes:
        logits: Next-token scores, ``(B, T, vocab_size)``.
        loss: Mean cross-entropy, or None when no targets were supplied.
        attentions: Per-block attention probabilities, each
            ``(B, n_head, T, T)``, or None unless explicitly requested.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class GPT(nn.Module):
    """A monolingual decoder-only transformer language model."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = (
            nn.Embedding(cfg.block_size, cfg.d_model) if cfg.use_positional else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            # Both residual-writing projections are named "proj"; the fused qkv
            # and the MLP's inner layer are not, so this hits exactly the two
            # per block that the depth scaling is meant for.
            if name.endswith("proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        # Tie after initialisation so the head simply aliases the embedding
        # rather than overwriting it.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Normal(0, 0.02) on weights, zeros on biases -- the GPT-2 scheme."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Trainable parameters, counting a tied matrix once."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> GPTOutput:
        """Run the language model.

        Args:
            idx: Token ids, ``(B, T)``.
            targets: Next-token ids, ``(B, T)``, already shifted by the data
                loader so that ``targets[:, t]`` is the token following
                ``idx[:, t]``. Positions equal to -1 are ignored by the loss.
            return_attention: Collect every block's attention probabilities.
                Off during training -- it retains ``n_layer`` tensors of shape
                ``(B, n_head, T, T)``, which is a lot of memory.

        Returns:
            A :class:`GPTOutput`.
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.cfg.block_size}: "
                "there is no positional embedding for it"
            )

        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(T, device=idx.device)
            # (T, d) broadcasts over the batch: every sequence in the batch
            # gets the same position vectors, which is the point.
            x = x + self.position_embedding(positions)
        x = self.drop(x)

        attentions = [] if return_attention else None
        for block in self.blocks:
            if return_attention:
                x, probs = block(x, return_attention=True)
                attentions.append(probs)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten batch and time: cross-entropy is over vocabulary, and
            # every position is an independent prediction. reshape rather than
            # view because targets may be a non-contiguous slice.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return GPTOutput(
            logits=logits,
            loss=loss,
            attentions=tuple(attentions) if attentions is not None else None,
        )
