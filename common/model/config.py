"""Model geometry, and the parameter arithmetic that picks it.

The brief fixes a budget -- roughly 25M trainable parameters per model -- and
asks for the exact count and a justification of the depth/width trade-off. Both
come from here, so the number in the report is computed from the same config
object the model is built from and cannot drift away from it.

**The arithmetic.** With model dimension ``d``, ``L`` layers and feed-forward
width ``4d``, each block holds ``4d^2`` of attention (the Q, K, V and output
projections) and ``8d^2`` of feed-forward, so ``12d^2`` per block. The
embedding table adds ``V*d`` on top, once, because the output head is tied to
it.

At ``V = 16000`` (the size both Phase 1 sweeps selected) that gives:

===========================  ==========  ==========  ===========
geometry                     embedding   blocks      total
===========================  ==========  ==========  ===========
d=384, L=10, h=6              6.14M       17.74M      24.09M
d=384, L=11, h=6              6.14M       19.52M      25.86M
d=512, L=6,  h=8              8.19M       18.91M      27.37M
===========================  ==========  ==========  ===========

``d=384, L=10`` is the chosen geometry. Depth is preferred to width at a fixed
budget because each additional block is another sequential transformation of
the residual stream -- more composition -- while width only widens each one;
ten layers is well inside the range where a pre-norm stack trains without
tricks. Head dimension is ``384/6 = 64``, the size GPU tensor cores are
happiest with.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GPTConfig:
    """Geometry and regularisation for one monolingual model.

    Attributes:
        vocab_size: Size of this language's vocabulary, from its Phase 1
            tokenizer. Never shared between Model H and Model L.
        block_size: Maximum sequence length. Learned positional embeddings
            make this a hard ceiling: the table has exactly ``block_size``
            rows, so position ``block_size`` has no embedding to look up.
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads. Must divide ``d_model``.
        d_model: Residual stream width.
        d_ff: Feed-forward inner width. Defaults to ``4 * d_model``.
        dropout: Applied to embeddings, attention probabilities and sublayer
            outputs.
        bias: Whether linear layers carry bias terms.
        tie_embeddings: Share the input embedding matrix with the output
            projection. Saves ``vocab_size * d_model`` parameters -- 6.1M here,
            a quarter of the whole budget.
        use_positional: Whether position is injected at all. False is the
            optional ablation the brief describes as a bonus; it exists as a
            flag so the ablated model runs the same code path as the standard
            one.
    """

    vocab_size: int
    block_size: int = 512
    n_layer: int = 10
    n_head: int = 6
    d_model: int = 384
    d_ff: int | None = None
    dropout: float = 0.1
    bias: bool = True
    tie_embeddings: bool = True
    use_positional: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_head:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_head={self.n_head}"
            )
        if self.d_ff is None:
            object.__setattr__(self, "d_ff", 4 * self.d_model)

    @property
    def d_head(self) -> int:
        """Width of the subspace each attention head works in."""
        return self.d_model // self.n_head

    def parameter_count(self) -> dict[str, int]:
        """Trainable parameters, analytically, broken down by component.

        Computed from the geometry rather than by instantiating the model, so
        a geometry can be costed before anything is built. ``common.model.gpt``
        asserts the assembled model agrees with this, which is what makes the
        breakdown trustworthy as a report table rather than just a comment.
        """
        d, ff = self.d_model, self.d_ff
        bias = int(self.bias)

        embedding = self.vocab_size * d
        positional = self.block_size * d if self.use_positional else 0

        attention = (d * 3 * d + 3 * d * bias) + (d * d + d * bias)
        feed_forward = (d * ff + ff * bias) + (ff * d + d * bias)
        norms = 2 * (d + d * bias)  # two LayerNorms: a weight, and a bias if used
        blocks = self.n_layer * (attention + feed_forward + norms)

        head = 0 if self.tie_embeddings else self.vocab_size * d

        counts = {
            "embedding": embedding,
            "positional": positional,
            "blocks": blocks,
            "final_norm": d + d * bias,
            "lm_head": head,
        }
        counts["total"] = sum(counts.values())
        return counts

    def with_(self, **changes) -> "GPTConfig":
        """A copy with some fields replaced, for sweeps and ablations."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        """Plain dict, for writing into checkpoints and run logs."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GPTConfig":
        """Load a geometry from one of the per-language ``model.yaml`` files.

        Unknown keys are an error rather than a silent no-op: a typo in a
        config file would otherwise train a model that quietly differs from
        the one the file describes.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"{path}: unknown keys {sorted(unknown)}; "
                f"expected only {sorted(f.name for f in fields(cls))}"
            )
        return cls(**raw)
