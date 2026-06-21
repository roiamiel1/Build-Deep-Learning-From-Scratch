"""Stage 30: Transformer block.

Canonical Transformer block from earlier pieces: MultiHeadAttention (stage_29),
a position-wise feed-forward MLP (stage_12), two hand-derived LayerNorms, and
residuals. LayerNorm standardizes each TOKEN over its FEATURES (last axis).
Shapes ``(B, L, D)`` or ``(L, D)``; ``norm=`` selects pre-norm (GPT-2+) or
post-norm (original paper). Pure NumPy (LayerNorm is outside the Tensor engine).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# MultiHeadAttention (29), MLP (12), Tensor (09) -- re-exported for later stages.
from dlfs import stage_import

Stage29_MultiHeadAttention = stage_import("stage_29", "MultiHeadAttention")
Stage12_MLP = stage_import("stage_12", "MLP")
Stage9_Tensor = stage_import("stage_09", "Tensor")


class LayerNorm:
    """Layer norm over the last (feature) axis with learnable affine
    ``y = gamma * x_hat + beta`` (gamma/beta shape (D,)); hand-derived backward."""

    def __init__(self, normalized_dim: int, *, eps: float = 1e-5) -> None:
        # TODO: store dim/eps, init gamma=ones(D)/beta=zeros(D) + grads, backward cache.
        raise NotImplementedError("LayerNorm.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Standardize each token over its features (reduce last axis, BIASED var),
        then affine. x ``(..., D)`` -> same shape. Cache for backward."""
        # TODO: implement the per-token standardize + affine; cache; return out.
        raise NotImplementedError("LayerNorm.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for :meth:`forward`."""
        raise NotImplementedError("LayerNorm.__call__")

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backprop through LayerNorm: grad_out ``(..., D)`` -> dL/dx (same shape);
        set gamma_grad/beta_grad. (stage_23's standardize backward over axis=-1.)"""
        # TODO: read cache; set gamma_grad/beta_grad; compute and return dx.
        raise NotImplementedError("LayerNorm.backward")

    def parameters(self) -> List[np.ndarray]:
        """Return the learnable parameters ``[gamma, beta]``."""
        raise NotImplementedError("LayerNorm.parameters")

    def zero_grad(self) -> None:
        """Reset ``gamma_grad`` and ``beta_grad`` to zeros."""
        raise NotImplementedError("LayerNorm.zero_grad")

    def __repr__(self) -> str:
        raise NotImplementedError("LayerNorm.__repr__")


class TransformerBlock:
    """One Transformer block: residual MHA sublayer (stage_29) + residual
    position-wise FFN sublayer (stage_12, D->d_ff->D, ReLU). ``norm`` selects
    pre-norm (GPT-2+) or post-norm (original paper) placement of the LayerNorms."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        norm: str = "pre",
        eps: float = 1e-5,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: validate norm; build attn, ln1/ln2 = LayerNorm(d_model, eps), and
        #       the position-wise FFN MLP (distinct seeds for attn and ffn).
        raise NotImplementedError("TransformerBlock.__init__")

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Run one block (shape-preserving ``(B, L, D)`` or ``(L, D)``); ``mask`` is
        an optional additive attention mask (None = full attention)."""
        # TODO: implement the "pre" and "post" residual + norm wirings.
        raise NotImplementedError("TransformerBlock.forward")

    def __call__(
        self, x: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Alias for :meth:`forward`."""
        raise NotImplementedError("TransformerBlock.__call__")

    def parameters(self) -> List[np.ndarray]:
        """Return every learnable parameter in a stable order (e.g. attn, ln1, ffn, ln2)."""
        # TODO: concatenate attn/ln1/ffn/ln2 parameters().
        raise NotImplementedError("TransformerBlock.parameters")

    def __repr__(self) -> str:
        raise NotImplementedError("TransformerBlock.__repr__")
