"""Stage 28: Single-head scaled dot-product self-attention.

Built on the stage_09 ``Tensor`` engine. For a sequence X (T, d_model):
    Q,K,V = X@W_q, X@W_k, X@W_v;  S = (Q@K.T)/sqrt(d_k) (+mask);
    A = softmax_rows(S);  O = A@V.  All via Tensor ops so backward() supplies grads.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from dlfs import stage_import

# Tensor (09); softmax/cross_entropy_loss (13).
Stage9_Tensor = stage_import("stage_09", "Tensor")
Stage13_softmax, Stage13_cross_entropy_loss = stage_import(
    "stage_13", "softmax", "cross_entropy_loss"
)

# Canonical public name so tests / later stages can ``from code import Tensor``.
Tensor = Stage9_Tensor


def _as_tensor(x) -> "Tensor":
    """Return ``x`` as a ``Tensor`` (pass through if it already is one)."""
    # TODO: implement
    raise NotImplementedError("_as_tensor")


def causal_mask(T: int) -> np.ndarray:
    """Additive (T, T) causal mask: 0.0 for j <= i, -1e9 above diagonal (so position
    i cannot attend to future j > i)."""
    # TODO: implement
    raise NotImplementedError("causal_mask")


class SelfAttention:
    """Single-head scaled dot-product self-attention on a (T, d_model) sequence.
    W_q/W_k/W_v are (d_model, d_k); last_attn holds the most recent attention A."""

    def __init__(
        self,
        d_model: int,
        d_k: int,
        causal: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Q/K/V projection weights (d_model, d_k), scaled 1/sqrt(d_model)."""
        self.d_model: int = d_model
        self.d_k: int = d_k
        self.causal: bool = causal
        self.W_q: "Tensor"
        self.W_k: "Tensor"
        self.W_v: "Tensor"
        self.last_attn: Optional["Tensor"] = None
        # TODO: build rng from seed; create W_q/W_k/W_v; set last_attn = None.
        raise NotImplementedError("TODO: init W_q, W_k, W_v")

    def softmax_rows(self, s: "Tensor") -> "Tensor":
        """Stable row-wise softmax of a (T, T) Tensor (each row sums to 1) using the
        detached-max trick so no gradient flows through the shift. Tensor ops only."""
        # TODO: implement
        raise NotImplementedError("softmax_rows")

    def forward(self, x) -> "Tensor":
        """Scaled dot-product self-attention (T, d_model) -> (T, d_k): computes
        O = softmax_rows((Q@K.T)/sqrt(d_k) [+ mask]) @ V; stores A in self.last_attn."""
        # TODO: implement
        raise NotImplementedError("forward")

    def __call__(self, x) -> "Tensor":
        """Alias for ``forward(x)``."""
        # TODO: implement
        raise NotImplementedError("__call__")

    def attention_weights(self, x) -> np.ndarray:
        """Run forward(x) and return the (T, T) attention matrix as a NumPy array."""
        # TODO: implement
        raise NotImplementedError("attention_weights")

    def parameters(self) -> List["Tensor"]:
        """Return the learnable parameter Tensors: [W_q, W_k, W_v]."""
        # TODO: implement
        raise NotImplementedError("parameters")

    def zero_grad(self) -> None:
        """Reset ``.grad`` to zeros for every parameter."""
        # TODO: implement
        raise NotImplementedError("zero_grad")

    def __repr__(self) -> str:
        """Return e.g. ``SelfAttention(d_model=8, d_k=4, causal=False)``."""
        return (
            f"SelfAttention(d_model={self.d_model}, d_k={self.d_k}, "
            f"causal={self.causal})"
        )
