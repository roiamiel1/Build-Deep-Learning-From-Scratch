"""Stage 29: Multi-Head Attention (MHA).

Runs ``h`` scaled-dot-product attention heads (each on a ``d_k = d_model // h``
subspace) in parallel, concatenates them, and mixes with a learned ``W_o``.
Composes stage_28's ``SelfAttention``; all gradients flow via ``Tensor.backward``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from dlfs import stage_import

# Tensor (09); SelfAttention/causal_mask (28) -- the head composed h times.
Stage9_Tensor = stage_import("stage_09", "Tensor")
Stage28_SelfAttention, Stage28_causal_mask = stage_import(
    "stage_28", "SelfAttention", "causal_mask"
)

# Re-export under canonical public names.
Tensor = Stage9_Tensor
SelfAttention = Stage28_SelfAttention
causal_mask = Stage28_causal_mask


class MultiHeadAttention:
    """Multi-head self-attention over ``h`` stage_28 ``SelfAttention`` heads.

    Computes ``concat(head_1(x), ..., head_h(x)) @ W_o``. Each head projects
    ``d_model -> d_k = d_model // h``; ``W_o`` has shape ``(d_model, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        causal: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Build the ``h`` heads and the ``W_o`` output projection (ValueError if
        d_model % h != 0; distinct per-head seed)."""
        self.d_model: int = d_model
        self.h: int = h
        self.d_k: int
        self.causal: bool = causal
        self.heads: List["Stage28_SelfAttention"]
        self.W_o: "Stage9_Tensor"
        self.last_attn: Optional[List["Stage9_Tensor"]] = None
        # TODO: validate divisibility; build h heads + W_o (init 1/sqrt(d_model)); last_attn=None.
        raise NotImplementedError("MultiHeadAttention.__init__")

    def forward(self, x) -> "Stage9_Tensor":
        """Run all heads on ``x``, concatenate (Tensor ops, no detach), apply ``W_o``;
        (T, d_model) -> (T, d_model). Records self.last_attn per head."""
        # TODO: run heads, concat to (T, d_model), record last_attn, return concat @ W_o.
        raise NotImplementedError("MultiHeadAttention.forward")

    def __call__(self, x) -> "Stage9_Tensor":
        """Alias for :meth:`forward`."""
        # TODO: delegate to forward
        raise NotImplementedError("MultiHeadAttention.__call__")

    def attention_weights(self, x) -> List[np.ndarray]:
        """Run ``forward(x)``; return each head's ``(T, T)`` attention as NumPy."""
        # TODO: run forward, return per-head last_attn.data
        raise NotImplementedError("MultiHeadAttention.attention_weights")

    def parameters(self) -> List["Stage9_Tensor"]:
        """Return all learnable params: every head's, then ``W_o`` (stable order)."""
        # TODO: gather each head's parameters, then append W_o
        raise NotImplementedError("MultiHeadAttention.parameters")

    def zero_grad(self) -> None:
        """Reset ``.grad`` to zeros for every parameter."""
        # TODO: zero_grad each parameter
        raise NotImplementedError("MultiHeadAttention.zero_grad")

    def __repr__(self) -> str:
        """Return e.g. ``MultiHeadAttention(d_model=12, h=4, causal=False)``."""
        return (
            f"MultiHeadAttention(d_model={self.d_model}, h={self.h}, "
            f"causal={self.causal})"
        )


# Alias expected by stage_30's loader (it looks for "MultiHeadAttention" OR "MHA").
MHA = MultiHeadAttention
