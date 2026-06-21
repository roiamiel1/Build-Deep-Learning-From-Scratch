"""Stage 11: Dense / Linear layer.

Fully-connected layer on top of stage_09's autodiff ``Tensor``:
``Z = X @ W + b`` with X:(B, n_in), W:(n_in, n_out), b:(n_out,), Z:(B, n_out).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from dlfs import stage_import

# Tensor (stage_09): autodiff engine. Neuron (stage_10): a Dense column.
Stage9_Tensor = stage_import("stage_09", "Tensor")
Stage10_Neuron = stage_import("stage_10", "Neuron")

# Re-export under the canonical public name.
Tensor = Stage9_Tensor


class Dense:
    """Fully-connected (linear) layer ``Z = X @ W + b`` built on stage_09 Tensor."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: init leaf Tensors W (n_in, n_out) and b (n_out,); store dims/bias.
        raise NotImplementedError("Dense.__init__")

    def __call__(self, x) -> "Tensor":
        """Forward affine pass; (n_in,) -> (n_out,) or (B, n_in) -> (B, n_out).

        Bias is expanded to (B, n_out) via Tensor ops (no broadcasting backward
        until stage_12), e.g. ``ones((B, 1)) @ b.reshape(1, n_out)``.
        """
        # TODO: implement the forward pass; let Tensor.backward supply gradients.
        raise NotImplementedError("Dense.__call__")

    def parameters(self) -> List["Tensor"]:
        """Return learnable params: [W, b] with bias, else [W]."""
        # TODO: return the parameter list.
        raise NotImplementedError("Dense.parameters")

    def zero_grad(self) -> None:
        """Reset every parameter's gradient to zeros."""
        # TODO: zero each parameter's grad.
        raise NotImplementedError("Dense.zero_grad")

    def __repr__(self) -> str:
        # TODO: summarize n_in, n_out, bias.
        raise NotImplementedError("Dense.__repr__")
