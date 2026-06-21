"""Stage 12: MLP.

A multilayer perceptron: a chain of pure-linear ``Dense`` layers (stage_11) with
a nonlinearity applied between them, built on the autodiff ``Tensor`` (stage_09).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# Building blocks from earlier stages (re-exported for tests / later stages).
from dlfs import stage_import

Stage9_Tensor = stage_import("stage_09", "Tensor")
Stage11_Dense = stage_import("stage_11", "Dense")


class MLP:
    """A multilayer perceptron: ``Dense`` layers + activations. sizes
    ``[n_in, ..., n_out]`` builds len(sizes)-1 Dense layers; activation follows
    each hidden layer, out_activation the last (each in {"tanh","relu","none"})."""

    def __init__(
        self,
        sizes: Sequence[int],
        activation: str = "tanh",
        out_activation: str = "none",
        seed: Optional[int] = None,
    ) -> None:
        # TODO: validate args; build the Dense layers (per-layer derived seeds).
        raise NotImplementedError("MLP.__init__")

    @staticmethod
    def _apply_activation(z: "Stage9_Tensor", name: str) -> "Stage9_Tensor":
        """Apply named pointwise activation via the Tensor's own methods; raise on unknown name."""
        # TODO: dispatch "none"/"tanh"/"relu" to z / z.tanh() / z.relu().
        raise NotImplementedError("MLP._apply_activation")

    def forward(self, x) -> "Stage9_Tensor":
        """Chain layers, applying activation after each (out_activation after the
        last). x ``(n_in,)`` or ``(batch, n_in)`` -> ``(n_out,)`` / ``(batch, n_out)``."""
        # TODO: chain layers with the right activation per layer.
        raise NotImplementedError("MLP.forward")

    def __call__(self, x) -> "Stage9_Tensor":
        """Alias for :meth:`forward`."""
        # TODO: delegate to forward.
        raise NotImplementedError("MLP.__call__")

    def parameters(self) -> List["Stage9_Tensor"]:
        """Return every learnable parameter from every layer, flattened in layer order."""
        # TODO: flatten each layer's parameters().
        raise NotImplementedError("MLP.parameters")

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        # TODO: zero each parameter's grad.
        raise NotImplementedError("MLP.zero_grad")

    def __repr__(self) -> str:
        # TODO: summarize sizes and activations.
        raise NotImplementedError("MLP.__repr__")
