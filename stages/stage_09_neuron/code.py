"""Stage 09: Neuron.

A single neuron ``y = phi(x @ w + b)`` built on stage_08's autodiff ``Tensor``.
Wire up the forward expression only; gradients flow through ``Tensor.backward()``.
"""

from __future__ import annotations

import numpy as np

from dlfs import stage_import

# Tensor engine from stage_08, re-exported as the public ``Tensor``.
Stage8_Tensor = stage_import("stage_08", "Tensor")
Tensor = Stage8_Tensor

class Neuron:
    """A single neuron: ``y = phi(x @ w + b)``, built on stage_08's ``Tensor``."""

    _NONE_ACTIVATION = str(None).lower()
    _VALID_ACTIVATIONS = ["tanh", "relu", _NONE_ACTIVATION]

    def __init__(self, n_in: int, activation: str = "tanh", seed: int | None = None):
        """Construct leaf params: w shape (n_in,), scalar bias b=0; activation in {tanh, relu, none}."""
        rng = np.random.default_rng(seed=seed)
        self.w = Tensor(rng.random((n_in,)))
        self.b = Tensor(rng.random())
        activation = activation.lower().strip()
        assert activation in Neuron._VALID_ACTIVATIONS
        self.activation = activation

    def __call__(self, x) -> "Stage8_Tensor":
        """Forward pass: z = x @ w + b then phi(z). x shape (n_in,) or (batch, n_in)."""
        z = x @ self.w + self.b
        if self.activation == Neuron._NONE_ACTIVATION:
            return z
        else:
            return getattr(z, self.activation)()

    def parameters(self) -> list:
        """Return the learnable parameters as ``[self.w, self.b]``."""
        return [self.w, self.b]

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> str:
        """e.g. ``Neuron(n_in=3, activation='tanh')``."""
        return f"Neuron(n_in={self.w.shape[0]}, activation='{self.activation}')"
