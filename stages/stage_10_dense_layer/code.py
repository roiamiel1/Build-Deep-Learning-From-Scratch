"""Stage 10: Dense / Linear layer.

Fully-connected layer on top of stage_08's autodiff ``Tensor``:
``Z = X @ W + b`` with X:(B, n_in), W:(n_in, n_out), b:(n_out,), Z:(B, n_out).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from dlfs import stage_import

# Tensor (stage_08): autodiff engine. Neuron (stage_09): a Dense column.
Stage8_Tensor = stage_import("stage_08", "Tensor")
Stage9_Neuron = stage_import("stage_09", "Neuron")

# Re-export under the canonical public name.
Tensor = Stage8_Tensor


class Dense:
    """Fully-connected (linear) layer ``Z = X @ W + b`` built on stage_08 Tensor."""
    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.W = Tensor(np.random.default_rng(seed=seed).random((n_in, n_out)))
        if bias:
            self.b = Tensor(np.zeros(n_out,))
        else:
            self.b = None

    @property
    def n_in(self):
        return self.W.shape[0]
    
    @property
    def n_out(self):
        return self.W.shape[1]

    def __call__(self, x: "Tensor") -> "Tensor":
        """Forward affine pass; (n_in,) -> (n_out,) or (B, n_in) -> (B, n_out).

        Bias is expanded to (B, n_out) via Tensor ops (no broadcasting backward
        until stage_11), e.g. ``ones((B, 1)) @ b.reshape(1, n_out)``.
        """
        assert isinstance(x, Tensor)
        batch_size = 0 if x.data.ndim <= 1 else x.shape[0]

        z = x @ self.W

        if self.b is not None:
            if batch_size > 0:
                z += Tensor(np.ones((batch_size, 1))) @ self.b.reshape(1, self.n_out)
            else:
                z += self.b

        return z

    def parameters(self) -> List["Tensor"]:
        """Return learnable params: [W, b] with bias, else [W]."""
        if self.b is None:
            return [self.W]
        else:
            return [self.W, self.b]

    def zero_grad(self) -> None:
        """Reset every parameter's gradient to zeros."""
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> str:
        return f"Dense(n_in={self.n_in}, n_out={self.n_out})"
