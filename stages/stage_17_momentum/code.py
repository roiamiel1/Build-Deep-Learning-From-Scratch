"""Stage 17: Momentum.

SGD with a velocity buffer (heavy-ball momentum) and an optional Nesterov
variant, built by subclassing the plain ``SGD`` from stage_14.
"""

from __future__ import annotations

from typing import Callable, Iterable, List

import numpy as np

# Tensor (09), SGD (14).
from dlfs import stage_import

Stage9_Tensor = stage_import("stage_09", "Tensor")
Stage14_SGD = stage_import("stage_14", "SGD")

# Re-export the autodiff Tensor under its canonical public name.
Tensor = Stage9_Tensor


class SGDMomentum(Stage14_SGD):
    """SGD with momentum (heavy-ball) and an optional Nesterov variant.

    Update with ``g = p.grad``:
        v <- beta * v + g
        p <- p - lr * v                  # heavy-ball (nesterov=False)
        p <- p - lr * (g + beta * v)     # nesterov=True
    Extends stage_14 ``SGD`` (same params/lr contract, inherited zero_grad).
    ``beta=0`` reduces exactly to plain SGD.
    """

    def __init__(
        self,
        params: Iterable["Tensor"],
        lr: float,
        beta: float = 0.9,
        nesterov: bool = False,
    ) -> None:
        # Defer params/lr setup to the stage_14 SGD ctor (single source of truth).
        super().__init__(params, lr)
        # TODO: validate 0 <= beta < 1, store beta/nesterov, allocate one
        # zeros velocity buffer per param (aligned with self.params).
        raise NotImplementedError("SGDMomentum.__init__")

    def step(self) -> None:
        """Apply one momentum update to every parameter in place.

        Velocity buffers persist across calls (write updated v back into
        self.velocities). Does NOT zero grads (that's the inherited zero_grad).
        """
        # TODO: implement the heavy-ball / Nesterov update.
        raise NotImplementedError("SGDMomentum.step")

    def reset(self) -> None:
        """Zero all velocity buffers; leaves p.data and p.grad untouched."""
        # TODO: reset every velocity buffer to zeros.
        raise NotImplementedError("SGDMomentum.reset")

    def __repr__(self) -> str:
        # TODO: summarize lr, beta, nesterov.
        raise NotImplementedError("SGDMomentum.__repr__")


# Backwards-compatible public alias used by the tests.
Momentum = SGDMomentum


def quadratic_descent(
    optimizer_factory: Callable[["Tensor"], "Stage14_SGD"],
    x0: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    steps: int,
) -> List[float]:
    """Minimize ``f(x) = 0.5 * x^T A x - b^T x`` (A SPD) using the closed-form
    gradient ``grad f(x) = A @ x - b``; return ``f(x)`` after each of ``steps`` updates."""
    # TODO: build a leaf p from x0, run `steps` optimizer updates, record f(x) each step.
    raise NotImplementedError("quadratic_descent")
