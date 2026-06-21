"""Stage 18: Adam (Adaptive Moment Estimation).

Top of the optimizer chain: imports the Optimizer base / SGD / Momentum from
prior stages and adds RMSProp, Adam (momentum + 2nd-moment EMA + bias
correction), and AdamW. Optimizers consume p.grad; they never call backward().
NumPy + stdlib only.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

# Optimizer base/SGD (14), Momentum (17); Adam subclasses the base.
from dlfs import stage_import

Stage14_Optimizer, Stage14_SGD = stage_import("stage_14", "Optimizer", "SGD")
Stage17_Momentum = stage_import("stage_17", "Momentum")

# Canonical re-exports so the whole optimizer family imports from here.
Optimizer = Stage14_Optimizer
SGD = Stage14_SGD
Momentum = Stage17_Momentum


class RMSProp(Stage14_Optimizer):
    """RMSProp: per-parameter step scaled by an EMA of the squared gradient.

    g = p.grad + weight_decay*p.data; v = beta*v + (1-beta)*g**2;
    p.data -= lr * g / (sqrt(v) + eps).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        beta: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        # Plumbing (provided): store hyper-params and allocate one EMA buffer per param.
        super().__init__(params)
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Apply one in-place RMSProp update to every parameter; leave p.grad untouched."""
        # TODO: implement the RMSProp update (update self.v, then p.data).
        raise NotImplementedError("RMSProp.step")


class Adam(Stage14_Optimizer):
    """Adam: 1st-moment EMA + 2nd-moment EMA + bias correction. On step t with g:
        m = beta1*m + (1-beta1)*g;  v = beta2*v + (1-beta2)*g**2
        m_hat = m/(1-beta1**t);  v_hat = v/(1-beta2**t)
        p.data -= lr * m_hat / (sqrt(v_hat) + eps)
    Coupled weight decay (folded into g); see AdamW for decoupled."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        # Plumbing (provided): store hyper-params, init step counter, allocate m/v buffers.
        super().__init__(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def _effective_grad(self, p):
        """The gradient the update uses (coupled decay folded in). Hook overridden by AdamW."""
        # TODO: return the effective gradient including coupled weight decay.
        raise NotImplementedError("Adam._effective_grad")

    def step(self) -> None:
        """Apply one in-place Adam update (increment t, update m/v with bias correction, step p.data)."""
        # TODO: implement the Adam update (mutate self.m / self.v / p.data); leave p.grad.
        raise NotImplementedError("Adam.step")


class AdamW(Adam):
    """Adam with decoupled weight decay: decay applied to p.data directly, not folded into g.

    Same constructor signature as Adam.
    """

    def _effective_grad(self, p):
        """AdamW's adaptive update uses the raw gradient only (decay is decoupled)."""
        # TODO: return the raw gradient (no weight decay folded in).
        raise NotImplementedError("AdamW._effective_grad")

    def step(self) -> None:
        """Run the standard Adam update, then apply decoupled weight decay to p.data."""
        # TODO: super().step() then decoupled decay on p.data. Document decay ordering.
        raise NotImplementedError("AdamW.step")
