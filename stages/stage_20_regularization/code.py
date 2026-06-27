"""Stage 20: L1 & L2 regularization.

Adds a weight penalty to the loss: L_tilde = L + lam * R, where
R2 = 0.5 * sum(theta**2) (grad theta) and R1 = sum(|theta|) (grad sign(theta)).
Reuses the stage_11 Tensor engine and stage_12 losses; penalties backprop
through Tensor.backward(). Allowed: numpy (forward only), stdlib, imported Tensor/losses.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

# Tensor (09); _as_tensor/mse/mae/cross_entropy losses (13).
from dlfs import stage_import

Stage11_Tensor = stage_import("stage_11", "Tensor")
(
    Stage12__as_tensor,
    Stage12_mse_loss,
    Stage12_mae_loss,
    Stage12_cross_entropy_loss,
) = stage_import(
    "stage_12", "_as_tensor", "mse_loss", "mae_loss", "cross_entropy_loss"
)

# Re-export engine + losses under canonical names for downstream stages/tests.
# Prefer stage_11's broadcasting Tensor; fall back if stage_11 not ready.
Tensor = Stage11_Tensor
_as_tensor = Stage12__as_tensor
mse_loss = Stage12_mse_loss
mae_loss = Stage12_mae_loss
cross_entropy_loss = Stage12_cross_entropy_loss


def _abs(t: "Stage11_Tensor") -> "Stage11_Tensor":
    """Elementwise |t| via the relu identity |t| = relu(t) + relu(-t), so the
    sub-gradient sign(t) flows through autodiff (do NOT use np.abs on the graph)."""
    # TODO: implement |t| from Tensor ops so its gradient is sign(t)
    raise NotImplementedError("_abs")


def l2_penalty(params: Iterable, lam: float = 1.0) -> "Stage11_Tensor":
    """L2 (ridge) penalty: returns scalar Tensor lam * 0.5 * sum(theta**2).
    Built from Tensor ops so grad lam*theta comes from backward(). lam==0 -> 0 Tensor."""
    # TODO: implement the L2 penalty as a scalar Tensor (no hand-written grad)
    raise NotImplementedError("l2_penalty")


def l1_penalty(params: Iterable, lam: float = 1.0) -> "Stage11_Tensor":
    """L1 (lasso) penalty: returns scalar Tensor lam * sum(|theta|), using _abs
    so the sub-gradient lam*sign(theta) flows through backward(). lam==0 -> 0 Tensor."""
    # TODO: implement the L1 penalty as a scalar Tensor via _abs
    raise NotImplementedError("l1_penalty")


def regularized_loss(
    loss: "Stage11_Tensor",
    params: Iterable,
    *,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "Stage11_Tensor":
    """Add penalties to a stage_12 data loss: L_tilde = loss + l1*sum|theta| +
    l2*0.5*sum(theta**2). With l1==l2==0 the result must equal loss."""
    # TODO: coerce loss via stage_12's _as_tensor, then add the l1/l2 penalties
    raise NotImplementedError("regularized_loss")


def l2_grad_equals_weight_decay(
    params: Iterable, lam: float
) -> List[np.ndarray]:
    """Plain-NumPy reference (no autodiff): analytic L2-penalty gradient lam*p.data
    per param, equal to stage_17's coupled weight_decay term."""
    # TODO: implement the analytic per-param L2 gradient in NumPy
    raise NotImplementedError("l2_grad_equals_weight_decay")
