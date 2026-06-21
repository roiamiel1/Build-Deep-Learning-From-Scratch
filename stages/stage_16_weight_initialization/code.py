"""Stage 16: Weight initialization.

Turns the variance-propagation rule Var(z) = n_in * Var(W) * Var(x) into the
Xavier/Glorot (tanh/linear) and He/Kaiming (relu) schemes plus a harness that
measures activation statistics across depth. Imports Dense (stage_11) and Tensor
(stage_09) as-is; builds samplers and a measurement harness on top.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# Dense (stage_11) and Tensor (stage_09) via the shared dlfs shim, used as-is.
from dlfs import stage_import

Stage11_Dense = stage_import("stage_11", "Dense")
Stage9_Tensor = stage_import("stage_09", "Tensor")

Dense = Stage11_Dense
Tensor = Stage9_Tensor


def xavier_uniform(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Xavier/Glorot uniform init: U[-a, a], a = gain*sqrt(6/(n_in+n_out)). Returns (n_in, n_out)."""
    # TODO: sample U[-a, a] so Var(W) = gain**2 * 2 / (n_in + n_out).
    raise NotImplementedError("xavier_uniform")


def xavier_normal(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Xavier/Glorot normal init: N(0, std**2), std = gain*sqrt(2/(n_in+n_out)). Returns (n_in, n_out)."""
    # TODO: sample N(0, std**2) with std = gain*sqrt(2/(n_in+n_out)).
    raise NotImplementedError("xavier_normal")


def he_normal(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming normal init for ReLU nets: N(0, 2/n_in). Returns (n_in, n_out)."""
    # TODO: sample N(0, 2/n_in) -- numerator 2 accounts for ReLU halving variance.
    raise NotImplementedError("he_normal")


def he_uniform(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming uniform init for ReLU nets: U[-a, a], a = sqrt(6/n_in). Returns (n_in, n_out)."""
    # TODO: sample U[-a, a] so Var(W) = 2 / n_in.
    raise NotImplementedError("he_uniform")


def init_dense(layer: "Stage11_Dense", W: np.ndarray, b: Optional[np.ndarray] = None) -> None:
    """Overwrite a stage_11 Dense layer's params in place (swap .data on the same
    leaf Tensors, reset .grad). W (n_in, n_out), b (n_out,)."""
    # TODO: validate shape, swap .data on the same Tensor objects, reset .grad.
    raise NotImplementedError("init_dense")


def _apply_activation(t: "Stage9_Tensor", activation: str) -> "Stage9_Tensor":
    """Apply an elementwise activation Tensor-op by name: {"tanh", "relu", "none"}."""
    # TODO: dispatch to the matching Tensor op; raise ValueError on unknown name.
    raise NotImplementedError("_apply_activation")


def forward_activation_stats(
    sizes: List[int],
    init_fn,
    activation: str,
    *,
    n_samples: int = 512,
    seed: Optional[int] = None,
) -> List[dict]:
    """Measure post-activation statistics through a deep stack of Dense layers
    (each init via init_fn, `activation` applied after each). Returns one dict per
    layer with keys "mean", "std", "saturated" (abs>0.98), "dead" (==0)."""
    # TODO: build/init each layer, forward a noise batch, collect per-layer stats.
    raise NotImplementedError("forward_activation_stats")
