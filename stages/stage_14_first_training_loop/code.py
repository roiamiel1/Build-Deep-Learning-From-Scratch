"""Stage 14: First training loop.

Wires the imported Tensor/MLP/mse_loss/SGD into the canonical learning loop
(forward -> loss -> backward -> step -> zero_grad) plus toy datasets, an
accuracy metric, and a loss plotter. Every gradient comes from Tensor.backward().
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from dlfs import stage_import

# Framework pieces built in earlier stages, re-exported under canonical names.
Stage11_Tensor = stage_import("stage_11", "Tensor")
Stage11_MLP = stage_import("stage_11", "MLP")
Stage12_mse_loss = stage_import("stage_12", "mse_loss")
Stage13_SGD = stage_import("stage_13", "SGD")
Stage11_Dense = stage_import("stage_11", "Dense")

Tensor = Stage11_Tensor
MLP = Stage11_MLP
mse_loss = Stage12_mse_loss
SGD = Stage13_SGD
Dense = Stage11_Dense


def make_moons(
    n: int = 200, noise: float = 0.1, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-moons. Returns X (n, 2) and y in {-1, +1} (n,)."""
    # TODO: generate the half-moons dataset
    raise NotImplementedError("make_moons")


def make_spiral(
    n_per_class: int = 100,
    n_classes: int = 2,
    noise: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """n_classes-arm spiral. Returns X (n_per_class*n_classes, 2) and y in {-1, +1}."""
    # TODO: generate the spiral dataset (only binary needed here)
    raise NotImplementedError("make_spiral")


def accuracy(pred: "Stage11_Tensor", y) -> float:
    """Binary accuracy: fraction where sign(pred) == sign(y) (read-only, off-graph)."""
    # TODO: compute accuracy from sign(pred) vs sign(y)
    raise NotImplementedError("accuracy")


def train(
    model: "Stage11_MLP",
    X,
    y,
    *,
    lr: float = 0.1,
    epochs: int = 200,
    optimizer: Optional["Stage13_SGD"] = None,
) -> List[float]:
    """Run the training loop (forward -> mse_loss -> backward -> step -> zero_grad);
    return per-epoch loss history. Defaults to SGD(model.parameters(), lr) if None."""
    # TODO: implement the training loop
    raise NotImplementedError("train")


def plot_loss(history: Sequence[float], path: Optional[str] = None):
    """Plot training loss vs epoch; save to path if given. Returns the Figure."""
    # TODO: plot the loss history with matplotlib
    raise NotImplementedError("plot_loss")
