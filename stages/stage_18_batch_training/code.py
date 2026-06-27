"""Stage 18: Mini-batch training.

Additive on top of stage_14's full-batch driver: split the data into mini-batches
and run forward/backward/step once per batch. Adds only the data-feeding layer
(``iterate_minibatches`` + ``train_minibatch``) plus noise/analysis helpers.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Reused framework: autodiff, model, loss, optimizers, datasets + full-batch driver.
from dlfs import stage_import

Stage11_Tensor = stage_import("stage_11", "Tensor")
Stage11_MLP = stage_import("stage_11", "MLP")
Stage12_mse_loss = stage_import("stage_12", "mse_loss")
Stage13_Optimizer, Stage13_SGD = stage_import("stage_13", "Optimizer", "SGD")
Stage17_Adam = stage_import("stage_17", "Adam")
Stage14_make_moons, Stage14_make_spiral, Stage14_accuracy, Stage14_train = stage_import(
    "stage_14", "make_moons", "make_spiral", "accuracy", "train"
)

# Re-export under canonical names for this stage's callers and later stages.
Tensor = Stage11_Tensor
MLP = Stage11_MLP
mse_loss = Stage12_mse_loss
Optimizer = Stage13_Optimizer
SGD = Stage13_SGD
Adam = Stage17_Adam
make_moons = Stage14_make_moons
make_spiral = Stage14_make_spiral
accuracy = Stage14_accuracy
train = Stage14_train


def iterate_minibatches(
    X,
    y,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generator yielding ``(X_b, y_b)`` mini-batches partitioning the dataset.
    X (N, n_in), y (N,) or (N, 1); ValueError if lengths differ or batch_size not in [1, N]."""
    # TODO: yield consecutive batches over a (optionally shuffled) permutation.
    raise NotImplementedError("iterate_minibatches")


def train_minibatch(
    model: "Stage11_MLP",
    X,
    y,
    *,
    lr: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
    optimizer: Optional["Stage13_Optimizer"] = None,
    drop_last: bool = False,
) -> Dict[str, object]:
    """Train ``model`` with mini-batch gradient descent; return loss history
    ``{"batch_loss", "epoch_loss", "steps"}`` (epoch_loss = size-weighted mean)."""
    # TODO: loop epochs x mini-batches running the canonical train step per batch.
    raise NotImplementedError("train_minibatch")


def gradient_noise(
    model: "Stage11_MLP",
    X,
    y,
    batch_size: int,
    *,
    n_batches: int,
    seed: Optional[int] = None,
) -> float:
    """Estimate the variance of the mini-batch gradient at a fixed model: mean over
    coordinates of the per-coordinate variance (should fall like sigma**2/batch_size)."""
    # TODO: collect per-batch flattened grads, then mean of per-coordinate var.
    raise NotImplementedError("gradient_noise")


def epochs_to_threshold(history: Sequence[float], threshold: float) -> int:
    """Return the 1-based epoch index where loss first reaches ``threshold``, else -1."""
    # TODO: scan history for the first value <= threshold.
    raise NotImplementedError("epochs_to_threshold")


def plot_batch_comparison(
    histories: Mapping[str, Sequence[float]],
    path: Optional[str] = None,
):
    """Plot several labelled epoch-loss curves on shared axes; save to ``path`` or return fig."""
    # TODO: plot one curve per history entry; save if path given, else return fig.
    raise NotImplementedError("plot_batch_comparison")
