"""Stage 15: First training loop.

Wires the imported Tensor/MLP/mse_loss/SGD into the canonical learning loop
(forward -> loss -> backward -> step -> zero_grad) plus toy datasets, an
accuracy metric, and a loss plotter. Every gradient comes from Tensor.backward().
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from dlfs import stage_import

# Framework pieces built in earlier stages, re-exported under canonical names.
Stage12_Tensor = stage_import("stage_12", "Tensor")
Stage11_MLP = stage_import("stage_11", "MLP")
Stage12_mse_loss = stage_import("stage_12", "mse_loss")
Stage14_SGD = stage_import("stage_14", "SGD")
Stage11_Dense = stage_import("stage_11", "Dense")

Tensor = Stage12_Tensor
MLP = Stage11_MLP
mse_loss = Stage12_mse_loss
SGD = Stage14_SGD
Dense = Stage11_Dense


def make_moons(
    n: int = 200, noise: float = 0.1, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-moons. Returns X (n, 2) and y in {-1, +1} (n,).

    Generates a random toy dataset used to train and test the training loop:
    a non-linearly-separable 2-class problem the MLP must learn to split.
    `seed` makes the draw reproducible; `noise` is the per-point Gaussian jitter.
    """
    rng = np.random.default_rng(seed)
    n_out = n // 2          # outer (upper) moon, label +1
    n_in = n - n_out        # inner (lower) moon, label -1

    # Outer moon: upper half-circle centered at origin.
    t_out = np.linspace(0.0, np.pi, n_out)
    x_out = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)

    # Inner moon: lower half-circle, shifted right and down so the two interlock.
    t_in = np.linspace(0.0, np.pi, n_in)
    x_in = np.stack([1.0 - np.cos(t_in), 0.5 - np.sin(t_in)], axis=1)

    X = np.concatenate([x_out, x_in], axis=0).astype(np.float64)
    y = np.concatenate([np.ones(n_out), -np.ones(n_in)]).astype(np.float64)

    X += rng.normal(0.0, noise, size=X.shape)
    return Tensor(X), Tensor(y)


def make_spiral(
    n_per_class: int = 100,
    n_classes: int = 2,
    noise: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """n_classes-arm spiral. Returns X (n_per_class*n_classes, 2) and y in {-1, +1}.

    Generates a random toy dataset used to train and test the training loop:
    interleaved spiral arms, a harder non-linearly-separable 2-class problem.
    `seed` makes the draw reproducible; `noise` is the per-point angular jitter.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * n_classes, 2), dtype=np.float64)
    y = np.zeros(n_per_class * n_classes, dtype=np.float64)

    for c in range(n_classes):
        idx = slice(c * n_per_class, (c + 1) * n_per_class)
        r = np.linspace(0.0, 1.0, n_per_class)                       # radius 0 -> 1
        # Each arm starts a full turn offset from the last; noise jitters the angle.
        theta = (
            np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class)
            + rng.normal(0.0, noise, size=n_per_class)
        )
        X[idx] = np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1)
        # Binary labels in {-1, +1}: even arms -> +1, odd arms -> -1.
        y[idx] = 1.0 if c % 2 == 0 else -1.0

    return Tensor(X), Tensor(y)


def accuracy(pred: "Stage12_Tensor", y) -> float:
    """Binary accuracy: fraction where sign(pred) == sign(y) (read-only, off-graph)."""
    # TODO: compute accuracy from sign(pred) vs sign(y)
    raise NotImplementedError("accuracy")


def train(
    model: "Stage11_MLP",
    X: "Stage12_Tensor",
    y: "Stage12_Tensor",
    *,
    lr: float = 0.1,
    epochs: int = 200,
    optimizer: Optional["Stage14_SGD"] = None,
) -> List[float]:
    """Run the training loop (forward -> mse_loss -> backward -> step -> zero_grad);
    return per-epoch loss history. Defaults to SGD(model.parameters(), lr) if None.

    ``X`` and ``y`` must be ``Tensor`` instances (X: (N, n_in), y: (N,) or (N, 1));
    raise TypeError otherwise. Wrap the NumPy arrays from make_moons/make_spiral
    with ``Tensor(...)`` before calling."""
    # TODO: implement the training loop
    raise NotImplementedError("train")


def plot_loss(history: Sequence[float], path: Optional[str] = None):
    """Plot training loss vs epoch; save to path if given. Returns the Figure."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(range(1, len(history) + 1), history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    if path is not None:
        fig.savefig(path)
    else:
        plt.show()
    return fig
