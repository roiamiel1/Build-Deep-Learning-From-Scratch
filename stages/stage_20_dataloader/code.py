"""Stage 20: DataLoader.

Packages stage_19's one-shot batch logic as reusable objects: an indexable
``Dataset``, a re-iterable ``DataLoader`` (iterator protocol, re-shuffles each
epoch), ``train_val_split``, and a ``train_with_loader`` driver. No new gradients.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from dlfs import stage_import

# Tensor (09); MLP/mse_loss/SGD/train_minibatch (19).
Stage9_Tensor = stage_import("stage_09", "Tensor")
Stage19_MLP, Stage19_mse_loss, Stage19_SGD, Stage19_train_minibatch = stage_import(
    "stage_19", "MLP", "mse_loss", "SGD", "train_minibatch"
)

# Re-export imported pieces under canonical public names.
Tensor = Stage9_Tensor
MLP = Stage19_MLP
mse_loss = Stage19_mse_loss
SGD = Stage19_SGD
train_minibatch = Stage19_train_minibatch


class Dataset:
    """Map-style dataset wrapping (X, y) via __len__ + __getitem__. X (N, n_in),
    y (N,) or (N, 1); ValueError if lengths differ."""

    def __init__(self, X, y) -> None:
        # TODO: store X, y as float64 ndarrays; validate matching first dim.
        raise NotImplementedError("Dataset.__init__")

    def __len__(self) -> int:
        """Return N, the number of examples."""
        # TODO: implement
        raise NotImplementedError("Dataset.__len__")

    def __getitem__(self, index):
        """Return ``(X[index], y[index])``; index may be int, slice, or int array."""
        # TODO: implement (DataLoader uses index-array selection for a batch)
        raise NotImplementedError("Dataset.__getitem__")


class DataLoader:
    """Iterate a ``Dataset`` in mini-batches, re-shuffling each epoch; each
    ``__iter__`` is a fresh pass. batch_size must be >= 1 (else ValueError)."""

    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: validate batch_size, store config; do NOT build the iterator here.
        raise NotImplementedError("DataLoader.__init__")

    def __len__(self) -> int:
        """Return batches per epoch: N // batch_size (drop_last) else ceil(N / batch_size)."""
        # TODO: implement
        raise NotImplementedError("DataLoader.__len__")

    def __iter__(self) -> Iterator[Tuple["Stage9_Tensor", "Stage9_Tensor"]]:
        """Yield ``(X_b, y_b)`` Tensor batches for one epoch (fresh order, optional
        drop_last). X_b: (b, n_in), y_b: (b,) or (b, 1)."""
        # TODO: implement the per-epoch generator.
        raise NotImplementedError("DataLoader.__iter__")


def train_val_split(
    X,
    y,
    val_frac: float,
    *,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``(X, y)`` into disjoint train/val (last round(val_frac*N) held out).
    Returns (X_tr, y_tr, X_val, y_val); requires 0 <= val_frac < 1."""
    # TODO: implement
    raise NotImplementedError("train_val_split")


def train_with_loader(
    model: "Stage19_MLP",
    loader: "DataLoader",
    *,
    lr: float = 0.1,
    epochs: int = 50,
    optimizer: Optional["Stage19_SGD"] = None,
) -> Dict[str, object]:
    """Train ``model`` by iterating ``loader`` for ``epochs`` epochs (forward ->
    mse_loss -> backward -> step -> zero_grad). Returns
    {"batch_loss": list, "epoch_loss": list, "steps": int}."""
    # TODO: build optimizer if None, then run the per-epoch / per-batch loop.
    raise NotImplementedError("train_with_loader")
