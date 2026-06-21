"""Stage 26: CNN classifier (project).

Compose prior building blocks into a small convolutional image classifier and
train it end-to-end. No new layer math here -- this stage only wires imported
layers into a network (``CNN``) and drives a train/eval loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dlfs import stage_import

# Tensor (09); Dense (11); Conv2D/MaxPool2D/Flatten (25); cross_entropy_loss (13); Adam (18).
# NOTE Conv2D signature is Conv2D(out_channels, in_channels, kernel_size, ...).
Stage9_Tensor = stage_import("stage_09", "Tensor")
Stage11_Dense = stage_import("stage_11", "Dense")
Stage25_Conv2D, Stage25_MaxPool2D, Stage25_Flatten = stage_import(
    "stage_25", "Conv2D", "MaxPool2D", "Flatten"
)
Stage13_cross_entropy_loss = stage_import("stage_13", "cross_entropy_loss")
Stage18_Adam = stage_import("stage_18", "Adam")

# Re-export under canonical public names (Conv2d/MaxPool2d are lowercase aliases).
Tensor = Stage9_Tensor
Dense = Stage11_Dense
Conv2D = Stage25_Conv2D
MaxPool2D = Stage25_MaxPool2D
Flatten = Stage25_Flatten
cross_entropy_loss = Stage13_cross_entropy_loss
Adam = Stage18_Adam
Conv2d = Stage25_Conv2D
MaxPool2d = Stage25_MaxPool2D

__all__ = [
    "Tensor",
    "Dense",
    "Conv2D",
    "Conv2d",
    "MaxPool2D",
    "MaxPool2d",
    "Flatten",
    "cross_entropy_loss",
    "Adam",
    "CNN",
    "accuracy",
    "train_cnn",
    "make_digit_blobs",
]


class CNN:
    """LeNet-style classifier: conv-relu-pool x2 -> flatten -> dense head. in_shape
    is (C, H, W); ReLU applied by forward(); flat_dim derived, not hardcoded."""

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        n_classes: int,
        *,
        conv_channels: Sequence[int] = (8, 16),
        kernel_size: int = 3,
        hidden: int = 64,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: build the conv tower + dense head; derive flat_dim from spatial arithmetic
        raise NotImplementedError("CNN.__init__")

    def forward(self, x) -> "Stage9_Tensor":
        """Run a (B, C, H, W) batch -> (B, n_classes) logits (no softmax)."""
        # TODO: implement the forward pass (relu via Tensor.relu, no head activation).
        raise NotImplementedError("CNN.forward")

    def __call__(self, x) -> "Stage9_Tensor":
        """Alias for :meth:`forward`."""
        # TODO: delegate to self.forward.
        raise NotImplementedError("CNN.__call__")

    def parameters(self) -> List["Stage9_Tensor"]:
        """Every learnable parameter from every sub-layer, in forward order."""
        # TODO: collect params from sub-layers that expose parameters()
        raise NotImplementedError("CNN.parameters")

    def train(self) -> "CNN":
        """Set training mode on self and every sub-layer; return self."""
        # TODO: implement train-mode propagation.
        raise NotImplementedError("CNN.train")

    def eval(self) -> "CNN":
        """Set eval mode on self and every sub-layer; return self."""
        # TODO: implement eval-mode propagation.
        raise NotImplementedError("CNN.eval")

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        # TODO: implement gradient reset.
        raise NotImplementedError("CNN.zero_grad")

    def __repr__(self) -> str:
        # TODO: summarize in_shape, n_classes, flat_dim.
        raise NotImplementedError("CNN.__repr__")


def accuracy(logits, targets) -> float:
    """Fraction of rows where argmax(logits) == target. logits (B, C), targets (B,)."""
    # TODO: implement accuracy.
    raise NotImplementedError("accuracy")


def train_cnn(
    model: "CNN",
    train_loader,
    *,
    epochs: int,
    lr: float = 1e-3,
    val_loader=None,
    optimizer=None,
    verbose: bool = False,
) -> Dict[str, list]:
    """Train ``model`` with Adam over ``epochs`` passes of ``train_loader``. Returns
    history ``{"train_loss", "val_loss", "val_acc", "steps"}`` (val empty if none)."""
    # TODO: implement the train loop (forward/loss/backward/step; optional validation).
    raise NotImplementedError("train_cnn")


def make_digit_blobs(
    n_per_class: int = 64,
    *,
    img_size: int = 8,
    n_classes: int = 2,
    noise: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tiny MNIST-like single-channel set: per-class binary stroke templates + noise.
    Returns (X (N,1,S,S), y (N,) int), rows shuffled."""
    # TODO: build per-class binary templates, add noise, stack and shuffle.
    raise NotImplementedError("make_digit_blobs")
