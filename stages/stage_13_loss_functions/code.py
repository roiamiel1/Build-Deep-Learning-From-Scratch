"""Stage 13: Loss functions (mse, mae, cross-entropy) built on the Tensor engine.

All losses use Tensor ops so gradients flow through Tensor.backward(); NumPy is
allowed for forward array creation only (e.g. one-hot labels), never for grads.
"""

from __future__ import annotations

import numpy as np

# Reuse the autodiff Tensor engine from stage_09 (broadcast backward from stage_12).
from dlfs import stage_import

Stage9_Tensor = stage_import("stage_09", "Tensor")


def _as_tensor(x) -> "Stage9_Tensor":
    """Return x as a Tensor (pass through if it already is one)."""
    # TODO: coerce x to a Tensor
    raise NotImplementedError("_as_tensor")


def one_hot(targets, num_classes: int) -> np.ndarray:
    """Turn 1-D integer class labels (B,) into a (B, num_classes) one-hot ndarray."""
    # TODO: build the one-hot matrix with NumPy (pure forward construction)
    raise NotImplementedError("one_hot")


def mse_loss(pred, target) -> "Stage9_Tensor":
    """Mean squared error: scalar Tensor L = mean( (pred - target)**2 )."""
    # TODO: implement MSE using Tensor ops and a mean reduction
    raise NotImplementedError("mse_loss")


def mae_loss(pred, target) -> "Stage9_Tensor":
    """Mean absolute error: scalar Tensor L = mean( |pred - target| ).

    Build abs from Tensor ops (e.g. |d| = relu(d) + relu(-d)); never hand-write grads.
    """
    # TODO: implement MAE using Tensor ops and a mean reduction
    raise NotImplementedError("mae_loss")


def log_softmax(logits) -> "Stage9_Tensor":
    """Numerically stable log-softmax over the last axis; (B, C) -> (B, C). Use the
    log-sum-exp trick: subtract the per-row max as a CONSTANT (no grad through shift)."""
    # TODO: implement stable log-softmax via log-sum-exp
    raise NotImplementedError("log_softmax")


def softmax(logits) -> "Stage9_Tensor":
    """Stable softmax over the last axis; rows of the (B, C) output sum to 1."""
    # TODO: implement softmax (reuse the stable log_softmax path)
    raise NotImplementedError("softmax")


def cross_entropy_loss(logits, targets) -> "Stage9_Tensor":
    """Mean softmax cross-entropy; logits (B, C), targets int indices or one-hot.
    L = -(1/B) * sum_b sum_c y_{b,c} * log p_{b,c}; build from log_softmax (don't
    special-case the gradient)."""
    # TODO: implement cross-entropy via log_softmax + one-hot targets
    raise NotImplementedError("cross_entropy_loss")
