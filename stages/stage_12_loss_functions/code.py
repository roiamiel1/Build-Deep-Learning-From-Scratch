"""Stage 12: Loss functions (mse, mae, cross-entropy) + sum/mean reductions.

This stage does two things on top of the autodiff ``Tensor``:

1. ADDS the ``sum`` / ``mean`` *reduction* ops (with correct backward) to the
   ``Tensor``.  stage_08 deliberately shipped no reductions and deferred them to
   here; every loss below needs ``.sum()`` / ``.mean()`` to collapse a per-element
   array down to a scalar objective, so this is where they live.
2. ADDS the workhorse losses (``mse_loss``, ``mae_loss``, ``cross_entropy_loss``
   with a stable ``softmax`` / ``log_softmax`` via log-sum-exp) built on those
   reductions.

All losses use Tensor ops so gradients flow through ``Tensor.backward()``; NumPy
is allowed for forward array creation only (e.g. one-hot labels), never for grads.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

# Reuse the autodiff Tensor engine.  We extend the broadcast-capable Tensor from
# stage_11 (it adds the broadcasting-correct backward); THIS stage subclasses it
# to add the ``sum`` / ``mean`` reductions the losses need.
from dlfs import stage_import

Stage11_Tensor = stage_import("stage_11", "Tensor")


class Tensor(Stage11_Tensor):
    """The autodiff ``Tensor`` extended with ``sum`` / ``mean`` reductions.

    Subclasses the broadcast-capable base (stage_11's ``Tensor`` if present, else
    stage_08's) and ADDS the two reduction ops the losses need.  A reduction maps
    an array down (optionally along ``axis``) to fewer elements; its backward
    must *re-expand* the upstream grad back over the reduced axes.
    """

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdims: bool = False) -> "Tensor":
        """Sum over ``axis`` (all elements if ``axis is None``). z = self.sum(...).

        TODO (forward): z.data = np.sum(self.data, axis=axis, keepdims=keepdims),
            returning a child Tensor with _prev=(self,), _op="sum".
        TODO (backward): each input element contributed exactly once to its output
            cell, so its local grad is 1.  dL/d(self) = the upstream grad
            *expanded* (broadcast) back to ``self.shape`` over the reduced axes.
            With keepdims=False you must first restore the reduced axes as size-1
            (e.g. np.expand_dims over the summed axes, or reshape) before
            broadcasting to self.shape; with keepdims=True it already broadcasts.
            Accumulate via the engine's grad-accumulation path (the same one
            __add__/__mul__ use) so it matches self.shape.
        """
        raise NotImplementedError("Tensor.sum")

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> "Tensor":
        """Mean over ``axis`` (all elements if ``axis is None``). z = self.mean(...).

        TODO (forward): z.data = np.mean(self.data, axis=axis, keepdims=keepdims),
            returning a child Tensor with _prev=(self,), _op="mean".
        TODO (backward): mean = sum / N where N is the number of reduced elements
            (the product of the reduced-axis sizes; self.data.size when axis is
            None).  So this is exactly the ``sum`` backward but each expanded grad
            is additionally divided by N: dL/d(self) = (upstream grad expanded to
            self.shape over the reduced axes) / N.  Expand the same way as sum
            (restore reduced axes to size-1 first when keepdims=False), then
            broadcast to self.shape and accumulate.
        """
        raise NotImplementedError("Tensor.mean")


def _as_tensor(x) -> "Tensor":
    """Return x as THIS stage's Tensor (the one with reductions).

    Coercing to this Tensor (not the bare base) is what lets a loss call
    ``.sum()`` / ``.mean()`` on a coerced input.
    """
    # TODO: if x is already an instance of this stage's Tensor, return it;
    #       otherwise wrap its data in Tensor(...) so the reduction ops are
    #       available on the result.
    raise NotImplementedError("_as_tensor")


def one_hot(targets, num_classes: int) -> np.ndarray:
    """Turn 1-D integer class labels (B,) into a (B, num_classes) one-hot ndarray."""
    # TODO: build the one-hot matrix with NumPy (pure forward construction)
    raise NotImplementedError("one_hot")


def mse_loss(pred, target) -> "Tensor":
    """Mean squared error: scalar Tensor L = mean( (pred - target)**2 )."""
    # TODO: coerce via _as_tensor, then implement MSE using Tensor ops and .mean()
    raise NotImplementedError("mse_loss")


def mae_loss(pred, target) -> "Tensor":
    """Mean absolute error: scalar Tensor L = mean( |pred - target| ).

    Build abs from Tensor ops (e.g. |d| = relu(d) + relu(-d)); never hand-write grads.
    """
    # TODO: coerce via _as_tensor, then implement MAE using Tensor ops and .mean()
    raise NotImplementedError("mae_loss")


def log_softmax(logits) -> "Tensor":
    """Numerically stable log-softmax over the last axis; (B, C) -> (B, C). Use the
    log-sum-exp trick: subtract the per-row max as a CONSTANT (no grad through shift)."""
    # TODO: implement stable log-softmax via log-sum-exp (uses .sum(axis=1, keepdims=True))
    raise NotImplementedError("log_softmax")


def softmax(logits) -> "Tensor":
    """Stable softmax over the last axis; rows of the (B, C) output sum to 1."""
    # TODO: implement softmax (reuse the stable log_softmax path)
    raise NotImplementedError("softmax")


def cross_entropy_loss(logits, targets) -> "Tensor":
    """Mean softmax cross-entropy; logits (B, C), targets int indices or one-hot.
    L = -(1/B) * sum_b sum_c y_{b,c} * log p_{b,c}; build from log_softmax (don't
    special-case the gradient)."""
    # TODO: implement cross-entropy via log_softmax + one-hot targets, reduced with
    #       .sum() over classes and .mean() over the batch
    raise NotImplementedError("cross_entropy_loss")
