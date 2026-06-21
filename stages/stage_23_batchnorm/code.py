"""Stage 23: Batch Normalization (1-D).

Self-contained module with hand-written forward/backward on raw NumPy arrays (not
the Tensor engine). Per feature: x_hat=(x-mu)/sqrt(var+eps); y=gamma*x_hat+beta.
Train uses batch stats + updates running (EMA) buffers; eval uses the buffers.
"""

from __future__ import annotations

from typing import List

import numpy as np

# Tensor (09), imported for shape/parity use.
from dlfs import stage_import

Stage9_Tensor = stage_import("stage_09", "Tensor")


class BatchNorm1d:
    """BatchNorm over (B, num_features): standardize each column, then affine y = gamma*x_hat + beta."""

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        # TODO: init params (gamma ones/beta zeros), buffers, grads, training flag, cache.
        raise NotImplementedError("BatchNorm1d.__init__")

    def train(self) -> "BatchNorm1d":
        """Switch to train mode (batch stats, update buffers); return self."""
        # TODO: set training True and return self.
        raise NotImplementedError("BatchNorm1d.train")

    def eval(self) -> "BatchNorm1d":
        """Switch to eval mode (running buffers, no updates); return self."""
        # TODO: set training False and return self.
        raise NotImplementedError("BatchNorm1d.eval")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize x (B, num_features) + affine. Train: BIASED batch var, EMA-update
        buffers with UNBIASED var, cache for backward. Eval: running buffers only."""
        # TODO: implement the train and eval branches.
        raise NotImplementedError("BatchNorm1d.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for :meth:`forward`."""
        # TODO: delegate to forward.
        raise NotImplementedError("BatchNorm1d.__call__")

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backprop dL/dy (B, num_features) -> dL/dx; set gamma_grad/beta_grad (only
        valid after a train forward). Collapsed standardization backward, with
        g_hat = grad_out * gamma:
            dx = (istd / B) * (B*g_hat - g_hat.sum(0) - x_hat*(g_hat*x_hat).sum(0))
        """
        # TODO: read cache (error if empty), compute param grads, return dx.
        raise NotImplementedError("BatchNorm1d.backward")

    def parameters(self) -> List[np.ndarray]:
        """Return learnable params [gamma, beta] (buffers are excluded)."""
        # TODO: return the learnable parameters.
        raise NotImplementedError("BatchNorm1d.parameters")

    def zero_grad(self) -> None:
        """Reset gamma_grad and beta_grad to zeros."""
        # TODO: zero the parameter grads.
        raise NotImplementedError("BatchNorm1d.zero_grad")

    def __repr__(self) -> str:
        # TODO: include num_features, eps, momentum, training.
        raise NotImplementedError("BatchNorm1d.__repr__")
