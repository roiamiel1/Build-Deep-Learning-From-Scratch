"""Stage 27: Attention mathematics.

Scaled dot-product attention as pure NumPy reference functions (forward + backward);
no autodiff Tensor, no layer class. Row-wise softmax means over the LAST axis (keys).

Shapes (unbatched; a leading batch axis B broadcasts):
    Q (L_q, d_k)  K (L_k, d_k)  V (L_k, d_v)
    S = Q @ K.T / sqrt(d_k) -> (L_q, L_k);  A = softmax_rows(S);  O = A @ V -> (L_q, d_v)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the LAST axis. x: (..., n).

    Stability: subtract the per-row max before exponentiating (constant shift
    cancels in the ratio).
    """
    # TODO: implement the numerically stable row-wise softmax
    raise NotImplementedError("softmax_rows")


def softmax_backward(dA: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Backprop through row-wise softmax: given dL/dA and the softmax output A,
    return dL/dS. JVP per row: dL/ds_i = a_i * (g_i - sum_j g_j a_j).

    A is the softmax OUTPUT (not the logits); do not recompute the softmax.
    """
    # TODO: implement the row-wise softmax Jacobian-vector product
    raise NotImplementedError("softmax_backward")


def attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention forward, d_k = Q.shape[-1]:
        S = Q @ K.T / sqrt(d_k); S += mask (optional); A = softmax_rows(S); O = A @ V

    1/sqrt(d_k) keeps logits unit-scale so softmax does not saturate; mask is
    ADDITIVE (0 allowed, -inf forbidden). Returns O (..., L_q, d_v) and A (for backward).
    """
    # TODO: implement the scaled dot-product attention forward pass
    raise NotImplementedError("attention_forward")


def attention_backward(
    dO: np.ndarray,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    A: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical gradients of scaled dot-product attention, d_k = Q.shape[-1]:
        dV = A.T @ dO;  dA = dO @ V.T;  dS = softmax_backward(dA, A)
        dQ = (dS @ K) / sqrt(d_k);  dK = (dS.T @ Q) / sqrt(d_k)

    mask is an additive constant (no gradient). Returns dQ, dK, dV matching input shapes.
    """
    # TODO: implement the scaled dot-product attention backward pass
    raise NotImplementedError("attention_backward")
