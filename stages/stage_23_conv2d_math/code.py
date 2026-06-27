"""Stage 23: Convolution mathematics (shape formulas + forward/backward by hand).

Pure-function conv math (no autodiff, no layer class): output-size formulas,
cross-correlation forward, im2col/col2im, and the three backward rules. NCHW
layout: x (N,C_in,H,W), w (C_out,C_in,kH,kW), b (C_out,), y (N,C_out,H_out,W_out).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Stage_08 Mat / stage_11 Tensor used here only as shape oracles.
from dlfs import stage_import

Stage7_Mat = stage_import("stage_07", "Mat")
Stage11_Tensor = stage_import("stage_11", "Tensor")


# Output-size formulas
def conv_output_size(
    in_size: int,
    kernel: int,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> int:
    """Output length of one spatial axis under conv with stride/pad/dilation.

    out = floor((in_size + 2*pad - dilation*(kernel-1) - 1) / stride) + 1.
    Raise ValueError on bad args or output size < 1.
    """
    raise NotImplementedError("TODO: floor((in + 2*pad - dil*(k-1) - 1)/stride) + 1")


def conv2d_output_shape(
    x_shape: Tuple[int, int, int, int],
    w_shape: Tuple[int, int, int, int],
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> Tuple[int, int, int, int]:
    """Full output shape (N, C_out, H_out, W_out). Raise ValueError if C_in mismatch."""
    raise NotImplementedError("TODO: (N, C_out, H_out, W_out)")


def im2col_shape(
    x_shape: Tuple[int, int, int, int],
    w_shape: Tuple[int, int, int, int],
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    """Shape of the unrolled im2col matrix: (N*H_out*W_out, C_in*kH*kW)."""
    raise NotImplementedError("TODO: (N*H_out*W_out, C_in*kH*kW)")


# im2col / col2im
def im2col(
    x: np.ndarray,
    kH: int,
    kW: int,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """Unroll every receptive field of x (N,C_in,H,W) into a row of x_col,
    shape (N*H_out*W_out, C_in*kH*kW)."""
    raise NotImplementedError("TODO: gather dilated patches into rows")


def col2im(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    kH: int,
    kW: int,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """Adjoint of im2col: scatter-ADD columns back to image positions (overlaps
    sum, not a true inverse); returns x_shape."""
    raise NotImplementedError("TODO: scatter-add patches (overlaps sum), crop pad")


# Forward (cross-correlation via im2col + matmul)
def conv2d_forward(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """Reference conv2d forward (cross-correlation, NO kernel flip).

    y_col = im2col(x) @ w.reshape(C_out,-1).T + b, reshaped to (N, C_out, H_out, W_out).
    """
    raise NotImplementedError("TODO: y = (im2col(x) @ w_row.T + b) reshaped to NCHW")


# Backward (matmul gradients from stage_07 + col2im)
def conv2d_backward(
    dy: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical gradients of conv2d wrt input, kernel, bias.

    With G = dy as y_col (N*Ho*Wo, C_out): dw_row = G.T @ x_col, dx = col2im(G @ w_row),
    db = G.sum(axis=0). Returns (dx, dw, db) shaped like (x, w, b).
    """
    raise NotImplementedError("TODO: dw=G.T@x_col, dx=col2im(G@w_row), db=G.sum(0)")
