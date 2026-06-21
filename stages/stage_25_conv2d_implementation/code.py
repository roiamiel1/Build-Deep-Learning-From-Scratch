"""Stage 25: Conv2D (im2col + matmul), pooling, and Flatten, built on the
stage_09 autodiff ``Tensor``. Adds im2col/col2im, Conv2D, MaxPool2D, AvgPool2D,
Flatten with hand-wired backward closures.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

# Tensor engine from stage_09: Tensor(data, _prev=(), _op="").
from dlfs import stage_import

Stage9_Tensor = stage_import("stage_09", "Tensor")


def conv_output_size(in_size: int, k: int, stride: int, pad: int) -> int:
    """Output spatial size: floor((in_size + 2*pad - k) / stride) + 1."""
    # TODO: return the integer output size using the formula above.
    raise NotImplementedError("conv_output_size")


def im2col(
    x: np.ndarray,
    kh: int,
    kw: int,
    stride: int,
    pad: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Unfold receptive fields of ``x`` (N, C, H, W) into rows. Returns
    (cols (N*H'*W', C*kh*kw), x_padded_shape) for col2im."""
    # TODO: zero-pad x, gather fields per output position, reorder; return (cols, x_padded.shape).
    raise NotImplementedError("im2col")


def col2im(
    cols: np.ndarray,
    x_padded_shape: Tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride: int,
    pad: int,
) -> np.ndarray:
    """Adjoint of ``im2col``: scatter-ADD rows back to pixels (overlaps summed).
    cols (N*H'*W', C*kh*kw) -> (N, C, H, W)."""
    # TODO: scatter-add each kernel-position slab into a padded canvas, strip pad.
    raise NotImplementedError("col2im")


class Conv2D:
    """2-D convolution via im2col + matmul, with a hand-wired backward.

    W: (out_channels, in_channels, k, k) He-init; b: (out_channels,) or None.
    """

    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: He-init W (std=sqrt(2/(in_channels*k*k))), zero bias if used; store config.
        raise NotImplementedError("Conv2D.__init__")

    def __call__(self, x: Union["Stage9_Tensor", np.ndarray]) -> "Stage9_Tensor":
        """Forward (N, C_in, H, W) -> (N, C_out, H', W') via im2col + matmul."""
        # TODO: implement forward via im2col + matmul; assign out._backward (+= grads).
        raise NotImplementedError("Conv2D.__call__")

    def parameters(self) -> List["Stage9_Tensor"]:
        """Learnable parameters: [W, b] (or [W] when no bias)."""
        # TODO: implement
        raise NotImplementedError("Conv2D.parameters")

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        # TODO: implement
        raise NotImplementedError("Conv2D.zero_grad")

    def __repr__(self) -> str:
        # TODO: brief config repr.
        raise NotImplementedError("Conv2D.__repr__")


class MaxPool2D:
    """Max pooling over k x k windows; backward routes grad to arg-max only.

    stride defaults to kernel_size (non-overlapping); overlapping windows ADD.
    """

    def __init__(self, kernel_size: int, stride: Optional[int] = None) -> None:
        # TODO: store kernel_size and stride (default stride = kernel_size).
        raise NotImplementedError("MaxPool2D.__init__")

    def __call__(self, x: Union["Stage9_Tensor", np.ndarray]) -> "Stage9_Tensor":
        """Forward (N, C, H, W) -> (N, C, H', W'); max over each window. Backward
        routes grad to the cached arg-max (0 elsewhere)."""
        # TODO: implement max pooling forward + cached-argmax backward.
        raise NotImplementedError("MaxPool2D.__call__")

    def __repr__(self) -> str:
        # TODO: brief config repr.
        raise NotImplementedError("MaxPool2D.__repr__")


class AvgPool2D:
    """Average pooling over k x k windows; backward spreads grad uniformly.

    stride defaults to kernel_size (non-overlapping); overlapping windows ADD.
    """

    def __init__(self, kernel_size: int, stride: Optional[int] = None) -> None:
        # TODO: store kernel_size and stride (default stride = kernel_size).
        raise NotImplementedError("AvgPool2D.__init__")

    def __call__(self, x: Union["Stage9_Tensor", np.ndarray]) -> "Stage9_Tensor":
        """Forward (N, C, H, W) -> (N, C, H', W'); mean over each window. Backward
        scatter-ADDs g/(kh*kw) into every covered input position."""
        # TODO: implement average pooling forward + uniform-spread backward.
        raise NotImplementedError("AvgPool2D.__call__")

    def __repr__(self) -> str:
        # TODO: brief config repr.
        raise NotImplementedError("AvgPool2D.__repr__")


class Flatten:
    """Differentiable reshape (N, C, H, W) -> (N, C*H*W); bridges to Dense.

    Backward reshapes the upstream grad straight back to the input shape.
    """

    def __call__(self, x: Union["Stage9_Tensor", np.ndarray]) -> "Stage9_Tensor":
        """Forward (N, ...) -> (N, prod(...)); backward reshapes grad back."""
        # TODO: implement the flatten forward + reshape-back backward.
        raise NotImplementedError("Flatten.__call__")

    def __repr__(self) -> str:
        # TODO: return "Flatten()".
        raise NotImplementedError("Flatten.__repr__")
