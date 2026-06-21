"""Stage 24 tests: convolution shape formulas, forward, and gradients.

Run: pytest stage_24_conv2d_math/test.py

This stage is math only (no autodiff). We check:
  - the output-size / shape / im2col-shape formulas against a table of cases,
  - `conv2d_forward` against a naive triple-loop cross-correlation,
  - `col2im` sums overlapping receptive fields (never overwrites),
  - `conv2d_backward` (dx, dw, db) against central-difference numerical gradients:
        df/dt ~= (f(t + eps) - f(t - eps)) / (2 * eps).
"""

import numpy as np
import pytest

from code import (
    col2im,
    conv2d_backward,
    conv2d_forward,
    conv2d_output_shape,
    conv_output_size,
    im2col,
    im2col_shape,
)


EPS = 1e-5
TOL = 1e-4
rng = np.random.default_rng(24)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def naive_conv2d(x, w, b, stride=1, pad=0, dilation=1):
    """Reference cross-correlation by explicit loops (independent oracle)."""
    N, C_in, H, W = x.shape
    C_out, C_in2, kH, kW = w.shape
    assert C_in == C_in2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - dilation * (kH - 1) - 1) // stride + 1
    Wo = (W + 2 * pad - dilation * (kW - 1) - 1) // stride + 1
    y = np.zeros((N, C_out, Ho, Wo))
    for n in range(N):
        for o in range(C_out):
            for i in range(Ho):
                for j in range(Wo):
                    acc = 0.0
                    for c in range(C_in):
                        for u in range(kH):
                            for v in range(kW):
                                acc += (
                                    xp[n, c, i * stride + u * dilation,
                                       j * stride + v * dilation]
                                    * w[o, c, u, v]
                                )
                    y[n, o, i, j] = acc + b[o]
    return y


def central_diff(f, t):
    """Central-difference gradient of scalar f(array) wrt every entry of t."""
    g = np.zeros_like(t, dtype=np.float64)
    it = np.nditer(t, flags=["multi_index"], op_flags=["readwrite"])
    for _ in it:
        idx = it.multi_index
        orig = t[idx]
        t[idx] = orig + EPS
        fp = f(t)
        t[idx] = orig - EPS
        fm = f(t)
        t[idx] = orig
        g[idx] = (fp - fm) / (2 * EPS)
    return g


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# --------------------------------------------------------------------------- #
# output-size formula                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "in_size,k,s,p,d,expected",
    [
        (5, 3, 1, 0, 1, 3),    # plain
        (5, 3, 1, 1, 1, 5),    # 'same' with pad=1
        (7, 3, 2, 0, 1, 3),    # stride 2
        (8, 2, 2, 0, 1, 4),    # exact tiling
        (7, 3, 1, 0, 2, 3),    # dilation 2 -> kernel spans 5
        (32, 5, 1, 2, 1, 32),  # same conv
        (1, 1, 1, 0, 1, 1),    # degenerate
    ],
)
def test_conv_output_size(in_size, k, s, p, d, expected):
    got = conv_output_size(in_size, k, s, p, d)
    assert got == expected, (
        f"conv_output_size({in_size},{k},s={s},p={p},d={d}) = {got}, want {expected}"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(in_size=5, kernel=3, stride=0),    # stride < 1
        dict(in_size=5, kernel=0),              # kernel < 1
        dict(in_size=5, kernel=3, pad=-1),      # pad < 0
        dict(in_size=5, kernel=3, dilation=0),  # dilation < 1
        dict(in_size=3, kernel=5),              # kernel bigger than input -> out < 1
    ],
)
def test_conv_output_size_invalid_raises(kwargs):
    with pytest.raises(ValueError):
        conv_output_size(**kwargs)


def test_conv2d_output_shape():
    shp = conv2d_output_shape((2, 3, 7, 7), (4, 3, 3, 3), stride=2, pad=1, dilation=1)
    assert shp == (2, 4, 4, 4), f"got {shp}"


def test_conv2d_output_shape_channel_mismatch_raises():
    with pytest.raises(ValueError):
        conv2d_output_shape((2, 3, 7, 7), (4, 5, 3, 3))  # C_in 3 vs 5


def test_im2col_shape():
    # x=(2,3,5,5), w=(4,3,3,3), stride1 pad0 -> Ho=Wo=3
    shp = im2col_shape((2, 3, 5, 5), (4, 3, 3, 3), stride=1, pad=0, dilation=1)
    assert shp == (2 * 3 * 3, 3 * 3 * 3), f"got {shp}"  # (18, 27)


# --------------------------------------------------------------------------- #
# im2col / col2im                                                             #
# --------------------------------------------------------------------------- #
def test_im2col_shape_matches_helper():
    x = rng.standard_normal((2, 3, 6, 5))
    cols = im2col(x, 3, 3, stride=1, pad=1, dilation=1)
    expect = im2col_shape(x.shape, (4, 3, 3, 3), stride=1, pad=1, dilation=1)
    assert cols.shape == expect, f"im2col shape {cols.shape} != helper {expect}"


def test_col2im_is_adjoint_of_im2col():
    """<im2col(x), c> == <x, col2im(c)> for random x, c (adjoint property)."""
    x = rng.standard_normal((2, 3, 6, 5))
    cols_like = rng.standard_normal(
        im2col_shape(x.shape, (1, 3, 3, 3), stride=1, pad=1, dilation=1)
    )
    lhs = float(np.sum(im2col(x, 3, 3, stride=1, pad=1) * cols_like))
    rhs = float(np.sum(x * col2im(cols_like, x.shape, 3, 3, stride=1, pad=1)))
    assert abs(lhs - rhs) < TOL, f"adjoint broken: {lhs} vs {rhs}"


def test_col2im_overlaps_sum():
    """A stride-1 3x3 over a 3x3 input: the center pixel is hit 9 times.

    col2im of all-ones columns must place the count of contributing patches at
    each pixel (overlaps SUM, never overwrite).
    """
    x_shape = (1, 1, 3, 3)
    cols = np.ones(im2col_shape(x_shape, (1, 1, 3, 3), stride=1, pad=1))
    out = col2im(cols, x_shape, 3, 3, stride=1, pad=1)
    # center pixel participates in all 9 receptive fields
    assert out[0, 0, 1, 1] == pytest.approx(9.0), f"center count {out[0,0,1,1]}"
    # a corner participates in only 4
    assert out[0, 0, 0, 0] == pytest.approx(4.0), f"corner count {out[0,0,0,0]}"


# --------------------------------------------------------------------------- #
# forward                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "stride,pad,dilation",
    [(1, 0, 1), (1, 1, 1), (2, 1, 1), (1, 0, 2), (2, 2, 1)],
)
def test_conv2d_forward_matches_naive(stride, pad, dilation):
    x = rng.standard_normal((2, 3, 7, 6))
    w = rng.standard_normal((4, 3, 3, 3))
    b = rng.standard_normal(4)
    got = conv2d_forward(x, w, b, stride, pad, dilation)
    ref = naive_conv2d(x, w, b, stride, pad, dilation)
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    assert maxdiff(got, ref) < TOL, f"forward max diff {maxdiff(got, ref):.2e}"


def test_conv2d_forward_shape_matches_helper():
    x = rng.standard_normal((2, 3, 9, 8))
    w = rng.standard_normal((5, 3, 3, 3))
    b = rng.standard_normal(5)
    y = conv2d_forward(x, w, b, stride=2, pad=1, dilation=1)
    assert y.shape == conv2d_output_shape(x.shape, w.shape, 2, 1, 1)


# --------------------------------------------------------------------------- #
# backward: gradcheck vs central differences                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "stride,pad,dilation",
    [(1, 0, 1), (1, 1, 1), (2, 1, 1), (1, 0, 2)],
)
def test_conv2d_backward_gradcheck(stride, pad, dilation):
    x = rng.standard_normal((2, 2, 6, 5))
    w = rng.standard_normal((3, 2, 3, 3))
    b = rng.standard_normal(3)

    # Non-trivial upstream so a wrong reduction axis / transpose is caught.
    y0 = conv2d_forward(x, w, b, stride, pad, dilation)
    dy = rng.standard_normal(y0.shape)

    dx, dw, db = conv2d_backward(dy, x, w, b, stride, pad, dilation)
    assert dx.shape == x.shape, f"dx shape {dx.shape} != {x.shape}"
    assert dw.shape == w.shape, f"dw shape {dw.shape} != {w.shape}"
    assert db.shape == b.shape, f"db shape {db.shape} != {b.shape}"

    # scalar loss L = sum(dy * y) so that dL/dy == dy exactly.
    def loss_x(xx):
        return float(np.sum(dy * conv2d_forward(xx, w, b, stride, pad, dilation)))

    def loss_w(ww):
        return float(np.sum(dy * conv2d_forward(x, ww, b, stride, pad, dilation)))

    def loss_b(bb):
        return float(np.sum(dy * conv2d_forward(x, w, bb, stride, pad, dilation)))

    ndx = central_diff(loss_x, x.copy())
    ndw = central_diff(loss_w, w.copy())
    ndb = central_diff(loss_b, b.copy())

    assert maxdiff(dx, ndx) < TOL, f"dx gradcheck max diff {maxdiff(dx, ndx):.2e}"
    assert maxdiff(dw, ndw) < TOL, f"dw gradcheck max diff {maxdiff(dw, ndw):.2e}"
    assert maxdiff(db, ndb) < TOL, f"db gradcheck max diff {maxdiff(db, ndb):.2e}"


def test_bias_grad_is_sum_over_spatial_and_batch():
    """dL/db[o] = sum over (n, i, j) of dy[n, o, i, j]."""
    x = rng.standard_normal((2, 2, 5, 5))
    w = rng.standard_normal((3, 2, 3, 3))
    b = rng.standard_normal(3)
    dy = rng.standard_normal(conv2d_output_shape(x.shape, w.shape, 1, 0, 1))
    _, _, db = conv2d_backward(dy, x, w, b, stride=1, pad=0, dilation=1)
    expected = dy.sum(axis=(0, 2, 3))
    assert maxdiff(db, expected) < TOL, f"db != sum(dy): {maxdiff(db, expected):.2e}"
