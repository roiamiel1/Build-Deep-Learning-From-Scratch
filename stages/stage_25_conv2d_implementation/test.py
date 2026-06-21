"""Tests for stage_25: Conv2D (im2col + matmul) and pooling.

Strategy
--------
* Forward correctness is checked against INDEPENDENT, naive NumPy references
  (explicit sliding-window loops) so a correct im2col path cannot "agree with
  itself".
* im2col / col2im are verified as adjoints via the inner-product identity
  <im2col(x), y> == <x, col2im(y)>.
* Gradients are verified with central-difference gradient checking against a
  scalar loss L = sum(out), comparing the perturbation estimate to the
  analytic ``.grad`` filled by ``Tensor.backward()``.

Run:  pytest stage_25_conv2d_implementation/test.py
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load this stage's code.py and the stage_09 Tensor.
# ---------------------------------------------------------------------------
_HERE = _os.path.dirname(_os.path.abspath(__file__))


def _load(prefix: str):
    root = _os.path.dirname(_HERE)
    for d in sorted(x for x in _os.listdir(root) if x.startswith(prefix)):
        path = _os.path.join(root, d, "code.py")
        if _os.path.exists(path):
            spec = _ilu.spec_from_file_location(f"_{d}_test_code", path)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f"no code.py under {prefix}*")


code = _load("stage_25")
Tensor = _load("stage_09").Tensor

EPS = 1e-5
ATOL = 1e-6


# ---------------------------------------------------------------------------
# Independent naive references.
# ---------------------------------------------------------------------------
def naive_conv2d(x, W, b, stride, pad):
    """Reference conv: x (N,Cin,H,W), W (Cout,Cin,kh,kw), b (Cout,) or None."""
    N, Cin, H, Wd = x.shape
    Cout, _, kh, kw = W.shape
    Hp = (H + 2 * pad - kh) // stride + 1
    Wp = (Wd + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((N, Cout, Hp, Wp))
    for n in range(N):
        for co in range(Cout):
            for oh in range(Hp):
                for ow in range(Wp):
                    hs, ws = oh * stride, ow * stride
                    patch = xp[n, :, hs:hs + kh, ws:ws + kw]
                    out[n, co, oh, ow] = np.sum(patch * W[co])
            if b is not None:
                out[n, co] += b[co]
    return out


def naive_maxpool(x, k, stride):
    N, C, H, Wd = x.shape
    Hp = (H - k) // stride + 1
    Wp = (Wd - k) // stride + 1
    out = np.zeros((N, C, Hp, Wp))
    for n in range(N):
        for c in range(C):
            for oh in range(Hp):
                for ow in range(Wp):
                    hs, ws = oh * stride, ow * stride
                    out[n, c, oh, ow] = np.max(x[n, c, hs:hs + k, ws:ws + k])
    return out


def naive_avgpool(x, k, stride):
    N, C, H, Wd = x.shape
    Hp = (H - k) // stride + 1
    Wp = (Wd - k) // stride + 1
    out = np.zeros((N, C, Hp, Wp))
    for n in range(N):
        for c in range(C):
            for oh in range(Hp):
                for ow in range(Wp):
                    hs, ws = oh * stride, ow * stride
                    out[n, c, oh, ow] = np.mean(x[n, c, hs:hs + k, ws:ws + k])
    return out


# ---------------------------------------------------------------------------
# conv_output_size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "in_size,k,stride,pad,expected",
    [(5, 3, 1, 0, 3), (5, 3, 1, 1, 5), (7, 3, 2, 0, 3), (8, 2, 2, 0, 4)],
)
def test_conv_output_size(in_size, k, stride, pad, expected):
    assert code.conv_output_size(in_size, k, stride, pad) == expected


# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------
def test_im2col_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 6, 5))
    kh = kw = 3
    stride, pad = 1, 0
    cols, _ = code.im2col(x, kh, kw, stride, pad)
    Hp = code.conv_output_size(6, kh, stride, pad)
    Wp = code.conv_output_size(5, kw, stride, pad)
    assert cols.shape == (2 * Hp * Wp, 3 * kh * kw), (
        f"im2col shape {cols.shape} != expected {(2 * Hp * Wp, 3 * kh * kw)}"
    )


@pytest.mark.parametrize("stride,pad", [(1, 0), (2, 0), (1, 1), (2, 1)])
def test_im2col_col2im_are_adjoint(stride, pad):
    """<im2col(x), y> must equal <x, col2im(y)> for random x, y."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 7, 6))
    kh = kw = 3
    cols, xp_shape = code.im2col(x, kh, kw, stride, pad)
    y = rng.standard_normal(cols.shape)
    lhs = np.sum(cols * y)
    back = code.col2im(y, xp_shape, kh, kw, stride, pad)
    assert back.shape == x.shape, f"col2im shape {back.shape} != {x.shape}"
    rhs = np.sum(x * back)
    assert np.isclose(lhs, rhs, atol=1e-8), (
        f"im2col/col2im not adjoint: <im2col(x),y>={lhs} vs <x,col2im(y)>={rhs}"
    )


# ---------------------------------------------------------------------------
# Conv2D forward vs naive reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "N,Cin,Cout,H,Wd,k,stride,pad,bias",
    [
        (2, 3, 4, 6, 6, 3, 1, 0, True),
        (1, 1, 2, 5, 7, 3, 1, 1, True),
        (2, 2, 3, 8, 8, 3, 2, 1, False),
        (3, 4, 2, 7, 5, 2, 2, 0, True),
    ],
)
def test_conv2d_forward_matches_naive(N, Cin, Cout, H, Wd, k, stride, pad, bias):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((N, Cin, H, Wd))
    conv = code.Conv2D(Cout, Cin, k, stride=stride, padding=pad, bias=bias, seed=7)
    out = conv(x)
    b = conv.b.data if bias else None
    ref = naive_conv2d(x, conv.W.data, b, stride, pad)
    assert out.shape == ref.shape, f"conv out shape {out.shape} != {ref.shape}"
    assert np.allclose(out.data, ref, atol=1e-10), "Conv2D forward != naive reference"


# ---------------------------------------------------------------------------
# Conv2D gradient checks (central differences)
# ---------------------------------------------------------------------------
def _central_diff(f, arr, eps=EPS):
    """Numerical grad of scalar f() w.r.t. each entry of array `arr` (in place)."""
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = arr[idx]
        arr[idx] = orig + eps
        fp = f()
        arr[idx] = orig - eps
        fm = f()
        arr[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def test_conv2d_gradcheck_weight():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 2, 6, 6))
    conv = code.Conv2D(3, 2, 3, stride=1, padding=1, bias=True, seed=1)

    def loss():
        return float(np.sum(conv(x).data))

    num = _central_diff(loss, conv.W.data)
    conv.zero_grad()
    out = conv(x)
    out.backward()
    assert np.allclose(conv.W.grad, num, atol=ATOL), (
        "Conv2D dL/dW mismatch:\nanalytic=%s\nnumeric=%s"
        % (conv.W.grad.ravel()[:6], num.ravel()[:6])
    )


def test_conv2d_gradcheck_bias():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 2, 5, 5))
    conv = code.Conv2D(3, 2, 3, stride=2, padding=1, bias=True, seed=2)

    def loss():
        return float(np.sum(conv(x).data))

    num = _central_diff(loss, conv.b.data)
    conv.zero_grad()
    out = conv(x)
    out.backward()
    assert np.allclose(conv.b.grad, num, atol=ATOL), (
        f"Conv2D dL/db mismatch: analytic={conv.b.grad} numeric={num}"
    )


def test_conv2d_gradcheck_input():
    rng = np.random.default_rng(5)
    x_arr = rng.standard_normal((2, 2, 6, 5))
    conv = code.Conv2D(2, 2, 3, stride=1, padding=0, bias=True, seed=3)

    def loss():
        return float(np.sum(conv(x_arr).data))

    num = _central_diff(loss, x_arr)
    xt = Tensor(x_arr)
    conv.zero_grad()
    out = conv(xt)
    out.backward()
    assert np.allclose(xt.grad, num, atol=ATOL), (
        "Conv2D dL/dx mismatch:\nanalytic=%s\nnumeric=%s"
        % (xt.grad.ravel()[:6], num.ravel()[:6])
    )


# ---------------------------------------------------------------------------
# MaxPool2D
# ---------------------------------------------------------------------------
def test_maxpool_forward_matches_naive():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2, 3, 8, 8))
    pool = code.MaxPool2D(2)  # stride defaults to 2 (non-overlapping)
    out = pool(x)
    ref = naive_maxpool(x, 2, 2)
    assert out.shape == ref.shape, f"maxpool shape {out.shape} != {ref.shape}"
    assert np.allclose(out.data, ref, atol=1e-12), "MaxPool2D forward != naive"


def test_maxpool_default_stride_is_kernel():
    pool = code.MaxPool2D(3)
    assert pool.stride == 3, "MaxPool2D stride should default to kernel_size"


def test_maxpool_gradcheck():
    # Distinct values so the arg-max in each window is unambiguous (max is then
    # locally differentiable and central differences are valid).
    rng = np.random.default_rng(7)
    x_arr = rng.permutation(2 * 1 * 6 * 6).astype(np.float64).reshape(2, 1, 6, 6)
    pool = code.MaxPool2D(2)

    def loss():
        return float(np.sum(pool(x_arr).data))

    num = _central_diff(loss, x_arr, eps=0.4)  # large eps OK: max is piecewise-linear
    xt = Tensor(x_arr)
    out = pool(xt)
    out.backward()
    assert np.allclose(xt.grad, num, atol=ATOL), (
        "MaxPool2D dL/dx mismatch (grad should be a 0/1 gate at arg-max)"
    )


# ---------------------------------------------------------------------------
# AvgPool2D
# ---------------------------------------------------------------------------
def test_avgpool_forward_matches_naive():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((2, 3, 8, 6))
    pool = code.AvgPool2D(2)
    out = pool(x)
    ref = naive_avgpool(x, 2, 2)
    assert out.shape == ref.shape, f"avgpool shape {out.shape} != {ref.shape}"
    assert np.allclose(out.data, ref, atol=1e-12), "AvgPool2D forward != naive"


def test_avgpool_gradcheck():
    rng = np.random.default_rng(9)
    x_arr = rng.standard_normal((2, 2, 6, 6))
    pool = code.AvgPool2D(2)

    def loss():
        return float(np.sum(pool(x_arr).data))

    num = _central_diff(loss, x_arr)
    xt = Tensor(x_arr)
    out = pool(xt)
    out.backward()
    assert np.allclose(xt.grad, num, atol=ATOL), (
        "AvgPool2D dL/dx mismatch (each input grad should be 1/(k*k))"
    )
    # Every covered position gets exactly 1/(k*k) for a sum-loss, non-overlapping.
    assert np.allclose(xt.grad, 1.0 / 4.0, atol=ATOL), (
        "AvgPool2D grad should be uniform 1/(k*k) for non-overlapping windows"
    )


# ---------------------------------------------------------------------------
# Integration: conv -> pool composes and backprops end to end.
# ---------------------------------------------------------------------------
def test_conv_then_pool_backprops():
    rng = np.random.default_rng(10)
    x_arr = rng.standard_normal((2, 2, 8, 8))
    conv = code.Conv2D(3, 2, 3, stride=1, padding=1, bias=True, seed=4)
    pool = code.MaxPool2D(2)

    def loss():
        return float(np.sum(pool(conv(x_arr)).data))

    num_W = _central_diff(loss, conv.W.data)
    conv.zero_grad()
    out = pool(conv(x_arr))
    out.backward()
    assert np.allclose(conv.W.grad, num_W, atol=ATOL), (
        "conv->maxpool end-to-end dL/dW mismatch"
    )
