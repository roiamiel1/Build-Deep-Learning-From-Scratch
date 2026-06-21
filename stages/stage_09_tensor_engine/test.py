"""Tests for Stage 09: Tensor engine.

Verifies the N-dimensional autodiff `Tensor`:
  * forward values and operator overloading (incl. reflected ops with numbers),
  * gradient accumulation on reused nodes (e.g. y = x*x + x),
  * elementwise array backward shapes/values,
  * central-difference gradient checks against the analytic `.grad` filled by
    `.backward()`.

This stage has no reduction op yet (`.sum()` arrives in stage_13), so backward
seeds the output with ones_like and we therefore gradcheck on SCALAR (0-d)
tensors, where the network output IS the scalar loss -- no reduction needed.
The chain-rule code under test is identical for scalars and arrays; the array
tests separately confirm the engine works elementwise at full shape.

Run: pytest stage_09_tensor_engine/test.py
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# Make the shared `dlfs` shim importable (it lives at the curriculum root) so
# this stage's code.py can `from dlfs import stage_import` to pull in stage_06's
# `Value`, the scalar engine this `Tensor` unifies onto arrays.
sys.path.insert(0, _ROOT)

# `code.py` defines this stage's `Tensor` (the new n-d core) and re-imports
# `Value` from stage_06 via dlfs.stage_import for the `from_value` bridge.
try:
    from code import Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_09 Tensor not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
TOL = 1e-6


# --------------------------------------------------------------------------- #
# central-difference gradcheck on a scalar-valued function of one scalar input
# --------------------------------------------------------------------------- #
def numeric_grad_scalar(f, x, eps=EPS):
    """Central-difference d f / d x for scalar float x and scalar output f."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def analytic_grad_scalar(build, x):
    """leaf = Tensor(x); out = build(leaf) (0-d); backward; return leaf.grad."""
    leaf = Tensor(float(x))
    out = build(leaf)
    out.backward()
    return float(leaf.grad)


# A library of scalar functions: (engine build via Tensor, plain-float ref).
SCALAR_FUNCS = {
    "add_const":   (lambda t: t + 3.0,            lambda x: x + 3.0),
    "radd_const":  (lambda t: 3.0 + t,            lambda x: 3.0 + x),
    "mul_const":   (lambda t: t * 2.5,            lambda x: x * 2.5),
    "rmul_const":  (lambda t: 2.5 * t,            lambda x: 2.5 * x),
    "neg":         (lambda t: -t,                 lambda x: -x),
    "sub_const":   (lambda t: t - 1.5,            lambda x: x - 1.5),
    "rsub_const":  (lambda t: 5.0 - t,            lambda x: 5.0 - x),
    "pow2":        (lambda t: t ** 2,             lambda x: x ** 2),
    "pow3":        (lambda t: t ** 3,             lambda x: x ** 3),
    "powhalf":     (lambda t: t ** 0.5,           lambda x: x ** 0.5),
    "div_const":   (lambda t: t / 4.0,            lambda x: x / 4.0),
    "rdiv_const":  (lambda t: 6.0 / t,            lambda x: 6.0 / x),
    "relu":        (lambda t: t.relu(),           lambda x: max(0.0, x)),
    "self_mul":    (lambda t: t * t,              lambda x: x * x),
    "self_add":    (lambda t: t + t,              lambda x: x + x),
    "reuse":       (lambda t: t * t + t,          lambda x: x * x + x),
    "composite":   (lambda t: (t * 2.0 + 1.0).relu() ** 2,
                    lambda x: max(0.0, x * 2.0 + 1.0) ** 2),
    "ratio":       (lambda t: (t + 1.0) / (t * t + 2.0),
                    lambda x: (x + 1.0) / (x * x + 2.0)),
}

# Inputs chosen positive for pow with fractional exponent / division stability,
# and away from the ReLU kink at 0.
SCALAR_INPUTS = [0.7, 1.3, 2.0, 3.5, -0.4]


@pytest.mark.parametrize("name", list(SCALAR_FUNCS))
@pytest.mark.parametrize("x", SCALAR_INPUTS)
def test_scalar_gradcheck(name, x):
    build, ref = SCALAR_FUNCS[name]
    # skip points where the reference is undefined (fractional pow / div of neg)
    if name in ("powhalf", "rdiv_const", "div_const") and x <= 0:
        pytest.skip("function undefined / unstable for non-positive input")
    if name == "relu" and abs(x) < 1e-3:
        pytest.skip("ReLU kink")

    ana = analytic_grad_scalar(build, x)
    num = numeric_grad_scalar(ref, x)
    assert np.isclose(ana, num, rtol=1e-4, atol=TOL), (
        f"{name} at x={x}: analytic grad {ana} != numeric {num}"
    )


@pytest.mark.parametrize("name", list(SCALAR_FUNCS))
@pytest.mark.parametrize("x", SCALAR_INPUTS)
def test_scalar_forward(name, x):
    build, ref = SCALAR_FUNCS[name]
    if name in ("powhalf", "rdiv_const", "div_const") and x <= 0:
        pytest.skip("function undefined for non-positive input")
    out = build(Tensor(float(x)))
    assert np.isclose(float(out.data), ref(x), rtol=1e-6, atol=1e-9), (
        f"{name} at x={x}: forward {float(out.data)} != reference {ref(x)}"
    )


# --------------------------------------------------------------------------- #
# construction / basic invariants
# --------------------------------------------------------------------------- #
def test_construct_from_scalar_list_array():
    a = Tensor(3.0)
    b = Tensor([1.0, 2.0, 3.0])
    c = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    for t in (a, b, c):
        assert isinstance(t.data, np.ndarray)
        assert t.data.dtype == np.float64
        assert t.grad.shape == t.data.shape
        assert np.all(t.grad == 0.0)
        assert t._prev == ()
        assert t._op == ""


def test_shape_property():
    t = Tensor(np.zeros((2, 3, 4)))
    assert t.shape == (2, 3, 4)


def test_repr_mentions_data_and_grad():
    r = repr(Tensor([1.0, 2.0]))
    assert "data" in r and "grad" in r


# --------------------------------------------------------------------------- #
# elementwise array forward + backward
# --------------------------------------------------------------------------- #
def test_array_add_forward_backward():
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0, 6.0])
    z = x + y
    assert np.allclose(z.data, [5.0, 7.0, 9.0])
    z.backward()
    # d(sum-seeded ones)/dx = ones, same for y
    assert np.allclose(x.grad, np.ones(3))
    assert np.allclose(y.grad, np.ones(3))


def test_array_mul_forward_backward():
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0, 6.0])
    z = x * y
    assert np.allclose(z.data, [4.0, 10.0, 18.0])
    z.backward()
    # seed ones: dz/dx = y, dz/dy = x
    assert np.allclose(x.grad, [4.0, 5.0, 6.0])
    assert np.allclose(y.grad, [1.0, 2.0, 3.0])


def test_array_pow_backward():
    x = Tensor([1.0, 2.0, 3.0])
    z = x ** 3
    assert np.allclose(z.data, [1.0, 8.0, 27.0])
    z.backward()
    assert np.allclose(x.grad, 3.0 * np.array([1.0, 2.0, 3.0]) ** 2)


def test_array_relu_backward():
    x = Tensor([-2.0, -0.5, 0.5, 3.0])
    z = x.relu()
    assert np.allclose(z.data, [0.0, 0.0, 0.5, 3.0])
    z.backward()
    assert np.allclose(x.grad, [0.0, 0.0, 1.0, 1.0])


def test_2d_chain_backward():
    x = Tensor([[1.0, -2.0], [3.0, 4.0]])
    z = (x * 2.0).relu()
    z.backward()
    expected = 2.0 * (x.data > 0)
    assert np.allclose(x.grad, expected)
    assert x.grad.shape == x.data.shape


# --------------------------------------------------------------------------- #
# gradient accumulation on reused nodes
# --------------------------------------------------------------------------- #
def test_reuse_accumulates_array():
    x = Tensor([1.0, 2.0, 3.0])
    z = x * x + x          # dz/dx = 2x + 1
    z.backward()
    assert np.allclose(x.grad, 2.0 * np.array([1.0, 2.0, 3.0]) + 1.0)


def test_reuse_accumulates_scalar():
    x = Tensor(2.0)
    # f = x*x*x + x*x  ->  f' = 3x^2 + 2x = 12 + 4 = 16
    z = x * x * x + x * x
    z.backward()
    assert np.isclose(float(x.grad), 16.0)


def test_shared_subexpression():
    a = Tensor(3.0)
    b = a + 1.0           # 4
    c = a * 2.0           # 6
    out = b * c           # 24 ; d/da = (a+1)*2 + 2a = 2a+2+2a = 4a+2 = 14
    out.backward()
    assert np.isclose(float(a.grad), 14.0)


# --------------------------------------------------------------------------- #
# graph plumbing
# --------------------------------------------------------------------------- #
def test_prev_and_op_recorded():
    x = Tensor(1.0)
    y = Tensor(2.0)
    z = x + y
    assert set(z._prev) == {x, y} or set(z._prev) == {y, x}
    assert z._op == "+"
    p = x ** 2
    assert p._op == "**"
    assert p._prev == (x,)
    r = x.relu()
    assert r._op == "relu"


def test_backward_seeds_output_with_ones():
    x = Tensor([1.0, 2.0])
    y = x * 1.0
    y.backward()
    # grad of y wrt itself is ones -> grad flows as ones into x (since *1)
    assert np.allclose(x.grad, np.ones(2))


def test_zero_grad_resets():
    x = Tensor([1.0, 2.0, 3.0])
    z = x * x
    z.backward()
    assert not np.allclose(x.grad, 0.0)
    x.zero_grad()
    assert np.allclose(x.grad, 0.0)
    assert x.grad.shape == x.data.shape


def test_numbers_on_both_sides():
    x = Tensor(4.0)
    out = 10.0 - x / 2.0 + 1.0   # = 10 - 2 + 1 = 9 ; d/dx = -0.5
    assert np.isclose(float(out.data), 9.0)
    out.backward()
    assert np.isclose(float(x.grad), -0.5)
