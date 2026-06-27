"""Tests for Stage 08: Tensor engine.

Verifies the N-dimensional autodiff `Tensor`:
  * forward values and operator overloading (incl. reflected ops with numbers),
  * gradient accumulation on reused nodes (e.g. y = x*x + x),
  * elementwise array backward shapes/values,
  * central-difference gradient checks against the analytic `.grad` filled by
    `.backward()`.

This stage has no reduction op yet (`.sum()` arrives in stage_12), so backward
seeds the output with ones_like and we therefore gradcheck on SCALAR (0-d)
tensors, where the network output IS the scalar loss -- no reduction needed.
The chain-rule code under test is identical for scalars and arrays; the array
tests separately confirm the engine works elementwise at full shape.

Run: pytest stage_08_tensor_engine/test.py
"""
import os as _os
import sys as _sys

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# Make the shared `dlfs` shim importable (it lives at the curriculum root) so
# this stage's code.py can `from dlfs import stage_import` to pull in stage_05's
# `Value`, the scalar engine this `Tensor` unifies onto arrays.
sys.path.insert(0, _ROOT)

# `code.py` defines this stage's `Tensor` (the new n-d core) and re-imports
# `Value` from stage_05 via dlfs.stage_import for the `from_value` bridge.
try:
    # --- resolve sibling code.py (avoid stdlib `code` collision) ---
    import importlib.util as _ilu
    _THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_THIS_DIR)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    _spec = _ilu.spec_from_file_location(
        "code", _os.path.join(_THIS_DIR, "code.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["code"] = _mod
    _spec.loader.exec_module(_mod)
    from code import Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_08 Tensor not importable yet: {exc}",
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
    "tanh":        (lambda t: t.tanh(),           lambda x: math.tanh(x)),
    "exp":         (lambda t: t.exp(),            lambda x: math.exp(x)),
    "log":         (lambda t: t.log(),            lambda x: math.log(x)),
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

    if name == "log" and x <= 0:
        pytest.skip("log undefined for non-positive input")

    ana = analytic_grad_scalar(build, x)
    num = numeric_grad_scalar(ref, x)
    assert np.isclose(ana, num, rtol=1e-4, atol=TOL), (
        f"{name} at x={x}: analytic grad {ana} != numeric {num}"
    )


@pytest.mark.parametrize("name", list(SCALAR_FUNCS))
@pytest.mark.parametrize("x", SCALAR_INPUTS)
def test_scalar_forward(name, x):
    build, ref = SCALAR_FUNCS[name]
    if name in ("powhalf", "rdiv_const", "div_const", "log") and x <= 0:
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


def test_data_is_assignable():
    # Downstream code (e.g. Neuron training / gradcheck) overwrites params in
    # place via `t.data = new_array`. The `data` property must expose a setter;
    # a read-only property raises "property 'data' has no setter".
    t = Tensor([1.0, 2.0])
    t.data = np.array([3.0, 4.0])
    assert np.allclose(t.data, [3.0, 4.0])
    assert isinstance(t.data, np.ndarray) and t.data.dtype == np.float64


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


# --------------------------------------------------------------------------- #
# elementwise transcendentals on NON-scalar arrays
# --------------------------------------------------------------------------- #
# These must work elementwise at full shape, not just on 0-d tensors. A
# `math.tanh`/`math.exp`/`math.log` implementation passes the scalar gradchecks
# above but raises "only 0-dimensional arrays can be converted to Python
# scalars" the moment a real (n,) input arrives (e.g. a Neuron's tanh over a
# batch). Use np.tanh / np.exp / np.log so the forward + local grad broadcast.
def test_array_tanh_forward_backward():
    x = Tensor([-1.0, 0.0, 2.0])
    z = x.tanh()
    assert np.allclose(z.data, np.tanh(x.data))
    z.backward()
    assert np.allclose(x.grad, 1.0 - np.tanh(x.data) ** 2)
    assert x.grad.shape == x.data.shape


def test_array_exp_forward_backward():
    x = Tensor([-1.0, 0.5, 1.5])
    z = x.exp()
    assert np.allclose(z.data, np.exp(x.data))
    z.backward()
    # d/dx exp(x) = exp(x)
    assert np.allclose(x.grad, np.exp(x.data))
    assert x.grad.shape == x.data.shape


def test_array_log_forward_backward():
    x = Tensor([0.5, 1.0, 3.0])
    z = x.log()
    assert np.allclose(z.data, np.log(x.data))
    z.backward()
    # d/dx log(x) = 1/x
    assert np.allclose(x.grad, 1.0 / x.data)
    assert x.grad.shape == x.data.shape


def test_2d_tanh_forward_backward():
    x = Tensor([[-1.0, 0.0], [0.5, 2.0]])
    z = x.tanh()
    assert np.allclose(z.data, np.tanh(x.data))
    z.backward()
    assert np.allclose(x.grad, 1.0 - np.tanh(x.data) ** 2)
    assert x.grad.shape == x.data.shape


# --------------------------------------------------------------------------- #
# 0-d scalar operand broadcasts into an (n,)/(2-d) tensor
# --------------------------------------------------------------------------- #
# A Neuron stores its bias as a 0-d Tensor and computes `z = x @ w + b`. For a
# batched input `x @ w` is (batch,), so `(batch,) + 0-d` must broadcast and the
# bias must collect the summed upstream grad. A strict equal-shape `_at_shape`
# assert rejects this even though the single-example `() + 0-d` case slips
# through, so only a batched/array test catches the gap.
def test_scalar_tensor_broadcasts_into_vector_add():
    v = Tensor([1.0, 2.0, 3.0])
    b = Tensor(0.5)  # 0-d
    z = v + b
    assert np.allclose(z.data, [1.5, 2.5, 3.5])
    z.backward()
    assert np.allclose(v.grad, np.ones(3))
    # 0-d bias receives the sum of the upstream ones across the broadcast axis.
    assert np.isclose(float(b.grad), 3.0)
    assert b.grad.shape == b.data.shape


def test_scalar_tensor_broadcasts_into_vector_mul():
    v = Tensor([1.0, 2.0, 3.0])
    s = Tensor(2.0)  # 0-d
    z = v * s
    assert np.allclose(z.data, [2.0, 4.0, 6.0])
    z.backward()
    # dz/dv = s (broadcast), dz/ds = sum(v) over the broadcast axis
    assert np.allclose(v.grad, [2.0, 2.0, 2.0])
    assert np.isclose(float(s.grad), 6.0)
    assert s.grad.shape == s.data.shape


def test_2d_chain_backward():
    x = Tensor([[1.0, -2.0], [3.0, 4.0]])
    z = (x * 2.0).relu()
    z.backward()
    expected = 2.0 * (x.data > 0)
    assert np.allclose(x.grad, expected)
    assert x.grad.shape == x.data.shape


# --------------------------------------------------------------------------- #
# matmul (@): forward value + gradient rule dL/dA = G@B.T, dL/dB = A.T@G
# --------------------------------------------------------------------------- #
def test_matmul_forward():
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[5.0, 6.0], [7.0, 8.0]])
    C = A @ B
    assert np.allclose(C.data, np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[5.0, 6.0], [7.0, 8.0]]))
    assert C._op == "@"
    assert set(C._prev) == {A, B}


def test_matmul_backward_2d():
    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # (2,3)
    B = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3,2)
    C = A @ B                                          # (2,2)
    C.backward()                                       # seeds G = ones((2,2))
    G = np.ones((2, 2))
    assert np.allclose(A.grad, G @ B.data.T), "dL/dA must equal G @ B.T"
    assert np.allclose(B.grad, A.data.T @ G), "dL/dB must equal A.T @ G"
    assert A.grad.shape == A.data.shape and B.grad.shape == B.data.shape


def test_matmul_vector_cases():
    # (n,) @ (n,m) -> (m,)  : the dense-layer single-example case
    x = Tensor([1.0, 2.0])               # (2,)
    W = Tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])  # (2,3)
    y = x @ W                            # (3,)
    assert np.allclose(y.data, x.data @ W.data)
    y.backward()
    assert x.grad.shape == x.data.shape and W.grad.shape == W.data.shape


def test_matmul_vector_dot_produces_scalar_backward():
    # (n,) @ (n,) -> 0-d scalar : the Neuron's `x @ w` single-example pre-activation.
    # backward must NOT do `result.grad @ other.data.T` blindly: when result is 0-d
    # that crashes ("Input operand 0 does not have enough dimensions"). For a dot
    # product z = a . b the grads are dz/da = b, dz/db = a (seeded with ones -> 1.0).
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    z = a @ b                             # 0-d
    assert z.shape == ()
    assert np.isclose(float(z.data), 1 * 4 + 2 * 5 + 3 * 6)
    z.backward()                          # seeds G = 1.0 (0-d)
    assert np.allclose(a.grad, b.data), "dL/da of a.b must equal b"
    assert np.allclose(b.grad, a.data), "dL/db of a.b must equal a"
    assert a.grad.shape == a.data.shape and b.grad.shape == b.data.shape


# --------------------------------------------------------------------------- #
# reshape: pure rearrangement, grad flows back under the inverse reshape
# --------------------------------------------------------------------------- #
def test_reshape_forward_varargs():
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    z = x.reshape(2, 3)
    assert z.shape == (2, 3)
    assert np.allclose(z.data, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_reshape_forward_tuple():
    # must accept a single tuple arg too: t.reshape((2, 3))
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    z = x.reshape((3, 2))
    assert z.shape == (3, 2)
    assert np.allclose(z.data, x.data.reshape(3, 2))


def test_reshape_minus_one_flattens():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    z = x.reshape(-1)                       # NumPy infers the (4,) shape
    assert z.shape == (4,)
    assert np.allclose(z.data, [1.0, 2.0, 3.0, 4.0])


def test_reshape_backward_routes_grad_to_original_shape():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2,3)
    z = x.reshape(6)                                  # (6,)
    z.backward()                                      # seeds ones((6,))
    # pure rearrangement: every grad entry is 1, reshaped back to x's shape
    assert x.grad.shape == x.data.shape
    assert np.allclose(x.grad, np.ones((2, 3)))


def test_reshape_backward_through_op_preserves_values():
    # grad must travel the inverse reshape with the RIGHT per-entry values,
    # not just the right shape. Square after reshape so dz/dx = 2x, then
    # confirm each entry lands back in its original (2,2) slot.
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    z = (x.reshape(4) ** 2)                 # (4,), entrywise x**2
    z.backward()
    assert np.allclose(x.grad, 2.0 * x.data)
    assert x.grad.shape == x.data.shape


def test_reshape_records_graph():
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    z = x.reshape(2, 2)
    assert z._prev == (x,)


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
