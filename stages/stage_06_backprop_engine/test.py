"""Tests for Stage 6: Scalar backprop engine.

We verify:
  - forward values of every op,
  - the canonical reused-node example (grad accumulation),
  - that every analytical gradient from .backward() matches a central-difference
    numerical gradient within tolerance,
  - that __pow__ rejects a Value exponent,
  - that grads accumulate (+=) and can be zeroed for independent passes.

Run:  pytest stage_06_backprop_engine/test.py
"""

import importlib.util
import math
import os
import sys

import pytest

# This stage's ``code.py`` extends stage_05's reverse-mode ``Value`` via
# ``dlfs.stage_import``, so the curriculum root (which holds the ``dlfs``
# package) must be importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Load this stage's ``code.py`` by file path. A plain ``from code import ...``
# would collide with Python's standard-library ``code`` module (which pytest's
# debugger imports), so we load the sibling file explicitly. This runs the
# tests against THIS stage's extended ``Value`` (a subclass of stage_05's),
# regardless of which directory pytest is invoked from.
_spec = importlib.util.spec_from_file_location(
    "stage_06_code", os.path.join(_THIS_DIR, "code.py")
)
_code = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_code)
Value = _code.Value

TOL = 1e-5
EPS = 1e-6


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #
def numerical_grad(f, xs, i, eps=EPS):
    """Central-difference d f / d x_i, where f maps a list of floats -> float.

    (f(x_i + eps) - f(x_i - eps)) / (2 * eps)
    """
    plus = list(xs)
    minus = list(xs)
    plus[i] += eps
    minus[i] -= eps
    return (f(plus) - f(minus)) / (2 * eps)


def analytical_grads(build, xs):
    """Build a Value graph from floats `xs` via `build`, run backward, return grads.

    `build(values) -> Value` constructs the output Value from a list of leaf
    Values. Returns the list of leaf .grad values in the same order.
    """
    leaves = [Value(x) for x in xs]
    out = build(leaves)
    out.backward()
    return [leaf.grad for leaf in leaves], out.data


def check_gradcheck(name, build, xs):
    """Assert analytical grads match central-difference grads for each input."""
    grads, _ = analytical_grads(build, xs)

    def f(vals):
        # Forward-only evaluation using the same expression on plain floats:
        # rebuild with Value but read .data (no backward needed).
        leaves = [Value(v) for v in vals]
        return build(leaves).data

    for i in range(len(xs)):
        ng = numerical_grad(f, xs, i)
        assert grads[i] == pytest.approx(ng, abs=TOL, rel=1e-4), (
            f"[{name}] grad mismatch for input {i}: "
            f"analytical={grads[i]:.8f} numerical={ng:.8f}"
        )


# ---------------------------------------------------------------------------- #
# Forward values
# ---------------------------------------------------------------------------- #
def test_forward_add_mul():
    a = Value(2.0)
    b = Value(3.0)
    assert (a + b).data == pytest.approx(5.0)
    assert (a * b).data == pytest.approx(6.0)


def test_forward_scalar_coercion():
    a = Value(2.0)
    assert (a + 1).data == pytest.approx(3.0)
    assert (1 + a).data == pytest.approx(3.0)
    assert (2 * a).data == pytest.approx(4.0)
    assert (a * 2).data == pytest.approx(4.0)
    assert (1 - a).data == pytest.approx(-1.0)
    assert (a - 1).data == pytest.approx(1.0)
    assert (-a).data == pytest.approx(-2.0)


def test_forward_unary_ops():
    x = Value(0.7)
    assert x.tanh().data == pytest.approx(math.tanh(0.7), abs=TOL)
    assert x.exp().data == pytest.approx(math.exp(0.7), abs=TOL)
    assert Value(-2.0).relu().data == pytest.approx(0.0)
    assert Value(3.0).relu().data == pytest.approx(3.0)
    assert (Value(2.0) ** 3).data == pytest.approx(8.0)


# ---------------------------------------------------------------------------- #
# The canonical reused-node example
# ---------------------------------------------------------------------------- #
def test_canonical_reused_node():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = c + a
    d.backward()
    # d = a*b + a -> dd/da = b + 1 = 4, dd/db = a = 2
    assert a.grad == pytest.approx(4.0, abs=TOL), f"a.grad={a.grad}, expected 4.0"
    assert b.grad == pytest.approx(2.0, abs=TOL), f"b.grad={b.grad}, expected 2.0"
    assert d.data == pytest.approx(8.0)


def test_node_reused_many_times_accumulates():
    a = Value(3.0)
    # b = a + a + a -> db/da = 3
    b = a + a + a
    b.backward()
    assert a.grad == pytest.approx(3.0, abs=TOL), f"a.grad={a.grad}, expected 3.0"


# ---------------------------------------------------------------------------- #
# Gradient checks for each op via central differences
# ---------------------------------------------------------------------------- #
def test_gradcheck_add():
    check_gradcheck("add", lambda v: v[0] + v[1], [1.5, -2.3])


def test_gradcheck_mul():
    check_gradcheck("mul", lambda v: v[0] * v[1], [1.5, -2.3])


def test_gradcheck_pow():
    check_gradcheck("pow", lambda v: v[0] ** 3, [1.7])
    check_gradcheck("pow_neg", lambda v: v[0] ** -2, [1.7])
    check_gradcheck("pow_frac", lambda v: v[0] ** 0.5, [2.3])


def test_gradcheck_tanh():
    check_gradcheck("tanh", lambda v: v[0].tanh(), [0.6])
    check_gradcheck("tanh_neg", lambda v: v[0].tanh(), [-1.2])


def test_gradcheck_exp():
    check_gradcheck("exp", lambda v: v[0].exp(), [0.4])


def test_gradcheck_relu():
    # Use points away from the kink at 0 so central differences are valid.
    check_gradcheck("relu_pos", lambda v: v[0].relu(), [1.3])
    check_gradcheck("relu_neg", lambda v: v[0].relu(), [-0.8])


def test_gradcheck_sub_neg():
    check_gradcheck("sub", lambda v: v[0] - v[1], [2.0, 5.0])
    check_gradcheck("neg", lambda v: -v[0], [3.3])


# ---------------------------------------------------------------------------- #
# Composite expressions exercising the chain rule + reuse together
# ---------------------------------------------------------------------------- #
def test_gradcheck_composite_with_reuse():
    # f = tanh(a*b + a) * exp(b) ; a appears twice
    def build(v):
        a, b = v
        return (a * b + a).tanh() * b.exp()

    check_gradcheck("composite_reuse", build, [0.5, -0.7])


def test_gradcheck_neuron_like():
    # f = relu(w1*x1 + w2*x2 + bias) ; a tiny neuron pre-activation
    def build(v):
        w1, w2, x1, x2, bias = v
        return (w1 * x1 + w2 * x2 + bias).relu()

    check_gradcheck("neuron", build, [0.3, -0.5, 1.1, 2.0, 0.2])


def test_gradcheck_polynomial():
    # f = 3*a**2 + a*b - b**3
    def build(v):
        a, b = v
        return 3 * a ** 2 + a * b - b ** 3

    check_gradcheck("poly", build, [1.4, -0.9])


# ---------------------------------------------------------------------------- #
# Engine invariants
# ---------------------------------------------------------------------------- #
def test_pow_rejects_value_exponent():
    a = Value(2.0)
    b = Value(3.0)
    with pytest.raises((AssertionError, TypeError)):
        _ = a ** b


def test_grads_accumulate_across_backward_calls():
    # Two backward passes WITHOUT zeroing should accumulate (engine uses +=).
    a = Value(2.0)
    b = Value(3.0)
    d = a * b
    d.backward()
    g1 = a.grad
    d.backward()  # no zeroing -> seed self.grad=1 again and re-accumulate
    assert a.grad == pytest.approx(2 * g1, abs=TOL), (
        f"expected accumulation: a.grad={a.grad}, 2*g1={2 * g1}"
    )


def test_leaf_grads_start_at_zero():
    a = Value(5.0)
    assert a.grad == pytest.approx(0.0)


def test_repr_contains_data_and_grad():
    a = Value(2.0)
    a.grad = 4.0
    s = repr(a)
    assert "2.0" in s and "4.0" in s, f"repr missing data/grad: {s}"
