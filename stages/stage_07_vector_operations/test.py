"""Stage 7 tests: Vec operations + central-difference gradient checks.

Run: pytest stage_07_vector_operations/test.py

`code.py` imports the finished `Value` from stage_06 via `dlfs.stage_import`
(stage_06 already supplies `relu`) and adds the new `Vec` class on top. Both are
re-exported from `code`, so the imports below are unchanged. Gradients are
verified against numerical central differences:
df/dx ~= (f(x+eps) - f(x-eps)) / (2*eps).
"""

import math
import random

import pytest

from code import Value, Vec


EPS = 1e-6
TOL = 1e-5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def numeric_grad(f, xs, i, eps=EPS):
    """Central-difference d f / d xs[i], where f takes a list of floats."""
    plus = list(xs)
    minus = list(xs)
    plus[i] += eps
    minus[i] -= eps
    return (f(plus) - f(minus)) / (2 * eps)


def approx(a, b, tol=TOL):
    return abs(a - b) <= tol + tol * abs(b)


# ---------------------------------------------------------------------------
# container basics
# ---------------------------------------------------------------------------
def test_construction_and_container_protocol():
    v = Vec([1.0, 2.0, Value(3.0)])
    assert len(v) == 3
    assert isinstance(v[0], Value)
    assert v[0].data == 1.0
    assert v[2].data == 3.0
    assert [x.data for x in v] == [1.0, 2.0, 3.0]


def test_existing_value_not_rewrapped():
    a = Value(5.0)
    v = Vec([a, 1.0])
    assert v[0] is a, "an entry that is already a Value must be kept, not re-wrapped"


# ---------------------------------------------------------------------------
# forward correctness
# ---------------------------------------------------------------------------
def test_elementwise_add_sub_mul_forward():
    a = Vec([1.0, 2.0, 3.0])
    b = Vec([4.0, 5.0, 6.0])
    assert [x.data for x in (a + b)] == [5.0, 7.0, 9.0]
    assert [x.data for x in (b - a)] == [3.0, 3.0, 3.0]
    assert [x.data for x in (a * b)] == [4.0, 10.0, 18.0]


def test_scalar_broadcast_forward():
    a = Vec([1.0, 2.0, 3.0])
    assert [x.data for x in (a + 10)] == [11.0, 12.0, 13.0]
    assert [x.data for x in (a * 2)] == [2.0, 4.0, 6.0]
    # reflected forms
    assert [x.data for x in (10 + a)] == [11.0, 12.0, 13.0]
    assert [x.data for x in (2 * a)] == [2.0, 4.0, 6.0]
    assert [x.data for x in (10 - a)] == [9.0, 8.0, 7.0]


def test_dot_and_sum_forward():
    a = Vec([1.0, 2.0, 3.0])
    b = Vec([4.0, 5.0, 6.0])
    assert isinstance(a.dot(b), Value)
    assert a.dot(b).data == pytest.approx(32.0)
    assert a.sum().data == pytest.approx(6.0)


def test_relu_forward():
    v = Vec([-2.0, -0.5, 0.0, 1.5])
    assert [x.data for x in v.relu()] == [0.0, 0.0, 0.0, 1.5]


def test_length_mismatch_raises():
    a = Vec([1.0, 2.0])
    b = Vec([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        _ = a + b
    with pytest.raises(ValueError):
        _ = a.dot(b)


def test_no_input_mutation():
    a = Vec([1.0, 2.0])
    b = Vec([3.0, 4.0])
    _ = a + b
    assert [x.data for x in a] == [1.0, 2.0]
    assert [x.data for x in b] == [3.0, 4.0]


# ---------------------------------------------------------------------------
# gradient checks
# ---------------------------------------------------------------------------
def test_dot_gradient_matches_numeric():
    a_vals = [1.5, -2.0, 0.7]
    b_vals = [0.3, 1.1, -0.9]

    a = Vec(a_vals)
    b = Vec(b_vals)
    out = a.dot(b)
    out.backward()

    # analytical: d/d a_i = b_i, d/d b_i = a_i
    def f_a(xs):
        return Vec(xs).dot(Vec(b_vals)).data

    def f_b(xs):
        return Vec(a_vals).dot(Vec(xs)).data

    for i in range(len(a_vals)):
        g = numeric_grad(f_a, a_vals, i)
        assert approx(a[i].grad, g), f"a[{i}].grad {a[i].grad} != numeric {g}"
    for i in range(len(b_vals)):
        g = numeric_grad(f_b, b_vals, i)
        assert approx(b[i].grad, g), f"b[{i}].grad {b[i].grad} != numeric {g}"


def test_sum_gradient_is_one_each():
    vals = [3.0, -1.0, 4.0]
    v = Vec(vals)
    out = v.sum()
    out.backward()
    for i in range(len(vals)):
        assert approx(v[i].grad, 1.0), f"sum grad of element {i} should be 1"


def test_elementwise_mul_then_sum_gradient():
    # f = sum(a * b); d/d a_i = b_i, d/d b_i = a_i
    a_vals = [0.5, -1.2, 2.0]
    b_vals = [3.0, 0.4, -0.7]
    a = Vec(a_vals)
    b = Vec(b_vals)
    out = (a * b).sum()
    out.backward()

    def f_a(xs):
        return (Vec(xs) * Vec(b_vals)).sum().data

    def f_b(xs):
        return (Vec(a_vals) * Vec(xs)).sum().data

    for i in range(len(a_vals)):
        assert approx(a[i].grad, numeric_grad(f_a, a_vals, i))
        assert approx(b[i].grad, numeric_grad(f_b, b_vals, i))


def test_scalar_broadcast_accumulates_summed_gradient():
    # f = sum(a + c) with c a single shared Value broadcast across the vector.
    # d f / d c = number of elements (each contributes 1, summed by += accumulation).
    a_vals = [1.0, 2.0, 3.0, 4.0]
    c = Value(0.5)
    a = Vec(a_vals)
    out = (a + c).sum()
    out.backward()
    assert approx(c.grad, float(len(a_vals))), (
        f"broadcast scalar grad {c.grad} should be sum of per-element grads "
        f"= {len(a_vals)}"
    )
    # and each a_i still gets grad 1
    for i in range(len(a_vals)):
        assert approx(a[i].grad, 1.0)


def test_scalar_broadcast_mul_gradient():
    # f = sum(a * c); d f / d c = sum_i a_i ; d f / d a_i = c
    a_vals = [1.0, -2.0, 3.5]
    c_val = 2.0
    a = Vec(a_vals)
    c = Value(c_val)
    out = (a * c).sum()
    out.backward()
    assert approx(c.grad, sum(a_vals))
    for i in range(len(a_vals)):
        assert approx(a[i].grad, c_val)


def test_relu_gradient_matches_numeric():
    vals = [-1.3, -0.0001, 0.0001, 2.4]
    v = Vec(vals)
    out = v.relu().sum()
    out.backward()

    def f(xs):
        return Vec(xs).relu().sum().data

    for i in range(len(vals)):
        g = numeric_grad(f, vals, i)
        assert approx(v[i].grad, g, tol=1e-4), (
            f"relu grad element {i}: {v[i].grad} vs numeric {g}"
        )


def test_composite_neuron_preactivation_gradient():
    # The whole point: a neuron pre-activation s = w . x + b, then relu.
    random.seed(0)
    n = 5
    w_vals = [random.uniform(-1, 1) for _ in range(n)]
    x_vals = [random.uniform(-1, 1) for _ in range(n)]
    b_val = 0.3

    w = Vec(w_vals)
    x = Vec(x_vals)
    b = Value(b_val)
    out = (w.dot(x) + b).relu()
    out.backward()

    def f_w(ws):
        s = Vec(ws).dot(Vec(x_vals)) + Value(b_val)
        return s.relu().data

    def f_b(bs):
        s = Vec(w_vals).dot(Vec(x_vals)) + Value(bs[0])
        return s.relu().data

    for i in range(n):
        assert approx(w[i].grad, numeric_grad(f_w, w_vals, i), tol=1e-4)
    assert approx(b.grad, numeric_grad(f_b, [b_val], 0), tol=1e-4)
