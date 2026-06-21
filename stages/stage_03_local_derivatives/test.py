"""Tests for Stage 3: Local derivatives.

We test the per-op local-derivative functions against:
  1. their hand-derived closed forms, and
  2. a central-difference numerical estimate of each partial.

We also test the ``local_derivatives`` dispatcher against a minimal ``Value``
stub that mirrors the stage_02 interface (``.data``, ``._op`` and operand-ordered
parents), so this stage's tests run without standing up a full stage_02 graph.
The dispatcher itself is the new behavior this stage adds on top of stage_02's
imported ``Value``.
"""

import math

import pytest

from code import (
    d_add,
    d_sub,
    d_mul,
    d_div,
    d_neg,
    local_derivatives,
)

EPS = 1e-6
TOL = 1e-6


# ---------------------------------------------------------------------------
# Central-difference helpers
# ---------------------------------------------------------------------------


def central_diff_partial(f, args, i, eps=EPS):
    """Estimate df/d(args[i]) at ``args`` via central differences.

    ``f`` takes positional float args and returns a float.
    """
    plus = list(args)
    minus = list(args)
    plus[i] += eps
    minus[i] -= eps
    return (f(*plus) - f(*minus)) / (2 * eps)


# Forward functions matching each op (used only to numerically check grads).
_F = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "neg": lambda a: -a,
}


# ---------------------------------------------------------------------------
# Minimal Value stub mirroring stage_02's Value interface
# ---------------------------------------------------------------------------


class StubNode:
    """Minimal stand-in for the stage_02 ``Value``.

    Exposes ``data`` and ``_op`` -- the core fields ``local_derivatives``
    reads -- plus operand-ordered parents under both ``_inputs`` (an ordered
    tuple) and ``_prev`` (a set, as stage_02 stores them), so the dispatcher
    works however it recovers operand order. A leaf uses ``_op == ""``.
    """

    def __init__(self, value, op="", inputs=()):
        self.data = float(value)
        self.value = self.data  # alias, in case a stage exposes ``.value``
        self._op = op
        self._inputs = tuple(inputs)
        self._prev = set(inputs)


def leaf(v):
    return StubNode(v)


# ---------------------------------------------------------------------------
# d_add
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.5, 4.25), (0.0, -7.0)])
def test_d_add_closed_form(a, b):
    da, db = d_add(a, b)
    assert da == pytest.approx(1.0)
    assert db == pytest.approx(1.0)


@pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.5, 4.25), (0.0, -7.0)])
def test_d_add_gradcheck(a, b):
    da, db = d_add(a, b)
    nda = central_diff_partial(_F["add"], (a, b), 0)
    ndb = central_diff_partial(_F["add"], (a, b), 1)
    assert da == pytest.approx(nda, abs=TOL), f"d_add da analytic={da} numeric={nda}"
    assert db == pytest.approx(ndb, abs=TOL), f"d_add db analytic={db} numeric={ndb}"


# ---------------------------------------------------------------------------
# d_sub
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.5, 4.25), (9.0, -7.0)])
def test_d_sub_closed_form(a, b):
    da, db = d_sub(a, b)
    assert da == pytest.approx(1.0)
    assert db == pytest.approx(-1.0)


@pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.5, 4.25), (9.0, -7.0)])
def test_d_sub_gradcheck(a, b):
    da, db = d_sub(a, b)
    nda = central_diff_partial(_F["sub"], (a, b), 0)
    ndb = central_diff_partial(_F["sub"], (a, b), 1)
    assert da == pytest.approx(nda, abs=TOL), f"d_sub da analytic={da} numeric={nda}"
    assert db == pytest.approx(ndb, abs=TOL), f"d_sub db analytic={db} numeric={ndb}"


# ---------------------------------------------------------------------------
# d_mul
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.5, 4.25), (6.0, -7.0)])
def test_d_mul_closed_form(a, b):
    da, db = d_mul(a, b)
    assert da == pytest.approx(b), "dz/da of a*b must equal b"
    assert db == pytest.approx(a), "dz/db of a*b must equal a"


@pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.5, 4.25), (6.0, -7.0)])
def test_d_mul_gradcheck(a, b):
    da, db = d_mul(a, b)
    nda = central_diff_partial(_F["mul"], (a, b), 0)
    ndb = central_diff_partial(_F["mul"], (a, b), 1)
    assert da == pytest.approx(nda, abs=TOL), f"d_mul da analytic={da} numeric={nda}"
    assert db == pytest.approx(ndb, abs=TOL), f"d_mul db analytic={db} numeric={ndb}"


# ---------------------------------------------------------------------------
# d_div  (asymmetric!)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.5, 4.25), (6.0, -7.0)])
def test_d_div_closed_form(a, b):
    da, db = d_div(a, b)
    assert da == pytest.approx(1.0 / b), "dz/da of a/b must equal 1/b"
    assert db == pytest.approx(-a / b ** 2), "dz/db of a/b must equal -a/b**2"


@pytest.mark.parametrize("a,b", [(1.0, 2.0), (-3.5, 4.25), (6.0, -7.0)])
def test_d_div_gradcheck(a, b):
    da, db = d_div(a, b)
    nda = central_diff_partial(_F["div"], (a, b), 0)
    ndb = central_diff_partial(_F["div"], (a, b), 1)
    assert da == pytest.approx(nda, abs=TOL), f"d_div da analytic={da} numeric={nda}"
    assert db == pytest.approx(ndb, abs=TOL), f"d_div db analytic={db} numeric={ndb}"


def test_d_div_not_symmetric():
    # Division is not symmetric: the two partials must differ in general.
    da, db = d_div(3.0, 2.0)
    assert da != pytest.approx(db)


def test_d_div_zero_denominator_raises():
    with pytest.raises(ZeroDivisionError):
        d_div(1.0, 0.0)


# ---------------------------------------------------------------------------
# d_neg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [1.0, -3.5, 0.0, 12.0])
def test_d_neg_closed_form(a):
    (da,) = d_neg(a)
    assert da == pytest.approx(-1.0)


@pytest.mark.parametrize("a", [1.0, -3.5, 12.0])
def test_d_neg_gradcheck(a):
    (da,) = d_neg(a)
    nda = central_diff_partial(_F["neg"], (a,), 0)
    assert da == pytest.approx(nda, abs=TOL), f"d_neg analytic={da} numeric={nda}"


# ---------------------------------------------------------------------------
# local_derivatives dispatcher
# ---------------------------------------------------------------------------


def test_local_derivatives_add():
    z = StubNode(5.0, op="+", inputs=(leaf(2.0), leaf(3.0)))
    assert local_derivatives(z) == pytest.approx((1.0, 1.0))


def test_local_derivatives_sub():
    z = StubNode(-1.0, op="-", inputs=(leaf(2.0), leaf(3.0)))
    assert local_derivatives(z) == pytest.approx((1.0, -1.0))


def test_local_derivatives_mul():
    a, b = 2.0, 3.0
    z = StubNode(a * b, op="*", inputs=(leaf(a), leaf(b)))
    # dz/da = b, dz/db = a -- in input order.
    assert local_derivatives(z) == pytest.approx((b, a))


def test_local_derivatives_div():
    a, b = 6.0, 4.0
    z = StubNode(a / b, op="/", inputs=(leaf(a), leaf(b)))
    assert local_derivatives(z) == pytest.approx((1.0 / b, -a / b ** 2))


def test_local_derivatives_neg():
    z = StubNode(-7.0, op="neg", inputs=(leaf(7.0),))
    assert local_derivatives(z) == pytest.approx((-1.0,))


def test_local_derivatives_leaf_returns_empty():
    assert local_derivatives(leaf(3.0)) == ()


def test_local_derivatives_order_matches_inputs():
    # For mul the partials are (b, a); the result order must follow inputs.
    a, b = 10.0, 0.5
    z = StubNode(a * b, op="*", inputs=(leaf(a), leaf(b)))
    da, db = local_derivatives(z)
    assert da == pytest.approx(b)
    assert db == pytest.approx(a)


def test_local_derivatives_unknown_op_raises():
    bogus = StubNode(0.0, op="???", inputs=(leaf(1.0), leaf(2.0)))
    with pytest.raises(ValueError):
        local_derivatives(bogus)
