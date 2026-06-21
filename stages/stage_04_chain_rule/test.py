"""Tests for Stage 4: Chain rule.

Run with:  pytest stage_04_chain_rule/test.py

These run against this stage's ``code.py``, which reuses ``Value`` (stage_02)
and the ``d_*`` locals (stage_03) via ``dlfs.stage_import`` and ADDS the
chain-rule plumbing on top -- so the cumulative framework is exercised.

Gradient checks use central differences (f(x+eps) - f(x-eps)) / (2*eps) and
compare against the ANALYTICAL derivative your code returns. NumPy is used only
to build/compare the numerical reference; your code must not use it.
"""

import math

import numpy as np
import pytest

from code import (
    accumulate,
    backward_pass,
    chain,
    f_branch,
    f_chain,
)

TOL = 1e-5
EPS = 1e-6


def central_diff(value_fn, x, eps=EPS):
    """Numerical derivative of a scalar->scalar function via central differences."""
    return (value_fn(x + eps) - value_fn(x - eps)) / (2.0 * eps)


# ---------------------------------------------------------------------------
# chain: product of locals along a straight path
# ---------------------------------------------------------------------------

def test_chain_product():
    assert chain([2.0, 3.0, 4.0]) == pytest.approx(24.0)


def test_chain_single():
    assert chain([7.5]) == pytest.approx(7.5)


def test_chain_empty_is_identity():
    assert chain([]) == pytest.approx(1.0), "empty chain must return 1.0"


def test_chain_with_zero():
    assert chain([5.0, 0.0, 9.0]) == pytest.approx(0.0)


def test_chain_order_independent():
    assert chain([1.5, -2.0, 4.0]) == pytest.approx(chain([4.0, 1.5, -2.0]))


# ---------------------------------------------------------------------------
# accumulate: sum over merging paths
# ---------------------------------------------------------------------------

def test_accumulate_sum():
    assert accumulate([1.0, 2.0, 3.0]) == pytest.approx(6.0)


def test_accumulate_empty_is_zero():
    assert accumulate([]) == pytest.approx(0.0), "empty accumulate must return 0.0"


def test_accumulate_negatives():
    assert accumulate([5.0, -2.0, -1.0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# f_chain: straight chain u = 3x+1, v = u^2, y = tanh(v)
# ---------------------------------------------------------------------------

def _f_chain_value(x):
    u = 3.0 * x + 1.0
    v = u * u
    return math.tanh(v)


@pytest.mark.parametrize("x", [-1.3, -0.4, 0.0, 0.2, 0.7])
def test_f_chain_value_matches_forward(x):
    y, _ = f_chain(x)
    assert y == pytest.approx(_f_chain_value(x), abs=1e-9)


@pytest.mark.parametrize("x", [-1.3, -0.4, 0.0, 0.2, 0.7])
def test_f_chain_gradcheck(x):
    _, dy_dx = f_chain(x)
    num = central_diff(_f_chain_value, x)
    assert dy_dx == pytest.approx(num, abs=TOL, rel=1e-4), (
        f"f_chain dy/dx at x={x}: analytical={dy_dx}, numerical={num}"
    )


def test_f_chain_known_point():
    # x = 0: u = 1, v = 1, y = tanh(1).
    # dy/dx = (1 - tanh(1)^2) * 2u * 3 = (1 - tanh(1)^2) * 6
    expected = (1.0 - math.tanh(1.0) ** 2) * 6.0
    _, dy_dx = f_chain(0.0)
    assert dy_dx == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# f_branch: a = x+1, b = x^2, y = a*b  (x feeds both branches)
# ---------------------------------------------------------------------------

def _f_branch_value(x):
    a = x + 1.0
    b = x * x
    return a * b


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.1, 3.0])
def test_f_branch_value_matches_forward(x):
    y, _ = f_branch(x)
    assert y == pytest.approx(_f_branch_value(x), abs=1e-9)


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.1, 3.0])
def test_f_branch_gradcheck(x):
    _, dy_dx = f_branch(x)
    num = central_diff(_f_branch_value, x)
    assert dy_dx == pytest.approx(num, abs=TOL, rel=1e-4), (
        f"f_branch dy/dx at x={x}: analytical={dy_dx}, numerical={num}"
    )


def test_f_branch_known_point():
    # y = (x+1)*x^2 = x^3 + x^2  ->  dy/dx = 3x^2 + 2x. At x=2: 12 + 4 = 16.
    _, dy_dx = f_branch(2.0)
    assert dy_dx == pytest.approx(16.0, abs=1e-9)


# ---------------------------------------------------------------------------
# backward_pass: full reverse pass over f_branch's expression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.1, 3.0])
def test_backward_pass_seed(x):
    grads = backward_pass(x)
    assert grads["y"] == pytest.approx(1.0), "must seed dy/dy = 1.0"


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.1, 3.0])
def test_backward_pass_keys(x):
    grads = backward_pass(x)
    assert set(grads.keys()) == {"y", "a", "b", "x"}


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.1, 3.0])
def test_backward_pass_intermediates(x):
    # a = x+1, b = x^2, y = a*b  =>  dy/da = b, dy/db = a
    grads = backward_pass(x)
    assert grads["a"] == pytest.approx(x * x, abs=1e-9)          # dy/da = b
    assert grads["b"] == pytest.approx(x + 1.0, abs=1e-9)        # dy/db = a


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.1, 3.0])
def test_backward_pass_matches_f_branch(x):
    grads = backward_pass(x)
    _, dy_dx = f_branch(x)
    assert grads["x"] == pytest.approx(dy_dx, abs=1e-9), (
        "backward_pass dy/dx must equal f_branch dy/dx"
    )


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.1, 3.0])
def test_backward_pass_gradcheck(x):
    grads = backward_pass(x)
    num = central_diff(_f_branch_value, x)
    assert grads["x"] == pytest.approx(num, abs=TOL, rel=1e-4), (
        f"backward_pass dy/dx at x={x}: analytical={grads['x']}, numerical={num}"
    )
