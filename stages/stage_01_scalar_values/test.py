"""Tests for Stage 01: Scalar values & arithmetic.

Run with:  pytest stage_01_scalar_values/test.py

This is the ORIGIN stage: it imports nothing from earlier stages, so the test
loads this stage's own ``code.py`` by file path (a stage's tests always run
against its own ``code.py``). Later stages will pull this ``Value`` forward via
``dlfs.stage_import`` and extend it.

There are no analytic gradients in this stage, so there is nothing to compare a
hand-derived derivative against. Instead, the final tests use the symmetric
CENTRAL DIFFERENCE
    f'(x) ~= (f(x + h) - f(x - h)) / (2h)
to confirm that derivatives of `Value` expressions exist and match the slope we
expect (e.g. d/da (a*b) = b). From stage 06 on, this same central-difference
recipe becomes the gradient check against your analytic `.backward()`.
"""

import importlib.util
import os

import pytest

# --- import the student's code.py living next to this test file ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "stage_01_code", os.path.join(_HERE, "code.py")
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
Value = _MOD.Value


# ---------------------------------------------------------------------------
# Construction & repr
# ---------------------------------------------------------------------------

def test_data_is_float():
    v = Value(3)
    assert isinstance(v.data, float), "Value.data must be coerced to float"
    assert v.data == 3.0


def test_grad_initialized_zero():
    v = Value(1.5)
    assert v.grad == 0.0, "Value.grad must be initialized to 0.0 (unused this stage)"


def test_repr():
    assert repr(Value(4.0)) == "Value(data=4.0)"


# ---------------------------------------------------------------------------
# Forward arithmetic matches raw-float arithmetic
# ---------------------------------------------------------------------------

def test_add():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == pytest.approx(-1.0)


def test_mul():
    a, b = Value(2.0), Value(-3.0)
    assert (a * b).data == pytest.approx(-6.0)


def test_sub():
    a, b = Value(2.0), Value(-3.0)
    assert (a - b).data == pytest.approx(5.0)


def test_div():
    a, b = Value(6.0), Value(3.0)
    assert (a / b).data == pytest.approx(2.0)


def test_neg():
    a = Value(2.0)
    assert (-a).data == pytest.approx(-2.0)


def test_pow():
    a = Value(3.0)
    assert (a ** 2).data == pytest.approx(9.0)
    assert (a ** -1).data == pytest.approx(1.0 / 3.0)


def test_pow_rejects_value_exponent():
    a = Value(3.0)
    with pytest.raises((AssertionError, TypeError)):
        _ = a ** Value(2.0)


def test_compound_expression():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b + c          # 2*-3 + 10 = 4
    assert d.data == pytest.approx(4.0)


def test_op_returns_new_value():
    a, b = Value(2.0), Value(3.0)
    out = a + b
    assert isinstance(out, Value)
    assert out is not a and out is not b


# ---------------------------------------------------------------------------
# Mixing Value with plain ints/floats on either side
# ---------------------------------------------------------------------------

def test_mixed_left_operand():
    a = Value(4.0)
    assert (a + 1).data == pytest.approx(5.0)
    assert (a * 3).data == pytest.approx(12.0)
    assert (a - 1.5).data == pytest.approx(2.5)
    assert (a / 2).data == pytest.approx(2.0)


def test_mixed_right_operand():
    a = Value(4.0)
    assert (1 + a).data == pytest.approx(5.0)
    assert (3 * a).data == pytest.approx(12.0)
    assert (10 - a).data == pytest.approx(6.0)
    assert (8 / a).data == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Numerical derivative check (central differences).
# No analytic gradients exist yet; we only confirm the slope of the forward
# function is what calculus predicts. d/da (a*b + c) = b ; d/da (a**2) = 2a.
# ---------------------------------------------------------------------------

def _central_diff(f, x, h=1e-6):
    """(f(x+h) - f(x-h)) / (2h), evaluated on plain floats via Value."""
    fp = f(x + h)
    fm = f(x - h)
    return (fp - fm) / (2.0 * h)


def test_central_difference_linear():
    # f(a) = a*b + c with b = -3, c = 10  =>  df/da = b = -3
    b, c = -3.0, 10.0
    f = lambda a: (Value(a) * Value(b) + Value(c)).data
    slope = _central_diff(f, 2.0)
    assert slope == pytest.approx(b, abs=1e-4), (
        f"central-difference slope of a*b+c w.r.t. a should be b={b}, got {slope}"
    )


def test_central_difference_quadratic():
    # f(a) = a**2  =>  df/da = 2a ; at a=3 the slope is 6
    f = lambda a: (Value(a) ** 2).data
    slope = _central_diff(f, 3.0)
    assert slope == pytest.approx(6.0, abs=1e-4), (
        f"central-difference slope of a**2 at a=3 should be 6, got {slope}"
    )


def test_central_difference_reciprocal():
    # f(a) = 1/a  =>  df/da = -1/a**2 ; at a=2 the slope is -0.25
    f = lambda a: (1.0 / Value(a)).data
    slope = _central_diff(f, 2.0)
    assert slope == pytest.approx(-0.25, abs=1e-4), (
        f"central-difference slope of 1/a at a=2 should be -0.25, got {slope}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
