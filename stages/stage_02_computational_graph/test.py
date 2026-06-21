"""Tests for stage 2: Computational graph.

Validates that `Value` records its operation graph (`_prev`, `_op`) while
forward arithmetic stays identical to stage_01.

This stage does NOT implement a backward pass, so there are no analytical
`grad` fields to gradient-check yet. To still honor central-difference
gradient checking "where gradients exist", we treat the *forward* DAG as a
differentiable function of its leaf inputs and verify the local derivatives
the README derives ( d(a+b)/da = 1, d(a*b)/da = b ) numerically, by
re-running the forward graph with each leaf perturbed by +/-eps. When the
real backward pass arrives in a later stage, the SAME central-difference
recipe will be reused against analytical `.grad` values.
"""

import importlib.util
import os
import sys

import pytest

# This stage's ``code.py`` extends stage_01's ``Value`` via ``dlfs.stage_import``,
# so the curriculum root (which holds the ``dlfs`` package) must be importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import this stage's ``code.py`` by file path. A plain ``from code import ...``
# would collide with Python's standard-library ``code`` module (and pytest's
# debugger imports it), so we load the sibling file explicitly. This works no
# matter what directory pytest is invoked from. The extended ``Value`` (subclass
# of stage_01.Value) is what these tests exercise.
_spec = importlib.util.spec_from_file_location(
    "stage_02_code", os.path.join(_THIS_DIR, "code.py")
)
_code = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_code)
Value = _code.Value
trace = _code.trace


# --------------------------------------------------------------------------
# central-difference helper (reused verbatim by later stages vs analytical grad)
# --------------------------------------------------------------------------
def numgrad(f, x, eps=1e-6):
    """Central-difference derivative of scalar f at scalar x.

    (f(x + eps) - f(x - eps)) / (2 * eps)
    """
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


# --------------------------------------------------------------------------
# forward arithmetic must be unchanged from stage_01
# --------------------------------------------------------------------------
def test_leaf_data_is_float():
    a = Value(3)
    assert isinstance(a.data, float), "data must be stored as a float"
    assert a.data == 3.0


def test_add_forward():
    a, b = Value(2.0), Value(5.0)
    assert (a + b).data == 7.0, "addition forward value changed from stage_01"


def test_mul_forward():
    a, b = Value(2.0), Value(5.0)
    assert (a * b).data == 10.0, "multiplication forward value changed"


def test_compound_forward():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    out = a * b + c
    assert out.data == 4.0, "(a*b + c) forward value incorrect"


def test_number_coercion_and_reflected_ops():
    a = Value(4.0)
    assert (a + 1).data == 5.0, "Value + number failed"
    assert (1 + a).data == 5.0, "number + Value (__radd__) failed"
    assert (a * 3).data == 12.0, "Value * number failed"
    assert (3 * a).data == 12.0, "number * Value (__rmul__) failed"


# --------------------------------------------------------------------------
# graph bookkeeping: _prev and _op
# --------------------------------------------------------------------------
def test_leaf_has_empty_graph():
    a = Value(1.0)
    assert a._prev == set(), "a leaf must have no parents"
    assert a._op == "", "a leaf must have empty _op"
    assert a.grad == 0.0, "grad must default to 0.0"


def test_add_records_parents_and_op():
    a, b = Value(2.0), Value(3.0)
    out = a + b
    assert out._op == "+", "add result must have _op == '+'"
    assert out._prev == {a, b}, "add result must record both operands as parents"


def test_mul_records_parents_and_op():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    assert out._op == "*", "mul result must have _op == '*'"
    assert out._prev == {a, b}, "mul result must record both operands as parents"


def test_coerced_operand_becomes_value_parent():
    a = Value(2.0)
    out = a + 1
    assert len(out._prev) == 2, "coerced number must appear as a Value parent"
    assert a in out._prev, "original Value must be a parent"
    parents = list(out._prev)
    other = parents[0] if parents[1] is a else parents[1]
    assert isinstance(other, Value), "coerced operand must be wrapped in Value"
    assert other.data == 1.0, "coerced operand must carry the number's data"


def test_self_reuse_dedups_in_prev():
    a = Value(3.0)
    out = a * a
    assert out.data == 9.0, "a*a forward value incorrect"
    assert out._prev == {a}, "a*a must store a single parent (set dedup)"
    assert len(out._prev) == 1


# --------------------------------------------------------------------------
# trace(): full DAG enumeration without duplicates or infinite loops
# --------------------------------------------------------------------------
def test_trace_simple_graph():
    a, b = Value(2.0), Value(3.0)
    out = a + b
    nodes, edges = trace(out)
    assert nodes == {a, b, out}, "trace must return every reachable node"
    assert edges == {(a, out), (b, out)}, "trace must return each parent->child edge"


def test_trace_compound_graph():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    e = a * b          # node e
    out = e + c        # node out
    nodes, edges = trace(out)
    assert nodes == {a, b, c, e, out}
    assert edges == {(a, e), (b, e), (e, out), (c, out)}


def test_trace_terminates_on_reused_node():
    a = Value(3.0)
    out = a * a
    nodes, edges = trace(out)  # must not loop / duplicate
    assert nodes == {a, out}
    assert edges == {(a, out)}


def test_trace_leaf():
    a = Value(7.0)
    nodes, edges = trace(a)
    assert nodes == {a}
    assert edges == set()


# --------------------------------------------------------------------------
# central-difference check of the README's local derivatives via the DAG
# --------------------------------------------------------------------------
def test_numgrad_addition_local_derivative():
    # f(a) = a + b ; d f / d a should be 1.0
    b = 3.0

    def f(a):
        return (Value(a) + Value(b)).data

    g = numgrad(f, 2.0)
    assert g == pytest.approx(1.0, abs=1e-4), (
        f"d(a+b)/da should be 1 (chain-rule local grad); numgrad gave {g}"
    )


def test_numgrad_multiplication_local_derivative():
    # f(a) = a * b ; d f / d a should be b
    b = 4.0

    def f(a):
        return (Value(a) * Value(b)).data

    g = numgrad(f, 5.0)
    assert g == pytest.approx(b, abs=1e-4), (
        f"d(a*b)/da should equal b={b}; numgrad gave {g}"
    )


def test_numgrad_compound_local_derivative():
    # f(a) = a*b + c ; d f / d a should be b
    b, c = -3.0, 10.0

    def f(a):
        return (Value(a) * Value(b) + Value(c)).data

    g = numgrad(f, 2.0)
    assert g == pytest.approx(b, abs=1e-4), (
        f"d(a*b+c)/da should equal b={b}; numgrad gave {g}"
    )


def test_repr_mentions_data_and_op():
    a = Value(2.0)
    out = a + Value(3.0)
    assert "data=" in repr(out)
    assert "op=" in repr(out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
