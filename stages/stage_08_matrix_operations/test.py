"""Stage 8 tests: Mat operations + central-difference gradient checks.

Run: pytest stage_08_matrix_operations/test.py

These tests assume a finished `Value` (stage_06/07) and the `Mat` class in
code.py. A `Mat` is a 2-D grid of `Value`s; gradients flow through the scalar
engine, so we reduce to a scalar `Value` (`sum`/`mean`) and call `.backward()`.
Gradients are verified against numerical central differences:
    df/dx ~= (f(x+eps) - f(x-eps)) / (2*eps).
"""

import os
import random
import sys

import numpy as np
import pytest

# Ensure this stage's directory is importable so `from code import ...` works
# regardless of how pytest is launched. `code.py` itself pulls `Value` (stage_06)
# and `Vec` (stage_07) in via dlfs.stage_import and builds `Mat` on top of them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from code import Mat, Value


EPS = 1e-6
TOL = 1e-5
random.seed(8)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def rand_grid(rows, cols):
    """A plain list-of-lists of floats."""
    return [[random.uniform(-2, 2) for _ in range(cols)] for _ in range(rows)]


def flatten(grid):
    return [x for row in grid for x in row]


def grads_of(mat):
    """Extract .grad of every element as a list-of-lists of floats."""
    return [[mat[i][j].grad for j in range(mat.cols)] for i in range(mat.rows)]


def np_of(grid):
    return np.array(grid, dtype=np.float64)


def numeric_grad_grid(f, grid):
    """Central-difference gradient of scalar `f(grid_floats)` w.r.t. each cell.

    `f` takes a list-of-lists of floats and returns a Python float.
    Returns a list-of-lists matching `grid`.
    """
    rows, cols = len(grid), len(grid[0])
    g = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            up = [r[:] for r in grid]
            dn = [r[:] for r in grid]
            up[i][j] += EPS
            dn[i][j] -= EPS
            g[i][j] = (f(up) - f(dn)) / (2 * EPS)
    return g


def assert_grid_close(actual, expected, name):
    a, e = np_of(actual), np_of(expected)
    assert a.shape == e.shape, f"{name}: shape mismatch {a.shape} vs {e.shape}"
    diff = float(np.max(np.abs(a - e)))
    assert diff < TOL, f"{name}: max abs diff {diff:.3e} exceeds tol {TOL:.1e}"


# --------------------------------------------------------------------------- #
# container basics                                                            #
# --------------------------------------------------------------------------- #
def test_construction_and_shape():
    m = Mat([[1.0, 2.0, 3.0], [4.0, 5.0, Value(6.0)]])
    assert m.shape == (2, 3), f"expected (2,3), got {m.shape}"
    assert isinstance(m[0][0], Value)
    assert m[1][2].data == 6.0


def test_existing_value_not_rewrapped():
    a = Value(7.0)
    m = Mat([[a, 1.0]])
    assert m[0][0] is a, "an entry already a Value must be kept, not re-wrapped"


def test_ragged_rows_raise():
    with pytest.raises(ValueError):
        Mat([[1.0, 2.0], [3.0]])


# --------------------------------------------------------------------------- #
# matmul: forward                                                             #
# --------------------------------------------------------------------------- #
def test_matmul_forward_matches_numpy():
    a, b = rand_grid(3, 4), rand_grid(4, 5)
    c = Mat(a) @ Mat(b)
    assert c.shape == (3, 5), f"expected (3,5), got {c.shape}"
    cdata = [[c[i][j].data for j in range(5)] for i in range(3)]
    assert_grid_close(cdata, (np_of(a) @ np_of(b)).tolist(), "matmul forward")


def test_matmul_inner_dim_mismatch_raises():
    with pytest.raises(ValueError):
        Mat(rand_grid(3, 4)) @ Mat(rand_grid(5, 6))


# --------------------------------------------------------------------------- #
# matmul: the two key gradient formulas                                       #
# --------------------------------------------------------------------------- #
def test_matmul_grad_formulas_exact():
    """With L = sum(A@B), G = ones(m,n), so dL/dA = ones@B.T and dL/dB = A.T@ones."""
    a, b = rand_grid(3, 4), rand_grid(4, 5)
    A, B = Mat(a), Mat(b)
    (A @ B).sum().backward()

    G = np.ones((3, 5))
    assert_grid_close(grads_of(A), (G @ np_of(b).T).tolist(), "dL/dA = G@B.T")
    assert_grid_close(grads_of(B), (np_of(a).T @ G).tolist(), "dL/dB = A.T@G")


def test_matmul_gradcheck():
    """Asymmetric shapes (4x3, 3x2) so a swapped A.T/B.T would be caught."""
    a, b = rand_grid(4, 3), rand_grid(3, 2)

    def loss_a(grid):
        return float((np_of(grid) @ np_of(b)).sum())

    def loss_b(grid):
        return float((np_of(a) @ np_of(grid)).sum())

    A, B = Mat(a), Mat(b)
    (A @ B).sum().backward()

    assert_grid_close(grads_of(A), numeric_grad_grid(loss_a, a), "matmul gradcheck A")
    assert_grid_close(grads_of(B), numeric_grad_grid(loss_b, b), "matmul gradcheck B")


# --------------------------------------------------------------------------- #
# transpose                                                                   #
# --------------------------------------------------------------------------- #
def test_transpose_forward_and_T_property():
    a = rand_grid(2, 5)
    m = Mat(a)
    t = m.transpose()
    assert t.shape == (5, 2)
    tdata = [[t[i][j].data for j in range(2)] for i in range(5)]
    assert_grid_close(tdata, np_of(a).T.tolist(), "transpose forward")
    tp = m.T
    tpdata = [[tp[i][j].data for j in range(2)] for i in range(5)]
    assert_grid_close(tpdata, np_of(a).T.tolist(), ".T property forward")


def test_transpose_gradcheck():
    """L = sum((A.T) @ M) routes grad through transpose then matmul."""
    a = rand_grid(3, 4)
    mm = rand_grid(3, 2)  # multiplies A.T which is (4,3)

    def loss(grid):
        return float((np_of(grid).T @ np_of(mm)).sum())

    A = Mat(a)
    (A.T @ Mat(mm)).sum().backward()
    assert_grid_close(grads_of(A), numeric_grad_grid(loss, a), "transpose gradcheck")


# --------------------------------------------------------------------------- #
# reshape                                                                     #
# --------------------------------------------------------------------------- #
def test_reshape_forward():
    a = rand_grid(2, 6)
    r = Mat(a).reshape(3, 4)
    assert r.shape == (3, 4)
    rdata = [[r[i][j].data for j in range(4)] for i in range(3)]
    assert_grid_close(rdata, np_of(a).reshape(3, 4).tolist(), "reshape forward")


def test_reshape_bad_size_raises():
    with pytest.raises(ValueError):
        Mat(rand_grid(2, 6)).reshape(3, 5)


def test_reshape_gradcheck():
    a = rand_grid(2, 6)

    def loss(grid):
        return float(np_of(grid).reshape(3, 4).sum())

    A = Mat(a)
    A.reshape(3, 4).sum().backward()
    assert_grid_close(grads_of(A), numeric_grad_grid(loss, a), "reshape gradcheck")
    # pure reshape+sum -> gradient is all ones
    assert_grid_close(grads_of(A), [[1.0] * 6 for _ in range(2)], "reshape grad is ones")


# --------------------------------------------------------------------------- #
# sum                                                                         #
# --------------------------------------------------------------------------- #
def test_sum_forward_and_grad():
    a = rand_grid(3, 4)
    A = Mat(a)
    s = A.sum()
    assert abs(s.data - float(np_of(a).sum())) < TOL, "sum forward"
    s.backward()
    assert_grid_close(grads_of(A), [[1.0] * 4 for _ in range(3)], "sum grad is ones")


def test_sum_gradcheck():
    a = rand_grid(3, 4)

    def loss(grid):
        return float(np_of(grid).sum())

    A = Mat(a)
    A.sum().backward()
    assert_grid_close(grads_of(A), numeric_grad_grid(loss, a), "sum gradcheck")


# --------------------------------------------------------------------------- #
# mean                                                                        #
# --------------------------------------------------------------------------- #
def test_mean_forward_and_grad():
    a = rand_grid(3, 4)
    A = Mat(a)
    m = A.mean()
    assert abs(m.data - float(np_of(a).mean())) < TOL, "mean forward"
    m.backward()
    n = 3 * 4
    assert_grid_close(grads_of(A), [[1.0 / n] * 4 for _ in range(3)], "mean grad is 1/N")


def test_mean_gradcheck():
    a = rand_grid(3, 4)

    def loss(grid):
        return float(np_of(grid).mean())

    A = Mat(a)
    A.mean().backward()
    assert_grid_close(grads_of(A), numeric_grad_grid(loss, a), "mean gradcheck")


# --------------------------------------------------------------------------- #
# composition: a linear-layer-ish chain exercises matmul+mean together        #
# --------------------------------------------------------------------------- #
def test_chained_matmul_mean_gradcheck():
    x, w = rand_grid(5, 3), rand_grid(3, 2)

    def loss_x(grid):
        return float((np_of(grid) @ np_of(w)).mean())

    def loss_w(grid):
        return float((np_of(x) @ np_of(grid)).mean())

    X, W = Mat(x), Mat(w)
    (X @ W).mean().backward()
    assert_grid_close(grads_of(X), numeric_grad_grid(loss_x, x), "chain gradcheck dL/dx")
    assert_grid_close(grads_of(W), numeric_grad_grid(loss_w, w), "chain gradcheck dL/dw")


def test_reused_matrix_accumulates():
    """A square matrix used as BOTH factors must SUM its gradients (Value uses +=)."""
    a = rand_grid(2, 2)
    A = Mat(a)
    (A @ A).sum().backward()  # L = sum(A @ A)
    G = np.ones((2, 2))
    an = np_of(a)
    expected = (G @ an.T + an.T @ G).tolist()  # left-factor grad + right-factor grad
    assert_grid_close(grads_of(A), expected, "reused-matrix grad accumulation")
