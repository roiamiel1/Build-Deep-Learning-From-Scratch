"""Tests for Stage 11: Dense / Linear layer.

These tests verify the forward shapes of a fully-connected layer (single-vector
and batched, with and without bias) and gradient-check a scalar reduction of its
output with respect to the weights, bias, and input using central differences:

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

compared against the analytical gradients produced by ``Tensor.backward()`` from
stage_09. ``Dense`` lives in this stage's ``code.py`` and is built on the
``Tensor`` imported from stage_09 through ``dlfs.stage_import``. If stage_09's
``Tensor`` or stage_11's ``Dense`` is not yet implemented, the suite skips
rather than erroring, so you can run it incrementally.

Run with:  pytest stage_11_dense_layer/test.py
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# Make the shared `dlfs` shim importable (it lives at the curriculum root).
sys.path.insert(0, _ROOT)

# --- Import the things under test, skipping cleanly if not ready yet. --------
# `code.py` re-exports `Tensor` (from stage_09 via dlfs.stage_import) and defines
# this stage's `Dense` on top of it.
try:
    from code import Dense, Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_11 Dense / stage_09 Tensor not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Return the underlying numpy array of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def make_tensor(arr):
    """Build a Tensor from a numpy array, tolerating different ctor kwargs."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr)
    except TypeError:  # pragma: no cover - defensive for alt signatures
        return Tensor(arr)


def scalar_out(t):
    """Reduce a Tensor's output to a python float for finite-diff probing."""
    return float(np.sum(as_array(t)))


def central_diff(f, x, eps=EPS):
    """Numerical gradient of scalar-valued f at numpy point x (any shape)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


# --- Construction & forward shapes ------------------------------------------
def test_param_shapes_with_bias():
    layer = Dense(4, 3, bias=True, seed=0)
    assert as_array(layer.W).shape == (4, 3), "W must have shape (n_in, n_out)"
    assert as_array(layer.b).shape == (3,), "b must have shape (n_out,)"
    params = layer.parameters()
    assert len(params) == 2, "parameters() must return [W, b] when bias=True"


def test_param_shapes_no_bias():
    layer = Dense(4, 3, bias=False, seed=0)
    assert as_array(layer.W).shape == (4, 3), "W must have shape (n_in, n_out)"
    assert layer.b is None, "b must be None when bias=False"
    params = layer.parameters()
    assert len(params) == 1, "parameters() must return [W] when bias=False"


def test_single_input_shape():
    layer = Dense(3, 5, seed=1)
    x = make_tensor([0.5, -1.0, 2.0])
    y = layer(x)
    assert as_array(y).shape == (5,), "single (n_in,) input must yield (n_out,) output"


def test_batched_input_shape():
    layer = Dense(3, 4, seed=2)
    x = make_tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])
    y = layer(x)
    assert as_array(y).shape == (2, 4), "(B, n_in) input must yield (B, n_out) output"


def test_repr():
    layer = Dense(2, 3, bias=True, seed=0)
    r = repr(layer)
    assert "Dense" in r and "2" in r and "3" in r


def test_reproducible_with_seed():
    a = Dense(5, 6, seed=123)
    b = Dense(5, 6, seed=123)
    assert np.allclose(as_array(a.W), as_array(b.W)), "same seed -> same weights"


# --- Forward correctness (matches plain numpy affine map) -------------------
def test_forward_matches_numpy():
    layer = Dense(3, 2, bias=True, seed=4)
    X = np.array([[0.5, -1.0, 2.0], [1.0, 0.5, -0.5]])
    expected = X @ as_array(layer.W) + as_array(layer.b)  # (2, 2)
    y = layer(make_tensor(X))
    assert np.allclose(as_array(y), expected, atol=ATOL), (
        f"forward mismatch:\n got     ={as_array(y)}\n expected={expected}"
    )


def test_forward_no_bias_matches_numpy():
    layer = Dense(3, 2, bias=False, seed=4)
    X = np.array([[0.5, -1.0, 2.0], [1.0, 0.5, -0.5]])
    expected = X @ as_array(layer.W)
    y = layer(make_tensor(X))
    assert np.allclose(as_array(y), expected, atol=ATOL), "unbiased forward mismatch"


# --- Gradient check: output w.r.t. weights ----------------------------------
@pytest.mark.parametrize("bias", [True, False])
def test_gradcheck_wrt_W(bias):
    layer = Dense(4, 3, bias=bias, seed=3)
    X = np.array([[0.7, -1.3, 0.2, 2.1], [0.1, 0.9, -0.4, 1.0]])

    # Analytical grad via backward on sum(Z).
    layer.zero_grad()
    Z = layer(make_tensor(X))
    Z.backward()
    g_analytic = as_array(layer.W.grad)

    W0 = as_array(layer.W).copy()

    def f(W):
        layer.W.data = W.copy()
        return scalar_out(layer(make_tensor(X)))

    g_num = central_diff(f, W0)
    layer.W.data = W0  # restore

    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"[bias={bias}] dL/dW mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- Gradient check: output w.r.t. bias (sums over the batch) ----------------
def test_gradcheck_wrt_b_batched():
    layer = Dense(3, 4, bias=True, seed=5)
    X = np.array(
        [[0.4, -0.9, 1.7], [1.0, 0.5, -0.5], [-0.3, 0.8, 1.1]]
    )  # B = 3

    layer.zero_grad()
    Z = layer(make_tensor(X))
    Z.backward()
    g_analytic = as_array(layer.b.grad).reshape(-1)

    b0 = as_array(layer.b).copy()

    def f(b):
        layer.b.data = b.copy().reshape(np.asarray(layer.b.data).shape)
        return scalar_out(layer(make_tensor(X)))

    g_num = central_diff(f, b0.reshape(-1))
    layer.b.data = b0  # restore

    # With sum(Z) as the loss, each bias entry receives +1 from every example.
    assert np.allclose(g_analytic, np.full_like(g_analytic, X.shape[0]), atol=ATOL), (
        f"dL/db should equal batch size on each entry, got {g_analytic}"
    )
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"dL/db mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- Gradient check: output w.r.t. input ------------------------------------
@pytest.mark.parametrize("bias", [True, False])
def test_gradcheck_wrt_X(bias):
    layer = Dense(4, 3, bias=bias, seed=6)
    X0 = np.array([[0.3, -0.6, 1.2, -2.0], [0.5, 1.1, -0.7, 0.2]])

    layer.zero_grad()
    x = make_tensor(X0)
    Z = layer(x)
    Z.backward()
    g_analytic = as_array(x.grad)

    def f(Xv):
        return scalar_out(layer(make_tensor(Xv)))

    g_num = central_diff(f, X0.copy())

    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"[bias={bias}] dL/dX mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- Gradient check: single (1-D) input -------------------------------------
def test_gradcheck_single_input_wrt_W():
    layer = Dense(3, 2, bias=True, seed=7)
    x_np = np.array([0.5, -1.0, 2.0])

    layer.zero_grad()
    Z = layer(make_tensor(x_np))
    Z.backward()
    g_analytic = as_array(layer.W.grad)

    W0 = as_array(layer.W).copy()

    def f(W):
        layer.W.data = W.copy()
        return scalar_out(layer(make_tensor(x_np)))

    g_num = central_diff(f, W0)
    layer.W.data = W0

    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"single-input dL/dW mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- zero_grad clears accumulated gradients ---------------------------------
def test_zero_grad():
    layer = Dense(3, 2, bias=True, seed=8)
    Z = layer(make_tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]]))
    Z.backward()
    assert np.any(as_array(layer.W.grad) != 0.0), "W.grad should populate after backward"
    layer.zero_grad()
    assert np.allclose(as_array(layer.W.grad), 0.0), "zero_grad must clear W.grad"
    assert np.allclose(as_array(layer.b.grad), 0.0), "zero_grad must clear b.grad"


# --- Layer equals a stack of independent neurons (column-wise) --------------
def test_columns_are_independent():
    """Output column j depends only on W[:, j] (and b[j]); a sanity check that
    the layer is n_out parallel affine maps, not something entangled."""
    layer = Dense(2, 3, bias=False, seed=9)
    X = np.array([[1.0, -2.0]])
    Z = as_array(layer(make_tensor(X)))  # (1, 3)
    W = as_array(layer.W)
    for j in range(3):
        assert np.isclose(Z[0, j], X[0] @ W[:, j], atol=ATOL), (
            f"output column {j} must equal X @ W[:, {j}]"
        )
