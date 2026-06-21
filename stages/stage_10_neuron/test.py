"""Tests for Stage 10: Neuron.

These tests verify the forward shapes of a single neuron and gradient-check its
output with respect to the weights, bias, and input using central differences:

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

compared against the analytical gradients produced by ``Tensor.backward()`` from
stage_09. ``Neuron`` lives in this stage's ``code.py`` and is built on the
``Tensor`` imported from stage_09 through ``dlfs.stage_import``. If stage_09 or
stage_10's ``Neuron`` is not yet implemented, the suite skips rather than
erroring, so you can run it incrementally.

Run with:  pytest stage_10_neuron/test.py
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
# this stage's `Neuron` on top of it.
try:
    from code import Neuron, Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_10 Neuron / stage_09 Tensor not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Return the underlying numpy array of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def make_tensor(arr, requires_grad=True):
    """Build a Tensor from a numpy array, tolerating different ctor kwargs."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=requires_grad)
    except TypeError:
        return Tensor(arr)


def scalar_out(t):
    """Reduce a Tensor's output to a python float for finite-diff probing."""
    a = as_array(t)
    return float(np.sum(a))


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
@pytest.mark.parametrize("activation", ["tanh", "relu", "none"])
def test_param_shapes(activation):
    n = Neuron(4, activation=activation, seed=0)
    assert as_array(n.w).shape == (4,), "weight vector must have shape (n_in,)"
    assert as_array(n.b).size == 1, "bias must be a scalar tensor"
    params = n.parameters()
    assert len(params) == 2, "parameters() must return [w, b]"


def test_single_input_is_scalar():
    n = Neuron(3, activation="tanh", seed=1)
    x = make_tensor([0.5, -1.0, 2.0])
    y = n(x)
    assert as_array(y).size == 1, "single (n_in,) input must yield a scalar output"


def test_batched_input_shape():
    n = Neuron(3, activation="relu", seed=2)
    x = make_tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])
    y = n(x)
    assert as_array(y).shape == (2,), "(batch, n_in) input must yield (batch,) output"


def test_repr():
    n = Neuron(2, activation="tanh", seed=0)
    r = repr(n)
    assert "Neuron" in r and "tanh" in r


def test_reproducible_with_seed():
    a = Neuron(5, seed=123)
    b = Neuron(5, seed=123)
    assert np.allclose(as_array(a.w), as_array(b.w)), "same seed -> same weights"


# --- Activation correctness (forward) ---------------------------------------
def test_relu_clips_negative():
    n = Neuron(2, activation="relu", seed=0)
    # Force a known negative pre-activation by overwriting params.
    n.w.data = np.array([1.0, 1.0])
    n.b.data = np.asarray(n.b.data) * 0.0  # zero bias, keep shape
    x = make_tensor([-3.0, -1.0])  # z = -4 < 0  -> relu -> 0
    assert np.isclose(scalar_out(n(x)), 0.0), "relu(neg) must be 0"


def test_tanh_range():
    n = Neuron(3, activation="tanh", seed=7)
    x = make_tensor([10.0, 10.0, 10.0])
    assert -1.0 <= scalar_out(n(x)) <= 1.0, "tanh output must lie in [-1, 1]"


# --- Gradient checks: output w.r.t. weights ---------------------------------
@pytest.mark.parametrize("activation", ["tanh", "relu", "none"])
def test_gradcheck_wrt_w(activation):
    n = Neuron(4, activation=activation, seed=3)
    x_np = np.array([0.7, -1.3, 0.2, 2.1])

    # Analytical grad via backward.
    n.zero_grad()
    x = make_tensor(x_np, requires_grad=False)
    y = n(x)
    y.backward()
    g_analytic = as_array(n.w.grad)

    # Numerical grad: vary w, keep x fixed.
    w0 = as_array(n.w).copy()

    def f(w):
        n.w.data = w.copy()
        return scalar_out(n(make_tensor(x_np, requires_grad=False)))

    g_num = central_diff(f, w0)
    n.w.data = w0  # restore

    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"[{activation}] dY/dw mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- Gradient check: output w.r.t. bias -------------------------------------
@pytest.mark.parametrize("activation", ["tanh", "relu", "none"])
def test_gradcheck_wrt_b(activation):
    n = Neuron(3, activation=activation, seed=4)
    x_np = np.array([0.4, -0.9, 1.7])

    n.zero_grad()
    y = n(make_tensor(x_np, requires_grad=False))
    y.backward()
    g_analytic = as_array(n.b.grad).reshape(-1)

    b0 = as_array(n.b).copy()

    def f(b):
        n.b.data = b.copy().reshape(np.asarray(n.b.data).shape)
        return scalar_out(n(make_tensor(x_np, requires_grad=False)))

    g_num = central_diff(f, b0.reshape(-1))
    n.b.data = b0  # restore

    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"[{activation}] dY/db mismatch: analytic={g_analytic}, numeric={g_num}"
    )


# --- Gradient check: output w.r.t. input ------------------------------------
@pytest.mark.parametrize("activation", ["tanh", "relu", "none"])
def test_gradcheck_wrt_x(activation):
    n = Neuron(4, activation=activation, seed=5)
    x_np = np.array([0.3, -0.6, 1.2, -2.0])

    n.zero_grad()
    x = make_tensor(x_np, requires_grad=True)
    y = n(x)
    y.backward()
    g_analytic = as_array(x.grad)

    def f(xv):
        return scalar_out(n(make_tensor(xv, requires_grad=False)))

    g_num = central_diff(f, x_np.copy())

    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"[{activation}] dY/dx mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- Gradient check on a batch (sum of outputs) -----------------------------
def test_gradcheck_batch_wrt_w():
    n = Neuron(3, activation="tanh", seed=6)
    X = np.array([[0.5, -1.0, 2.0], [1.0, 0.5, -0.5], [-0.3, 0.8, 1.1]])

    n.zero_grad()
    y = n(make_tensor(X, requires_grad=False))
    y.backward()  # backward on a (batch,) tensor sums contributions
    g_analytic = as_array(n.w.grad)

    w0 = as_array(n.w).copy()

    def f(w):
        n.w.data = w.copy()
        out = n(make_tensor(X, requires_grad=False))
        return scalar_out(out)  # sum over batch

    g_num = central_diff(f, w0)
    n.w.data = w0

    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"batched dY/dw mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- zero_grad clears accumulated gradients ---------------------------------
def test_zero_grad():
    n = Neuron(3, activation="tanh", seed=8)
    y = n(make_tensor([1.0, 2.0, 3.0], requires_grad=False))
    y.backward()
    assert np.any(as_array(n.w.grad) != 0.0), "grad should be populated after backward"
    n.zero_grad()
    assert np.allclose(as_array(n.w.grad), 0.0), "zero_grad must clear w.grad"
    assert np.allclose(as_array(n.b.grad), 0.0), "zero_grad must clear b.grad"
