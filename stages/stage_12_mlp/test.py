"""Tests for Stage 12: MLP.

These tests verify the forward shapes of a multilayer perceptron, that it
gathers all of its layers' parameters, and they gradient-check a scalar loss
(the sum of the network output) with respect to every layer's weights and bias
and with respect to the input, using central differences::

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

compared against the analytical gradients produced by ``Tensor.backward()`` from
stage_09 (propagated through the ``Dense`` layers from stage_11). ``MLP`` lives in
this stage's ``code.py`` and is built on the ``Dense`` (stage_11) and ``Tensor``
(stage_09) classes imported through ``dlfs.stage_import``. If any earlier stage
is not yet implemented, the suite skips rather than erroring, so you can run it
incrementally.

Run with:  pytest stage_12_mlp/test.py
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
# `code.py` binds `Stage9_Tensor` (stage_09) and `Stage11_Dense` (stage_11) via
# dlfs.stage_import and defines this stage's `MLP` on top of them.
try:
    from code import MLP, Stage11_Dense, Stage9_Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_12 MLP / stage_11 Dense / stage_09 Tensor not importable yet: {exc}",
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
        return Stage9_Tensor(arr, requires_grad=requires_grad)
    except TypeError:
        return Stage9_Tensor(arr)


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


def build_net(sizes, activation="tanh", out_activation="none", seed=0):
    return MLP(sizes, activation=activation, out_activation=out_activation, seed=seed)


# --- Construction & structure ------------------------------------------------
def test_layer_count_and_widths():
    net = build_net([3, 5, 4, 2], seed=0)
    assert len(net.layers) == 3, "an MLP over [3,5,4,2] must have 3 Dense layers"


def test_every_layer_is_a_dense():
    net = build_net([4, 8, 1], seed=0)
    for layer in net.layers:
        assert isinstance(layer, Stage11_Dense), "each MLP layer must be a stage_11 Dense"


def test_parameters_are_collected_from_all_layers():
    net = build_net([3, 6, 6, 2], seed=1)
    params = net.parameters()
    # 3 Dense layers, 2 params (weight + bias) each -> 6 parameter tensors.
    assert len(params) == 6, "parameters() must flatten all layers' params (2 per layer)"
    expected = [p for layer in net.layers for p in layer.parameters()]
    assert len(params) == len(expected)
    # Identity check: the same tensor objects, not copies.
    for p, q in zip(params, expected):
        assert p is q, "parameters() must return the actual layer parameter tensors"


def test_repr_mentions_sizes_and_activation():
    net = build_net([2, 16, 1], activation="tanh", out_activation="none", seed=0)
    r = repr(net)
    assert "MLP" in r and "tanh" in r


def test_reproducible_with_seed():
    a = build_net([3, 7, 2], seed=123)
    b = build_net([3, 7, 2], seed=123)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert np.allclose(as_array(pa), as_array(pb)), "same seed -> same weights"


def test_distinct_layers_have_distinct_weights():
    # Per-layer seed derivation should make the two weight matrices differ
    # (they also have different shapes here, but check the values are not a
    # trivially-shared object).
    net = build_net([4, 4, 4], seed=0)
    w0 = as_array(net.layers[0].parameters()[0])
    w1 = as_array(net.layers[1].parameters()[0])
    assert not np.allclose(w0, w1), "different layers should not share identical weights"


# --- Forward shapes ----------------------------------------------------------
def test_single_input_shape():
    net = build_net([3, 5, 2], seed=2)
    y = net(make_tensor([0.5, -1.0, 2.0], requires_grad=False))
    assert as_array(y).shape == (2,), "(n_in,) input must yield (n_out,) output"


def test_batched_input_shape():
    net = build_net([3, 5, 2], seed=2)
    X = make_tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]], requires_grad=False)
    y = net(X)
    assert as_array(y).shape == (2, 2), "(batch, n_in) input must yield (batch, n_out)"


def test_call_is_forward():
    net = build_net([2, 4, 3], seed=5)
    x = make_tensor([0.3, -0.7], requires_grad=False)
    y1 = net(x)
    y2 = net.forward(make_tensor([0.3, -0.7], requires_grad=False))
    assert np.allclose(as_array(y1), as_array(y2)), "__call__ must equal forward()"


# --- Depth + nonlinearity actually matters -----------------------------------
def test_hidden_nonlinearity_is_not_affine():
    """A 1->H->1 MLP with a tanh hidden layer must NOT be an affine function of x.

    For an affine map g, the second difference g(x+h) - 2 g(x) + g(x-h) is zero.
    A genuine nonlinearity makes it nonzero.
    """
    net = build_net([1, 8, 1], activation="tanh", out_activation="none", seed=11)

    def g(xv):
        return scalar_out(net(make_tensor(np.array([xv]), requires_grad=False)))

    h = 0.5
    x0 = 0.3
    second_diff = g(x0 + h) - 2.0 * g(x0) + g(x0 - h)
    assert abs(second_diff) > 1e-6, (
        "MLP with a hidden nonlinearity must not collapse to an affine map"
    )


# --- Gradient checks: scalar loss w.r.t. every parameter ---------------------
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_gradcheck_wrt_all_params(activation):
    sizes = [3, 5, 4, 2]
    net = build_net(sizes, activation=activation, out_activation="none", seed=7)
    x_np = np.array([0.7, -1.3, 0.2])

    # Analytical gradients via one backward pass on sum(output).
    net.zero_grad()
    out = net(make_tensor(x_np, requires_grad=False))
    loss = out.sum() if hasattr(out, "sum") else out
    # If Tensor lacks .sum(), backward() on the (n_out,) tensor seeds ones,
    # which equals d(sum(out)). Either path matches our scalar_out finite diff.
    loss.backward()

    params = net.parameters()
    analytic = [as_array(p.grad).copy() for p in params]

    saved = [as_array(p).copy() for p in params]

    def f_factory(k):
        def f(pv):
            params[k].data = pv.copy().reshape(saved[k].shape)
            val = scalar_out(net(make_tensor(x_np, requires_grad=False)))
            params[k].data = saved[k].copy()
            return val
        return f

    for k, p in enumerate(params):
        g_num = central_diff(f_factory(k), saved[k].copy())
        assert np.allclose(analytic[k], g_num, atol=ATOL, rtol=RTOL), (
            f"[{activation}] dLoss/dparam[{k}] (shape {saved[k].shape}) mismatch:\n"
            f" analytic=\n{analytic[k]}\n numeric =\n{g_num}"
        )


# --- Gradient check: scalar loss w.r.t. the input ----------------------------
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_gradcheck_wrt_input(activation):
    net = build_net([4, 6, 3], activation=activation, out_activation="none", seed=9)
    x_np = np.array([0.3, -0.6, 1.2, -2.0])

    net.zero_grad()
    x = make_tensor(x_np, requires_grad=True)
    out = net(x)
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    g_analytic = as_array(x.grad)

    def f(xv):
        return scalar_out(net(make_tensor(xv, requires_grad=False)))

    g_num = central_diff(f, x_np.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"[{activation}] dLoss/dx mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- Gradient check over a batch (sum of all outputs) ------------------------
def test_gradcheck_batch_wrt_first_layer():
    net = build_net([3, 5, 2], activation="tanh", out_activation="none", seed=6)
    X = np.array([[0.5, -1.0, 2.0], [1.0, 0.5, -0.5], [-0.3, 0.8, 1.1]])

    net.zero_grad()
    out = net(make_tensor(X, requires_grad=False))
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    W1 = net.layers[0].parameters()[0]
    g_analytic = as_array(W1.grad).copy()

    saved = as_array(W1).copy()

    def f(w):
        W1.data = w.copy().reshape(saved.shape)
        val = scalar_out(net(make_tensor(X, requires_grad=False)))
        W1.data = saved.copy()
        return val

    g_num = central_diff(f, saved.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"batched dLoss/dW1 mismatch:\n analytic=\n{g_analytic}\n numeric =\n{g_num}"
    )


# --- zero_grad clears accumulated gradients across all layers -----------------
def test_zero_grad_clears_every_param():
    net = build_net([3, 4, 2], activation="tanh", seed=8)
    out = net(make_tensor([1.0, 2.0, 3.0], requires_grad=False))
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    assert any(np.any(as_array(p.grad) != 0.0) for p in net.parameters()), (
        "some gradient should be populated after backward"
    )
    net.zero_grad()
    for p in net.parameters():
        assert np.allclose(as_array(p.grad), 0.0), "zero_grad must clear every param grad"
