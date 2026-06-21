"""Tests for Stage 15: First training loop.

Covers the full loop and each piece feeding it:
  * datasets (make_moons / make_spiral) -- shapes, label set, determinism;
  * MLP -- forward shapes, parameter collection, non-linearity present;
  * mse_loss -- correct scalar value and a central-difference gradcheck of the
    loss w.r.t. a parameter Tensor against the autodiff `.grad`;
  * SGD -- in-place update direction and that step does not extend the graph;
  * zero_grad -- grads reset to zeros, and that WITHOUT a zero in between two
    `.backward()` calls provably DOUBLE the grad (the accumulation bug the loop
    must avoid);
  * train -- loss history length, loss decreases, and >= 90% accuracy on a
    noiseless moons set.

Run: pytest stage_15_first_training_loop/test.py
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
# `code.py` defines this stage's `train` / `make_moons` / `make_spiral` /
# `accuracy` and re-exports `MLP` (stage_12), `mse_loss` (stage_13), `SGD`
# (stage_14), `Tensor` (stage_09), and `Dense`, all via dlfs.stage_import.
try:
    from code import (
        MLP,
        SGD,
        accuracy,
        make_moons,
        make_spiral,
        mse_loss,
        train,
        Tensor,
        Dense,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_15 (or a dependency stage_09/12/13/14) not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
TOL = 1e-6


# --------------------------------------------------------------------------- #
# helper: skip a test cleanly if a *dependency* (stage_09..14 / Dense) isn't
# implemented yet, while still failing if THIS stage is the problem.
# --------------------------------------------------------------------------- #
def _requires(fn, *args, **kwargs):
    """Call fn; if a NotImplementedError bubbles up from a dependency, skip."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"depends on unimplemented piece: {e}")


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def test_make_moons_shapes_and_labels():
    X, y = _requires(make_moons, n=200, noise=0.0, seed=0)
    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert set(np.unique(y)).issubset({-1.0, 1.0})
    assert len(np.unique(y)) == 2
    assert X.dtype == np.float64


def test_make_moons_deterministic():
    X1, y1 = _requires(make_moons, n=120, noise=0.1, seed=7)
    X2, y2 = _requires(make_moons, n=120, noise=0.1, seed=7)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def test_make_moons_noise_changes_points():
    Xa, _ = _requires(make_moons, n=120, noise=0.0, seed=3)
    Xb, _ = _requires(make_moons, n=120, noise=0.3, seed=3)
    assert not np.allclose(Xa, Xb)


def test_make_spiral_shapes_and_labels():
    X, y = _requires(make_spiral, n_per_class=80, n_classes=2, noise=0.1, seed=1)
    assert X.shape == (160, 2)
    assert y.shape == (160,)
    assert set(np.unique(y)).issubset({-1.0, 1.0})
    assert len(np.unique(y)) == 2


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
def test_mlp_forward_shape():
    model = _requires(MLP, [2, 8, 1], activation="tanh", seed=0)
    X = np.random.default_rng(0).standard_normal((16, 2))
    out = _requires(model.__call__, X)
    assert isinstance(out, Tensor)
    assert out.shape == (16, 1)


def test_mlp_parameters_collected():
    model = _requires(MLP, [2, 8, 4, 1], activation="tanh", seed=0)
    params = _requires(model.parameters)
    # 3 Dense layers, each contributes W (and b unless disabled) -> >= 3 tensors.
    assert all(isinstance(p, Tensor) for p in params)
    assert len(params) >= 3
    # parameters of a fresh Dense should match what MLP exposes for one layer.
    one = _requires(Dense, 2, 8, seed=0)
    assert len(one.parameters()) <= len(params)


def test_mlp_is_nonlinear():
    """A 2-layer MLP with activation must NOT be an affine map of the input
    (otherwise it could never separate moons/spirals)."""
    model = _requires(MLP, [2, 16, 1], activation="tanh", seed=2)
    f = lambda v: float(_requires(model.__call__, np.array([v])).data.reshape(-1)[0])
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    mid = (a + b) / 2.0
    # affine => f(mid) == (f(a)+f(b))/2 exactly; non-linear breaks this.
    fa, fb, fm = f(a), f(b), f(mid)
    assert not np.isclose(fm, 0.5 * (fa + fb), atol=1e-6), (
        "MLP behaves like an affine map; activation is missing between layers"
    )


# --------------------------------------------------------------------------- #
# mse_loss
# --------------------------------------------------------------------------- #
def test_mse_loss_value():
    pred = Tensor([[1.0], [2.0], [3.0]])
    target = np.array([[0.0], [0.0], [0.0]])
    loss = _requires(mse_loss, pred, target)
    assert loss.shape == ()
    # mean of [1, 4, 9] = 14/3
    assert np.isclose(float(loss.data), 14.0 / 3.0)


def test_mse_loss_is_scalar_and_backprops():
    pred = Tensor([[0.5], [-0.5]])
    loss = _requires(mse_loss, pred, np.array([[1.0], [-1.0]]))
    assert loss.shape == ()
    _requires(loss.backward)
    assert pred.grad.shape == pred.data.shape
    assert not np.allclose(pred.grad, 0.0)


# --------------------------------------------------------------------------- #
# central-difference gradcheck: d(mse_loss)/d(param) vs autodiff .grad
# --------------------------------------------------------------------------- #
def test_mse_gradcheck_wrt_parameter():
    """Perturb one entry of a Dense weight by +/- eps, recompute the loss
    forward-only, and compare the central difference to the autodiff grad."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 2))
    y = np.array([[1.0], [-1.0], [1.0], [-1.0], [1.0], [-1.0]])

    model = _requires(MLP, [2, 5, 1], activation="tanh", seed=1)

    # --- analytic grad via backward ---
    model.zero_grad()
    pred = _requires(model.__call__, X)
    loss = _requires(mse_loss, pred, y)
    _requires(loss.backward)
    params = _requires(model.parameters)
    p = params[0]                       # first weight matrix
    g_analytic = np.array(p.grad, copy=True)

    # --- numeric grad: central difference on every entry of p.data ---
    def loss_at(param_data):
        saved = np.array(p.data, copy=True)
        p.data = param_data
        out = mse_loss(model(X), y)
        val = float(out.data)
        p.data = saved
        return val

    g_numeric = np.zeros_like(p.data)
    flat = p.data.reshape(-1)
    for i in range(flat.size):
        idx = np.unravel_index(i, p.data.shape)
        base = np.array(p.data, copy=True)
        up = np.array(base, copy=True); up[idx] += EPS
        dn = np.array(base, copy=True); dn[idx] -= EPS
        g_numeric[idx] = (loss_at(up) - loss_at(dn)) / (2 * EPS)

    assert np.allclose(g_analytic, g_numeric, rtol=1e-4, atol=1e-5), (
        f"param gradcheck failed:\nanalytic=\n{g_analytic}\nnumeric=\n{g_numeric}"
    )


# --------------------------------------------------------------------------- #
# SGD optimizer
# --------------------------------------------------------------------------- #
def test_sgd_step_moves_against_gradient():
    p = Tensor([[1.0, -2.0], [3.0, 4.0]])
    p.grad = np.array([[0.5, 0.5], [-1.0, 2.0]])
    before = np.array(p.data, copy=True)
    opt = _requires(SGD, [p], lr=0.1)
    _requires(opt.step)
    expected = before - 0.1 * np.array([[0.5, 0.5], [-1.0, 2.0]])
    assert np.allclose(p.data, expected)


def test_sgd_step_does_not_extend_graph():
    """step() must mutate .data in place, not create a new differentiable node:
    the parameter object identity and its leaf-ness must be preserved."""
    p = Tensor([1.0, 2.0, 3.0])
    p.grad = np.ones_like(p.data)
    opt = _requires(SGD, [p], lr=0.5)
    _requires(opt.step)
    # still a leaf (no parents recorded by the update)
    assert p.parents == ()
    assert p.operation == ""


def test_sgd_zero_grad():
    p = Tensor([1.0, 2.0])
    p.grad = np.array([5.0, -3.0])
    opt = _requires(SGD, [p], lr=0.1)
    _requires(opt.zero_grad)
    assert np.allclose(p.grad, 0.0)
    assert p.grad.shape == p.data.shape


# --------------------------------------------------------------------------- #
# zero-grad / accumulation: the bug the training loop must avoid
# --------------------------------------------------------------------------- #
def test_backward_accumulates_without_zero():
    """Two backward() calls with NO zero in between double the grad."""
    x = Tensor([1.0, 2.0, 3.0])
    loss1 = _requires(mse_loss, x, np.array([0.0, 0.0, 0.0]))
    _requires(loss1.backward)
    g_once = np.array(x.grad, copy=True)

    # second pass WITHOUT zeroing -> accumulates onto the same .grad
    loss2 = mse_loss(x, np.array([0.0, 0.0, 0.0]))
    loss2.backward()
    assert np.allclose(x.grad, 2.0 * g_once), (
        "grad did not accumulate; either backward overwrote or graph reused"
    )


def test_mlp_zero_grad_resets_params():
    model = _requires(MLP, [2, 4, 1], activation="tanh", seed=0)
    X = np.random.default_rng(1).standard_normal((5, 2))
    y = np.ones((5, 1))
    pred = _requires(model.__call__, X)
    loss = _requires(mse_loss, pred, y)
    _requires(loss.backward)
    params = _requires(model.parameters)
    assert any(not np.allclose(p.grad, 0.0) for p in params)
    _requires(model.zero_grad)
    for p in params:
        assert np.allclose(p.grad, 0.0)


# --------------------------------------------------------------------------- #
# accuracy metric
# --------------------------------------------------------------------------- #
def test_accuracy_perfect_and_half():
    pred = Tensor([[2.0], [-3.0], [0.7], [-0.1]])
    y = np.array([1.0, -1.0, 1.0, -1.0])
    assert np.isclose(_requires(accuracy, pred, y), 1.0)
    y_bad = np.array([-1.0, 1.0, 1.0, -1.0])
    assert np.isclose(_requires(accuracy, pred, y_bad), 0.5)


# --------------------------------------------------------------------------- #
# the full training loop
# --------------------------------------------------------------------------- #
def test_train_returns_history_and_decreases_loss():
    X, y = _requires(make_moons, n=160, noise=0.1, seed=0)
    model = _requires(MLP, [2, 16, 1], activation="tanh", seed=0)
    history = _requires(train, model, X, y, lr=0.1, epochs=50)
    assert isinstance(history, list)
    assert len(history) == 50
    assert all(np.isfinite(history))
    # loss should fall well below where it started.
    assert history[-1] < history[0]
    assert history[-1] < 0.9 * history[0]


def test_train_fits_noiseless_moons():
    X, y = _requires(make_moons, n=200, noise=0.0, seed=0)
    model = _requires(MLP, [2, 16, 1], activation="tanh", seed=0)
    _requires(train, model, X, y, lr=0.2, epochs=400)
    pred = model(X)
    acc = accuracy(pred, y)
    assert acc >= 0.90, f"expected >= 90% train accuracy on clean moons, got {acc:.3f}"


def test_train_uses_provided_optimizer():
    X, y = _requires(make_moons, n=120, noise=0.1, seed=0)
    model = _requires(MLP, [2, 8, 1], activation="tanh", seed=0)
    opt = _requires(SGD, model.parameters(), lr=0.1)
    history = _requires(train, model, X, y, epochs=20, optimizer=opt)
    assert len(history) == 20
    assert history[-1] <= history[0]
