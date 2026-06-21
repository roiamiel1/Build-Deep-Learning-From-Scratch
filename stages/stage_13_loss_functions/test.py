"""Tests for Stage 13: Loss functions.

Checks forward values against plain-NumPy reference losses and gradient-checks
each loss with respect to its (differentiable) prediction / logits using central
differences:

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

compared against the analytic gradient produced by ``Tensor.backward()`` from
stage_09. The losses live in this stage's ``code.py`` and are built on the
``Tensor`` imported from stage_09 through ``dlfs.stage_import``. If stage_09's
``Tensor`` or this stage's losses are not implemented yet, the suite skips
cleanly instead of erroring.

Run with:  pytest stage_13_loss_functions/test.py
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
# this stage's loss functions on top of it.
try:
    from code import (
        Tensor,
        cross_entropy_loss,
        log_softmax,
        mae_loss,
        mse_loss,
        one_hot,
        softmax,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_13 losses / stage_09 Tensor not importable yet: {exc}",
        allow_module_level=True,
    )

RNG = np.random.default_rng(13)
EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Underlying ndarray of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def loss_value(t):
    """Scalar python float held by a (0-d) loss Tensor."""
    return float(as_array(t).reshape(-1)[0]) if as_array(t).size == 1 else float(np.sum(as_array(t)))


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


def analytic_grad(loss_fn, pred_np, *rest):
    """Build the loss from a fresh prediction Tensor, backprop, return its grad."""
    p = Tensor(pred_np.copy())
    if hasattr(p, "zero_grad"):
        p.zero_grad()
    L = loss_fn(p, *rest)
    L.backward()
    return as_array(p.grad)


# =========================================================================== #
# one_hot helper
# =========================================================================== #
def test_one_hot_shape_and_values():
    oh = one_hot([0, 2, 1], num_classes=3)
    assert oh.shape == (3, 3)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    assert np.allclose(oh, expected), f"one_hot wrong:\n{oh}"


# =========================================================================== #
# MSE
# =========================================================================== #
def test_mse_forward_matches_numpy():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    targ = np.array([1.5, 1.0, 3.5, 2.0])
    got = loss_value(mse_loss(Tensor(pred), Tensor(targ)))
    ref = float(np.mean((pred - targ) ** 2))
    assert np.isclose(got, ref, atol=ATOL), f"MSE forward: got {got}, want {ref}"


def test_mse_zero_when_equal():
    a = np.array([[1.0, -2.0], [3.0, 4.0]])
    assert np.isclose(loss_value(mse_loss(Tensor(a), Tensor(a))), 0.0, atol=ATOL)


def test_mse_gradcheck():
    pred_np = RNG.normal(size=(3, 4))
    targ_np = RNG.normal(size=(3, 4))

    g_analytic = analytic_grad(mse_loss, pred_np, Tensor(targ_np))

    def f(p):
        return loss_value(mse_loss(Tensor(p), Tensor(targ_np)))

    g_num = central_diff(f, pred_np.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"MSE gradcheck mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


def test_mse_grad_formula():
    """dL/dpred should equal 2*(pred-target)/N."""
    pred_np = np.array([0.5, -1.0, 2.0, 0.0])
    targ_np = np.array([1.0, 0.0, 1.0, -1.0])
    g = analytic_grad(mse_loss, pred_np, Tensor(targ_np))
    expected = 2.0 * (pred_np - targ_np) / pred_np.size
    assert np.allclose(g, expected, atol=ATOL), f"got {g}, want {expected}"


# =========================================================================== #
# MAE
# =========================================================================== #
def test_mae_forward_matches_numpy():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    targ = np.array([1.5, 1.0, 3.5, 2.0])
    got = loss_value(mae_loss(Tensor(pred), Tensor(targ)))
    ref = float(np.mean(np.abs(pred - targ)))
    assert np.isclose(got, ref, atol=ATOL), f"MAE forward: got {got}, want {ref}"


def test_mae_gradcheck_away_from_kink():
    # Keep pred - target well away from 0 so |.| is differentiable everywhere.
    targ_np = RNG.normal(size=(3, 4))
    pred_np = targ_np + RNG.choice([-1.0, 1.0], size=(3, 4)) * (
        1.0 + np.abs(RNG.normal(size=(3, 4)))
    )

    g_analytic = analytic_grad(mae_loss, pred_np, Tensor(targ_np))

    def f(p):
        return loss_value(mae_loss(Tensor(p), Tensor(targ_np)))

    g_num = central_diff(f, pred_np.copy())
    assert np.allclose(g_analytic, g_num, atol=1e-5, rtol=RTOL), (
        f"MAE gradcheck mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


def test_mae_grad_is_sign_over_n():
    pred_np = np.array([2.0, -3.0, 0.5, 5.0])
    targ_np = np.array([0.0, 0.0, 0.0, 0.0])
    g = analytic_grad(mae_loss, pred_np, Tensor(targ_np))
    expected = np.sign(pred_np - targ_np) / pred_np.size
    assert np.allclose(g, expected, atol=ATOL), f"got {g}, want {expected}"


# =========================================================================== #
# softmax / log-softmax
# =========================================================================== #
def test_softmax_rows_sum_to_one():
    z = RNG.normal(size=(5, 7))
    p = as_array(softmax(Tensor(z)))
    assert np.allclose(p.sum(axis=1), 1.0, atol=ATOL), f"rows sum: {p.sum(axis=1)}"
    assert np.all(p >= 0.0), "softmax must be non-negative"


def test_softmax_matches_numpy_reference():
    z = RNG.normal(size=(4, 6))
    p = as_array(softmax(Tensor(z)))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(p, ref, atol=ATOL), f"softmax mismatch:\n{p}\n{ref}"


def test_log_softmax_matches_log_of_softmax():
    z = RNG.normal(size=(4, 5))
    lp = as_array(log_softmax(Tensor(z)))
    ref = np.log(as_array(softmax(Tensor(z))))
    assert np.allclose(lp, ref, atol=ATOL), f"log_softmax mismatch:\n{lp}\n{ref}"


def test_softmax_numerically_stable_large_logits():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    p = as_array(softmax(Tensor(z)))
    assert np.all(np.isfinite(p)), "softmax overflowed on large logits"
    e = np.exp(z - z.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(p, ref, atol=ATOL)


def test_log_softmax_stable_large_logits():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    lp = as_array(log_softmax(Tensor(z)))
    assert np.all(np.isfinite(lp)), "log_softmax produced inf/nan on large logits"


# =========================================================================== #
# cross-entropy
# =========================================================================== #
def test_cross_entropy_forward_matches_numpy():
    z = RNG.normal(size=(4, 5))
    t = np.array([0, 2, 4, 1])
    got = loss_value(cross_entropy_loss(Tensor(z), t))
    # reference
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    ref = float(np.mean(-np.log(p[np.arange(4), t])))
    assert np.isclose(got, ref, atol=ATOL), f"CE forward: got {got}, want {ref}"


def test_cross_entropy_accepts_onehot_targets():
    z = RNG.normal(size=(3, 4))
    t_idx = np.array([0, 3, 1])
    t_oh = one_hot(t_idx, 4)
    a = loss_value(cross_entropy_loss(Tensor(z), t_idx))
    b = loss_value(cross_entropy_loss(Tensor(z), t_oh))
    assert np.isclose(a, b, atol=ATOL), f"index vs one-hot CE differ: {a} vs {b}"


def test_cross_entropy_stable_large_logits():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    t = np.array([2])
    L = loss_value(cross_entropy_loss(Tensor(z), t))
    assert np.isfinite(L), "cross-entropy overflowed on large logits"


def test_cross_entropy_gradcheck_wrt_logits():
    z_np = RNG.normal(size=(4, 5))
    t = np.array([0, 2, 4, 1])

    g_analytic = analytic_grad(cross_entropy_loss, z_np, t)

    def f(z):
        return loss_value(cross_entropy_loss(Tensor(z), t))

    g_num = central_diff(f, z_np.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"CE gradcheck mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


def test_cross_entropy_grad_is_p_minus_y_over_b():
    """The fused gradient must equal (softmax(logits) - onehot) / B."""
    z_np = RNG.normal(size=(4, 5))
    t = np.array([0, 2, 4, 1])
    B = z_np.shape[0]

    g_analytic = analytic_grad(cross_entropy_loss, z_np, t)

    p = as_array(softmax(Tensor(z_np)))
    y = one_hot(t, z_np.shape[1])
    expected = (p - y) / B

    assert np.allclose(g_analytic, expected, atol=ATOL), (
        f"CE grad != (p - y)/B:\n analytic={g_analytic}\n expected={expected}"
    )
