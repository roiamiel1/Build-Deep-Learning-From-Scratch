"""Tests for Stage 21: L1 & L2 regularization.

The penalties are differentiable ``Tensor`` functions, so we gradcheck them with
central differences: for a scalar penalty ``R`` and a weight ``Tensor`` ``w``,

    dR/dw_i  ~=  (R(w + eps e_i) - R(w - eps e_i)) / (2 eps)

and compare against the analytic gradient that ``Tensor.backward()`` deposits in
``w.grad`` (== ``lam * w`` for L2, ``lam * sign(w)`` for L1). We also assert the
forward values match NumPy references and that the L2 penalty gradient equals the
``stage_18`` coupled ``weight_decay`` term ``lam * p.data``.

If stage_21's (or stage_09's) ``code.py`` is not implemented yet, the suite skips
cleanly.

Run with:  pytest stage_21_regularization/test.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import the things under test, skipping cleanly if not ready yet. --------
try:
    from code import (
        Tensor,
        l1_penalty,
        l2_penalty,
        regularized_loss,
        l2_grad_equals_weight_decay,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_21 regularization not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-5


def _maybe_skip(thunk):
    """Run ``thunk`` (which builds Tensors / calls penalties); skip on TODO."""
    try:
        return thunk()
    except NotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"stage_21 piece not implemented yet: {exc}")


def central_diff_penalty(penalty_fn, w_array, lam):
    """Central-difference gradient of a scalar penalty w.r.t. an array param.

    Rebuilds a fresh single-param Tensor for each perturbed forward pass so no
    stale graph/grad leaks between evaluations.
    """
    w = np.asarray(w_array, dtype=float)
    grad = np.zeros_like(w)
    it = np.nditer(w, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = w[idx]

        w[idx] = orig + EPS
        fp = float(penalty_fn([Tensor(w.copy())], lam).data)
        w[idx] = orig - EPS
        fm = float(penalty_fn([Tensor(w.copy())], lam).data)

        w[idx] = orig
        grad[idx] = (fp - fm) / (2 * EPS)
        it.iternext()
    return grad


# ===========================================================================
# Forward values
# ===========================================================================
def test_l2_forward_matches_numpy():
    w1 = np.array([1.0, -2.0, 3.0])
    w2 = np.array([[0.5, -0.5], [2.0, 1.0]])
    lam = 0.3
    out = _maybe_skip(lambda: l2_penalty([Tensor(w1), Tensor(w2)], lam))
    expected = lam * 0.5 * (np.sum(w1 ** 2) + np.sum(w2 ** 2))
    assert np.isclose(float(out.data), expected, atol=ATOL, rtol=RTOL), (
        f"l2_penalty forward mismatch: got {float(out.data)}, expected {expected}"
    )


def test_l1_forward_matches_numpy():
    w1 = np.array([1.0, -2.0, 3.0])
    w2 = np.array([[0.5, -0.5], [2.0, 1.0]])
    lam = 0.7
    out = _maybe_skip(lambda: l1_penalty([Tensor(w1), Tensor(w2)], lam))
    expected = lam * (np.sum(np.abs(w1)) + np.sum(np.abs(w2)))
    assert np.isclose(float(out.data), expected, atol=ATOL, rtol=RTOL), (
        f"l1_penalty forward mismatch: got {float(out.data)}, expected {expected}"
    )


def test_penalties_are_scalar():
    out2 = _maybe_skip(lambda: l2_penalty([Tensor(np.arange(6.0).reshape(2, 3))]))
    out1 = _maybe_skip(lambda: l1_penalty([Tensor(np.arange(6.0).reshape(2, 3))]))
    assert np.ndim(out2.data) == 0, "l2_penalty must return a scalar (0-d) Tensor"
    assert np.ndim(out1.data) == 0, "l1_penalty must return a scalar (0-d) Tensor"


def test_lam_zero_is_zero():
    w = np.array([3.0, -4.0, 5.0])
    z2 = _maybe_skip(lambda: l2_penalty([Tensor(w)], 0.0))
    z1 = _maybe_skip(lambda: l1_penalty([Tensor(w)], 0.0))
    assert np.isclose(float(z2.data), 0.0, atol=ATOL), "l2_penalty(lam=0) must be 0"
    assert np.isclose(float(z1.data), 0.0, atol=ATOL), "l1_penalty(lam=0) must be 0"


# ===========================================================================
# Gradcheck: L2
# ===========================================================================
def test_l2_gradcheck_central_difference():
    w = np.array([0.7, -1.3, 0.2, 2.1])
    lam = 0.4
    t = Tensor(w.copy())
    out = _maybe_skip(lambda: l2_penalty([t], lam))
    _maybe_skip(lambda: out.backward())

    g_num = central_diff_penalty(l2_penalty, w, lam)
    assert np.allclose(t.grad, g_num, atol=1e-6, rtol=1e-5), (
        f"l2 gradcheck failed:\n analytic={t.grad}\n numeric ={g_num}"
    )


def test_l2_grad_equals_lam_times_w():
    """The closed-form L2 penalty gradient is lam * w."""
    w = np.array([1.0, -2.0, 3.0, 0.0])
    lam = 0.25
    t = Tensor(w.copy())
    out = _maybe_skip(lambda: l2_penalty([t], lam))
    _maybe_skip(lambda: out.backward())
    assert np.allclose(t.grad, lam * w, atol=ATOL), (
        f"d(lam*R2)/dw must equal lam*w: got {t.grad}, expected {lam * w}"
    )


# ===========================================================================
# Gradcheck: L1
# ===========================================================================
def test_l1_gradcheck_central_difference():
    # Keep entries away from 0 so the sub-gradient kink is not probed.
    w = np.array([0.8, -1.5, 0.3, 2.4])
    lam = 0.6
    t = Tensor(w.copy())
    out = _maybe_skip(lambda: l1_penalty([t], lam))
    _maybe_skip(lambda: out.backward())

    g_num = central_diff_penalty(l1_penalty, w, lam)
    assert np.allclose(t.grad, g_num, atol=1e-6, rtol=1e-5), (
        f"l1 gradcheck failed:\n analytic={t.grad}\n numeric ={g_num}"
    )


def test_l1_grad_equals_lam_times_sign():
    w = np.array([2.0, -3.0, 0.5, -0.1])
    lam = 0.5
    t = Tensor(w.copy())
    out = _maybe_skip(lambda: l1_penalty([t], lam))
    _maybe_skip(lambda: out.backward())
    assert np.allclose(t.grad, lam * np.sign(w), atol=ATOL), (
        f"d(lam*R1)/dw must equal lam*sign(w): got {t.grad}, "
        f"expected {lam * np.sign(w)}"
    )


# ===========================================================================
# Multi-parameter penalties accumulate into each param's grad
# ===========================================================================
def test_multi_param_grads():
    wa = np.array([1.0, -2.0])
    wb = np.array([[3.0, -4.0]])
    lam = 0.2
    ta, tb = Tensor(wa.copy()), Tensor(wb.copy())
    out = _maybe_skip(lambda: l2_penalty([ta, tb], lam))
    _maybe_skip(lambda: out.backward())
    assert np.allclose(ta.grad, lam * wa, atol=ATOL), "param A L2 grad wrong"
    assert np.allclose(tb.grad, lam * wb, atol=ATOL), "param B L2 grad wrong"


# ===========================================================================
# Equivalence with stage_18 coupled weight decay
# ===========================================================================
def test_l2_grad_matches_weight_decay_helper():
    """l2_grad_equals_weight_decay returns lam*p.data == stage_18 wd term."""
    wa = np.array([1.0, -2.0, 3.0])
    wb = np.array([[0.5], [-1.5]])
    lam = 0.3
    ta, tb = Tensor(wa.copy()), Tensor(wb.copy())

    ref = _maybe_skip(lambda: l2_grad_equals_weight_decay([ta, tb], lam))
    assert np.allclose(ref[0], lam * wa, atol=ATOL)
    assert np.allclose(ref[1], lam * wb, atol=ATOL)

    # And it must match what the autodiff engine produces for the L2 penalty.
    out = _maybe_skip(lambda: l2_penalty([ta, tb], lam))
    _maybe_skip(lambda: out.backward())
    assert np.allclose(ta.grad, ref[0], atol=ATOL), (
        "autodiff L2 grad must equal the coupled weight-decay term lam*p.data"
    )
    assert np.allclose(tb.grad, ref[1], atol=ATOL)


# ===========================================================================
# regularized_loss
# ===========================================================================
def test_regularized_loss_zero_lambdas_is_bare_loss():
    loss = Tensor(2.5)
    w = Tensor(np.array([1.0, -2.0, 3.0]))
    out = _maybe_skip(lambda: regularized_loss(loss, [w], l1=0.0, l2=0.0))
    assert np.isclose(float(out.data), 2.5, atol=ATOL), (
        "regularized_loss with l1=l2=0 must equal the bare loss"
    )


def test_regularized_loss_adds_both_penalties():
    loss_val = 1.0
    w = np.array([1.0, -2.0, 3.0])
    l1, l2 = 0.1, 0.2
    out = _maybe_skip(
        lambda: regularized_loss(Tensor(loss_val), [Tensor(w)], l1=l1, l2=l2)
    )
    expected = loss_val + l1 * np.sum(np.abs(w)) + l2 * 0.5 * np.sum(w ** 2)
    assert np.isclose(float(out.data), expected, atol=ATOL, rtol=RTOL), (
        f"regularized_loss forward mismatch: got {float(out.data)}, "
        f"expected {expected}"
    )


def test_regularized_loss_grad_combines_data_and_penalty():
    """Grad of (data_loss + l2 penalty) w.r.t. w is dLoss/dw + l2*w."""
    # Use a simple data loss that depends on w: L = sum(w**2 * 0)  -> 0 here,
    # so the gradient on w is purely the penalty (keeps the check exact).
    w = np.array([1.0, -2.0, 3.0])
    l2 = 0.5
    t = Tensor(w.copy())
    bare = Tensor(0.0)  # constant data loss; no dependence on w
    out = _maybe_skip(lambda: regularized_loss(bare, [t], l1=0.0, l2=l2))
    _maybe_skip(lambda: out.backward())
    assert np.allclose(t.grad, l2 * w, atol=ATOL), (
        f"regularized_loss penalty grad wrong: got {t.grad}, expected {l2 * w}"
    )


# ===========================================================================
# L2 shrinkage: larger lam -> smaller fitted weights (a tiny ridge fit)
# ===========================================================================
def test_l2_shrinks_weights_more_with_larger_lambda():
    """Closed-form-ish: minimize 0.5*(w-target)**2 + 0.5*lam*w**2 by GD.

    The optimum is w* = target / (1 + lam); larger lam -> smaller |w*|.
    We run a few gradient steps using the autodiff penalty gradient and check
    the trend, which exercises the penalty inside an actual descent loop.
    """
    target = np.array([4.0, -6.0, 2.0])

    def fit(lam, steps=400, lr=0.1):
        w = Tensor(np.zeros_like(target))
        for _ in range(steps):
            w.zero_grad()
            data = ((w - Tensor(target)) ** 2)  # (w-target)^2 elementwise
            # mean over elements as the data loss, plus L2 penalty on w
            data_loss = data * 0.5
            # reduce to scalar via the engine's sum
            scalar_data = _sum_to_scalar(data_loss)
            total = regularized_loss(scalar_data, [w], l1=0.0, l2=lam)
            total.backward()
            w.data = w.data - lr * w.grad
        return w.data.copy()

    def _sum_to_scalar(t):
        # Reduce any Tensor to a scalar using whatever sum the engine exposes.
        for name in ("sum", "total", "reduce_sum"):
            fn = getattr(t, name, None)
            if callable(fn):
                try:
                    return fn()
                except TypeError:
                    return fn(None)
        raise NotImplementedError(
            "stage_09 Tensor needs a sum-to-scalar reduction for this test"
        )

    try:
        w_small = fit(0.0)
        w_large = fit(2.0)
    except NotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"reduction/penalty not ready: {exc}")

    assert np.all(np.abs(w_large) < np.abs(w_small) + 1e-9), (
        "larger L2 lambda must shrink fitted weights toward zero "
        f"(|w(lam=2)|={np.abs(w_large)} vs |w(lam=0)|={np.abs(w_small)})"
    )
