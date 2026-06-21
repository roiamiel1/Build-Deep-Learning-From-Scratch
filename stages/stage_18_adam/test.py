"""Tests for Stage 18: Adam (+ the RMSProp / AdamW it builds on).

This stage sits on top of the optimizer chain. ``Optimizer`` (base) and ``SGD``
come from ``stage_14`` and ``Momentum`` from ``stage_17``; this stage's
``code.py`` re-exports them (via ``dlfs.stage_import``) and ADDS ``RMSProp``,
``Adam`` and ``AdamW``. The tests import everything from this stage's ``code.py``
so they run against the extended family.

These tests treat an optimizer as an *update rule* over parameter objects that
expose ``.data`` and ``.grad`` (both ``np.ndarray``) -- exactly the contract of
the leaf ``Tensor``s returned by ``parameters()`` in stages 11-12. To keep the
suite self-contained we use a tiny ``Param`` stand-in with that same interface;
no autodiff engine is required to verify an update rule.

Where gradients exist we still gradcheck with central differences: for the
convex quadratic ``L(theta) = 0.5 * sum(theta**2)`` the analytic gradient is
``g = theta``, and we confirm the finite-difference gradient matches that before
feeding it to the optimizer:

    dL/dtheta_i ~= (L(theta + eps e_i) - L(theta - eps e_i)) / (2 eps)

If stage_18 (or the stage_14 / stage_17 it imports) is not implemented yet, the
suite skips cleanly.

Run with:  pytest stage_18_adam/test.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import the things under test, skipping cleanly if not ready yet. --------
# Optimizer / SGD originate in stage_14, Momentum in stage_17; this stage's
# code.py re-exports them and defines RMSProp / Adam / AdamW on top.
try:
    from code import Optimizer, SGD, Momentum, RMSProp, Adam, AdamW
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_18 optimizers not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
ATOL = 1e-8
RTOL = 1e-6


# --- A minimal parameter object matching the Tensor .data/.grad contract -----
class Param:
    """Tiny stand-in for a leaf ``Tensor``: just ``.data`` and ``.grad``."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)


def make_params(arrays):
    return [Param(a) for a in arrays]


def quadratic_loss(thetas):
    """L = 0.5 * sum(theta**2) over a list of arrays; grad is theta itself."""
    return 0.5 * sum(float(np.sum(t ** 2)) for t in thetas)


def central_diff(f, x, eps=EPS):
    """Numerical gradient of scalar f at numpy point x (any shape)."""
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


def set_grad_quadratic(params):
    """Set p.grad = p.data for every param (gradient of 0.5*sum(theta**2))."""
    for p in params:
        p.grad = p.data.copy()


# ===========================================================================
# Gradcheck: the gradient we feed the optimizer is the true gradient
# ===========================================================================
def test_quadratic_gradcheck():
    """g = theta is the analytic gradient of 0.5*sum(theta**2); verify numerically."""
    theta = np.array([0.7, -1.3, 0.2, 2.1])
    g_analytic = theta.copy()  # what we feed the optimizer

    def f(x):
        return 0.5 * float(np.sum(x ** 2))

    g_num = central_diff(f, theta.copy())
    assert np.allclose(g_analytic, g_num, atol=1e-6, rtol=1e-5), (
        f"quadratic gradient mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# ===========================================================================
# Base API
# ===========================================================================
def test_zero_grad_clears():
    params = make_params([np.array([1.0, 2.0]), np.array([[3.0]])])
    for p in params:
        p.grad = np.ones_like(p.data) * 5.0
    opt = SGD(params, lr=0.1)
    opt.zero_grad()
    for p in params:
        assert np.allclose(p.grad, 0.0), "zero_grad must clear every p.grad"
        assert p.grad.shape == p.data.shape, "grad shape must match data shape"


def test_base_step_not_implemented():
    params = make_params([np.array([1.0])])
    base = Optimizer(params)
    with pytest.raises(NotImplementedError):
        base.step()


def test_step_leaves_grad_untouched():
    params = make_params([np.array([1.0, -2.0, 3.0])])
    set_grad_quadratic(params)
    saved = params[0].grad.copy()
    Adam(params, lr=1e-3).step()
    assert np.allclose(params[0].grad, saved), "step() must not modify p.grad"


# ===========================================================================
# SGD / Momentum (imported from stage_14 / stage_17, used here unchanged)
# ===========================================================================
def test_sgd_plain_step():
    """stage_14 SGD is exactly p.data -= lr * p.grad."""
    lr = 0.05
    data = np.array([1.0, -2.0, 3.0])
    grad = np.array([0.5, 0.5, -1.0])
    p = Param(data.copy())
    p.grad = grad.copy()
    SGD([p], lr=lr).step()
    assert np.allclose(p.data, data - lr * grad), (
        "plain SGD must be exactly p.data -= lr*p.grad"
    )


def test_momentum_accumulates():
    """stage_17 Momentum: velocity grows as g, g(1+b), g(1+b+b^2), ..."""
    lr, b = 0.1, 0.9
    g = np.array([1.0, -2.0])
    p = Param(np.zeros(2))
    opt = Momentum([p], lr=lr, beta=b)

    v = np.zeros(2)
    theta = p.data.copy()
    for _ in range(5):
        p.grad = g.copy()
        opt.step()
        v = b * v + g
        theta = theta - lr * v
        assert np.allclose(p.data, theta, atol=ATOL), (
            "Momentum velocity update mismatch"
        )


# ===========================================================================
# RMSProp
# ===========================================================================
def test_rmsprop_reference_update():
    lr, beta, eps = 0.01, 0.99, 1e-8
    data = np.array([1.0, -2.0, 0.5])
    grad = np.array([0.3, -0.7, 2.0])
    p = Param(data.copy())
    p.grad = grad.copy()
    RMSProp([p], lr=lr, beta=beta, eps=eps).step()

    v = (1 - beta) * grad ** 2
    expected = data - lr * grad / (np.sqrt(v) + eps)
    assert np.allclose(p.data, expected, atol=ATOL), (
        f"RMSProp update mismatch:\n got     ={p.data}\n expected={expected}"
    )


# ===========================================================================
# Adam
# ===========================================================================
def test_adam_step_counter():
    p = Param(np.array([1.0]))
    opt = Adam([p], lr=1e-3)
    assert opt.t == 0, "Adam.t must start at 0"
    p.grad = np.array([1.0])
    opt.step()
    assert opt.t == 1, "first step() must make t == 1"
    opt.step()
    assert opt.t == 2, "t must increment each step"


def test_adam_reference_first_step():
    """One Adam step from m=v=0 against a hand-computed reference."""
    lr = 1e-3
    b1, b2, eps = 0.9, 0.999, 1e-8
    data = np.array([1.0, -2.0, 3.0])
    grad = np.array([0.5, -1.5, 0.25])
    p = Param(data.copy())
    p.grad = grad.copy()

    Adam([p], lr=lr, betas=(b1, b2), eps=eps).step()

    m = (1 - b1) * grad
    v = (1 - b2) * grad ** 2
    m_hat = m / (1 - b1 ** 1)
    v_hat = v / (1 - b2 ** 1)
    expected = data - lr * m_hat / (np.sqrt(v_hat) + eps)
    assert np.allclose(p.data, expected, atol=ATOL), (
        f"Adam first-step mismatch:\n got     ={p.data}\n expected={expected}"
    )


def test_adam_reference_two_steps():
    """Two Adam steps with changing gradients; checks the EMA carry-over."""
    lr, b1, b2, eps = 1e-2, 0.9, 0.999, 1e-8
    data = np.array([0.5, -0.5])
    g1 = np.array([1.0, -2.0])
    g2 = np.array([0.5, 0.5])
    p = Param(data.copy())
    opt = Adam([p], lr=lr, betas=(b1, b2), eps=eps)

    m = np.zeros(2)
    v = np.zeros(2)
    theta = data.copy()
    for t, g in enumerate([g1, g2], start=1):
        p.grad = g.copy()
        opt.step()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        assert np.allclose(p.data, theta, atol=ATOL), (
            f"Adam multi-step mismatch at t={t}:\n got={p.data}\n exp={theta}"
        )


def test_adam_betas_zero_is_signed_step():
    """betas=(0,0): m_hat=g, v_hat=g**2 -> update ~ -lr*sign(g) (eps tiny)."""
    lr = 1e-2
    grad = np.array([2.0, -5.0, 0.1])
    p = Param(np.zeros(3))
    p.grad = grad.copy()
    Adam([p], lr=lr, betas=(0.0, 0.0), eps=1e-12).step()
    expected = -lr * np.sign(grad)
    assert np.allclose(p.data, expected, atol=1e-6), (
        f"Adam betas=(0,0) must give a signed step:\n got={p.data}\n exp={expected}"
    )


def test_adam_bias_correction_constant_grad():
    """With a constant gradient g, the bias-corrected m_hat equals g exactly.

    Unrolling m_t = (1-b1) sum b1^{t-i} g = g (1 - b1^t); dividing by (1-b1^t)
    recovers g. We reconstruct m_hat from internal buffers if exposed; otherwise
    we verify the resulting step has magnitude ~ lr (since m_hat/sqrt(v_hat) ~ 1).
    """
    lr, b1, b2, eps = 1e-3, 0.9, 0.999, 1e-8
    g = np.array([3.0])
    p = Param(np.array([0.0]))
    opt = Adam([p], lr=lr, betas=(b1, b2), eps=eps)

    prev = p.data.copy()
    for t in range(1, 6):
        p.grad = g.copy()
        opt.step()
        step_mag = abs(float((p.data - prev).item()))
        prev = p.data.copy()
        # For constant grad, m_hat -> g and v_hat -> g**2, so the step magnitude
        # approaches lr * |g| / |g| = lr from the very first step.
        assert abs(step_mag - lr) < 5e-4, (
            f"bias-corrected Adam step at t={t} should be ~lr={lr}, got {step_mag}"
        )


def test_adam_converges_on_quadratic():
    rng = np.random.default_rng(0)
    p = Param(rng.uniform(-3, 3, size=10))
    opt = Adam([p], lr=0.1)
    for _ in range(500):
        set_grad_quadratic([p])
        opt.step()
    assert quadratic_loss([p.data]) < 1e-3, (
        "Adam should drive the convex quadratic loss near zero"
    )


# ===========================================================================
# AdamW
# ===========================================================================
def test_adamw_decouples_decay():
    """AdamW(wd>0) ends closer to zero than Adam(wd=0) on a zero-gradient param.

    With g = 0 the adaptive Adam step is ~0, so any shrinkage is purely the
    decoupled decay term lr*wd*theta.
    """
    start = np.array([4.0, -4.0])

    p_plain = Param(start.copy())
    p_plain.grad = np.zeros(2)
    Adam([p_plain], lr=1e-2, weight_decay=0.0).step()

    p_wd = Param(start.copy())
    p_wd.grad = np.zeros(2)
    AdamW([p_wd], lr=1e-2, weight_decay=0.5).step()

    assert np.all(np.abs(p_wd.data) < np.abs(p_plain.data)), (
        "AdamW decoupled decay must shrink params toward zero vs plain Adam"
    )


def test_adamw_differs_from_coupled_adam():
    """Decoupled (AdamW) and coupled (Adam wd) updates should not coincide."""
    data = np.array([1.0, -2.0, 3.0])
    grad = np.array([0.5, -1.0, 0.25])

    p_w = Param(data.copy())
    p_w.grad = grad.copy()
    AdamW([p_w], lr=1e-2, weight_decay=0.1).step()

    p_c = Param(data.copy())
    p_c.grad = grad.copy()
    Adam([p_c], lr=1e-2, weight_decay=0.1).step()

    assert not np.allclose(p_w.data, p_c.data), (
        "AdamW (decoupled) and Adam coupled weight decay should differ"
    )


# ===========================================================================
# Convergence of all optimizers on a convex quadratic
# ===========================================================================
@pytest.mark.parametrize(
    "make_opt,steps",
    [
        (lambda ps: Momentum(ps, lr=0.2, beta=0.9), 300),
        (lambda ps: RMSProp(ps, lr=0.05), 800),
        (lambda ps: Adam(ps, lr=0.1), 600),
        (lambda ps: AdamW(ps, lr=0.1, weight_decay=0.0), 600),
    ],
)
def test_all_optimizers_converge(make_opt, steps):
    rng = np.random.default_rng(1)
    p = Param(rng.uniform(-2, 2, size=8))
    opt = make_opt([p])
    for _ in range(steps):
        set_grad_quadratic([p])
        opt.step()
    assert quadratic_loss([p.data]) < 1e-2, (
        f"{type(opt).__name__} failed to minimize the convex quadratic"
    )


# ===========================================================================
# Multiple parameters with independent state
# ===========================================================================
def test_independent_per_param_state():
    """Each parameter must keep its own moment buffers (no shared state)."""
    a = Param(np.array([1.0, 1.0]))
    b = Param(np.array([10.0, 10.0, 10.0]))  # different shape on purpose
    opt = Adam([a, b], lr=1e-2)
    for _ in range(3):
        a.grad = np.array([1.0, -1.0])
        b.grad = np.array([2.0, 2.0, 2.0])
        opt.step()
    # Shapes preserved and both moved -> buffers were allocated per-parameter.
    assert a.data.shape == (2,) and b.data.shape == (3,)
    assert not np.allclose(a.data, [1.0, 1.0])
    assert not np.allclose(b.data, [10.0, 10.0, 10.0])
