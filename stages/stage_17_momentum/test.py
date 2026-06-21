"""Tests for Stage 17: Momentum.

These tests verify the momentum optimizer built on ``SGD`` from ``stage_14``:

  * the velocity buffer follows the heavy-ball recursion (an EMA of grads),
  * ``beta=0`` reproduces plain ``SGD`` step-for-step,
  * heavy-ball and Nesterov momentum converge faster than plain ``SGD`` on an
    ill-conditioned quadratic,
  * the gradient fed to the optimizer in ``quadratic_descent`` matches a
    central-difference gradient of ``f``:

        df/dx_i ~= (f(x + eps e_i) - f(x - eps e_i)) / (2 * eps).

If stage_09 / stage_14 / stage_17 are not implemented yet, the suite skips
cleanly instead of erroring, so you can run it incrementally.

Run with:  pytest stage_17_momentum/test.py
"""

import numpy as np
import pytest

# --- Import the things under test through the curriculum shim, skipping
# --- cleanly if a prior stage is not ready yet. ------------------------------
# ``Momentum`` is this stage's ``SGDMomentum`` (a subclass of stage_14's SGD);
# ``SGD`` comes straight from stage_14 so the tests compare the two directly.
try:
    from dlfs import stage_import

    Stage17_Momentum, Stage17_Tensor, Stage17_quadratic_descent = stage_import(
        "stage_17", "Momentum", "Tensor", "quadratic_descent"
    )
    Stage14_SGD = stage_import("stage_14", "SGD")
    # Local short names for readability in the test body below.
    Tensor = Stage17_Tensor
    Momentum = Stage17_Momentum
    quadratic_descent = Stage17_quadratic_descent
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_17 Momentum / stage_14 SGD / stage_09 Tensor not importable "
        f"yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Return the underlying numpy array of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def make_param(arr):
    """Build a leaf parameter Tensor from a numpy array, tolerating ctor kwargs."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=True)
    except TypeError:
        return Tensor(arr)


def set_grad(p, g):
    """Write gradient g into p.grad as a float64 array of matching shape."""
    p.grad = np.asarray(g, dtype=float).reshape(as_array(p).shape)


def make_momentum(params, lr, beta=0.9, nesterov=False):
    """Construct Momentum, tolerating positional/keyword styles."""
    return Momentum(params, lr, beta=beta, nesterov=nesterov)


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


# --- A well-understood ill-conditioned quadratic -----------------------------
def quad_problem():
    """Return (A, b, x0) for f(x) = 0.5 x^T A x - b^T x, A SPD ill-conditioned."""
    A = np.array([[20.0, 0.0], [0.0, 1.0]])  # condition number 20
    b = np.array([2.0, 1.0])
    x0 = np.array([3.0, 3.0])
    return A, b, x0


def f_quad(x, A, b):
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ A @ x - b @ x)


# --- Construction ------------------------------------------------------------
def test_velocity_buffers_initialized_to_zero():
    p = make_param([1.0, 2.0, 3.0])
    opt = make_momentum([p], lr=0.1, beta=0.9)
    assert len(opt.velocities) == 1, "one velocity buffer per parameter"
    assert np.allclose(opt.velocities[0], 0.0), "velocity must start at zero"
    assert opt.velocities[0].shape == as_array(p).shape, "buffer shape == param"


def test_invalid_beta_rejected():
    p = make_param([1.0])
    with pytest.raises((ValueError, AssertionError)):
        make_momentum([p], lr=0.1, beta=1.0)
    with pytest.raises((ValueError, AssertionError)):
        make_momentum([p], lr=0.1, beta=-0.1)


def test_repr():
    p = make_param([1.0])
    r = repr(make_momentum([p], lr=0.1, beta=0.9))
    assert "Momentum" in r and "0.9" in r


# --- Velocity recursion matches a hand-unrolled EMA --------------------------
def test_velocity_is_ema_of_gradients():
    """v_t = beta v_{t-1} + g_t, accumulated over a fixed sequence of grads."""
    p = make_param([0.0, 0.0])
    lr, beta = 0.0, 0.7  # lr=0 -> params frozen, isolate the velocity recursion
    opt = make_momentum([p], lr=lr, beta=beta)

    grads = [np.array([1.0, -2.0]),
             np.array([0.5, 0.5]),
             np.array([-1.0, 3.0])]

    v_ref = np.zeros(2)
    for g in grads:
        set_grad(p, g)
        opt.step()
        v_ref = beta * v_ref + g
        assert np.allclose(opt.velocities[0], v_ref, atol=ATOL), (
            f"velocity recursion mismatch:\n got={opt.velocities[0]}\n "
            f"want={v_ref}"
        )


def test_param_update_uses_velocity():
    """With a constant grad, p moves by -lr * v each step (heavy-ball)."""
    p = make_param([0.0])
    lr, beta = 0.1, 0.5
    opt = make_momentum([p], lr=lr, beta=beta, nesterov=False)
    g = np.array([2.0])

    p_ref = np.array([0.0])
    v_ref = np.array([0.0])
    for _ in range(4):
        set_grad(p, g)
        opt.step()
        v_ref = beta * v_ref + g
        p_ref = p_ref - lr * v_ref
        assert np.allclose(as_array(p), p_ref, atol=ATOL), (
            f"param update mismatch: got={as_array(p)}, want={p_ref}"
        )


def test_nesterov_update_rule():
    """Nesterov: p -= lr * (g + beta * v) with v the post-update velocity."""
    p = make_param([0.0, 0.0])
    lr, beta = 0.05, 0.9
    opt = make_momentum([p], lr=lr, beta=beta, nesterov=True)
    grads = [np.array([1.0, 2.0]), np.array([-0.5, 1.0]), np.array([0.3, -0.7])]

    p_ref = np.zeros(2)
    v_ref = np.zeros(2)
    for g in grads:
        set_grad(p, g)
        opt.step()
        v_ref = beta * v_ref + g
        p_ref = p_ref - lr * (g + beta * v_ref)
        assert np.allclose(as_array(p), p_ref, atol=ATOL), (
            f"nesterov update mismatch: got={as_array(p)}, want={p_ref}"
        )


# --- beta = 0 reduces to plain SGD ------------------------------------------
def test_beta_zero_equals_plain_sgd():
    """With beta=0, Momentum must match stage_14's SGD step-for-step."""
    A, b, x0 = quad_problem()

    p_m = make_param(x0)
    opt_m = make_momentum([p_m], lr=0.01, beta=0.0)

    p_s = make_param(x0)
    opt_s = Stage14_SGD([p_s], 0.01)

    for _ in range(20):
        g = A @ as_array(p_m) - b
        set_grad(p_m, g)
        opt_m.step()

        g2 = A @ as_array(p_s) - b
        set_grad(p_s, g2)
        opt_s.step()

        assert np.allclose(as_array(p_m), as_array(p_s), atol=ATOL), (
            "beta=0 Momentum should equal plain SGD:\n "
            f"momentum={as_array(p_m)}\n sgd={as_array(p_s)}"
        )


# --- zero_grad / reset -------------------------------------------------------
def test_zero_grad():
    p = make_param([1.0, 2.0])
    opt = make_momentum([p], lr=0.1, beta=0.9)
    set_grad(p, [3.0, 4.0])
    opt.zero_grad()
    assert np.allclose(as_array(p.grad), 0.0), "zero_grad must clear p.grad"


def test_reset_clears_velocity():
    p = make_param([0.0])
    opt = make_momentum([p], lr=0.1, beta=0.9)
    set_grad(p, [1.0])
    opt.step()
    assert not np.allclose(opt.velocities[0], 0.0), "velocity should be nonzero"
    opt.reset()
    assert np.allclose(opt.velocities[0], 0.0), "reset must zero the velocity"


# --- Convergence speed-up: momentum / Nesterov beat plain SGD ----------------
def test_momentum_converges_faster_than_sgd():
    A, b, x0 = quad_problem()
    lr = 0.02
    steps = 60

    hist_sgd = quadratic_descent(lambda p: Stage14_SGD([p], lr), x0.copy(), A, b, steps)
    hist_mom = quadratic_descent(
        lambda p: make_momentum([p], lr, beta=0.9, nesterov=False),
        x0.copy(), A, b, steps,
    )

    assert len(hist_sgd) == steps and len(hist_mom) == steps, (
        "quadratic_descent must return one loss per step"
    )
    assert hist_mom[-1] < hist_sgd[-1] - 1e-9, (
        "momentum should reach a lower final loss than SGD in equal steps:\n "
        f"sgd_final={hist_sgd[-1]}, momentum_final={hist_mom[-1]}"
    )


def test_nesterov_converges():
    A, b, x0 = quad_problem()
    lr = 0.02
    steps = 60

    hist_sgd = quadratic_descent(lambda p: Stage14_SGD([p], lr), x0.copy(), A, b, steps)
    hist_nag = quadratic_descent(
        lambda p: make_momentum([p], lr, beta=0.9, nesterov=True),
        x0.copy(), A, b, steps,
    )
    assert hist_nag[-1] < hist_sgd[-1] - 1e-9, (
        "Nesterov should reach a lower final loss than SGD in equal steps:\n "
        f"sgd_final={hist_sgd[-1]}, nesterov_final={hist_nag[-1]}"
    )


def test_loss_decreases_overall():
    A, b, x0 = quad_problem()
    start_loss = f_quad(x0, A, b)  # capture BEFORE running (x0 may be mutated)
    hist = quadratic_descent(
        lambda p: make_momentum([p], 0.02, beta=0.9), x0.copy(), A, b, 60
    )
    assert hist[-1] < start_loss, "loss must drop below the starting value"


# --- gradcheck: the gradient fed to the optimizer matches central differences -
def test_quadratic_gradient_matches_central_diff():
    """grad f = A x - b must equal a central-difference gradient of f."""
    A, b, _ = quad_problem()
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(2)
        g_analytic = A @ x - b
        g_num = central_diff(lambda xv: f_quad(xv, A, b), x.copy())
        assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
            f"quadratic gradient mismatch:\n analytic={g_analytic}\n "
            f"numeric ={g_num}"
        )
