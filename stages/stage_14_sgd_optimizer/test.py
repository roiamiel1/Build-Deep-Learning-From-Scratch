"""Tests for Stage 14: SGD optimizer.

These tests verify the optimizer mechanics on top of the ``Tensor`` engine from
stage_09:

  * ``zero_grad()`` clears every parameter's accumulated gradient,
  * a single ``SGD.step()`` moves each ``p.data`` by exactly ``-lr * p.grad``,
  * the optimizer mutates the SAME tensor objects the "model" holds,
  * ``Optimizer.step()`` (the base class) is abstract, and
  * an end-to-end loop on a convex quadratic loss drives the loss monotonically
    down, with each step's analytic gradient matching a central-difference
    gradcheck:

        df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

If stage_09's ``Tensor`` or this stage's ``SGD`` is not yet implemented, the
suite skips rather than erroring, so you can run it incrementally.

Run with:  pytest stage_14_sgd_optimizer/test.py
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # curriculum root, so `import dlfs` works
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

# --- Import the things under test, skipping cleanly if not ready yet. --------
# code.py pulls Tensor from stage_09 via dlfs.stage_import; importing it here
# runs that import (it must succeed even while the SGD bodies are skeletons).
try:
    from code import SGD, Optimizer, Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_14 SGD / stage_09 Tensor not importable yet: {exc}",
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
    """Build a Tensor from array-like data, tolerating ctor kwarg differences."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=True)
    except TypeError:
        return Tensor(arr)


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


def set_grad(p, g):
    """Stamp a known gradient onto a parameter (so step() has something to use)."""
    p.grad = np.array(g, dtype=float).reshape(as_array(p).shape)


# --- Construction -----------------------------------------------------------
def test_collects_params_from_iterator():
    # A generator must not be exhausted by the optimizer keeping it as-is.
    ps = [make_tensor([1.0, 2.0]), make_tensor([3.0])]
    opt = SGD((p for p in ps), lr=0.1)
    assert len(opt.params) == 2, "SGD must materialize the params iterable into a list"


def test_lr_must_be_positive():
    p = make_tensor([1.0])
    with pytest.raises((ValueError, AssertionError)):
        SGD([p], lr=0.0)
    with pytest.raises((ValueError, AssertionError)):
        SGD([p], lr=-0.5)


def test_repr_mentions_lr_and_count():
    p = make_tensor([1.0, 2.0])
    r = repr(SGD([p], lr=0.05))
    assert "SGD" in r and "0.05" in r


def test_base_step_is_abstract():
    p = make_tensor([1.0])
    base = Optimizer([p])
    with pytest.raises(NotImplementedError):
        base.step()


# --- Core update rule -------------------------------------------------------
def test_step_applies_exact_update():
    p1 = make_tensor([1.0, -2.0, 3.0])
    p2 = make_tensor([[0.5, -0.5]])
    set_grad(p1, [0.1, 0.2, -0.3])
    set_grad(p2, [[2.0, -4.0]])

    before1 = as_array(p1).copy()
    before2 = as_array(p2).copy()
    lr = 0.1
    SGD([p1, p2], lr=lr).step()

    assert np.allclose(as_array(p1), before1 - lr * np.array([0.1, 0.2, -0.3]), atol=ATOL), (
        "p1.data must move by -lr * p1.grad"
    )
    assert np.allclose(as_array(p2), before2 - lr * np.array([[2.0, -4.0]]), atol=ATOL), (
        "p2.data must move by -lr * p2.grad"
    )


def test_step_does_not_modify_grad():
    p = make_tensor([1.0, 2.0])
    set_grad(p, [0.5, -0.5])
    g_before = as_array(p.grad).copy()
    SGD([p], lr=0.3).step()
    assert np.allclose(as_array(p.grad), g_before, atol=ATOL), (
        "step() must not clear or alter p.grad (that is zero_grad's job)"
    )


def test_step_mutates_in_place_same_object():
    p = make_tensor([4.0, 5.0])
    set_grad(p, [1.0, 1.0])
    opt = SGD([p], lr=0.5)
    same = opt.params[0]
    opt.step()
    # The optimizer must hold the very same Tensor object as the caller.
    assert same is p, "optimizer must keep the same Tensor objects, not copies"
    assert np.allclose(as_array(p), [3.5, 4.5], atol=ATOL), "in-place update expected"


def test_step_skips_none_grad():
    p = make_tensor([1.0, 2.0])
    p.grad = None  # simulate a param that never received a gradient
    before = as_array(p).copy()
    SGD([p], lr=0.1).step()  # must not raise
    assert np.allclose(as_array(p), before, atol=ATOL), "params with grad=None are skipped"


def test_zero_grad_clears_all():
    p1 = make_tensor([1.0, 2.0])
    p2 = make_tensor([3.0])
    set_grad(p1, [9.0, 9.0])
    set_grad(p2, [9.0])
    opt = SGD([p1, p2], lr=0.1)
    opt.zero_grad()
    assert np.allclose(as_array(p1.grad), 0.0), "zero_grad must clear p1.grad"
    assert np.allclose(as_array(p2.grad), 0.0), "zero_grad must clear p2.grad"
    assert as_array(p1.grad).shape == as_array(p1).shape, "grad shape must match data"


# --- Gradient comes from autodiff, not hand-coding --------------------------
def test_grad_matches_central_difference():
    """The .grad SGD consumes is produced by Tensor.backward(); verify it.

    Loss: L(w) = sum( (w - target)**2 ), a convex quadratic. dL/dw = 2(w - target).
    """
    w_np = np.array([0.5, -1.0, 2.0])
    target = np.array([1.0, 0.0, -1.0])

    def loss_value(w):
        wt = Tensor(np.asarray(w, dtype=float))
        diff = wt - Tensor(target)
        return float(np.sum(as_array(diff) ** 2))

    w = make_tensor(w_np)
    diff = w - Tensor(target)
    loss = (diff * diff)
    # reduce to a scalar so .backward() seeds a scalar; sum-of-squares.
    loss = loss.sum() if hasattr(loss, "sum") else loss
    loss.backward()
    g_analytic = as_array(w.grad)

    g_num = central_diff(loss_value, w_np.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"autodiff grad must match central difference:\n"
        f" analytic={g_analytic}\n numeric ={g_num}"
    )


# --- End-to-end: SGD minimizes a convex quadratic ---------------------------
def test_sgd_drives_quadratic_loss_down():
    """Minimize L(w) = sum((w - target)**2) with the full backward/step/zero loop.

    Loss must decrease (nearly) monotonically and converge near the minimum.
    """
    target = np.array([1.0, -2.0, 0.5])
    w = make_tensor([0.0, 0.0, 0.0])
    opt = SGD([w], lr=0.1)

    losses = []
    for _ in range(200):
        opt.zero_grad()
        diff = w - Tensor(target)
        loss = diff * diff
        loss = loss.sum() if hasattr(loss, "sum") else loss
        loss.backward()
        losses.append(float(as_array(loss).sum()))
        opt.step()

    assert losses[-1] < losses[0], "loss must decrease over training"
    assert losses[-1] < 1e-4, f"SGD should converge near the minimum, got {losses[-1]}"
    assert np.allclose(as_array(w), target, atol=1e-2), (
        f"weights should approach the target; got {as_array(w)} vs {target}"
    )
    # Monotone-ish: no step should increase the loss for this small lr.
    for prev, cur in zip(losses, losses[1:]):
        assert cur <= prev + 1e-9, "loss increased during SGD on a convex quadratic"


def test_step_count_changes_params_progressively():
    w = make_tensor([10.0])
    opt = SGD([w], lr=0.1)
    history = [as_array(w).copy()]
    for _ in range(5):
        opt.zero_grad()
        # L = w**2 -> dL/dw = 2w
        loss = w * w
        loss = loss.sum() if hasattr(loss, "sum") else loss
        loss.backward()
        opt.step()
        history.append(as_array(w).copy())
    # |w| strictly shrinks each step (0 < lr*2 < 1 -> contraction factor 0.8).
    mags = [float(np.abs(h).sum()) for h in history]
    for prev, cur in zip(mags, mags[1:]):
        assert cur < prev, "each SGD step should shrink |w| toward the minimum at 0"
