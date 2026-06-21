"""Stage 23 tests: Batch Normalization (1-D).

Covers:
  * forward standardization (mean ~ 0, var ~ 1) and the gamma/beta affine,
  * the hand-derived backward via central-difference gradcheck w.r.t. x, gamma,
    and beta,
  * running-stat (EMA) updates and the train/eval mode split.

Allowed tools mirror code.py: NumPy + stdlib only. No autodiff library.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # curriculum root, so `import dlfs` works
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

# --------------------------------------------------------------------------- #
# Import BatchNorm1d from this stage's code.py. code.py pulls Tensor from
# stage_09 via dlfs.stage_import; importing it here runs that import (it must
# succeed even while the BatchNorm1d bodies are still skeletons).
# --------------------------------------------------------------------------- #
try:
    from code import BatchNorm1d
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_23 BatchNorm1d / stage_09 Tensor not importable yet: {exc}",
        allow_module_level=True,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _scalar_loss(y: np.ndarray, w: np.ndarray) -> float:
    """A fixed, asymmetric scalar reduction of the BN output.

    Using a non-trivial weight matrix ``w`` (instead of a plain sum) makes the
    gradients per-element distinct, so the gradcheck actually exercises the
    full backward rather than a degenerate constant.
    """
    return float(np.sum(y * w))


def _numeric_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar ``f`` at array ``x``."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
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


def _rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# construction / shapes / defaults
# --------------------------------------------------------------------------- #
def test_init_shapes_and_defaults():
    bn = BatchNorm1d(4)
    assert bn.gamma.shape == (4,) and bn.beta.shape == (4,)
    assert np.allclose(bn.gamma, 1.0), "gamma must init to ones"
    assert np.allclose(bn.beta, 0.0), "beta must init to zeros"
    assert bn.running_mean.shape == (4,) and bn.running_var.shape == (4,)
    assert np.allclose(bn.running_mean, 0.0), "running_mean inits to zeros"
    assert np.allclose(bn.running_var, 1.0), "running_var inits to ones"
    assert bn.training is True, "layer starts in train mode"
    assert bn.parameters() == [bn.gamma, bn.beta] or (
        len(bn.parameters()) == 2
    ), "parameters() returns [gamma, beta] only (buffers excluded)"


def test_forward_shape():
    bn = BatchNorm1d(5)
    x = _rng(1).normal(size=(8, 5))
    y = bn.forward(x)
    assert y.shape == (8, 5), "BN preserves (B, C) shape"


# --------------------------------------------------------------------------- #
# forward standardization
# --------------------------------------------------------------------------- #
def test_train_forward_standardizes():
    bn = BatchNorm1d(3)  # gamma=1, beta=0
    x = _rng(2).normal(loc=5.0, scale=3.0, size=(64, 3))
    y = bn.forward(x)
    # Each feature (column) should be ~ zero-mean, unit-var (biased var).
    assert np.allclose(y.mean(axis=0), 0.0, atol=1e-6), (
        f"per-feature mean should be ~0, got {y.mean(axis=0)}"
    )
    assert np.allclose(y.var(axis=0), 1.0, atol=1e-3), (
        f"per-feature var should be ~1, got {y.var(axis=0)}"
    )


def test_gamma_beta_affine():
    bn = BatchNorm1d(3)
    bn.gamma = np.array([2.0, 0.5, 3.0])
    bn.beta = np.array([1.0, -1.0, 0.5])
    x = _rng(3).normal(size=(50, 3))
    y = bn.forward(x)
    # After affine, mean -> beta, std -> |gamma| (var -> gamma**2).
    assert np.allclose(y.mean(axis=0), bn.beta, atol=1e-6), (
        "per-feature mean should equal beta"
    )
    assert np.allclose(y.var(axis=0), bn.gamma ** 2, atol=1e-3), (
        "per-feature var should equal gamma**2"
    )


# --------------------------------------------------------------------------- #
# backward: param grads as plain sums
# --------------------------------------------------------------------------- #
def test_param_grad_formulas():
    bn = BatchNorm1d(4)
    bn.gamma = _rng(10).normal(size=4)
    bn.beta = _rng(11).normal(size=4)
    x = _rng(12).normal(size=(16, 4))
    y = bn.forward(x)
    g = _rng(13).normal(size=y.shape)
    bn.backward(g)
    # beta_grad = sum_B g ; gamma_grad = sum_B g * x_hat. Recover x_hat from y.
    x_hat = (y - bn.beta) / bn.gamma
    assert np.allclose(bn.beta_grad, g.sum(axis=0), atol=1e-9), (
        "beta_grad must be the column-sum of grad_out"
    )
    assert np.allclose(bn.gamma_grad, (g * x_hat).sum(axis=0), atol=1e-9), (
        "gamma_grad must be column-sum of grad_out * x_hat"
    )


# --------------------------------------------------------------------------- #
# central-difference gradchecks (train mode)
# --------------------------------------------------------------------------- #
def test_gradcheck_input_x():
    B, C = 12, 5
    x = _rng(20).normal(size=(B, C))
    gamma = _rng(21).normal(size=C)
    beta = _rng(22).normal(size=C)
    w = _rng(23).normal(size=(B, C))  # loss weights, fixed

    def make_bn():
        bn = BatchNorm1d(C)
        bn.gamma = gamma.copy()
        bn.beta = beta.copy()
        return bn

    # analytical dL/dx
    bn = make_bn()
    y = bn.forward(x)
    dy = w.copy()  # d(sum(y*w))/dy = w
    dx = bn.backward(dy)

    # numeric dL/dx (recompute a FRESH forward each probe so batch stats track x)
    def loss_of_x(xx):
        return _scalar_loss(make_bn().forward(xx), w)

    num = _numeric_grad(loss_of_x, x.copy())
    assert np.allclose(dx, num, atol=1e-6), (
        f"dL/dx mismatch: max abs diff {np.abs(dx - num).max():.2e}"
    )


def test_gradcheck_gamma():
    B, C = 10, 4
    x = _rng(30).normal(size=(B, C))
    gamma = _rng(31).normal(size=C)
    beta = _rng(32).normal(size=C)
    w = _rng(33).normal(size=(B, C))

    def make_bn(g):
        bn = BatchNorm1d(C)
        bn.gamma = g.copy()
        bn.beta = beta.copy()
        return bn

    bn = make_bn(gamma)
    y = bn.forward(x)
    bn.backward(w.copy())
    ana = bn.gamma_grad

    def loss_of_gamma(g):
        return _scalar_loss(make_bn(g).forward(x), w)

    num = _numeric_grad(loss_of_gamma, gamma.copy())
    assert np.allclose(ana, num, atol=1e-6), (
        f"dL/dgamma mismatch: max abs diff {np.abs(ana - num).max():.2e}"
    )


def test_gradcheck_beta():
    B, C = 10, 4
    x = _rng(40).normal(size=(B, C))
    gamma = _rng(41).normal(size=C)
    beta = _rng(42).normal(size=C)
    w = _rng(43).normal(size=(B, C))

    def make_bn(bvec):
        bn = BatchNorm1d(C)
        bn.gamma = gamma.copy()
        bn.beta = bvec.copy()
        return bn

    bn = make_bn(beta)
    bn.forward(x)
    bn.backward(w.copy())
    ana = bn.beta_grad

    def loss_of_beta(bvec):
        return _scalar_loss(make_bn(bvec).forward(x), w)

    num = _numeric_grad(loss_of_beta, beta.copy())
    assert np.allclose(ana, num, atol=1e-6), (
        f"dL/dbeta mismatch: max abs diff {np.abs(ana - num).max():.2e}"
    )


# --------------------------------------------------------------------------- #
# running statistics (EMA) and eval mode
# --------------------------------------------------------------------------- #
def test_running_stats_update():
    bn = BatchNorm1d(3, momentum=0.1)
    rng = _rng(50)
    rm0 = bn.running_mean.copy()
    rv0 = bn.running_var.copy()
    x = rng.normal(loc=2.0, scale=4.0, size=(128, 3))
    bn.forward(x)
    # one update => running moves toward batch stats, away from init.
    assert not np.allclose(bn.running_mean, rm0), "running_mean must update in train"
    assert not np.allclose(bn.running_var, rv0), "running_var must update in train"
    # exact EMA formula check (biased mean; unbiased var for the buffer).
    expected_mean = (1 - 0.1) * rm0 + 0.1 * x.mean(axis=0)
    expected_var = (1 - 0.1) * rv0 + 0.1 * x.var(axis=0, ddof=1)
    assert np.allclose(bn.running_mean, expected_mean, atol=1e-9), (
        "running_mean EMA uses the biased batch mean"
    )
    assert np.allclose(bn.running_var, expected_var, atol=1e-9), (
        "running_var EMA uses the UNBIASED (ddof=1) batch variance"
    )


def test_eval_uses_running_not_batch():
    bn = BatchNorm1d(2, momentum=1.0)  # momentum=1 => running := this batch's stats
    rng = _rng(60)
    x_train = rng.normal(loc=3.0, scale=2.0, size=(256, 2))
    bn.train()
    bn.forward(x_train)  # running_mean/var now equal x_train's biased mean / unbiased var
    bn.eval()
    # A different batch, with totally different stats.
    x_eval = rng.normal(loc=-10.0, scale=0.1, size=(256, 2))
    y = bn.forward(x_eval)
    # Eval must normalize by the RUNNING stats, so y's stats reflect x_eval
    # relative to x_train -- NOT zero-mean/unit-var of x_eval itself.
    assert not np.allclose(y.mean(axis=0), 0.0, atol=1e-1), (
        "eval must NOT standardize using the eval batch's own mean"
    )
    # Reproduce eval forward by hand from the buffers.
    expected = (x_eval - bn.running_mean) / np.sqrt(bn.running_var + bn.eps)
    expected = bn.gamma * expected + bn.beta
    assert np.allclose(y, expected, atol=1e-9), (
        "eval forward must use running_mean/running_var with eps"
    )


def test_eval_does_not_update_buffers():
    bn = BatchNorm1d(3)
    bn.eval()
    rm = bn.running_mean.copy()
    rv = bn.running_var.copy()
    bn.forward(_rng(70).normal(size=(40, 3)))
    assert np.allclose(bn.running_mean, rm), "eval forward must not touch running_mean"
    assert np.allclose(bn.running_var, rv), "eval forward must not touch running_var"


def test_eval_is_per_example_deterministic():
    """In eval mode the output for one row is independent of the rest of the batch."""
    bn = BatchNorm1d(3, momentum=1.0)
    rng = _rng(80)
    bn.train()
    bn.forward(rng.normal(size=(100, 3)))  # populate buffers
    bn.eval()
    row = rng.normal(size=(1, 3))
    y_alone = bn.forward(row)
    # Same row embedded in a larger batch -> identical output for that row.
    batch = np.concatenate([row, rng.normal(size=(7, 3))], axis=0)
    y_batch = bn.forward(batch)
    assert np.allclose(y_alone[0], y_batch[0], atol=1e-12), (
        "eval output for a row must not depend on the other rows in the batch"
    )


def test_train_eval_toggle():
    bn = BatchNorm1d(2)
    assert bn.training is True
    assert bn.eval().training is False, "eval() returns self and sets training=False"
    assert bn.train().training is True, "train() returns self and sets training=True"


# --------------------------------------------------------------------------- #
# zero_grad
# --------------------------------------------------------------------------- #
def test_zero_grad_resets_param_grads():
    bn = BatchNorm1d(3)
    x = _rng(90).normal(size=(16, 3))
    bn.forward(x)
    bn.backward(_rng(91).normal(size=(16, 3)))
    assert not np.allclose(bn.gamma_grad, 0.0) or not np.allclose(bn.beta_grad, 0.0)
    bn.zero_grad()
    assert np.allclose(bn.gamma_grad, 0.0), "zero_grad must reset gamma_grad"
    assert np.allclose(bn.beta_grad, 0.0), "zero_grad must reset beta_grad"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
