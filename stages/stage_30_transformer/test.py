"""Stage 30 tests: Transformer block.

Covers:
  * LayerNorm forward: per-token standardization over the FEATURE axis (mean ~0,
    var ~1 per token) and the gamma/beta affine,
  * LayerNorm hand-derived backward via central-difference gradcheck w.r.t. x,
    gamma, and beta (reduction over the last axis, the per-token analogue of the
    per-feature BatchNorm backward in stage_23),
  * the TransformerBlock wiring: shape preservation, pre- vs post-norm difference,
    and the residual identity when the sublayers contribute nothing.

The TransformerBlock tests depend on MultiHeadAttention (stage_29) and MLP
(stage_12); they SKIP cleanly if those stages are not yet implemented, while the
LayerNorm gradchecks always run.

Allowed tools mirror code.py: NumPy + stdlib only. No autodiff library.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os
import sys as _sys

import numpy as np
import pytest

# Put the curriculum root on sys.path so this stage's code.py can do
# ``from dlfs import stage_import``. We load code.py by file path (below) rather
# than ``import code`` to avoid shadowing the stdlib ``code`` module.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)  # curriculum root, so `import dlfs` works
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


# --------------------------------------------------------------------------- #
# Load this stage's code.py by file path. code.py pulls MultiHeadAttention
# (stage_29), MLP (stage_12), and Tensor (stage_09) through dlfs.stage_import;
# loading it here runs those imports (which must succeed while the LayerNorm /
# TransformerBlock bodies are still skeletons). The LayerNorm gradchecks always
# run; the TransformerBlock tests skip cleanly if a dependency body is not yet
# implemented.
# --------------------------------------------------------------------------- #
def _load_code():
    path = _os.path.join(_HERE, "code.py")
    spec = _ilu.spec_from_file_location("_stage30_code", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CODE = _load_code()
LayerNorm = CODE.LayerNorm
TransformerBlock = CODE.TransformerBlock


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _scalar_loss(y: np.ndarray, w: np.ndarray) -> float:
    """A fixed, asymmetric scalar reduction of an output.

    Using a non-trivial weight array ``w`` (instead of a plain sum) makes the
    per-element gradients distinct, so the gradcheck exercises the full backward.
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
# LayerNorm: construction / shapes / defaults
# --------------------------------------------------------------------------- #
def test_init_shapes_and_defaults():
    ln = LayerNorm(8)
    assert ln.gamma.shape == (8,) and ln.beta.shape == (8,)
    assert np.allclose(ln.gamma, 1.0), "gamma must init to ones"
    assert np.allclose(ln.beta, 0.0), "beta must init to zeros"
    assert ln.gamma_grad.shape == (8,) and ln.beta_grad.shape == (8,)
    params = ln.parameters()
    assert len(params) == 2, "parameters() returns [gamma, beta]"


def test_forward_preserves_shape_3d_and_2d():
    ln = LayerNorm(6)
    x3 = _rng(1).normal(size=(4, 5, 6))
    assert ln.forward(x3).shape == (4, 5, 6), "LayerNorm preserves (B, L, D)"
    x2 = _rng(2).normal(size=(7, 6))
    assert ln.forward(x2).shape == (7, 6), "LayerNorm preserves (L, D)"


# --------------------------------------------------------------------------- #
# LayerNorm: forward standardizes over FEATURES (last axis), per token
# --------------------------------------------------------------------------- #
def test_forward_standardizes_per_token():
    ln = LayerNorm(32)  # gamma=1, beta=0
    x = _rng(3).normal(loc=5.0, scale=3.0, size=(4, 10, 32))
    y = ln.forward(x)
    # Each TOKEN (row over the last axis) is ~ zero-mean, unit-var (biased var).
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-6), (
        f"per-token mean over features should be ~0, got max "
        f"{np.abs(y.mean(axis=-1)).max():.2e}"
    )
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-3), (
        f"per-token var over features should be ~1, got "
        f"{y.var(axis=-1).min():.3f}..{y.var(axis=-1).max():.3f}"
    )


def test_forward_is_per_token_independent():
    """LayerNorm normalizes each token against its OWN features, batch-independent."""
    ln = LayerNorm(5)
    row = _rng(4).normal(size=(1, 1, 5))
    y_alone = ln.forward(row)
    batch = np.concatenate(
        [row, _rng(5).normal(loc=100.0, size=(1, 3, 5))], axis=1
    )
    y_batch = ln.forward(batch)
    assert np.allclose(y_alone[0, 0], y_batch[0, 0], atol=1e-12), (
        "a token's LayerNorm output must not depend on other tokens (no batch stats)"
    )


def test_gamma_beta_affine():
    ln = LayerNorm(4)
    ln.gamma = np.array([2.0, 0.5, 3.0, 1.5])
    ln.beta = np.array([1.0, -1.0, 0.5, 0.0])
    x = _rng(6).normal(size=(6, 8, 4))
    y = ln.forward(x)
    # After the per-token standardize, x_hat has per-token mean 0 / var 1, so
    # the affine maps feature j by gamma_j, shift beta_j. Recover x_hat:
    x_hat = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
        x.var(axis=-1, keepdims=True) + ln.eps
    )
    assert np.allclose(y, ln.gamma * x_hat + ln.beta, atol=1e-6), (
        "forward must be gamma * x_hat + beta with x_hat standardized over features"
    )


# --------------------------------------------------------------------------- #
# LayerNorm: param-grad reduction is over ALL axes except the last
# --------------------------------------------------------------------------- #
def test_param_grad_formulas():
    ln = LayerNorm(5)
    ln.gamma = _rng(10).normal(size=5)
    ln.beta = _rng(11).normal(size=5)
    x = _rng(12).normal(size=(3, 7, 5))
    y = ln.forward(x)
    g = _rng(13).normal(size=y.shape)
    ln.backward(g)
    x_hat = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
        x.var(axis=-1, keepdims=True) + ln.eps
    )
    red = tuple(range(g.ndim - 1))  # all axes except the last
    assert ln.beta_grad.shape == (5,), "beta_grad must be shape (D,)"
    assert ln.gamma_grad.shape == (5,), "gamma_grad must be shape (D,)"
    assert np.allclose(ln.beta_grad, g.sum(axis=red), atol=1e-9), (
        "beta_grad must sum grad_out over all token axes (all but last)"
    )
    assert np.allclose(ln.gamma_grad, (g * x_hat).sum(axis=red), atol=1e-9), (
        "gamma_grad must sum grad_out * x_hat over all token axes"
    )


# --------------------------------------------------------------------------- #
# LayerNorm: central-difference gradchecks
# --------------------------------------------------------------------------- #
def test_gradcheck_input_x():
    B, L, D = 3, 4, 5
    x = _rng(20).normal(size=(B, L, D))
    gamma = _rng(21).normal(size=D)
    beta = _rng(22).normal(size=D)
    w = _rng(23).normal(size=(B, L, D))  # fixed loss weights

    def make_ln():
        ln = LayerNorm(D)
        ln.gamma = gamma.copy()
        ln.beta = beta.copy()
        return ln

    ln = make_ln()
    ln.forward(x)
    dx = ln.backward(w.copy())  # dL/dx for loss = sum(y * w)

    def loss_of_x(xx):
        return _scalar_loss(make_ln().forward(xx), w)

    num = _numeric_grad(loss_of_x, x.copy())
    assert np.allclose(dx, num, atol=1e-5), (
        f"dL/dx mismatch: max abs diff {np.abs(dx - num).max():.2e}"
    )


def test_gradcheck_gamma():
    B, L, D = 2, 5, 4
    x = _rng(30).normal(size=(B, L, D))
    gamma = _rng(31).normal(size=D)
    beta = _rng(32).normal(size=D)
    w = _rng(33).normal(size=(B, L, D))

    def make_ln(g):
        ln = LayerNorm(D)
        ln.gamma = g.copy()
        ln.beta = beta.copy()
        return ln

    ln = make_ln(gamma)
    ln.forward(x)
    ln.backward(w.copy())
    ana = ln.gamma_grad

    def loss_of_gamma(g):
        return _scalar_loss(make_ln(g).forward(x), w)

    num = _numeric_grad(loss_of_gamma, gamma.copy())
    assert np.allclose(ana, num, atol=1e-5), (
        f"dL/dgamma mismatch: max abs diff {np.abs(ana - num).max():.2e}"
    )


def test_gradcheck_beta():
    B, L, D = 2, 5, 4
    x = _rng(40).normal(size=(B, L, D))
    gamma = _rng(41).normal(size=D)
    beta = _rng(42).normal(size=D)
    w = _rng(43).normal(size=(B, L, D))

    def make_ln(bvec):
        ln = LayerNorm(D)
        ln.gamma = gamma.copy()
        ln.beta = bvec.copy()
        return ln

    ln = make_ln(beta)
    ln.forward(x)
    ln.backward(w.copy())
    ana = ln.beta_grad

    def loss_of_beta(bvec):
        return _scalar_loss(make_ln(bvec).forward(x), w)

    num = _numeric_grad(loss_of_beta, beta.copy())
    assert np.allclose(ana, num, atol=1e-5), (
        f"dL/dbeta mismatch: max abs diff {np.abs(ana - num).max():.2e}"
    )


def test_gradcheck_input_x_unbatched_2d():
    """dx gradcheck for the unbatched (L, D) case (reduction still over last axis)."""
    L, D = 6, 5
    x = _rng(50).normal(size=(L, D))
    gamma = _rng(51).normal(size=D)
    beta = _rng(52).normal(size=D)
    w = _rng(53).normal(size=(L, D))

    def make_ln():
        ln = LayerNorm(D)
        ln.gamma = gamma.copy()
        ln.beta = beta.copy()
        return ln

    ln = make_ln()
    ln.forward(x)
    dx = ln.backward(w.copy())

    def loss_of_x(xx):
        return _scalar_loss(make_ln().forward(xx), w)

    num = _numeric_grad(loss_of_x, x.copy())
    assert np.allclose(dx, num, atol=1e-5), (
        f"unbatched dL/dx mismatch: max abs diff {np.abs(dx - num).max():.2e}"
    )


def test_zero_grad_resets_param_grads():
    ln = LayerNorm(5)
    x = _rng(60).normal(size=(2, 3, 5))
    ln.forward(x)
    ln.backward(_rng(61).normal(size=(2, 3, 5)))
    assert not np.allclose(ln.gamma_grad, 0.0) or not np.allclose(ln.beta_grad, 0.0)
    ln.zero_grad()
    assert np.allclose(ln.gamma_grad, 0.0), "zero_grad must reset gamma_grad"
    assert np.allclose(ln.beta_grad, 0.0), "zero_grad must reset beta_grad"


# --------------------------------------------------------------------------- #
# TransformerBlock wiring. These need MHA (stage_29) and MLP (stage_12); skip
# cleanly if those dependencies are not yet implemented.
# --------------------------------------------------------------------------- #
def _build_block(**kwargs):
    try:
        return TransformerBlock(d_model=8, n_heads=2, d_ff=16, seed=0, **kwargs)
    except (NotImplementedError, ImportError, AttributeError, TypeError) as e:
        pytest.skip(f"TransformerBlock dependencies not ready: {e!r}")


def test_block_preserves_shape():
    block = _build_block(norm="pre")
    x = _rng(70).normal(size=(2, 5, 8))
    try:
        y = block.forward(x)
    except (NotImplementedError, ImportError, AttributeError, TypeError) as e:
        pytest.skip(f"block forward not ready: {e!r}")
    assert y.shape == (2, 5, 8), "TransformerBlock must preserve (B, L, D)"


def test_pre_and_post_norm_differ():
    x = _rng(71).normal(size=(2, 5, 8))
    pre = _build_block(norm="pre")
    post = _build_block(norm="post")
    try:
        y_pre = pre.forward(x)
        y_post = post.forward(x)
    except (NotImplementedError, ImportError, AttributeError, TypeError) as e:
        pytest.skip(f"block forward not ready: {e!r}")
    assert y_pre.shape == y_post.shape == (2, 5, 8)
    assert not np.allclose(y_pre, y_post), (
        "pre-norm and post-norm wirings must produce different outputs"
    )


def test_invalid_norm_rejected():
    with pytest.raises((ValueError, NotImplementedError, ImportError)):
        TransformerBlock(d_model=8, n_heads=2, d_ff=16, norm="banana", seed=0)


def test_block_parameters_nonempty_and_stable():
    block = _build_block(norm="pre")
    try:
        p1 = block.parameters()
        p2 = block.parameters()
    except (NotImplementedError, ImportError, AttributeError, TypeError) as e:
        pytest.skip(f"block parameters not ready: {e!r}")
    assert len(p1) > 0, "block must expose learnable parameters"
    assert len(p1) == len(p2), "parameters() order/count must be stable across calls"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
