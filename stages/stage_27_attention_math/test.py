"""Tests for Stage 27: Attention mathematics.

Checks the scaled dot-product attention forward against a naive per-query loop
reference, verifies softmax stability and the row-sum property, and gradient-
checks the analytic ``attention_backward`` against central differences:

    df/dx ~= (f(x + eps) - f(x - eps)) / (2 * eps)

The forward (``attention_forward``) is the numerical oracle that the central-
difference checks differentiate; the analytic gradients come from
``attention_backward``. If the stage is not implemented yet the suite skips
cleanly instead of erroring.

Run with:  pytest stage_27_attention_math/test.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import the things under test, skipping cleanly if not ready yet. --------
try:
    from code import (
        attention_backward,
        attention_forward,
        softmax_backward,
        softmax_rows,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_27 attention math not importable yet: {exc}",
        allow_module_level=True,
    )

RNG = np.random.default_rng(27)
EPS = 1e-6
ATOL = 1e-4
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def _ready(fn, *args, **kwargs):
    """Run ``fn``; skip the test if any required piece is still a stub."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"not implemented yet: {exc}")


def central_diff(f, x, eps=EPS):
    """Numerical gradient of scalar-valued ``f`` at numpy point ``x`` (any shape).

    Central differences: (f(x+eps) - f(x-eps)) / (2*eps), one entry at a time.
    """
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


def naive_attention(Q, K, V, mask=None):
    """Reference forward written as explicit per-query loops (unbatched 2-D)."""
    Lq, dk = Q.shape
    Lk, dv = V.shape
    O = np.zeros((Lq, dv), dtype=float)
    A = np.zeros((Lq, Lk), dtype=float)
    scale = 1.0 / np.sqrt(dk)
    for i in range(Lq):
        scores = np.array([np.dot(Q[i], K[j]) * scale for j in range(Lk)])
        if mask is not None:
            scores = scores + mask[i]
        m = np.max(scores)
        e = np.exp(scores - m)
        a = e / np.sum(e)
        A[i] = a
        O[i] = sum(a[j] * V[j] for j in range(Lk))
    return O, A


# =========================================================================== #
# softmax_rows
# =========================================================================== #
def test_softmax_rows_sum_to_one():
    x = RNG.normal(size=(5, 7))
    p = _ready(softmax_rows, x)
    assert np.allclose(p.sum(axis=-1), 1.0, atol=ATOL), f"rows sum: {p.sum(axis=-1)}"
    assert np.all(p >= 0.0), "softmax must be non-negative"


def test_softmax_rows_matches_reference():
    x = RNG.normal(size=(4, 6))
    p = _ready(softmax_rows, x)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    ref = e / e.sum(axis=-1, keepdims=True)
    assert np.allclose(p, ref, atol=ATOL), f"softmax mismatch:\n{p}\n{ref}"


def test_softmax_rows_stable_large_logits():
    x = np.array([[1000.0, 1001.0, 1002.0]])
    p = _ready(softmax_rows, x)
    assert np.all(np.isfinite(p)), "softmax overflowed on large logits"
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    ref = e / e.sum(axis=-1, keepdims=True)
    assert np.allclose(p, ref, atol=ATOL)


def test_softmax_rows_last_axis_batched():
    x = RNG.normal(size=(3, 4, 5))
    p = _ready(softmax_rows, x)
    assert p.shape == x.shape
    assert np.allclose(p.sum(axis=-1), 1.0, atol=ATOL), "batched rows must sum to 1"


# =========================================================================== #
# softmax_backward (gradcheck the standalone JVP)
# =========================================================================== #
def test_softmax_backward_gradcheck():
    x = RNG.normal(size=(4, 6))
    # arbitrary upstream weighting so the scalar loss has a non-trivial grad
    w = RNG.normal(size=(4, 6))

    def f(z):
        return float(np.sum(softmax_rows(z) * w))

    A = _ready(softmax_rows, x)
    dA = w  # d/dA of sum(A * w)
    dS_analytic = _ready(softmax_backward, dA, A)
    dS_num = central_diff(f, x.copy())
    assert np.allclose(dS_analytic, dS_num, atol=ATOL, rtol=RTOL), (
        f"softmax_backward gradcheck mismatch:\n analytic={dS_analytic}\n numeric ={dS_num}"
    )


def test_softmax_backward_constant_upstream_is_zero():
    """A uniform upstream gradient should produce ~0 dS (softmax is shift/scale invariant in that direction)."""
    x = RNG.normal(size=(3, 5))
    A = _ready(softmax_rows, x)
    dA = np.ones_like(A)  # constant per row
    dS = _ready(softmax_backward, dA, A)
    assert np.allclose(dS, 0.0, atol=ATOL), f"constant upstream should give 0 dS, got {dS}"


# =========================================================================== #
# attention_forward
# =========================================================================== #
def test_attention_forward_matches_naive_reference():
    Lq, Lk, dk, dv = 3, 4, 5, 6
    Q = RNG.normal(size=(Lq, dk))
    K = RNG.normal(size=(Lk, dk))
    V = RNG.normal(size=(Lk, dv))
    O, A = _ready(attention_forward, Q, K, V)
    O_ref, A_ref = naive_attention(Q, K, V)
    assert O.shape == (Lq, dv), f"O shape {O.shape}, want {(Lq, dv)}"
    assert A.shape == (Lq, Lk), f"A shape {A.shape}, want {(Lq, Lk)}"
    assert np.allclose(O, O_ref, atol=ATOL), f"forward O mismatch:\n{O}\n{O_ref}"
    assert np.allclose(A, A_ref, atol=ATOL), f"forward A mismatch:\n{A}\n{A_ref}"


def test_attention_weights_rows_sum_to_one():
    Q = RNG.normal(size=(3, 4))
    K = RNG.normal(size=(5, 4))
    V = RNG.normal(size=(5, 2))
    _, A = _ready(attention_forward, Q, K, V)
    assert np.allclose(A.sum(axis=-1), 1.0, atol=ATOL), f"weight rows: {A.sum(axis=-1)}"
    assert np.all(A >= 0.0), "attention weights must be non-negative"


def test_attention_uses_sqrt_dk_scale():
    """The score matrix feeding softmax must be Q@K.T / sqrt(d_k), not Q@K.T."""
    Lq, Lk, dk, dv = 2, 3, 8, 4
    Q = RNG.normal(size=(Lq, dk))
    K = RNG.normal(size=(Lk, dk))
    V = RNG.normal(size=(Lk, dv))
    _, A = _ready(attention_forward, Q, K, V)
    scaled = (Q @ K.T) / np.sqrt(dk)
    A_scaled = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    A_scaled = A_scaled / A_scaled.sum(axis=-1, keepdims=True)
    unscaled = Q @ K.T
    A_unscaled = np.exp(unscaled - unscaled.max(axis=-1, keepdims=True))
    A_unscaled = A_unscaled / A_unscaled.sum(axis=-1, keepdims=True)
    assert np.allclose(A, A_scaled, atol=ATOL), "weights do not match the 1/sqrt(d_k)-scaled softmax"
    assert not np.allclose(A_scaled, A_unscaled, atol=ATOL), (
        "test setup degenerate: scaled and unscaled softmax coincide"
    )


def test_attention_forward_stable_large_logits():
    Q = np.array([[100.0, 100.0]])
    K = np.array([[100.0, 100.0], [-100.0, -100.0]])
    V = np.array([[1.0], [2.0]])
    O, A = _ready(attention_forward, Q, K, V)
    assert np.all(np.isfinite(O)) and np.all(np.isfinite(A)), "attention overflowed"


def test_attention_forward_batched():
    B, Lq, Lk, dk, dv = 2, 3, 4, 5, 6
    Q = RNG.normal(size=(B, Lq, dk))
    K = RNG.normal(size=(B, Lk, dk))
    V = RNG.normal(size=(B, Lk, dv))
    O, A = _ready(attention_forward, Q, K, V)
    assert O.shape == (B, Lq, dv), f"batched O shape {O.shape}"
    assert A.shape == (B, Lq, Lk), f"batched A shape {A.shape}"
    # each batch element must match the unbatched reference
    for b in range(B):
        O_ref, A_ref = naive_attention(Q[b], K[b], V[b])
        assert np.allclose(O[b], O_ref, atol=ATOL), f"batch {b} O mismatch"
        assert np.allclose(A[b], A_ref, atol=ATOL), f"batch {b} A mismatch"


# =========================================================================== #
# attention_backward -- central-difference gradcheck
# =========================================================================== #
def _scalar_loss(Q, K, V, W, mask=None):
    """A scalar built from the attention output: L = sum(O * W)."""
    O, _ = attention_forward(Q, K, V, mask=mask)
    return float(np.sum(O * W))


def _analytic_grads(Q, K, V, W, mask=None):
    O, A = attention_forward(Q, K, V, mask=mask)
    dO = W  # d/dO of sum(O * W)
    return attention_backward(dO, Q, K, V, A, mask=mask)


def test_attention_backward_gradcheck_unbatched():
    Lq, Lk, dk, dv = 3, 4, 5, 6
    Q = RNG.normal(size=(Lq, dk))
    K = RNG.normal(size=(Lk, dk))
    V = RNG.normal(size=(Lk, dv))
    W = RNG.normal(size=(Lq, dv))

    dQ, dK, dV = _ready(_analytic_grads, Q, K, V, W)

    dQ_num = central_diff(lambda q: _scalar_loss(q, K, V, W), Q.copy())
    dK_num = central_diff(lambda k: _scalar_loss(Q, k, V, W), K.copy())
    dV_num = central_diff(lambda v: _scalar_loss(Q, K, v, W), V.copy())

    assert dQ.shape == Q.shape, f"dQ shape {dQ.shape}, want {Q.shape}"
    assert dK.shape == K.shape, f"dK shape {dK.shape}, want {K.shape}"
    assert dV.shape == V.shape, f"dV shape {dV.shape}, want {V.shape}"
    assert np.allclose(dQ, dQ_num, atol=ATOL, rtol=RTOL), (
        f"dQ gradcheck mismatch:\n analytic={dQ}\n numeric ={dQ_num}"
    )
    assert np.allclose(dK, dK_num, atol=ATOL, rtol=RTOL), (
        f"dK gradcheck mismatch:\n analytic={dK}\n numeric ={dK_num}"
    )
    assert np.allclose(dV, dV_num, atol=ATOL, rtol=RTOL), (
        f"dV gradcheck mismatch:\n analytic={dV}\n numeric ={dV_num}"
    )


def test_attention_backward_gradcheck_batched():
    B, Lq, Lk, dk, dv = 2, 2, 3, 4, 3
    Q = RNG.normal(size=(B, Lq, dk))
    K = RNG.normal(size=(B, Lk, dk))
    V = RNG.normal(size=(B, Lk, dv))
    W = RNG.normal(size=(B, Lq, dv))

    dQ, dK, dV = _ready(_analytic_grads, Q, K, V, W)

    dQ_num = central_diff(lambda q: _scalar_loss(q, K, V, W), Q.copy())
    dK_num = central_diff(lambda k: _scalar_loss(Q, k, V, W), K.copy())
    dV_num = central_diff(lambda v: _scalar_loss(Q, K, v, W), V.copy())

    assert np.allclose(dQ, dQ_num, atol=ATOL, rtol=RTOL), "batched dQ gradcheck mismatch"
    assert np.allclose(dK, dK_num, atol=ATOL, rtol=RTOL), "batched dK gradcheck mismatch"
    assert np.allclose(dV, dV_num, atol=ATOL, rtol=RTOL), "batched dV gradcheck mismatch"


# =========================================================================== #
# masking
# =========================================================================== #
def test_mask_zeros_out_weights():
    """A -inf mask on a key must drive its attention weight to ~0."""
    Lq, Lk, dk, dv = 2, 4, 5, 3
    Q = RNG.normal(size=(Lq, dk))
    K = RNG.normal(size=(Lk, dk))
    V = RNG.normal(size=(Lk, dv))
    mask = np.zeros((Lq, Lk))
    mask[:, 0] = -np.inf  # forbid key 0 for every query
    _, A = _ready(attention_forward, Q, K, V, mask=mask)
    assert np.allclose(A[:, 0], 0.0, atol=ATOL), f"masked key should get ~0 weight, got {A[:, 0]}"
    assert np.allclose(A.sum(axis=-1), 1.0, atol=ATOL), "rows must still sum to 1 under masking"


def test_mask_gradcheck():
    """With a finite-but-large negative mask, gradients still match central differences."""
    Lq, Lk, dk, dv = 3, 4, 5, 4
    Q = RNG.normal(size=(Lq, dk))
    K = RNG.normal(size=(Lk, dk))
    V = RNG.normal(size=(Lk, dv))
    W = RNG.normal(size=(Lq, dv))
    mask = np.zeros((Lq, Lk))
    mask[0, 1] = -1e9  # large finite negative so f(x+-eps) stays finite

    dQ, dK, dV = _ready(_analytic_grads, Q, K, V, W, mask=mask)

    dQ_num = central_diff(lambda q: _scalar_loss(q, K, V, W, mask=mask), Q.copy())
    dK_num = central_diff(lambda k: _scalar_loss(Q, k, V, W, mask=mask), K.copy())
    dV_num = central_diff(lambda v: _scalar_loss(Q, K, v, W, mask=mask), V.copy())

    assert np.allclose(dQ, dQ_num, atol=ATOL, rtol=RTOL), "masked dQ gradcheck mismatch"
    assert np.allclose(dK, dK_num, atol=ATOL, rtol=RTOL), "masked dK gradcheck mismatch"
    assert np.allclose(dV, dV_num, atol=ATOL, rtol=RTOL), "masked dV gradcheck mismatch"
