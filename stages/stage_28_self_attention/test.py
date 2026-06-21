"""Tests for Stage 28: Self-Attention.

Checks the forward construction of single-head scaled dot-product self-attention
(shapes, row-stochastic attention weights, the 1/sqrt(d_k) scaling, the causal
mask) against plain-NumPy references, and gradient-checks a scalar loss w.r.t.
each projection weight and the input using central differences:

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

compared against the analytic gradient produced by ``Tensor.backward()`` from
stage_09. If stage_09's ``Tensor`` or this stage's ``SelfAttention`` is not
implemented yet, the suite skips cleanly instead of erroring.

Run with:  pytest stage_28_self_attention/test.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import the things under test, skipping cleanly if not ready yet. --------
try:
    from code import SelfAttention, Tensor, causal_mask
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_28 self-attention / stage_09 Tensor not importable yet: {exc}",
        allow_module_level=True,
    )

RNG = np.random.default_rng(28)
EPS = 1e-6
ATOL = 1e-5
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Underlying ndarray of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def try_call(fn, *a, **k):
    """Run ``fn``; skip the test if the skeleton is still unimplemented."""
    try:
        return fn(*a, **k)
    except NotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"not implemented yet: {exc}")


def central_diff(f, x, eps=EPS):
    """Numerical gradient of scalar-valued f at numpy point x (any shape)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = float(f(x))
        x[idx] = orig - eps
        fm = float(f(x))
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def numpy_softmax_rows(s):
    """Reference stable row-wise softmax (NumPy)."""
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def numpy_attention(x, Wq, Wk, Wv, causal):
    """Reference forward pass entirely in NumPy."""
    d_k = Wq.shape[1]
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    S = (Q @ K.T) / np.sqrt(d_k)
    if causal:
        S = S + causal_mask(x.shape[0])
    A = numpy_softmax_rows(S)
    return A, A @ V


def make_module(d_model=6, d_k=4, causal=False, seed=0):
    return try_call(SelfAttention, d_model, d_k, causal=causal, seed=seed)


# ---------------------------------------------------------------------------
# causal_mask
# ---------------------------------------------------------------------------
def test_causal_mask_structure():
    M = try_call(causal_mask, 5)
    M = np.asarray(M, dtype=float)
    assert M.shape == (5, 5), "causal_mask must be (T, T)"
    # on/below diagonal == 0
    assert np.allclose(np.tril(M), 0.0), "lower triangle (incl diag) must be 0"
    # strictly above diagonal is a large negative number
    upper = M[np.triu_indices(5, k=1)]
    assert np.all(upper < -1e6), "strictly-upper entries must be very negative"


# ---------------------------------------------------------------------------
# forward shapes & weights
# ---------------------------------------------------------------------------
def test_param_shapes():
    sa = make_module(d_model=6, d_k=4, seed=1)
    params = try_call(sa.parameters)
    assert len(params) == 3, "parameters() must return [W_q, W_k, W_v]"
    for p in params:
        assert as_array(p).shape == (6, 4), "each weight must be (d_model, d_k)"


def test_forward_output_shape():
    T, d_model, d_k = 5, 6, 4
    sa = make_module(d_model=d_model, d_k=d_k, seed=2)
    x = RNG.standard_normal((T, d_model))
    o = try_call(sa.forward, x)
    assert as_array(o).shape == (T, d_k), "output O must be (T, d_k)"


def test_attention_matrix_shape_and_rows_sum_to_one():
    T, d_model, d_k = 7, 6, 4
    sa = make_module(d_model=d_model, d_k=d_k, seed=3)
    x = RNG.standard_normal((T, d_model))
    A = try_call(sa.attention_weights, x)
    A = np.asarray(A, dtype=float)
    assert A.shape == (T, T), "attention matrix A must be (T, T)"
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-8), "each row of A must sum to 1"
    assert np.all(A >= -1e-12), "attention weights must be non-negative"


def test_seed_reproducible():
    a = make_module(d_model=6, d_k=4, seed=42)
    b = make_module(d_model=6, d_k=4, seed=42)
    for pa, pb in zip(try_call(a.parameters), try_call(b.parameters)):
        assert np.allclose(as_array(pa), as_array(pb)), "same seed -> same weights"


# ---------------------------------------------------------------------------
# numerical agreement with a pure-NumPy reference
# ---------------------------------------------------------------------------
def test_forward_matches_numpy_reference():
    T, d_model, d_k = 5, 6, 4
    sa = make_module(d_model=d_model, d_k=d_k, causal=False, seed=4)
    Wq, Wk, Wv = (as_array(p) for p in try_call(sa.parameters))
    x = RNG.standard_normal((T, d_model))
    A_ref, O_ref = numpy_attention(x, Wq, Wk, Wv, causal=False)
    O = try_call(sa.forward, x)
    assert np.allclose(as_array(O), O_ref, atol=1e-8), "output mismatch vs NumPy ref"
    assert np.allclose(as_array(sa.last_attn), A_ref, atol=1e-8), "A mismatch vs ref"


def test_scaling_by_sqrt_dk_is_present():
    """Reconstruct scores from public weights; only the 1/sqrt(d_k)-scaled
    version should reproduce the module's attention matrix."""
    T, d_model, d_k = 4, 5, 4
    sa = make_module(d_model=d_model, d_k=d_k, causal=False, seed=5)
    Wq, Wk, Wv = (as_array(p) for p in try_call(sa.parameters))
    x = RNG.standard_normal((T, d_model))
    A = np.asarray(try_call(sa.attention_weights, x), dtype=float)

    Q, K = x @ Wq, x @ Wk
    A_scaled = numpy_softmax_rows((Q @ K.T) / np.sqrt(d_k))
    A_unscaled = numpy_softmax_rows(Q @ K.T)
    assert np.allclose(A, A_scaled, atol=1e-8), "scores must be divided by sqrt(d_k)"
    assert not np.allclose(A, A_unscaled, atol=1e-6), "scaling appears to be missing"


# ---------------------------------------------------------------------------
# causal mask behaviour
# ---------------------------------------------------------------------------
def test_causal_attention_is_lower_triangular():
    T, d_model, d_k = 6, 5, 4
    sa = make_module(d_model=d_model, d_k=d_k, causal=True, seed=6)
    x = RNG.standard_normal((T, d_model))
    A = np.asarray(try_call(sa.attention_weights, x), dtype=float)
    upper = A[np.triu_indices(T, k=1)]
    assert np.allclose(upper, 0.0, atol=1e-8), "future positions must get 0 weight"
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-8), "causal rows still sum to 1"
    # first query attends only to itself
    assert np.allclose(A[0, 0], 1.0, atol=1e-8), "row 0 must be all on position 0"


# ---------------------------------------------------------------------------
# gradient checks (central differences vs Tensor.backward)
# ---------------------------------------------------------------------------
def _scalar_loss_of_output(O):
    """Deterministic scalar reduction of the (T, d_k) output for gradcheck."""
    Od = as_array(O)
    # fixed, non-symmetric weights so the loss exercises every output element
    W = np.linspace(0.1, 1.0, Od.size).reshape(Od.shape)
    return W


def _run_and_loss(sa, x_arr):
    """Forward + weighted-sum scalar loss as a Tensor (graph intact)."""
    x = Tensor(x_arr)
    x.requires_grad = True if hasattr(x, "requires_grad") else None  # best-effort
    O = sa.forward(x)
    W = _scalar_loss_of_output(O)
    loss = (O * Tensor(W)).sum()
    return x, O, loss


@pytest.mark.parametrize("causal", [False, True])
def test_gradcheck_wrt_weights(causal):
    T, d_model, d_k = 5, 6, 4
    sa = make_module(d_model=d_model, d_k=d_k, causal=causal, seed=7)
    x_arr = RNG.standard_normal((T, d_model))
    Wq0, Wk0, Wv0 = (as_array(p).copy() for p in try_call(sa.parameters))

    # analytic grads via backward
    try_call(sa.zero_grad)
    _, _, loss = _run_and_loss(sa, x_arr)
    try_call(loss.backward)
    g_q = as_array(sa.W_q.grad if hasattr(sa.W_q, "grad") else sa.parameters()[0].grad)
    g_k = as_array(sa.parameters()[1].grad)
    g_v = as_array(sa.parameters()[2].grad)

    names = ["W_q", "W_k", "W_v"]
    base = [Wq0, Wk0, Wv0]
    analytic = [g_q, g_k, g_v]

    for which in range(3):
        def f(Wvar, which=which):
            s2 = SelfAttention(d_model, d_k, causal=causal, seed=7)
            mats = [Wq0, Wk0, Wv0]
            mats[which] = Wvar
            s2.W_q, s2.W_k, s2.W_v = (Tensor(m) for m in mats)
            O = s2.forward(x_arr)
            return float((as_array(O) * _scalar_loss_of_output(O)).sum())

        num = central_diff(f, base[which])
        assert np.allclose(num, analytic[which], atol=ATOL, rtol=RTOL), (
            f"gradcheck failed for {names[which]} (causal={causal}):\n"
            f"max abs diff = {np.max(np.abs(num - analytic[which])):.3e}"
        )


@pytest.mark.parametrize("causal", [False, True])
def test_gradcheck_wrt_input(causal):
    T, d_model, d_k = 5, 6, 4
    sa = make_module(d_model=d_model, d_k=d_k, causal=causal, seed=8)
    x_arr = RNG.standard_normal((T, d_model))

    # analytic grad on the input Tensor
    try_call(sa.zero_grad)
    x = Tensor(x_arr)
    O = try_call(sa.forward, x)
    loss = (O * Tensor(_scalar_loss_of_output(O))).sum()
    try_call(loss.backward)
    g_x = as_array(x.grad)

    def f(xvar):
        O = sa.forward(xvar)
        return float((as_array(O) * _scalar_loss_of_output(O)).sum())

    num = central_diff(f, x_arr)
    assert np.allclose(num, g_x, atol=ATOL, rtol=RTOL), (
        f"gradcheck failed for input x (causal={causal}):\n"
        f"max abs diff = {np.max(np.abs(num - g_x)):.3e}"
    )
