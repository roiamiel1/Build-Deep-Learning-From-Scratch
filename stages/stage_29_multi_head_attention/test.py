"""Tests for Stage 29: Multi-Head Attention.

MHA is built ON stage_28's single-head ``SelfAttention`` plus a learned output
projection ``W_o``; every gradient flows through ``Tensor.backward()`` (stage_09),
so we gradient-check a scalar loss against central differences:

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

We check forward shape, that ``h=1`` reduces to one ``SelfAttention`` head
followed by ``W_o``, that a bad head count raises ``ValueError``, and that the
analytic gradients for ``W_o``, each head's ``W_q/W_k/W_v``, and the input ``x``
match the numerical ones. If stage_09's ``Tensor`` / stage_28's ``SelfAttention``
or this stage's ``MultiHeadAttention`` is not implemented yet, the suite skips
cleanly instead of erroring.

Run with:  pytest stage_29_multi_head_attention/test.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import the things under test, skipping cleanly if not ready yet. --------
try:
    from code import MHA, MultiHeadAttention, SelfAttention, Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_29 MHA / stage_28 SelfAttention / stage_09 Tensor not "
        f"importable yet: {exc}",
        allow_module_level=True,
    )

RNG = np.random.default_rng(29)
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


def make_mha(d_model=12, h=4, causal=False, seed=0):
    return try_call(MultiHeadAttention, d_model, h, causal=causal, seed=seed)


# --- Construction ------------------------------------------------------------
def test_alias_is_same_class():
    assert MHA is MultiHeadAttention, "MHA must alias MultiHeadAttention"


def test_bad_head_count_raises():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=10, h=3)  # 10 not divisible by 3


def test_holds_h_self_attention_heads():
    mha = make_mha(d_model=12, h=4, seed=1)
    assert len(mha.heads) == 4, "must build one head per h"
    assert all(isinstance(hd, SelfAttention) for hd in mha.heads), (
        "MHA must reuse stage_28 SelfAttention heads (not reimplement attention)"
    )
    assert mha.d_k == 3, "d_k must be d_model // h"


# --- Forward -----------------------------------------------------------------
def test_forward_shape():
    T, d_model, h = 5, 12, 4
    mha = make_mha(d_model, h, seed=2)
    x = RNG.standard_normal((T, d_model))
    out = try_call(mha.forward, x)
    assert as_array(out).shape == (T, d_model), "forward must preserve (T, d_model)"


def test_call_matches_forward():
    T, d_model, h = 4, 8, 2
    mha = make_mha(d_model, h, seed=3)
    x = RNG.standard_normal((T, d_model))
    a = as_array(try_call(mha.forward, x))
    b = as_array(try_call(mha.__call__, x))
    assert np.allclose(a, b, atol=1e-12), "__call__ must equal forward"


def test_parameters_count():
    d_model, h = 12, 4
    mha = make_mha(d_model, h, seed=4)
    params = try_call(mha.parameters)
    # 3 weights per head (W_q, W_k, W_v) + 1 output projection W_o.
    assert len(params) == 3 * h + 1, "parameters() = per-head W_q/W_k/W_v + W_o"


def test_h1_reduces_to_single_head_plus_Wo():
    """With h=1, MHA == one SelfAttention head's output projected by W_o."""
    T, d_model = 5, 6
    mha = make_mha(d_model, h=1, seed=5)
    x = RNG.standard_normal((T, d_model))
    out = as_array(try_call(mha.forward, x))
    head_out = as_array(try_call(mha.heads[0].forward, x))  # (T, d_model)  d_k == d_model
    ref = head_out @ as_array(mha.W_o)
    assert np.allclose(out, ref, atol=ATOL), "h=1 must equal head(x) @ W_o"


# --- Gradient checks (central differences vs Tensor.backward) ----------------
def _scalar_loss_of_output(O):
    """A deterministic scalar reduction of the (T, d_model) output Tensor."""
    return O.sum() if hasattr(O, "sum") else float(np.sum(as_array(O)))


def test_gradcheck_weights():
    T, d_model, h = 4, 8, 2
    x = RNG.standard_normal((T, d_model))

    mha = make_mha(d_model, h, seed=7)
    out = try_call(mha.forward, x)
    loss = _scalar_loss_of_output(out)
    try_call(mha.zero_grad)
    try_call(loss.backward)

    params = mha.parameters()  # [h0:Wq,Wk,Wv, h1:Wq,Wk,Wv, ..., W_o]
    analytic = [as_array(p.grad) for p in params]

    def loss_for_param(i):
        def f(p_arr):
            p_arr = np.asarray(p_arr, dtype=float)
            orig = params[i].data
            params[i].data = p_arr
            try:
                o = mha.forward(x)
                return float(as_array(o).sum())
            finally:
                params[i].data = orig
        return f

    for i, p in enumerate(params):
        num = central_diff(loss_for_param(i), as_array(p))
        assert np.allclose(num, analytic[i], atol=ATOL, rtol=RTOL), (
            f"param[{i}] gradcheck mismatch: "
            f"max|num-analytic|={np.max(np.abs(num - analytic[i])):.2e}"
        )


def test_gradcheck_input():
    T, d_model, h = 4, 8, 2
    mha = make_mha(d_model, h, seed=8)
    x_arr = RNG.standard_normal((T, d_model))

    x = Tensor(x_arr)
    if hasattr(x, "requires_grad"):
        x.requires_grad = True
    out = try_call(mha.forward, x)
    loss = _scalar_loss_of_output(out)
    try_call(mha.zero_grad)
    try_call(loss.backward)
    g_x = as_array(x.grad)

    def f(xx):
        return float(as_array(mha.forward(Tensor(np.asarray(xx, dtype=float)))).sum())

    num = central_diff(f, x_arr)
    assert np.allclose(num, g_x, atol=ATOL, rtol=RTOL), (
        f"input gradcheck mismatch: max|num-analytic|={np.max(np.abs(num - g_x)):.2e}"
    )


def test_causal_attention_is_lower_triangular():
    """With causal=True, every head's attention map is lower-triangular."""
    T, d_model, h = 4, 8, 2
    mha = make_mha(d_model, h, causal=True, seed=9)
    x = RNG.standard_normal((T, d_model))
    maps = try_call(mha.attention_weights, x)
    upper = np.triu_indices(T, k=1)
    for A in maps:
        A = as_array(A)
        assert np.allclose(A[upper], 0.0, atol=1e-8), "future positions must be 0"
        assert np.allclose(A.sum(axis=1), 1.0, atol=1e-8), "rows must sum to 1"
