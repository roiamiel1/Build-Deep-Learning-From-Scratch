"""Tests for Stage 34: Capstone -- tiny Transformer (char-level LM).

Verifies the end-to-end integration: the tokenizer round-trips, ``get_batch``
produces correctly-shifted (X, Y) windows, the model forward returns
``(B, L, V)`` logits, ``lm_loss`` is a scalar ~ ``ln(V)`` for a fresh model, the
loss DROPS below that baseline after a short training run, and ``sample`` emits
in-vocabulary text. The gradient check uses central differences,

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

against the analytic gradient from ``Tensor.backward()`` (stage_08), applied to
the output-head weights of the model. Because the capstone leans on stages
09/13/28/30/32, any test whose dependencies are still skeletons SKIPS cleanly
instead of erroring.

Run with:  pytest stage_34_capstone_transformer/test.py
"""
import os as _os
import sys as _sys

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# Make the shared `dlfs` shim importable (it lives at the curriculum root).
sys.path.insert(0, _ROOT)

# --- Import things under test, skipping cleanly if not ready. ----------------
try:
    # --- resolve sibling code.py (avoid stdlib `code` collision) ---
    import importlib.util as _ilu
    _THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_THIS_DIR)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    _spec = _ilu.spec_from_file_location(
        "code", _os.path.join(_THIS_DIR, "code.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["code"] = _mod
    _spec.loader.exec_module(_mod)
    from code import (
        CharTokenizer,
        TransformerLM,
        Stage11_Tensor,
        get_batch,
        lm_loss,
        sample,
        train_lm,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_34 capstone / a prior stage not importable yet: {exc}",
        allow_module_level=True,
    )

if Stage11_Tensor is None:  # pragma: no cover
    pytest.skip("stage_11 Tensor not available", allow_module_level=True)

RNG = np.random.default_rng(35)
EPS = 1e-5
ATOL = 1e-4
RTOL = 1e-3

# A tiny, highly repetitive corpus -- a small Transformer can memorize it fast.
CORPUS = ("hello world. " * 40) + ("the quick brown fox. " * 40)

# Small model config so the suite is fast.
D_MODEL = 16
N_HEADS = 2
D_FF = 32
N_LAYERS = 2
BLOCK = 16


# --- helpers -----------------------------------------------------------------
def as_array(t):
    """Underlying ndarray of a Tensor (arrays pass through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def try_call(fn, *a, **k):
    """Run ``fn``; skip the test if the skeleton is still unimplemented."""
    try:
        return fn(*a, **k)
    except NotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"not implemented yet: {exc}")


def central_diff(f, x, eps=EPS):
    """Numerical gradient of scalar-valued ``f`` at numpy point ``x``."""
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


def make_tokenizer():
    return try_call(CharTokenizer, CORPUS)


def make_model(tok, seed=0):
    return try_call(
        TransformerLM,
        tok.vocab_size,
        D_MODEL,
        N_HEADS,
        D_FF,
        N_LAYERS,
        BLOCK,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def test_tokenizer_roundtrip():
    tok = make_tokenizer()
    s = "hello world."
    ids = try_call(tok.encode, s)
    ids = np.asarray(ids)
    assert ids.ndim == 1 and ids.shape[0] == len(s), "encode must give (len(s),) ids"
    assert ids.dtype.kind in "iu", "encoded ids must be integers"
    assert try_call(tok.decode, ids) == s, "decode(encode(s)) must round-trip"


def test_tokenizer_vocab_is_sorted_unique():
    tok = make_tokenizer()
    V = tok.vocab_size
    assert V == len(set(CORPUS)), "vocab_size must equal #unique chars"
    assert list(tok.chars) == sorted(set(CORPUS)), "vocab must be sorted-unique"
    ids = np.asarray(try_call(tok.encode, CORPUS))
    assert ids.min() >= 0 and ids.max() < V, "all ids must lie in [0, V)"


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------
def test_get_batch_shapes_and_shift():
    tok = make_tokenizer()
    data = np.asarray(try_call(tok.encode, CORPUS))
    rng = np.random.default_rng(1)
    X, Y = try_call(get_batch, data, BLOCK, 8, rng=rng)
    X, Y = np.asarray(X), np.asarray(Y)
    assert X.shape == (8, BLOCK), "X must be (B, L)"
    assert Y.shape == (8, BLOCK), "Y must be (B, L)"
    assert X.dtype.kind in "iu" and Y.dtype.kind in "iu", "ids must be integer"
    # Y is X shifted by one: reconstruct each window's start and compare.
    for b in range(X.shape[0]):
        # find a start index whose window matches row b of X
        # (windows are contiguous, so Y[b, :-1] must equal X[b, 1:])
        assert np.array_equal(Y[b, :-1], X[b, 1:]), (
            "Y must be the next-token shift of X (Y[:-1] == X[1:])"
        )


# ---------------------------------------------------------------------------
# forward shapes & baseline loss
# ---------------------------------------------------------------------------
def test_forward_logits_shape():
    tok = make_tokenizer()
    model = make_model(tok, seed=2)
    data = np.asarray(try_call(tok.encode, CORPUS))
    X, _ = try_call(get_batch, data, BLOCK, 4, rng=np.random.default_rng(2))
    logits = try_call(model.forward, X)
    assert as_array(logits).shape == (4, BLOCK, tok.vocab_size), (
        "forward must return (B, L, V) logits"
    )


def test_fresh_loss_near_ln_vocab():
    tok = make_tokenizer()
    model = make_model(tok, seed=3)
    data = np.asarray(try_call(tok.encode, CORPUS))
    X, Y = try_call(get_batch, data, BLOCK, 8, rng=np.random.default_rng(3))
    logits = try_call(model.forward, X)
    loss = try_call(lm_loss, logits, Y)
    val = float(as_array(loss))
    base = float(np.log(tok.vocab_size))
    assert as_array(loss).shape == (), "lm_loss must be a scalar Tensor"
    assert abs(val - base) < 0.6 * base, (
        f"fresh model loss {val:.3f} should be near ln(V)={base:.3f}"
    )


def test_parameters_unique_and_nonempty():
    tok = make_tokenizer()
    model = make_model(tok, seed=4)
    params = try_call(model.parameters)
    assert len(params) > 0, "parameters() must be non-empty"
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids)), "each parameter must appear exactly once"


# ---------------------------------------------------------------------------
# gradient check (central differences vs Tensor.backward) on the head weights
# ---------------------------------------------------------------------------
def _head_weight_param(model):
    """Return the output head's weight Parameter (the (D, V) projection)."""
    head = getattr(model, "head", None)
    if head is None:
        pytest.skip("model has no .head attribute to gradcheck")
    # The Linear head exposes its params via parameters(); the weight is the
    # 2-D one whose last axis is the vocab size.
    for p in try_call(head.parameters):
        if as_array(p).ndim == 2:
            return p
    pytest.skip("could not locate a 2-D head weight to gradcheck")


def test_gradcheck_head_weights():
    tok = make_tokenizer()
    model = make_model(tok, seed=5)
    data = np.asarray(try_call(tok.encode, CORPUS))
    X, Y = try_call(get_batch, data, BLOCK, 2, rng=np.random.default_rng(5))

    W = _head_weight_param(model)
    W0 = as_array(W).copy()

    # analytic grad via backward
    try_call(model.zero_grad) if hasattr(model, "zero_grad") else None
    for p in try_call(model.parameters):
        if hasattr(p, "grad"):
            p.grad = np.zeros_like(as_array(p))
    loss = try_call(lm_loss, try_call(model.forward, X), Y)
    try_call(loss.backward)
    analytic = as_array(W.grad)

    def f(Wvar):
        W.data = np.asarray(Wvar, dtype=float)
        out = lm_loss(model.forward(X), Y)
        return float(as_array(out))

    num = central_diff(f, W0)
    W.data = W0  # restore

    assert np.allclose(num, analytic, atol=ATOL, rtol=RTOL), (
        "head-weight gradcheck failed:\n"
        f"max abs diff = {np.max(np.abs(num - analytic)):.3e}"
    )


# ---------------------------------------------------------------------------
# training reduces the loss; sampling is in-vocabulary
# ---------------------------------------------------------------------------
def test_training_reduces_loss():
    tok = make_tokenizer()
    model = make_model(tok, seed=6)
    data = np.asarray(try_call(tok.encode, CORPUS))
    base = float(np.log(tok.vocab_size))
    history = try_call(
        train_lm,
        model,
        data,
        block_size=BLOCK,
        batch_size=16,
        steps=120,
        lr=3e-3,
        seed=6,
    )
    assert len(history) == 120, "history must have one entry per step"
    start = float(np.mean(history[:5]))
    end = float(np.mean(history[-5:]))
    assert end < start, f"training must reduce loss (start {start:.3f} -> end {end:.3f})"
    assert end < base, (
        f"trained loss {end:.3f} must fall below ln(V)={base:.3f} baseline"
    )


def test_sample_in_vocabulary():
    tok = make_tokenizer()
    model = make_model(tok, seed=7)
    out = try_call(
        sample,
        model,
        tok,
        prompt="hello",
        max_new_tokens=30,
        block_size=BLOCK,
        temperature=1.0,
        rng=np.random.default_rng(7),
    )
    assert isinstance(out, str), "sample must return a str"
    assert out.startswith("hello"), "sample must keep the prompt as a prefix"
    assert len(out) == len("hello") + 30, "sample must add max_new_tokens chars"
    vocab = set(tok.chars)
    assert set(out) <= vocab, "every generated character must be in the vocabulary"
