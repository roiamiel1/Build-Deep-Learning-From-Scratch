"""Tests for Stage 22: Dropout.

Covers inverted dropout on top of the ``Tensor`` from stage_09 and the ``MLP``
from stage_12:

  * eval mode is the exact identity (no sampling, no scaling, deterministic);
  * train mode zeroes a ~``1 - p_keep`` fraction of elements and scales the
    survivors by ``1 / p_keep`` (inverted scaling);
  * over many draws the train-mode mean ~= the input (expectation preserved);
  * central-difference gradcheck of ``dL/dx`` with the mask HELD FIXED:
        df/dx ~= (f(x + eps) - f(x - eps)) / (2 * eps)
    where dropped coords -> 0 and kept coords -> 1 / p_keep;
  * ``MLPDropout`` train/eval mode wiring and parameter collection.

To hold the random mask fixed across finite-difference probes we re-seed the
layer's RNG to the same value before every forward call, so the SAME Bernoulli
mask is drawn each time (the analytic backward then matches the numeric slope).

If an earlier stage is not yet implemented, the suite skips rather than
erroring. Run with:  pytest stage_22_dropout/test.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import the things under test, skipping cleanly if not ready yet. --------
try:
    from code import Dropout, MLPDropout, Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_22 Dropout / stage_12 MLP / stage_09 Tensor not importable yet: {exc}",
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
    """Build a Tensor from a numpy array, tolerating different ctor kwargs."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=True)
    except TypeError:
        return Tensor(arr)


def scalar_out(t):
    """Reduce a Tensor's data to a python float (sum) for finite-diff probing."""
    return float(np.sum(as_array(t)))


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


def reseed(layer, seed):
    """Force the dropout layer's RNG back to a known seed (same mask next draw)."""
    layer._rng = np.random.default_rng(seed)


# --- Construction & validation -----------------------------------------------
def test_invalid_p_keep_raises():
    for bad in (0.0, -0.1, 1.5, 2.0):
        with pytest.raises(ValueError):
            Dropout(bad)


def test_p_keep_one_is_allowed():
    d = Dropout(1.0)
    assert d.p_keep == 1.0


def test_defaults_to_training_mode():
    d = Dropout(0.5)
    assert d.training is True, "Dropout should start in training mode"


def test_no_parameters():
    d = Dropout(0.5)
    assert list(d.parameters()) == [], "Dropout has no learnable parameters"


def test_train_eval_chainable():
    d = Dropout(0.5)
    assert d.eval() is d and d.training is False
    assert d.train() is d and d.training is True


def test_repr_mentions_p_keep():
    d = Dropout(0.3)
    assert "0.3" in repr(d) or "p_keep" in repr(d)


# --- Eval mode is the identity ----------------------------------------------
def test_eval_mode_is_identity():
    d = Dropout(0.5, seed=0).eval()
    x_np = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    y = d(make_tensor(x_np))
    assert np.allclose(as_array(y), x_np), "eval-mode dropout must return input unchanged"
    assert d.mask is None, "eval mode must not store a mask"


def test_eval_mode_is_deterministic():
    d = Dropout(0.5, seed=0).eval()
    x_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    a = as_array(d(make_tensor(x_np)))
    b = as_array(d(make_tensor(x_np)))
    assert np.allclose(a, b), "repeated eval calls must be identical"


def test_p_keep_one_train_is_identity():
    # With p_keep == 1, every unit is kept and scaled by 1/1 -> identity.
    d = Dropout(1.0, seed=3).train()
    x_np = np.array([1.0, -2.0, 3.0, -4.0])
    y = as_array(d(make_tensor(x_np)))
    assert np.allclose(y, x_np), "p_keep=1 train mode must be the identity"


# --- Train mode: masking + inverted scaling ---------------------------------
def test_train_mode_zeros_and_scales():
    p = 0.5
    d = Dropout(p, seed=7).train()
    x_np = np.full((50, 50), 2.0)
    y = as_array(d(make_tensor(x_np)))
    # Each output element is either 0 (dropped) or x / p (kept, inverted scale).
    kept_val = 2.0 / p
    is_zero = np.isclose(y, 0.0)
    is_scaled = np.isclose(y, kept_val)
    assert np.all(is_zero | is_scaled), (
        "every train-mode output must be 0 (dropped) or x / p_keep (kept)"
    )
    frac_dropped = is_zero.mean()
    assert abs(frac_dropped - (1 - p)) < 0.05, (
        f"~{1 - p:.2f} of units should be dropped, got {frac_dropped:.3f}"
    )


def test_mask_matches_output():
    d = Dropout(0.6, seed=11).train()
    x_np = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = as_array(d(make_tensor(x_np)))
    # self.mask holds the scale array s = m / p_keep; output = x * s.
    assert d.mask is not None, "train mode must store the sampled scale in self.mask"
    assert np.allclose(y, x_np * d.mask), "output must equal x * stored mask scale"


def test_inverted_scaling_preserves_mean():
    # Averaging many independent train-mode passes should recover the input
    # (E[m/p] = 1), which is the whole point of inverted dropout.
    p = 0.5
    d = Dropout(p, seed=123).train()
    x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = 20000
    acc = np.zeros_like(x_np)
    for _ in range(n):
        acc += as_array(d(make_tensor(x_np)))
    mean = acc / n
    assert np.allclose(mean, x_np, atol=0.1), (
        f"inverted dropout mean should ~= input; got {mean} vs {x_np}"
    )


def test_mask_resampled_each_call():
    d = Dropout(0.5, seed=5).train()
    x_np = np.ones((100,))
    a = as_array(d(make_tensor(x_np)))
    b = as_array(d(make_tensor(x_np)))
    assert not np.allclose(a, b), "train mode must resample the mask on each call"


# --- Gradient check: dL/dx with the mask held fixed --------------------------
@pytest.mark.parametrize("p_keep", [0.5, 0.7, 1.0])
def test_gradcheck_wrt_input(p_keep):
    seed = 42
    d = Dropout(p_keep, seed=seed).train()
    x_np = np.array([0.7, -1.3, 0.2, 2.5, -0.4])

    # Analytic gradient: one forward (fixing the mask via re-seed) + backward.
    reseed(d, seed)
    x = make_tensor(x_np)
    out = d(x)
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    g_analytic = as_array(x.grad)
    fixed_scale = np.asarray(d.mask, dtype=float).copy()  # m / p_keep, held fixed

    # Numeric gradient: re-seed before EVERY probe so the same mask is drawn.
    def f(xv):
        reseed(d, seed)
        return scalar_out(d(make_tensor(xv)))

    g_num = central_diff(f, x_np.copy())

    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"[p_keep={p_keep}] dL/dx mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )
    # Kept coords -> 1/p_keep, dropped coords -> 0.
    expected = fixed_scale  # since loss = sum(out), dL/dx = scale exactly
    assert np.allclose(g_analytic, expected, atol=ATOL), (
        f"[p_keep={p_keep}] expected dL/dx == mask scale; got {g_analytic} vs {expected}"
    )


def test_dropped_units_get_zero_gradient():
    p = 0.5
    seed = 99
    d = Dropout(p, seed=seed).train()
    x_np = np.linspace(-1.0, 1.0, 12)
    reseed(d, seed)
    x = make_tensor(x_np)
    out = d(x)
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    g = as_array(x.grad)
    scale = np.asarray(d.mask, dtype=float)
    dropped = np.isclose(scale, 0.0)
    assert np.allclose(g[dropped], 0.0), "dropped units must receive zero gradient"
    assert np.allclose(g[~dropped], 1.0 / p), "kept units must get gradient 1/p_keep"


# --- MLPDropout integration --------------------------------------------------
def test_mlpdropout_eval_shapes():
    net = MLPDropout([3, 5, 4, 2], p_keep=0.5, seed=0).eval()
    y1 = net(make_tensor([0.5, -1.0, 2.0]))
    assert as_array(y1).shape == (2,), "(n_in,) input -> (n_out,) output"
    X = make_tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])
    y2 = net(X)
    assert as_array(y2).shape == (2, 2), "(batch, n_in) input -> (batch, n_out)"


def test_mlpdropout_one_dropout_per_hidden_layer():
    net = MLPDropout([3, 5, 4, 2], p_keep=0.5, seed=0)
    # sizes has 4 entries -> 3 Dense layers -> 2 hidden layers -> 2 Dropouts.
    assert len(net.layers) == 3, "an MLP over [3,5,4,2] must have 3 Dense layers"
    assert len(net.dropouts) == 2, "one Dropout per HIDDEN layer (not the output)"


def test_mlpdropout_eval_is_deterministic_and_identity_dropout():
    net = MLPDropout([2, 8, 1], p_keep=0.5, seed=1).eval()
    x = np.array([0.3, -0.7])
    a = as_array(net(make_tensor(x)))
    b = as_array(net(make_tensor(x)))
    assert np.allclose(a, b), "eval-mode MLPDropout must be deterministic"


def test_mlpdropout_train_is_stochastic():
    net = MLPDropout([2, 16, 1], p_keep=0.5, seed=2).train()
    x = np.array([0.3, -0.7])
    a = as_array(net(make_tensor(x)))
    b = as_array(net(make_tensor(x)))
    assert not np.allclose(a, b), "train-mode MLPDropout should differ across calls"


def test_mlpdropout_mode_flips_every_dropout():
    net = MLPDropout([2, 4, 4, 1], p_keep=0.5, seed=0)
    net.eval()
    assert net.training is False
    assert all(not d.training for d in net.dropouts), "eval() must flip every Dropout"
    net.train()
    assert net.training is True
    assert all(d.training for d in net.dropouts), "train() must flip every Dropout"


def test_mlpdropout_parameters_are_dense_only():
    net = MLPDropout([3, 6, 6, 2], p_keep=0.5, seed=1)
    params = net.parameters()
    # 3 Dense layers, 2 params each (W, b) -> 6 parameter tensors; Dropout adds none.
    assert len(params) == 6, "parameters() must collect only Dense params (2 per layer)"
    expected = [p for layer in net.layers for p in layer.parameters()]
    for p, q in zip(params, expected):
        assert p is q, "parameters() must return the actual Dense parameter tensors"


def test_mlpdropout_eval_gradcheck_wrt_first_layer():
    # In eval mode dropout is identity, so the network is a plain MLP and the
    # weight gradients must match finite differences exactly.
    net = MLPDropout([3, 5, 2], p_keep=0.5, activation="tanh", seed=6).eval()
    X = np.array([[0.5, -1.0, 2.0], [1.0, 0.5, -0.5]])

    net.zero_grad()
    out = net(make_tensor(X))
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    W1 = net.layers[0].parameters()[0]
    g_analytic = as_array(W1.grad).copy()

    saved = as_array(W1).copy()

    def f(w):
        W1.data = w.copy().reshape(saved.shape)
        val = scalar_out(net(make_tensor(X)))
        W1.data = saved.copy()
        return val

    g_num = central_diff(f, saved.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"eval-mode dL/dW1 mismatch:\n analytic=\n{g_analytic}\n numeric =\n{g_num}"
    )
