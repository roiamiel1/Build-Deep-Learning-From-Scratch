"""Tests for Stage 16: Weight initialization.

What these tests check:
  * Each initializer returns the right shape and an EMPIRICAL variance that
    matches its target Var(W) from the variance-propagation derivation, and is
    reproducible under a fixed seed.
  * The measurement harness ``forward_activation_stats`` reproduces the known
    failure / success modes:
        - matched init (Xavier+tanh, He+relu) keeps per-layer std stable,
        - a tiny init makes activation std collapse toward 0 with depth
          (vanishing signal),
        - a large init drives tanh into saturation and ReLU units dead.
  * A freshly (He/Xavier) initialized ``stage_11`` ``Dense`` still gradchecks:
    init only sets the STARTING weights, the layer's backward is unchanged. We
    compare central differences against the autodiff ``.grad``:

        df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps).

Tests that need the (skeleton) ``Dense`` / ``Tensor`` from sibling stages skip
cleanly until those are implemented, so this suite is runnable incrementally.

Run with:  pytest stage_16_weight_initialization/test.py
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # curriculum root, so `import dlfs` works
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

# --- Import things under test, skipping cleanly if not ready yet. ------------
# code.py pulls Dense (stage_11) and Tensor (stage_09) via dlfs.stage_import;
# importing it here runs those imports (they must succeed while this stage's
# initializer / harness bodies are still skeletons).
try:
    from code import (
        xavier_uniform,
        xavier_normal,
        he_normal,
        he_uniform,
        init_dense,
        forward_activation_stats,
        Dense,
        Tensor,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_16 initializers (or stage_11/stage_09 deps) not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- helpers -----------------------------------------------------------------
def as_array(t):
    """Underlying numpy array of a Tensor (or pass arrays straight through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def make_tensor(arr, requires_grad=True):
    """Build a Tensor, tolerating constructors with or without requires_grad."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=requires_grad)
    except TypeError:
        return Tensor(arr)


def scalar_out(t):
    """Reduce a Tensor's output to a python float (sum) for finite-diff probing."""
    return float(np.sum(as_array(t)))


def central_diff(f, x, eps=EPS):
    """Central-difference gradient of scalar-valued f at numpy point x."""
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


def skip_if_unimplemented(fn, *args, **kwargs):
    """Call fn; skip the test if it (or a dependency) is still a skeleton."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"depends on unimplemented code: {exc}")


# ============================================================================
# Initializer shapes
# ============================================================================
@pytest.mark.parametrize(
    "fn", [xavier_uniform, xavier_normal, he_normal, he_uniform]
)
def test_init_shape(fn):
    W = skip_if_unimplemented(fn, 7, 5, seed=0)
    W = np.asarray(W)
    assert W.shape == (7, 5), f"{fn.__name__} must return shape (n_in, n_out)"
    assert W.dtype == np.float64, f"{fn.__name__} must return float64"


# ============================================================================
# Reproducibility with a fixed seed
# ============================================================================
@pytest.mark.parametrize(
    "fn", [xavier_uniform, xavier_normal, he_normal, he_uniform]
)
def test_init_reproducible(fn):
    a = skip_if_unimplemented(fn, 16, 12, seed=42)
    b = skip_if_unimplemented(fn, 16, 12, seed=42)
    assert np.allclose(a, b), f"{fn.__name__}: same seed must give same weights"
    c = skip_if_unimplemented(fn, 16, 12, seed=43)
    assert not np.allclose(a, c), f"{fn.__name__}: different seed must differ"


# ============================================================================
# Empirical variance matches the derived target Var(W)
# ============================================================================
def test_xavier_variance_target():
    n_in, n_out = 200, 300
    target = 2.0 / (n_in + n_out)  # Var(W) for Xavier
    Wu = skip_if_unimplemented(xavier_uniform, n_in, n_out, seed=0)
    Wn = skip_if_unimplemented(xavier_normal, n_in, n_out, seed=0)
    vu, vn = np.var(Wu), np.var(Wn)
    assert np.isclose(vu, target, rtol=0.10), (
        f"xavier_uniform var {vu:.3e} should match 2/(n_in+n_out)={target:.3e}"
    )
    assert np.isclose(vn, target, rtol=0.10), (
        f"xavier_normal var {vn:.3e} should match 2/(n_in+n_out)={target:.3e}"
    )


def test_he_variance_target():
    n_in, n_out = 256, 128
    target = 2.0 / n_in  # Var(W) for He/Kaiming
    Wn = skip_if_unimplemented(he_normal, n_in, n_out, seed=0)
    Wu = skip_if_unimplemented(he_uniform, n_in, n_out, seed=0)
    vn, vu = np.var(Wn), np.var(Wu)
    assert np.isclose(vn, target, rtol=0.10), (
        f"he_normal var {vn:.3e} should match 2/n_in={target:.3e}"
    )
    assert np.isclose(vu, target, rtol=0.10), (
        f"he_uniform var {vu:.3e} should match 2/n_in={target:.3e}"
    )


def test_xavier_gain_scales_variance():
    n_in, n_out = 128, 128
    base = skip_if_unimplemented(xavier_normal, n_in, n_out, gain=1.0, seed=1)
    scaled = skip_if_unimplemented(xavier_normal, n_in, n_out, gain=2.0, seed=1)
    # Variance scales with gain**2 -> std scales with gain.
    assert np.isclose(np.std(scaled) / np.std(base), 2.0, rtol=0.10), (
        "gain=2 should double the std of xavier_normal"
    )


def test_init_zero_mean():
    W = skip_if_unimplemented(he_normal, 256, 256, seed=2)
    assert abs(np.mean(W)) < 0.02, "He init should be (approximately) zero-mean"


# ============================================================================
# init_dense overwrites params in place but keeps the leaf Tensors
# ============================================================================
def test_init_dense_in_place():
    layer = skip_if_unimplemented(Dense, 4, 3, bias=True, seed=0)
    W_old = layer.W  # same object must survive
    b_old = layer.b
    newW = skip_if_unimplemented(he_normal, 4, 3, seed=7)
    skip_if_unimplemented(init_dense, layer, newW, b=np.zeros(3))
    assert layer.W is W_old, "init_dense must keep the SAME W Tensor object"
    assert layer.b is b_old, "init_dense must keep the SAME b Tensor object"
    assert np.allclose(as_array(layer.W), newW), "W.data must be overwritten"
    assert np.allclose(as_array(layer.b), 0.0), "b.data must be overwritten to 0"
    assert as_array(layer.W.grad).shape == newW.shape, "grad reset to W shape"


def test_init_dense_shape_mismatch_raises():
    layer = skip_if_unimplemented(Dense, 4, 3, bias=True, seed=0)
    with pytest.raises(ValueError):
        skip_if_unimplemented(init_dense, layer, np.zeros((4, 5)))


# ============================================================================
# Activation-statistics harness: success and failure modes
# ============================================================================
def _stats(init_fn, activation, sizes=None, **kw):
    if sizes is None:
        sizes = [64] * 8
    return skip_if_unimplemented(
        forward_activation_stats, sizes, init_fn, activation,
        n_samples=512, seed=0, **kw,
    )


def test_stats_length_and_keys():
    sizes = [32, 32, 32, 32]
    stats = _stats(lambda i, o: xavier_normal(i, o, seed=0), "tanh", sizes=sizes)
    assert len(stats) == len(sizes) - 1, "one stats dict per layer"
    for d in stats:
        for k in ("mean", "std", "saturated", "dead"):
            assert k in d, f"each stats dict must contain key {k!r}"


def test_xavier_tanh_keeps_std_stable():
    stats = _stats(lambda i, o: xavier_normal(i, o, seed=0), "tanh")
    first, last = stats[0]["std"], stats[-1]["std"]
    # Matched init: std should not collapse and not explode across depth.
    assert last > 0.2 * first, (
        f"Xavier+tanh std collapsed across depth: first={first:.3f} last={last:.3f}"
    )
    assert stats[-1]["saturated"] < 0.5, (
        "Xavier+tanh should not saturate most units"
    )


def test_he_relu_keeps_signal_alive():
    stats = _stats(lambda i, o: he_normal(i, o, seed=0), "relu")
    first, last = stats[0]["std"], stats[-1]["std"]
    assert last > 0.2 * first, (
        f"He+relu std collapsed across depth: first={first:.3f} last={last:.3f}"
    )


def test_tiny_init_vanishes():
    # Deliberately tiny weights: per-layer factor n_in*Var(W) << 1 -> std -> 0.
    tiny = lambda i, o: np.random.default_rng(0).normal(0.0, 0.01, size=(i, o))
    stats = _stats(tiny, "tanh")
    assert stats[-1]["std"] < 0.1 * stats[0]["std"], (
        "tiny init should make activation std vanish with depth"
    )


def test_large_tanh_saturates():
    # Deliberately large weights push tanh into its flat tails.
    big = lambda i, o: np.random.default_rng(0).normal(0.0, 1.0, size=(i, o))
    stats = _stats(big, "tanh")
    assert stats[-1]["saturated"] > 0.5, (
        "large init should saturate most tanh units (|out| > 0.98)"
    )


def test_large_relu_kills_units():
    big = lambda i, o: np.random.default_rng(0).normal(0.0, 1.0, size=(i, o))
    stats = _stats(big, "relu")
    # With large symmetric weights ~ half of pre-activations are negative -> dead.
    assert stats[-1]["dead"] > 0.3, (
        "large init should leave a substantial dead-ReLU fraction"
    )


# ============================================================================
# A freshly initialized Dense still gradchecks (init != gradient)
# ============================================================================
def _make_inited_dense(init_fn, n_in, n_out, seed):
    layer = Dense(n_in, n_out, bias=True, seed=seed)
    init_dense(layer, init_fn(n_in, n_out, seed=seed), b=np.zeros(n_out))
    return layer


def test_gradcheck_wrt_W_after_init():
    layer = skip_if_unimplemented(_make_inited_dense, he_normal, 4, 3, 0)
    X = np.array([[0.7, -1.3, 0.2, 2.1], [1.0, 0.5, -0.5, 0.3]])

    layer.zero_grad()
    y = skip_if_unimplemented(layer, make_tensor(X, requires_grad=False))
    y.backward()
    g_analytic = as_array(layer.W.grad)

    W0 = as_array(layer.W).copy()

    def f(W):
        layer.W.data = W.copy()
        return scalar_out(layer(make_tensor(X, requires_grad=False)))

    g_num = central_diff(f, W0)
    layer.W.data = W0
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"dL/dW mismatch after init:\n analytic={g_analytic}\n numeric ={g_num}"
    )


def test_gradcheck_wrt_b_after_init():
    layer = skip_if_unimplemented(_make_inited_dense, xavier_normal, 3, 4, 1)
    X = np.array([[0.4, -0.9, 1.7], [0.2, 0.1, -0.3], [1.1, -1.0, 0.5]])

    layer.zero_grad()
    y = skip_if_unimplemented(layer, make_tensor(X, requires_grad=False))
    y.backward()
    g_analytic = as_array(layer.b.grad).reshape(-1)

    b0 = as_array(layer.b).reshape(-1).copy()

    def f(b):
        layer.b.data = b.copy().reshape(np.asarray(layer.b.data).shape)
        return scalar_out(layer(make_tensor(X, requires_grad=False)))

    g_num = central_diff(f, b0)
    layer.b.data = b0.reshape(np.asarray(layer.b.data).shape)
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"dL/db mismatch after init: analytic={g_analytic}, numeric={g_num}"
    )


def test_gradcheck_wrt_X_after_init():
    layer = skip_if_unimplemented(_make_inited_dense, he_normal, 4, 2, 2)
    X0 = np.array([[0.3, -0.6, 1.2, -2.0]])

    layer.zero_grad()
    x = make_tensor(X0, requires_grad=True)
    y = skip_if_unimplemented(layer, x)
    y.backward()
    g_analytic = as_array(x.grad)

    def f(xv):
        return scalar_out(layer(make_tensor(xv, requires_grad=False)))

    g_num = central_diff(f, X0.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"dL/dX mismatch after init:\n analytic={g_analytic}\n numeric ={g_num}"
    )
