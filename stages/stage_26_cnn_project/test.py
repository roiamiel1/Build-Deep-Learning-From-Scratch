"""Tests for Stage 26: CNN classifier (project).

Verifies the composed convolutional classifier built on the prior stages:

  * ``Flatten`` round-trips shape ``(B, C, H, W) -> (B, C*H*W)`` and its
    gradient is the identity-up-to-layout reshape backward, checked with
    central differences:   df/dx ~= (f(x + eps) - f(x - eps)) / (2 * eps);
  * ``CNN.forward`` returns ``(B, n_classes)`` logits for several input shapes,
    with ``flat_dim`` DERIVED (not hardcoded) -- tested across input sizes;
  * end-to-end gradcheck: for one batch, every parameter's ``.grad`` from
    ``cross_entropy_loss(model(X_b), y_b).backward()`` matches the
    central-difference estimate (loosened tolerance for the deep stack);
  * ``accuracy`` matches a NumPy reference;
  * ``train_cnn`` drives training loss down and reaches high train accuracy on a
    tiny synthetic 2-class digit set.

This stage depends on stage_09 (Tensor), stage_11 (Dense), stage_13
(cross_entropy_loss), stage_18 (Adam), stage_20 (DataLoader/Dataset) and
stage_25 (Conv2D/MaxPool2D/Flatten). If any of those is not implemented yet, the
suite skips cleanly instead of erroring.

Run with:  pytest stage_26_cnn_project/test.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import the things under test, skipping cleanly if not ready yet. --------
try:
    from code import (
        CNN,
        Adam,
        Conv2d,
        Dense,
        Flatten,
        MaxPool2d,
        Tensor,
        accuracy,
        cross_entropy_loss,
        make_digit_blobs,
        train_cnn,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_26 CNN or a prior dependency (stage_09/11/13/18/20/25) "
        f"not importable yet: {exc}",
        allow_module_level=True,
    )

# Load the DataLoader / Dataset from stage_20 via the shared dlfs shim (the same
# mechanism the module under test uses), so we can build loaders in the
# convergence test.
try:
    from dlfs import stage_import

    Stage20_DataLoader, Stage20_Dataset = stage_import(
        "stage_20", "DataLoader", "Dataset"
    )
except (ImportError, NotImplementedError):  # pragma: no cover
    Stage20_DataLoader = None
    Stage20_Dataset = None

RNG = np.random.default_rng(26)
EPS = 1e-5
ATOL_FLAT = 1e-6
ATOL_E2E = 1e-4
RTOL = 1e-3


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Underlying ndarray of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def make_tensor(arr):
    """Build a Tensor from a numpy array, tolerating different ctor kwargs."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=True)
    except TypeError:
        return Tensor(arr)


def scalar(t):
    """Python float of a scalar (0-d / size-1) Tensor."""
    a = as_array(t)
    return float(a.reshape(-1)[0]) if a.size == 1 else float(a.sum())


def skip_if_unbuilt(fn, *args, **kwargs):
    """Call ``fn``; if a skeleton body raises NotImplementedError, skip."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"skeleton not implemented yet: {exc}")


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------
def test_flatten_shape_roundtrip():
    """Flatten maps (B, C, H, W) -> (B, C*H*W) preserving values row-wise."""
    x = RNG.standard_normal((4, 3, 5, 6))
    flat = skip_if_unbuilt(Flatten)
    out = skip_if_unbuilt(flat, make_tensor(x))
    assert out.shape == (4, 3 * 5 * 6), f"expected (4, 90), got {out.shape}"
    np.testing.assert_allclose(
        as_array(out),
        x.reshape(4, -1),
        atol=1e-12,
        err_msg="Flatten must preserve element order (B, C, H, W) -> (B, C*H*W).",
    )


def test_flatten_no_parameters():
    """Flatten owns no learnable parameters."""
    flat = skip_if_unbuilt(Flatten)
    assert skip_if_unbuilt(flat.parameters) == [], "Flatten.parameters() must be []."


def test_flatten_gradcheck():
    """Central-difference gradcheck of dL/dx through Flatten.

    Loss is a fixed random linear functional of the flattened output, so the
    analytic gradient (reshape backward) must equal the numeric slope exactly
    up to fp error (a reshape is an identity-up-to-layout map).
    """
    shape = (2, 2, 3, 4)
    x0 = RNG.standard_normal(shape)
    w = RNG.standard_normal((shape[0], np.prod(shape[1:])))  # functional weights

    def forward_loss(xarr):
        flat = Flatten()
        out = flat(make_tensor(xarr))
        # scalar = sum(out * w) : a linear functional with known gradient.
        loss = (out * make_tensor(w)).sum()
        return loss

    # Analytic grad via backward.
    flat = skip_if_unbuilt(Flatten)
    xt = make_tensor(x0)
    out = skip_if_unbuilt(flat, xt)
    loss = skip_if_unbuilt(lambda: (out * make_tensor(w)).sum())
    skip_if_unbuilt(loss.backward)
    g_analytic = np.array(xt.grad, dtype=float)

    # Numeric grad via central differences.
    g_num = np.zeros_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp = x0.copy(); xp[idx] += EPS
        xm = x0.copy(); xm[idx] -= EPS
        fp = scalar(forward_loss(xp))
        fm = scalar(forward_loss(xm))
        g_num[idx] = (fp - fm) / (2 * EPS)
        it.iternext()

    np.testing.assert_allclose(
        g_analytic, g_num, atol=ATOL_FLAT, rtol=RTOL,
        err_msg="Flatten backward (reshape) must match the numeric gradient.",
    )


# ---------------------------------------------------------------------------
# CNN forward / shape derivation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("in_shape,n_classes", [
    ((1, 8, 8), 2),
    ((1, 16, 16), 3),
    ((3, 12, 12), 4),
])
def test_cnn_forward_shape(in_shape, n_classes):
    """CNN.forward returns (B, n_classes) logits; flat_dim is derived."""
    B = 5
    model = skip_if_unbuilt(
        CNN, in_shape, n_classes, conv_channels=(4, 8), kernel_size=3,
        hidden=16, seed=0,
    )
    x = RNG.standard_normal((B, *in_shape))
    logits = skip_if_unbuilt(model, x)
    assert logits.shape == (B, n_classes), (
        f"expected logits shape {(B, n_classes)}, got {logits.shape}"
    )
    # flat_dim must be the derived c2 * (H//4) * (W//4).
    C, H, W = in_shape
    expected_flat = 8 * (H // 4) * (W // 4)
    assert model.flat_dim == expected_flat, (
        f"flat_dim must be DERIVED ({expected_flat}); got {model.flat_dim} "
        f"(do not hardcode for one input size)."
    )


def test_cnn_parameters_nonempty_and_unique():
    """CNN collects parameters from all sub-layers; no duplicates."""
    model = skip_if_unbuilt(CNN, (1, 8, 8), 2, conv_channels=(4, 8), seed=1)
    params = skip_if_unbuilt(model.parameters)
    assert len(params) >= 6, (
        "expected >= 6 params (2 conv W/b + 2 dense W/b); got "
        f"{len(params)}"
    )
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids)), "parameters() must not repeat tensors."


def test_cnn_train_eval_chainable():
    """train()/eval() return self and set the training flag."""
    model = skip_if_unbuilt(CNN, (1, 8, 8), 2, seed=2)
    assert skip_if_unbuilt(model.train) is model
    assert model.training is True
    assert skip_if_unbuilt(model.eval) is model
    assert model.training is False


# ---------------------------------------------------------------------------
# end-to-end gradcheck through the whole network
# ---------------------------------------------------------------------------
def test_cnn_end_to_end_gradcheck():
    """Central-difference gradcheck of every parameter through the full net.

    Tiny model + tiny batch so the finite-difference loop is cheap. The loss is
    cross_entropy_loss(model(X), y); for a subset of coordinates per parameter
    we compare the analytic grad (from backward) to the numeric slope.
    """
    in_shape = (1, 8, 8)
    n_classes = 3
    B = 4
    model = skip_if_unbuilt(
        CNN, in_shape, n_classes, conv_channels=(2, 3), kernel_size=3,
        hidden=5, seed=7,
    )
    X = RNG.standard_normal((B, *in_shape))
    y = RNG.integers(0, n_classes, size=B)

    params = skip_if_unbuilt(model.parameters)

    def loss_value():
        model.zero_grad()
        logits = model(X)
        return cross_entropy_loss(logits, y)

    # Analytic gradients.
    loss = skip_if_unbuilt(loss_value)
    skip_if_unbuilt(loss.backward)
    analytic = [np.array(p.grad, dtype=float) for p in params]

    rng = np.random.default_rng(123)
    for p, g_an in zip(params, analytic):
        flat = p.data.reshape(-1)
        n = flat.size
        # probe up to 4 random coordinates per parameter tensor.
        probes = rng.choice(n, size=min(4, n), replace=False)
        for j in probes:
            orig = flat[j]
            flat[j] = orig + EPS
            fp = scalar(loss_value())
            flat[j] = orig - EPS
            fm = scalar(loss_value())
            flat[j] = orig
            num = (fp - fm) / (2 * EPS)
            an = g_an.reshape(-1)[j]
            assert abs(an - num) <= ATOL_E2E + RTOL * abs(num), (
                f"end-to-end gradcheck failed: analytic={an:.3e} "
                f"numeric={num:.3e} (param shape {p.data.shape}, idx {j})."
            )


# ---------------------------------------------------------------------------
# accuracy metric
# ---------------------------------------------------------------------------
def test_accuracy_matches_numpy():
    """accuracy(logits, targets) equals the NumPy argmax reference."""
    logits = RNG.standard_normal((20, 4))
    y = RNG.integers(0, 4, size=20)
    ref = float((np.argmax(logits, axis=1) == y).mean())
    got = skip_if_unbuilt(accuracy, make_tensor(logits), y)
    assert abs(got - ref) < 1e-12, f"accuracy={got} expected {ref}"


def test_accuracy_perfect_and_zero():
    """Sanity bounds: a perfectly aligned set scores 1.0, anti-aligned 0.0."""
    logits = np.array([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    assert skip_if_unbuilt(accuracy, make_tensor(logits), np.array([0, 1, 0])) == 1.0
    assert skip_if_unbuilt(accuracy, make_tensor(logits), np.array([1, 0, 1])) == 0.0


# ---------------------------------------------------------------------------
# synthetic data + convergence
# ---------------------------------------------------------------------------
def test_make_digit_blobs_shapes():
    """Synthetic digit set has channels-first images and integer labels."""
    X, y = skip_if_unbuilt(
        make_digit_blobs, n_per_class=10, img_size=8, n_classes=2, seed=0
    )
    assert X.shape == (20, 1, 8, 8), f"got X shape {X.shape}"
    assert y.shape == (20,), f"got y shape {y.shape}"
    assert set(np.unique(y).tolist()) == {0, 1}, "labels must cover all classes."


@pytest.mark.slow
def test_train_cnn_converges():
    """train_cnn drives loss down and reaches high train accuracy.

    A tiny 2-class digit set + a small CNN should overfit to >= 0.90 train
    accuracy within a few epochs. Skipped if stage_20's DataLoader is unbuilt.
    """
    if Stage20_DataLoader is None or Stage20_Dataset is None:
        pytest.skip("stage_20 DataLoader/Dataset not importable yet.")

    X, y = skip_if_unbuilt(
        make_digit_blobs, n_per_class=48, img_size=8, n_classes=2,
        noise=0.25, seed=3,
    )
    ds = skip_if_unbuilt(Stage20_Dataset, X, y)
    loader = skip_if_unbuilt(
        Stage20_DataLoader, ds, batch_size=16, shuffle=True, seed=3
    )
    model = skip_if_unbuilt(
        CNN, (1, 8, 8), 2, conv_channels=(6, 12), kernel_size=3, hidden=24, seed=3
    )
    hist = skip_if_unbuilt(train_cnn, model, loader, epochs=8, lr=2e-3)

    losses = hist["train_loss"]
    assert losses[-1] < losses[0], (
        f"training loss should decrease: first={losses[0]:.3f} "
        f"last={losses[-1]:.3f}"
    )

    # Final train accuracy over the whole set in eval mode.
    model.eval()
    logits = skip_if_unbuilt(model, X)
    acc = skip_if_unbuilt(accuracy, logits, y)
    assert acc >= 0.90, f"expected >= 0.90 train accuracy, got {acc:.3f}"
