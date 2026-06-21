"""Tests for Stage 34: Capstone -- CIFAR-10.

Verifies the CIFAR-10 capstone built on the prior stages:

  * ``BatchNorm2d`` forward differs between TRAIN (batch stats) and EVAL (running
    buffers), and its analytic backward (dL/dx, dL/dgamma, dL/dbeta) matches a
    central-difference estimate:  df/dx ~= (f(x + eps) - f(x - eps)) / (2 * eps);
  * data augmentation (random_crop / random_horizontal_flip / normalize /
    Augment) preserves shape, is gradient-free, and is deterministic-normalize in
    eval mode;
  * the LR schedules (cosine_lr / step_lr) hit the right values at boundaries and
    are monotone where they should be;
  * ``ConvNet.forward`` returns ``(B, n_classes)`` logits for several input
    shapes, with ``flat_dim`` DERIVED (not hardcoded);
  * end-to-end gradcheck: for one batch, every parameter's ``.grad`` from
    ``cross_entropy_loss(model(X), y).backward()`` matches the numeric slope
    (loosened tolerance for the deep BN stack);
  * ``train_cifar`` drives loss down and reaches high train accuracy on a tiny
    synthetic CIFAR-like set, recording the per-step learning rate.

Depends on stage_09 (Tensor), stage_11 (Dense), stage_25 (Conv2D/MaxPool2D/
Flatten), and the stage_32 mytorch API (cross_entropy_loss / Adam / DataLoader /
Dataset, re-exporting stages 13 / 18 / 20). If any is not implemented yet, the
suite skips cleanly.

Run with:  pytest stage_34_capstone_cifar10/test.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import the things under test, skipping cleanly if not ready yet. --------
try:
    from code import (
        Adam,
        Augment,
        BatchNorm2d,
        ConvNet,
        Conv2D,
        Tensor,
        accuracy,
        cosine_lr,
        cross_entropy_loss,
        make_cifar_like,
        normalize,
        random_crop,
        random_horizontal_flip,
        step_lr,
        train_cifar,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_34 capstone or a prior dependency "
        f"(stage_09/11/13/18/20/25/26) not importable yet: {exc}",
        allow_module_level=True,
    )

# Load DataLoader / Dataset through the shared dlfs shim (stage_32 re-exports the
# stage_20 plumbing; the code module imports them the same way).
try:
    from dlfs import stage_import

    Stage32_DataLoader, Stage32_Dataset = stage_import("stage_32", "DataLoader", "Dataset")
except (ImportError, NotImplementedError):  # pragma: no cover
    Stage32_DataLoader = None
    Stage32_Dataset = None

RNG = np.random.default_rng(34)
EPS = 1e-5
ATOL_BN = 1e-5
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
    """Python float of a scalar (0-d / size-1) Tensor, else sum."""
    a = as_array(t)
    return float(a.reshape(-1)[0]) if a.size == 1 else float(a.sum())


def skip_if_unbuilt(fn, *args, **kwargs):
    """Call ``fn``; if a skeleton body raises NotImplementedError, skip."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"skeleton not implemented yet: {exc}")


# ---------------------------------------------------------------------------
# BatchNorm2d forward: train vs eval
# ---------------------------------------------------------------------------
def test_batchnorm2d_train_standardizes_per_channel():
    """In train mode each channel of the output is ~zero-mean, unit-var."""
    x = RNG.standard_normal((8, 3, 5, 5)) * 2.0 + 1.0
    bn = skip_if_unbuilt(BatchNorm2d, 3)
    out = skip_if_unbuilt(bn, make_tensor(x))
    assert out.shape == (8, 3, 5, 5), f"shape changed: {out.shape}"
    o = as_array(out)
    # gamma=1, beta=0 initially -> output == x_hat -> per-channel mean ~0, var ~1.
    mean_c = o.mean(axis=(0, 2, 3))
    var_c = o.var(axis=(0, 2, 3))
    np.testing.assert_allclose(
        mean_c, np.zeros(3), atol=1e-6,
        err_msg="BatchNorm2d train output must be per-channel zero-mean.",
    )
    np.testing.assert_allclose(
        var_c, np.ones(3), atol=1e-3,
        err_msg="BatchNorm2d train output must be per-channel unit-variance.",
    )


def test_batchnorm2d_train_updates_running_buffers():
    """A train forward moves running_mean / running_var off their init."""
    bn = skip_if_unbuilt(BatchNorm2d, 3, momentum=0.5)
    rm0 = np.array(bn.running_mean, dtype=float)
    rv0 = np.array(bn.running_var, dtype=float)
    x = RNG.standard_normal((6, 3, 4, 4)) + 3.0
    skip_if_unbuilt(bn, make_tensor(x))
    assert not np.allclose(bn.running_mean, rm0), (
        "running_mean must update during a train forward."
    )
    assert not np.allclose(bn.running_var, rv0), (
        "running_var must update during a train forward."
    )


def test_batchnorm2d_eval_uses_buffers_no_update():
    """Eval forward uses the running buffers and does NOT update them."""
    bn = skip_if_unbuilt(BatchNorm2d, 2)
    skip_if_unbuilt(bn.eval)
    rm0 = np.array(bn.running_mean, dtype=float)
    rv0 = np.array(bn.running_var, dtype=float)
    x = RNG.standard_normal((5, 2, 3, 3))
    out = skip_if_unbuilt(bn, make_tensor(x))
    assert out.shape == (5, 2, 3, 3)
    np.testing.assert_allclose(
        bn.running_mean, rm0, atol=1e-12,
        err_msg="eval forward must NOT touch running_mean.",
    )
    np.testing.assert_allclose(
        bn.running_var, rv0, atol=1e-12,
        err_msg="eval forward must NOT touch running_var.",
    )


def test_batchnorm2d_parameters_excludes_buffers():
    """parameters() returns [gamma, beta] only; buffers stay out."""
    bn = skip_if_unbuilt(BatchNorm2d, 4)
    params = skip_if_unbuilt(bn.parameters)
    assert len(params) == 2, "BatchNorm2d.parameters() must be [gamma, beta]."
    for p in params:
        assert as_array(p).shape == (4,), "gamma/beta must have shape (C,)."


# ---------------------------------------------------------------------------
# BatchNorm2d backward: central-difference gradcheck
# ---------------------------------------------------------------------------
def _bn_loss(bn, xarr, w):
    """Scalar loss = sum(bn(x) * w): a linear functional with a known gradient."""
    out = bn(make_tensor(xarr))
    return (out * make_tensor(w)).sum()


def test_batchnorm2d_gradcheck_input():
    """Central-difference gradcheck of dL/dx through BatchNorm2d (train mode)."""
    shape = (4, 2, 3, 3)
    x0 = RNG.standard_normal(shape)
    w = RNG.standard_normal(shape)

    bn = skip_if_unbuilt(BatchNorm2d, 2)
    xt = make_tensor(x0)
    out = skip_if_unbuilt(bn, xt)
    loss = skip_if_unbuilt(lambda: (out * make_tensor(w)).sum())
    skip_if_unbuilt(loss.backward)
    g_analytic = np.array(xt.grad, dtype=float)

    g_num = np.zeros_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp = x0.copy(); xp[idx] += EPS
        xm = x0.copy(); xm[idx] -= EPS
        fp = scalar(_bn_loss(BatchNorm2d(2), xp, w))
        fm = scalar(_bn_loss(BatchNorm2d(2), xm, w))
        g_num[idx] = (fp - fm) / (2 * EPS)
        it.iternext()

    np.testing.assert_allclose(
        g_analytic, g_num, atol=ATOL_BN, rtol=RTOL,
        err_msg="BatchNorm2d input-gradient must match central differences.",
    )


def test_batchnorm2d_gradcheck_gamma_beta():
    """Central-difference gradcheck of dL/dgamma and dL/dbeta."""
    shape = (5, 3, 2, 2)
    x0 = RNG.standard_normal(shape)
    w = RNG.standard_normal(shape)

    bn = skip_if_unbuilt(BatchNorm2d, 3)
    # perturb params off the trivial init so the check is non-degenerate.
    bn.gamma.data[:] = RNG.standard_normal(3) * 0.5 + 1.0
    bn.beta.data[:] = RNG.standard_normal(3) * 0.5
    gamma0 = np.array(bn.gamma.data, dtype=float)
    beta0 = np.array(bn.beta.data, dtype=float)

    out = skip_if_unbuilt(bn, make_tensor(x0))
    loss = skip_if_unbuilt(lambda: (out * make_tensor(w)).sum())
    skip_if_unbuilt(loss.backward)
    g_gamma = np.array(bn.gamma.grad, dtype=float)
    g_beta = np.array(bn.beta.grad, dtype=float)

    def loss_with_params(gamma_arr, beta_arr):
        b = BatchNorm2d(3)
        b.gamma.data[:] = gamma_arr
        b.beta.data[:] = beta_arr
        return scalar(_bn_loss_param(b, x0, w))

    num_gamma = np.zeros(3)
    num_beta = np.zeros(3)
    for c in range(3):
        gp = gamma0.copy(); gp[c] += EPS
        gm = gamma0.copy(); gm[c] -= EPS
        num_gamma[c] = (loss_with_params(gp, beta0) - loss_with_params(gm, beta0)) / (2 * EPS)
        bp = beta0.copy(); bp[c] += EPS
        bm = beta0.copy(); bm[c] -= EPS
        num_beta[c] = (loss_with_params(gamma0, bp) - loss_with_params(gamma0, bm)) / (2 * EPS)

    np.testing.assert_allclose(
        g_gamma, num_gamma, atol=ATOL_BN, rtol=RTOL,
        err_msg="BatchNorm2d gamma-gradient must match central differences.",
    )
    np.testing.assert_allclose(
        g_beta, num_beta, atol=ATOL_BN, rtol=RTOL,
        err_msg="BatchNorm2d beta-gradient must match central differences.",
    )


def _bn_loss_param(bn, xarr, w):
    """Like _bn_loss but reuses an already-configured bn (params preset)."""
    out = bn(make_tensor(xarr))
    return (out * make_tensor(w)).sum()


# ---------------------------------------------------------------------------
# Augmentation: shape, eval determinism, gradient-free
# ---------------------------------------------------------------------------
def test_random_crop_preserves_shape():
    x = RNG.standard_normal((4, 3, 8, 8))
    out = skip_if_unbuilt(
        random_crop, x, pad=2, rng=np.random.default_rng(0)
    )
    assert out.shape == x.shape, f"random_crop changed shape: {out.shape}"


def test_random_horizontal_flip_shape_and_flips_some():
    x = RNG.standard_normal((64, 3, 6, 6))
    out = skip_if_unbuilt(
        random_horizontal_flip, x, p=1.0, rng=np.random.default_rng(0)
    )
    assert out.shape == x.shape
    # p=1.0 -> every image flipped along width.
    np.testing.assert_allclose(
        out, x[:, :, :, ::-1], atol=1e-12,
        err_msg="p=1.0 must flip every image along the width axis.",
    )


def test_normalize_per_channel():
    x = RNG.standard_normal((10, 3, 5, 5))
    mean = [0.1, 0.2, 0.3]
    std = [2.0, 0.5, 1.5]
    out = skip_if_unbuilt(normalize, x, mean, std)
    ref = (x - np.array(mean).reshape(1, 3, 1, 1)) / np.array(std).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(out, ref, atol=1e-12, err_msg="normalize mismatch.")


def test_augment_eval_is_normalize_only_and_returns_array():
    """In eval mode Augment applies normalize only (no crop / no flip)."""
    x = RNG.standard_normal((6, 3, 8, 8))
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    aug = skip_if_unbuilt(
        Augment, pad=4, flip_p=0.5, mean=mean, std=std, seed=0
    )
    skip_if_unbuilt(aug.eval)
    out = skip_if_unbuilt(aug, x)
    assert isinstance(out, np.ndarray), "Augment must return a NumPy array."
    np.testing.assert_allclose(
        out, x, atol=1e-12,
        err_msg="eval Augment with mean=0/std=1 must be the identity.",
    )


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------
def test_cosine_lr_endpoints_and_midpoint():
    base, mn, T = 0.1, 0.0, 100
    assert abs(skip_if_unbuilt(cosine_lr, 0, T, base_lr=base, min_lr=mn) - base) < 1e-12
    assert abs(skip_if_unbuilt(cosine_lr, T, T, base_lr=base, min_lr=mn) - mn) < 1e-9
    mid = skip_if_unbuilt(cosine_lr, T // 2, T, base_lr=base, min_lr=mn)
    assert abs(mid - 0.5 * base) < 1e-9, f"cosine midpoint should be base/2, got {mid}"


def test_cosine_lr_monotone_decreasing():
    T = 50
    vals = [skip_if_unbuilt(cosine_lr, t, T, base_lr=1.0) for t in range(T + 1)]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(T)), (
        "cosine_lr must be non-increasing over [0, T]."
    )


def test_step_lr_drops():
    base, de, g = 1.0, 10, 0.1
    assert abs(skip_if_unbuilt(step_lr, 0, base_lr=base, drop_every=de, gamma=g) - 1.0) < 1e-12
    assert abs(skip_if_unbuilt(step_lr, 9, base_lr=base, drop_every=de, gamma=g) - 1.0) < 1e-12
    assert abs(skip_if_unbuilt(step_lr, 10, base_lr=base, drop_every=de, gamma=g) - 0.1) < 1e-12
    assert abs(skip_if_unbuilt(step_lr, 20, base_lr=base, drop_every=de, gamma=g) - 0.01) < 1e-12


# ---------------------------------------------------------------------------
# ConvNet forward / shape derivation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("in_shape,n_classes,channels", [
    ((3, 32, 32), 10, (4, 8)),
    ((3, 16, 16), 5, (4, 8)),
    ((1, 8, 8), 3, (2, 4)),
])
def test_convnet_forward_shape(in_shape, n_classes, channels):
    """ConvNet.forward returns (B, n_classes) logits; flat_dim is derived."""
    B = 3
    model = skip_if_unbuilt(
        ConvNet, in_shape, n_classes, channels=channels, hidden=8, seed=0
    )
    x = RNG.standard_normal((B, *in_shape))
    logits = skip_if_unbuilt(model, x)
    assert logits.shape == (B, n_classes), (
        f"expected {(B, n_classes)} logits, got {logits.shape}"
    )
    C, H, W = in_shape
    S = len(channels)
    expected_flat = channels[-1] * (H // (2 ** S)) * (W // (2 ** S))
    assert model.flat_dim == expected_flat, (
        f"flat_dim must be DERIVED ({expected_flat}); got {model.flat_dim}."
    )


def test_convnet_parameters_nonempty_and_unique():
    """ConvNet collects params from conv/bn/dense; no duplicates."""
    model = skip_if_unbuilt(ConvNet, (3, 16, 16), 5, channels=(4, 8), seed=1)
    params = skip_if_unbuilt(model.parameters)
    # 2 stages * (2 conv W/b + 2 bn gamma/beta) + 2 dense W/b = many params.
    assert len(params) >= 12, f"expected many params, got {len(params)}"
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids)), "parameters() must not repeat tensors."


def test_convnet_train_eval_propagates_to_batchnorm():
    """train()/eval() return self and toggle BatchNorm2d sub-layers."""
    model = skip_if_unbuilt(ConvNet, (3, 16, 16), 5, channels=(4, 8), seed=2)
    assert skip_if_unbuilt(model.eval) is model
    bns = [l for l in model.layers if isinstance(l, BatchNorm2d)]
    assert bns, "ConvNet must contain BatchNorm2d layers."
    assert all(b.training is False for b in bns), "eval() must propagate to BN."
    skip_if_unbuilt(model.train)
    assert all(b.training is True for b in bns), "train() must propagate to BN."


# ---------------------------------------------------------------------------
# end-to-end gradcheck through the whole network
# ---------------------------------------------------------------------------
def test_convnet_end_to_end_gradcheck():
    """Central-difference gradcheck of every parameter through the full net.

    Tiny model + tiny batch so the finite-difference loop is cheap. Probe a few
    random coordinates per parameter tensor.
    """
    in_shape = (3, 8, 8)
    n_classes = 3
    B = 4
    model = skip_if_unbuilt(
        ConvNet, in_shape, n_classes, channels=(2, 3), hidden=4, seed=7
    )
    X = RNG.standard_normal((B, *in_shape))
    y = RNG.integers(0, n_classes, size=B)
    params = skip_if_unbuilt(model.parameters)

    def loss_value():
        model.zero_grad()
        logits = model(X)
        return cross_entropy_loss(logits, y)

    loss = skip_if_unbuilt(loss_value)
    skip_if_unbuilt(loss.backward)
    analytic = [np.array(p.grad, dtype=float) for p in params]

    rng = np.random.default_rng(123)
    for p, g_an in zip(params, analytic):
        flat = p.data.reshape(-1)
        n = flat.size
        probes = rng.choice(n, size=min(3, n), replace=False)
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
# metric + synthetic data + convergence
# ---------------------------------------------------------------------------
def test_accuracy_matches_numpy():
    logits = RNG.standard_normal((20, 5))
    y = RNG.integers(0, 5, size=20)
    ref = float((np.argmax(logits, axis=1) == y).mean())
    got = skip_if_unbuilt(accuracy, make_tensor(logits), y)
    assert abs(got - ref) < 1e-12, f"accuracy={got} expected {ref}"


def test_make_cifar_like_shapes():
    X, y = skip_if_unbuilt(
        make_cifar_like, n_per_class=6, img_size=16, n_classes=4, seed=0
    )
    assert X.shape == (24, 3, 16, 16), f"got X shape {X.shape}"
    assert y.shape == (24,), f"got y shape {y.shape}"
    assert set(np.unique(y).tolist()) == {0, 1, 2, 3}, "labels must cover all classes."


@pytest.mark.slow
def test_train_cifar_converges_and_records_lr():
    """train_cifar drives loss down, reaches high train accuracy, logs the LR.

    A tiny few-class CIFAR-like set + a small ConvNet should overfit within a few
    epochs. Skipped if stage_20's DataLoader is unbuilt.
    """
    if Stage32_DataLoader is None or Stage32_Dataset is None:
        pytest.skip("stage_20 DataLoader/Dataset not importable yet.")

    X, y = skip_if_unbuilt(
        make_cifar_like, n_per_class=24, img_size=16, n_classes=3,
        noise=0.2, seed=3,
    )
    ds = skip_if_unbuilt(Stage32_Dataset, X, y)
    loader = skip_if_unbuilt(Stage32_DataLoader, ds, batch_size=12, shuffle=True, seed=3)
    model = skip_if_unbuilt(
        ConvNet, (3, 16, 16), 3, channels=(8, 16), hidden=32, seed=3
    )
    hist = skip_if_unbuilt(
        train_cifar, model, loader, epochs=8, base_lr=2e-3, schedule="cosine"
    )

    losses = hist["train_loss"]
    assert losses[-1] < losses[0], (
        f"training loss should decrease: first={losses[0]:.3f} "
        f"last={losses[-1]:.3f}"
    )
    assert len(hist["lr"]) == hist["steps"], "lr must be recorded once per step."
    assert hist["lr"][0] >= hist["lr"][-1] - 1e-12, "cosine LR should anneal down."

    model.eval()
    logits = skip_if_unbuilt(model, X)
    acc = skip_if_unbuilt(accuracy, logits, y)
    assert acc >= 0.85, f"expected >= 0.85 train accuracy, got {acc:.3f}"
