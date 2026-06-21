"""Tests for Stage 19: Mini-batch training.

These tests verify the data-feeding layer on top of the stage_15 training stack
(``MLP``, ``mse_loss``, ``SGD``, ``make_moons``) and the stage_09 ``Tensor``:

  * ``iterate_minibatches`` partitions the dataset exactly (covers every row
    once; correct batch counts; ``drop_last``; ``shuffle`` reproducibility;
    bad-input ``ValueError``s),
  * ``train_minibatch`` runs ``epochs * ceil(N/B)`` steps, drives the epoch loss
    down, and -- with ``batch_size == N`` and ``shuffle=False`` -- reproduces a
    hand-written full-batch loop step-for-step,
  * ``gradient_noise`` falls as batch size grows, matching ``sigma**2 / B``,
  * ``epochs_to_threshold`` finds the first crossing,
  * the ``p.grad`` produced by one batched ``mse_loss(...).backward()`` matches a
    central-difference gradient:

        dL/dp ~= (L(p + eps) - L(p - eps)) / (2 * eps).

If stage_09 / stage_15 / this stage are not implemented yet, the suite skips
cleanly instead of erroring, so you can run it incrementally.

Run with:  pytest stage_19_batch_training/test.py
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# Make the shared `dlfs` shim importable (it lives at the curriculum root) so
# `code.py`'s `stage_import` calls resolve the prior stages.
sys.path.insert(0, _ROOT)

# --- Import the things under test, skipping cleanly if not ready yet. --------
# `code.py` ADDS iterate_minibatches / train_minibatch / gradient_noise /
# epochs_to_threshold, and re-exports MLP (stage_12), mse_loss (stage_13),
# SGD (stage_14), Tensor (stage_09), make_moons (stage_15) -- all via
# dlfs.stage_import.
try:
    from code import (
        MLP,
        SGD,
        Tensor,
        epochs_to_threshold,
        gradient_noise,
        iterate_minibatches,
        make_moons,
        mse_loss,
        train_minibatch,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_19 / stage_15 / stage_09 not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Return the underlying numpy array of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


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


def toy_data(n=40, seed=0):
    """A small reproducible moons dataset for the loop / noise tests."""
    X, y = make_moons(n=n, noise=0.1, seed=seed)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


# ===========================================================================
# iterate_minibatches: exact partitioning
# ===========================================================================
def test_batches_cover_every_row_once_no_shuffle():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    batches = list(iterate_minibatches(X, y, batch_size=3, shuffle=False))
    # ceil(10 / 3) == 4 batches; sizes 3,3,3,1
    assert [b[0].shape[0] for b in batches] == [3, 3, 3, 1], "wrong batch sizes"
    Xcat = np.concatenate([b[0] for b in batches], axis=0)
    ycat = np.concatenate([b[1] for b in batches], axis=0)
    assert np.array_equal(Xcat, X), "no-shuffle batches must recover X in order"
    assert np.array_equal(ycat, y), "no-shuffle batches must recover y in order"


def test_drop_last_drops_partial_batch():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    batches = list(
        iterate_minibatches(X, y, batch_size=3, shuffle=False, drop_last=True)
    )
    assert len(batches) == 10 // 3, "drop_last must yield N // B batches"
    assert all(b[0].shape[0] == 3 for b in batches), "kept batches are full-size"


def test_shuffle_is_a_permutation_and_aligned():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = X[:, 0].copy()  # y[i] == X[i, 0] so we can check alignment after shuffle
    batches = list(iterate_minibatches(X, y, batch_size=5, shuffle=True, seed=1))
    Xcat = np.concatenate([b[0] for b in batches], axis=0)
    ycat = np.concatenate([b[1] for b in batches], axis=0)
    # All 12 rows present exactly once (as a set of first coords).
    assert sorted(Xcat[:, 0].tolist()) == sorted(X[:, 0].tolist()), (
        "shuffled batches must be a permutation covering every row once"
    )
    # X_b and y_b stayed paired through the shuffle.
    assert np.array_equal(ycat, Xcat[:, 0]), "shuffle must keep X_b and y_b aligned"


def test_shuffle_seed_is_reproducible_and_varies():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float)
    a = np.concatenate(
        [b[1] for b in iterate_minibatches(X, y, 4, shuffle=True, seed=7)]
    )
    b = np.concatenate(
        [b[1] for b in iterate_minibatches(X, y, 4, shuffle=True, seed=7)]
    )
    c = np.concatenate(
        [b[1] for b in iterate_minibatches(X, y, 4, shuffle=True, seed=8)]
    )
    assert np.array_equal(a, b), "same seed must give the same order"
    assert not np.array_equal(a, c), "different seed should (almost surely) differ"


def test_bad_batch_size_raises():
    X = np.zeros((5, 2))
    y = np.zeros(5)
    for bad in (0, -1, 6):
        with pytest.raises(ValueError):
            list(iterate_minibatches(X, y, batch_size=bad))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        list(iterate_minibatches(np.zeros((5, 2)), np.zeros(4), batch_size=2))


# ===========================================================================
# train_minibatch: step counts, history, and full-batch equivalence
# ===========================================================================
def test_step_count_matches_batches_per_epoch():
    X, y = toy_data(n=40, seed=0)
    model = MLP([2, 8, 1], activation="tanh", seed=0)
    epochs, B = 3, 16
    out = train_minibatch(
        model, X, y, lr=0.05, epochs=epochs, batch_size=B, shuffle=True, seed=0
    )
    n_batches = int(np.ceil(len(X) / B))
    assert out["steps"] == epochs * n_batches, "steps == epochs * ceil(N/B)"
    assert len(out["batch_loss"]) == out["steps"], "one batch_loss per step"
    assert len(out["epoch_loss"]) == epochs, "one epoch_loss per epoch"


def test_epoch_loss_decreases():
    X, y = toy_data(n=60, seed=1)
    model = MLP([2, 16, 1], activation="tanh", seed=1)
    out = train_minibatch(
        model, X, y, lr=0.1, epochs=80, batch_size=16, shuffle=True, seed=1
    )
    hist = out["epoch_loss"]
    assert hist[-1] < hist[0], "epoch loss must fall over training"
    assert hist[-1] < 0.6 * hist[0], "loss should drop meaningfully, not crawl"


def test_full_batch_matches_manual_loop():
    """batch_size == N with shuffle=False must equal a hand-rolled full-batch
    loop on the SAME initial model -- i.e. mini-batch reduces to stage_15."""
    X, y = toy_data(n=24, seed=2)

    # Model A: trained via train_minibatch in full-batch mode.
    model_a = MLP([2, 8, 1], activation="tanh", seed=3)
    out = train_minibatch(
        model_a, X, y, lr=0.1, epochs=10, batch_size=len(X), shuffle=False
    )

    # Model B: identical init, manual full-batch loop using the stage_15 stack.
    model_b = MLP([2, 8, 1], activation="tanh", seed=3)
    opt = SGD(model_b.parameters(), 0.1)
    manual = []
    yb = np.asarray(y, dtype=float).reshape(-1, 1)
    for _ in range(10):
        pred = model_b(X)
        loss = mse_loss(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        manual.append(float(loss.data))

    assert np.allclose(out["epoch_loss"], manual, atol=ATOL, rtol=RTOL), (
        "full-batch train_minibatch must match a manual full-batch loop:\n "
        f"train_minibatch={out['epoch_loss']}\n manual         ={manual}"
    )
    assert out["steps"] == 10, "full-batch => one step per epoch"


# ===========================================================================
# gradient_noise: Var[g_hat_B] ~ sigma**2 / B
# ===========================================================================
def test_gradient_noise_decreases_with_batch_size():
    X, y = toy_data(n=128, seed=4)
    model = MLP([2, 16, 1], activation="tanh", seed=5)
    noise_small = gradient_noise(model, X, y, batch_size=8, n_batches=40, seed=0)
    noise_large = gradient_noise(model, X, y, batch_size=64, n_batches=40, seed=0)
    assert noise_small > noise_large, (
        "gradient noise must shrink as batch size grows (sigma**2 / B):\n "
        f"B=8 -> {noise_small}, B=64 -> {noise_large}"
    )
    assert noise_large >= 0.0, "variance is non-negative"


def test_gradient_noise_does_not_move_model():
    X, y = toy_data(n=64, seed=6)
    model = MLP([2, 8, 1], activation="tanh", seed=7)
    before = [as_array(p).copy() for p in model.parameters()]
    _ = gradient_noise(model, X, y, batch_size=16, n_batches=10, seed=0)
    after = [as_array(p) for p in model.parameters()]
    for b, a in zip(before, after):
        assert np.allclose(b, a, atol=0.0), "gradient_noise must not update params"


# ===========================================================================
# epochs_to_threshold
# ===========================================================================
def test_epochs_to_threshold_first_crossing():
    hist = [1.0, 0.8, 0.5, 0.3, 0.25]
    assert epochs_to_threshold(hist, 0.55) == 3, "1-based index of first <= thr"
    assert epochs_to_threshold(hist, 1.0) == 1, "epoch 1 already meets threshold"
    assert epochs_to_threshold(hist, 0.1) == -1, "never reached => -1"


# ===========================================================================
# gradcheck: batched mse_loss backward vs central differences
# ===========================================================================
def test_batched_mse_backward_matches_central_diff():
    """One batched forward/backward through the MLP must produce a .grad equal to
    a central-difference gradient of the scalar batch loss w.r.t. a parameter."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 2))            # a batch of B=6
    yb = rng.choice([-1.0, 1.0], size=(6, 1))  # targets, shaped to broadcast

    model = MLP([2, 5, 1], activation="tanh", seed=11)
    params = model.parameters()
    p = params[0]  # gradcheck the first parameter tensor (the first layer's W)
    p0 = as_array(p).copy()

    # Analytic grad from autodiff.
    model.zero_grad() if hasattr(model, "zero_grad") else None
    for q in params:
        q.grad = np.zeros_like(as_array(q))
    loss = mse_loss(model(X), yb)
    loss.backward()
    g_analytic = as_array(p.grad)

    # Numeric grad: perturb p.data entrywise, recompute the forward loss only.
    def loss_at(p_data):
        p.data = np.asarray(p_data, dtype=float).reshape(p0.shape)
        out = mse_loss(model(X), yb)
        return float(as_array(out))

    g_num = central_diff(loss_at, p0.copy())
    p.data = p0  # restore

    assert g_analytic.shape == g_num.shape, "grad shape must match param shape"
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        "autodiff grad of batched mse_loss must match central differences:\n "
        f"analytic=\n{g_analytic}\n numeric=\n{g_num}"
    )
