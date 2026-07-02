"""Tests for Stage 15: First training loop.

What these tests check:
  * dataset fixtures (defined HERE, not in code.py) -- ``make_moons`` /
    ``make_spiral`` return Tensors with X ``(N, 2)``, y ``(N, 1)`` in {-1, +1},
    both classes present, deterministic under a fixed seed;
  * ``accuracy`` -- correct fraction, plain-float return, Tensor/ndarray and
    ``(N, 1)``/``(N,)`` inputs, sign(0) counts as wrong, ValueError on size
    mismatch, and it stays off-graph (no grads touched);
  * ``train`` -- returns the ``{"loss", "accuracy"}`` history (one float per
    epoch); a few epochs are EXACTLY equal (losses and final parameters) to a
    hand-rolled forward/loss/backward/step/zero loop on an identically-seeded
    model, which is the test that catches a shuffled step order or a stray
    extra update; loss falls; a provided optimizer is used (``lr`` ignored);
    grads are left zeroed (this is what catches a missing ``zero_grad``);
    ``(N,)`` and ``(N, 1)`` targets train identically; TypeError for
    non-Tensor inputs, ValueError for bad shapes;
  * end-to-end -- the MLP fits noiseless moons to >= 90% accuracy and the
    recorded history agrees with a fresh ``accuracy(model(X), y)``.

Run with:  pytest stages/stage_15_first_training_loop/test.py
"""
import os as _os
import sys as _sys

import numpy as np
import pytest

# --- Import the things under test, skipping cleanly if not ready yet. --------
# `code.py` ADDS this stage's `accuracy` / `train` (and ships `make_moons` /
# `make_spiral` / `plot_history`); it re-exports `Tensor` (stage_12), `MLP`
# (stage_11), `mse_loss` (stage_12) and `SGD` (stage_14) via dlfs.stage_import.
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
        MLP,
        SGD,
        Tensor,
        accuracy,
        mse_loss,
        plot_history,
        train,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_15 (or a dependency stage_08..14) not importable yet: {exc}",
        allow_module_level=True,
    )


# --------------------------------------------------------------------------- #
# helper: skip cleanly while a piece (this stage's TODOs or an earlier-stage
# dependency) is still raising NotImplementedError.
# --------------------------------------------------------------------------- #
def _requires(fn, *args, **kwargs):
    """Call fn; skip the test if a NotImplementedError bubbles up."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"depends on unimplemented piece: {e}")


def _T(data):
    """Build a Tensor, skipping the test while the engine is unimplemented."""
    return _requires(Tensor, data)


# --------------------------------------------------------------------------- #
# toy-dataset fixtures: the data the loop is trained on
# --------------------------------------------------------------------------- #
def make_moons(n=200, noise=0.1, seed=None):
    """Two interleaving half-moons: a 2-class problem no linear model solves.

    Returns ``(X, y)`` as Tensors: ``X`` of shape ``(n, 2)``, ``y`` of shape
    ``(n, 1)`` with labels in ``{-1.0, +1.0}``.  ``y`` is a column on purpose:
    it matches the model's ``(n, 1)`` output so ``pred - y`` never broadcasts
    to ``(n, n)``.  Deterministic given ``seed``; ``noise`` is the per-point
    Gaussian jitter.
    """
    rng = np.random.default_rng(seed)
    n_out = n // 2          # outer (upper) moon, label +1
    n_in = n - n_out        # inner (lower) moon, label -1

    # Outer moon: upper half-circle centered at the origin.
    t_out = np.linspace(0.0, np.pi, n_out)
    x_out = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)

    # Inner moon: lower half-circle, shifted right and down so the two interlock.
    t_in = np.linspace(0.0, np.pi, n_in)
    x_in = np.stack([1.0 - np.cos(t_in), 0.5 - np.sin(t_in)], axis=1)

    X = np.concatenate([x_out, x_in], axis=0).astype(np.float64)
    y = np.concatenate([np.ones(n_out), -np.ones(n_in)]).astype(np.float64)

    X += rng.normal(0.0, noise, size=X.shape)
    return Tensor(X), Tensor(y.reshape(-1, 1))


def make_spiral(n_per_class=100, n_classes=2, noise=0.2, seed=None):
    """Interleaved spiral arms: a harder non-linearly-separable 2-class problem.

    Returns ``(X, y)`` as Tensors: ``X`` of shape ``(n_per_class * n_classes, 2)``,
    ``y`` of shape ``(n_per_class * n_classes, 1)`` in ``{-1.0, +1.0}`` (even
    arms +1, odd arms -1).  Deterministic given ``seed``; ``noise`` jitters the
    angle of each point.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * n_classes, 2), dtype=np.float64)
    y = np.zeros(n_per_class * n_classes, dtype=np.float64)

    for c in range(n_classes):
        idx = slice(c * n_per_class, (c + 1) * n_per_class)
        r = np.linspace(0.0, 1.0, n_per_class)                       # radius 0 -> 1
        # Each arm starts a full turn offset from the last; noise jitters the angle.
        theta = (
            np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class)
            + rng.normal(0.0, noise, size=n_per_class)
        )
        X[idx] = np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1)
        y[idx] = 1.0 if c % 2 == 0 else -1.0

    return Tensor(X), Tensor(y.reshape(-1, 1))


# --------------------------------------------------------------------------- #
# dataset fixtures: sanity of the contract the loop is built against
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("maker,kwargs,n_expected", [
    (make_moons, dict(n=200, noise=0.1, seed=0), 200),
    (make_moons, dict(n=201, noise=0.1, seed=0), 201),          # odd n splits fine
    (make_spiral, dict(n_per_class=80, n_classes=2, noise=0.1, seed=1), 160),
])
def test_datasets_types_shapes_labels(maker, kwargs, n_expected):
    X, y = _requires(maker, **kwargs)
    assert isinstance(X, Tensor) and isinstance(y, Tensor)
    assert X.shape == (n_expected, 2)
    assert y.shape == (n_expected, 1), "y must be a (N, 1) column, not (N,)"
    assert X.data.dtype == np.float64
    labels = set(np.unique(y.data))
    assert labels == {-1.0, 1.0}, f"labels must be exactly {{-1, +1}}, got {labels}"


@pytest.mark.parametrize("maker,kwargs", [
    (make_moons, dict(n=120, noise=0.1)),
    (make_spiral, dict(n_per_class=60, noise=0.1)),
])
def test_datasets_deterministic_given_seed(maker, kwargs):
    Xa, ya = _requires(maker, seed=7, **kwargs)
    Xb, yb = _requires(maker, seed=7, **kwargs)
    assert np.array_equal(Xa.data, Xb.data)
    assert np.array_equal(ya.data, yb.data)
    Xc, _ = _requires(maker, seed=8, **kwargs)
    assert not np.allclose(Xa.data, Xc.data), "different seed must move the points"


def test_make_moons_noise_jitters_points():
    Xa, _ = _requires(make_moons, n=120, noise=0.0, seed=3)
    Xb, _ = _requires(make_moons, n=120, noise=0.3, seed=3)
    assert not np.allclose(Xa.data, Xb.data)


# --------------------------------------------------------------------------- #
# accuracy: off-graph sign agreement
# --------------------------------------------------------------------------- #
def test_accuracy_perfect_and_half():
    pred = _T([[2.0], [-3.0], [0.7], [-0.1]])
    y = _T([[1.0], [-1.0], [1.0], [-1.0]])
    acc = _requires(accuracy, pred, y)
    assert isinstance(acc, float)
    assert np.isclose(acc, 1.0)
    y_half = _T([[-1.0], [1.0], [1.0], [-1.0]])   # first two flipped
    assert np.isclose(_requires(accuracy, pred, y_half), 0.5)


def test_accuracy_mixed_types_and_shapes():
    """Tensor or ndarray, (N, 1) or (N,) -- all four combinations agree."""
    p_col = _T([[1.0], [-2.0], [3.0], [-4.0]])
    y_flat = np.array([1.0, -1.0, -1.0, -1.0])
    expected = 0.75
    assert np.isclose(_requires(accuracy, p_col, y_flat), expected)
    assert np.isclose(_requires(accuracy, p_col.data, y_flat), expected)
    assert np.isclose(_requires(accuracy, p_col, _T(y_flat)), expected)
    assert np.isclose(
        _requires(accuracy, p_col.data.reshape(-1), _T(y_flat.reshape(-1, 1))),
        expected,
    )


def test_accuracy_zero_prediction_counts_wrong():
    pred = _T([[0.0], [1.0]])
    y = _T([[1.0], [1.0]])
    assert np.isclose(_requires(accuracy, pred, y), 0.5)


def test_accuracy_size_mismatch_raises():
    pred = _T([[1.0], [2.0], [3.0]])
    y = _T([[1.0], [-1.0]])
    with pytest.raises(ValueError):
        _requires(accuracy, pred, y)


def test_accuracy_is_off_graph():
    """A metric must not touch grads or extend the graph."""
    pred = _T([[2.0], [-1.0]])
    y = _T([[1.0], [-1.0]])
    _requires(accuracy, pred, y)
    assert np.allclose(pred.grad, 0.0), "accuracy must not write grads"
    assert np.allclose(y.grad, 0.0), "accuracy must not write grads"


# --------------------------------------------------------------------------- #
# train: the canonical loop, pinned against a hand-rolled reference
# --------------------------------------------------------------------------- #
def test_train_returns_history_dict():
    X, y = _requires(make_moons, n=160, noise=0.1, seed=0)
    model = _requires(MLP, [2, 16, 1], activation="tanh", seed=0)
    history = _requires(train, model, X, y, lr=0.1, epochs=50)
    assert set(history.keys()) == {"loss", "accuracy"}
    assert len(history["loss"]) == 50
    assert len(history["accuracy"]) == 50
    assert all(isinstance(v, float) and np.isfinite(v) for v in history["loss"])
    assert all(
        isinstance(v, float) and 0.0 <= v <= 1.0 for v in history["accuracy"]
    )


def test_train_matches_manual_loop_exactly():
    """train() for 5 epochs == a hand-rolled forward/loss/backward/step/zero
    loop on an identically-seeded model.  This equality catches a shuffled
    step order and any stray extra update (a missing zero_grad is caught by
    test_train_leaves_grads_zeroed below)."""
    X, y = _requires(make_moons, n=64, noise=0.1, seed=2)

    model_a = _requires(MLP, [2, 8, 1], activation="tanh", seed=3)
    history = _requires(train, model_a, X, y, lr=0.1, epochs=5)

    model_b = _requires(MLP, [2, 8, 1], activation="tanh", seed=3)
    opt = _requires(SGD, model_b.parameters(), lr=0.1)
    manual_losses = []
    for _ in range(5):
        pred = model_b(X)
        loss = mse_loss(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        manual_losses.append(float(loss.data))

    assert np.allclose(history["loss"], manual_losses, rtol=0, atol=0), (
        "train() must be the canonical loop exactly:\n"
        f" train ={history['loss']}\n manual={manual_losses}"
    )
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert np.array_equal(p_a.data, p_b.data), (
            "parameters after train() differ from the manual loop "
            "(zero_grad missing or step order wrong?)"
        )


def test_train_loss_decreases():
    X, y = _requires(make_moons, n=160, noise=0.1, seed=0)
    model = _requires(MLP, [2, 16, 1], activation="tanh", seed=0)
    history = _requires(train, model, X, y, lr=0.1, epochs=50)
    losses = history["loss"]
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.9 * losses[0], "loss should drop meaningfully, not crawl"


def test_train_uses_provided_optimizer_and_ignores_lr():
    X, y = _requires(make_moons, n=120, noise=0.1, seed=0)
    model = _requires(MLP, [2, 8, 1], activation="tanh", seed=0)
    opt = _requires(SGD, model.parameters(), lr=0.05)
    # lr=1e6 would explode instantly if train built its own optimizer from it.
    history = _requires(train, model, X, y, lr=1e6, epochs=20, optimizer=opt)
    assert len(history["loss"]) == 20
    assert all(np.isfinite(v) for v in history["loss"])
    assert history["loss"][-1] <= history["loss"][0]


def test_train_leaves_grads_zeroed():
    """The loop ends on zero_grad, so the caller can backward() fresh."""
    X, y = _requires(make_moons, n=40, noise=0.1, seed=0)
    model = _requires(MLP, [2, 4, 1], activation="tanh", seed=0)
    _requires(train, model, X, y, lr=0.1, epochs=3)
    for p in model.parameters():
        assert np.allclose(p.grad, 0.0)
        assert p.grad.shape == p.data.shape


def test_train_accepts_flat_y_identically():
    """y as (N,) must train exactly like the same y as (N, 1)."""
    X, y = _requires(make_moons, n=60, noise=0.1, seed=1)
    y_flat = Tensor(y.data.reshape(-1))

    model_col = _requires(MLP, [2, 8, 1], activation="tanh", seed=4)
    hist_col = _requires(train, model_col, X, y, lr=0.1, epochs=5)
    model_flat = _requires(MLP, [2, 8, 1], activation="tanh", seed=4)
    hist_flat = _requires(train, model_flat, X, y_flat, lr=0.1, epochs=5)

    assert np.allclose(hist_col["loss"], hist_flat["loss"], rtol=0, atol=0), (
        "(N,) y must be normalized to (N, 1) -- did pred - y broadcast to (N, N)?"
    )


def test_train_rejects_non_tensor_inputs():
    X, y = _requires(make_moons, n=40, noise=0.1, seed=0)
    model = _requires(MLP, [2, 8, 1], activation="tanh", seed=0)
    with pytest.raises(Exception):
        _requires(train, model, X.data, y, epochs=1)         # X is ndarray
    with pytest.raises(Exception):
        _requires(train, model, X, y.data, epochs=1)         # y is ndarray
    with pytest.raises(Exception):
        _requires(train, model, X.data, y.data, epochs=1)    # both


def test_train_rejects_bad_shapes():
    X, y = _requires(make_moons, n=40, noise=0.1, seed=0)
    model = _requires(MLP, [2, 8, 1], activation="tanh", seed=0)
    with pytest.raises(ValueError):
        _requires(train, model, X, Tensor(np.ones((39, 1))), epochs=1)  # N mismatch
    with pytest.raises(ValueError):
        _requires(train, model, X, Tensor(np.ones((40, 2))), epochs=1)  # y not a column
    with pytest.raises(ValueError):
        _requires(train, model, Tensor(np.ones(40)), y, epochs=1)       # X not 2-D


# --------------------------------------------------------------------------- #
# end-to-end: the loop actually learns
# --------------------------------------------------------------------------- #
def test_train_fits_noiseless_moons():
    X, y = _requires(make_moons, n=200, noise=0.0, seed=0)
    model = _requires(MLP, [2, 16, 1], activation="tanh", seed=0)
    history = _requires(train, model, X, y, lr=0.2, epochs=400)
    final_acc = _requires(accuracy, model(X), y)
    assert final_acc >= 0.90, (
        f"expected >= 90% train accuracy on clean moons, got {final_acc:.3f}"
    )
    # the recorded history must agree with a fresh evaluation of the metric.
    assert np.isclose(history["accuracy"][-1], final_acc, atol=0.05)


# --------------------------------------------------------------------------- #
# plot helper (provided): smoke test, skipped when matplotlib is absent
# --------------------------------------------------------------------------- #
def test_plot_history_saves_figure(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    history = {"loss": [1.0, 0.5, 0.25], "accuracy": [0.5, 0.75, 1.0]}
    out = tmp_path / "history.png"
    fig = plot_history(history, path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None
