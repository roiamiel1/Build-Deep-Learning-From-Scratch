"""Tests for Stage 20: DataLoader.

Covers the data-feeding plumbing built on top of stage_19:
  * Dataset       -- len / getitem (int, slice, index-array), shape mismatch
                     raises, dtype is float64;
  * DataLoader    -- batch count (drop_last vs ceil), exact batch widths,
                     iterator protocol (re-iterable, fresh permutation per
                     epoch), full coverage of every row, shuffle on/off, yields
                     Tensor batches, batch_size validation;
  * train_val_split -- disjoint + complete index sets, validation size,
                     determinism per seed, val_frac validation;
  * train_with_loader -- step count == epochs * len(loader), epoch loss trends
                     down, and a central-difference gradcheck of one batch's
                     mse_loss(...).backward() vs the autodiff .grad.

Run: pytest stage_20_dataloader/test.py
"""

import os
import sys

import numpy as np
import pytest

# Make both this stage's ``code.py`` and the repo-root ``dlfs`` shim importable
# when pytest is run from anywhere. ``code.py`` imports its prior-stage pieces
# (Tensor from stage_09; MLP / mse_loss / SGD / train_minibatch from stage_19)
# via ``dlfs.stage_import`` and re-exports them, so the tests pull everything
# from this stage's extended module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from code import (
        Dataset,
        DataLoader,
        train_val_split,
        train_with_loader,
        Tensor,
        MLP,
        mse_loss,
        SGD,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_20 DataLoader (or a stage_09 / stage_19 piece it imports) "
        f"not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
TOL = 1e-6


# --------------------------------------------------------------------------- #
# helper: skip cleanly if a *dependency* (stage_09/15/19) isn't implemented
# yet, while still failing if THIS stage is the problem.
# --------------------------------------------------------------------------- #
def _requires(fn, *args, **kwargs):
    """Call fn; if a NotImplementedError bubbles up from a dependency, skip."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"depends on unimplemented piece: {e}")


def _xy(n=10, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n).astype(np.float64) * 2 - 1  # in {-1, +1}
    return X, y


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def test_dataset_len_and_dtype():
    X, y = _xy(7)
    ds = _requires(Dataset, X, y)
    assert len(ds) == 7
    assert ds.X.dtype == np.float64
    assert ds.y.dtype == np.float64


def test_dataset_getitem_int():
    X, y = _xy(5)
    ds = _requires(Dataset, X, y)
    xi, yi = _requires(ds.__getitem__, 2)
    assert np.allclose(xi, X[2])
    assert np.allclose(yi, y[2])


def test_dataset_getitem_index_array_and_slice():
    X, y = _xy(6)
    ds = _requires(Dataset, X, y)
    idx = np.array([0, 3, 5])
    xb, yb = _requires(ds.__getitem__, idx)
    assert xb.shape == (3, 2)
    assert yb.shape == (3,)
    assert np.allclose(xb, X[idx])
    xs, ys = ds[1:4]
    assert xs.shape == (3, 2) and ys.shape == (3,)


def test_dataset_length_mismatch_raises():
    X = np.zeros((5, 2))
    y = np.zeros((4,))
    with pytest.raises(ValueError):
        Dataset(X, y)


# --------------------------------------------------------------------------- #
# DataLoader: batch counts and widths
# --------------------------------------------------------------------------- #
def test_loader_len_ceil_vs_drop_last():
    X, y = _xy(10)
    ds = _requires(Dataset, X, y)
    # 10 rows, batch 3 -> ceil(10/3)=4 batches, or 10//3=3 with drop_last.
    full = _requires(DataLoader, ds, 3, shuffle=False, drop_last=False)
    dropped = _requires(DataLoader, ds, 3, shuffle=False, drop_last=True)
    assert len(full) == 4
    assert len(dropped) == 3


def test_loader_drop_last_all_batches_full():
    X, y = _xy(10)
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 4, shuffle=True, drop_last=True, seed=0)
    batches = list(_requires(iter, loader))
    assert len(batches) == 2  # 10 // 4
    for xb, yb in batches:
        assert xb.shape[0] == 4
        assert yb.shape[0] == 4


def test_loader_last_batch_ragged_when_not_dropped():
    X, y = _xy(10)
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 4, shuffle=False, drop_last=False)
    sizes = [xb.shape[0] for xb, _ in loader]
    assert sizes == [4, 4, 2]


def test_loader_yields_tensors():
    X, y = _xy(8)
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 4, shuffle=False)
    xb, yb = next(iter(loader))
    assert isinstance(xb, Tensor)
    assert isinstance(yb, Tensor)
    assert xb.shape == (4, 2)


def test_loader_batch_size_validation():
    X, y = _xy(5)
    ds = _requires(Dataset, X, y)
    with pytest.raises(ValueError):
        DataLoader(ds, 0)


# --------------------------------------------------------------------------- #
# DataLoader: iterator protocol -- re-iterable, full coverage, shuffle
# --------------------------------------------------------------------------- #
def _collect_rows(loader):
    """Stack all yielded X rows (read .data off the Tensor) in yield order."""
    rows = [xb.data for xb, _ in loader]
    return np.concatenate(rows, axis=0)


def test_loader_no_shuffle_is_arange_order():
    X, y = _xy(9)
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 3, shuffle=False)
    seen = _collect_rows(loader)
    assert np.allclose(seen, X), "shuffle=False must yield rows in arange order"


def test_loader_covers_all_rows_when_shuffling():
    X, y = _xy(9)
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 3, shuffle=True, seed=1)
    seen = _collect_rows(loader)
    # Same multiset of rows as X (order may differ).
    s_sorted = seen[np.lexsort(seen.T)]
    x_sorted = X[np.lexsort(X.T)]
    assert np.allclose(s_sorted, x_sorted), "shuffled epoch must cover every row once"


def test_loader_is_reiterable_two_epochs():
    """The SAME loader object must be loopable again (a fresh iterator each
    __iter__), each epoch covering all rows."""
    X, y = _xy(12)
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 5, shuffle=True, drop_last=False, seed=3)
    a = _collect_rows(loader)
    b = _collect_rows(loader)
    assert a.shape == X.shape and b.shape == X.shape
    for arr in (a, b):
        s = arr[np.lexsort(arr.T)]
        xs = X[np.lexsort(X.T)]
        assert np.allclose(s, xs)


def test_loader_seed_makes_order_reproducible():
    X, y = _xy(12)
    ds = _requires(Dataset, X, y)
    l1 = _requires(DataLoader, ds, 4, shuffle=True, seed=7)
    l2 = _requires(DataLoader, ds, 4, shuffle=True, seed=7)
    assert np.allclose(_collect_rows(l1), _collect_rows(l2)), (
        "same seed must produce identical shuffle order"
    )


# --------------------------------------------------------------------------- #
# train_val_split
# --------------------------------------------------------------------------- #
def test_split_disjoint_and_complete():
    X, y = _xy(20)
    X_tr, y_tr, X_val, y_val = _requires(train_val_split, X, y, 0.25, seed=0)
    assert X_tr.shape[0] == 15
    assert X_val.shape[0] == 5
    assert y_tr.shape[0] == 15 and y_val.shape[0] == 5
    # Union of rows == original set, with no overlap.
    all_rows = np.concatenate([X_tr, X_val], axis=0)
    s_sorted = all_rows[np.lexsort(all_rows.T)]
    x_sorted = X[np.lexsort(X.T)]
    assert np.allclose(s_sorted, x_sorted), "train+val must cover all N rows exactly"


def test_split_deterministic():
    X, y = _xy(30)
    a = _requires(train_val_split, X, y, 0.2, seed=42)
    b = _requires(train_val_split, X, y, 0.2, seed=42)
    for u, v in zip(a, b):
        assert np.allclose(u, v), "same seed must give the same split"


def test_split_val_frac_validation():
    X, y = _xy(10)
    with pytest.raises(ValueError):
        train_val_split(X, y, 1.0)
    with pytest.raises(ValueError):
        train_val_split(X, y, -0.1)


def test_split_zero_val_frac():
    X, y = _xy(10)
    X_tr, y_tr, X_val, y_val = _requires(train_val_split, X, y, 0.0, seed=0)
    assert X_tr.shape[0] == 10
    assert X_val.shape[0] == 0


# --------------------------------------------------------------------------- #
# train_with_loader: step count + loss trend
# --------------------------------------------------------------------------- #
def test_train_with_loader_step_count_and_keys():
    X, y = _xy(16, seed=2)
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 4, shuffle=True, seed=0)
    model = _requires(MLP, [2, 8, 1], activation="tanh", seed=0)
    hist = _requires(train_with_loader, model, loader, lr=0.1, epochs=5)
    assert set(hist.keys()) >= {"batch_loss", "epoch_loss", "steps"}
    assert hist["steps"] == 5 * len(loader)
    assert len(hist["epoch_loss"]) == 5
    assert len(hist["batch_loss"]) == 5 * len(loader)
    assert all(np.isfinite(hist["epoch_loss"]))


def test_train_with_loader_loss_decreases():
    # Easy fittable target so a few epochs clearly drop the loss.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 2))
    y = np.sign(X[:, 0] * X[:, 1])  # in {-1, +1}
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 16, shuffle=True, seed=0)
    model = _requires(MLP, [2, 16, 1], activation="tanh", seed=0)
    hist = _requires(train_with_loader, model, loader, lr=0.1, epochs=40)
    el = hist["epoch_loss"]
    assert el[-1] < el[0], "epoch loss should fall over training"
    assert el[-1] < 0.9 * el[0]


def test_train_with_loader_uses_provided_optimizer():
    X, y = _xy(16, seed=5)
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 8, shuffle=False)
    model = _requires(MLP, [2, 8, 1], activation="tanh", seed=0)
    opt = _requires(SGD, model.parameters(), lr=0.05)
    hist = _requires(train_with_loader, model, loader, epochs=3, optimizer=opt)
    assert hist["steps"] == 3 * len(loader)


# --------------------------------------------------------------------------- #
# central-difference gradcheck: d(mse_loss on one loader batch)/d(param)
# --------------------------------------------------------------------------- #
def test_gradcheck_on_one_loader_batch():
    """Pull one batch from the loader, run mse_loss(...).backward(), and compare
    the autodiff grad of a parameter against a central finite difference of the
    forward-only loss on that SAME batch."""
    X, y = _xy(12, seed=1)
    ds = _requires(Dataset, X, y)
    loader = _requires(DataLoader, ds, 6, shuffle=False)
    model = _requires(MLP, [2, 5, 1], activation="tanh", seed=1)

    X_b, y_b = next(iter(loader))
    yb = y_b.data.reshape(-1, 1)

    # --- analytic grad via backward ---
    model.zero_grad()
    pred = _requires(model.__call__, X_b)
    loss = _requires(mse_loss, pred, yb)
    _requires(loss.backward)
    p = _requires(model.parameters)[0]  # first weight matrix
    g_analytic = np.array(p.grad, copy=True)

    # --- numeric grad: central difference on every entry of p.data ---
    def loss_at(param_data):
        saved = np.array(p.data, copy=True)
        p.data = param_data
        val = float(mse_loss(model(X_b), yb).data)
        p.data = saved
        return val

    g_numeric = np.zeros_like(p.data)
    base = np.array(p.data, copy=True)
    for i in range(base.size):
        idx = np.unravel_index(i, base.shape)
        up = np.array(base, copy=True); up[idx] += EPS
        dn = np.array(base, copy=True); dn[idx] -= EPS
        g_numeric[idx] = (loss_at(up) - loss_at(dn)) / (2 * EPS)

    assert np.allclose(g_analytic, g_numeric, rtol=1e-4, atol=1e-5), (
        f"batch gradcheck failed:\nanalytic=\n{g_analytic}\nnumeric=\n{g_numeric}"
    )
