"""Tests for Stage 33: Capstone -- MNIST.

Covers the integration capstone that points the stage_32 ``mytorch`` framework
at MNIST:
  * load_mnist_idx     -- round-trips a synthetic IDX byte buffer (plain + gz),
                          and rejects wrong magic numbers;
  * preprocess         -- (N,784) in [0,1] when flatten, else (N,1,28,28);
  * make_loaders       -- yields Tensor batches of the right shape, optional
                          validation split (stage_32 DataLoader);
  * build_mlp          -- correct logit shape (B, 10) from a Sequential MLP;
  * evaluate           -- {"loss","acc"} on a loader, perfect classifier -> 1.0;
  * train_mnist        -- step count, falling loss, >= 0.90 acc on a tiny set;
  * run_capstone       -- end-to-end synthetic fallback returns a test_acc;
  * gradcheck          -- one-batch cross_entropy_loss(model(X),y).backward()
                          vs central differences on a parameter.

Run: pytest stage_33_capstone_mnist/test.py
"""

import gzip
import importlib.util as _ilu
import os
import struct
import sys
import tempfile

import numpy as np
import pytest

# Put the curriculum root on sys.path so this stage's code.py can do
# ``from dlfs import stage_import``. We load code.py BY FILE PATH (below) rather
# than ``import code`` to avoid shadowing the stdlib ``code`` module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # curriculum root, so `import dlfs` works
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_code():
    """Load this stage's code.py by file path (it imports stage_32 via dlfs)."""
    path = os.path.join(_HERE, "code.py")
    spec = _ilu.spec_from_file_location("_stage33_code", path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    CODE = _load_code()
    load_mnist_idx = CODE.load_mnist_idx
    preprocess = CODE.preprocess
    make_loaders = CODE.make_loaders
    build_mlp = CODE.build_mlp
    evaluate = CODE.evaluate
    train_mnist = CODE.train_mnist
    run_capstone = CODE.run_capstone
    make_synthetic_digits = CODE.make_synthetic_digits
    Tensor = CODE.Tensor
    cross_entropy_loss = CODE.cross_entropy_loss
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_33 capstone / stage_32 framework not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
TOL = 1e-4


# --------------------------------------------------------------------------- #
# helper: skip cleanly if a *dependency* (stage_09/12/13/18/20/26) isn't
# implemented yet, while still failing if THIS stage is the problem.
# --------------------------------------------------------------------------- #
def _requires(fn, *args, **kwargs):
    """Call fn; if a NotImplementedError bubbles up from a dependency, skip."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"depends on unimplemented piece: {e}")


# --------------------------------------------------------------------------- #
# build a synthetic IDX byte buffer (the canonical MNIST on-disk format)
# --------------------------------------------------------------------------- #
def _write_idx_images(path, X_uint8, gz=False):
    n, rows, cols = X_uint8.shape
    header = struct.pack(">IIII", 2051, n, rows, cols)
    payload = header + X_uint8.astype(np.uint8).tobytes()
    opener = gzip.open if gz else open
    with opener(path, "wb") as f:
        f.write(payload)


def _write_idx_labels(path, y_uint8, gz=False):
    n = y_uint8.shape[0]
    header = struct.pack(">II", 2049, n)
    payload = header + y_uint8.astype(np.uint8).tobytes()
    opener = gzip.open if gz else open
    with opener(path, "wb") as f:
        f.write(payload)


# --------------------------------------------------------------------------- #
# load_mnist_idx
# --------------------------------------------------------------------------- #
def test_load_idx_roundtrip_plain():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, size=(5, 28, 28)).astype(np.uint8)
    y = rng.integers(0, 10, size=(5,)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        ip = os.path.join(d, "img-idx3-ubyte")
        lp = os.path.join(d, "lab-idx1-ubyte")
        _write_idx_images(ip, X)
        _write_idx_labels(lp, y)
        Xo, yo = _requires(load_mnist_idx, ip, lp)
    assert Xo.shape == (5, 28, 28), f"images must be (N,28,28), got {Xo.shape}"
    assert yo.shape == (5,), f"labels must be (N,), got {yo.shape}"
    assert np.allclose(Xo, X.astype(np.float64)), "pixel values must round-trip"
    assert np.array_equal(np.asarray(yo).astype(int), y.astype(int)), "labels must round-trip"


def test_load_idx_roundtrip_gzip():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 256, size=(4, 28, 28)).astype(np.uint8)
    y = rng.integers(0, 10, size=(4,)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        ip = os.path.join(d, "img-idx3-ubyte.gz")
        lp = os.path.join(d, "lab-idx1-ubyte.gz")
        _write_idx_images(ip, X, gz=True)
        _write_idx_labels(lp, y, gz=True)
        Xo, yo = _requires(load_mnist_idx, ip, lp)
    assert Xo.shape == (4, 28, 28)
    assert np.allclose(Xo, X.astype(np.float64)), "gz pixel values must round-trip"


def test_load_idx_bad_magic_raises():
    with tempfile.TemporaryDirectory() as d:
        ip = os.path.join(d, "bad-img")
        lp = os.path.join(d, "lab-idx1-ubyte")
        # wrong image magic
        with open(ip, "wb") as f:
            f.write(struct.pack(">IIII", 9999, 1, 28, 28) + bytes(28 * 28))
        _write_idx_labels(lp, np.array([0], dtype=np.uint8))
        with pytest.raises((ValueError,)):
            load_mnist_idx(ip, lp)


# --------------------------------------------------------------------------- #
# preprocess
# --------------------------------------------------------------------------- #
def test_preprocess_flatten_range_and_shape():
    X = np.full((3, 28, 28), 255.0)
    out = _requires(preprocess, X, flatten=True)
    assert out.shape == (3, 784), f"flatten must give (N,784), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "normalized pixels must be in [0,1]"
    assert np.allclose(out, 1.0), "255 -> 1.0 after /255 normalization"


def test_preprocess_channels_first_shape():
    X = np.zeros((2, 28, 28))
    out = _requires(preprocess, X, flatten=False)
    assert out.shape == (2, 1, 28, 28), f"cnn input must be (N,1,28,28), got {out.shape}"


def test_preprocess_no_normalize():
    X = np.full((1, 28, 28), 128.0)
    out = _requires(preprocess, X, flatten=True, normalize=False)
    assert np.allclose(out, 128.0), "normalize=False must leave raw values"


# --------------------------------------------------------------------------- #
# make_loaders
# --------------------------------------------------------------------------- #
def test_make_loaders_mlp_batch_shape():
    X, y = make_synthetic_digits(n_per_class=6, n_classes=10, seed=0)
    train_loader, val_loader = _requires(
        make_loaders, X, y, batch_size=16, flatten=True, val_frac=0.0, seed=0
    )
    assert val_loader is None, "val_frac=0 must give val_loader=None"
    Xb, yb = next(iter(train_loader))
    assert isinstance(Xb, Tensor) and isinstance(yb, Tensor)
    assert Xb.shape[1] == 784, f"MLP batch features must be 784, got {Xb.shape}"


def test_make_loaders_cnn_batch_shape():
    X, y = make_synthetic_digits(n_per_class=4, n_classes=10, seed=0)
    train_loader, _ = _requires(
        make_loaders, X, y, batch_size=8, flatten=False, val_frac=0.0, seed=0
    )
    Xb, _ = next(iter(train_loader))
    assert Xb.shape[1:] == (1, 28, 28), f"CNN batch must be (B,1,28,28), got {Xb.shape}"


def test_make_loaders_validation_split():
    X, y = make_synthetic_digits(n_per_class=10, n_classes=10, seed=0)  # N=100
    train_loader, val_loader = _requires(
        make_loaders, X, y, batch_size=16, flatten=True, val_frac=0.2, seed=0
    )
    assert val_loader is not None, "val_frac>0 must give a validation loader"
    n_train = sum(xb.shape[0] for xb, _ in train_loader)
    n_val = sum(xb.shape[0] for xb, _ in val_loader)
    assert n_train + n_val == 100, "split must cover all rows"
    assert n_val == 20, f"20% of 100 should be 20 val rows, got {n_val}"


# --------------------------------------------------------------------------- #
# model builders: logit shape
# --------------------------------------------------------------------------- #
def test_build_mlp_logit_shape():
    net = _requires(build_mlp, seed=0)
    X = Tensor(np.zeros((7, 784)))
    logits = _requires(net.__call__, X)
    assert logits.shape == (7, 10), f"MLP must output (B,10), got {logits.shape}"


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def test_evaluate_keys_and_range():
    X, y = make_synthetic_digits(n_per_class=4, n_classes=10, seed=1)
    train_loader, _ = _requires(
        make_loaders, X, y, batch_size=16, flatten=True, val_frac=0.0, seed=0
    )
    net = _requires(build_mlp, seed=0)
    m = _requires(evaluate, net, train_loader)
    assert set(m.keys()) >= {"loss", "acc"}
    assert 0.0 <= m["acc"] <= 1.0, "accuracy must be a fraction in [0,1]"
    assert np.isfinite(m["loss"]), "loss must be finite"


# --------------------------------------------------------------------------- #
# train_mnist: step count, falling loss, target accuracy on a tiny set
# --------------------------------------------------------------------------- #
def test_train_mnist_step_count_and_keys():
    X, y = make_synthetic_digits(n_per_class=8, n_classes=10, seed=2)
    train_loader, _ = _requires(
        make_loaders, X, y, batch_size=16, flatten=True, val_frac=0.0, seed=0
    )
    net = _requires(build_mlp, seed=0)
    hist = _requires(train_mnist, net, train_loader, epochs=3, lr=1e-3)
    assert set(hist.keys()) >= {"train_loss", "val_loss", "val_acc", "steps"}
    assert hist["steps"] == 3 * len(train_loader), "steps must be epochs * batches"
    assert len(hist["train_loss"]) == 3
    assert all(np.isfinite(hist["train_loss"]))


def test_train_mnist_converges_on_small_set():
    # Low-noise, separable synthetic digits so a few epochs clearly learn it.
    X, y = make_synthetic_digits(n_per_class=24, n_classes=10, noise=0.2, seed=3)
    train_loader, _ = _requires(
        make_loaders, X, y, batch_size=32, flatten=True, val_frac=0.0, seed=0
    )
    net = _requires(build_mlp, seed=0)
    hist = _requires(train_mnist, net, train_loader, epochs=25, lr=2e-3)
    tl = hist["train_loss"]
    assert tl[-1] < tl[0], "training loss should fall"
    m = _requires(evaluate, net, train_loader)
    assert m["acc"] >= 0.90, f"should reach >=0.90 train acc, got {m['acc']:.3f}"


def test_train_mnist_records_validation():
    X, y = make_synthetic_digits(n_per_class=10, n_classes=10, noise=0.2, seed=4)
    train_loader, val_loader = _requires(
        make_loaders, X, y, batch_size=16, flatten=True, val_frac=0.25, seed=0
    )
    net = _requires(build_mlp, seed=0)
    hist = _requires(
        train_mnist, net, train_loader, epochs=2, lr=1e-3, val_loader=val_loader
    )
    assert len(hist["val_acc"]) == 2, "val metrics recorded per epoch when val_loader given"
    assert all(0.0 <= a <= 1.0 for a in hist["val_acc"])


# --------------------------------------------------------------------------- #
# run_capstone: end-to-end synthetic fallback
# --------------------------------------------------------------------------- #
def test_run_capstone_synthetic_fallback_mlp():
    # No data_dir -> synthetic fallback; must still return a sane test_acc.
    out = _requires(
        run_capstone, None, model="mlp", epochs=15, batch_size=32, lr=2e-3, seed=0
    )
    assert set(out.keys()) >= {"test_acc", "test_loss", "history", "model", "source"}
    assert out["source"] == "synthetic", "no data_dir must use the synthetic fallback"
    assert out["model"] == "mlp"
    assert 0.0 <= out["test_acc"] <= 1.0
    assert out["test_acc"] >= 0.5, f"capstone should learn synthetic digits, got {out['test_acc']:.3f}"


def test_run_capstone_loads_real_idx(tmp_path):
    # Drop a tiny synthetic-but-real IDX dataset on disk and confirm the
    # capstone reads it (source == "mnist") rather than falling back.
    rng = np.random.default_rng(5)
    Xtr = rng.integers(0, 256, size=(40, 28, 28)).astype(np.uint8)
    ytr = rng.integers(0, 10, size=(40,)).astype(np.uint8)
    Xte = rng.integers(0, 256, size=(10, 28, 28)).astype(np.uint8)
    yte = rng.integers(0, 10, size=(10,)).astype(np.uint8)
    d = str(tmp_path)
    _write_idx_images(os.path.join(d, "train-images-idx3-ubyte"), Xtr)
    _write_idx_labels(os.path.join(d, "train-labels-idx1-ubyte"), ytr)
    _write_idx_images(os.path.join(d, "t10k-images-idx3-ubyte"), Xte)
    _write_idx_labels(os.path.join(d, "t10k-labels-idx1-ubyte"), yte)
    out = _requires(
        run_capstone, d, model="mlp", epochs=1, batch_size=16, lr=1e-3, seed=0
    )
    assert out["source"] == "mnist", "must load the IDX files from data_dir"
    assert 0.0 <= out["test_acc"] <= 1.0


# --------------------------------------------------------------------------- #
# central-difference gradcheck on one batch (the integration property)
# --------------------------------------------------------------------------- #
def test_gradcheck_one_batch_mlp():
    """One batch: cross_entropy_loss(model(X), y).backward() vs central
    difference on the first weight matrix. This is the load-bearing property of
    the capstone: gradients flow end to end through the imported layers."""
    X, y = make_synthetic_digits(n_per_class=2, n_classes=10, seed=6)
    train_loader, _ = _requires(
        make_loaders, X, y, batch_size=8, flatten=True, val_frac=0.0, seed=0
    )
    net = _requires(build_mlp, seed=0)
    X_b, y_b = next(iter(train_loader))

    # analytic grad via backward
    net.zero_grad()
    logits = _requires(net.__call__, X_b)
    loss = _requires(cross_entropy_loss, logits, y_b)
    _requires(loss.backward)
    p = _requires(net.parameters)[0]  # first weight matrix
    g_analytic = np.array(p.grad, copy=True)

    # numeric grad: central difference on a few entries of p.data (full sweep
    # of a 784x128 matrix is wasteful; sample a deterministic subset).
    base = np.array(p.data, copy=True)

    def loss_at(param_data):
        saved = np.array(p.data, copy=True)
        p.data = param_data
        val = float(cross_entropy_loss(net(X_b), y_b).data)
        p.data = saved
        return val

    rng = np.random.default_rng(0)
    flat_idx = rng.choice(base.size, size=min(20, base.size), replace=False)
    for fi in flat_idx:
        idx = np.unravel_index(fi, base.shape)
        up = np.array(base, copy=True); up[idx] += EPS
        dn = np.array(base, copy=True); dn[idx] -= EPS
        num = (loss_at(up) - loss_at(dn)) / (2 * EPS)
        assert abs(num - g_analytic[idx]) <= TOL + 1e-3 * abs(num), (
            f"gradcheck failed at {idx}: analytic={g_analytic[idx]:.6e} "
            f"numeric={num:.6e}"
        )
