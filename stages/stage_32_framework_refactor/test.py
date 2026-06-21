"""Tests for Stage 32: Framework refactor (``mytorch``).

This stage introduces no new math, so the testing has two halves:

  1. ORGANIZATIONAL contract -- the parts every framework needs:
       * Parameter is a grad-requiring leaf Tensor.
       * Module.parameters() recursively gathers every Parameter exactly once,
         in stable order, through nested Sequential.
       * zero_grad() clears all grads; train()/eval() toggle `training`
         recursively.
       * an optimizer built from model.parameters() updates the SAME tensor
         objects the model holds (in place).
       * a tiny Sequential trains with Adam + CrossEntropyLoss.

  2. CORRECTNESS via central-difference gradient checking on Linear:
         df/dx ~= (f(x + eps) - f(x - eps)) / (2 * eps)
     compared against the analytic grad from Tensor.backward() (stage_09).

If a prior stage (Tensor / Dense / losses / optimizers / DataLoader) or this
stage's framework is not implemented yet, the suite SKIPS cleanly.

Run with:  pytest stage_32_framework_refactor/test.py
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# Make the shared `dlfs` shim importable (it lives at the curriculum root).
sys.path.insert(0, _ROOT)

# --- Import the framework under test, skipping cleanly if not ready. ---------
try:
    from code import (
        Adam,
        CrossEntropyLoss,
        DataLoader,
        Dataset,
        Linear,
        Module,
        MSELoss,
        Parameter,
        ReLU,
        SGD,
        Sequential,
        Tensor,
    )
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_32 mytorch / a prior stage not importable yet: {exc}",
        allow_module_level=True,
    )

RNG = np.random.default_rng(32)
EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- Helpers -----------------------------------------------------------------
def _array(t):
    """Underlying ndarray of a Tensor (or pass an array through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def _scalar(t):
    """Python float held by a 0-d / size-1 loss Tensor."""
    a = _array(t)
    return float(a.reshape(-1)[0]) if a.size == 1 else float(np.sum(a))


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


def _try(make):
    """Run `make()`, skipping cleanly if the relevant piece is unimplemented."""
    try:
        return make()
    except NotImplementedError as exc:  # pragma: no cover
        pytest.skip(f"piece not implemented yet: {exc}")


# ===========================================================================
# Parameter
# ===========================================================================
def test_parameter_is_grad_requiring_leaf_tensor():
    p = _try(lambda: Parameter(np.array([[1.0, 2.0], [3.0, 4.0]])))
    assert isinstance(p, Tensor), "Parameter must subclass the stage_09 Tensor"
    assert getattr(p, "requires_grad", False) is True, "Parameter.requires_grad must be True"
    assert _array(p).shape == (2, 2)
    # It is a leaf: no parents / operation from being constructed.
    assert getattr(p, "operation", "") == "", "a fresh Parameter must be a leaf"


# ===========================================================================
# Module registration + recursive parameters()
# ===========================================================================
def _tiny_net():
    return Sequential(
        Linear(3, 4, seed=0),
        ReLU(),
        Linear(4, 2, seed=1),
    )


def test_parameters_collected_recursively_and_in_order():
    net = _try(_tiny_net)
    params = _try(net.parameters)
    # Two Linear layers, each W + b  => 4 parameter tensors.
    assert len(params) == 4, f"expected 4 params (W,b,W,b), got {len(params)}"
    shapes = [_array(p).shape for p in params]
    assert shapes == [(3, 4), (4,), (4, 2), (2,)], f"params out of order: {shapes}"
    for p in params:
        assert isinstance(p, Parameter), "parameters() must return Parameter leaves"


def test_parameters_deduplicated_by_identity():
    # A nested Module that registers the SAME Parameter twice (weight tying).
    shared = _try(lambda: Linear(2, 2, seed=7))

    class Tied(Module):
        def __init__(self):
            super().__init__()
            self.a = shared
            self.b = shared  # same object registered again

        def forward(self, x):
            return self.b(self.a(x))

    net = _try(Tied)
    params = _try(net.parameters)
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids)), "parameters() must not return duplicates"
    # Only the two unique params of `shared`.
    assert len(params) == 2


def test_zero_grad_clears_all_parameter_grads():
    net = _try(_tiny_net)
    for p in net.parameters():
        p.grad = np.ones_like(_array(p))
    _try(net.zero_grad)
    for p in net.parameters():
        assert np.allclose(_array(p.grad), 0.0), "zero_grad must clear every grad"


def test_train_eval_toggle_recursively():
    net = _try(_tiny_net)
    _try(net.eval)
    assert net.training is False
    for m in net._modules.values():
        assert m.training is False, "eval() must propagate to children"
    _try(net.train)
    assert net.training is True
    for m in net._modules.values():
        assert m.training is True, "train() must propagate to children"


# ===========================================================================
# Forward shapes
# ===========================================================================
def test_linear_forward_shape():
    lin = _try(lambda: Linear(3, 5, seed=2))
    x = Tensor(RNG.standard_normal((6, 3)))
    out = _try(lambda: lin(x))
    assert _array(out).shape == (6, 5)


def test_sequential_forward_shape_and_chaining():
    net = _try(_tiny_net)
    x = Tensor(RNG.standard_normal((4, 3)))
    out = _try(lambda: net(x))
    assert _array(out).shape == (4, 2)


def test_relu_forward_matches_numpy():
    r = _try(ReLU)
    x = Tensor(np.array([[-1.0, 0.0, 2.0], [3.0, -4.0, 0.5]]))
    out = _try(lambda: r(x))
    assert np.allclose(_array(out), np.maximum(_array(x), 0.0))


# ===========================================================================
# Gradient checking: Linear (the only module with parameters + a gradient)
# ===========================================================================
def test_linear_gradcheck_wrt_W_and_b():
    lin = _try(lambda: Linear(3, 2, seed=5))
    X = RNG.standard_normal((4, 3))

    # A fixed downstream weighting so the scalar loss depends on every output.
    G = RNG.standard_normal((4, 2))

    def loss_from_params(W_np, b_np):
        lin.W.data = W_np.copy()
        if lin.b is not None:
            lin.b.data = b_np.copy()
        out = lin(Tensor(X.copy()))
        # scalar L = sum(G * out)  -> dL/dout = G
        return float(np.sum(G * _array(out)))

    W0 = _array(lin.W).copy()
    b0 = _array(lin.b).copy() if lin.b is not None else np.zeros(2)

    # Analytic grads via Tensor.backward().
    lin.W.data = W0.copy()
    if lin.b is not None:
        lin.b.data = b0.copy()
    lin.zero_grad()
    out = lin(Tensor(X.copy()))
    scalar = (out * Tensor(G)).sum()  # uses Tensor ops; backward fills grads
    _try(scalar.backward)
    gW = _array(lin.W.grad).copy()
    gb = _array(lin.b.grad).copy() if lin.b is not None else None

    # Numerical grads.
    num_W = central_diff(lambda W: loss_from_params(W, b0), W0)
    assert np.allclose(gW, num_W, atol=1e-4, rtol=1e-3), (
        f"dL/dW mismatch\nanalytic=\n{gW}\nnumeric=\n{num_W}"
    )
    if lin.b is not None:
        num_b = central_diff(lambda b: loss_from_params(W0, b), b0)
        assert np.allclose(gb, num_b, atol=1e-4, rtol=1e-3), (
            f"dL/db mismatch\nanalytic=\n{gb}\nnumeric=\n{num_b}"
        )


# ===========================================================================
# Optimizer integration: updates must hit the SAME tensor objects.
# ===========================================================================
def test_optimizer_updates_model_parameters_in_place():
    net = _try(_tiny_net)
    params = net.parameters()
    opt = _try(lambda: SGD(params, lr=0.1))
    # Make sure the optimizer holds the very same objects (identity).
    held = {id(p) for p in opt.params}
    assert held == {id(p) for p in params}, "optimizer must hold the model's params"

    before = [_array(p).copy() for p in params]
    for p in params:
        p.grad = np.ones_like(_array(p))
    _try(opt.step)
    for p, b in zip(params, before):
        assert not np.allclose(_array(p), b), "step() must mutate p.data in place"
    # And the model still references the updated tensors.
    assert all(id(a) == id(b) for a, b in zip(net.parameters(), params))


# ===========================================================================
# End-to-end: tiny classifier trains with Adam + CrossEntropyLoss.
# ===========================================================================
def test_tiny_classifier_trains():
    # Two linearly separable 2-D blobs -> 2 classes.
    n = 40
    Xa = RNG.standard_normal((n, 2)) + np.array([2.0, 2.0])
    Xb = RNG.standard_normal((n, 2)) + np.array([-2.0, -2.0])
    X = np.vstack([Xa, Xb])
    y = np.array([0] * n + [1] * n)

    net = _try(lambda: Sequential(Linear(2, 8, seed=3), ReLU(), Linear(8, 2, seed=4)))
    crit = _try(CrossEntropyLoss)
    opt = _try(lambda: Adam(net.parameters(), lr=0.05))

    def loss_now():
        return _scalar(crit(net(Tensor(X)), y))

    start = _try(loss_now)
    for _ in range(80):
        net.zero_grad()
        loss = crit(net(Tensor(X)), y)
        _try(loss.backward)
        opt.step()
    end = loss_now()
    assert end < start, f"loss did not decrease: start={start:.4f} end={end:.4f}"
    assert end < 0.3, f"toy classifier failed to fit: final loss {end:.4f}"


# ===========================================================================
# DataLoader is re-exported and still iterates batches.
# ===========================================================================
def test_dataloader_reexported_and_iterates():
    X = RNG.standard_normal((10, 3))
    y = RNG.integers(0, 2, size=10)
    ds = _try(lambda: Dataset(X, y))
    loader = _try(lambda: DataLoader(ds, batch_size=4, shuffle=False))
    batches = list(_try(lambda: list(loader)))
    assert len(batches) == 3, "10 rows / batch 4 (no drop) -> 3 batches"
    xb, yb = batches[0]
    assert _array(xb).shape == (4, 3)


# ===========================================================================
# MSELoss module wraps the stage_13 functional loss.
# ===========================================================================
def test_mseloss_module_matches_numpy():
    crit = _try(MSELoss)
    pred = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    target = np.array([[1.5, 2.5], [2.5, 3.5]])
    out = _try(lambda: crit(pred, target))
    expected = float(np.mean((_array(pred) - target) ** 2))
    assert np.isclose(_scalar(out), expected, atol=ATOL, rtol=RTOL)
