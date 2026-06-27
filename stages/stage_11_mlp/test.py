"""Tests for Stage 11: MLP.

These tests verify the forward shapes of a multilayer perceptron, that it
gathers all of its layers' parameters, and they gradient-check a scalar loss
(the sum of the network output) with respect to every layer's weights and bias
and with respect to the input, using central differences::

    df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps)

compared against the analytical gradients produced by ``Tensor.backward()`` from
stage_08 (propagated through the ``Dense`` layers from stage_10). ``MLP`` lives in
this stage's ``code.py`` and is built on the ``Dense`` (stage_10) and ``Tensor``
(stage_08) classes imported through ``dlfs.stage_import``. If any earlier stage
is not yet implemented, the suite skips rather than erroring, so you can run it
incrementally.

Run with:  pytest stage_11_mlp/test.py
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

# --- Import the things under test, skipping cleanly if not ready yet. --------
# `code.py` binds the prior-stage `Tensor` (stage_08) and `Dense` (stage_10)
# engines via dlfs.stage_import and defines this stage's broadcasting `Tensor`,
# `Dense`, and `MLP` on top of them; the tests import only this stage's classes.
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
    from code import MLP, Dense, Tensor
except (ImportError, NotImplementedError) as exc:  # pragma: no cover
    pytest.skip(
        f"stage_11 MLP / stage_10 Dense / stage_08 Tensor not importable yet: {exc}",
        allow_module_level=True,
    )

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4


# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Return the underlying numpy array of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def make_tensor(arr, requires_grad=True):
    """Build a Tensor from a numpy array, tolerating different ctor kwargs."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=requires_grad)
    except TypeError:
        return Tensor(arr)


def scalar_out(t):
    """Reduce a Tensor's output to a python float for finite-diff probing."""
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


def build_net(sizes, activation="tanh", out_activation="none", seed=0):
    return MLP(sizes, activation=activation, out_activation=out_activation, seed=seed)


# --- Construction & structure ------------------------------------------------
def test_layer_count_and_widths():
    net = build_net([3, 5, 4, 2], seed=0)
    assert len(net.layers) == 3, "an MLP over [3,5,4,2] must have 3 Dense layers"


def test_every_layer_is_a_dense():
    net = build_net([4, 8, 1], seed=0)
    for layer in net.layers:
        assert isinstance(layer, Dense), "each MLP layer must be a stage_11 Dense"


def test_parameters_are_collected_from_all_layers():
    net = build_net([3, 6, 6, 2], seed=1)
    params = net.parameters()
    # 3 Dense layers, 2 params (weight + bias) each -> 6 parameter tensors.
    assert len(params) == 6, "parameters() must flatten all layers' params (2 per layer)"
    expected = [p for layer in net.layers for p in layer.parameters()]
    assert len(params) == len(expected)
    # Identity check: the same tensor objects, not copies.
    for p, q in zip(params, expected):
        assert p is q, "parameters() must return the actual layer parameter tensors"


def test_repr_mentions_sizes_and_activation():
    net = build_net([2, 16, 1], activation="tanh", out_activation="none", seed=0)
    r = repr(net)
    assert "MLP" in r and "tanh" in r


def test_reproducible_with_seed():
    a = build_net([3, 7, 2], seed=123)
    b = build_net([3, 7, 2], seed=123)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert np.allclose(as_array(pa), as_array(pb)), "same seed -> same weights"


def test_distinct_layers_have_distinct_weights():
    # Per-layer seed derivation should make the two weight matrices differ
    # (they also have different shapes here, but check the values are not a
    # trivially-shared object).
    net = build_net([4, 4, 4], seed=0)
    w0 = as_array(net.layers[0].parameters()[0])
    w1 = as_array(net.layers[1].parameters()[0])
    assert not np.allclose(w0, w1), "different layers should not share identical weights"


# --- Forward shapes ----------------------------------------------------------
def test_single_input_shape():
    net = build_net([3, 5, 2], seed=2)
    y = net(make_tensor([0.5, -1.0, 2.0], requires_grad=False))
    assert as_array(y).shape == (2,), "(n_in,) input must yield (n_out,) output"


def test_batched_input_shape():
    net = build_net([3, 5, 2], seed=2)
    X = make_tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]], requires_grad=False)
    y = net(X)
    assert as_array(y).shape == (2, 2), "(batch, n_in) input must yield (batch, n_out)"


def test_call_is_forward():
    net = build_net([2, 4, 3], seed=5)
    x = make_tensor([0.3, -0.7], requires_grad=False)
    y1 = net(x)
    y2 = net.forward(make_tensor([0.3, -0.7], requires_grad=False))
    assert np.allclose(as_array(y1), as_array(y2)), "__call__ must equal forward()"


# --- Input type guard: forward/__call__ only accept a Tensor -----------------
@pytest.mark.parametrize(
    "bad_x",
    [
        [0.3, -0.7],                       # raw python list
        np.array([0.3, -0.7]),            # bare numpy array (not a Tensor)
        0.5,                               # scalar
    ],
)
def test_forward_and_call_reject_non_tensor(bad_x):
    """``MLP.forward`` and ``MLP.__call__`` must reject any ``x`` that is not a
    ``Tensor`` -- raising an error rather than running the graph on a raw value.

    The net runs the autodiff graph through ``x``; a raw list/array/scalar has no
    ``.data``/``.backward`` and would either crash deep inside a layer or silently
    skip gradient tracking. Guard at the boundary instead. Either a ``TypeError``
    (explicit type check) or an ``AssertionError`` (``assert isinstance(...)``) is
    accepted -- the contract is only that it raises. A real ``Tensor`` (this
    stage's broadcasting subclass) must still pass.
    """
    net = build_net([2, 4, 3], seed=5)
    with pytest.raises((TypeError, AssertionError)):
        net.forward(bad_x)
    with pytest.raises((TypeError, AssertionError)):
        net(bad_x)


def test_forward_and_call_accept_tensor():
    """The type guard must NOT reject a genuine ``Tensor`` (the happy path)."""
    net = build_net([2, 4, 3], seed=5)
    x = make_tensor([0.3, -0.7], requires_grad=False)
    # Neither call raises; both produce the (n_out,) output.
    assert as_array(net.forward(x)).shape == (3,)
    assert as_array(net(make_tensor([0.3, -0.7], requires_grad=False))).shape == (3,)


# --- Depth + nonlinearity actually matters -----------------------------------
def test_hidden_nonlinearity_is_not_affine():
    """A 1->H->1 MLP with a tanh hidden layer must NOT be an affine function of x.

    For an affine map g, the second difference g(x+h) - 2 g(x) + g(x-h) is zero.
    A genuine nonlinearity makes it nonzero.
    """
    net = build_net([1, 8, 1], activation="tanh", out_activation="none", seed=11)

    def g(xv):
        return scalar_out(net(make_tensor(np.array([xv]), requires_grad=False)))

    h = 0.5
    x0 = 0.3
    second_diff = g(x0 + h) - 2.0 * g(x0) + g(x0 - h)
    assert abs(second_diff) > 1e-6, (
        "MLP with a hidden nonlinearity must not collapse to an affine map"
    )


# --- Gradient checks: scalar loss w.r.t. every parameter ---------------------
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_gradcheck_wrt_all_params(activation):
    sizes = [3, 5, 4, 2]
    # seed/input chosen so no pre-activation lands on the ReLU kink (z == 0).
    # At exactly z == 0 ReLU is non-differentiable: central differences yield
    # 0.5 while any subgradient choice (PyTorch uses 0) gives 0 or 1, so a
    # gradcheck there is meaningless. See test_relu_subgradient_at_zero for the
    # explicit relu'(0) == 0 contract.
    net = build_net(sizes, activation=activation, out_activation="none", seed=0)
    x_np = np.array([0.7, 1.3, 0.2])

    # Analytical gradients via one backward pass on sum(output).
    net.zero_grad()
    out = net(make_tensor(x_np, requires_grad=False))
    loss = out.sum() if hasattr(out, "sum") else out
    # If Tensor lacks .sum(), backward() on the (n_out,) tensor seeds ones,
    # which equals d(sum(out)). Either path matches our scalar_out finite diff.
    loss.backward()

    params = net.parameters()
    analytic = [as_array(p.grad).copy() for p in params]

    saved = [as_array(p).copy() for p in params]

    def f_factory(k):
        def f(pv):
            params[k].data = pv.copy().reshape(saved[k].shape)
            val = scalar_out(net(make_tensor(x_np, requires_grad=False)))
            params[k].data = saved[k].copy()
            return val
        return f

    for k, p in enumerate(params):
        g_num = central_diff(f_factory(k), saved[k].copy())
        assert np.allclose(analytic[k], g_num, atol=ATOL, rtol=RTOL), (
            f"[{activation}] dLoss/dparam[{k}] (shape {saved[k].shape}) mismatch:\n"
            f" analytic=\n{analytic[k]}\n numeric =\n{g_num}"
        )


# --- ReLU subgradient convention at the kink (z == 0) ------------------------
def test_relu_subgradient_at_zero():
    """ReLU is non-differentiable at z == 0; we adopt PyTorch's convention that
    relu'(0) == 0. This is asserted on the analytic backward directly -- NOT via
    central differences, which give 0.5 at the kink (no subgradient matches that)
    and so cannot be used to gradcheck exactly at zero. test_gradcheck_wrt_all_params
    deliberately avoids landing any unit on z == 0 for that reason.
    """
    x = make_tensor(np.array([-2.0, 0.0, 3.0]), requires_grad=True)
    y = x.relu()
    loss = y.sum() if hasattr(y, "sum") else y
    loss.backward()  # seeds ones, so x.grad == relu'(x) elementwise
    g = as_array(x.grad)
    assert np.allclose(g, [0.0, 0.0, 1.0]), (
        f"relu'(0) must be 0 (PyTorch convention); got grad {g}"
    )


# --- Gradient check: scalar loss w.r.t. the input ----------------------------
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_gradcheck_wrt_input(activation):
    net = build_net([4, 6, 3], activation=activation, out_activation="none", seed=9)
    x_np = np.array([0.3, -0.6, 1.2, -2.0])

    net.zero_grad()
    x = make_tensor(x_np, requires_grad=True)
    out = net(x)
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    g_analytic = as_array(x.grad)

    def f(xv):
        return scalar_out(net(make_tensor(xv, requires_grad=False)))

    g_num = central_diff(f, x_np.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"[{activation}] dLoss/dx mismatch:\n analytic={g_analytic}\n numeric ={g_num}"
    )


# --- Gradient check over a batch (sum of all outputs) ------------------------
def test_gradcheck_batch_wrt_first_layer():
    net = build_net([3, 5, 2], activation="tanh", out_activation="none", seed=6)
    X = np.array([[0.5, -1.0, 2.0], [1.0, 0.5, -0.5], [-0.3, 0.8, 1.1]])

    net.zero_grad()
    out = net(make_tensor(X, requires_grad=False))
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    W1 = net.layers[0].parameters()[0]
    g_analytic = as_array(W1.grad).copy()

    saved = as_array(W1).copy()

    def f(w):
        W1.data = w.copy().reshape(saved.shape)
        val = scalar_out(net(make_tensor(X, requires_grad=False)))
        W1.data = saved.copy()
        return val

    g_num = central_diff(f, saved.copy())
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        f"batched dLoss/dW1 mismatch:\n analytic=\n{g_analytic}\n numeric =\n{g_num}"
    )


# --- zero_grad clears accumulated gradients across all layers -----------------
def test_zero_grad_clears_every_param():
    net = build_net([3, 4, 2], activation="tanh", seed=8)
    out = net(make_tensor([1.0, 2.0, 3.0], requires_grad=False))
    loss = out.sum() if hasattr(out, "sum") else out
    loss.backward()
    assert any(np.any(as_array(p.grad) != 0.0) for p in net.parameters()), (
        "some gradient should be populated after backward"
    )
    net.zero_grad()
    for p in net.parameters():
        assert np.allclose(as_array(p.grad), 0.0), "zero_grad must clear every param grad"


# --- Broadcasting backward (the stage_11 Tensor subclass) --------------------
# stage_08's Tensor only allows equal-shaped elementwise operands; this stage's
# `Tensor` subclass adds broadcasting forward + unbroadcast backward, which
# stage_12's stable softmax (`(B,C) - (B,1)`) and bias-row broadcasting need.
def bcast_tensor(arr, requires_grad=True):
    """Build a *broadcasting* Tensor (the stage_11 subclass), tolerating ctors."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=requires_grad)
    except TypeError:
        return Tensor(arr)


def test_tensor_is_broadcasting_subclass():
    """The exported `Tensor` must be a prior-stage engine extended in this stage.

    Don't name the base class (this stage is the boundary that introduces the
    subclass); instead prove the relationship structurally: `Tensor` has a
    strict ancestor also named ``Tensor`` (the engine it extends), and it really
    broadcasts -- differently-shaped operands forward + each grad unbroadcasts
    back to its own shape, which the base engine refuses.
    """
    base = Tensor.__mro__[1]  # immediate ancestor: the engine being extended
    assert base is not Tensor and base.__name__ == "Tensor", (
        "stage_11 must export a Tensor that subclasses the prior-stage Tensor"
    )
    # Behavioural proof of the extension: (2,3)+(3,) broadcasts and unbroadcasts.
    a = bcast_tensor(np.ones((2, 3)))
    b = bcast_tensor(np.ones((3,)))
    z = a + b
    assert as_array(z).shape == (2, 3)
    z.backward()
    assert as_array(b.grad).shape == (3,), (
        "the stage_11 Tensor must unbroadcast a (3,) operand's grad back to (3,)"
    )


def test_broadcast_add_row_vector_forward_and_grad():
    """(2,3) + (3,): forward equals numpy broadcast; (3,) grad is the column
    sums (shape (3,)) and (2,3) grad is ones (seed=ones)."""
    a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b_np = np.array([10.0, 20.0, 30.0])                   # (3,)
    a = bcast_tensor(a_np)
    b = bcast_tensor(b_np)

    z = a + b
    # Forward matches numpy broadcasting.
    assert as_array(z).shape == (2, 3)
    assert np.allclose(as_array(z), a_np + b_np)

    z.backward()  # seeds grad = ones_like(z), i.e. all-ones (2,3)
    # d/da of sum(a + b) is ones at a's shape; d/db is the column-sums.
    assert as_array(a.grad).shape == (2, 3)
    assert np.allclose(as_array(a.grad), np.ones((2, 3)))
    assert as_array(b.grad).shape == (3,), "broadcast (3,) operand grad must keep shape (3,)"
    assert np.allclose(as_array(b.grad), np.ones((2, 3)).sum(axis=0))  # [2, 2, 2]


def test_broadcast_sub_keepdims_column_forward_and_grad():
    """(B,C) - (B,1): forward broadcasts the (B,1) column; the (B,1) operand's
    grad sums across the C axis (shape (B,1)), as stage_12 softmax needs."""
    B, C = 4, 3
    x_np = np.arange(B * C, dtype=float).reshape(B, C)  # (B, C)
    m_np = np.array([[0.5], [1.0], [-2.0], [3.0]])      # (B, 1)
    x = bcast_tensor(x_np)
    m = bcast_tensor(m_np)

    z = x - m
    assert as_array(z).shape == (B, C)
    assert np.allclose(as_array(z), x_np - m_np)

    z.backward()  # seed ones (B, C)
    # d/dx of sum(x - m) is ones (B, C); d/dm is -1 summed across C -> (B, 1).
    assert as_array(x.grad).shape == (B, C)
    assert np.allclose(as_array(x.grad), np.ones((B, C)))
    assert as_array(m.grad).shape == (B, 1), "broadcast (B,1) operand grad must keep shape (B,1)"
    assert np.allclose(as_array(m.grad), -np.ones((B, C)).sum(axis=1, keepdims=True))  # all -C


def test_broadcast_mul_scales_grad_by_other_operand():
    """(2,3) * (3,): multiply backward applies the local factor at the broadcast
    shape, then unbroadcasts -- the (3,) operand's grad is the column-sums of the
    OTHER operand's data."""
    a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b_np = np.array([2.0, 3.0, 4.0])                      # (3,)
    a = bcast_tensor(a_np)
    b = bcast_tensor(b_np)

    z = a * b
    assert np.allclose(as_array(z), a_np * b_np)

    z.backward()  # seed ones (2, 3)
    # d/da = ones * b broadcast -> b tiled over rows; d/db = (ones * a) summed over rows.
    assert np.allclose(as_array(a.grad), np.broadcast_to(b_np, (2, 3)))
    assert as_array(b.grad).shape == (3,)
    assert np.allclose(as_array(b.grad), a_np.sum(axis=0))  # [5, 7, 9]


# --- Exhaustive broadcasting gradcheck over every broadcasting binary op ------
# The two overridden primitives are __add__ / __mul__; __sub__, __truediv__,
# __neg__ and the reflected forms (__radd__/__rmul__/__rsub__/__rtruediv__)
# compose out of them and so must broadcast + unbroadcast correctly too. The
# tests below numerically gradient-check L = sum(op(a, b)) w.r.t. BOTH operands
# across every broadcast pattern: equal shapes (no-op unbroadcast), rank
# promotion ((n,)->(B,n)), size-1 stretch ((B,1)->(B,C)), BOTH operands
# stretched ((R,1) & (1,C) -> (R,C)), and combined rank+size-1.

# Binary ops under test. Each entry: (name, callable(a, b)). The callable uses
# the Python operator so dispatch routes through the real dunder being tested.
_BINARY_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
]

# Broadcast shape pairs (shape_a, shape_b). Both broadcast to a common shape.
_BCAST_SHAPE_PAIRS = [
    ((2, 3), (2, 3)),   # equal -> unbroadcast is a no-op on both
    ((2, 3), (3,)),     # rank promotion on b: (3,) -> (2,3)
    ((3,), (2, 3)),     # rank promotion on a (left operand smaller)
    ((4, 3), (4, 1)),   # size-1 stretch on b's last axis (keepdims case)
    ((4, 1), (4, 3)),   # size-1 stretch on a's last axis
    ((2, 1), (1, 3)),   # BOTH operands stretched -> (2,3)
    ((2, 3, 4), (4,)),  # 3-D rank promotion: (4,) -> (2,3,4)
    ((2, 3, 4), (3, 1)),  # combined rank-promo + size-1 stretch on b
]


def _grad_pair(op, a_np, b_np):
    """Analytic grads of L = sum(op(a, b)) w.r.t. a and b via Tensor.backward()."""
    a = bcast_tensor(a_np)
    b = bcast_tensor(b_np)
    z = op(a, b)
    z.backward()  # seeds ones over z's (broadcast) shape -> dL/dz = 1
    return as_array(a.grad), as_array(b.grad), as_array(z)


def _pos(shape, lo=1.0):
    """Strictly-positive ramp of `shape` (so division and 1/b are well-behaved)."""
    n = int(np.prod(shape))
    return (np.arange(n, dtype=float) + lo).reshape(shape)


@pytest.mark.parametrize("op_name,op", _BINARY_OPS)
@pytest.mark.parametrize("shape_a,shape_b", _BCAST_SHAPE_PAIRS)
def test_broadcast_binary_op_gradcheck(op_name, op, shape_a, shape_b):
    """L = sum(op(a, b)): analytic grads match central differences, and each
    operand's grad keeps that operand's ORIGINAL shape (unbroadcast correctness)."""
    a_np = _pos(shape_a)
    b_np = _pos(shape_b, lo=2.0)  # keep b away from 0 for div / b**-1

    ga, gb, z = _grad_pair(op, a_np, b_np)

    # Forward matches numpy's own broadcast of the same operator.
    np_op = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
             "div": np.true_divide}[op_name]
    assert z.shape == np.broadcast_shapes(shape_a, shape_b)
    assert np.allclose(z, np_op(a_np, b_np))

    # Each operand's grad is reduced back to its own shape.
    assert ga.shape == shape_a, f"{op_name}: a.grad shape {ga.shape} != {shape_a}"
    assert gb.shape == shape_b, f"{op_name}: b.grad shape {gb.shape} != {shape_b}"

    # Numeric gradient of sum(op(a, b)) w.r.t. a (b fixed) and w.r.t. b (a fixed).
    num_a = central_diff(lambda av: float(np.sum(np_op(av, b_np))), a_np.copy())
    num_b = central_diff(lambda bv: float(np.sum(np_op(a_np, bv))), b_np.copy())

    assert np.allclose(ga, num_a, atol=ATOL, rtol=RTOL), (
        f"{op_name} dL/da [{shape_a} vs {shape_b}] mismatch:\n analytic={ga}\n numeric={num_a}"
    )
    assert np.allclose(gb, num_b, atol=ATOL, rtol=RTOL), (
        f"{op_name} dL/db [{shape_a} vs {shape_b}] mismatch:\n analytic={gb}\n numeric={num_b}"
    )


# --- Reflected ops: `scalar OP tensor` must broadcast the python number -------
@pytest.mark.parametrize("op_name,op", [
    ("radd", lambda s, t: s + t),       # number.__add__ -> NotImplemented -> Tensor.__radd__
    ("rsub", lambda s, t: s - t),       # Tensor.__rsub__: (-self) + other
    ("rmul", lambda s, t: s * t),       # Tensor.__rmul__
    ("rtruediv", lambda s, t: s / t),   # Tensor.__rtruediv__: (self**-1) * other
])
def test_broadcast_reflected_scalar_op_gradcheck(op_name, op):
    """`scalar OP tensor` (a python number on the left) coerces+broadcasts and
    backprops the grad to the (2,3) tensor; gradcheck against central diff."""
    scalar = 3.0
    t_np = _pos((2, 3), lo=2.0)  # positive -> safe for scalar / t
    t = bcast_tensor(t_np)

    z = op(scalar, t)
    np_op = {"radd": np.add, "rsub": np.subtract,
             "rmul": np.multiply, "rtruediv": np.true_divide}[op_name]
    assert as_array(z).shape == (2, 3)
    assert np.allclose(as_array(z), np_op(scalar, t_np))

    z.backward()
    g = as_array(t.grad)
    assert g.shape == (2, 3), f"{op_name}: tensor grad must keep (2,3), got {g.shape}"
    num = central_diff(lambda tv: float(np.sum(np_op(scalar, tv))), t_np.copy())
    assert np.allclose(g, num, atol=ATOL, rtol=RTOL), (
        f"{op_name} dL/dt mismatch:\n analytic={g}\n numeric={num}"
    )


# --- Both operands broadcast simultaneously: outer-sum forward + both grads ----
def test_broadcast_both_operands_stretched_add():
    """(2,1) + (1,3) -> (2,3): forward is the outer sum; backward sends the (2,1)
    operand the row-sums (keepdims) and the (1,3) operand the column-sums."""
    a_np = np.array([[1.0], [2.0]])        # (2, 1)
    b_np = np.array([[10.0, 20.0, 30.0]])  # (1, 3)
    a = bcast_tensor(a_np)
    b = bcast_tensor(b_np)

    z = a + b
    assert as_array(z).shape == (2, 3)
    assert np.allclose(as_array(z), a_np + b_np)

    z.backward()  # seed ones (2, 3)
    # a stretched over axis 1 -> grad sums axis 1 keepdims -> (2,1) of value 3 (C).
    assert as_array(a.grad).shape == (2, 1)
    assert np.allclose(as_array(a.grad), np.full((2, 1), 3.0))
    # b stretched over axis 0 -> grad sums axis 0 keepdims -> (1,3) of value 2 (R).
    assert as_array(b.grad).shape == (1, 3)
    assert np.allclose(as_array(b.grad), np.full((1, 3), 2.0))


def test_broadcast_div_keepdims_column_grad():
    """(B,C) / (B,1): the (B,1) denominator's grad sums across C back to (B,1)
    (the quotient-rule local factor -a/b**2, unbroadcast)."""
    B, C = 3, 4
    a_np = _pos((B, C))
    b_np = _pos((B, 1), lo=2.0)
    a = bcast_tensor(a_np)
    b = bcast_tensor(b_np)

    z = a / b
    assert as_array(z).shape == (B, C)
    assert np.allclose(as_array(z), a_np / b_np)

    z.backward()
    assert as_array(a.grad).shape == (B, C)
    assert as_array(b.grad).shape == (B, 1)
    # d/da = 1/b broadcast; d/db = sum_C(-a / b**2), keepdims.
    assert np.allclose(as_array(a.grad), np.broadcast_to(1.0 / b_np, (B, C)))
    assert np.allclose(as_array(b.grad), (-a_np / (b_np ** 2)).sum(axis=1, keepdims=True))
