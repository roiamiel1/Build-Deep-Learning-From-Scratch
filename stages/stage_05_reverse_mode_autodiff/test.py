"""Stage 5 tests: reverse-mode autodiff (topo sort + reverse accumulation).

Run: pytest stage_05_reverse_mode_autodiff/test.py

Gradients are verified against central-difference numerical gradients:
    df/dx ~= (f(x + eps) - f(x - eps)) / (2 * eps)
"""

import pytest

# This stage's extended Value (subclass of stage_02's Value) lives in its own
# code.py; import it through the curriculum shim so the cumulative chain is real.
from dlfs import stage_import

# ``Node`` is an alias for this stage's ``Value`` (it subclasses stage_02's
# graph node and adds the reverse-mode machinery the tests exercise).
Stage5_Value, Stage5_topo_sort = stage_import("stage_05", "Value", "topo_sort")
Node = Stage5_Value
topo_sort = Stage5_topo_sort

EPS = 1e-6
TOL = 1e-4


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def zero_grads(*nodes):
    for n in nodes:
        n.grad = 0.0


def numeric_grad(f, xs, i):
    """Central-difference dL/dx_i, where f(list_of_floats) -> float."""
    plus = list(xs)
    minus = list(xs)
    plus[i] += EPS
    minus[i] -= EPS
    return (f(plus) - f(minus)) / (2 * EPS)


# --------------------------------------------------------------------------- #
# Forward correctness
# --------------------------------------------------------------------------- #
def test_add_forward():
    a, b = Node(2.0), Node(3.0)
    assert (a + b).data == pytest.approx(5.0)


def test_mul_forward():
    a, b = Node(2.0), Node(-4.0)
    assert (a * b).data == pytest.approx(-8.0)


def test_scalar_promotion_both_sides():
    a = Node(3.0)
    assert (a + 1.0).data == pytest.approx(4.0)
    assert (1.0 + a).data == pytest.approx(4.0)
    assert (a * 2.0).data == pytest.approx(6.0)
    assert (2.0 * a).data == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# Topological sort
# --------------------------------------------------------------------------- #
def test_topo_sort_dependencies_before_dependents():
    a, b = Node(1.0), Node(2.0)
    c = a * b
    d = c + a
    order = topo_sort(d)
    pos = {id(n): i for i, n in enumerate(order)}
    # every parent must come before its child in the order
    for child in order:
        for parent in child._prev:
            assert pos[id(parent)] < pos[id(child)], (
                "parent must appear before child in topological order"
            )


def test_topo_sort_counts_each_node_once():
    a = Node(3.0)
    b = a * a            # a is shared
    c = b + a            # a shared again
    order = topo_sort(c)
    ids = [id(n) for n in order]
    assert len(ids) == len(set(ids)), "topo_sort must emit each node exactly once"
    # distinct nodes here: a, b, c  -> 3
    assert len(order) == 3


def test_topo_sort_root_is_last():
    a, b = Node(1.0), Node(2.0)
    out = a + b
    order = topo_sort(out)
    assert order[-1] is out, "root (output) must be last in topological order"


# --------------------------------------------------------------------------- #
# Backward: seed and basic local rules
# --------------------------------------------------------------------------- #
def test_backward_seeds_output_grad():
    a, b = Node(2.0), Node(5.0)
    out = a * b
    out.backward()
    assert out.grad == pytest.approx(1.0), "output node must be seeded with grad 1"


def test_add_backward():
    a, b = Node(2.0), Node(-3.0)
    zero_grads(a, b)
    out = a + b
    out.backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)


def test_mul_backward():
    a, b = Node(2.0), Node(-3.0)
    zero_grads(a, b)
    out = a * b
    out.backward()
    assert a.grad == pytest.approx(b.data)   # = -3
    assert b.grad == pytest.approx(a.data)   # =  2


# --------------------------------------------------------------------------- #
# Accumulation along multiple paths (the whole point of reverse mode)
# --------------------------------------------------------------------------- #
def test_reused_input_add_accumulates():
    # f(x) = x + x = 2x  -> df/dx = 2
    x = Node(7.0)
    zero_grads(x)
    out = x + x
    out.backward()
    assert x.grad == pytest.approx(2.0), "x+x must accumulate grad via +="


def test_reused_input_square_accumulates():
    # f(a) = a * a = a^2  -> df/da = 2a
    a = Node(4.0)
    zero_grads(a)
    out = a * a
    out.backward()
    assert a.grad == pytest.approx(2 * a.data), "a*a needs grad accumulation"


def test_diamond_graph_sums_paths():
    # x -> a, x -> b, out = a*b with a = x+1, b = x*2
    # out(x) = (x+1)*(2x) = 2x^2 + 2x -> dout/dx = 4x + 2
    x = Node(3.0)
    zero_grads(x)
    a = x + 1.0
    b = x * 2.0
    out = a * b
    # zero intermediate grads too
    for n in topo_sort(out):
        n.grad = 0.0
    out.backward()
    assert x.grad == pytest.approx(4 * x.data + 2)


# --------------------------------------------------------------------------- #
# Numerical gradient checks (central differences)
# --------------------------------------------------------------------------- #
def _build_expr(vals):
    """A small mixed expression; returns (output_node, [input_nodes]).

    f(x, y, z) = (x*y + z) * (x + y)  +  x*x
    """
    x, y, z = (Node(v) for v in vals)
    out = (x * y + z) * (x + y) + x * x
    return out, [x, y, z]


def _forward_only(vals):
    out, _ = _build_expr(vals)
    return out.data


def test_gradcheck_mixed_expression():
    vals = [1.5, -2.0, 0.7]
    out, inputs = _build_expr(vals)
    for n in topo_sort(out):
        n.grad = 0.0
    out.backward()
    for i, node in enumerate(inputs):
        ng = numeric_grad(_forward_only, vals, i)
        assert node.grad == pytest.approx(ng, abs=TOL), (
            f"analytical grad {node.grad} != numerical {ng} for input {i}"
        )


def test_gradcheck_deep_chain():
    # f(x) = ((((x+1)*x)+1)*x) ... force many accumulation steps
    def f(vals):
        x = Node(vals[0])
        t = x
        for _ in range(5):
            t = (t + 1.0) * x
        return t

    vals = [1.1]
    out = f(vals)
    for n in topo_sort(out):
        n.grad = 0.0
    out.backward()
    # recover the single input node (the leaf with _op == "")
    leaves = [n for n in topo_sort(out) if n._op == ""]
    # the variable leaf is the one used many times; constants (1.0) are also leaves.
    # numerical grad of the scalar function:
    ng = numeric_grad(lambda v: f(v).data, vals, 0)
    # grad of x = sum over all leaves equal to x; here the variable leaf carries it.
    var_leaf = max(leaves, key=lambda n: abs(n.grad))
    assert var_leaf.grad == pytest.approx(ng, abs=TOL), (
        f"deep-chain grad {var_leaf.grad} != numerical {ng}"
    )


def test_no_autozero_in_backward():
    # backward must NOT zero other nodes; calling twice should double grads.
    x = Node(3.0)
    x.grad = 0.0
    out = x * 2.0
    out.backward()
    g1 = x.grad
    out.backward()  # grads accumulate again (no auto-zero)
    assert x.grad == pytest.approx(2 * g1), (
        "backward must not auto-zero; repeated calls accumulate"
    )
