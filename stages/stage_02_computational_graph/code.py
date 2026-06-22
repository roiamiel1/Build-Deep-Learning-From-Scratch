"""Stage 2: Computational graph.

stage_01's ``Value`` already records its provenance: every result node stores the
parents it was built from (``_prev``), the op that built it (``_op``), and a no-op
``_backward`` hook. Those fields were inert plumbing there — nothing read them.
This stage makes the graph FIRST-CLASS by adding the part that uses it: a
``trace(root)`` walker that enumerates the DAG, plus a graph-aware ``__repr__``
that surfaces ``_op``.

Because stage_01's operators already build their results through ``_make`` (which
constructs ``type(self)(...)`` and wires up ``_prev``/``_op``), this stage does
NOT touch ``__add__``/``__mul__``/``__pow__`` — subclassing alone makes the whole
DAG out of this stage's ``Value``. Forward arithmetic is identical to stage_01.
Allowed tools: Python standard library ONLY.
"""

from dlfs import stage_import

# stage_01 Value (extend, do not rewrite)
Stage1_Value = stage_import("stage_01", "Value")


class Value(Stage1_Value):
    """stage_01 ``Value``, extended so its operation graph is first-class.

    stage_01 already set, on every result node:
    - ``_prev``: the **set** of parent ``Value``s it was built from (``set()`` for
      a leaf). A set so a reused operand (``a * a``) is a single parent and the
      DAG walk terminates.
    - ``_op``: a string label for the op (``'+'``, ``'*'``, ``''`` for leaves).
    - ``_backward``: a no-op closure (stage_03 installs the real per-op rule).

    Operand ORDER is not stored: a set is unordered, and that is fine because the
    stage_03 gradient closures capture each operand directly (so ``a - b`` and
    ``a / b`` know which side is which without the node remembering order).

    This stage ADDS the consumers of that bookkeeping: a graph-aware ``__repr__``
    and the module-level ``trace`` walker. The arithmetic operators are inherited
    unchanged — ``_make`` already constructs this subclass and records the edges.
    """

    def __repr__(self):
        """Graph-aware debug string, e.g. ``Value(data=3.0, op='+')``."""
        args = [
            f"data={self.data}",
            f"prev='{self._prev}'" if self._prev else "",
            f"op='{self._op}'" if self._op else "",
        ]

        return f"Value({', '.join(args)})"


def trace(root):
    """Walk the graph backward from `root`, returning (nodes, edges).

    nodes: set of every Value reachable from root (including root).
    edges: set of (parent, child) tuples, one per _prev link.
    Use a visited set so the DAG walk terminates on reused nodes (e.g. a * a).
    """
    assert isinstance(root, Value)

    nodes = set([root])
    edges = set()

    for n in root._prev:
        n_nodes, n_edges = trace(n)
        nodes = nodes.union(n_nodes)
        edges = edges.union(n_edges)
        edges.add((n, root))


    return nodes, edges
