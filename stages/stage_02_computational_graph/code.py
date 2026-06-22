"""Stage 2: Computational graph.

Subclass stage_01 `Value` to make its operation graph first-class: every node
tracks parents (`_prev`) and op (`_op`), plus a `trace(root)` DAG walker.
No gradients here; forward arithmetic stays identical to stage_01.
Allowed tools: Python standard library ONLY.
"""

from dlfs import stage_import

# stage_01 Value (extend, do not rewrite)
Stage1_Value = stage_import("stage_01", "Value")


class Value(Stage1_Value):
    """stage_01 `Value`, extended so its operation graph is first-class.

    Each result node records its provenance two ways:
    - ``_prev``: the *set* of parent Values (dedups ``a * a``; used by ``trace``).
    - ``_inputs``: the *operand-ordered tuple* of parents (keeps left/right, so
      asymmetric ops in later stages — ``a - b``, ``a / b`` — can tell inputs
      apart). A set cannot preserve order, so the ordered tuple is stored too.
    Leaves have ``_prev == set()`` and ``_inputs == ()``.
    """

    def __init__(self, data, _inputs=(), _op=""):
        """Init stage_01 fields, then record graph provenance.

        Stores ``_inputs`` (operand-ordered tuple), ``_prev`` (set(_inputs), for
        ``trace``/dedup), and ``_op`` (the op label, '' for leaves).
        """
        # TODO: call stage_01 __init__ for data/grad, then set
        # self._inputs (tuple), self._prev (set(self._inputs)), self._op
        raise NotImplementedError("stage_02: implement Value.__init__")

    def __add__(self, other):
        """Return self + other as a Value recording this addition.

        Pass operands in order so the result has _inputs=(self, other), _op='+'.
        """
        # TODO: implement graph-recording addition (coerce numbers to Value)
        raise NotImplementedError("stage_02: implement Value.__add__")

    def __mul__(self, other):
        """Return self * other as a Value recording this multiply.

        Pass operands in order so the result has _inputs=(self, other), _op='*'.
        """
        # TODO: implement graph-recording multiply (coerce numbers to Value)
        raise NotImplementedError("stage_02: implement Value.__mul__")

    def __repr__(self):
        """Graph-aware debug string, e.g. ``Value(data=3.0, op='+')``."""
        # TODO: implement repr surfacing data and _op
        raise NotImplementedError("stage_02: implement Value.__repr__")


def trace(root):
    """Walk the graph backward from `root`, returning (nodes, edges).

    nodes: set of every Value reachable from root (including root).
    edges: set of (parent, child) tuples, one per _prev link.
    Use a visited set so the DAG walk terminates on reused nodes (e.g. a * a).
    """
    # TODO: implement the backward graph walk
    raise NotImplementedError("stage_02: implement trace")
