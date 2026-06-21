"""Stage 5: Reverse-mode autodiff.

Subclasses stage_02's ``Value`` to add per-op ``_backward`` closures,
``topo_sort`` for DAG ordering, and ``backward()`` (seed grad=1, walk reverse).
Skeleton only. Python stdlib only; no NumPy/autodiff libs.
"""

from __future__ import annotations

from typing import Iterable, List, Set

from dlfs import stage_import

# Reuse stage_02's computation-graph node; subclass it below.
Stage2_Value = stage_import("stage_02", "Value")


class Value(Stage2_Value):
    """Scalar graph node that pushes gradients backward.

    Inherits data/grad/_prev/_op from stage_02; adds a ``_backward`` closure
    (default no-op for leaves) installed per-op by the overridden ops below.
    """

    def __init__(self, data, _children: Iterable["Value"] = (), _op: str = ""):
        # TODO: delegate to super().__init__, then add a no-op _backward
        raise NotImplementedError

    def __add__(self, other: "Value | float") -> "Value":
        """Return ``self + other`` and install its backward rule (d_add -> 1, 1)."""
        # TODO: build the sum node and accumulate (+=) the add local rule onto inputs
        raise NotImplementedError

    def __mul__(self, other: "Value | float") -> "Value":
        """Return ``self * other`` and install its backward rule (d_mul -> b, a)."""
        # TODO: build the product node and accumulate (+=) the product rule onto inputs
        raise NotImplementedError

    # __radd__ / __rmul__ inherited from stage_02; they delegate here.

    def backward(self) -> None:
        """Compute dL/d(node) for all nodes (L = self): topo_sort, seed grad=1,
        then call _backward in reverse order. Does not zero other grads."""
        # TODO: implement reverse-mode traversal
        raise NotImplementedError


def topo_sort(root: Value) -> List[Value]:
    """Return nodes reachable from ``root`` in topological order (dependencies
    first); reverse it for the backward pass. Each node emitted once."""
    order: List[Value] = []
    visited: Set[Value] = set()

    def build(v: Value) -> None:
        # TODO: DFS post-order using `visited`, appending to `order`
        raise NotImplementedError

    # TODO: build(root); return order
    raise NotImplementedError
