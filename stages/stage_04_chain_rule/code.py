"""Stage 4: The backward pass.

stage_03 gave every result node a ``_backward`` closure that pushes the local
derivative one edge. This stage adds the **global** reverse pass that runs those
closures in the right order: ``backward()`` topologically sorts the graph, seeds
the output's ``grad`` to 1, and walks the nodes in reverse, calling each
``_backward``. After ``loss.backward()`` every node's ``.grad`` holds
``d(loss)/d(node)`` — automatic differentiation, complete for scalars.

This is the micrograd ``Value.backward``. Later stages reuse it unchanged
(stage_05 only adds more ops; stage_08 lifts the same algorithm to arrays).
Allowed tools: Python stdlib only.
"""

from __future__ import annotations

from typing import List, Set

from dlfs import stage_import

# stage_03 Value: graph node whose ops install per-edge `_backward` closures.
Stage3_Value = stage_import("stage_03", "Value")


class Value(Stage3_Value):
    """stage_03 `Value` plus a global ``backward()`` reverse pass.

    Inherits ``data``/``grad``/``_prev``/``_op``/``_backward`` and the
    gradient-installing ``__add__``/``__mul__``/``__pow__`` (and the derived
    ops). Adds only the topological reverse walk.
    """

    def backward(self) -> None:
        """Run reverse-mode autodiff with ``self`` as the output (loss).

        (1) Build a topological order of every node reachable from ``self``
            (parents before children) via DFS post-order over ``_prev``.
        (2) Seed ``self.grad = 1.0`` (d(self)/d(self) = 1).
        (3) Walk the order in REVERSE, calling each node's ``_backward`` so
            gradients flow from the output back to every leaf. Closures use
            ``+=``, so reused nodes accumulate correctly.

        Does not zero grads first; callers zero between passes if needed.
        """
        topo = topo_sort(self)
        self.grad = 1.0
        for v in reversed(topo): 
            v._backward()


def topo_sort(root: "Value") -> List["Value"]:
    """Return nodes reachable from ``root`` in topological order.

    Dependencies first (a parent appears before any child that uses it); the
    backward pass walks this list in reverse. Each node is emitted exactly once
    even when the graph is a DAG with reused nodes (use a ``visited`` set).
    """
    order: List["Value"] = []
    visited: Set["Value"] = set()

    def build(v: "Value") -> None:
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            order.append(v)

    build(root)
    return order
