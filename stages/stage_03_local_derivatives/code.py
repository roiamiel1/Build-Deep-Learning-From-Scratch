"""Stage 3: Local derivatives.

Adds per-op local-derivative rules on top of stage_02's ``Value`` graph: one
honest partial per input, evaluated at the stored forward values. No global
backward pass yet (see README for the hand-derived rules). Stdlib only.
"""

from __future__ import annotations

# stage_02 Value: exposes .data (forward value), ._op (op label), ._prev (parents).
from dlfs import stage_import

Stage2_Value = stage_import("stage_02", "Value")


# Per-op local derivatives: take input forward values (floats), return one
# partial per input, in operand order.


def d_add(a: float, b: float) -> tuple[float, float]:
    """Local derivatives of ``z = a + b``; returns ``(dz/da, dz/db)``."""
    raise NotImplementedError  # TODO: implement local derivatives of add


def d_sub(a: float, b: float) -> tuple[float, float]:
    """Local derivatives of ``z = a - b``; returns ``(dz/da, dz/db)``."""
    raise NotImplementedError  # TODO: implement local derivatives of sub


def d_mul(a: float, b: float) -> tuple[float, float]:
    """Local derivatives of ``z = a * b``; returns ``(dz/da, dz/db)``."""
    raise NotImplementedError  # TODO: implement local derivatives of mul


def d_div(a: float, b: float) -> tuple[float, float]:
    """Local derivatives of ``z = a / b``; returns ``(dz/da, dz/db)``. Raises ZeroDivisionError if ``b == 0``."""
    raise NotImplementedError  # TODO: implement local derivatives of div (guard b == 0)


def d_neg(a: float) -> tuple[float]:
    """Local derivative of unary negation ``z = -a``; returns ``(dz/da,)``."""
    raise NotImplementedError  # TODO: implement local derivative of neg


# Dispatcher: reads a node's ._op, pulls each input's forward value, and calls
# the matching d_* rule. Leaf/constant nodes return ().


def _input_value(inp) -> float:
    """Return an input's forward value (its ``.data`` if a Value, else the number)."""
    raise NotImplementedError  # TODO: implement reading an input's forward value


def _ordered_inputs(node) -> tuple:
    """Return ``node``'s parent inputs in operand order; () for a leaf/constant.

    Order matters for asymmetric ops ('-', '/'); recover it from your stage_02 Value.
    """
    raise NotImplementedError  # TODO: implement recovering operand-ordered parents


def local_derivatives(node) -> tuple[float, ...]:
    """Local derivatives of ``node``'s output wrt each input, in operand order.

    Dispatch on ``node._op`` (+ - * / neg); () for leaves; ValueError on unknown op.
    """
    raise NotImplementedError  # TODO: implement op dispatch over the stage_02 graph
