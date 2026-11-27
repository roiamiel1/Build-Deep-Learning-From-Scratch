"""Stage 05: Scalar reverse-mode autodiff engine.

Imports stage_04's complete reverse-mode ``Value`` (graph + per-op ``_backward``
+ ``backward()``) and subclasses it, adding only the new ops (tanh, exp, relu),
thin re-blessing __add__/__mul__/__pow__, and a grad-aware __repr__. Inherited
graph/accumulation/backward machinery is reused, not re-derived. SKELETON ONLY.
Stdlib only (no autodiff libraries).
"""

from __future__ import annotations

import math
from typing import Union

from dlfs import stage_import

# Complete scalar autodiff Value (graph + backward()) from stage_04; extended here.
Stage4_Value = stage_import("stage_04", "Value")

Number = Union[int, float]


class Value(Stage4_Value):
    """Scalar node in a differentiable graph; extends stage_04's reverse-mode Value.

    Inherits __init__/_backward, differentiable __add__/__mul__/__pow__, reflected
    ops, backward(), and the derived ops (__neg__/__sub__/__rsub__/__truediv__).
    This stage only adds the remaining primitive ops and their gradient rules.
    """

    # No operator overrides needed: stage_01's `_make` builds every result via
    # `type(self)(...)`, so the inherited `+`/`*`/`**` already return THIS class
    # and the new unary ops below chain on any intermediate.

    # New primitive ops. Each builds a Value and installs a _backward closure
    # that accumulates into inputs with += (never =).

    def tanh(self) -> "Value":
        """Return tanh(self). Local rule: dt/dx = 1 - tanh(x)**2."""
        data = math.tanh(self.data)
        _return = self._make(data, [self], "tanh")
        def _backward():
            self.grad += (1 - data**2) * _return.grad
        _return._backward = _backward
        return _return
            

    def exp(self) -> "Value":
        """Return exp(self). Local rule: de/dx = exp(x)."""
        data = math.exp(self.data)
        _return = self._make(data, [self], "exp")

        def _backward():
            self.grad += data * _return.grad

        _return._backward = _backward
        return _return

    def relu(self) -> "Value":
        """Return ReLU(self) = max(0, self). Local rule: dr/dx = 1 if x > 0 else 0."""
        data = max(0, self.data)
        _return = self._make(data, [self], "relu")

        def _backward():
            self.grad += 1 * _return.grad if data > 0 else 0

        _return._backward = _backward
        return _return

    def __repr__(self) -> str:
        """Return e.g. ``Value(data=2.0, grad=4.0)``."""
        return f"Value(data={self.data}, grad={self.grad})"
