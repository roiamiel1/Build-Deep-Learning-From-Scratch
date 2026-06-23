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

    # Re-bless inherited core ops so they return a stage-06 Value (lets the new
    # unary ops chain on intermediates); delegates math to stage_04.

    def __add__(self, other: Union["Value", Number]) -> "Value":
        """Return self + other via stage_04's __add__, re-tagged as stage-06 Value."""
        return super(Value, self).__add__(other)

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        """Return self * other via stage_04's __mul__, re-tagged as stage-06 Value."""
        return super(Value, self).__mul__(other)

    # New primitive ops. Each builds a Value and installs a _backward closure
    # that accumulates into inputs with += (never =).

    def __pow__(self, n: Number) -> "Value":
        """Return self ** n via stage_04's __pow__, re-tagged as stage-06 Value.

        The forward + ``_backward`` (d/dx = n*x**(n-1)) already live in stage_03;
        here just delegate to super() and re-bless so results chain as stage-06.
        """
        return super(Value, self).__pow__(n)

    def tanh(self) -> "Value":
        """Return tanh(self). Local rule: dt/dx = 1 - tanh(x)**2."""
        out = self._make(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1 - math.tanh(self.data)**2)

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        """Return exp(self). Local rule: de/dx = exp(x)."""
        out = self._make(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.grad * math.exp(self.data)

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        """Return ReLU(self) = max(0, self). Local rule: dr/dx = 1 if x > 0 else 0."""
        out = self._make(max(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0.0 else 0.0)

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        """Return e.g. ``Value(data=2.0, grad=4.0)``."""
        return f"Value(data={self.data}, grad={self.grad})"
