"""Stage 6: Scalar reverse-mode autodiff engine.

Imports stage_05's reverse-mode ``Value`` and subclasses it, adding only the new
ops (__pow__, tanh, exp, relu), thin re-blessing __add__/__mul__, and a
grad-aware __repr__. Inherited graph/accumulation machinery is reused, not
re-derived. SKELETON ONLY. Stdlib only (no autodiff libraries).
"""

from __future__ import annotations

from typing import Union

from dlfs import stage_import

# Reverse-mode Value (graph + backward()) from stage_05; extended here.
Stage5_Value = stage_import("stage_05", "Value")

Number = Union[int, float]


class Value(Stage5_Value):
    """Scalar node in a differentiable graph; extends stage_05's reverse-mode Value.

    Inherits __init__/_backward, differentiable __add__/__mul__, reflected ops,
    backward(), and the derived ops (__neg__/__sub__/__rsub__/__truediv__). This
    stage only completes the remaining primitive ops and their gradient rules.
    """

    # Re-bless inherited core ops so they return a stage-06 Value (lets unary
    # ops and __pow__ chain on intermediates); delegates math to stage_05.

    def __add__(self, other: Union["Value", Number]) -> "Value":
        """Return self + other via stage_05's __add__, re-tagged as stage-06 Value."""
        # TODO: delegate to super().__add__, then re-bless the result's class
        raise NotImplementedError

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        """Return self * other via stage_05's __mul__, re-tagged as stage-06 Value."""
        # TODO: delegate to super().__mul__, then re-bless the result's class
        raise NotImplementedError

    # New primitive ops. Each builds a Value and installs a _backward closure
    # that accumulates into inputs with += (never =).

    def __pow__(self, n: Number) -> "Value":
        """Return self ** n for constant int/float n (NOT a Value). d/dx = n*x**(n-1)."""
        # TODO: assert n is a constant int/float, then implement forward + backward
        raise NotImplementedError

    def tanh(self) -> "Value":
        """Return tanh(self). Local rule: dt/dx = 1 - tanh(x)**2."""
        # TODO: implement the forward + backward pass for tanh
        raise NotImplementedError

    def exp(self) -> "Value":
        """Return exp(self). Local rule: de/dx = exp(x)."""
        # TODO: implement the forward + backward pass for exp
        raise NotImplementedError

    def relu(self) -> "Value":
        """Return ReLU(self) = max(0, self). Local rule: dr/dx = 1 if x > 0 else 0."""
        # TODO: implement the forward + backward pass for relu
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return e.g. ``Value(data=2.0, grad=4.0)``."""
        # TODO: implement the grad-aware repr
        raise NotImplementedError
