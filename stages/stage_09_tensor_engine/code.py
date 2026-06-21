"""Stage 09: Tensor engine -- one N-dimensional reverse-mode autodiff class.

`Tensor` collapses the scalar/Vec/Mat graphs (stages 06-08) onto a single
NumPy-backed node (one data array + one grad array). Every later stage imports
this `Tensor`. Binary ops are restricted to EQUAL-SHAPED operands (or a numeric
constant); general broadcasting-gradient reduction arrives in stage_12.
Allowed tools: Python stdlib + NumPy (forward math / storage only).
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from dlfs import stage_import


def _load_value():
    """Return the scalar `Value` class from stage_06 (alias Stage6_Value)."""
    Stage6_Value = stage_import("stage_06", "Value")
    return Stage6_Value


# An operand we accept: a Tensor, or a raw number/array we wrap.
Operand = Union["Tensor", float, int, np.ndarray, list]


class Tensor:
    """An N-dimensional value node in a reverse-mode autodiff graph.

    Mirrors stage_06's scalar `Value` API but on whole arrays (one ndarray of
    data and one of grad per node). This is the engine every later stage uses.
    """

    def __init__(
        self,
        data: Operand,
        _prev: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        """Wrap `data` as a float64 ndarray and init an autodiff node
        (data, grad zeros, _prev, _op, _backward no-op leaf)."""
        raise NotImplementedError("TODO: store data/grad/_prev/_op/_backward")

    @staticmethod
    def _coerce(other: Operand) -> "Tensor":
        """Return `other` as a Tensor (wrap raw numbers/arrays; pass Tensors through)."""
        raise NotImplementedError("TODO: wrap non-Tensor operands in Tensor(...)")

    @classmethod
    def from_value(cls, v) -> "Tensor":
        """Bridge a stage_06 scalar `Value` into a 0-d `Tensor` leaf (lifts v.data only)."""
        raise NotImplementedError("TODO: implement the Value -> 0-d Tensor leaf bridge")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the underlying data array."""
        raise NotImplementedError("TODO: return self.data.shape")

    @staticmethod
    def _accumulate(grad_into: "Tensor", incoming: np.ndarray) -> None:
        """Add `incoming` into grad_into.grad; sum-to-scalar if grad_into is 0-d
        (the only broadcast case this stage allows; full reduction is stage_12)."""
        raise NotImplementedError("TODO: sum-to-scalar for 0-d operand, else +=")

    # elementwise ops (equal-shaped operands only this stage)
    def __add__(self, other: Operand) -> "Tensor":
        """Elementwise add (route both grads through Tensor._accumulate)."""
        raise NotImplementedError("TODO: implement add + its _backward")

    def __mul__(self, other: Operand) -> "Tensor":
        """Elementwise multiply (route both grads through Tensor._accumulate)."""
        raise NotImplementedError("TODO: implement mul + its _backward")

    def __pow__(self, c: Union[int, float]) -> "Tensor":
        """Raise to a CONSTANT power. z = self ** c."""
        raise NotImplementedError("TODO: implement pow-by-constant + its _backward")

    def relu(self) -> "Tensor":
        """Elementwise ReLU. z = max(0, self)."""
        raise NotImplementedError("TODO: implement relu + its _backward")

    # ops derived from the primitives above (no new _backward)
    def __neg__(self) -> "Tensor":
        """-self, implemented as self * -1."""
        raise NotImplementedError("TODO: return self * -1")

    def __sub__(self, other: Operand) -> "Tensor":
        """self - other, implemented as self + (-other)."""
        raise NotImplementedError("TODO: return self + (-coerced other)")

    def __rsub__(self, other: Operand) -> "Tensor":
        """other - self (for `number - tensor`)."""
        raise NotImplementedError("TODO: return coerced(other) - self")

    def __truediv__(self, other: Operand) -> "Tensor":
        """self / other, implemented as self * other ** -1."""
        raise NotImplementedError("TODO: return self * (coerced(other) ** -1)")

    def __rtruediv__(self, other: Operand) -> "Tensor":
        """other / self (for `number / tensor`)."""
        raise NotImplementedError("TODO: return coerced(other) * (self ** -1)")

    def __radd__(self, other: Operand) -> "Tensor":
        """Reflected add so `number + tensor` works."""
        raise NotImplementedError("TODO: return self + other")

    def __rmul__(self, other: Operand) -> "Tensor":
        """Reflected mul so `number * tensor` works."""
        raise NotImplementedError("TODO: return self * other")

    # autodiff
    def backward(self) -> None:
        """Reverse-mode autodiff: topo-sort, seed grad=ones, reverse-walk _backward
        (grads accumulate with `+=` so reused tensors sum their contributions)."""
        raise NotImplementedError("TODO: topo-sort, seed grad, reverse-iterate _backward")

    def zero_grad(self) -> None:
        """Reset this tensor's gradient to zeros (same shape as data)."""
        raise NotImplementedError("TODO: self.grad = np.zeros_like(self.data)")

    def __repr__(self) -> str:
        """Return 'Tensor(data=..., grad=...)'."""
        raise NotImplementedError("TODO: return formatted repr with data and grad")
