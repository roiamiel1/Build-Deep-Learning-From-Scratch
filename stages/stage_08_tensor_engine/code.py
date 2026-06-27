"""Stage 08: Tensor engine -- one N-dimensional reverse-mode autodiff class.

`Tensor` collapses the scalar/Vec/Mat graphs (stages 06-08) onto a single
NumPy-backed node (one data array + one grad array). Every later stage imports
this `Tensor`. Binary ops are restricted to EQUAL-SHAPED operands (or a numeric
constant); general broadcasting-gradient reduction arrives in stage_11.
Allowed tools: Python stdlib + NumPy (forward math / storage only).
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from dlfs import stage_import


def _load_value():
    """Return the scalar `Value` class from stage_05 (alias Stage5_Value)."""
    Stage5_Value = stage_import("stage_05", "Value")
    return Stage5_Value


# An operand we accept: a Tensor, or a raw number/array we wrap.
Operand = Union["Tensor", float, int, np.ndarray, list]


class Tensor:
    """An N-dimensional value node in a reverse-mode autodiff graph.

    Mirrors stage_05's scalar `Value` API but on whole arrays (one ndarray of
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
        """Bridge a stage_05 scalar `Value` into a 0-d `Tensor` leaf (lifts v.data only)."""
        raise NotImplementedError("TODO: implement the Value -> 0-d Tensor leaf bridge")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the underlying data array."""
        raise NotImplementedError("TODO: return self.data.shape")

    def reshape(self, *shape: int) -> "Tensor":
        """Return a Tensor viewing this data under a new shape. z = self.reshape(shape).

        Pure rearrangement -- no entry is created, destroyed, or combined, so the
        chain rule is just the inverse rearrangement: each upstream grad entry
        belongs to exactly one input entry. Forward `self.data.reshape(shape)`;
        backward reshape `out.grad` back to `self.data.shape` and accumulate.
        Accepts dims as varargs (`t.reshape(2, 3)`) or one tuple (`t.reshape((2, 3))`),
        and a `-1` placeholder NumPy infers (`t.reshape(-1)` flattens)."""
        raise NotImplementedError("TODO: reshape forward + _backward (reshape grad back to self.shape)")

    @staticmethod
    def _accumulate(grad_into: "Tensor", incoming: np.ndarray) -> None:
        """Add `incoming` into grad_into.grad; sum-to-scalar if grad_into is 0-d
        (the only broadcast case this stage allows; full reduction is stage_11)."""
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

    def tanh(self) -> "Tensor":
        """Elementwise tanh. z = tanh(self); local grad g * (1 - z**2)."""
        raise NotImplementedError("TODO: implement tanh + its _backward")

    def exp(self) -> "Tensor":
        """Elementwise exp. z = exp(self); local grad g * z."""
        raise NotImplementedError("TODO: implement exp + its _backward")

    def log(self) -> "Tensor":
        """Elementwise natural log. z = log(self); local grad g / self."""
        raise NotImplementedError("TODO: implement log + its _backward")

    def __matmul__(self, other: Operand) -> "Tensor":
        """Matrix product z = self @ other (the ``@`` operator).

        For 2-D z = A @ B with upstream grad G: dL/dA = G @ B.T, dL/dB = A.T @ G.

        Must also cover the 1-D operand forms the neuron / dense layers use:
        (n,)@(n,) -> scalar, (n,)@(n,m) -> (m,), and (b,n)@(n,) -> (b,) batched.
        A 1-D operand makes the bare G@B.T / A.T@G rule wrong (``.T`` is a no-op
        on 1-D, and the right grad is an outer product). The clean fix: promote
        each 1-D operand to 2-D (left -> (1,n) row, right -> (n,1) column),
        apply the 2-D rule above, then squeeze the inserted axis back out so each
        grad matches its operand's original shape. (No general broadcasting
        beyond this 1-D<->2-D promotion; that arrives in stage_11.)"""
        raise NotImplementedError("TODO: matmul + _backward; promote 1-D operands, then G@B.T / A.T@G")

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
