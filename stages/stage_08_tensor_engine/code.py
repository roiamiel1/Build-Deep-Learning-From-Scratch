"""Stage 08: Tensor engine -- one N-dimensional reverse-mode autodiff class.

`Tensor` collapses the scalar/Vec/Mat graphs (stages 06-08) onto a single
NumPy-backed node (one data array + one grad array). Every later stage imports
this `Tensor`. Binary ops are restricted to EQUAL-SHAPED operands (or a numeric
constant); general broadcasting-gradient reduction arrives in stage_11.
Allowed tools: Python stdlib + NumPy (forward math / storage only).
"""

from __future__ import annotations

from typing import Tuple, Union, Set, List

import numpy as np

from dlfs import stage_import


Stage5_Value = stage_import("stage_05", "Value")


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
        
        assert data is not None
        assert isinstance(data, (Tensor, float, int, np.ndarray, list))
        assert _op is not None and isinstance(_op, str)
        assert _prev is not None and isinstance(_prev, tuple)
        assert all(isinstance(x, Tensor) for x in _prev)

        if isinstance(data, (float, int, list)): 
            # data in Number or list - init by wraping it with np.array
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, Tensor): 
            # data in Tensor - init by deep copy it's value
            self.data = data.data.copy().astype(np.float64)
        elif isinstance(data, np.ndarray): 
            self.data = data.copy().astype(np.float64)
        else:
            raise AssertionError(f"Unsupported Tensor initilize by data type {type(data)}")

        self.grad = np.zeros(self.data.shape, dtype=np.float64)
        self._prev = _prev
        self._op = _op
        self._backward = lambda: None

    def _make_tensor(self, *args, **kwargs) -> "Tensor":
        """Build a result node of THIS instance's runtime class.

        Every node-building op below routes its child construction through here
        (``self._make_tensor(data, prev, op)``) instead of calling ``Tensor(...)``
        directly. Because the class is ``type(self)``, a chained graph keeps the
        most-derived subclass at every node: a later stage that subclasses
        ``Tensor`` (e.g. stage_11's broadcasting ``Tensor``) inherits every op
        unchanged and its results are still ITS class -- no op needs re-overriding
        just to swap the constructor. Signature mirrors ``__init__``."""
        return type(self)(*args, **kwargs)

    def _coerce(self, other: Operand) -> "Tensor":
        """Return `other` as a Tensor (wrap raw numbers/arrays; pass Tensors through).

        Wrap via ``self._make_tensor(...)`` (== ``type(self)(...)``) so a coerced raw
        operand becomes THIS instance's runtime class, keeping a subclass alive across
        the chain (same reason the node-building ops route through ``_make_tensor``)."""
        if isinstance(other, Tensor):
            return other

        return self._make_tensor(other)

    @classmethod
    def from_value(cls, v) -> "Tensor":
        """Bridge a stage_05 scalar `Value` into a 0-d `Tensor` leaf (lifts v.data only)."""
        assert isinstance(v, Stage5_Value)
        return Tensor(v.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the underlying data array."""
        return self.data.shape
        
    @property
    def prev(self) -> np.ndarray:
        """Underlying grad prev."""
        return self._prev

    def reshape(self, *shape: int) -> "Tensor":
        """Return a Tensor viewing this data under a new shape. z = self.reshape(shape).

        Pure rearrangement -- no entry is created, destroyed, or combined, so the
        chain rule is just the inverse rearrangement: each upstream grad entry
        belongs to exactly one input entry. Forward `self.data.reshape(shape)`;
        backward reshape `out.grad` back to `self.data.shape` and accumulate.
        Accepts dims as varargs (`t.reshape(2, 3)`) or one tuple (`t.reshape((2, 3))`),
        and a `-1` placeholder NumPy infers (`t.reshape(-1)` flattens).
        Build the child via ``self._make_tensor(...)`` so a subclass survives the chain."""
        result = self._make_tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            Tensor._accumulate(self, result.grad.reshape(self.data.shape)) # -> we just change the shape - it won't affect the gardient.

        result._backward = _backward
        return result

    @staticmethod
    def _accumulate(grad_into: "Tensor", incoming: np.ndarray) -> None:
        """Add `incoming` into grad_into.grad; sum-to-scalar if grad_into is 0-d
        (the only broadcast case this stage allows; full reduction is stage_11)."""
        if grad_into.data.ndim == 0:
            incoming = np.sum(incoming)

        grad_into.grad += incoming

    # elementwise ops (equal-shaped operands only this stage)
    # Every op below builds its result via ``self._make_tensor(...)`` (NOT a bare
    # ``Tensor(...)``) so the node carries the caller's runtime class through the graph.
    def __add__(self, other: Operand) -> "Tensor":
        """Elementwise add (build via self._make_tensor; route both grads through Tensor._accumulate)."""
        other = self._coerce(other)

        result = self._make_tensor(self.data + other.data, _prev=(self, other), _op="+")

        def _backward():
            Tensor._accumulate(self, result.grad)
            Tensor._accumulate(other, result.grad)

        result._backward = _backward
        return result

    def __mul__(self, other: Operand) -> "Tensor":
        """Elementwise multiply (build via self._make_tensor; route both grads through Tensor._accumulate)."""
        other = self._coerce(other)

        result = self._make_tensor(self.data * other.data, _prev=(self, other), _op="*")

        def _backward():
            Tensor._accumulate(self, result.grad * other.data)
            Tensor._accumulate(other, result.grad * self.data)

        result._backward = _backward
        return result

    def __pow__(self, c: Union[int, float]) -> "Tensor":
        """Raise to a CONSTANT power. z = self ** c (build via self._make_tensor)."""
        result = self._make_tensor(self.data ** c, _prev=(self,), _op="**")

        def _backward():
            Tensor._accumulate(self, result.grad * c * (self.data ** (c - 1.0)))

        result._backward = _backward
        return result

    def relu(self) -> "Tensor":
        """Elementwise ReLU. z = max(0, self) (build via self._make_tensor)."""
        result = self._make_tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            Tensor._accumulate(self, result.grad * np.where(self.data > 0, 1, 0))

        result._backward = _backward
        return result

    def tanh(self) -> "Tensor":
        """Elementwise tanh. z = tanh(self); local grad g * (1 - z**2) (build via self._make_tensor)."""
        result = self._make_tensor(np.tanh(self.data), (self,), "tanh")

        def _backward():
            Tensor._accumulate(self, result.grad * (1 - np.tanh(self.data)**2))

        result._backward = _backward
        return result

    def exp(self) -> "Tensor":
        """Elementwise exp. z = exp(self); local grad g * z (build via self._make_tensor)."""
        result = self._make_tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            Tensor._accumulate(self, result.grad * np.exp(self.data))

        result._backward = _backward
        return result

    def log(self) -> "Tensor":
        """Elementwise natural log. z = log(self); local grad g / self (build via self._make_tensor)."""
        result = self._make_tensor(np.log(self.data), (self,), "log")

        def _backward():
            Tensor._accumulate(self, result.grad / self.data)

        result._backward = _backward
        return result

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
        beyond this 1-D<->2-D promotion; that arrives in stage_11.)
        Build the result via ``self._make_tensor(...)`` so a subclass survives the chain."""
        other = self._coerce(other)
        result = self._make_tensor(self.data @ other.data, (self, other), "@")

        # Inserted-axis bookkeeping for the 1-D -> 2-D promotion. A 1-D left
        # operand becomes a (1,n) row (axis 0 inserted); a 1-D right operand
        # becomes an (n,1) column (axis 1 inserted). result.data picks up those
        # same axes, so its upstream grad must be promoted to (1,1)/(1,m)/(n,1)
        # before the 2-D rule, then the inserted axes squeezed back out.
        l1d = self.data.ndim == 1
        r1d = other.data.ndim == 1

        def _backward():
            A = self.data[np.newaxis, :] if l1d else self.data        # (m,k)
            B = other.data[:, np.newaxis] if r1d else other.data      # (k,n)
            G = result.grad                                            # (m,n) once promoted
            if l1d:
                G = G[np.newaxis, :] if not r1d else G[np.newaxis, np.newaxis]
            elif r1d:
                G = G[:, np.newaxis]

            dA = G @ B.T          # (m,k) -> matches A
            dB = A.T @ G          # (k,n) -> matches B
            if l1d:
                dA = dA[0]        # drop inserted row axis -> (k,)
            if r1d:
                dB = dB[:, 0]     # drop inserted col axis -> (k,)

            Tensor._accumulate(self, dA)
            Tensor._accumulate(other, dB)

        result._backward = _backward
        return result

    # ops derived from the primitives above (no new _backward)
    def __neg__(self) -> "Tensor":
        """-self, implemented as self * -1."""
        return self * -1

    def __sub__(self, other: Operand) -> "Tensor":
        """self - other, implemented as self + (-other)."""
        return self + (-other)

    def __rsub__(self, other: Operand) -> "Tensor":
        """other - self (for `number - tensor`)."""
        return (-self) + other

    def __truediv__(self, other: Operand) -> "Tensor":
        """self / other, implemented as self * other ** -1."""
        return self * (other ** -1)

    def __rtruediv__(self, other: Operand) -> "Tensor":
        """other / self (for `number / tensor`)."""
        return (self ** -1) * other

    def __radd__(self, other: Operand) -> "Tensor":
        """Reflected add so `number + tensor` works."""
        return self + other

    def __rmul__(self, other: Operand) -> "Tensor":
        """Reflected mul so `number * tensor` works."""
        return self * other

    # autodiff
    def backward(self) -> None:
        """Reverse-mode autodiff: topo-sort, seed grad=ones, reverse-walk _backward
        (grads accumulate with `+=` so reused tensors sum their contributions)."""
        topo = Tensor.topo_sort(self)

        for v in topo:
            v.zero_grad()

        self.grad += 1.0

        for v in reversed(topo): 
            v._backward()

    @staticmethod
    def topo_sort(root: "Tensor") -> List["Tensor"]:
        """Return nodes reachable from ``root`` in topological order.

        Dependencies first (a parent appears before any child that uses it); the
        backward pass walks this list in reverse. Each node is emitted exactly once
        even when the graph is a DAG with reused nodes (use a ``visited`` set).
        """
        order: List["Tensor"] = []
        visited: Set["Tensor"] = set()

        def build(v: "Tensor") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                order.append(v)

        build(root)
        return order

    def zero_grad(self) -> None:
        """Reset this tensor's gradient to zeros (same shape as data)."""
        self.grad = np.zeros(self.data.shape, dtype=np.float64)

    def __repr__(self) -> str:
        """Return 'Tensor(data=..., grad=...)'."""
        return f"Tensor(data={self.data}, grad={self.grad})"
