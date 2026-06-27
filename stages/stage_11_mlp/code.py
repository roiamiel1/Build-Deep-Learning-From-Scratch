"""Stage 11: MLP + broadcasting-aware Tensor.

Two things ship here:

* ``MLP`` -- a multilayer perceptron: a chain of pure-linear ``Dense`` layers
  (stage_10) with a nonlinearity applied between them, built on the autodiff
  ``Tensor`` (stage_08).
* ``Tensor`` -- a thin subclass of the stage_08 ``Tensor`` that finally adds the
  **broadcasting-correct backward** stage_08 deferred to here.  The stage_08
  engine restricts elementwise binary ops to EQUAL-SHAPED operands; this stage
  overrides ``__add__``/``__mul__`` so differently-shaped-but-broadcastable
  operands forward via NumPy broadcasting AND each parent's gradient is reduced
  (the "unbroadcast" rule) back to that parent's original shape.  stage_12's
  stable softmax (``logits - logsumexp(..., keepdims=True)``, i.e. ``(B,C)-(B,1)``)
  relies on this.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# Building blocks from earlier stages (re-exported for tests / later stages).
from dlfs import stage_import

Stage8_Tensor = stage_import("stage_08", "Tensor")
Stage10_Dense = stage_import("stage_10", "Dense")

# An operand the broadcasting binary ops accept: a Tensor, or a raw
# number/array we wrap (mirrors stage_08's ``Operand`` alias).
Operand = Union["Tensor", float, int, np.ndarray, list]


class Tensor(Stage8_Tensor):
    """The stage_08 autodiff ``Tensor`` extended with broadcasting backward.

    stage_08 keeps elementwise binary ops to equal-shaped operands and defers
    "general broadcasting gradient reduction to stage_11" -- that is this class.

    Only the ops that genuinely CHANGE behaviour are overridden here:
    ``__add__`` and ``__mul__`` (broadcasting forward + ``_unbroadcast`` backward).
    Every other node-building op (``__pow__``, ``relu``, ``tanh``, ``exp``,
    ``log``, ``reshape``, ``@``) is INHERITED from stage_08 unchanged: each builds
    its child via ``self._make_tensor(...)``, which is ``type(self)(...)``, so a
    chained graph like ``(x @ W).relu()`` keeps producing stage_11 ``Tensor``
    nodes and this class's broadcasting ``__add__`` stays reachable for any later
    op keyed on the node's type. Derived ops (``__sub__``, ``__truediv__``,
    ``__neg__`` and the reflected forms) compose the primitives, so they too stay
    stage_11 with no new ``_backward``.
    """
    def _coerce(self, other: Operand) -> "Tensor":
        """Return `other` as a Tensor (wrap raw numbers/arrays; pass Tensors through).

        Wrap via ``self._make_tensor(...)`` (== ``type(self)(...)``) so a coerced raw
        operand becomes THIS instance's runtime class, keeping a subclass alive across
        the chain (same reason the node-building ops route through ``_make_tensor``)."""
        """Return `other` as a Tensor (pass Tensors through, else wrap in Tensor)."""
        if isinstance(other, Tensor):
            return other

        return self._make_tensor(other)

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Sum a broadcasted ``grad`` back to an operand's original ``shape``.

        Forward broadcast *copies* an operand to fill the output; chain rule says
        a copied value's grad is the SUM over its copies. So undo a broadcast by
        summing the copied axes away (stage_08's ``+=`` would shape-mismatch /
        be wrong without this). Two cases compose:
          1. RANK PROMOTION (e.g. bias ``(N,)`` -> ``(B,N)``): sum extra leading
             axes -- ``while grad.ndim > len(shape): grad = grad.sum(axis=0)``.
          2. SIZE-1 STRETCH (e.g. softmax ``(B,1)`` -> ``(B,C)``): sum each
             stretched size-1 axis with ``keepdims=True``.
        Then ``reshape(shape)`` (no-op after 1&2). Equal shapes pass through.
        E.g. ``(2,3)+(3,)``: ``a`` gets ``grad`` as-is, ``b`` gets ``grad.sum(axis=0)``.
        """
        while grad.ndim > len(shape): 
            # This is for batching -> where the first axies is the "batch dim", 
            # we sum along this axis according to the chain rule.
            grad = grad.sum(axis=0)

        for i in range(len(shape)):
            if shape[i] == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad.reshape(shape)

    def __add__(self, other: "Operand") -> "Tensor":
        """Broadcasting elementwise add: ``z = self + other``.

        Forward via NumPy broadcasting (``self.data + other.data``); in
        ``_backward`` push ``self._unbroadcast(out.grad, self.shape)`` to ``self``
        and ``self._unbroadcast(out.grad, other.shape)`` to ``other`` so each
        parent's grad is reduced back to its own shape.  Equal-shaped operands
        reduce to the stage_08 behaviour (unbroadcast is then a no-op)."""
        other = self._coerce(other)
        out = self._make_tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: "Operand") -> "Tensor":
        """Broadcasting elementwise multiply: ``z = self * other``.

        Forward via NumPy broadcasting (``self.data * other.data``); local grads
        are ``g * other`` for ``self`` and ``g * self`` for ``other`` (each
        evaluated at the BROADCAST shape), then unbroadcast back to each parent's
        original shape before accumulating."""
        other = self._coerce(other)
        out = self._make_tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad * other.data, self.shape)
            other.grad += Tensor._unbroadcast(out.grad * self.data,  other.shape)

        out._backward = _backward
        return out

    # NOTE: ``__pow__``, ``relu``, ``tanh``, ``exp``, ``log``, ``reshape`` and
    # ``__matmul__`` are NOT overridden here. stage_08's versions build their
    # child via ``self._make_tensor(...)`` (== ``type(self)(...)``), so when
    # called on a stage_11 instance they already return stage_11 ``Tensor`` nodes
    # -- the subclass survives a chained graph (e.g. ``(x @ W).relu()``) with no
    # re-implementation needed. Their equal-shape (unary / promotion) grads make
    # stage_08's ``_accumulate`` equivalent to an unbroadcast no-op, so only the
    # broadcasting ``__add__``/``__mul__`` above genuinely change behaviour.


# ``Tensor`` (above) is this stage's public broadcasting-capable autodiff node.
# stage_12's stable softmax needs ``(B,C) - (B,1)`` to backprop correctly; that
# broadcasting backward is delivered HERE, satisfying stage_12's reliance even
# though stage_12 keeps importing the engine via ``stage_import("stage_08",
# "Tensor")`` (its construction sites just feed differently-shaped operands).


class Dense(Stage10_Dense):
    """stage_10 ``Dense`` re-expressed on this stage's broadcasting ``Tensor``.

    stage_10 had no broadcasting backward, so it faked the bias add with the
    matmul trick ``z += ones((B,1)) @ b.reshape(1, n_out)``.  This stage's
    ``Tensor.__add__`` unbroadcasts, so the bias add is just ``z + b`` over a
    ``(B, n_out) + (n_out,)`` broadcast -- the grad reduces back to ``(n_out,)``.

    Params (``W``, ``b``) are rebuilt as this stage's ``Tensor`` so ``z + b``
    routes through the broadcasting ``__add__``.  ``z = x @ W`` is built by the
    inherited ``__matmul__``, which constructs its child via ``self._make_tensor``
    (== ``type(self)``); since ``W`` is a stage_11 ``Tensor``, ``z`` is one too,
    so plain ``z + b`` already keys on this stage's broadcasting ``__add__`` --
    no unbound-call trick needed.  ``parameters``/``zero_grad``/``n_in``/``n_out``
    are inherited as-is.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.W = Tensor(np.random.default_rng(seed=seed).random((n_in, n_out)))
        if bias:
            self.b = Tensor(np.zeros(n_out,))
        else:
            self.b = None

    def __call__(self, x) -> "Tensor":
        """Forward affine pass; ``(n_in,) -> (n_out,)`` or ``(B, n_in) -> (B, n_out)``.

        Bias add is now a plain broadcast: ``(B, n_out) + (n_out,)`` for a batch,
        ``(n_out,) + (n_out,)`` for a single input -- both handled by this stage's
        unbroadcasting ``Tensor.__add__``."""
        z = x @ self.W

        if self.b is not None:
            z += self.b

        return z

class MLP:
    """A multilayer perceptron: ``Dense`` layers + activations. sizes
    ``[n_in, ..., n_out]`` builds len(sizes)-1 Dense layers; activation follows
    each hidden layer, out_activation the last (each in {"tanh","relu","none"})."""

    _NONE_ACTIVATION = str(None).lower()
    _VALID_ACTIVATIONS = ["tanh", "relu", _NONE_ACTIVATION]

    def __init__(
        self,
        sizes: Sequence[int],
        activation: str = "tanh",
        out_activation: str = "none",
        seed: Optional[int] = None,
    ) -> None:
        assert len(sizes) > 1
        assert activation in MLP._VALID_ACTIVATIONS
        assert out_activation in MLP._VALID_ACTIVATIONS

        self.layers = []
        self.activation = activation
        self.out_activation = out_activation
        
        rng = np.random.default_rng(seed=seed)

        for (i, j) in zip(sizes[:-1], sizes[1:]): # for sizes [a, b, c] it will be [(a, b), (b, c), ...]
            self.layers.append(Dense(i, j, seed=rng.integers(2**32)))

    @staticmethod
    def _apply_activation(z: "Stage8_Tensor", name: str) -> "Stage8_Tensor":
        """Apply named pointwise activation via the Tensor's own methods; raise on unknown name."""
        assert name in MLP._VALID_ACTIVATIONS

        if name == MLP._NONE_ACTIVATION:
            return z

        return getattr(z, name)()

    def forward(self, x: "Stage8_Tensor") -> "Stage8_Tensor":
        """Chain layers, applying activation after each (out_activation after the
        last). x ``(n_in,)`` or ``(batch, n_in)`` -> ``(n_out,)`` / ``(batch, n_out)``."""
        assert isinstance(x, Tensor)

        for l in self.layers:
            x = l(x)
            x = MLP._apply_activation(x, self.activation)

        return MLP._apply_activation(x, self.out_activation)

    def __call__(self, x: "Stage8_Tensor") -> "Stage8_Tensor":
        """Alias for :meth:`forward`."""
        assert isinstance(x, Tensor)
        return self.forward(x)

    def parameters(self) -> List["Stage8_Tensor"]:
        """Return every learnable parameter from every layer, flattened in layer order."""
        return sum((l.parameters() for l in self.layers), [])

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        for l in self.layers:
            l.zero_grad()

    def __repr__(self) -> str:
        return f"MLP([{', '.join(f'({l.n_in}, {l.n_out})' for l in self.layers)}], activation='{self.activation}', out_activation='{self.out_activation}')"
