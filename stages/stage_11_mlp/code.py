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

    Implement EVERY node-building op here so each builds a stage_11 ``Tensor``
    directly, rather than inheriting stage_08's ops (which hardcode the base
    ``Tensor(...)`` constructor). If an op were inherited, a chained graph like
    ``(x @ W).relu()`` would produce a *stage_08* node mid-chain and silently
    drop this class's broadcasting ``__add__`` for any later op keyed on it. By
    re-implementing each op so its result is a stage_11 ``Tensor``, the
    broadcasting-capable class propagates through the whole graph.

    * Broadcasting binary ops (``__add__``/``__mul__``): forward via NumPy
      broadcasting, then *unbroadcast* each parent's grad (sum the broadcast
      axes, reshape) back to its own shape via ``_unbroadcast``.
    * Non-broadcasting ops (``__pow__``, ``relu``, ``tanh``, ``exp``, ``log``,
      ``reshape``, ``@``): same math as stage_08, but the result node is built
      as this class so the subclass survives the chain.
    * Derived ops (``__sub__``, ``__truediv__``, ``__neg__`` and the reflected
      forms) are inherited -- they compose the primitives above, so they stay
      stage_11 with no new ``_backward``.
    """

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
        # TODO: implement the unbroadcast reduction described above:
        #   - while grad.ndim > len(shape): grad = grad.sum(axis=0)
        #   - for each axis i where shape[i] == 1 and grad.shape[i] > 1:
        #         grad = grad.sum(axis=i, keepdims=True)
        #   - return grad.reshape(shape)
        raise NotImplementedError("Tensor._unbroadcast")

    def __add__(self, other: "Operand") -> "Tensor":
        """Broadcasting elementwise add: ``z = self + other``.

        Forward via NumPy broadcasting (``self.data + other.data``); in
        ``_backward`` push ``self._unbroadcast(out.grad, self.shape)`` to ``self``
        and ``self._unbroadcast(out.grad, other.shape)`` to ``other`` so each
        parent's grad is reduced back to its own shape.  Equal-shaped operands
        reduce to the stage_08 behaviour (unbroadcast is then a no-op)."""
        # TODO: coerce other; out = Tensor(self.data + other.data, (self, other), "+");
        #       _backward: each parent.grad += _unbroadcast(out.grad, parent.shape).
        raise NotImplementedError("Tensor.__add__")

    def __mul__(self, other: "Operand") -> "Tensor":
        """Broadcasting elementwise multiply: ``z = self * other``.

        Forward via NumPy broadcasting (``self.data * other.data``); local grads
        are ``g * other`` for ``self`` and ``g * self`` for ``other`` (each
        evaluated at the BROADCAST shape), then unbroadcast back to each parent's
        original shape before accumulating."""
        # TODO: coerce other; out = Tensor(self.data * other.data, (self, other), "*");
        #       _backward: self.grad  += _unbroadcast(out.grad * other.data, self.shape)
        #                  other.grad += _unbroadcast(out.grad * self.data,  other.shape).
        raise NotImplementedError("Tensor.__mul__")

    # Non-broadcasting ops: same math as stage_08, but build the result as THIS
    # class (Tensor(...) here resolves to stage_11's Tensor) so the subclass
    # survives a chained graph instead of decaying to a stage_08 node.
    def __pow__(self, c: Union[int, float]) -> "Tensor":
        """Raise to a CONSTANT power. z = self ** c (local grad g * c * self**(c-1))."""
        # TODO: out = Tensor(self.data ** c, (self,), "**");
        #       _backward: self.grad += out.grad * c * (self.data ** (c - 1.0)).
        raise NotImplementedError("Tensor.__pow__")

    def relu(self) -> "Tensor":
        """Elementwise ReLU. z = max(0, self); local grad g * (self > 0)."""
        # TODO: out = Tensor(np.maximum(self.data, 0.0), (self,), "relu");
        #       _backward: self.grad += out.grad * (self.data > 0).
        raise NotImplementedError("Tensor.relu")

    def tanh(self) -> "Tensor":
        """Elementwise tanh. z = tanh(self); local grad g * (1 - z**2)."""
        # TODO: out = Tensor(np.tanh(self.data), (self,), "tanh");
        #       _backward: self.grad += out.grad * (1.0 - np.tanh(self.data) ** 2).
        raise NotImplementedError("Tensor.tanh")

    def exp(self) -> "Tensor":
        """Elementwise exp. z = exp(self); local grad g * z."""
        # TODO: out = Tensor(np.exp(self.data), (self,), "exp");
        #       _backward: self.grad += out.grad * np.exp(self.data).
        raise NotImplementedError("Tensor.exp")

    def log(self) -> "Tensor":
        """Elementwise natural log. z = log(self); local grad g / self."""
        # TODO: out = Tensor(np.log(self.data), (self,), "log");
        #       _backward: self.grad += out.grad / self.data.
        raise NotImplementedError("Tensor.log")

    def reshape(self, *shape) -> "Tensor":
        """Pure rearrangement; backward reshapes the grad back to self's shape."""
        # TODO: out = Tensor(self.data.reshape(*shape), (self,), "reshape");
        #       _backward: self.grad += out.grad.reshape(self.data.shape).
        raise NotImplementedError("Tensor.reshape")

    def __matmul__(self, other: "Operand") -> "Tensor":
        """Matrix product z = self @ other; same 1-D<->2-D promotion rule as
        stage_08 (dL/dA = G@B.T, dL/dB = A.T@G with inserted axes squeezed back
        out), but the result node is a stage_11 Tensor so the subclass survives
        the chain (e.g. ``(x @ W) + b`` routes through this class's broadcasting
        ``__add__`` for the bias)."""
        # TODO: coerce other; out = Tensor(self.data @ other.data, (self, other), "@");
        #       promote 1-D operands to 2-D, apply G@B.T / A.T@G, squeeze the
        #       inserted axes back out, then self.grad += dA; other.grad += dB.
        raise NotImplementedError("Tensor.__matmul__")


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
    routes through the broadcasting ``__add__``.  Because Python keys ``z + b``
    on ``z``'s type and ``z = x @ W`` is built by the *inherited* stage_08
    ``__matmul__`` (a stage_08 ``Tensor``), the add is invoked *unbound* via
    ``Tensor.__add__(z, b)`` so the broadcasting path runs no matter z's runtime
    class.  ``parameters``/``zero_grad``/``n_in``/``n_out`` are inherited as-is.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: build W (n_in, n_out) and, if bias, b (n_out,) as THIS stage's
        #       broadcasting Tensor (stage_10 builds them as the stage_08 engine,
        #       whose add can't broadcast the bias row).
        raise NotImplementedError("Dense.__init__")

    def __call__(self, x) -> "Tensor":
        """Forward affine pass; ``(n_in,) -> (n_out,)`` or ``(B, n_in) -> (B, n_out)``.

        Bias add is now a plain broadcast: ``(B, n_out) + (n_out,)`` for a batch,
        ``(n_out,) + (n_out,)`` for a single input -- both handled by this stage's
        unbroadcasting ``Tensor.__add__``."""
        # TODO: z = x @ self.W; if bias, add it. NOTE Python keys ``z + b`` on
        #       z's type, and ``z = x @ W`` is built by the inherited stage_08
        #       __matmul__ (a stage_08 Tensor whose add can't broadcast), so
        #       invoke the broadcasting add UNBOUND: ``Tensor.__add__(z, self.b)``.
        raise NotImplementedError("Dense.__call__")


class MLP:
    """A multilayer perceptron: ``Dense`` layers + activations. sizes
    ``[n_in, ..., n_out]`` builds len(sizes)-1 Dense layers; activation follows
    each hidden layer, out_activation the last (each in {"tanh","relu","none"})."""

    def __init__(
        self,
        sizes: Sequence[int],
        activation: str = "tanh",
        out_activation: str = "none",
        seed: Optional[int] = None,
    ) -> None:
        # TODO: validate args; build the Dense layers (per-layer derived seeds).
        raise NotImplementedError("MLP.__init__")

    @staticmethod
    def _apply_activation(z: "Stage8_Tensor", name: str) -> "Stage8_Tensor":
        """Apply named pointwise activation via the Tensor's own methods; raise on unknown name."""
        # TODO: dispatch "none"/"tanh"/"relu" to z / z.tanh() / z.relu().
        raise NotImplementedError("MLP._apply_activation")

    def forward(self, x) -> "Stage8_Tensor":
        """Chain layers, applying activation after each (out_activation after the
        last). x ``(n_in,)`` or ``(batch, n_in)`` -> ``(n_out,)`` / ``(batch, n_out)``."""
        # TODO: chain layers with the right activation per layer.
        raise NotImplementedError("MLP.forward")

    def __call__(self, x) -> "Stage8_Tensor":
        """Alias for :meth:`forward`."""
        # TODO: delegate to forward.
        raise NotImplementedError("MLP.__call__")

    def parameters(self) -> List["Stage8_Tensor"]:
        """Return every learnable parameter from every layer, flattened in layer order."""
        # TODO: flatten each layer's parameters().
        raise NotImplementedError("MLP.parameters")

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zeros."""
        # TODO: zero each parameter's grad.
        raise NotImplementedError("MLP.zero_grad")

    def __repr__(self) -> str:
        # TODO: summarize sizes and activations.
        raise NotImplementedError("MLP.__repr__")
