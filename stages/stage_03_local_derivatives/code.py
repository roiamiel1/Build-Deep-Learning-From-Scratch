"""Stage 3: Local derivatives as `_backward` closures.

In stage_02 every result node reserved a no-op ``_backward`` hook. Here we fill
it in: each op installs a closure that, *given the output's* ``grad``, pushes the
local derivative onto each input's ``grad`` (with ``+=`` so a reused operand
accumulates from every consumer). This is the per-edge half of backprop — there
is still NO global pass that runs these closures in order; that is stage_04's
``backward()``. The closures capture their operands directly, which is why the
asymmetric ops (`-`, `/`) need no operand-order bookkeeping on the node.

Each op override is two lines of new work: let ``super()`` build the result node
(it already does the forward math and records ``_prev``/``_op``), then attach the
gradient closure. You never re-do the forward arithmetic here.

Local rules (output grad ``g`` flows back as):
  z = a + b   -> a.grad += g,        b.grad += g
  z = a * b   -> a.grad += b * g,    b.grad += a * g
  z = a ** c  -> a.grad += c*a**(c-1) * g          (c constant)
  z = -a      -> falls out of a * -1
  z = a - b   -> falls out of a + (-b)
  z = a / b   -> falls out of a * b**-1

Allowed tools: Python stdlib only.
"""

from __future__ import annotations

from typing import Union

from dlfs import stage_import

# stage_02 graph node (set `_prev`, `_op`, no-op `_backward`); subclass it.
Stage2_Value = stage_import("stage_02", "Value")

Number = Union[int, float]


class Value(Stage2_Value):
    """stage_02 graph `Value`, extended so each op installs its `_backward` rule.

    ``__add__``/``__mul__``/``__pow__`` let ``super()`` build the result node (the
    forward math + ``_prev``/``_op`` recording is inherited unchanged) and then set
    ``out._backward`` to a closure that accumulates the local derivative onto each
    operand's ``.grad`` using the output's ``.grad``. The derived ops (`-`, unary
    `-`, `/`) inherit from stage_01 and compose out of `+`/`*`/`**`, so they get
    correct gradients for free — no separate closure needed.

    Coerce ``other`` to a ``Value`` first (as the inherited op does), then capture
    ``self`` and ``other`` directly in the closure — that is how the asymmetric ops
    know which side is which without the node storing operand order.
    """

    def __add__(self, other: "Value | Number") -> "Value":
        """self + other; install _backward: a.grad += out.grad; b.grad += out.grad."""
        other = other if isinstance(other, Value) else Value(other)
        out = super().__add__(other)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | Number") -> "Value":
        """self * other; install _backward: a.grad += b*out.grad; b.grad += a*out.grad."""
        other = other if isinstance(other, Value) else Value(other)
        out = super().__mul__(other)

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, c: Number) -> "Value":
        """self ** c (c a constant int/float); install _backward: a.grad += c*a**(c-1)*out.grad."""
        out = super().__pow__(c)

        def _backward():
            self.grad += out.grad * c * (self.data ** (c - 1.0))

        out._backward = _backward
        return out

    # __neg__ / __sub__ / __rsub__ / __truediv__ / __rtruediv__ are INHERITED from
    # stage_01 and compose out of __add__/__mul__/__pow__ above, so their gradients
    # flow through the closures installed here — do not re-implement them.

    def __repr__(self) -> str:
        """e.g. ``Value(data=3.0, grad=0.0)`` (grad still 0 until stage_04 backward)."""
        return f"Value(data={self.data}, grad={self.grad})"
