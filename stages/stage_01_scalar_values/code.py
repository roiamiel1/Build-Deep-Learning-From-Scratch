"""Stage 01: Scalar values & arithmetic — the seed of the autodiff engine.

Defines ``Value``, a wrapper around a single scalar with operator overloading
(+ - * / ** neg and reflected forms). No gradients and no graph *walk* yet — but
the class is shaped from the start the way the finished engine needs it, so later
stages only ADD behavior and never rewrite these operators:

* The constructor already takes ``(data, _children=(), _op="")`` and stores the
  graph fields ``_prev`` / ``_op`` / ``_backward``. They are inert plumbing here
  (nothing reads them yet); stage_02 adds the ``trace`` walk that uses ``_prev``,
  stage_03 fills the ``_backward`` hook, stage_04 walks it for ``.backward()``.
* Every operator builds its result through ``self._make(...)``, which constructs
  ``type(self)(...)`` — so when a later stage subclasses ``Value``, the SAME
  operators return that subclass and record the operands, with no override needed.

Standard library only.
"""


class Value:
    """A scalar (stored as float) with arithmetic via operator overloading.

    Fields:
    - ``data``: the wrapped number (float).
    - ``grad``: 0.0 here; filled by the stage_04 ``.backward()`` pass.
    - ``_prev``: the set of operand ``Value``s this result was built from
      (``set()`` for a leaf). A set so a reused operand (``a * a``) is one parent.
    - ``_op``: a string label for the op that built this node (``''`` for a leaf).
    - ``_backward``: a no-op closure here; stage_03 installs the per-op gradient
      rule on it. Reserving the field now keeps the field set stable across stages.

    Operand ORDER is not stored: ``_prev`` is a set, which is fine because the
    stage_03 gradient closures capture each operand directly, so ``a - b`` and
    ``a / b`` know which side is which without the node remembering order.
    """

    def __init__(self, data, _children=(), _op=""):
        """Store ``data`` as float, init ``grad``, and record graph provenance."""
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def _make(self, data, _children, _op):
        """Build a result node of THIS class: ``type(self)(data, _children, _op)``.

        Using ``type(self)`` (not ``Value``) means a subclass's operators return the
        subclass, so the whole graph is uniform and later stages need no op rewrites.
        """
        return type(self)(data, _children, _op)

    def __repr__(self):
        """Return 'Value(data=<x>)'."""
        return f"Value(data={self.data})"

    def __add__(self, other):
        """self + other; record (self, other) and op '+'. Wrap a number operand."""
        other = other if isinstance(other, Value) else type(self)(other)
        return self._make(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        """self * other; record (self, other) and op '*'. Wrap a number operand."""
        other = other if isinstance(other, Value) else type(self)(other)
        return self._make(self.data * other.data, (self, other), "*")

    def __pow__(self, exponent):
        """self ** exponent (int/float, not Value); record (self,) and op f'**{exponent}'."""
        assert isinstance(exponent, (int, float))
        return self._make(self.data ** exponent, (self,), f"**{exponent}")

    def __neg__(self):
        """Return -self (as self * -1)."""
        return self * -1

    def __sub__(self, other):
        """Return self - other (as self + (-other))."""
        return self + (-other)

    def __truediv__(self, other):
        """Return self / other (as self * other ** -1)."""
        return self * other ** -1

    # reflected operators: enable `2 * a`, `1 + a`, `3 - a`, `6 / a`

    def __radd__(self, other):
        """Return other + self."""
        return self + other

    def __rmul__(self, other):
        """Return other * self."""
        return self * other

    def __rsub__(self, other):
        """Return other - self (as other + (-self))."""
        return other + (-self)

    def __rtruediv__(self, other):
        """Return other / self (as other * self ** -1)."""
        return other * self ** -1
