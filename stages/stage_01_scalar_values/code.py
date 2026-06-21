"""Stage 01: Scalar values & arithmetic — the seed of the autodiff engine.

Defines ``Value``, a wrapper around a single scalar with operator overloading
(+ - * / ** neg and reflected forms). No graph bookkeeping or .backward() yet;
those arrive in later stages that subclass this. Standard library only.
"""


class Value:
    """A scalar (stored as float) with arithmetic via operator overloading.

    data: the wrapped number. grad: always 0.0 here; filled by later stages.
    """

    def __init__(self, data):
        """Store data as float; init grad to 0.0."""
        # TODO: set self.data (float) and self.grad
        raise NotImplementedError

    def __repr__(self):
        """Return 'Value(data=<x>)'."""
        # TODO
        raise NotImplementedError

    def __add__(self, other):
        """Return a new Value: self + other (wrap other if it's a number)."""
        # TODO
        raise NotImplementedError

    def __mul__(self, other):
        """Return a new Value: self * other (wrap other if it's a number)."""
        # TODO
        raise NotImplementedError

    def __pow__(self, exponent):
        """Return a new Value: self ** exponent (exponent is int/float, not Value)."""
        # TODO
        raise NotImplementedError

    def __neg__(self):
        """Return -self (as self * -1)."""
        # TODO
        raise NotImplementedError

    def __sub__(self, other):
        """Return self - other (as self + (-other))."""
        # TODO
        raise NotImplementedError

    def __truediv__(self, other):
        """Return self / other (as self * other ** -1)."""
        # TODO
        raise NotImplementedError

    # reflected operators: enable `2 * a`, `1 + a`, `3 - a`, `6 / a`

    def __radd__(self, other):
        """Return other + self."""
        # TODO
        raise NotImplementedError

    def __rmul__(self, other):
        """Return other * self."""
        # TODO
        raise NotImplementedError

    def __rsub__(self, other):
        """Return other - self (as other + (-self))."""
        # TODO
        raise NotImplementedError

    def __rtruediv__(self, other):
        """Return other / self (as other * self ** -1)."""
        # TODO
        raise NotImplementedError
