"""Stage 06: Vector operations.

A thin ``Vec`` container built ON TOP of the stage_05 scalar ``Value`` engine;
every vector op composes scalar ``Value`` ops so autodiff flows through stage_05.
NumPy may be used for forward array creation only, never for gradients.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, List, Union

# Plumbing: make the curriculum root importable so `dlfs` resolves.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dlfs import stage_import  # noqa: E402

# Reuse stage_05's completed scalar engine; re-export `Value` for later stages.
Stage5_Value = stage_import("stage_05", "Value")
Value = Stage5_Value


Scalar = Union[int, float, Stage5_Value]


class Vec:
    """A 1-D vector of stage_05 ``Value`` scalars; ops compose ``Value`` ops so
    gradients flow through stage_05. No ``backward()`` (reduce to a scalar first)."""

    def __init__(self, data: Iterable[Scalar]) -> None:
        """Store ``data`` as a list of ``Value``, wrapping non-``Value`` entries."""
        self.data = [x if isinstance(x, Stage5_Value) else Stage5_Value(x) for x in data]

    # --- container protocol ---
    def __len__(self) -> int:
        """Number of elements in the vector."""
        return len(self.data)

    def __getitem__(self, i: int) -> Stage5_Value:
        """Return the i-th ``Value`` (no copy)."""
        return self.data[i]

    def __iter__(self):
        """Iterate over the underlying ``Value`` scalars."""
        return self.data.__iter__()

    def __repr__(self) -> str:
        return f"Vec({[v.data for v in self.data]})"

    # --- elementwise ops (Vec op Vec, or Vec op scalar broadcast) ---
    def __add__(self, other: "Vec | Scalar") -> "Vec":
        """Elementwise add; ``other`` is an equal-length ``Vec`` or a broadcast scalar."""
        other = other if isinstance(other, Vec) else Vec([other] * len(self))
        assert len(self) == len(other)
        return Vec(a + b for (a, b) in zip(self.data, other.data))

    def __sub__(self, other: "Vec | Scalar") -> "Vec":
        """Elementwise subtract (see ``__add__`` for broadcast/length rules)."""
        other = other if isinstance(other, Vec) else Vec([other] * len(self))
        assert len(self) == len(other)
        return Vec(a - b for (a, b) in zip(self.data, other.data))

    def __mul__(self, other: "Vec | Scalar") -> "Vec":
        """Elementwise (Hadamard) multiply (see ``__add__`` for the rules)."""
        other = other if isinstance(other, Vec) else Vec([other] * len(self))
        assert len(self) == len(other)
        return Vec(a * b for (a, b) in zip(self.data, other.data))

    # reflected scalar ops: scalar OP vec
    def __radd__(self, other: Scalar) -> "Vec":
        """Return ``other + self`` (commutative)."""
        return self + other

    def __rmul__(self, other: Scalar) -> "Vec":
        """Return ``other * self`` (commutative)."""
        return self * other

    def __rsub__(self, other: Scalar) -> "Vec":
        """Return ``other - self`` (e.g. ``10 - vec``)."""
        return Vec([other] * len(self)) - self

    # --- reductions (return a scalar Value) ---
    def dot(self, other: "Vec") -> Stage5_Value:
        """Dot product ``sum_i self[i] * other[i]`` as one ``Value``; ValueError if lengths differ."""
        return (self * other).sum()

    def sum(self) -> Stage5_Value:
        """Return ``sum_i self[i]`` as a single ``Value``."""
        assert len(self) >= 1
        total = self[0]
        for i in range(1, len(self)):
            total += self[i]
        return total

    # --- elementwise activation ---
    def relu(self) -> "Vec":
        """Elementwise ReLU; return a new ``Vec`` using stage_05 ``Value.relu()``."""
        return Vec(x.relu() for x in self)
