"""Stage 7: Vector operations.

A thin ``Vec`` container built ON TOP of the stage_06 scalar ``Value`` engine;
every vector op composes scalar ``Value`` ops so autodiff flows through stage_06.
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

# Reuse stage_06's completed scalar engine; re-export `Value` for later stages.
Stage6_Value = stage_import("stage_06", "Value")
Value = Stage6_Value


Scalar = Union[int, float, Stage6_Value]


class Vec:
    """A 1-D vector of stage_06 ``Value`` scalars; ops compose ``Value`` ops so
    gradients flow through stage_06. No ``backward()`` (reduce to a scalar first)."""

    def __init__(self, data: Iterable[Scalar]) -> None:
        """Store ``data`` as a list of ``Value``, wrapping non-``Value`` entries."""
        self.data: List[Stage6_Value] = []
        raise NotImplementedError("TODO: wrap data into a list of Value")

    # --- container protocol ---
    def __len__(self) -> int:
        """Number of elements in the vector."""
        raise NotImplementedError("TODO")

    def __getitem__(self, i: int) -> Stage6_Value:
        """Return the i-th ``Value`` (no copy)."""
        raise NotImplementedError("TODO")

    def __iter__(self):
        """Iterate over the underlying ``Value`` scalars."""
        raise NotImplementedError("TODO")

    def __repr__(self) -> str:
        return f"Vec({[v.data for v in self.data]})"

    # --- elementwise ops (Vec op Vec, or Vec op scalar broadcast) ---
    def __add__(self, other: "Vec | Scalar") -> "Vec":
        """Elementwise add; ``other`` is an equal-length ``Vec`` or a broadcast scalar."""
        raise NotImplementedError("TODO: elementwise add with scalar broadcast")

    def __sub__(self, other: "Vec | Scalar") -> "Vec":
        """Elementwise subtract (see ``__add__`` for broadcast/length rules)."""
        raise NotImplementedError("TODO")

    def __mul__(self, other: "Vec | Scalar") -> "Vec":
        """Elementwise (Hadamard) multiply (see ``__add__`` for the rules)."""
        raise NotImplementedError("TODO")

    # reflected scalar ops: scalar OP vec
    def __radd__(self, other: Scalar) -> "Vec":
        """Return ``other + self`` (commutative)."""
        raise NotImplementedError("TODO")

    def __rmul__(self, other: Scalar) -> "Vec":
        """Return ``other * self`` (commutative)."""
        raise NotImplementedError("TODO")

    def __rsub__(self, other: Scalar) -> "Vec":
        """Return ``other - self`` (e.g. ``10 - vec``)."""
        raise NotImplementedError("TODO")

    # --- reductions (return a scalar Value) ---
    def dot(self, other: "Vec") -> Stage6_Value:
        """Dot product ``sum_i self[i] * other[i]`` as one ``Value``; ValueError if lengths differ."""
        raise NotImplementedError("TODO: dot product from Value ops")

    def sum(self) -> Stage6_Value:
        """Return ``sum_i self[i]`` as a single ``Value``."""
        raise NotImplementedError("TODO")

    # --- elementwise activation ---
    def relu(self) -> "Vec":
        """Elementwise ReLU; return a new ``Vec`` using stage_06 ``Value.relu()``."""
        raise NotImplementedError("TODO")
