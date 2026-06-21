"""Stage 8: Matrix operations.

A `Mat` (2-D matrix of `Value` scalars) on top of the stage_06 scalar engine;
gradients flow through `Value`. For C = A @ B: dL/dA = G @ B.T, dL/dB = A.T @ G.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from typing import Iterable, List, Tuple, Union

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from dlfs import stage_import  # noqa: E402

# Reuse the scalar engine (stage_06) and vector container (stage_07) unchanged.
Stage6_Value = stage_import("stage_06", "Value")
Stage7_Vec = stage_import("stage_07", "Vec")

# Re-export under canonical public names for later stages / tests.
Value = Stage6_Value
Vec = Stage7_Vec

Scalar = Union[int, float, Stage6_Value]


class Mat:
    """A 2-D matrix of `Value` scalars (row-major `List[List[Value]]`). Ops compose
    scalar `Value` arithmetic; no `backward()` (reduce to a scalar first). Each op
    returns a NEW `Mat`."""

    def __init__(self, data: Iterable[Iterable[Scalar]]) -> None:
        """Build self.data: List[List[Value]] from a 2-D iterable; set rows/cols
        (wrap non-`Value` entries, keep shared nodes; ValueError on ragged rows)."""
        self.data: List[List[Stage6_Value]] = []
        self.rows: int = 0
        self.cols: int = 0
        raise NotImplementedError("TODO: build a List[List[Value]] and set rows/cols")

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def __getitem__(self, i: int) -> List[Stage6_Value]:
        """Return row `i` as a list of `Value`."""
        raise NotImplementedError("TODO")

    def __iter__(self):
        """Iterate over rows."""
        raise NotImplementedError("TODO")

    def __repr__(self) -> str:
        return f"Mat(shape={self.shape})"

    def matmul(self, other: "Mat") -> "Mat":
        """Matrix product C = self @ other; (m,k)@(k,n) -> (m,n), where
        C[i][j] = sum_p self[i][p] * other[p][j]. ValueError on inner-dim mismatch."""
        raise NotImplementedError("TODO: build C[i][j] = sum_p self[i][p]*other[p][j]")

    def __matmul__(self, other: "Mat") -> "Mat":
        """The `@` operator; delegates to `matmul`."""
        raise NotImplementedError("TODO: return self.matmul(other)")

    def transpose(self) -> "Mat":
        """Transpose: C[j][i] = self[i][j], shape (cols, rows); reuse the same
        `Value` objects so gradient flows back to the originals."""
        raise NotImplementedError("TODO: build transposed Mat reusing the Values")

    @property
    def T(self) -> "Mat":
        """Property alias for `transpose()`."""
        raise NotImplementedError("TODO: return self.transpose()")

    def reshape(self, rows: int, cols: int) -> "Mat":
        """Reshape to (rows, cols) row-major, reusing the `Value`s; ValueError if
        the element count changes."""
        raise NotImplementedError("TODO: row-major flatten then regroup, reusing Values")

    def sum(self) -> Stage6_Value:
        """Sum of all elements as one `Value`: s = sum_{i,j} self[i][j]."""
        raise NotImplementedError("TODO: sum every Value into one scalar")

    def mean(self) -> Stage6_Value:
        """Mean of all elements as one `Value`: sum() / N, N = rows*cols."""
        raise NotImplementedError("TODO: self.sum() scaled by 1/N")
