"""Stage 4: Chain rule.

Manually propagate derivatives through multi-op expressions by composing the
*local* derivatives from stage_03. No autodiff engine here -- plain floats with
explicit forward/backward code. Derivatives must be analytical (compose the
stage_03 locals); do NOT use finite differences. Skeleton only.
"""

from __future__ import annotations

import math
from typing import Dict, List

from dlfs import stage_import

# Reuse: graph node from stage_02, per-op local derivatives from stage_03.
Stage2_Value = stage_import("stage_02", "Value")
Stage3_d_add, Stage3_d_mul = stage_import("stage_03", "d_add", "d_mul")


def chain(locals_list: List[float]) -> float:
    """Chain rule along a single straight path: product of local derivatives.

    For ``x -> u -> ... -> y``, dy/dx is the product of the consecutive locals.
    Empty list returns 1.0 (multiplicative identity).
    """
    raise NotImplementedError("TODO: multiply the local derivatives; empty -> 1.0")


def accumulate(path_derivs: List[float]) -> float:
    """Multivariate chain rule: sum the per-path total derivatives.

    When one input reaches the output via several paths, dy/dx is their sum.
    Empty list returns 0.0.
    """
    raise NotImplementedError("TODO: sum the per-path derivatives; empty -> 0.0")


def f_chain(x: float) -> tuple[float, float]:
    """Straight chain: u = 3x + 1, v = u**2, y = tanh(v). Returns (y, dy/dx).

    Backward multiplies the three locals (use ``chain``). Analytical only.
    """
    raise NotImplementedError("TODO: forward pass for y, backward pass for dy/dx")


def f_branch(x: float) -> tuple[float, float]:
    """Branching: a = x + 1, b = x**2, y = a * b. Returns (y, dy/dx).

    ``x`` feeds both branches, so dy/dx sums over two paths (use ``accumulate``).
    Analytical only.
    """
    raise NotImplementedError("TODO: forward pass, then sum-over-paths backward")


def backward_pass(x: float) -> Dict[str, float]:
    """Full manual reverse pass over f_branch (a = x+1, b = x**2, y = a*b).

    Seed dy/dy = 1.0 and propagate backward, accumulating at ``x``. Returns
    {"y","a","b","x"} -> d(output)/d(node); "x" must equal f_branch(x)[1].
    """
    raise NotImplementedError("TODO: build the {name: derivative} reverse-pass dict")
