"""dlfs — shared import shim for the *Build Deep Learning From Scratch* curriculum.

Every stage lives in its own directory (``stage_09_tensor_engine/`` etc.) and
exposes its public symbols from ``code.py``.  Because those directory names are
not valid Python identifiers (``stage_09_tensor_engine`` is fine, but we never
want to hard-code the full slug), this module gives every stage ONE clean way to
pull symbols out of an earlier stage:

    from dlfs import stage_import

    # grab one symbol
    Tensor = stage_import("stage_09", "Tensor")

    # grab several at once (returns them in order)
    MLP, mse_loss, SGD = stage_import("stage_15", "MLP", "mse_loss", "SGD")

Design goals
------------
* Real imports, not copy-paste.  Each stage builds on the *code* of earlier
  stages, so the framework genuinely grows stage by stage.
* **Fail loud.**  If the earlier stage is not implemented yet, or does not
  expose the requested symbol, you get a precise, actionable error telling you
  exactly which stage to go implement — never a cryptic ``AttributeError`` or a
  silent ``None``.

How resolution works
---------------------
``stage_import`` finds the sibling directory whose name starts with the given
``stage_prefix`` (e.g. ``"stage_09"`` -> ``stage_09_tensor_engine/``), imports
its ``code.py`` exactly once (cached in ``sys.modules``), and returns the named
attributes.

Notes
-----
* Importing a stage executes its ``code.py`` at module top level.  Skeleton
  stages only *define* classes/functions whose *bodies* raise
  ``NotImplementedError``; defining them does not run those bodies, so importing
  a not-yet-finished stage still succeeds.  The error only surfaces when you
  *call* an unimplemented function in your tests — which is exactly what you
  want.
* A symbol that a stage forgot to define (typo, wrong name, not started) is
  caught here and reported clearly.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os
import sys as _sys
from types import ModuleType
from typing import Any

# Absolute path to the curriculum root (the directory that contains both this
# ``dlfs/`` package and all the ``stage_*`` directories).
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# Cache of already-imported stage modules, keyed by the resolved directory name.
_CACHE: dict[str, ModuleType] = {}


def _find_stage_dir(stage_prefix: str) -> str:
    """Return the single sibling directory name starting with ``stage_prefix``.

    Raises a clear error if zero or more than one match.
    """
    matches = sorted(
        d
        for d in _os.listdir(_ROOT)
        if d.startswith(stage_prefix) and _os.path.isdir(_os.path.join(_ROOT, d))
    )
    if not matches:
        raise ImportError(
            f"[dlfs] No stage directory starts with {stage_prefix!r} under "
            f"{_ROOT!r}.  Check the stage number you asked for."
        )
    if len(matches) > 1:
        raise ImportError(
            f"[dlfs] Ambiguous stage prefix {stage_prefix!r}: matched "
            f"{matches}.  Use a longer prefix."
        )
    return matches[0]


def _import_stage_module(stage_prefix: str) -> ModuleType:
    """Import (once) and return the ``code.py`` module of the named stage."""
    stage_dir = _find_stage_dir(stage_prefix)
    if stage_dir in _CACHE:
        return _CACHE[stage_dir]

    code_path = _os.path.join(_ROOT, stage_dir, "code.py")
    if not _os.path.exists(code_path):
        raise ImportError(
            f"[dlfs] {stage_dir}/code.py does not exist.  You need to "
            f"implement {stage_dir} before a later stage can import from it."
        )

    mod_name = f"dlfs._stages.{stage_dir}"
    spec = _ilu.spec_from_file_location(mod_name, code_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"[dlfs] Could not build an import spec for {code_path!r}.")
    module = _ilu.module_from_spec(spec)
    # Register before exec so intra-stage circular refs (rare) don't recurse.
    _sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except NotImplementedError as exc:  # pragma: no cover - guidance path
        raise ImportError(
            f"[dlfs] Importing {stage_dir}/code.py ran code that raised "
            f"NotImplementedError at module load time.  Earlier-stage skeletons "
            f"should only *define* unimplemented functions, not *call* them at "
            f"the top level.  Implement {stage_dir} first."
        ) from exc
    _CACHE[stage_dir] = module
    return module


def stage_import(stage_prefix: str, *names: str) -> Any:
    """Import one or more public symbols from an earlier stage's ``code.py``.

    Parameters
    ----------
    stage_prefix:
        A prefix that uniquely identifies the source stage directory, e.g.
        ``"stage_09"`` or ``"stage_14"``.
    *names:
        One or more attribute names to pull out of that stage's module.

    Returns
    -------
    The requested attribute if a single name is given, otherwise a tuple of
    attributes in the same order as ``names``.

    Raises
    ------
    ImportError
        If the stage directory / ``code.py`` is missing, or any requested name
        is not defined in it — with a message that names the stage and symbol so
        you know exactly what to implement.
    """
    if not names:
        raise TypeError("stage_import requires at least one symbol name to import.")

    module = _import_stage_module(stage_prefix)
    stage_dir = _find_stage_dir(stage_prefix)

    resolved = []
    for name in names:
        if not hasattr(module, name):
            available = sorted(
                a for a in vars(module) if not a.startswith("_")
            )
            raise ImportError(
                f"[dlfs] {stage_dir}/code.py does not define {name!r}.\n"
                f"        -> Implement {name} in {stage_dir} (or check the "
                f"spelling).\n"
                f"        Public symbols currently exported by that stage: "
                f"{available}"
            )
        resolved.append(getattr(module, name))

    return resolved[0] if len(resolved) == 1 else tuple(resolved)


__all__ = ["stage_import"]
