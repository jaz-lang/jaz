"""REPL result taxonomy and the Python REPL config schema.

- :class:`PythonREPLConfig` — the ``TypedDict`` schema for the Python REPL's sandbox settings
  (``allowed_imports``, ``allowed_attributes``, ``allow_raise``, ...), i.e. the keys
  :class:`PythonREPL` declares. Configure them on the REPL itself —
  ``jaz.configure(repl=PythonREPL(allowed_imports=[...]))``; this type documents the valid
  keys and type-checks a dict of them.
- :class:`ExecResult`, a union of :class:`Continue`, :class:`Return` and :class:`Raise`
  — the type of the result of a REPL execution.
"""

# The REPL *extension* API (``REPL``, ``PythonREPL``, ``register_repl``,
# ``REPL_LANGUAGE_MAP`` — for registering a custom REPL language) is re-exported for
# reachability but kept OUT of ``__all__``: custom-REPL registration is an internal
# extension seam, not part of the v1 public surface. Importing this package still
# registers the built-in Python REPL as a side effect.

from typing import TYPE_CHECKING

from .._warnings import make_lazy_getattr

# Importing `python_repl` registers the built-in Python REPL as a side effect (and pulls
# in `base`/`registry`), so the extension API can be resolved lazily below without
# affecting registration.
from .python_repl import PythonREPLConfig
from .types import Continue, ExecResult, Raise, Return

if TYPE_CHECKING:
    # Static types for the reachable-but-demoted REPL extension API (lazy at runtime).
    from .base import REPL as REPL
    from .python_repl import PythonREPL as PythonREPL
    from .registry import REPL_LANGUAGE_MAP as REPL_LANGUAGE_MAP
    from .registry import register_repl as register_repl

__all__ = [
    # Config schema for the public `repl_configs` knob.
    "PythonREPLConfig",
    # REPL execution-result taxonomy (hook `event.exec_result`; effect payloads).
    "ExecResult",
    "Continue",
    "Return",
    "Raise",
]

# Reachable-but-unsupported: the REPL *extension* API (registering a custom REPL
# language). Demoted — warn on access via `__getattr__`.
_DEMOTED = {
    "REPL": ("jaz.repl.base", "REPL"),
    "PythonREPL": ("jaz.repl.python_repl", "PythonREPL"),
    "REPL_LANGUAGE_MAP": ("jaz.repl.registry", "REPL_LANGUAGE_MAP"),
    "register_repl": ("jaz.repl.registry", "register_repl"),
}

__getattr__, __dir__ = make_lazy_getattr(__name__, __all__, _DEMOTED)
