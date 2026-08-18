"""REPL result taxonomy and the Python REPL config schema.

- :class:`PythonREPLConfig` — the ``TypedDict`` schema for the Python REPL's sandbox settings
  (``allowed_imports``, ``allowed_attributes``, ``allow_raise``, ...), i.e. the keys
  :class:`PythonREPL` declares. Configure them on the REPL itself —
  ``jaz.configure(repl=PythonREPL(allowed_imports=[...]))``; this type documents the valid
  keys and type-checks a dict of them.
- :class:`ExecResult`, a union of :class:`Continue`, :class:`Return` and :class:`Raise`
  — the type of the result of a REPL execution.
"""

# Public surface (in ``__all__``, reachable without a NonPublicAPIWarning): the base REPL +
# default concrete (``BaseREPL``/``PythonREPL``) and the execution-result taxonomy
# (``ExecResult``/``Continue``/``Return``/``Raise`` — these surface on hook ``event.exec_result``
# and in effect payloads, so callers touch them directly). Importing this package registers the
# built-in Python REPL as a side effect.
from .base import BaseREPL
from .python_repl import PythonREPL

# Re-exported for reachability (internal call sites import these from the package) but kept OUT
# of ``__all__`` — NOT part of the v1 public surface (user's executive call on the component
# packages' public API): the custom-REPL registration seam (``register_repl``/
# ``REPL_LANGUAGE_MAP``) and the sandbox config schema (``PythonREPLConfig``, configured via
# ``PythonREPL(allowed_imports=[...])`` rather than named directly). The three pluggable-component
# packages are uniform on this: each exposes only its base + default concrete (jaz.llm →
# BaseLLM/LiteLLM, jaz.protocol → BaseProtocol/CodeOnlyProtocol), with registries/registration
# and support types re-exported for reachability but excluded from the public promise. The
# redundant ``X as X`` alias is the ruff-blessed "intentional re-export, not in __all__" idiom.
#
# NOTE: these names previously emitted ``NonPublicAPIWarning`` on access (this package used
# ``make_lazy_getattr`` to demote them). That demotion machinery is deliberately gone: for
# uniformity with ``jaz.llm``/``jaz.protocol`` — which never warned on their non-``__all__``
# re-exports — the three component packages now signal "unsupported" purely via ``__all__``
# membership, not a runtime warning. A name here is off the supported path but no longer says so.
from .python_repl import PythonREPLConfig as PythonREPLConfig
from .registry import REPL_LANGUAGE_MAP as REPL_LANGUAGE_MAP
from .registry import register_repl as register_repl
from .types import Continue, ExecResult, Raise, Return

__all__ = [
    # Base + default concrete REPL.
    "BaseREPL",
    "PythonREPL",
    # REPL execution-result taxonomy (hook `event.exec_result`; effect payloads).
    "ExecResult",
    "Continue",
    "Return",
    "Raise",
]
