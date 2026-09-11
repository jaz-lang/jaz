"""REPL result taxonomy, the Python REPL config schema, and the per-invoke REPL state.

- :class:`PythonREPLConfig` — the ``TypedDict`` schema for the Python REPL's sandbox settings
  (``allowed_imports``, ``allowed_attributes``, ``allow_raise``, ...), i.e. the keys
  :class:`PythonREPL` declares. Configure them on the REPL itself —
  ``jaz.configure(repl=PythonREPL(allowed_imports=[...]))``; this type documents the valid
  keys and type-checks a dict of them.
- :class:`ExecResult`, a union of :class:`Continue`, :class:`Return` and :class:`Raise`
  — the type of the result of a REPL execution.
- :class:`REPLState` — one invoke's REPL state, which :meth:`BaseREPL.initialize` returns and
  every stateful method takes back. Needed to write a custom REPL.
"""

# Public surface (in ``__all__``, reachable without a NonPublicAPIWarning): the base REPL +
# default concrete (``BaseREPL``/``PythonREPL``) and the execution-result taxonomy
# (``ExecResult``/``Continue``/``Return``/``Raise`` — these surface on hook ``event.exec_result``
# and in effect payloads, so callers touch them directly). Importing this package registers the
# built-in Python REPL as a side effect.
from .base import BaseREPL

# Re-exported for reachability (internal call sites import these from the package) but kept OUT
# of ``__all__`` — NOT part of the v1 public surface (user's executive call on the component
# packages' public API): the sandbox config schema (``PythonREPLConfig``, configured via
# ``PythonREPL(allowed_imports=[...])`` rather than named directly), and ``REPLState``.
#
# ``jaz.protocol``'s ``ParsedCode`` is the settled precedent for ``REPLState``: a public ABC
# method (``BaseProtocol.parse``) returns it, it is re-exported ``as``, and it stays out of
# ``__all__`` because only implementers meet it. What would make it public instead is a *caller*
# being handed one by another route — that is why ``ExecResult`` is in ``__all__``, and
# ``REPLState``, which appears on no hook event and in no effect payload, is not.
#
# The other two component packages take the same shape (jaz.llm → BaseLLM/LiteLLM,
# jaz.protocol → BaseProtocol/CodeOnlyProtocol), and none of them warns on a non-``__all__``
# re-export: these names signal "unsupported" through ``__all__`` membership alone, not a
# runtime warning. The redundant ``X as X`` alias is the ruff-blessed "intentional re-export"
# idiom.
#
# NOTE the ordering below: ruff's isort merges this whole block, so ``PythonREPL`` — one of the
# two PUBLIC names — sorts under this comment rather than above it. Membership of ``__all__`` is
# the contract, not position in this list.
from .base import REPLState as REPLState
from .python_repl import PythonREPL
from .python_repl import PythonREPLConfig as PythonREPLConfig
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
