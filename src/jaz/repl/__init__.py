"""REPL result taxonomy and the base contract a REPL implementation satisfies.

- :class:`ExecResult`, a union of :class:`Continue`, :class:`Return` and :class:`Raise`
  — the type of the result of a REPL execution.
- :class:`BaseREPL`, generic in the per-invoke state type an implementation chooses, and
  :class:`PythonREPL`, the shipped in-process Python implementation.
- :class:`REPLHistoryEntry`, one entry of the ``__history__`` record :class:`PythonREPL` keeps.
"""

# Public surface (``__all__``): the base REPL + default concrete (``BaseREPL``/``PythonREPL``)
# and the execution-result taxonomy (``ExecResult``/``Continue``/``Return``/``Raise`` plus the
# terminal subset ``TerminalExecResult`` — these surface on hook ``event.exec_result``, on
# ``InvokeComplete.result``, and in effect payloads, so callers touch them directly).
#
# NO per-invoke state type is exported, and that is the design rather than an omission.
# ``BaseREPL`` is generic in ``StateT``: ``initialize`` returns one and the stateful methods take
# it back, but the type filling that slot is the *implementation's own*. An implementer writes
# `class MyREPL(BaseREPL[MyREPLState])` and names only a class they wrote, so the package has no
# type to advertise here — it does not own one.
#
# This replaced promoting a single concrete ``REPLState`` into ``__all__`` (EXECUTIVE CALL, user,
# 2026-09-13). That promotion was right that a type a public ABC's abstract methods force into an
# implementer's signatures cannot be left unadvertised — the rule that put ``LLMResponse`` in
# ``jaz.llm.__all__`` — and wrong about the remedy: it published the *host Python REPL's* fields
# (a namespace dict, a subinvoke, a session id) as the shape every REPL must inherit. Making
# the state a type parameter satisfies the same rule by removing the obligation instead of
# discharging it, and frees an out-of-process REPL from inheriting fields it cannot use.
#
# ``PythonREPLState`` is exported too (EXECUTIVE CALL, user, 2026-09-14), reversing an earlier
# call here that it was "the shipped REPL's private detail". That argument narrowed the rule to
# "no caller CONSTRUCTS one", and the rule is "a caller is HANDED one" — which they are:
# ``PythonREPL`` is public, and ``initialize()`` -> ``exec(state, ...)`` is the worked example in
# ``BaseREPL.initialize``'s own docstring. It also showed on the generated ``repl.md`` in seven
# places with no page behind it, which is the same defect as a published signature naming a
# private default.
#
# Exporting it does NOT re-create the problem the type parameter solved. That problem was
# *inheritance* — every REPL forced to carry the host Python REPL's fields. ``BaseREPL[StateT]``
# still frees an implementer to declare their own state; this only names the one the shipped
# REPL hands back, for the callers who already hold it.
#
# NOTE the ordering below: ruff's isort merges this whole block, so the names sort together
# regardless of this comment's position. Membership of ``__all__`` is the contract, not position.
from ._base import BaseREPL
from ._python_repl import PythonREPL, PythonREPLState, TracebackVerbosity
from ._repl_history import REPLHistoryEntry
from ._types import Continue, ExecResult, Raise, Return, TerminalExecResult

__all__ = [
    # Base + default concrete REPL, and the state that REPL hands back.
    "BaseREPL",
    "PythonREPL",
    "PythonREPLState",
    # A caller is handed one: `PythonREPL.traceback_verbosity` holds the coerced enum, and it is
    # the type `PythonREPL.__init__`'s published signature names.
    "TracebackVerbosity",
    # The record `PythonREPL.update_repl_history` appends — what an agent reads off `__history__`
    # and what a sub-agent's `return __history__` hands its parent — so a caller can name what
    # they are handed.
    "REPLHistoryEntry",
    # REPL execution-result taxonomy (hook `event.exec_result`; effect payloads).
    "ExecResult",
    "Continue",
    "Return",
    "Raise",
    # The terminal subset (`Return | Raise`), by the same rule the comment above states: a caller
    # is handed one. It is the declared type of `InvokeComplete.result`, so a hook author who
    # annotates that field needs a name for it — and the only one that existed was
    # `jaz.hooks.effects.TerminalExecResult`, reachable purely because `effects.py` imports it,
    # unadvertised, and in a package that does not own the type.
    "TerminalExecResult",
]
