"""The REPL execution-result taxonomy: the outcome of running one REPL turn."""

from typing import NamedTuple


class Continue(NamedTuple):
    """An :class:`ExecResult` for a turn that ran and continues to the next turn.

    Covers both a clean success and a recoverable error as one type, distinguished by
    ``exception``: they share control flow (feed ``output`` back, loop again) and payload,
    so the error case is a :class:`Continue` carrying an ``exception`` rather than a separate
    variant.

    Attributes:
        output: The output from the REPL execution, including error information on a
            recoverable error.
        exception: An optional exception object indicating any exception that occurred during
            execution; ``None`` on a clean success. (Terminal raises are :class:`Raise`
            instead of :class:`Continue`.)
    """

    output: str
    # BaseException (not Exception): only the fixed fatal set (KeyboardInterrupt, SystemExit,
    # FatalError, REPL timeouts, …) is propagated as a terminal Raise/abort; any *other*
    # non-Exception BaseException still surfaces here as a recoverable Continue, so the field
    # must admit it. Mirrors Raise.exception. Narrowing to Exception would be a behavior change
    # (reclassifying those as fatal), not a docstring fix.
    exception: BaseException | None = None


class Raise(NamedTuple):
    """An :class:`ExecResult` for a turn that finishes the REPL session by raising an exception
    that propagates out of the invoke.

    Attributes:
        exception: The exception object that was raised.
    """

    # Carries no ``output``: ``output`` is the agent-facing text fed back for the *next* turn,
    # which only a ``Continue`` has — a terminal ``Raise`` ends the invoke, so there is no next
    # turn to feed and its captured stdout has nowhere agent-facing to go. Dropping the field also
    # removes the supply-boundary asymmetry in ``_fold_carried_results`` (a carried ``Raise``'s own
    # output could never lead, so it was silently discarded); with no field there is nothing to
    # discard. Consumers that logged a terminal result's stdout now read it only off a ``Continue``.
    #
    # Known limitation — "terminal" is not guaranteed. A hook can *downgrade* a ``Raise`` /
    # ``Return`` to a recoverable ``Continue`` at ``REPLExecExit`` (``BudgetForcing``,
    # ``ReturnType``), and then there *is* a next turn — but the stdout the code printed before
    # finishing is already gone (``PythonREPL`` no longer captures it into the terminal result), so
    # the downgraded ``Continue`` leads with only the hook's own text. Earlier the fold carried the
    # terminal's ``output`` into the downgrade, so the agent still saw what it had printed.
    # Accepted because the Python REPL's ``reject_finish_on_printed_output`` (default ``True``)
    # already turns print-then-finish into a ``Continue`` *before* it can become a terminal result;
    # the loss is reachable only when that guard is off. Pinned by
    # ``test_terminal_original_contributes_no_output_to_downgrade``.

    exception: BaseException


class Return[ReturnT](NamedTuple):
    """An :class:`ExecResult` for a turn that finishes the REPL session by returning a value that
    propagates out of the invoke.

    Attributes:
        return_value: The value returned by the code.
    """

    # Carries no ``output``, for the same reason as ``Raise``: ``output`` is next-turn agent-facing
    # text, and a ``Return`` terminates the invoke (its value goes to the caller, not back to the
    # agent). See ``Raise`` for the fold-asymmetry rationale and for the known limitation when a
    # hook downgrades a terminal result back to a ``Continue``.

    return_value: ReturnT


type ExecResult[ReturnT] = Continue | Raise | Return[ReturnT]
"""The outcome of running one REPL turn: executing a turn's code yields exactly one of three
results.

- :class:`Continue` — the code ran and the session continues to the next turn.
- :class:`Return` — the code finished the run by returning a value.
- :class:`Raise` — the code finished the run by raising an exception.

An :class:`ExecResult` is what a hook receives as ``event.exec_result``, and what it wraps in
``jaz.hooks.effects.SupplyExecResult`` / :class:`ModifyExecResult` to supply or replace the outcome of a
turn.
"""


type TerminalExecResult[ReturnT] = Return[ReturnT] | Raise
"""The subset of :class:`ExecResult` a run can *terminate* on: a :class:`Return` or a
:class:`Raise`, never a :class:`Continue`.

A single REPL turn may end on any of the three :class:`ExecResult` kinds, but an *invoke* only
ever finishes on a terminal one — a :class:`Continue` appends an observation and loops again, so
by the time the invoke's result exists the loop has already broken on a :class:`Return` /
:class:`Raise`. This alias names that narrower domain so the invoke transform boundary can require
it — see ``jaz.hooks.effects.ModifyInvokeResult`` and ``jaz.hooks.events.InvokeComplete`` — rather
than accept the full :class:`ExecResult` union and lean on a runtime check.
"""

# Why a distinct alias rather than reusing ``ExecResult`` at the invoke boundary: the field type is
# the only place the "no Continue past an invoke's completion" invariant can be *stated*. Reusing
# ``ExecResult`` there let a hook construct ``ModifyExecResult(Continue(...))`` at ``InvokeComplete``
# with nothing rejecting it until a strippable ``assert isinstance(final, Return | Raise)`` deep in
# the dispatcher (and, under ``python -O``, a leaked ``Continue`` corrupting the invoke's outcome).
# The same downgrade-to-``Continue`` is *valid* at ``REPLExecComplete`` (it forces another turn) and
# meaningless at ``InvokeComplete`` (the loop has already broken), so the two Complete boundaries
# genuinely have different valid domains and cannot share one result type — even though they share
# the merge fold. (Executive call, this conversation: split the effect *wrapper* per boundary, keep
# the ``_fold_carried_results`` algorithm shared.)
#
# Named ``TerminalExecResult``, not ``InvokeResult``: this is the terminal *subset of an
# ``ExecResult``* (a REPL-turn property, defined here in REPL vocabulary), not an invoke-layer type —
# the invoke boundary merely consumes it, and the name ``InvokeResult`` would both invert the
# layering and collide with the existing ``InvokeOutcome`` (``Completed[Return | Raise] | Aborted |
# Failed``), two "invoke result" types meaning different things.
