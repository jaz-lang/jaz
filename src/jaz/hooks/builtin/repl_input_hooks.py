"""REPL-input validation as a hook: ``ValidateREPLInput``.

The pre-execution counterpart to :class:`~ValidateReturn`.
Where ``ValidateReturn`` inspects a ``return``'s *value* at ``REPLExecComplete`` (after the
code ran), ``ValidateREPLInput`` inspects the *source code* at ``REPLExecSend`` (the committed
input, before it runs) and can veto it. REPL-input validation lives in the effect system
rather than inside the REPL, so the REPL does not special-case it.

The two boundaries use different effects because they sit on opposite sides of execution:

- ``ValidateReturn`` uses ``ModifyExecResult`` at ``REPLExecComplete`` — it *transforms* a result
  that already exists (the ``Return``), folding onto the real output.
- ``ValidateREPLInput`` uses ``SupplyExecResult`` at ``REPLExecSend`` — it *supplies* a result
  in place of running the code at all, so a rejected input is never executed (matching the old
  ``_validate_input`` short-circuit, which returned an error result before ``exec``). Send is
  the committed-input veto slot: the validator sees exactly the input that would run.

At the ``max_failures`` cap the per-turn supply gives way to an ``Abort``: ending a run that
would otherwise keep going is control-plane authority, not a result (see the cap comment in
``on_repl_exec_send``).
"""

from __future__ import annotations

from collections.abc import Callable

from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Abort, Effect, SupplyExecResult
from jaz.hooks.events import InvokeExit, REPLExecSend
from jaz.repl.types import Continue
from jaz.string_utils import summarize_exception


class ValidateREPLInput(Hook):
    """Validate REPL input *before* execution, rejecting invalid input as a recoverable error.

    On rejection the code never runs: the agent is shown the validator's exception and retries.
    The agent may be rejected ``max_failures`` times and recover; the *next* rejection ends the
    invoke (the validator's exception aborts out), so ``max_failures=0`` makes the very first
    rejection terminal.

    Passed positionally it validates that one invoke. Under ``with`` the *same* validator is
    applied to every invoke in scope, nested ones included — so a sub-invoke's code is held to
    the same rule. Either way each invoke counts its own rejections against ``max_failures``
    rather than drawing down its parent's.

    Args:
        validator: Called with the input source string; it should raise iff the input is
            rejected. The exception it raises is shown to the agent as a recoverable error,
            except for the one past the cap, which aborts the invoke carrying that exception.
        max_failures: How many rejected inputs the agent may recover from before the next
            rejection aborts the invoke — a tolerance, not a total (``0`` aborts on the first
            rejection; ``2`` allows two recoveries, aborting on the third). Counted
            **cumulatively across the whole invoke**: a rejection on one turn and an unrelated one
            many turns later draw on the same budget, and a valid input does not reset it. ``None``
            (the default) never aborts on rejection — the agent may keep revising indefinitely, so
            ``IterationLimit`` / ``BudgetPool`` is the backstop, not this hook. Must be ``>= 0``.

    Raises:
        ValueError: At construction, if ``max_failures`` is negative.
    """

    # Handles ``REPLExecSend`` (the check) and ``InvokeExit`` (releasing the per-invoke count);
    # emits ``SupplyExecResult`` (a recoverable ``Continue``) per rejection, and ``Abort`` at
    # the ``max_failures`` cap.

    # Multiple suppliers compose: ``SupplyExecResult`` supplies at ``REPLExecSend`` fold by the
    # same precedence the transform boundary uses (see ``resolve_supply_results``), so two
    # validators — or a validator co-installed with another supplier like the evals
    # ``RestrictReturnValue`` — rejecting the *same* input each contribute a recoverable
    # ``Continue``: their outputs concatenate and their exceptions group into one rejection,
    # rather than raising a hard internal error. A propagating ``with ValidateREPLInput(fn):``
    # and a per-call one therefore stack cleanly. The one unresolvable case is two suppliers
    # naming *distinct* ``Return`` values (``ReturnValueConflictError``) — a validator only ever
    # supplies ``Continue``, so it never hits that. Two validators capping on the same input
    # each emit an ``Abort``; the aborts' exceptions group, so neither is dropped.

    def __init__(
        self,
        validator: Callable[[str], None],
        *,
        max_failures: int | None = None,
    ) -> None:
        self.validator = validator
        # ``max_failures`` is a count of *tolerated* rejections (``count > max_failures`` aborts),
        # so ``0`` is valid and meaningful — abort on the first rejection. Only a negative is
        # junk: it would abort before any rejection, and behaves identically to ``0``. Reject
        # negatives; ``None`` remains "never abort". (Same guard on ``ReturnType`` / ``ValidateReturn``.)
        if max_failures is not None and max_failures < 0:
            raise ValueError(f"max_failures must be >= 0, got {max_failures!r}")
        self.max_failures = max_failures
        # Per-invoke failure counts (keyed by invoke_id); see ValidateReturn / BudgetForcing
        # for why a single shared counter would be wrong across nested invokes.
        self._failures: dict[str, int] = {}

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        # Release the finished invoke's counter so the map doesn't grow across a long-lived
        # hook scope.
        self._failures.pop(event.invoke_id, None)
        return []

    def on_repl_exec_send(self, event: REPLExecSend) -> list[Effect]:
        try:
            self.validator(event.code)
        except Exception as exc:  # noqa: BLE001 — any validator exception is a rejection
            count = self._failures.get(event.invoke_id, 0) + 1
            self._failures[event.invoke_id] = count
            if self.max_failures is not None and count > self.max_failures:
                # Terminal: ABORT the invoke with the validator's exception. An Abort, not a
                # supplied Raise result, because this is the control-plane line: the cap
                # ends a run that would otherwise continue — the agent submitted more work,
                # and this hook overrules it — which is exactly the authority Abort names
                # (the REPL-exec span closes Aborted, and the stop is sticky: no
                # Complete-time transform can revoke it, where a supplied Raise result was
                # revocable — the Defect-3 hole of span_event_lifecycle.md). Contrast the
                # return validators (return_hooks.py): they only reshape a terminal the
                # agent ALREADY produced, so they correctly stay ModifyExecResult.
                # Unlike ValidateReturn there is no return value to augment the message
                # with — the offending input is the code itself, already visible.
                return [Abort(error=exc)]
            # Recoverable: supply a Continue carrying the validator's error, so the code is
            # skipped and the agent sees why and revises its next input. The output leads with an
            # explicit header naming the exception type + message (mirrors ValidateReturn), so the
            # <repl_output> block is self-describing instead of empty.
            return [
                SupplyExecResult(
                    result=Continue(
                        # ``summarize_exception`` rather than ``{type}: {exc}``: for an
                        # exception *group* (a validator aggregating several complaints)
                        # ``str()`` renders only a sub-exception count, dropping every child's
                        # text; the summary expands them.
                        output=(
                            "REPL input validation failed with "
                            f"{summarize_exception(exc)}"
                        ),
                        exception=exc,
                    )
                )
            ]
        return []
