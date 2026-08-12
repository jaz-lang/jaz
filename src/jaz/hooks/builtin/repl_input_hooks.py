"""REPL-input validation as a hook: ``ValidateREPLInput``.

The pre-execution counterpart to :class:`~jaz.hooks.builtin.return_hooks.ValidateReturn`.
Where ``ValidateReturn`` inspects a ``return``'s *value* at ``REPLExecExit`` (after the code
ran), ``ValidateREPLInput`` inspects the *source code* at ``REPLExecEnter`` (before it runs)
and can veto it. REPL-input validation lives in the effect system rather than inside the
REPL, so the REPL does not special-case it.

The two boundaries use different effects because they sit on opposite sides of execution:

- ``ValidateReturn`` uses ``ModifyResult`` at ``REPLExecExit`` — it *transforms* a result that
  already exists (the ``Return``), folding onto the real output.
- ``ValidateREPLInput`` uses ``OverrideResult`` at ``REPLExecEnter`` — it *supplies* a result
  in place of running the code at all, so a rejected input is never executed (matching the old
  ``_validate_input`` short-circuit, which returned an error result before ``exec``).
"""

from __future__ import annotations

from collections.abc import Callable

from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Effect, OverrideResult
from jaz.hooks.events import InvokeExit, REPLExecEnter
from jaz.repl.types import Continue, Raise
from jaz.string_utils import summarize_exception


class ValidateREPLInput(Hook):
    """Validate REPL input *before* execution, rejecting invalid input as a recoverable error.

    On rejection the code never runs: the agent is shown the validator's exception and retries.
    The ``max_failures``-th rejection ends the invoke by raising instead of being shown, so
    ``max_failures=1`` makes the very first rejection terminal.

    Passed positionally it validates that one invoke. Under ``with`` the *same* validator is
    applied to every invoke in scope, nested ones included — so a sub-invoke's code is held to
    the same rule. Either way each invoke counts its own rejections against ``max_failures``
    rather than drawing down its parent's.

    Args:
        validator: Called with the input source string; it should raise iff the input is
            rejected. The exception it raises is shown to the agent as a recoverable error,
            except for the ``max_failures``-th raise, which propagates out of the invoke.
        max_failures: Rejection count at which the invoke fails terminally. ``None`` (the
            default) never escalates — the agent may keep revising its input until the
            validator accepts it or another limit fires.
    """

    # Handles ``REPLExecEnter`` (the check) and ``InvokeExit`` (releasing the per-invoke count);
    # emits ``OverrideResult`` — a recoverable ``Continue`` on rejection, a terminal ``Raise``
    # at the cap.

    # Multiple suppliers compose: ``OverrideResult`` supplies at ``REPLExecEnter`` fold by the
    # same precedence the exit boundary uses (see ``resolve_override_results``), so two
    # validators — or a validator co-installed with another supplier like the evals
    # ``RestrictReturnValue`` — rejecting the *same* input each contribute a recoverable
    # ``Continue``: their outputs concatenate and their exceptions group into one rejection,
    # rather than raising a hard internal error. A propagating ``with ValidateREPLInput(fn):``
    # and a per-call one therefore stack cleanly. The one unresolvable case is two suppliers
    # naming *distinct* ``Return`` values (``ReturnValueConflictError``) — a validator only ever
    # supplies ``Continue``/``Raise``, so it never hits that.

    def __init__(
        self,
        validator: Callable[[str], None],
        *,
        max_failures: int | None = None,
    ) -> None:
        self.validator = validator
        self.max_failures = max_failures
        # Per-invoke failure counts (keyed by invoke_id); see ValidateReturn / BudgetForcing
        # for why a single shared counter would be wrong across nested invokes.
        self._failures: dict[str, int] = {}

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        # Release the finished invoke's counter so the map doesn't grow across a long-lived
        # hook scope.
        self._failures.pop(event.invoke_id, None)
        return []

    def on_repl_exec_enter(self, event: REPLExecEnter) -> list[Effect]:
        try:
            self.validator(event.code)
        except Exception as exc:  # noqa: BLE001 — any validator exception is a rejection
            count = self._failures.get(event.invoke_id, 0) + 1
            self._failures[event.invoke_id] = count
            if self.max_failures is not None and count >= self.max_failures:
                # Terminal: supply a Raise in place of running the code, ending the invoke.
                # Unlike ValidateReturn there is no return value to augment the message with —
                # the offending input is the code itself, already visible to the agent.
                return [OverrideResult(result=Raise(exception=exc))]
            # Recoverable: supply a Continue carrying the validator's error, so the code is
            # skipped and the agent sees why and revises its next input. The output leads with an
            # explicit header naming the exception type + message (mirrors ValidateReturn), so the
            # <repl_output> block is self-describing instead of empty.
            return [
                OverrideResult(
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
