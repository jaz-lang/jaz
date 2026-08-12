"""Hook that refuses an early ``return``/``raise`` for a configurable number of times.

"Budget forcing" encourages the agent to keep working instead of finishing
immediately: while active, an attempted ``return`` or ``raise`` is converted into a
recoverable Continue (via a ``ModifyResult`` at ``REPLExecExit`` carrying a
``Continue``) with a ``BudgetForcingError``, so the agent sees the error and continues.
A finish only ever comes from execution, so the conditional ``REPLExecExit`` is exactly
this hook's scope — it never wanted parse-failure turns (which produce no result to refuse).

Apply it as a context manager around an invoke::

    from jaz.hooks import BudgetForcing

    with BudgetForcing(refusals=2):
        result = jaz.invoke(jaz.ReturnType(str), task=task)

Behavior notes:
- The decision is evaluated at ``REPLExecExit``. A refusal (and counter
  decrement) only happens when the execution actually produced a
  Return/Raise, never for regular code, and never on the final
  iteration (``iteration == max_iterations``), where the agent must finish.
- The refusal budget is tracked **per invoke** (keyed by ``invoke_id``): each
  invoke in scope, including nested ``invoke()`` calls, gets its own fresh
  budget of ``refusals``. The hook propagates via contextvars, but a shared
  counter would let a parent's early refusals starve nested invokes (or vice
  versa), so each invoke's budget is independent.
"""

from jaz.exceptions import BudgetForcingError
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Effect, ModifyResult
from jaz.hooks.events import InvokeExit, REPLExecExit
from jaz.repl.types import Continue, Raise, Return
from jaz.string_utils import summarize_exception


def _refuse(message: str) -> ModifyResult:
    """Downgrade a finishing turn to a recoverable ``Continue`` carrying *message* as the refusal.

    One constructor for both the ``return`` and ``raise`` refusals so ``output`` is always derived
    from the exception rather than restated beside it.
    """
    exc = BudgetForcingError(message)
    return ModifyResult(result=Continue(output=summarize_exception(exc), exception=exc))


class BudgetForcing(Hook):
    """Refuse an attempted finish (``return``/``raise``) up to a given number of times.

    A refused finish is shown to the agent as a recoverable error, so it keeps working
    instead of stopping. After ``refusals`` refusals this hook stops refusing and lets the
    next one through.

    The agentic analog of the budget-forcing decoding strategy in s1: Simple Test-Time Scaling
    (Muennighoff et al., EMNLP 2025, https://aclanthology.org/2025.emnlp-main.1025/), which
    suppresses the model's end-of-thinking token to make it keep reasoning. Here the finish
    being refused is the agent's ``return``/``raise`` rather than a decoding token.

    Under ``with`` it applies to every invoke in scope, nested ones included, each counting
    its own refusals against ``refusals``. Passed positionally, only that invoke's finishes
    are refused.

    Args:
        refusals: Number of early ``return``/``raise`` attempts to refuse before
            allowing the agent to finish.
    """

    # Messages shown to the agent when an early ``return``/``raise`` is refused.
    BUDGET_FORCING_RETURN_MESSAGE = (
        "Budget forcing active: You cannot finish yet.\n\n"
        "Before returning a final answer, please:\n"
        "- Double-check your reasoning\n"
        "- Verify your solution is correct\n"
        "- Consider edge cases or alternative approaches\n"
        "- Test your answer if possible"
    )

    BUDGET_FORCING_RAISE_MESSAGE = (
        "Budget forcing active: You cannot raise an error yet.\n\n"
        "Before raising an error, please:\n"
        "- Reconsider whether the task is truly impossible\n"
        "- Try alternative approaches\n"
        "- Check if you misunderstood the requirements\n"
        "- Verify any assumptions you made"
    )

    def __init__(self, refusals: int) -> None:
        self.refusals = refusals
        # Per-invoke refusal budgets, keyed by invoke_id. Each invoke (including
        # nested ones) draws down its own fresh budget; see the module docstring
        # for why a single shared counter is wrong.
        self._refusals_remaining: dict[str, int] = {}

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        # Release a finished invoke's budget so the map doesn't grow unbounded
        # across many invokes in a long-lived hook scope.
        self._refusals_remaining.pop(event.invoke_id, None)
        return []

    def on_repl_exec_exit(self, event: REPLExecExit) -> list[Effect]:
        # Refuse an early ``return``/``raise`` produced by this iteration's execution.
        # TODO(#447): Restore cost-budget awareness. This hook used to also disable
        # forcing once the invoke's cost / LLM-calls budget was exhausted, so it
        # wouldn't refuse a finish (and burn an extra over-budget call) when
        # there was no budget left. The dropped logic was:
        #
        #     def _budgets_ok(self) -> bool:
        #         return self.cost_tracker is None or (
        #             not self.cost_tracker.is_llm_cost_budget_exhausted()
        #             and not self.cost_tracker.is_llm_calls_budget_exhausted()
        #         )
        #     ...
        #     forcing_active = (
        #         self.refusals_remaining > 0
        #         and event.iteration < event.max_iterations
        #         and self._budgets_ok()
        #     )
        #
        # It was removed because a user-facing hook can't reach the per-invoke
        # CostTracker (internal, no accessor). To bring it back, compose this
        # hook with budget *enforcement* once that is also a hook
        # (e.g. BudgetPool). Note: effect composition alone won't do it
        # -- on an over-budget finish, this hook emits a refusal ModifyResult and
        # enforcement emits nothing, so the refusal wins (the opposite of what we
        # want), and the model has no way for one hook to veto another's effect.
        # Proposed mechanisms:
        #   Option A (preferred, minimal): BudgetPool publishes "is the
        #     budget exhausted?" via a shared accessor (e.g. a contextvar set by
        #     invoke / the tracking hook). This hook reads it and skips forcing
        #     when exhausted -- restores _budgets_ok() without new effect
        #     semantics or per-hook CostTracker plumbing.
        #   Option B (richer): give result-override effects a priority and add a
        #     "preserve original" (veto) effect. On an over-budget finish,
        #     BudgetPool emits a higher-priority veto that keeps the
        #     original ``return``/``raise``, overriding this hook's refusal. (Today
        #     composition is only Raise > Error > original, with no cross-hook
        #     veto.)
        remaining = self._refusals_remaining.get(event.invoke_id, self.refusals)
        # TODO(#469): Restore last-iteration awareness (don't refuse a finish on the
        # last allowed iteration). The dropped guard was `event.iteration >= event.max_iterations`.
        # Restore it via IterationLimit publishing its state through a shared accessor
        # (Option A in the comment block above) once that mechanism exists.
        if remaining <= 0:
            return []

        # The refusal is applied as an exit-time ModifyResult carrying a Continue outside the
        # REPL: the resolve_modify_results fold turns the original Return/Raise into a
        # recoverable Continue carrying that refusal. Since #928 the carried Continue also
        # carries the refusal *text* in ``output`` (see below) rather than the empty output it
        # used to have — the fold prepends a terminal original's own output, which since #903 is
        # nothing, so ``output`` here is the only thing the agent would see.
        # The core agent loop is the single writer of __repl_history__ and records each entry
        # from the FINAL (post-override) result (agent.do_one_repl_iteration), so the refused
        # ``return``/``raise`` is recorded as the refusal error, not a clean success (#448).
        result = event.exec_result
        # ``output`` carries the agent-facing text (the ``Continue`` contract, #875-A), rendered
        # with ``summarize_exception`` so it reads ``"BudgetForcingError: <refusal>"``. The type
        # prefix is deliberate even though the message alone is already a complete instruction:
        # since #928 ``output`` is the *only* surface the agent sees, and the prefix is what tells
        # it this is a framework refusal rather than an error its own code produced. (No traceback:
        # the exception is constructed here and never raised, so ``__traceback__`` is ``None``. See
        # ``ReturnType.on_repl_exec_exit`` for the full str/repr/traceback rationale, applied
        # identically at every downgrade site.) Deriving ``output`` from the exception also keeps
        # the two from drifting apart.
        if isinstance(result, Return):
            self._refusals_remaining[event.invoke_id] = remaining - 1
            return [_refuse(self.BUDGET_FORCING_RETURN_MESSAGE)]
        if isinstance(result, Raise):
            self._refusals_remaining[event.invoke_id] = remaining - 1
            return [_refuse(self.BUDGET_FORCING_RAISE_MESSAGE)]

        return []


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``).
BudgetForcingHook = BudgetForcing
