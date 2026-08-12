"""Iteration-limit hook: per-level hard-stop on the REPL iteration cap.

This is a **level** limit (each invoke's own loop), the counterpart to the **pool**
budgets owned by :class:`~jaz.hooks.builtin.budget_pool.BudgetPool`. Activate it via a
propagating channel — ``with IterationLimit(max_iterations=50): jaz.invoke(...)`` — to cap
an invoke and its nested invokes. It carries the limit *values* as constructor params:

- ``max_iterations`` — the **hard** cap: once an iteration exceeds ``max_iterations``
  the hook aborts the invoke. The agent loop itself is unbounded (``while True``) —
  **this hook owns termination**.
- ``warn_fraction`` — as the agent approaches the cap (``warn_fraction *
  max_iterations``) it gets a warning prompt nudging it to finish first. There is no
  buffer / overshoot region and no result-overriding ``ModifyResult``.

Surfaced through the effect system: an ``Abort`` at ``LLMQueryEnter`` (the hard stop —
``span_llm_query`` turns it into a terminal ``Raise`` *before* the turn's LLM query) and a
transient ``AddMessages`` at the same ``on_llm_query_enter`` (opt-in per-turn status — off
by default, see ``show_status`` — plus the approaching-limit warning). No ``ModifyResult``.

The hard stop deliberately lives on the always-present ``LLMQueryEnter``, not the REPL
*execution* events: execution enter/exit only fire when the LLM response parses to runnable
code, so enforcement hung there silently never fires for perpetually-unparseable output (an
unbounded loop under a ``while True`` core). ``LLMQueryEnter`` fires unconditionally once
per turn — it is the always-present per-turn boundary that ``Abort`` targets.

Because the hard stop is hook-emitted (not a loop bound), a stricter
``IterationLimit`` stacked in the contextvar user tier tightens *both* the warning
and the hard cap (Aborts union — the stricter fires earlier, can't be loosened).
Termination depends on this hook running, so the dispatcher fails *loud* on a hook
exception (no silent swallow). There is deliberately no way to disable an active hook —
there is no generic ``disable_hook`` or ``clear_all_hooks`` escape hatch — so a hook that
owns termination can never be silently stripped for a scope.

This hook was split out of the former ``GovernanceHook``; the context-window
concern now lives in :class:`~jaz.hooks.builtin.context_window.ContextWindow`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from jaz.exceptions import BudgetExhaustedError
from jaz.hooks.builtin._must_exit import (
    deserialize_must_exit_warning,
    resolve_must_exit_warning,
    serialize_must_exit_warning,
)
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Abort, AddMessages, Effect
from jaz.hooks.events import LLMQueryEnter


@dataclass(eq=False, kw_only=True)
class IterationLimit(Hook):
    """Hard-stop the agent at its per-level REPL iteration limit.

    Aborts the agent with :class:`~jaz.exceptions.BudgetExhaustedError` if it does not finish
    within ``max_iterations`` iterations. From ``warn_fraction`` of the cap onward it is warned
    to finish, and on iteration ``max_iterations`` it is told that this is its last one.

    Under ``with`` the cap applies to every invoke in scope, nested ones included, each
    counting its own iterations against ``max_iterations``. Passed positionally to one
    ``jaz.invoke``, it caps only that invoke and not the invokes it nests.

    Args:
        max_iterations: Max number of iterations.
        warn_fraction: Fraction of ``max_iterations`` at which the agent starts being warned
            to finish. Default 0.9.
        must_exit_warning: Text emitted with the warning. ``None`` (default) renders the stock
            wording, which tells the agent to finish because it is nearing its iteration
            limit. Otherwise a plain string, or a callable ``(can_delegate: bool) -> str``
            called when the warning is emitted. Rendering to an empty string emits nothing.
        show_status: Show the agent a ``REPL iteration N of max M allowed.`` line each turn.
            Default False.

    Raises:
        ValueError: If ``warn_fraction`` is outside ``[0, 1]``.
    """

    # Mechanics: handles ``LLMQueryEnter``; emits ``Abort`` once ``iteration >
    # max_iterations``, at the top of that turn before its LLM query (the agent loop itself is
    # unbounded, so this hook owns termination), and ``AddMessages`` for the warning and the
    # optional status line. Note the strict ``>``: iteration ``max_iterations`` is a full,
    # normal turn the agent can finish on — it is the *last allowed* iteration, not the first
    # refused one.
    #
    # ``show_status`` default False means the limit is enforced *silently* — the agent is told
    # neither its iteration count nor the cap, only the approaching-limit warning (which, like
    # the budget force-finish nudge, carries no numbers). Mirrors ``BudgetPool.show_status``.
    #
    # Cut from the ``must_exit_warning`` description, which is agent-facing text a caller
    # configures, not a design they have to reason about: the stock warning is deliberately
    # finish-method-agnostic (it says to finish, not *how*, because the finish mechanism is the
    # REPL's own concern); the callable takes a bare ``can_delegate`` because the hook owns
    # *when* the warning appears and the callable owns *what* it says — see
    # ``_must_exit.resolve_must_exit_warning`` for the full contract and
    # ``TemplatedMustExitWarning`` for the serializable form.

    # A ``@dataclass(eq=False)`` — ``to_dict``/``from_dict`` derive the params from the fields.
    # ``must_exit_warning`` (a callable/str/None the generic path can't JSON-encode) carries a
    # ``field(metadata=...)`` (de)serializer: a plain string or ``TemplatedMustExitWarning``
    # round-trips; ``None`` or an arbitrary callable serializes to omitted → default warning.

    max_iterations: int
    warn_fraction: float = 0.9
    # must_exit_warning isn't JSON-safe (callable/str/None) — the field metadata carries the
    # shared (de)serializer, so the generic to_dict/from_dict round-trips a str or
    # TemplatedMustExitWarning and omits (→ default) a None/arbitrary-callable.
    must_exit_warning: str | Callable[[bool], str] | None = field(
        default=None,
        metadata={
            "to_dict": serialize_must_exit_warning,
            "from_dict": deserialize_must_exit_warning,
        },
    )
    show_status: bool = False

    def __post_init__(self) -> None:
        # Guard the public param: out-of-range values degrade silently otherwise —
        # >1 disables the warning entirely (the hard stop at max_iterations fires
        # first), <=0 warns from the very first turn (iteration 0 >= 0 * max).
        if not 0 <= self.warn_fraction <= 1:
            raise ValueError(
                f"warn_fraction must be in [0, 1], got {self.warn_fraction!r}"
            )

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Hard stop: once iterations exceed max_iterations, abort the invoke at the top of
        # the turn — before the LLM query, so no query is wasted on a turn that would be
        # discarded. LLMQueryEnter is the always-present per-turn boundary, so the hard stop
        # lives here as an Abort (#481); span_llm_query turns it into a terminal Raise that
        # propagates out of jaz.invoke(). This is the loop's termination backstop — owned by
        # the hook (the agent loop is unbounded), so a stacked stricter IterationLimit
        # tightens the hard cap too (Aborts union; the stricter one fires earlier). It lives
        # here, not the REPL *execution* events, because those are skipped when the LLM
        # response fails to parse, so a hard stop there would never fire for
        # perpetually-unparseable output. On the abort turn we emit only the Abort (no
        # warning add — the query is discarded anyway).
        if event.iteration > self.max_iterations:
            return [
                Abort(
                    error=BudgetExhaustedError(
                        f"No return value after {self.max_iterations} REPL iterations"
                    )
                )
            ]

        # Per-turn status line (always) + approaching-limit warning (from warn_fraction),
        # emitted as a transient AddMessages folded into this query. Two stacked
        # IterationLimitHooks each emit their own AddMessages; both are ordered and visible
        # (the fold's within-slot sort).

        # Per-turn status line ("iteration N of max M") — off by default
        # (show_status), so the limit is enforced without telling the agent
        # its position or the cap (mirrors BudgetPool.show_status). Opt in for an agent
        # that should pace itself against a visible iteration budget.
        texts: list[str | None] = []
        if self.show_status:
            texts.append(
                f"REPL iteration {event.iteration} of max {self.max_iterations} allowed."
            )

        # Approaching-limit warning: from warn_fraction of the cap, nudge the agent to
        # finish before the hard stop. The final allowed turn keeps the stronger phrasing.
        if event.iteration >= self.warn_fraction * self.max_iterations:
            is_last = event.iteration >= self.max_iterations
            texts.append(
                "NOTE: This is your last allowed REPL interaction."
                if is_last
                else "WARNING: You are approaching the REPL iteration limit. "
                "Finish soon."
            )
            texts.append(
                resolve_must_exit_warning(
                    self.must_exit_warning,
                    can_delegate=event.can_recurse,
                )
            )

        # One trailing user message for this hook (its fragments joined, blanks dropped),
        # appended to the query. Different hooks' messages are not coalesced — see AddMessages.
        content = "\n".join(t for t in texts if t)
        return [AddMessages([{"role": "user", "content": content}])]


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``).
IterationLimitHook = IterationLimit
