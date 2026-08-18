"""Iteration-limit hook: per-level hard-stop on the REPL iteration cap.

This is a **level** limit (each invoke's own loop), the counterpart to the **pool**
budgets owned by :class:`~BudgetPool`. Activate it via a
propagating channel — ``with IterationLimit(max_iterations=50): jaz.invoke(...)`` — to cap
an invoke and its nested invokes. It carries the limit *values* as constructor params:

- ``max_iterations`` — the **hard** cap, a *count*: iterations are 0-based, so indices
  ``0 .. max_iterations - 1`` run and the hook aborts the invoke on reaching
  ``max_iterations``. The agent loop itself is unbounded — **this hook owns termination**.
- ``warn_remaining`` — optional. When set, as the agent nears the cap (``max_iterations -
  iteration <= warn_remaining``, i.e. this many turns or fewer left, counting the current one)
  the caller's ``warning_text`` is emitted VERBATIM — the hook adds no framing of its own. This
  is an ABSOLUTE turn count, not a fraction of the cap. There is no buffer / overshoot region, no
  result-overriding ``ModifyExecResult``, and no separate last-turn notice; with no
  ``warn_remaining`` (⟺ no ``warning_text``, by the coupling) the hard stop is the only behaviour.

Surfaced through the effect system: an ``Abort`` at ``LLMQueryEnter`` (the hard stop —
``span_llm_query`` raises its carried exception *before* the turn's LLM query, closing the
span ``Aborted``) and a
transient ``AddMessages`` at the same ``on_llm_query_enter`` (the approaching-limit warning).
No ``ModifyExecResult``.

The hard stop deliberately lives on the always-present ``LLMQueryEnter``, not the REPL
*execution* events: execution enter/exit only fire when the LLM response parses to runnable
code, so enforcement hung there silently never fires for perpetually-unparseable output (an
unbounded loop under a ``while True`` core). ``LLMQueryEnter`` fires unconditionally once
per turn — it is the always-present per-turn boundary that ``Abort`` targets.

Because the hard stop is hook-emitted (not a loop bound), a stricter
``IterationLimit`` stacked in the contextvar user tier tightens *both* the warning
and the hard cap (aborts union — the stricter fires earlier, can't be loosened).
Termination depends on this hook running, so the dispatcher fails *loud* on a hook
exception (no silent swallow). There is deliberately no way to disable an active hook —
there is no generic ``disable_hook`` or ``clear_all_hooks`` escape hatch — so a hook that
owns termination can never be silently stripped for a scope.

This hook was split out of the former ``GovernanceHook``; the context-window
concern now lives in :class:`~ContextWindowWarning`.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

from jaz.exceptions import IterationLimitExhaustedError
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Abort, AddMessages, Effect
from jaz.hooks.events import LLMQueryEnter


@dataclass(eq=False)
class IterationLimit(Hook):
    """Hard-stop the agent at its per-level REPL iteration limit.

    Stops the agent with :class:`~jaz.exceptions.IterationLimitExhaustedError` if it does not finish
    within ``max_iterations`` iterations. Iterations are numbered from 0, so the agent runs
    indices ``0`` through ``max_iterations - 1`` and is aborted on reaching ``max_iterations``.

    If ``warn_remaining`` is set, once that many iterations or fewer are left the caller's
    ``warning_text`` is emitted verbatim to nudge the agent to finish. There is no separate
    last-turn notice: with ``warn_remaining`` unset (⟺ ``warning_text`` unset), the hard stop is
    the only behaviour.

    Under ``with`` the cap applies to every invoke in scope, nested ones included, each
    counting its own iterations against ``max_iterations``. Passed positionally to one
    ``jaz.invoke``, it caps only that invoke and not the invokes it nests.

    Args:
        max_iterations: Max number of iterations. Must be positive (>= 1).
        warning_text: The warning text, used verbatim — the only text this hook emits (an empty
            string emits nothing). ``None`` (default) means no warning text. Must be paired with
            ``warn_remaining``: both set, or both ``None``.
        warn_remaining: If set, warn the agent to finish once this many iterations or fewer
            remain (counting the current one). An absolute turn count, not a fraction of the cap.
            ``None`` (default) emits no warning at all. Must be paired with ``warning_text``
            (both set, or both ``None``).

    Raises:
        ValueError: If ``max_iterations`` is not positive; if exactly one of ``warning_text`` /
            ``warn_remaining`` is set (they are coupled); or if ``warn_remaining`` is not positive.
    """

    # Mechanics: handles ``LLMQueryEnter``; emits ``Abort`` once ``iteration >=
    # max_iterations``, at the top of that turn before its LLM query (the agent loop itself is
    # unbounded, so this hook owns termination), and ``AddMessages`` for the warning. Iterations
    # are 0-based and ``max_iterations`` is a count (#719), so index ``max_iterations - 1`` is a
    # full, normal turn the agent can finish on — the *last allowed* one — and index
    # ``max_iterations`` is the first refused.
    #
    # The limit is enforced *silently*: the agent is told neither its iteration count nor the cap,
    # only the approaching-limit warning (which, like the budget force-finish nudge, carries no
    # numbers). This used to be a ``show_status: bool = False`` opt-in emitting
    # ``"REPL iteration N of max M allowed."`` each turn, mirroring ``BudgetPool.show_status``.
    # Removed as dead surface (#719): nothing in ``src/`` or ``evals/`` ever set it — neither
    # construction site passes it (the eval harness's governance bundle, and the console's
    # ``IterationLimit(max_iterations=3)``), and the harness threads only ``warning_text``
    # through, so no eval YAML could reach it either. Only tests turned it on. ``BudgetPool``
    # was spared here, on the grounds that its cost/calls line does not carry the
    # count-vs-index question that motivated this removal — but that is a point about bug
    # risk, not usage, and it was equally dead; its ``show_status`` went the same way right
    # after. No builtin hook shows a per-turn status line now.
    #
    # This removal was accepted over a ``Hook.from_dict`` objection — that a persisted record
    # carrying ``"show_status": false`` would no longer reconstruct. That objection is void: hook
    # serialization was removed outright, so what a written trace holds is this hook's
    # ``__repr__``, and a field leaving the dataclass simply leaves the rendered string.
    #
    # ``warning_text`` is a plain ``str | None``, not a ``str | Callable | None``. The callable
    # ``(can_delegate: bool) -> str`` form was dropped (executive call): the only in-tree caller
    # of it was the eval harness passing a ``TemplatedMustExitWarning`` for a delegate-aware
    # template, and that now renders to a fixed string at config-build time (eval-side, in
    # ``evals/must_exit_template.py``). ``None`` means *no warning text* — core has no stock
    # fallback and no ``can_delegate`` at all; the eval harness supplies its own default string
    # (``STOCK_MUST_EXIT_WARNING``) when a config wires no custom text.

    # ``max_iterations`` is positional-or-keyword; everything after ``KW_ONLY`` is keyword-only
    # (so ``IterationLimit(10)`` works but the tunables must be named). A ``@dataclass(eq=False)``
    # so the generated ``__repr__`` lists the fields — what an observability consumer records; a
    # ``str | None`` warning_text renders as itself.

    max_iterations: int
    _: KW_ONLY
    warning_text: str | None = None
    warn_remaining: int | None = None

    def __post_init__(self) -> None:
        # max_iterations is a COUNT of allowed turns (indices 0 .. max_iterations - 1), so it must
        # be >= 1 — a value of 0 (or negative) aborts on iteration 0 before any turn runs.
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {self.max_iterations!r}")
        # ``warning_text`` and ``warn_remaining`` are coupled: both set, or both ``None``.
        # ``warning_text=None`` means "no warning text", so there is nothing to emit and a
        # ``warn_remaining`` threshold would have no message; conversely a ``warn_remaining``
        # threshold is meaningless without text to warn with. Both ``None`` (the default) is the
        # "no approaching warning at all" state — the hard stop and the last-turn NOTE still fire.
        if self.warning_text is None and self.warn_remaining is not None:
            raise ValueError(
                "warn_remaining must be None when warning_text is None "
                f"(no warning to emit), got warn_remaining={self.warn_remaining!r}"
            )
        if self.warning_text is not None and self.warn_remaining is None:
            raise ValueError(
                "warn_remaining must be set when warning_text is given "
                "(it says when to start warning)"
            )
        # warn_remaining must be strictly > 0. A value of 0 is useless — the approaching warning
        # fires when remaining <= warn_remaining, but remaining is >= 1 on every allowed turn
        # (index max-1 leaves 1), so a 0 threshold never fires. No upper bound — warn_remaining >=
        # max_iterations simply warns from iteration 0.
        if self.warn_remaining is not None and self.warn_remaining <= 0:
            raise ValueError(f"warn_remaining must be > 0, got {self.warn_remaining!r}")

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Hard stop: once iterations exceed max_iterations, abort the invoke at the top of
        # the turn — before the LLM query, so no query is wasted on a turn that would be
        # discarded. LLMQueryEnter is the always-present per-turn boundary, so the hard stop
        # lives here as an Abort (#481); span_llm_query raises the carried
        # IterationLimitExhaustedError out of jaz.invoke() (the invoke's spans close Aborted).
        # This is the loop's termination backstop — owned by
        # the hook (the agent loop is unbounded), so a stacked stricter IterationLimit
        # tightens the hard cap too (aborts union; the stricter one fires earlier). It lives
        # here, not the REPL *execution* events, because those are skipped when the LLM
        # response fails to parse, so a hard stop there would never fire for
        # perpetually-unparseable output. On the abort turn we emit only the Abort (no
        # warning add — the query is discarded anyway).
        #
        # ``>=``, not ``>``: ``iteration`` is a 0-based position and ``max_iterations`` is a
        # COUNT (#719), so the allowed turns are indices ``0 .. max_iterations - 1`` and index
        # ``max_iterations`` is the first refused one. Exactly ``max_iterations`` turns run,
        # unchanged from when this read ``> max_iterations`` against a 1-based counter.
        if event.iteration >= self.max_iterations:
            return [
                Abort(
                    error=IterationLimitExhaustedError(
                        f"No return value after {self.max_iterations} REPL iterations"
                    )
                )
            ]

        # Approaching-limit warning: emit the caller's ``warning_text`` VERBATIM, as a transient
        # AddMessages, once the agent is within ``warn_remaining`` turns of the cap. There is NO
        # hook-authored framing — no "you are approaching…" line, no last-turn NOTE. The message is
        # exactly what the caller supplied and nothing else (executive call: hooks prescribe no
        # agent-facing prose). So a hook with no ``warning_text`` (⟺ no ``warn_remaining``, by the
        # __post_init__ coupling) emits nothing at all — the hard stop above is then the only
        # behaviour. An empty ``warning_text`` likewise emits nothing.
        #
        # ``warn_remaining`` is an ABSOLUTE turn count, not a fraction of the cap: wrap-up cost is
        # roughly constant, so "do I have enough turns left to finish" is answered identically for a
        # cap of 10 and of 1000 (``remaining == max_iterations - iteration``, turns left including
        # this one). Two stacked IterationLimits each emit their own AddMessages; both are ordered
        # and visible (the fold's within-slot sort).
        remaining = self.max_iterations - event.iteration
        if (
            self.warn_remaining is not None
            and remaining <= self.warn_remaining
            and self.warning_text
        ):
            return [AddMessages([{"role": "user", "content": self.warning_text}])]
        return []


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_legacy_hook_names_still_importable``).
IterationLimitHook = IterationLimit
