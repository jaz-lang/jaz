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
- ``min_depth`` / ``max_depth`` — optional, both ``None`` by default. They restrict WHICH invokes
  in the propagating scope the cap applies to, by absolute recursion depth (1-based: the top-level
  invoke is depth 1). ``IterationLimit(50, min_depth=2)`` caps every delegated invoke at 50 turns
  and leaves the driver uncapped. They select invokes; they do not change what the cap means for a
  selected one, which still counts its own iterations against ``max_iterations``.

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
A depth window narrows *which invokes* a stacked hook governs, never how strictly: outside its
window a hook contributes nothing, so stacking can only add aborts and the union above still holds.
Termination depends on this hook running, and the dispatcher logs and swallows a hook
exception rather than propagating it (making that configurable is #612), so a bug in this
hook lifts the cap without stopping the run; the hook's own tests are the guard. There is
deliberately no way to disable an active hook — there is no generic ``disable_hook`` or
``clear_all_hooks`` escape hatch — so a hook that
owns termination can never be silently stripped for a scope.

This hook was split out of the former ``GovernanceHook``; the context-window
concern now lives in :class:`~ContextWindowWarning`.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

from jaz.exceptions import HookActivationError, IterationLimitExhaustedError
from jaz.hooks._dispatcher import Hook
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

    Under ``with`` the cap applies to every invoke in scope, nested ones included (or to those
    inside ``min_depth``/``max_depth``, when set), each counting its own iterations against
    ``max_iterations``. Passed positionally to one ``jaz.invoke``, it caps only that invoke and not
    the invokes it nests.

    Args:
        max_iterations: Max number of iterations. Must be positive (>= 1).
        warning_text: The warning text, used verbatim — the only text this hook emits (an empty
            string emits nothing). ``None`` (default) means no warning text. Must be paired with
            ``warn_remaining``: both set, or both ``None``.
        warn_remaining: If set, warn the agent to finish once this many iterations or fewer
            remain (counting the current one). An absolute turn count, not a fraction of the cap.
            ``None`` (default) emits no warning at all. Must be paired with ``warning_text``
            (both set, or both ``None``).
        min_depth: If set, apply nothing — neither the cap nor the warning — to invokes
            shallower than this. Absolute recursion depth, 1-based (a top-level invoke is depth 1),
            so ``min_depth=2`` means "delegated invokes only". ``None`` (default) sets no
            floor. Subject to the depth attribution gap :class:`~RecursionLimit` documents — see
            the note below the field definitions.
        max_depth: If set, apply nothing to invokes DEEPER than this, so ``max_depth=1`` means
            "the top-level invoke only". ``None`` (default) sets no ceiling.

    Raises:
        ValueError: If ``max_iterations`` is not positive; if exactly one of ``warning_text`` /
            ``warn_remaining`` is set (they are coupled); if ``warn_remaining`` is not positive; if
            ``min_depth`` or ``max_depth`` is less than 1; or if ``min_depth > max_depth``.
        HookActivationError: On ``setup()``, if a depth window is set and the hook is installed
            positionally on ``invoke`` rather than activated with ``with``.
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
    # Removed as dead surface (#719): no construction site ever set it — the console's
    # ``IterationLimit(max_iterations=3)`` does not, and no harness threaded it through from
    # config, only ``warning_text``. Only tests turned it on. ``BudgetPool``
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
    # ``(can_delegate: bool) -> str`` form was dropped (executive call): its only caller was a
    # harness passing a delegate-aware template, and such text renders to a fixed string at
    # config-build time, host-side. ``None`` means *no warning text* — core has no stock fallback
    # and no ``can_delegate`` at all.

    # ``max_iterations`` is positional-or-keyword; everything after ``KW_ONLY`` is keyword-only
    # (so ``IterationLimit(10)`` works but the tunables must be named). A ``@dataclass(eq=False)``
    # so the generated ``__repr__`` lists the fields — what an observability consumer records; a
    # ``str | None`` warning_text renders as itself.

    max_iterations: int
    _: KW_ONLY
    warning_text: str | None = None
    warn_remaining: int | None = None
    # A depth WINDOW rather than a single ``depth`` value, and ``None`` on each side rather than a
    # dedicated sentinel object: ``None`` is already this hook's spelling for "not configured"
    # (``warning_text``/``warn_remaining`` use it), and a half-open window is what the real cases
    # want — "subagents only" is ``min_depth=2`` with no ceiling, and "the driver only" is
    # ``max_depth=1`` with no floor. A single ``depth=`` would express neither.
    #
    # ``min_depth``/``max_depth`` and not the longer ``applies_from_depth``/``applies_to_depth``,
    # which these briefly were (executive call, reversing that rename). The objection to the short
    # names is real and was tried: a ``RecursionLimit`` sits immediately above this hook in the
    # configs the feature exists for, so two ``max_depth: 2`` with different meanings end up adjacent
    # in one hook list, and side by side in a trace's ``hooks`` field now that the window is in the
    # ``__repr__``. It was rejected anyway, because the collision is resolved by context — a reader
    # meeting ``max_depth`` ON AN ITERATION LIMIT reads it against that hook, and nobody guesses what
    # ``applies_to_depth`` means without being told. Clarity at the point of use beat global
    # uniqueness of the identifier.
    min_depth: int | None = None
    max_depth: int | None = None

    # True only for the dynamic extent of ``__enter__``, so the ``setup()`` it triggers can tell a
    # propagating install from a ``local_hooks`` one. Same mechanism, and the same TODO(#808), as
    # ``RecursionLimit``. ``init=False`` so it is lifecycle bookkeeping rather than a construction
    # param, and ``repr=False`` so it stays out of the recorded string — ``init=False`` alone would
    # not have, since the generated ``__repr__`` includes non-init fields.
    _entering_via_context: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        # max_iterations is a COUNT of allowed turns (indices 0 .. max_iterations - 1), so it must
        # be >= 1 — a value of 0 (or negative) aborts on iteration 0 before any turn runs.
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {self.max_iterations!r}")
        # ``warning_text`` and ``warn_remaining`` are coupled: both set, or both ``None``.
        # ``warning_text=None`` means "no warning text", so there is nothing to emit and a
        # ``warn_remaining`` threshold would have no message; conversely a ``warn_remaining``
        # threshold is meaningless without text to warn with. Both ``None`` (the default) is the
        # "no approaching warning at all" state — the hard stop still fires.
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
        # Depth is 1-based and there is always a root, so a bound below 1 is incoherent in the same
        # way ``RecursionLimit(max_depth=0)`` is. Unlike ``warning_text``/``warn_remaining`` the two
        # are NOT coupled — each side is independently optional, which is the point of a window.
        for name, value in (
            ("min_depth", self.min_depth),
            ("max_depth", self.max_depth),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1, got {value!r}")
        # An inverted window selects nothing, so the hook would silently do nothing at all — the
        # worst outcome for a hook that owns termination. Reject it at construction.
        if (
            self.min_depth is not None
            and self.max_depth is not None
            and self.min_depth > self.max_depth
        ):
            raise ValueError(
                f"min_depth must be <= max_depth, got min_depth={self.min_depth!r} "
                f"max_depth={self.max_depth!r} (an empty window would cap nothing)"
            )
        # NOT checked, and cannot be: a window the tree never reaches -- `min_depth=5` under a
        # `RecursionLimit(max_depth=2)`, or under an agent that simply never delegates -- has the same
        # silent no-op outcome as the inverted window above, and is refused only in the case this
        # object can see by itself. The reachable depth is a property of the whole hook stack and of
        # the agent's behaviour, neither of which is available at construction, and inferring it at
        # the first event would move the failure from "the config is wrong" to "the run is over".

    def __enter__(self):
        # Mark the `with` path so the setup() the base __enter__ calls next is recognized as a
        # propagating install. Cleared in `finally` so a failed __enter__ (the base's
        # double-activation guard raising before setup()) never leaves the flag set, which would let
        # a later local_hooks activation slip through unrejected.
        self._entering_via_context = True
        try:
            return super().__enter__()
        finally:
            self._entering_via_context = False

    def setup(self) -> None:
        # Reject a depth window on the local_hooks channel. setup() runs on exactly two paths --
        # `with` (via __enter__, which set the flag above) and local_hooks (called directly, flag
        # unset). A local hook is dispatched to exactly ONE invoke and does not propagate, so a
        # depth window there is either a no-op (the invoke is inside it) or silently disables the
        # hook entirely (it is not) -- and the caller already knows that invoke's depth, so the
        # window can express nothing they could not express by not installing the hook. Fail loud
        # rather than let a cap quietly not apply. The plain form stays legal on both channels:
        # `jaz.invoke(IterationLimit(50), ...)` is how a caller caps one invoke, and nothing here
        # changes that. TODO(#808): this infers the install channel from the __enter__/setup
        # routing; an explicit signal to setup() would be better than two hooks inferring it.
        if not self._entering_via_context and (
            self.min_depth is not None or self.max_depth is not None
        ):
            raise HookActivationError(
                "IterationLimit with min_depth/max_depth cannot be passed to a single invoke: "
                "a hook passed that way reaches exactly one invoke and does not propagate, so a depth window "
                "there cannot select anything the caller did not already choose. Install it with "
                "`with IterationLimit(...):`, or drop the depth window."
            )
        super().setup()

    def _applies_at(self, depth: int) -> bool:
        """Whether this hook governs an invoke at `depth` (1-based)."""
        # Inherits RecursionLimit's "Known gap" (#1033): an invoke reached through a passed-down
        # wrapper whose `jaz.invoke` runs on a thread that did not inherit the caller's context is
        # counted at the BINDING invoke's depth, not the caller's. There it lets a tree go one level
        # too deep; here it can present a delegated invoke as depth 1, so an `min_depth=2`
        # cap silently does not apply to it — the termination backstop quietly absent, which is the
        # outcome `__post_init__` and `setup()` both refuse configurations to avoid. Narrow (it needs
        # `allowed_imports` to permit threads), but "absolute recursion depth" reads as exact above
        # and is not, so do not build a safety argument on this window alone.
        if self.min_depth is not None and depth < self.min_depth:
            return False
        return not (self.max_depth is not None and depth > self.max_depth)

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Depth window first, so an out-of-band invoke gets NEITHER the hard stop nor the warning.
        # Gating only the abort would leave the driver being told to wrap up against a cap that
        # will never fire on it -- agent-facing text about a limit that does not apply.
        if not self._applies_at(event.depth):
            return []

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
            # The depth is in the message because it is the only place a consumer can read it. Both a
            # tree-wide cap and a windowed one raise the same exception type, and a *capped subagent*
            # and a *capped driver* mean opposite things to anyone reading a run: the first is a
            # routine per-task outcome, the second ends the run with every unreached task unearned.
            # Without the depth those are one bucket, and a downstream analysis splitting them has to
            # split on hook identity, which stops existing the moment one hook does both jobs.
            return [
                Abort(
                    error=IterationLimitExhaustedError(
                        f"No return value after {self.max_iterations} REPL iterations "
                        f"(depth {event.depth})"
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
