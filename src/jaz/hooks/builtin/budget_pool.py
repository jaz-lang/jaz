"""User-facing hook for LLM cost / call-count budget enforcement.

This hook **is the budget pool**: one shared allowance per hook scope. Apply it as a
context manager around an invoke::

    from jaz import invoke
    from jaz.hooks import BudgetPool

    with BudgetPool(cost_budget=1.0):
        result = invoke(task=task)

A single instance propagates (via contextvars) to all invokes in scope — but ONLY on the
``with`` path. Passed positionally (``jaz.invoke(BudgetPool(...), ...)``) it is bound to that
one invoke's dispatcher and never reaches invokes nested under it, so their cost is neither
counted nor enforced. Measured on the same root+child workload: 3 calls / $0.04 under ``with``,
2 calls / $0.02 positionally.

The hook owns the one budget and a **running scalar aggregate** — a total cost and a total
call count over every LLM call in scope. It is a *pure enforcer*: it keeps no per-invoke
records and writes no report. Per-invoke / per-tree cost accounting (with the full nesting
tree) lives in the ATIF trace (``ATIFTrace``), which records the same cost off the same
``LLMQueryExit`` event; anything that needs a cost breakdown reads that trace rather than this
hook. All invokes in scope draw down the ONE shared aggregate.

Enforcement is global (the one budget vs the aggregate), so *which* invoke triggered a
turn is irrelevant — any invoke continuing past the aggregate budget is stopped:

- **Hard-stop** (budget reached) → an ``Abort`` at ``LLMQueryEnter`` (before the turn's
  LLM query) once the aggregate cost/calls reach the budget. Reaching the budget IS the
  stop — there is no buffer. It fires before the *first* turn's query too, so an invoke
  born into an already-exhausted pool stops before any LLM call. It lives on the
  always-present ``LLMQueryEnter`` (fires unconditionally once per turn), not the REPL
  *execution* events (skipped when the LLM response fails to parse, which would
  otherwise pay for queries forever on perpetually-unparseable output).
- **Warning** (approaching the budget) → a transient ``AddMessages`` (at
  ``on_llm_query_enter``) once the remaining budget falls to an opt-in ``warn_*_remaining``
  headroom (absolute dollars / calls left, not a fraction). The message is the caller's
  ``warning_text`` VERBATIM — no hook-authored "$X remaining" line, no framing. Purely additive
  prompt text — there is **no** result-overriding ``ModifyExecResult``.
"""

from __future__ import annotations

import threading

from jaz.exceptions import BudgetPoolExhaustedError, ModelPricingUnavailableError
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Abort, AddMessages, Effect
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
)
from jaz.hooks.events.base import Completed


class BudgetPool(Hook):
    """Enforce one shared LLM cost/call budget over everything in the hook's scope.

    Every invoke the hook is active for draws on the *same* budget. Reaching ``cost_budget``
    or ``calls_budget`` is a hard stop: the invoke ends by raising
    :class:`~jaz.exceptions.BudgetPoolExhaustedError`. Before that, if a ``warn_*_remaining``
    headroom is set, the agent is warned to finish once the remaining budget falls to it.

    The scope covered by the budget pool depends on how it is activated:

    * ``with BudgetPool(...):`` — every ``jaz.invoke()`` in the block *and every invoke
      nested inside those* shares the one budget. Use this to cap a whole workflow.
    * ``jaz.invoke(BudgetPool(...), ...)`` — the pool covers **only that invoke's own LLM
      calls, excluding its sub-invokes.**

    The budget starts empty when the scope opens.

    This hook enforces only; it produces no cost report. For a per-invoke cost/token breakdown
    (including the full sub-invoke tree), read the ATIF trace written by ``ATIFTrace`` — it
    records the same cost off the same event.

    Note that if a single turn jumps from above the warning headroom straight past the budget,
    the agent is stopped with no warning at all. Set ``warn_cost_remaining`` /
    ``warn_calls_remaining`` appreciably larger than a typical turn's cost / a single call so
    the warning turn is not skipped.

    **Missing token usage degrades; it does not crash.** A provider may return a response
    with no usage block; this hook does not track tokens, so a missing usage block is a
    non-event for it. A missing *cost* (``cost_usd is None``) under a ``cost_budget`` is a
    different matter — the budget is then unenforceable and the hook aborts (see
    ``on_llm_query_exit``); with no ``cost_budget`` the call is booked as ``$0`` and the
    ``calls_budget`` keeps counting it.

    Args:
        cost_budget: Hard ceiling on total LLM cost in USD. ``None`` (default) means no cost
            limit; otherwise must be positive.
        calls_budget: Hard ceiling on total LLM calls. ``None`` (default) means no call limit;
            otherwise must be positive.
        warn_cost_remaining: If set, warn the agent to finish once the remaining cost budget
            (``cost_budget`` minus spend) falls to this many USD or fewer. Absolute headroom,
            not a fraction of the budget. ``None`` (default) emits no approaching cost warning;
            the hard stop at ``cost_budget`` still applies. Requires ``cost_budget`` (the budget it
            measures against) and must be paired with ``warning_text``.
        warn_calls_remaining: Same, in remaining LLM calls against ``calls_budget``. ``None``
            (default) emits no approaching calls warning. Requires ``calls_budget`` and must be
            paired with ``warning_text``.
        warning_text: The warning text, used verbatim — the only text this hook emits (an empty
            string emits nothing). ``None`` (default) means no warning at all. Coupled with the
            thresholds (like ``IterationLimit``): ``warning_text`` and at least one
            ``warn_*_remaining`` are either both set (a warning fires), or all ``None`` (no warning).

    Raises:
        ValueError: If ``cost_budget``, ``calls_budget``, ``warn_cost_remaining``, or
            ``warn_calls_remaining`` is set but not positive; if a ``warn_*_remaining`` is set
            without its matching budget (``cost_budget`` / ``calls_budget``); or if exactly one of
            ``warning_text`` / (a ``warn_*_remaining`` threshold) is set — they are coupled, so it
            is all-``None`` (no warning) or both.
    """

    # This hook is a PURE ENFORCER: it holds the one budget plus a running scalar aggregate
    # (`_total_cost`, `_total_calls`) over the LLM calls in scope, and nothing else. It once
    # kept a *forest* of per-invoke `CostTracker` nodes (one tree per top-level invoke, nested
    # invokes chained onto their parent) and flushed a JSON cost envelope at teardown via a
    # `json_path` param. That forest was removed (executive call): it duplicated the tree the
    # ATIF trace (`ATIFTrace`) already builds off the same `LLMQueryExit` event — same cost,
    # same nesting — so two structures tracked one fact. Enforcement only ever reads the
    # *aggregate* (one budget vs the sum), which a scalar carries exactly; the per-invoke
    # breakdown a report needs now has one home, the ATIF trace. Eval tooling that used to read
    # this hook's JSON envelope reads `final_metrics.total_cost_usd` from the ATIF trace instead
    # (summed over the root and its `subagent_trajectories`).
    #
    # Why the positional form excludes sub-invokes: it is not a BudgetPool quirk. A
    # positionally-passed hook is bound to that Agent's dispatcher and stays off the
    # contextvar, so nested invokes (which build their own dispatcher) never dispatch to it
    # — see `_activate_local_hooks` in `invoke.py`. The docstring states only the outcome.
    # The consequence, also left out of the docstring: for a *budget* the miss is silent. The
    # sub-invoke's cost is never counted, so the aggregate under-reports rather than erroring —
    # a delegating task can run up cost the pool never sees.
    #
    # `on_invoke_enter` records the invoke_id in `_entered` (the set of in-scope invokes) and
    # raises `RuntimeError` if an invoke reaches the hook without `setup()` having run. Not in
    # the docstring: `with` and the positional argument are the only supported activations and
    # both run `setup()`, so the raise is unreachable from documented usage — it exists to catch
    # a hook wired up through some other path. `_entered` is what distinguishes an in-scope
    # invoke from the non-loop `query()` path (which emits `LLMQueryEnter` with no preceding
    # `InvokeEnter`): accounting and enforcement skip an invoke_id that never entered.
    #
    # Mechanics kept out of the docstring: the hook handles `InvokeEnter`/`InvokeExit` (scope
    # membership), `LLMQueryEnter` (enforcement + warning), and `LLMQueryExit` (accounting:
    # += cost, += 1 call — AND the uncosted-model backstop, an `Abort` on the Completed arm that
    # reads the final post-transform cost). The hard
    # stop is an `Abort` at `LLMQueryEnter` — the top of the turn, before the query — chosen
    # because that event is the always-present per-turn boundary; REPL *execution* is skipped
    # for unparseable output, so enforcing there would let a turn slip past the budget. The
    # warning is an `AddMessages`. There is deliberately no buffer and no `ModifyExecResult`
    # rewrite of results: the only two levers are prompt text and the hard stop.
    #
    # Exhaustion raises rather than granting a final wrap-up turn — an executive call:
    # reaching the limit IS the stop. A "one last turn to RETURN something" affordance would
    # mean the budget can always be exceeded by one turn, which makes it not a ceiling.
    #
    # `warning_text` is a plain `str | None` (the callable form was dropped everywhere — see
    # IterationLimit). It is fully coupled with the warn_*_remaining thresholds, matching
    # IterationLimit: `warning_text=None` means no warning at all, and a threshold with no
    # warning_text (or vice-versa) raises (see the __init__ guards). So the only valid shapes are
    # all-None (no warning) and warning_text + at least one threshold (a warning fires).

    # The approaching-budget warning uses ABSOLUTE remaining headroom (`warn_cost_remaining` in
    # dollars, `warn_calls_remaining` in calls), not a fraction of the budget, and is OPT-IN
    # (both default None → no warning, only the hard stop). Executive call, three reasons:
    #   (1) Headroom is the stable quantity. What the agent needs is "enough left to wrap up",
    #       and wrap-up cost is roughly constant, so the threshold a user sets once should not
    #       move when they resize the budget — a fraction does the opposite (0.9 of $1 is $0.10
    #       of room, 0.9 of $100 is $10.00).
    #   (2) A fraction misbehaves at the small end: the old docstring's own correctness condition
    #       was `(1 - warn_fraction) * cost_budget` >> a typical turn's cost — an absolute-headroom
    #       statement wearing a fraction's clothes. Below that the one-turn jump from
    #       under-threshold to past-budget skipped the warning entirely.
    #   (3) There is no universal default absolute cost (a turn is $0.001 on one model, $1 on
    #       another), so rather than bake in a fraction that only looks universal, the warning is
    #       opt-in — matching LiteLLM's `soft_budget` (an explicit threshold set separately from
    #       the hard `max_budget`) and SWE-agent (which warns the agent not at all and just
    #       autosubmits at the cap). Sibling change: `IterationLimit.warn_remaining` — iterations,
    #       being a unitless count, CAN carry a small default and do (2); dollars cannot.
    def __init__(
        self,
        *,
        cost_budget: float | None = None,
        calls_budget: int | None = None,
        warn_cost_remaining: float | None = None,
        warn_calls_remaining: int | None = None,
        warning_text: str | None = None,
    ) -> None:
        self.cost_budget = cost_budget
        self.calls_budget = calls_budget
        # Budgets must be positive when set: a 0 (or negative) budget hard-stops before the first
        # LLM call (the aggregate 0 already meets it), which is never what a caller means.
        if cost_budget is not None and cost_budget <= 0:
            raise ValueError(f"cost_budget must be > 0, got {cost_budget!r}")
        if calls_budget is not None and calls_budget <= 0:
            raise ValueError(f"calls_budget must be > 0, got {calls_budget!r}")
        # Guard the public params: the headroom must be strictly > 0. A value of 0 (or negative)
        # is useless — the warning fires when remaining <= headroom, but the hard stop fires first
        # at remaining <= 0, so a 0 headroom's warning can never appear. None means "no approaching
        # warning for that budget". There is no upper bound: a headroom >= the budget just warns
        # from the first turn.
        if warn_cost_remaining is not None and warn_cost_remaining <= 0:
            raise ValueError(
                f"warn_cost_remaining must be > 0, got {warn_cost_remaining!r}"
            )
        if warn_calls_remaining is not None and warn_calls_remaining <= 0:
            raise ValueError(
                f"warn_calls_remaining must be > 0, got {warn_calls_remaining!r}"
            )
        # A headroom threshold needs its own budget to measure "remaining" against — without it the
        # warning silently never fires (``_warning_texts`` guards each arm with ``<budget> is not
        # None``). So a cost threshold requires ``cost_budget`` and a calls threshold requires
        # ``calls_budget``; a threshold paired only with the *other* budget is equally dead.
        if warn_cost_remaining is not None and cost_budget is None:
            raise ValueError(
                "warn_cost_remaining needs a cost_budget to measure remaining against"
            )
        if warn_calls_remaining is not None and calls_budget is None:
            raise ValueError(
                "warn_calls_remaining needs a calls_budget to measure remaining against"
            )
        # warning_text and the warn_*_remaining thresholds are fully coupled, like IterationLimit:
        # a warning is emitted iff BOTH a threshold and text are set. So warning_text=None means
        # "no warning emitted" — there is nothing for a threshold to trigger — and a threshold with
        # no text is equally a dead config (the text it would ride on is absent). Both directions
        # raise; the all-None default (no warning) and both-set (warning) are the only valid shapes.
        _has_threshold = (
            warn_cost_remaining is not None or warn_calls_remaining is not None
        )
        if warning_text is None and _has_threshold:
            raise ValueError(
                "warn_cost_remaining / warn_calls_remaining must be None when warning_text is "
                "None (no warning to emit)"
            )
        if warning_text is not None and not _has_threshold:
            raise ValueError(
                "warning_text needs a warn_cost_remaining or warn_calls_remaining threshold "
                "to be emitted (it is only shown with an approaching-budget warning)"
            )
        self.warn_cost_remaining = warn_cost_remaining
        self.warn_calls_remaining = warn_calls_remaining
        self.warning_text = warning_text
        # The running aggregate — the whole of what enforcement reads. Not per-invoke: the
        # budget is one shared allowance, so a scalar sum is exactly the enforced quantity.
        self._total_cost = 0.0
        self._total_calls = 0
        # In-scope invoke_ids (populated at InvokeEnter). Distinguishes a real in-scope invoke
        # from the non-loop query() path so accounting/enforcement skip the latter.
        self._entered: set[str] = set()
        self._setup_done = False
        # Guards the aggregate and the _entered set against concurrent sub-agent turns.
        self._lock = threading.RLock()

    # --- lifecycle ---

    def setup(self) -> None:
        """Reset the aggregate for this hook's scope (with/local_hooks entry)."""
        with self._lock:
            self._total_cost = 0.0
            self._total_calls = 0
            self._entered = set()
        self._setup_done = True

    def teardown(self, exc: BaseException | None = None) -> None:
        """No-op: this hook writes no report. Present so the scope-close contract is explicit."""
        # Nothing to flush — the aggregate is live enforcement state, not an artifact. (The
        # forest model wrote a JSON envelope here; that moved to the ATIF trace.)
        return

    # --- repr ---

    def __repr__(self) -> str:
        # Overridden so an observability trace says which budget applied, not just that a
        # BudgetPool was active. Only the enforcing params: `warning_text` is prose, which does
        # not constrain the run.
        return (
            f"BudgetPool(cost_budget={self.cost_budget!r}, "
            f"calls_budget={self.calls_budget!r}, "
            f"warn_cost_remaining={self.warn_cost_remaining!r}, "
            f"warn_calls_remaining={self.warn_calls_remaining!r})"
        )

    # --- events ---

    def on_invoke_enter(self, event: InvokeEnter) -> list[Effect]:
        with self._lock:
            if not self._setup_done:
                # Real raise (assert is stripped under `python -O`). Triggered if an invoke
                # reaches this hook without setup() having run.
                raise RuntimeError(
                    "BudgetPool received an invoke before setup() ran. "
                    "It is a scoped hook: activate it via "
                    "`with BudgetPool(...)` or "
                    "`jaz.invoke(BudgetPool(...), ...)`."
                )
            self._entered.add(event.invoke_id)
        # No entry hard-stop: an invoke born into an already-exhausted pool is caught
        # by the first on_llm_query_enter, which also fires before the first LLM
        # query (verified by the budget tests).
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        # An unregistered invoke ⇒ the non-loop query() path (no InvokeEnter) — book nothing.
        if event.invoke_id not in self._entered:
            return []
        # Non-completed arms (#892 outcome union) record no call: the query failed before
        # a response — there is no cost or call to book. (on_invoke_exit deliberately has NO
        # such guard: its _entered cleanup must run on every outcome.)
        if not isinstance(event.outcome, Completed):
            return []
        response = event.outcome.result
        # Observed-uncosted-turn BACKSTOP (#1071). It lives HERE (the Completed *Exit), not at
        # on_llm_query_complete, so it reads the FINAL, post-transform cost — the SAME value booked
        # just below — rather than the raw pre-transform one. That makes the two coherent: a
        # ModifyLLMResponse(cost_usd=…) re-pricing an otherwise-uncostable model now rescues the
        # turn (final cost non-None → no abort → booked on the re-priced value), and a genuinely
        # uncostable turn still aborts. Abort at the Completed *Exit still fires *before* the agent
        # acts on the response, so the uncosted turn's code never executes (the #1071 property is
        # preserved); the aborted turn books nothing (we return before the booking below). See
        # HookDispatcher._resolve_completed_exit_effects for why the abort is valid at Exit now.
        if response.cost_usd is None and self.cost_budget is not None:
            return [Abort(error=self._pricing_error(event.model))]
        # Book the call. An unknown cost (cost_usd None) is recorded as 0.0 — reachable only when
        # there is NO cost_budget (the branch above aborts the cost_budget case), i.e. a calls_budget
        # over an unpriceable model. `total_llm_calls` is incremented ONLY here, so booking every
        # completed call (including a free/unknown one) is what lets a calls_budget actually count.
        with self._lock:
            self._total_cost += (
                response.cost_usd if response.cost_usd is not None else 0.0
            )
            self._total_calls += 1
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # An unregistered invoke ⇒ the non-loop query() path (which emits no InvokeEnter, so is
        # never in _entered) — skip both the hard stop and the warning.
        if event.invoke_id not in self._entered:
            return []

        # Unenforceable-budget PRE-FLIGHT: the client says up front it cannot report a cost
        # for the model this query is about to use, so the budget is unenforceable before a
        # token is spent on it. This asks the BACKEND (`LLM.can_report_cost`) rather than
        # reading the price table here — only the client knows where its cost comes from. A
        # table lookup in this hook would false-abort every client that prices by another
        # route: `MockLLMClient` hands back its own cost and never consults the table, and
        # its default model name ("mock") is not in it. (The observed-uncosted-turn BACKSTOP
        # lives at on_llm_query_exit, on the post-transform cost — see there.)
        #
        # Pre-flight is first-turn-only in practice, but by consequence rather than by
        # construction: there is no iteration test here, it simply aborts, so no second turn
        # follows. Do not go looking for a turn guard.
        #
        # Aborts go via Abort, the supported termination channel: a hook that
        # raises is swallowed by the dispatcher.
        if self.cost_budget is not None and not event.can_report_cost:
            return [Abort(error=self._unreportable_cost_error(event.model))]

        # Hard-stop (aggregate budget reached): abort at the top of the turn, before the
        # LLM query. Global — independent of which invoke this is. Lives on the
        # always-present LLMQueryEnter (fires once per turn, before the first query too),
        # not REPL *execution* (skipped for unparseable output). Emitted as an Abort
        # (#481); span_llm_query raises BudgetPoolExhaustedError out of the invoke (its spans
        # close Aborted). On the abort turn we emit only the Abort (no warning add — the
        # query is discarded anyway).
        #
        # This per-turn re-check is load-bearing under the abort-propagation model: an aborted
        # CHILD invoke surfaces in its parent's REPL as an ordinary catchable
        # BudgetExhaustedError, so the parent keeps running — until ITS next turn reaches this
        # check and its own control plane stops it too. Each invoke in the tree is stopped by
        # its own per-turn check; the top-level one raises BudgetPoolExhaustedError bare to the
        # jaz.invoke() caller.
        err = self._budget_error()
        if err is not None:
            return [Abort(error=err)]

        # The approaching-budget warning is the hook's only agent-facing text: nothing is said
        # unless an opt-in warn_*_remaining headroom is set and the remaining budget has fallen
        # to it. Emitted as a transient AddMessages folded into this query. The text is the
        # caller's warning_text verbatim (an empty string ⇒ emit nothing); the hook adds no
        # framing. Different hooks' messages are not coalesced.
        content = "\n".join(t for t in self._warning_texts() if t)
        return [AddMessages([{"role": "user", "content": content}])] if content else []

    # The observed-uncosted-turn BACKSTOP (#1071) used to live on ``on_llm_query_complete`` here,
    # reading the *raw* pre-transform ``response.cost_usd``. It moved to ``on_llm_query_exit`` so it
    # reads the FINAL, post-transform cost — coherent with what that same method books, and so a
    # ``ModifyLLMResponse(cost_usd=…)`` re-price can rescue an otherwise-uncostable turn. It stays a
    # pre-*execution* abort: the Completed ``LLMQueryExit`` still fires before the agent acts on the
    # response (see ``on_llm_query_exit`` and ``HookDispatcher._resolve_completed_exit_effects``).

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        # Drop this invoke from the in-scope set. The aggregate is deliberately NOT decremented:
        # a finished invoke's cost still counts against the shared budget for the scope's life.
        with self._lock:
            self._entered.discard(event.invoke_id)
        return []

    # --- aggregate + enforcement (the hook is the pool) ---

    def _aggregate_cost(self) -> float:
        with self._lock:
            return self._total_cost

    def _aggregate_calls(self) -> int:
        with self._lock:
            return self._total_calls

    def _unreportable_cost_error(self, model: str) -> Exception:
        """The config error when the client says up front it cannot cost this model."""
        return ModelPricingUnavailableError(
            f"cost_budget=${self.cost_budget:.6f} was set, but the LLM client reports that it "
            f"cannot provide a cost for model {model!r}, so the cost budget cannot be "
            f"enforced. For the default client this means the model is missing from the "
            f"bundled price table — refresh it (see the regeneration command in "
            f"jaz/llm/pricing.py) or use calls_budget, which counts LLM calls and needs "
            f"no per-token rates. No LLM call was made for this model."
        )

    def _pricing_error(self, model: str) -> Exception:
        """The config error when a turn reported no cost while a ``cost_budget`` is set.

        Emitted by ``on_llm_query_exit`` the turn the uncosted response is observed.
        """
        return ModelPricingUnavailableError(
            f"cost_budget=${self.cost_budget:.6f} was set, but no cost was reported for model "
            f"{model!r}, so the cost budget cannot be enforced. Usually this "
            f"means the model is missing from the bundled price table — refresh it (see the "
            f"regeneration command in jaz/llm/pricing.py) or use calls_budget, which "
            f"counts LLM calls and needs no per-token rates."
        )

    def _budget_error(self) -> Exception | None:
        """The hard-stop error once the aggregate cost OR calls reaches the budget.

        Reaching the budget IS the hard stop — there is no buffer. Cost takes
        precedence over calls when both are exhausted.
        """
        if self.cost_budget is not None:
            total = self._aggregate_cost()
            if total >= self.cost_budget:
                return BudgetPoolExhaustedError(
                    f"LLM cost budget exhausted: ${total:.6f} spent, "
                    f"budget was ${self.cost_budget:.6f}"
                )
        if self.calls_budget is not None:
            calls = self._aggregate_calls()
            if calls >= self.calls_budget:
                return BudgetPoolExhaustedError(
                    f"LLM calls budget exhausted: {calls} calls made, "
                    f"budget was {self.calls_budget}"
                )
        return None

    def _warning_texts(self) -> list[str | None]:
        """The caller's ``warning_text`` when an approaching-budget threshold is crossed, else [].

        No hook-authored prose (no "$X remaining" line, no "Finish soon…"): the message is exactly
        the caller's ``warning_text`` — guaranteed non-``None`` when a threshold is set, by the
        coupling; an empty string emits nothing. The hard stop (``_budget_error``) runs before this
        in on_llm_query_enter, so a crossed threshold means remaining is still > 0.
        """
        crossed = (
            self.cost_budget is not None
            and self.warn_cost_remaining is not None
            and (self.cost_budget - self._aggregate_cost()) <= self.warn_cost_remaining
        ) or (
            self.calls_budget is not None
            and self.warn_calls_remaining is not None
            and (self.calls_budget - self._aggregate_calls())
            <= self.warn_calls_remaining
        )
        return [self.warning_text] if crossed else []
