"""User-facing hook for LLM cost / call-count budget tracking and enforcement.

This hook **is the budget pool**: one shared envelope per hook scope. Apply it as a
context manager around an invoke::

    from jaz.hooks import BudgetPool

    with BudgetPool(cost_budget=1.0):
        result = jaz.invoke(jaz.ReturnType(str), task=task)

A single instance propagates (via contextvars) to all invokes in scope — but ONLY on the
``with`` path. Passed positionally (``jaz.invoke(BudgetPool(...), ...)``) it is bound to that
one invoke's dispatcher and never reaches invokes nested under it, so their cost is neither
counted nor enforced. Measured on the same root+child workload: 3 calls / $0.04 under ``with``,
2 calls / $0.02 positionally. The hook owns
the one budget, the running aggregate, all enforcement, and a *forest* of per-invoke
``CostTracker`` nodes (``invoke_id -> node``): every top-level invoke under the hook is
its own root, and nested invokes chain off their real parent. There is no synthetic
pool node — the budget lives on the hook and the aggregate is the sum of the roots'
subtree totals. All top-level invokes therefore draw down ONE shared budget.

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
  ``on_llm_query_enter``) once the aggregate reaches ``warn_fraction`` of the budget.
  Purely additive prompt text — there is **no** result-overriding ``ModifyResult``.
- **Status** (opt-in via ``show_status``, default off) surfaces the current budget
  usage each turn via the same transient ``AddMessages``. Off by default, the hook is quiet
  until the agent approaches the budget (the warning); turn it on for a per-turn
  running tally.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path

from jaz.budget import CostTracker
from jaz.exceptions import BudgetExhaustedError, ModelPricingUnavailableError
from jaz.hooks.builtin._must_exit import resolve_must_exit_warning
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Abort, AddMessages, Effect
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
)

logger = logging.getLogger(__name__)

# --- agent-facing budget-status formatting ---
#
# The CostTracker nodes own the numbers; the agent-facing *status strings* live here
# (the hook's presentation concern). The status reflects the hook's one pool budget vs
# its running aggregate — there is a single budget, so every invoke sees the same line.


def _format_cost_budget_status(total: float, budget: float) -> str:
    """Agent-facing pool cost-budget status. Reports current usage only — never
    exhaustion (the hard stop owns that) and only called when a cost budget is set."""
    remaining = budget - total
    percent_used = (total / budget * 100) if budget > 0 else 0
    return (
        f"Budget: ${total:.6f} / ${budget:.6f} "
        f"({percent_used:.1f}% used, ${remaining:.6f} remaining)"
    )


def _format_calls_budget_status(total: int, budget: int) -> str:
    """Agent-facing pool calls-budget status (see ``_format_cost_budget_status``)."""
    remaining = budget - total
    percent_used = (total / budget * 100) if budget > 0 else 0
    return f"LLM calls: {total} / {budget} ({percent_used:.1f}% used, {remaining} remaining)"


class BudgetPool(Hook):
    """Enforce one shared LLM cost/call budget over everything in the hook's scope.

    Every invoke the hook is active for draws on the *same* budget. Reaching ``cost_budget``
    or ``calls_budget`` is a hard stop: the invoke ends by raising
    :class:`~jaz.exceptions.BudgetExhaustedError`. Before that, once spending crosses
    ``warn_fraction`` of a budget, the agent is warned to finish.

    The scope covered by the budget pool depends on how it is activated:

    * ``with BudgetPool(...):`` — every ``jaz.invoke()`` in the block *and every invoke
      nested inside those* shares the one budget. Use this to cap a whole workflow.
    * ``jaz.invoke(BudgetPool(...), ...)`` — the pool covers **only that invoke's own LLM
      calls, excluding its sub-invokes.**

    The budget starts empty when the scope opens.

    Note that if the one turn jumps from under ``warn_fraction`` to past the budget, then the
    agent is stopped with no warning at all. Thus, it is recommended that
    ``(1 - warn_fraction) * cost_budget`` be appreciably larger than the typical cost usage of
    a single turn.

    **Missing token usage degrades; it does not crash.** A provider may return a
    response with no usage block (``DefaultLLMClient`` then reports the token counts as
    ``None``); the missing counters are recorded as ``0`` with a once-per-model warning
    rather than raising, so the call is still booked and a ``calls_budget`` keeps
    counting it.

    Args:
        cost_budget: Hard ceiling on total LLM cost in USD. ``None`` (default) means no cost
            limit.
        calls_budget: Hard ceiling on total LLM calls. ``None`` (default) means no call limit.
        warn_fraction: Fraction of a budget at which the agent starts being warned to finish.
            Default 0.9.
        show_status: Show the agent a budget status line every turn ("Budget: $X / $Y (Z%
            used, ...)"). Default False, so the hook stays quiet until the warning.
        must_exit_warning: Text emitted with the approaching-budget warning. ``None``
            (default) renders the stock wording; otherwise a string, or a callable
            ``(can_delegate: bool) -> str`` called when the warning is emitted. Rendering to
            an empty string emits nothing.
        json_path: Write a JSON cost report here when the scope exits — the budget and
            totals, with each top-level invoke under ``invoke_calls``. One combined file
            however many top-level invokes ran.

    Raises:
        ValueError: If ``warn_fraction`` is outside ``[0, 1]``.
    """

    # Internally the tracked invokes are a *forest*: one `CostTracker` tree per top-level
    # invoke (`_roots`), with nested invokes chained onto their parent's tree. The budget is
    # enforced on the sum over all roots, which is what makes it one shared allowance rather
    # than a per-invoke one. The docstring says "everything in scope shares one budget"
    # instead — "forest" is our bookkeeping, not something a caller has to know.
    #
    # Why the positional form excludes sub-invokes: it is not a BudgetPool quirk. A
    # positionally-passed hook is bound to that Agent's dispatcher and stays off the
    # contextvar, so nested invokes (which build their own dispatcher) never dispatch to it
    # — see `_activate_local_hooks` in `invoke.py`. The docstring states only the outcome;
    # "dispatcher" is our machinery, not a word a caller should have to know.
    #
    # The consequence, also left out of the docstring: for a *budget* the miss is silent. The
    # sub-invoke's cost is never recorded, so the envelope under-reports rather than erroring —
    # a delegating task can run up cost the pool never sees while its report says the budget was
    # never approached. Measured on the same root+child workload: 3 calls / $0.04 under `with`,
    # 2 calls / $0.02 passed positionally.
    #
    # `on_invoke_enter` also raises `RuntimeError` if an invoke reaches the hook without
    # `setup()` having run. Not in the docstring: `with` and the positional argument are the
    # only supported activations and both run `setup()`, so the raise is unreachable from
    # documented usage — it exists to catch a hook wired up through some other path.
    #
    # Mechanics kept out of the docstring: the hook handles `InvokeEnter`/`InvokeExit` (forest
    # bookkeeping) and `LLMQueryEnter`/`LLMQueryExit` (enforcement and accounting). The hard
    # stop is an `Abort` at `LLMQueryEnter` — the top of the turn, before the query — chosen
    # because that event is the always-present per-turn boundary; REPL *execution* is skipped
    # for unparseable output, so enforcing there would let a turn slip past the budget. The
    # warning is an `AddMessages`. There is deliberately no buffer and no `ModifyResult`
    # rewrite of results: the only two levers are prompt text and the hard stop.
    #
    # Exhaustion raises rather than granting a final wrap-up turn — an executive call:
    # reaching the limit IS the stop. A "one last turn to RETURN something" affordance would
    # mean the budget can always be exceeded by one turn, which makes it not a ceiling.
    #
    # `setup()` opens the forest and `teardown()` flushes the JSON envelope; both run only on a
    # `with`/local activation. Hence the `_setup_done` guard in `on_invoke_enter`: an invoke
    # that never went through `setup()` raises instead of silently accumulating into state
    # that will never be flushed.
    #
    # The stock `must_exit_warning` is deliberately finish-method-agnostic — it says to finish,
    # not *how*, because the finish mechanism is the REPL's concern, not the budget's.

    def __init__(
        self,
        *,
        cost_budget: float | None = None,
        calls_budget: int | None = None,
        warn_fraction: float = 0.9,
        show_status: bool = False,
        must_exit_warning: str | Callable[[bool], str] | None = None,
        json_path: str | None = None,
    ) -> None:
        self.cost_budget = cost_budget
        self.calls_budget = calls_budget
        # Guard the public param: out-of-range values degrade silently otherwise —
        # >1 disables the warning entirely (the hard stop at the budget fires first),
        # <=0 warns from the very first turn (the fraction starts at 0.0 and 0 >= 0).
        if not 0 <= warn_fraction <= 1:
            raise ValueError(f"warn_fraction must be in [0, 1], got {warn_fraction!r}")
        self.warn_fraction = warn_fraction
        self.show_status = show_status
        self.must_exit_warning = must_exit_warning
        self.json_path = json_path
        # The forest: invoke_id -> accounting node, plus the roots (top-level invokes).
        self._nodes: dict[str, CostTracker] = {}
        self._roots: list[CostTracker] = []
        self._setup_done = False
        # Model from the first turn that reported no cost while a cost_budget was set; the
        # next on_llm_query_enter turns it into an Abort. See _pricing_error.
        self._uncosted_model: str | None = None
        # Guards the forest bookkeeping (the _nodes/_roots mutations and the aggregate
        # reads). Each node *tree* has its own lock for within-tree propagation; this
        # is the hook-level lock. Ordering is always hook-lock -> tree-lock.
        self._lock = threading.RLock()
        # Models already warned about missing token-usage data, so the degraded-
        # accounting warning fires once per model, not once per LLM call.
        self._degraded_warned: set[str] = set()

    # --- lifecycle ---

    def setup(self) -> None:
        """Open the forest for this hook's scope (with/local_hooks entry)."""
        self._nodes = {}
        self._roots = []
        self._uncosted_model = None
        # Per-scope, like _nodes/_roots: a pool entered a second time must warn afresh —
        # the suppression is once-per-scope, not once-per-process.
        self._degraded_warned = set()
        self._setup_done = True

    def teardown(self, exc: BaseException | None = None) -> None:
        """End any still-open nodes and write the one JSON envelope."""
        if not self._setup_done:
            return
        # Defensive: on an exceptional exit an invoke may not have produced its
        # InvokeExit, leaving a started-but-unended node. end() is required before
        # to_dict(), so close any stragglers before serializing.
        with self._lock:
            stragglers = list(self._nodes.values())
            roots = list(self._roots)
            self._nodes = {}
        for node in stragglers:
            if node.end_time is None:
                node.end()
        if self.json_path is not None:
            self._write_envelope(roots, self.json_path)

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
            parent = (
                self._nodes.get(event.parent_invoke_id)
                if event.parent_invoke_id is not None
                else None
            )
            if parent is not None:
                # Nested invoke: chain off its real parent (same tree + lock).
                node = parent.add_invoke_call()
            else:
                # Top-level invoke: a genuine new root of the forest.
                node = CostTracker(
                    depth=event.depth,
                )
                self._roots.append(node)
            node.start()
            self._nodes[event.invoke_id] = node
        # No entry hard-stop: an invoke born into an already-exhausted pool is caught
        # by the first on_llm_query_enter, which also fires before the first LLM
        # query (verified by the budget tests).
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        node = self._nodes.get(event.invoke_id)
        if node is None:
            return []
        response = event.response
        # Missing token usage is DATA, not a programming error: a provider can return a
        # response with no usage block (``DefaultLLMClient`` maps a missing ``usage`` to
        # prompt/completion/total = ``None``). The old ``assert ... is not None`` turned
        # that into a per-query hook exception — swallowed by the dispatcher but logged
        # with a full traceback on EVERY LLM call of an otherwise-working invoke — and,
        # worse, fired *before* ``add_llm_call``, so a ``calls_budget`` on such a model
        # silently counted zero and never fired. Degrade instead: record the call with 0
        # for the missing counters and warn once per model.
        prompt_tokens = response.prompt_tokens
        completion_tokens = response.completion_tokens
        total_tokens = response.total_tokens
        if prompt_tokens is None or completion_tokens is None or total_tokens is None:
            self._warn_degraded_once(
                event.model,
                f"No token usage info for model {event.model!r}; missing counters "
                "are recorded as 0 in the budget accounting.",
            )
            # `0 if x is None else x`, not `x or 0`: only a *missing* counter is
            # zero-filled; a legitimate 0 already reads as 0, and the explicit form says
            # "fill the None" rather than leaning on falsy-zero coinciding with it.
            prompt_tokens = 0 if prompt_tokens is None else prompt_tokens
            completion_tokens = 0 if completion_tokens is None else completion_tokens
            total_tokens = 0 if total_tokens is None else total_tokens
        # Query timing, unlike usage, IS a framework invariant: the agent stamps
        # start/end around the call (even on the OverrideResponse path) before
        # span.complete(), so None here is a jaz bug, not bad provider data.
        assert event.start_time is not None and event.end_time is not None
        # A missing cost used to be `assert response.cost is not None`. That was the worst
        # possible handling: the dispatcher swallows hook exceptions, so the assert produced
        # one log line and a cost budget that then silently enforced nothing — and under
        # `python -O` the assert vanishes entirely. Record the model instead and let the next
        # on_llm_query_enter abort through the supported channel. Only an enforcement problem
        # when a cost budget is set; otherwise an uncosted turn just contributes nothing.
        if response.cost is None and self.cost_budget is not None:
            self._uncosted_model = event.model
        # The turn is recorded either way, with an unknown cost booked as 0.0. Skipping
        # `add_llm_call` for an uncosted turn would be a second silent no-op of exactly the kind
        # this hook exists to prevent: `total_llm_calls` is incremented ONLY here, so a
        # `calls_budget` on an uncosted model would count zero forever and never fire. That is the
        # remedy `_pricing_error` recommends, so it has to work. Verified: cost=None with
        # `calls_budget=2` ran to the iteration limit before this, and stops at 2 after.
        #
        # Booking 0.0 conflates "free" with "unknown" in the cost total, which is acceptable
        # precisely because the pairing that would be misled by it — an uncosted turn under a
        # `cost_budget` — is aborted by the guard above rather than left to run on a total that
        # undercounts. With no cost budget there is nothing for the understated total to break,
        # and the tokens and per-call records stay accurate in the teardown envelope either way.
        node.add_llm_call(
            model=event.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=response.cost if response.cost is not None else 0.0,
            iteration=event.iteration,
            start_time=event.start_time,
            end_time=event.end_time,
        )
        return []

    def _warn_degraded_once(self, model: str, message: str) -> None:
        """Warn about degraded (zero-filled) accounting once per model.

        Once, because the condition is per-model and stable for the scope (a provider
        that omits usage keeps omitting it) — repeating it every LLM call would itself
        be the wall-of-noise this path exists to avoid.
        """
        with self._lock:
            if model in self._degraded_warned:
                return
            self._degraded_warned.add(model)
        logger.warning(message)

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # An unregistered invoke ⇒ the non-loop query() path (which emits no InvokeEnter, so
        # registers no node) — skip both the hard stop and the warning.
        if event.invoke_id not in self._nodes:
            return []

        # Unenforceable-budget guard, in two layers.
        #
        # (1) PRE-FLIGHT: the client says up front it cannot report a cost for the model this
        # query is about to use, so the budget is unenforceable before a token is spent on it.
        # This asks the BACKEND (`LLM.can_report_cost`) rather than reading the price
        # table here — only the client knows where its cost comes from. A table lookup in this
        # hook would false-abort every client that prices by another route: `MockLLMClient`
        # hands back its own cost and never consults the table, and its default model name
        # ("mock") is not in it.
        #
        # (2) BACKSTOP, one turn late: some earlier turn actually reported no cost. Kept even
        # with the pre-flight check, because `can_report_cost` is a promise a client can get
        # wrong — a stale table entry, a provider returning no usage — and the observed
        # condition is the ground truth. This layer costs the one turn already spent.
        # TODO(#1071): once #1059 (Abort@LLMQueryExit) lands, that one-turn cost is optional —
        # the backstop could abort directly at exit the turn the uncosted cost is observed,
        # dropping `_uncosted_model` (a field whose only job is to defer the decision to here).
        #
        # Ordered pre-flight first so that when both fire, the error names the model this
        # query would have used rather than whichever earlier one armed the backstop. They can
        # both be live at once because `_uncosted_model` persists across invokes in a pool
        # (cleared only in `setup()`): a completed invoke on model M1 can leave the backstop
        # armed, and a later invoke on an unpriceable M2 would otherwise abort naming M1.
        # Neither ordering can miss an abort — both conditions terminate — so this is purely
        # about which model the reader is pointed at, and the current query's is the useful one.
        #
        # Pre-flight is first-turn-only in practice, but by consequence rather than by
        # construction: there is no iteration test here, it simply aborts, so no second turn
        # follows. Do not go looking for a turn guard.
        #
        # Both abort via Abort@LLMQueryEnter, the supported termination channel: a hook that
        # raises is swallowed by the dispatcher.
        if self.cost_budget is not None and not event.can_report_cost:
            return [Abort(error=self._unreportable_cost_error(event.model))]

        err = self._pricing_error()
        if err is not None:
            return [Abort(error=err)]

        # Hard-stop (aggregate budget reached): abort at the top of the turn, before the LLM
        # query. Global — independent of which invoke this is. Lives on the always-present
        # LLMQueryEnter (fires once per turn, before the first query too), not REPL
        # *execution* (skipped for unparseable output). Emitted as an Abort (#481) →
        # span_llm_query turns it into a terminal Raise. On the abort turn we emit only the
        # Abort (no warning add — the query is discarded anyway).
        err = self._budget_error()
        if err is not None:
            return [Abort(error=err)]

        # The approaching-budget warning always fires (>= warn_fraction); the per-turn
        # status line is opt-in (show_status, default off) — quiet by default, surfacing
        # budget usage only once the agent is close to the limit. Emitted as a transient
        # AddMessages folded into this query.
        can_delegate = event.can_recurse
        texts: list[str | None] = []
        if self.show_status:
            texts.extend(self._status_texts())
        texts.extend(self._warning_texts(event, can_delegate))
        # One trailing user message for this hook (its status + warning fragments joined,
        # blanks dropped); empty ⇒ emit nothing. Different hooks' messages are not coalesced.
        content = "\n".join(t for t in texts if t)
        return [AddMessages([{"role": "user", "content": content}])] if content else []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        # End this invoke's node; the JSON envelope is written once for the whole
        # forest in teardown() (so multiple top-level invokes don't clobber json_path).
        # Pop from _nodes (the live invoke_id lookup) but deliberately leave the node in
        # _roots: the aggregate and the teardown envelope sum the roots, so a root must
        # outlive its invoke. _roots is therefore never pruned — a long hook scope
        # running many *sequential* top-level invokes retains every root tree (each with
        # its llm_calls list) for the hook's lifetime. This matches the old synthetic-pool
        # model (which held all children too): an unbounded-growth characteristic, not a leak.
        with self._lock:
            node = self._nodes.pop(event.invoke_id, None)
        if node is None:
            return []
        node.end()
        return []

    # --- aggregate + enforcement (the hook is the pool) ---

    def _aggregate_cost(self) -> float:
        with self._lock:
            return sum(r.total_llm_cost for r in self._roots)

    def _aggregate_calls(self) -> int:
        with self._lock:
            return sum(r.total_llm_calls for r in self._roots)

    def _unreportable_cost_error(self, model: str) -> Exception:
        """The config error when the client says up front it cannot cost this model."""
        return ModelPricingUnavailableError(
            f"cost_budget=${self.cost_budget:.6f} was set, but the LLM client reports that it "
            f"cannot provide a cost for model {model!r}, so the cost budget cannot be "
            f"enforced. For the default client this means the model is missing from the "
            f"bundled price table — refresh it (see the regeneration command in "
            f"jaz/providers/pricing.py) or use calls_budget, which counts LLM calls and needs "
            f"no per-token rates. No LLM call was made for this model."
        )

    def _pricing_error(self) -> Exception | None:
        """The config error when a turn reported no cost while a ``cost_budget`` is set.

        ``None`` until such a turn is seen. Set by ``on_llm_query_exit``.
        """
        if self._uncosted_model is None:
            return None
        return ModelPricingUnavailableError(
            f"cost_budget=${self.cost_budget:.6f} was set, but no cost was reported for model "
            f"{self._uncosted_model!r}, so the cost budget cannot be enforced. Usually this "
            f"means the model is missing from the bundled price table — refresh it (see the "
            f"regeneration command in jaz/providers/pricing.py) or use calls_budget, which "
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
                return BudgetExhaustedError(
                    f"LLM cost budget exhausted: ${total:.6f} spent, "
                    f"budget was ${self.cost_budget:.6f}"
                )
        if self.calls_budget is not None:
            calls = self._aggregate_calls()
            if calls >= self.calls_budget:
                return BudgetExhaustedError(
                    f"LLM calls budget exhausted: {calls} calls made, "
                    f"budget was {self.calls_budget}"
                )
        return None

    def _status_texts(self) -> list[str]:
        """Per-iteration pool budget status lines (one budget → every invoke sees it)."""
        texts: list[str] = []
        if self.cost_budget is not None:
            texts.append(
                _format_cost_budget_status(self._aggregate_cost(), self.cost_budget)
            )
        if self.calls_budget is not None:
            texts.append(
                _format_calls_budget_status(self._aggregate_calls(), self.calls_budget)
            )
        return texts

    def _warning_texts(
        self, event: LLMQueryEnter, can_delegate: bool
    ) -> list[str | None]:
        """Warn the agent as the aggregate approaches a budget (>= ``warn_fraction``).

        Purely additive prompt text: a numeric "you've used X%" line plus the shared
        must-exit warning (may be empty → dropped when building the message list). Fires every
        turn from the threshold up to (but not at) the budget, where the hard stop takes
        over.
        """
        used: list[str] = []
        if self.cost_budget is not None and self.cost_budget > 0:
            frac = self._aggregate_cost() / self.cost_budget
            if frac >= self.warn_fraction:
                used.append(f"{frac * 100:.0f}% of the cost budget")
        if self.calls_budget is not None and self.calls_budget > 0:
            frac = self._aggregate_calls() / self.calls_budget
            if frac >= self.warn_fraction:
                used.append(f"{frac * 100:.0f}% of the calls budget")
        if not used:
            return []
        return [
            f"WARNING: You have used {' and '.join(used)}. Finish soon "
            "before the budget is exhausted and the invoke is aborted.",
            resolve_must_exit_warning(
                self.must_exit_warning,
                can_delegate=can_delegate,
            ),
        ]

    # --- JSON envelope ---

    def _write_envelope(self, roots: list[CostTracker], json_path: str) -> None:
        """Write the one cost-tracking envelope: budget + aggregate + the root trees.

        The budget lives at the envelope level (not per node); each top-level invoke is
        a tree under ``invoke_calls``. Mirrors the old shape minus the per-node budgets.
        """
        envelope = {
            "llm_cost_budget": self.cost_budget,
            "llm_calls_budget": self.calls_budget,
            "n_invoke_calls": len(roots),
            "total_llm_cost": sum(r.total_llm_cost for r in roots),
            "total_llm_prompt_tokens": sum(r.total_llm_prompt_tokens for r in roots),
            "total_llm_completion_tokens": sum(
                r.total_llm_completion_tokens for r in roots
            ),
            "total_llm_calls": sum(r.total_llm_calls for r in roots),
            "total_invoke_calls": len(roots) + sum(r.total_invoke_calls for r in roots),
            "invoke_calls": [r.to_dict() for r in roots],
        }
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(envelope, f, indent=2)
