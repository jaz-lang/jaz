"""LLM query event, context, and span."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from jaz._llm_client import LLMResponse
from jaz.providers import MessageDict
from jaz.repl.types import ExecResult

from ..base import Event, ExecutionContext
from ..effects import AddMessages


@dataclass(frozen=True)
class LLMQueryEnter(Event):
    """Fired once per turn, before making an LLM API call.

    **Fires on every turn** — it is the always-present per-turn boundary, unlike the
    conditional REPL execution events (skipped when the response fails to parse) and
    :class:`LLMQueryExit` (skipped when the query never completes). That makes it the home of
    per-turn loop and budget control.

    Allowed effects:

    - :class:`Abort` — terminate the invoke; this is where the iteration and budget
      hard-stops fire.
    - :class:`AddMessages` — add a message to the prompt for this query.
    - :class:`DropMessages` — remove messages from the prompt for this query.
    - :class:`OverrideResponse` — supply a response and skip the LLM call.

    Attributes:
        messages: The message buffer as it stands before this query's edits are folded in.
        model: Which model is about to be called.
        iteration: The agent-loop iteration this query belongs to.
        can_recurse: Whether this invoke still has the recursive ``jaz.invoke`` tool.
        can_report_cost: Whether the client will report a per-call cost for this model
            (see :meth:`~jaz.providers.llm.LLM.can_report_cost`). Lets a cost budget be
            rejected before the first query instead of after paying for a turn.
    """

    messages: list[MessageDict]
    model: str
    iteration: int
    # Whether this invoke still has the recursive ``jaz.invoke`` tool (i.e. no
    # ``DisableRecursion`` was emitted at InvokeEnter — see ``RecursionLimit``). Present so
    # hooks that emit message edits here (the only context where AddMessages is valid) can
    # gate delegation guidance: ``can_delegate = event.can_recurse`` (the must-exit warnings).
    # With no ``RecursionLimit`` installed this is always ``True`` (the framework imposes no
    # recursion cap by default). This replaced the former ``max_depth: int`` field: the cap is
    # no longer a framework value the primitive knows — it lives in the opt-in hook — so the
    # primitive only knows the boolean "is recursion still available for this invoke".
    can_recurse: bool
    # Whether the client will populate ``LLMQueryExit.response.cost`` for this model, asked
    # of the backend itself (``LLM.can_report_cost``) rather than inferred from a price
    # table — only the client knows where its cost comes from. Carried as a plain bool rather
    # than exposing the client object, which would hand every hook the ability to issue
    # queries. Defaults ``True`` so a hand-constructed event (tests) keeps the permissive
    # reading: cost is assumed reportable unless the client says otherwise.
    can_report_cost: bool = True


@dataclass(frozen=True)
class LLMQueryExit(Event):
    """Fired after receiving LLM response.

    **Not every turn.** It is skipped whenever the query never completed: a hook emitted an
    :class:`Abort` at :class:`LLMQueryEnter` (the call never happened), or the call failed and
    its retries were exhausted. A response supplied by :class:`OverrideResponse` *does* fire it — no call is
    made, but the query completes normally.

    This event is purely observational - no effects can be emitted.

    Attributes:
        response: The ``LLMResponse`` the client returned — content, token counts, and cost.
        model: Which model was called; mirrors :class:`LLMQueryEnter`'s ``model``.
        iteration: The agent-loop iteration this query belongs to.
        start_time: When the call started, measured by the agent.
        end_time: When the call finished, measured by the agent.
        message_edits: The :class:`DropMessages` / :class:`AddMessages` edits composed for this query —
            the only way a hook can see which messages were *removed*.
    """

    # ``model`` / ``start_time`` / ``end_time`` stay **event** fields rather than folding into
    # ``response`` (a divergence from the #481 doc): they are *query* metadata, not *response*
    # content. The timing is wall-clock measured by the agent around the call, not returned by
    # the provider, so hanging it off the provider's ``LLMResponse`` would force that return
    # type to carry fields it never populates.
    #
    # ``message_edits`` is the only removal-visible surface: a dropped message leaves the
    # buffer, so per-message provenance on the survivors cannot reveal it. ``ConversationHistory``
    # resolves the drop indices against the query's enter snapshot to log removals + reasons.

    response: LLMResponse
    model: str
    iteration: int
    start_time: datetime | None = None
    end_time: datetime | None = None
    message_edits: MessageEdits | None = None


@dataclass
class MessageEdits:
    """Composed :class:`DropMessages` / :class:`AddMessages` edits for one query, split by
    persistence.

    :func:`jaz.hooks.effects.apply_message_edits` is folded twice over the *same*
    snapshot: the **persistent** view (carried forward into the buffer) uses only the
    persistent drops/adds; the **shown** view (sent to the model this turn) uses all of
    them. Drops compose by union; adds accumulate (their within-slot order is resolved
    deterministically by the fold).
    """

    persistent_drops: set[int] = field(default_factory=set)
    transient_drops: set[int] = field(default_factory=set)
    persistent_adds: list[AddMessages] = field(default_factory=list)
    transient_adds: list[AddMessages] = field(default_factory=list)

    @property
    def all_drops(self) -> set[int]:
        return self.persistent_drops | self.transient_drops

    @property
    def all_adds(self) -> list[AddMessages]:
        return [*self.persistent_adds, *self.transient_adds]

    @property
    def has_persistent(self) -> bool:
        return bool(self.persistent_drops or self.persistent_adds)


@dataclass
class LLMQueryContext(ExecutionContext):
    """Context for LLM query events.

    Hooks can:
    - Terminate the invoke via :class:`Abort` — this is the always-present per-turn point where
      loop/budget hard-stops live; the composed exceptions are collected in ``abort_errors``
      and resolved to a terminal :class:`Raise` by ``span_llm_query`` (before the LLM call).
    - Edit the messages before the query via :class:`DropMessages` / :class:`AddMessages`, each
      *transient* (a per-query view) or *persistent* (edits the carried-forward buffer);
      composition is order-independent (see :class:`MessageEdits`).
    - Override the response, skipping the API call (via OverrideResponse).
    """

    message_edits: MessageEdits = field(default_factory=MessageEdits)
    override_response: LLMResponse | None = None
    # Exceptions from ``Abort`` effects emitted at ``LLMQueryEnter``. Resolved to a terminal
    # ``Raise`` in ``span_llm_query`` (kept named for the exceptions they carry).
    abort_errors: list[Exception] = field(default_factory=list)


class LLMQuerySpan:
    """Span for LLM query.

    Usage:
        with dispatcher.span_llm_query(...) as span:
            response = llm.complete(...)  # an LLMResponse
            span.complete(response=response, iteration=..., start_time=..., end_time=...)
    """

    def __init__(self, ctx: LLMQueryContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        self._response: LLMResponse | None = None
        self._iteration: int | None = None
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        # Terminal ``Raise`` produced by an enter-time ``Abort``, if any. When set, the
        # caller skips the LLM call and returns this to terminate the invoke; the span fires
        # no ``LLMQueryExit`` (the query never happened). Set by ``span_llm_query`` at enter.
        self.abort: ExecResult | None = None

    def complete(
        self,
        *,
        response: LLMResponse,
        iteration: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        """Complete span with the LLM response and query timing.

        Args:
            response: The LLMResponse (content + token counts + cost) the client returned.
            iteration: The REPL iteration number.
            start_time: When the LLM call started (agent-measured).
            end_time: When the LLM call finished (agent-measured).

        Raises:
            RuntimeError: If span was already completed
        """
        if self._completed:
            raise RuntimeError("Span already completed")
        self._response = response
        self._iteration = iteration
        self._start_time = start_time
        self._end_time = end_time
        self._completed = True

    def is_completed(self) -> bool:
        """Check if span was completed."""
        return self._completed

    def get_response(self) -> LLMResponse:
        """Get the LLMResponse.

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        assert self._response is not None
        return self._response

    def get_iteration(self) -> int:
        """Get the iteration number.

        Returns:
            The iteration provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        assert self._iteration is not None
        return self._iteration

    def get_start_time(self) -> datetime | None:
        """Get the LLM call start time.

        Returns:
            The start_time provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        return self._start_time

    def get_end_time(self) -> datetime | None:
        """Get the LLM call end time.

        Returns:
            The end_time provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        return self._end_time


# Retry lives here with the rest of the LLMQuery* family (rather than a standalone
# module) — it is a query-scoped event, and co-locating keeps the whole family in one
# place. Unlike Enter/Exit it is a simple observational event with no span / enter-exit
# pair: it fires each time an in-flight LLM call fails and tenacity is about to retry it.


@dataclass(frozen=True)
class LLMQueryRetry(Event):
    """Fired when an LLM call fails and is about to be retried.

    **Not a per-turn event.** It fires once per retry attempt, so a turn whose first attempt
    succeeds never fires it, and a turn that retries three times fires it three times.

    This event is purely observational - no effects can be emitted.

    Attributes:
        model: The model the failed call was addressed to.
        attempt_number: Which attempt just failed, counting from 1.
        exception: The error that caused the retry.
        wait_seconds: How long the retry will wait before the next attempt.
        iteration: The agent-loop iteration the call belongs to.
    """

    model: str
    attempt_number: int
    exception: Exception
    wait_seconds: float
    iteration: int


@dataclass
class LLMQueryRetryContext(ExecutionContext):
    """Context for LLM retry events.

    This is a read-only context - LLM retries are informational.
    Hooks can only record metrics.
    """

    pass
