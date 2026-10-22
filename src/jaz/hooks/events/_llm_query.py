"""LLM query events, contexts, and span."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from jaz._messages import MessageDict
from jaz.llm._types import LLMResponse

from .._base import Event, ExecutionContext
from ..effects import AddMessages, ModifyLLMResponse
from ._base import Aborted, Completed, Failed

#: The LLM-query span's outcome union — what :attr:`LLMQueryExit.outcome` holds.
type LLMQueryOutcome = Completed[LLMResponse] | Aborted | Failed


@dataclass(frozen=True)
class LLMQueryEnter(Event):
    """Fired once per turn, before making an LLM API call.

    **Fires on every turn** — it is the always-present per-turn boundary, unlike the
    conditional REPL execution events (skipped when the response fails to parse). That
    makes it the home of per-turn loop and budget control.

    Allowed effects:

    - :class:`Abort` — abort the invoke; this is where the iteration and budget
      hard-stops fire.
    - :class:`AddMessages` — add a message to the prompt for this query.
    - :class:`DropMessages` — remove messages from the prompt for this query.

    Suppliers do not compose here: :class:`SupplyLLMResponse` is a :class:`LLMQuerySend`
    effect — a supplier decides on the basis of the message list, and the list it must see is
    the committed post-edit one, which at this event composition may still rewrite.

    Attributes:
        messages: The message buffer as it stands before this query's edits are folded in.
        iteration: The agent-loop iteration this query belongs to.
    """

    # ``Sequence`` in the annotation, ``tuple`` at runtime (see __post_init__): the
    # constructor accepts the caller's list, the type forbids mutation through the field.
    messages: Sequence[dict[str, Any]]
    iteration: int
    # No "may this invoke recurse" field. There used to be one (`can_recurse`), added so
    # message-editing hooks could gate delegation guidance on it; that consumer left core (a
    # harness can resolve it from `max_depth` at config time), and nothing under `src/`, `tests/`
    # or jaz-evals-internal reads it since. If a hook-facing signal is wanted again, expose it
    # once, on `InvokeSend`, computed from the bound tool — a hook reasons about what the REPL
    # has, not about a flag beside it. TODO(#1512) tracks that, with the migration paths a hook
    # has meanwhile.

    def __post_init__(self) -> None:
        super().__post_init__()
        # Sever the event's list from the loop's LIVE conversation buffer (the agent
        # passes it uncopied): without this, a hook appending/removing/reordering
        # ``event.messages`` silently edits the real conversation — the invoke events
        # made the equivalent mistake structurally impossible (read-only ``inputs``),
        # while this span handed hooks the live object (consistency audit C1). A tuple,
        # not a mutable list copy (review of this PR): one shared event instance is
        # dispatched to every hook, so a mutable copy would still let an earlier hook
        # alias-mutate what a later hook observes — and would desync the dispatcher's
        # Send-time edit resolution, which reads this field. Membership/order are pinned;
        # the dicts inside are still shared, the same deep/shallow ceiling as
        # ``InvokeEnter.inputs`` (see Event's docstring TODO). ``object.__setattr__`` is
        # the frozen-dataclass escape hatch for this framework-internal coercion.
        object.__setattr__(self, "messages", tuple(self.messages))


@dataclass
class MessageEdits:
    """Composed :class:`DropMessages` / :class:`AddMessages` edits for one query, split by
    persistence.

    :func:`jaz.hooks.effects._apply_message_edits` is folded twice over the *same*
    snapshot: the **persistent** view (carried forward into the buffer) uses only the
    persistent drops/adds; the **shown** view (sent to the model this turn) uses all of
    them. Drops compose by union; adds accumulate in hook order, persistent before transient
    (which is the within-slot order the fold applies).
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


# The resolved-edit record types below are PURE DATA on purpose: the resolution itself
# (raw drop indices / add slots → these self-contained records, against the enter
# snapshot) lives in the dispatcher, which holds both lists at emission time, so nothing here
# touches ``_resolve_edit_index``. Shipping bare indices instead would force every edit-consuming
# observer to keep its own snapshot of Enter's list to resolve against — rebuilding, one
# event earlier, exactly the machinery the pre-resolved records delete.


@dataclass(frozen=True)
class MessageDropRecord:
    """One message dropped from this query's committed input.

    Attributes:
        message: The dropped message itself (with its provenance stamp, so its stable
            per-message id is readable via :func:`jaz._provenance.provenance_of`).
        position: The resolved slot the message occupied in the enter-time snapshot when it
            was dropped -- the drop's input coordinate, mirroring :attr:`MessageAddRecord.position`.
            A live buffer index, so it drifts across turns; the message's id and iteration (via its
            provenance) are the stable identity.
        persistent: Whether the drop also removed the message from the carried-forward
            buffer (``True``) or applied to this query's view only (``False``).
    """

    message: dict[str, Any]
    position: int
    persistent: bool


@dataclass(frozen=True)
class MessageAddRecord:
    """One :class:`AddMessages` edit folded into this query's committed input.

    Attributes:
        messages: The inserted messages.
        position: The resolved slot in the enter-time snapshot the messages were inserted
            before (``len(snapshot)`` for an append).
        persistent: Whether the add also survives into later turns' buffers.
    """

    # Tuple-coerced like the events' message lists (see LLMQueryEnter.__post_init__):
    # these records ride a shared event (LLMQuerySend.message_edits), so a mutable list here
    # would let one hook alias-mutate what a later-dispatched hook observes.
    messages: Sequence[dict[str, Any]]
    position: int
    persistent: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))


@dataclass(frozen=True)
class ResolvedMessageEdits:
    """This query's composed message edits as self-contained, pre-resolved records.

    ``drops`` records :class:`DropMessages`, ``adds`` records :class:`AddMessages` — each
    sequence named by the verb of the effect it records.

    Ordering: ``drops`` by resolved position (one record per position, so that is a total
    order); ``adds`` in application order — by resolved slot, and within one slot as the fold
    applied them (persistent adds first, then transient, each in hook order).
    """

    # Tuple-coerced for the same shared-event reason as MessageAddRecord.messages.
    drops: Sequence[MessageDropRecord] = ()
    adds: Sequence[MessageAddRecord] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "drops", tuple(self.drops))
        object.__setattr__(self, "adds", tuple(self.adds))


@dataclass(frozen=True)
class LLMQuerySend(Event):
    """Fired when a query's input has committed — the post-edit message list is final.

    Fires after :class:`LLMQueryEnter`'s edits have composed, once per query, on **every**
    path where the input commits: before a live provider call, and equally when a
    :class:`SupplyLLMResponse` (e.g. :class:`TrajectoryReplay`) answers here and skips the call — the
    commitment of the input is real on both. It does not fire when the query is aborted at
    :class:`LLMQueryEnter` (the input never commits).

    Allowed effects:

    - :class:`SupplyLLMResponse` — supply a response, overriding the default producer (the
      provider call); the query still completes normally with it.
    - :class:`Abort` — abort the invoke, declining the committed input.

    Attributes:
        messages: The committed message list — the post-edit buffer view, the same shape
            as :attr:`LLMQueryEnter.messages` (provenance stamps intact). This turn's own
            :class:`AddMessages` additions carry no stamp yet — a persistent one is stamped
            after the query — so only the buffer's messages have a stable id here. The
            provider receives its wire projection, so a hook that forwards ``event.messages``
            to a provider itself must project them with
            :func:`jaz._provenance.to_wire_messages` first. The dicts are the buffer's
            own: rewriting one in place here changes the conversation for every later
            turn, exactly as at :class:`LLMQueryEnter`.
        iteration: The agent-loop iteration this query belongs to.
        message_edits: The enter-time edits that produced ``messages``, as self-contained records
            (dropped message content + persistence; added messages + resolved position).
    """

    # Send is a *default-action* event (span_event_lifecycle.md): emitting it IS the offer
    # of the committed input to result producers — the DOM model, where handlers may take
    # over (a supply effect answers inline, overriding the default producer; an abort
    # declines the input) and otherwise the default action runs (the provider call). That is
    # why it fires on the supplied path too, and why suppliers live here rather than at
    # Enter (a supplier decides on the basis of the input, and the input it must see is the
    # committed one — at Enter, composition may still rewrite it; a response cache keyed at
    # Enter would key on the proposal and return wrong hits under compaction).

    # ``Sequence`` in the annotation, ``tuple`` at runtime (see __post_init__): the
    # constructor accepts the caller's list, the type forbids mutation through the field.
    messages: Sequence[dict[str, Any]]
    iteration: int
    # ``message_edits``, not ``edits``: one ``<thing>_edits: Resolved<Thing>Edits`` per
    # editable field, named after the field it applies to (``code`` -> ``code_edits``,
    # ``namespace`` -> ``namespace_edits``), so the name says which Enter field the records
    # resolve against.
    message_edits: ResolvedMessageEdits = field(default_factory=ResolvedMessageEdits)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Sever the event's list from the committed list the PROVIDER is about to be
        # handed (the agent passes ``shown`` to this event and projects the same list to
        # wire form for the LLM call): a hook appending to or reordering ``event.messages``
        # here would otherwise edit the actual provider request. The tuple pins
        # membership, not contents — the dicts inside are the buffer's own, so a hook that
        # rewrites ``event.messages[i]["content"]`` at Send edits this query's request AND
        # the buffer for every later turn, exactly as at Enter (its __post_init__ has the
        # deep/shallow ceiling). Handing out the buffer's dicts rather than a per-Send copy
        # is deliberate: it is the "same shape as Enter" rule below, and it keeps
        # ``message_edits.drops[].message`` identity-equal to the buffer dicts it was dropped from.
        # Replay's NOT-covered list names this hazard; the attribute docstring warns users.
        #
        # EXECUTIVE CALL (user, 2026-09-18): ``messages`` is the stamped buffer view, not the
        # wire form. The wire projection is backend-keyed (``BaseLLM.extra_message_keys``),
        # so a committed list with it baked in would vary with the backend from an
        # identical buffer and identical edits; and the one field that changed shape
        # between its Enter and its Send would be this one — ``inputs`` and ``code`` do not,
        # and ``message_edits.drops[].message`` on this very event is stamped. ``Agent._to_wire``
        # projects at the provider call instead.
        object.__setattr__(self, "messages", tuple(self.messages))


@dataclass(frozen=True)
class LLMQueryComplete(Event):
    """Fired when the query's raw response exists — the response control boundary.

    Fires once per completed query, with the response the producer made: the provider's
    return, or the response a :class:`SupplyLLMResponse` supplied at :class:`LLMQuerySend`.
    It does not fire when the query was aborted before a response existed, or when the
    call raised (no raw response — only :class:`LLMQueryExit` fires then, with the failure).

    This is the earliest point a hook can decide "this run must not continue" on the basis
    of what came back — an unpriceable cost, a policy-violating completion, a provider
    degradation. The query itself is already finished and is never un-done: an abort
    here stops the turn *after* the response, before the agent acts on it, so the code the
    model just proposed is never executed.

    Allowed effects:

    - :class:`ModifyLLMResponse` — transform the response (content and/or token/cost fields) the
      agent then acts on. The transform is authoritative: the modified response is what the agent
      parses into its next turn and what :class:`LLMQueryExit` observers + cost accounting see.
    - :class:`Abort` — abort the invoke on the basis of the response.

    Attributes:
        response: The *raw* ``LLMResponse`` the producer returned (pre-transform) — content, token
            counts, and cost. A :class:`ModifyLLMResponse` composed here transforms it; the
            post-transform response is what :class:`LLMQueryExit` carries.
        iteration: The agent-loop iteration this query belongs to.
    """

    # Transform here, carry the transformed value on the Exit event: the #906 pattern.
    # This is the response-transform boundary — the ``ModifyExecResult`` analogue for responses,
    # via ``ModifyLLMResponse`` (the "modify slot" this event long reserved but left empty until a
    # consumer motivated it). The minefield the old comment warned of — "the observed exit and the
    # acted-on response must agree" — is resolved by making the transform authoritative and
    # single-sourced: ``span_llm_query`` writes the merged response back onto the span, so
    # ``LLMQueryExit`` and the content the agent parses read the *same* post-transform object (see
    # ``_resolve_llm_modify_results`` and ``Agent._query_with_messages``). ``LLMQueryComplete.response``
    # itself stays the *raw* response, mirroring ``InvokeComplete.result`` (hooks transform what they
    # observe here; Exit carries the result).
    #
    # ``BudgetPool``'s uncosted-model backstop does NOT live here: it runs on
    # ``on_llm_query_exit``'s ``Completed`` arm, so it reads the cost as *booked* — after any
    # ``ModifyLLMResponse`` re-price — which is what lets a hook supply the cost for a turn whose
    # backend promised one and then reported ``None``
    # (``test_modify_response_reprice_rescues_an_uncosted_model``). Moving it back here would read
    # the raw pre-transform response and break that rescue.

    response: LLMResponse
    iteration: int


@dataclass(frozen=True)
class LLMQueryExit(Event):
    """Fired when the LLM-query span closes, with the query's outcome.

    ``outcome`` is the tagged union :data:`LLMQueryOutcome`: :class:`Completed` carries the
    ``LLMResponse`` the query produced (``outcome.result``) — including a response supplied
    by :class:`SupplyLLMResponse` (no call is made, but the query completes normally);
    :class:`Aborted` means a hook's :class:`Abort` at one of this query's control stages
    ended the invoke (``outcome.exception`` carries the abort's error, which propagates
    after the event fires); :class:`Failed` carries the exception the call (or the span's
    own machinery) raised (``outcome.exception``; it still propagates after the event
    fires). The event fires on **every** path once the query span opened — an aborted
    query closes ``Aborted``, it no longer skips the exit.

    Allowed effects — narrow: a hook may return :class:`Abort` (and only ``Abort``), on **every**
    arm. On the :class:`Completed` arm this stops the run on the basis of the **final,
    post-transform** response — the thing :class:`LLMQueryComplete` can't see, since a
    ``ModifyLLMResponse`` there composes after every Complete hook has run — firing before the agent
    acts on the response. On the :class:`Aborted` / :class:`Failed` arms a terminating exception is
    already unwinding, so an abort here cannot *replace* it: it is folded *into* it via an
    ``ExceptionGroup`` (the in-flight exception stays present, so ``is_fatal`` still governs how far
    the combined error travels). Any effect other than ``Abort`` raises
    :class:`~jaz.exceptions.InvalidEffectError` on the ``Completed`` arm and is dropped with an
    error log on the abnormal arms (a loud raise there would replace the exception that must win).
    To transform the response (rather than abort on it), use :class:`ModifyLLMResponse` at
    :class:`LLMQueryComplete`.

    Timing: like every event, this exit carries only :attr:`~jaz.hooks.events.Event.timestamp`
    (its emission time); a span's duration is ``Exit.timestamp - Enter.timestamp``.

    Attributes:
        outcome: How the span ended, with its payload — match on the variant
            (``Completed[LLMResponse] | Aborted | Failed``).
        iteration: The agent-loop iteration this query belongs to.
    """

    # #892 settled the abnormal-arm rule: an abort raised while an exception is already
    # unwinding folds into that exception rather than replacing it.
    #
    # No time fields here: the former ``start_time``/``end_time`` (interval-on-record,
    # #1011) and ``call_start_time``/``call_end_time`` were replaced by the base
    # ``Event.timestamp`` — see its field comment in hooks/_base.py for the decision record
    # (uniform emission stamp; intervals are consumer arithmetic; the call fields had no
    # consumer after #1159).
    #
    # ``outcome`` has NO default on purpose: the pure tagged union replaced the former
    # ``response`` + ``outcome`` + ``exception`` field triple (whose parallel defaults let a
    # constructor produce an incoherent event, e.g. COMPLETED with response=None), so every
    # construction site must say how the span ended.
    #
    # No edits here: :class:`LLMQuerySend` carries them pre-resolved as ``message_edits``, which
    # is where their sole consumer (``TrajectoryRecorder``) reads them. ``Send`` owns the final
    # input; ``Exit`` owns the final outcome.

    outcome: LLMQueryOutcome
    iteration: int


@dataclass
class LLMQueryContext(ExecutionContext):
    """Context for LLM query *enter* events.

    Hooks can:
    - Abort the invoke via :class:`Abort` — this is the always-present per-turn point
      where loop/budget hard-stops live; the composed exceptions are collected in
      ``abort_errors`` and ``span_llm_query`` raises the combined exception (before the
      LLM call; the span closes ``Aborted``).
    - Edit the messages before the query via :class:`DropMessages` / :class:`AddMessages`, each
      *transient* (a per-query view) or *persistent* (edits the carried-forward buffer);
      composition is order-independent (see :class:`MessageEdits`).

    Response supply is NOT here: :class:`SupplyLLMResponse` composes at
    :class:`LLMQuerySend` (see :class:`LLMQuerySendContext`).
    """

    message_edits: MessageEdits = field(default_factory=MessageEdits)
    # Exceptions from ``Abort`` effects emitted at ``LLMQueryEnter``. ``span_llm_query``
    # raises their combination (kept named for the exceptions they carry).
    abort_errors: list[Exception] = field(default_factory=list)


@dataclass
class LLMQuerySendContext(ExecutionContext):
    """Context for LLM query *send* events — the supply boundary.

    Hooks can **supply** the response via :class:`SupplyLLMResponse` (the provider call is
    skipped and the supplied response is used, ``supplied_response``) or abort via
    :class:`Abort` (``abort_errors``). Composition of the supply is order-independent:
    identical overrides compose to one; two distinct overrides raise
    :class:`~jaz.exceptions.LLMResponseConflictError`.
    """

    supplied_response: LLMResponse | None = None
    abort_errors: list[Exception] = field(default_factory=list)


@dataclass
class LLMQueryCompleteContext(ExecutionContext):
    """Context for LLM query *complete* events — the response-transform boundary.

    The LLM-query twin of ``REPLExecCompleteContext``: hooks can transform the response via
    :class:`ModifyLLMResponse` (``modify_effects`` — ``span_llm_query`` resolves them onto the raw
    response and writes the result back on the span, so ``LLMQueryExit`` and the agent read the
    post-transform object), or abort via :class:`Abort` (``abort_errors`` — ``span_llm_query`` raises
    their combination; the span closes ``Aborted``, superseding any transform).
    """

    # ``modify_effects`` holds the ``ModifyLLMResponse``s; ``abort_errors`` the Abort exceptions.
    # The dispatcher's ``_compose_llm_query_complete`` is the only writer.
    modify_effects: list[ModifyLLMResponse] = field(default_factory=list)

    # Why this boundary exists at all: the response's first-existence point used to be
    # `LLMQueryExit`, which grew an Abort channel because denying it forced hooks to stash
    # state on themselves and re-decide at the *next* `LLMQueryEnter` (see
    # `BudgetPool`'s former `_uncosted_model` field). Be precise about what that deferral
    # cost, because it is easy to overstate: it was NOT an extra query — an abort at the next
    # enter returns before the call, and the turn that revealed the condition was already
    # spent. What it cost is an *execution*: between the two events the agent runs the code
    # from that very turn. For a policy-violating completion that is the whole ballgame. A
    # hook field whose only job is to defer a decision to a later event is the tell that an
    # effect site is missing. The pipeline gives the site a proper home: `Complete` is the
    # response-control stage. (`Exit` is not pure observation either: it accepts `Abort` on every
    # arm, which is where `BudgetPool`'s backstop lives. What no `Exit` accepts is a transform.)
    abort_errors: list[Exception] = field(default_factory=list)


class LLMQuerySpan:
    """Span for LLM query.

    Usage:
        with dispatcher.span_llm_query(enter_event) as span:
            span.send(shown_messages)      # fires LLMQuerySend; suppliers compose
            if span.supplied_response is not None:
                response = span.supplied_response
            else:
                response = llm.complete(...)
            span.complete(response=response)

    An ``Abort`` composed at any of the query's control stages never surfaces on the
    span: the ``span_llm_query`` context manager raises the carried exception (from the
    ``with`` statement at enter, from ``send()`` at the committed-input veto, from the
    block's close for a Complete-time one), after closing the span with an ``Aborted``
    outcome. The former per-stage abort fields (``abort`` / ``send_abort`` /
    ``complete_abort``) were the legacy laundering channel and are gone.
    """

    def __init__(self, ctx: LLMQueryContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        self._response: LLMResponse | None = None
        # No _iteration: the span deliberately has no iteration channel of its own — the
        # CM reads it off the enter event for every event it constructs, so one query's
        # events can never disagree about which iteration they belong to (the removed
        # complete(iteration=...) parameter was a second, mismatchable source;
        # consistency audit C3). Neither REPLExecSpan nor InvokeSpan ever had one.
        # Emitter for LLMQuerySend, installed by ``span_llm_query`` (the span is pure data
        # and must not import the dispatcher; the CM closes over the enter event + composed
        # edits so ``send()`` needs only the committed list, and stores the Send-composed
        # supply on ``supplied_response``). None on a hand-built span.
        self._emit_send: Callable[[list[MessageDict]], None] | None = None
        # Response supplied by a Send-composed ``SupplyLLMResponse`` (e.g. ``TrajectoryReplay``), if
        # any. When set, the caller uses it in place of calling the provider; the query
        # still completes normally with it. Set by ``send()``.
        self.supplied_response: LLMResponse | None = None

    def send(self, shown_messages: list[MessageDict]) -> None:
        """Fire :class:`LLMQuerySend` with this query's committed message list.

        Called by the agent loop once the enter-time edits have been folded into the list
        actually handed to the result producer — i.e. exactly when the input commits. A
        Send-composed supply is stored on :attr:`supplied_response`; a Send-composed
        :class:`Abort` raises its carried exception from this call (the span closes
        ``Aborted``).

        Raises:
            RuntimeError: If the span was not created by ``span_llm_query``.
        """
        if self._emit_send is None:
            raise RuntimeError(
                "LLMQuerySpan.send() requires a span created by span_llm_query"
            )
        self._emit_send(shown_messages)

    def complete(
        self,
        *,
        response: LLMResponse,
    ) -> None:
        """Complete span with the LLM response.

        Args:
            response: The LLMResponse (content + token counts + cost) the client returned.

        Raises:
            RuntimeError: If span was already completed
        """
        if self._completed:
            raise RuntimeError("Span already completed")
        self._response = response
        self._completed = True

    @property
    def is_completed(self) -> bool:
        """Whether the span has been completed."""
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

    def set_response(self, response: LLMResponse) -> None:
        """Replace the completed response with a ``LLMQueryComplete``-composed override.

        Called only by ``span_llm_query`` after emitting ``LLMQueryComplete`` and resolving any
        ``ModifyLLMResponse`` (mirrors ``InvokeSpan.set_final_result``). The caller then reads the
        (possibly overridden) response via :meth:`get_response` *after* the span closes — the
        object ``LLMQueryExit`` carries and the agent parses, so the transform is authoritative.
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        self._response = response


# Retry lives here with the rest of the LLMQuery* family (rather than a standalone
# module) — it is a query-scoped event, and co-locating keeps the whole family in one
# place. Unlike Enter/Exit it is a simple observational event with no span / enter-exit
# pair: it fires each time an in-flight LLM call fails and tenacity is about to retry it.


@dataclass(frozen=True)
class LLMQueryRetry(Event):
    """Fired when an LLM call fails and is about to be retried.

    **Not a per-turn event.** It fires once per retry attempt, so a turn whose first attempt
    succeeds never fires it, and a turn that retries three times fires it three times.

    This event is purely observational: no effect other than
    :class:`~jaz.hooks.effects.BlackboardWrite` is valid here.

    Attributes:
        attempt_number: Which attempt just failed, counting from 1.
        exception: The error that caused the retry.
        wait_seconds: How long the retry will wait before the next attempt.
        iteration: The agent-loop iteration the call belongs to.
    """

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
