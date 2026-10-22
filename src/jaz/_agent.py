import functools
import itertools
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._invoke import Prehook

from tenacity import RetryCallState

from ._config import Config
from ._messages import MessageDict
from ._provenance import (
    MessageKind,
    MessageProvenance,
    set_provenance,
    to_wire_messages,
)
from ._subinvoke import Subinvoke
from ._tokens import set_tokens, tokens_of
from .exceptions import (
    MissingDropTargetError,
    REPLInputConflictError,
    _JazInternalError,
)
from .hooks import Hook
from .hooks._dispatcher import HookDispatcher
from .hooks.effects import _apply_message_edits
from .hooks.events import (
    InvokeEnter,
    LLMQueryEnter,
    LLMQueryRetry,
    REPLExecEnter,
)
from .hooks.events._invoke import InvokeContext
from .hooks.events._llm_query import LLMQuerySpan, MessageEdits
from .hooks.events._repl_execution import REPLExecSpan
from .llm._base import BaseLLM
from .llm._types import LLMResponse
from .protocol import BaseProtocol
from .repl._base import BaseREPL
from .repl._types import (
    Continue,
    ExecResult,
    Raise,
    Return,
)


class Agent:
    def __init__(
        self,
        prehook: "Prehook | None" = None,
        *,
        config: Config,
        local_hooks: tuple[Hook, ...] = (),
    ) -> None:
        """Construct an agent. ``config`` is REQUIRED — there is no ambient fallback.

        The REPL, LLM and protocol all come off ``config``; there is nothing else to pass.
        To run against the process-wide configuration, pass it at the call site:
        ``Agent(config=jaz.get_config())``. That read is ambient, so it differs from the config an
        invoke runs under whenever a per-call override is in scope.
        """
        # Requiring `config` is #826 §3 (executive call): the `None`-defaulting shape silently read a
        # different config than the caller's, and it was the read shape #826 set out to retire — it
        # was the last ambient `get_config()` read in the codebase. Two other resolutions were
        # considered and rejected:
        #
        #   - *Resolve the fallback as if the Agent were a top-level invoke*. Rejected: an Agent
        #     has no recursion depth. It carries an optional `prehook` and is just an object a
        #     caller can drive at any level, so treating it as the top is an invented fact.
        #   - *Keep the ambient fallback* and document it as the one legitimate ambient read.
        #     Rejected: it leaves an implicit ambient read on a public constructor for no
        #     ergonomic gain.
        #
        # Requiring the argument resolves the tension instead of picking a side of it: the caller
        # decides, so no depth has to be invented and no ambient read remains in this path. The
        # framework never relied on the fallback anyway — `_build_invoke_setup` has always passed
        # the depth-resolved `config` explicitly (it is the only in-tree construction site), so this
        # was unreachable-in-practice surface, not a capability taken away. It is nonetheless a
        # breaking change to the public `jaz._agent.Agent` signature; direct construction was exported but
        # never documented (no example in README/docs/design).
        #
        # `config` is keyword-only; `prehook` is the first positional.
        # Taken as given: a `Config` is complete by construction, so there is nothing to resolve
        # and no missing-LLM case to handle here. The invoke path folds one at the top of the
        # invoke; a caller constructing an Agent directly passes `jaz.get_config()`, which
        # refuses there if nothing is configured.
        self.config: Config = config
        # The components come straight off the config — there is nothing left to build here.
        # This used to be three resolution blocks (a tag-or-instance branch for the backend, the
        # same for the protocol, and a language lookup for the REPL), each turning a
        # `{tag, params}` description into an object and each able to fail at `Agent.__init__`
        # for a config error written much earlier. Config now holds the configured components,
        # so a bad one raised at the `configure`/`ConfigOverride` call site that wrote it.
        # There is no `repl` parameter to reconcile with `config.repl` any more. It used to take
        # a language STRING, checked here against the configured REPL's declared name — a check
        # that existed only because the same fact was expressed twice. `invoke` derived the string
        # from `config.repl` in the first place, so the disagreement it policed was one the caller
        # had to construct deliberately.
        # The configured REPL is a stateless *template*: `initialize` returns a fresh per-invoke
        # state and the template holds no per-invoke state itself, so one config safely serves
        # many concurrent invokes (#1056, #1058).
        #
        # `BaseREPL[Any]`, and the threaded state is annotated `object`: the Agent is NOT generic
        # in the state type. It cannot usefully be — the REPL arrives from `Config`, which is
        # populated with whatever REPL instance the host configured, so no call site here has
        # a statically known state type to propagate. Making `Agent` (and therefore `Config`)
        # generic would push a type parameter through the whole public config surface to recover
        # a guarantee nothing could supply. Nothing is lost by threading it opaquely: an Agent
        # holds exactly one REPL and exactly one state built by that REPL's own `initialize`, so
        # there is no second state for the wrong one to be. The static safety the generic buys is
        # at the *implementation* boundary — `PythonREPL.exec` will not accept another REPL's
        # state — which is where the mix-up could actually happen.
        self.repl_template: BaseREPL[Any] = self.config.repl
        self.llm_client: BaseLLM = self.config.llm
        self.protocol: BaseProtocol = self.config.protocol
        self.prehook = prehook
        # Always a per-invoke dispatcher instance (never the shared singleton): the
        # dispatcher now holds this invoke's seeded blackboard (see seed_blackboard()
        # in invoke()/ainvoke()), which is per-call state that a shared singleton
        # could not carry. Local hooks (if any) ride along on the same instance.
        # See HookDispatcher class docstring for why per-instance beats a second
        # contextvar. A HookDispatcher is a trivial object, so per-invoke allocation
        # is cheap, and every field on a dispatcher is per-invoke state, so there is no
        # shared instance to reuse.
        self.dispatcher = HookDispatcher(local_hooks=local_hooks)
        # Per-query side-channel: the message edits composed at the last LLMQueryEnter,
        # stashed by _compose_shown_messages and applied to the carried-forward buffer by
        # _apply_persistent_message_edits (the loop resets it before each query). Carries
        # the composed edits out of the (mock-patched) _query_with_messages without
        # changing its signature.
        self._pending_message_edits: MessageEdits | None = None

        # Per-query side-channel (same idiom as _pending_message_edits): the full LLMResponse from
        # the last *real* query, stashed by _record_llm_response so _finalize_iteration can hand
        # the whole object to repl.update_repl_history without changing what
        # _query_with_messages returns (still the content str — so the many tests that patch
        # _query_with_messages to return a bare string keep working). Reset to None before each
        # query; on the mocked path it stays None and _llm_response_to_record synthesizes a
        # minimal LLMResponse from the content string, keeping the default record identical.
        self._last_llm_response: LLMResponse | None = None

        # Per-query side-channel (same idiom as the two above): the last query's committed
        # shown list, exactly what _compose_shown_messages returns — the buffer dicts (by
        # reference) plus any transient hook additions. _stamp_back_tokens needs it because a
        # token-native backend's TurnRecord.sent is index-aligned with this list, and only
        # these dict objects (not the wire copies _to_wire makes at the provider call) live
        # on in the buffer. Reset to None before each query; stays None on the mocked path.
        # TODO(#1161): this is the third per-query side-channel. Collapsing all three into one
        # record threaded as a local is blocked on the test seam, not on the field count: all
        # three are produced inside _query_with_messages and read outside it, and that method is
        # patched by the suite in ~150 places, so widening what it returns is a test migration
        # rather than a refactor. Half-done for this one: the stash IS _compose_shown_messages's
        # return value now (the wire projection moved to _to_wire), so the data dependency that
        # justified a separate channel is gone and only that plumbing remains.
        self._pending_shown_messages: list[MessageDict] | None = None

    def _reset_per_query_state(self) -> None:
        """Clear the per-query side-channels before a query runs."""
        # A fourth channel goes here, not at the two call sites (the sync and async loops).
        # Why these live on the instance at all is on the fields above.
        self._pending_message_edits = None
        self._last_llm_response = None
        self._pending_shown_messages = None

    def _make_llm_retry_fn(
        self,
        retry_fn: Any,
        messages: list[MessageDict],
        invoke_id: str,
        iteration: int,
        depth: int,
    ) -> tuple[Any, LLMQueryEnter]:
        """Bind the Config retry params + a hook-dispatching before-sleep onto
        ``retry_fn`` (``LLM._complete_with_retry`` / ``_acomplete_with_retry``) and return (fn, enter_event).
        All retry logic lives on the LLM; the Agent only supplies the parameters and
        the ``on_retry`` callback."""

        def log_retry_attempt(retry_state: RetryCallState) -> None:
            exception = retry_state.outcome.exception() if retry_state.outcome else None
            wait_seconds = (
                retry_state.next_action.sleep
                if retry_state.next_action is not None
                else 0.0
            )
            self.dispatcher.process_llm_query_retry(
                LLMQueryRetry(
                    config=self.config,
                    invoke_id=invoke_id,
                    attempt_number=retry_state.attempt_number,
                    exception=exception
                    if isinstance(exception, Exception)
                    else Exception(str(exception)),
                    wait_seconds=wait_seconds,
                    iteration=iteration,
                    depth=depth,
                )
            )

        # The Agent supplies `on_retry`, which closes over this invoke's id/iteration to
        # dispatch LLMQueryRetry — per-call observability rather than policy.
        #
        # No retry settings are threaded: the backend's `retry_policy` is the only source, and
        # `_complete_with_retry` reads it.
        #
        # This used to forward `retry_*` out of `llm.params` as per-call overrides, but only for
        # an instance tag: a tag-built client got them through the config boundary, while a pre-built
        # one never went through it, so the config keys would otherwise have been silently
        # ignored for instance tags. With the configured backend itself in the config there is
        # no second place for a retry setting to live, so there is nothing to reconcile.
        wrapped = functools.partial(retry_fn, on_retry=log_retry_attempt)

        enter_event = LLMQueryEnter(
            config=self.config,
            invoke_id=invoke_id,
            messages=messages,
            iteration=iteration,
            depth=depth,
        )
        return wrapped, enter_event

    def _record_llm_response(
        self,
        llm_response: LLMResponse,
        span: LLMQuerySpan,
    ) -> str:
        # This signature has shed iteration/depth (dead after #1156 C3) and the
        # start/end call-timing pair (dead with the Event.timestamp redesign — timing is
        # emission-stamped on every event; no per-call measurement remains). Dead
        # threaded parameters are exactly what a stale-based restack can silently
        # resurrect, hence the trail.
        """Track token usage/cost, complete the span, and return response content."""
        # Complete the span with the response object. Cost tracking + enforcement is
        # handled by BudgetPool (opt-in) observing LLMQueryExit.
        span.complete(response=llm_response)
        # Stash the whole object for the history record (see the field's init comment);
        # returning only .content keeps _query_with_messages's content-str contract intact.
        self._last_llm_response = llm_response
        # ``or ""`` is the single point where the wire's nullable content becomes a plain str, so
        # nothing downstream of this line has to carry a ``| None`` for it. ``LLMResponse.content``
        # stays ``str | None`` deliberately: it mirrors the provider schema (OpenAI types
        # ``ChatCompletionMessage.content`` as ``Optional[str]``, and the live LiteLLM extraction
        # defaults to None at every getattr hop), so the honesty lives in the provider layer and
        # the convenience lives here.
        #
        # What null content actually means is lost by this coercion, and that is the accepted
        # cost. It is real per the API contract — a refusal (the text is on
        # ``message.refusal``, which nothing reads), a reasoning model exhausting its budget with
        # ``finish_reason="length"``, a content filter, or an empty ``choices`` — but vanishingly
        # rare in practice: zero occurrences across ~432k files / 37GB of recorded runs in
        # ``logs/``, against 8476 parse errors of other kinds in the same corpus. So it is
        # normalized rather than routed, and an empty completion reaches ``parse`` as ``""``,
        # which rejects it the same way it rejected ``None``.
        #
        # This is a stopgap, not the design. Routing these four causes apart — retry the
        # transient ones (ideally perturbing the request, since a model at temperature 0 that
        # returns nothing returns nothing again), surface the agent-actionable ones with their
        # real reason — is #697's job, and exposure rises now that LiteLLM is the only backend
        # and the eval default is a reasoning model.
        return llm_response.content or ""

    def _query_with_messages(
        self,
        messages: list[MessageDict],
        *,
        iteration: int,
        depth: int,
        invoke_id: str,
    ) -> str:
        # An Abort composed at any of the query's stages raises its carried exception out
        # of this method (from the `with` statement for an enter-time hard-stop, from
        # span.send for the committed-input veto, from the block's close for a
        # Complete-time one) — the span CM closes the span Aborted first, and the invoke
        # span classifies the same exception Aborted on its own unwind (see
        # HookDispatcher._unwind_outcome). Content is coerced to ``str`` at
        # ``_record_llm_response`` (an empty completion projects to ``""``, #1141), so this
        # returns a plain ``str`` — never ``None``, never a laundered ``Raise``.
        complete_fn, enter_event = self._make_llm_retry_fn(
            self.llm_client._complete_with_retry,
            messages,
            invoke_id,
            iteration,
            depth,
        )
        with self.dispatcher.span_llm_query(enter_event) as span:
            shown = self._compose_shown_messages(span, messages)
            # LLMQuerySend — fires iff the input committed. ``shown`` IS the committed
            # input (the post-edit list; the result producer gets its ``_to_wire`` projection), so Send fires here
            # unconditionally: on the live-call path AND the supplied path below (the
            # commit is real on both; only the producer differs — Send is where a supplier
            # like TrajectoryReplay answers, with the committed list in view). It does not fire on
            # an enter-time abort — that raises out of the `with` statement before this
            # body runs, so no input ever commits (the span still closes, Aborted).
            span.send(shown)
            # Send-supplied response (e.g. TrajectoryReplay): the provider call is skipped.
            if span.supplied_response is not None:
                llm_response: LLMResponse = span.supplied_response
            else:
                # A provider error still raised after the retry policy propagates from
                # here: LLMQueryExit(Failed) fires from the span CM, the invoke span
                # closes Failed on the unwind, and the caller receives the exception.
                # TODO(#697): deliberately downgrading *selected* failures to a
                # recoverable Continue remains a legitimate future loop policy (as loop
                # policy — not the blanket Raise-conversion #687 used to do here).
                llm_response = complete_fn(messages=self._to_wire(shown))
            result = self._record_llm_response(llm_response, span)
        # Span closed: a Complete-stage ModifyLLMResponse may have replaced the response
        # (authoritative — see _resolve_llm_modify_results). If so, re-read it: its content is what
        # the agent parses next, and the same object is what LLMQueryExit carried and BudgetPool
        # booked, so content / observers / cost all agree. The identity check leaves the common
        # no-transform path (same object) untouched, and covers the supplied-response path too.
        final_response = span.get_response()
        if final_response is not llm_response:
            self._last_llm_response = final_response
            result = final_response.content or ""
        return result

    def _compose_shown_messages(
        self, span: LLMQuerySpan, messages: list[MessageDict]
    ) -> list[MessageDict]:
        """Fold this query's composed message edits into the list shown to the model.

        Shared by the sync and async query paths (only the awaited LLM call differs).
        ``messages`` is the per-query snapshot (the buffer plus any trailing instruction
        addition). The returned ``shown`` is all (persistent + transient) edits folded
        over it — subsuming the old ``DropMessages``-only view-filter, and byte-identical
        to ``messages`` when no edits were emitted.

        Persistence can't be applied here (this snapshot may be a transient copy, not the
        loop's real buffer), so the composed edits are stashed on
        :attr:`_pending_message_edits`; the loop applies any *persistent* ones to the
        buffer itself via :meth:`_apply_persistent_message_edits`. The loop resets the
        attribute before each query, so when the query is mocked away in tests it simply
        stays empty (no persistence).

        The returned list is the committed message list itself — the buffer's stamped dicts
        (by reference) plus the folded edits — and is what ``LLMQuerySend`` carries. The wire
        projection is NOT applied here: it is backend-specific (``BaseLLM.extra_message_keys``
        decides which sidecar keys survive), so a committed list that had it baked in would
        change with the backend from an identical buffer and identical edits, and would differ
        in shape from ``LLMQueryEnter.messages``. :meth:`_to_wire` projects it at the provider
        call instead — still one strip point, shared by the sync and async paths.
        """
        edits = span.ctx.message_edits
        self._pending_message_edits = edits
        shown = _apply_message_edits(messages, edits.all_drops, edits.all_adds)
        # Stashed for stamp-back: `shown` holds the buffer's own dicts by reference (the fold
        # never copies), while the wire projection at the provider call copies every
        # message — so this list is what stamp-back can mutate meaningfully.
        self._pending_shown_messages = shown
        return shown

    def _to_wire(self, shown: list[MessageDict]) -> list[MessageDict]:
        """Project the committed message list to the configured backend's wire form.

        The single **provider-egress** point: strips every reserved sidecar key the configured
        backend has not listed in ``BaseLLM.extra_message_keys``, so internal per-message keys
        never reach a model that doesn't read them. Applied at the provider call, AFTER
        ``LLMQuerySend`` has observed the committed list — the event carries the stamped
        buffer view, the provider gets its projection.
        """
        return to_wire_messages(shown, keep=self.llm_client.extra_message_keys)

    def _apply_persistent_message_edits(
        self, messages: list[MessageDict], iteration: int
    ) -> None:
        """Apply the last query's *persistent* message edits to the carried-forward buffer.

        ``messages`` is the loop's real buffer, and the edit indices anchor to exactly it:
        prompt additions now flow through the message-edit fold rather than being baked into
        a longer query snapshot ahead of it (``AddInstructionPrompt`` was removed in favour
        of ``AddMessages``), so the query snapshot coincides with the buffer and no
        trailing transient instruction has to be stripped. Fold the persistent subset over
        the buffer. A no-op unless a hook emitted a persistent ``DropMessages`` /
        ``AddMessages`` (context compaction).

        Inserted messages are stamped with ``ADDED`` provenance after the fold, over the
        persistent adds only: those are the dicts the fold put in the buffer, and an id means
        "this is a buffer message", which a transient add never becomes.
        """
        # The stamp used to be placed after the fold for a second, stronger reason: the fold
        # computed a ``json.dumps`` sort key over ``AddMessages.messages``, which a provenance
        # value is not serializable into. That sort key is gone (same-slot adds now apply in hook
        # order), so only the reason above still holds. It is enough on its own, but the
        # placement is no longer forced: the fold inserts these dicts by reference, so stamping
        # them before it would reach the same objects.
        #
        # A consequence worth knowing, because a reader of the trace meets it: at
        # ``LLMQuerySend`` -- where ``TrajectoryRecorder`` writes the turn's edit records -- no
        # add is stamped yet, so an added message is recorded without an id and is identified by
        # its record's position until the next enter.
        # Persistent edits are applied against the buffer, not the query snapshot, so the two coincide
        # for the next query — the invariant #660 established for `AddMessages`.
        edits = self._pending_message_edits
        if edits is None or not edits.has_persistent:
            return
        messages[:] = _apply_message_edits(
            messages, edits.persistent_drops, edits.persistent_adds
        )
        for add in edits.persistent_adds:
            for message in add.messages:
                set_provenance(
                    message,
                    MessageProvenance(
                        MessageKind.ADDED,
                        iteration=iteration,
                        persistent=True,
                    ),
                )

    @staticmethod
    def _stamp_seed_provenance(messages: list[MessageDict]) -> None:
        """Stamp every message in the initial seed as ``MessageKind.SEED``.

        ``messages`` is exactly ``render_initial_message_list``'s output, taken before any
        turn has run, so the whole list *is* the seed by definition — not just the system +
        user prompt. Stamping all of it (rather than picking out roles by position) is both
        simpler and correct for any protocol: ``CodeOnlyProtocol`` renders ``[system, user]``,
        but a custom protocol may seed extra context (e.g. few-shot ``assistant`` examples),
        and those are just as much the seed. Role no longer needs to disambiguate a kind
        (that was only for the former split SYSTEM/INPUT_PROMPT kinds); a consumer that wants
        the system prompt vs the user prompt reads the surviving message's ``role``."""
        for message in messages:
            set_provenance(message, MessageProvenance(MessageKind.SEED))

    def _stamp_back_tokens(
        self, messages: list[MessageDict], assistant_message: MessageDict
    ) -> None:
        """Copy a token-native response's per-message token stamps onto the buffer.

        A no-op unless the last query's response carries ``tokens`` (a ``TurnRecord``
        from a token-native backend such as ``SGLangLLM``): text backends, mocked
        queries and replay overrides leave the buffer untouched.
        """
        # This is the "stamp-back" edge of the token-rollout design (see jaz._tokens'
        # module comment for the rationale): the backend mints one frozen
        # TokenStamp per wire message inside complete(); here the agent attaches each
        # stamp to the buffer dict it came from, so the *next* send reuses the exact ids
        # (no re-render, no drift) and the stamps ride through edits/compaction by
        # reference. The agent treats stamps as opaque blobs and stays token-blind.
        #
        # Positional contract: record.sent[i] describes wire message i, and the wire list
        # is an element-wise projection of the stashed shown list — so
        # zip(shown, sent) pairs each stamp with the dict that produced it.
        # strict=True fails loud on a length mismatch — a genuine one-stamp-per-wire-message
        # violation, i.e. a backend bug worth raising on.
        #
        # A same-length misalignment is the dangerous one, and the content-guard alone does not
        # catch it: the backend's re-render check compares `content_text` only, while a stamp's
        # ids also depend on the ROLE, which the chat template renders as a header. So two
        # same-content messages that swapped roles pass the content-guard and keep ids rendered
        # under the wrong role. That is why the stamp-back below checks role as well as identity
        # and skips rather than stamps on a mismatch; `test_role_mismatch_skips_the_stamp_rather_
        # than_mis_stamping` pins it. Where the guard does hold, the cost is downstream only —
        # _extends (rollout export) stops chaining, so to_spans() splits there: degraded
        # export quality, not corrupted training ground truth.
        #
        # The identity check against the buffer is what skips (a) transient hook-added
        # messages — hook-owned dicts that are not in the buffer; their stamps live only
        # in the turn record, which is exactly their lifetime — and (b) messages a
        # persistent edit just dropped (_apply_persistent_message_edits runs before this).
        # The exit-abort path skips _record_response_and_parse entirely, so a terminal
        # turn is never stamped back — harmless: there is no next send to reuse it.
        response, shown = self._last_llm_response, self._pending_shown_messages
        if response is None or response.tokens is None or shown is None:
            return
        record = response.tokens
        # assistant_message is the turn _record_response_and_parse just appended and passes in
        # explicitly (rather than this method reaching for messages[-1] and assuming the caller
        # appended it last); its stamp is the record's sampled segment.
        set_tokens(assistant_message, record.sampled)
        # id()-as-key is safe only because `messages` and `shown` both hold strong refs to every
        # element for this whole call, so no dict can be collected and its address recycled while
        # buffer_ids is live. Contextual, not an intrinsic property of id().
        buffer_ids = {id(m) for m in messages}
        for message, stamp in zip(shown, record.sent, strict=True):
            # stamp.role == the message's role is a cheap extra alignment guard: strict=True
            # catches a length change but not a reorder or an equal-length merge/insert a
            # normalizing backend could introduce; a role mismatch means the positional contract
            # broke, so we skip rather than mis-stamp (it re-renders next turn — loud at export,
            # not silent). Deliberately NOT content_text: a hook mutating event.messages at
            # LLMQueryExit (fires before stamp-back) would trip that spuriously.
            if (
                id(message) in buffer_ids
                and stamp.role == message.get("role")
                and tokens_of(message) is not stamp
            ):
                # Unconditional overwrite with what was actually sent: on the reuse path
                # `stamp` IS the message's existing stamp (skipped above), so this only
                # fires for fresh renders — first sends and stale-stamp re-renders. A
                # stale stamp (hook edited the content after stamping) MUST be replaced,
                # or every later send would re-render under yet another message_id and
                # export would split at every turn instead of once at the edit.
                set_tokens(message, stamp)

    def _record_response_and_parse(
        self,
        messages: list[MessageDict],
        response_content: str,
        invoke_id: str,
        iteration: int,
        depth: int,
        state: object,
    ) -> REPLExecEnter:
        """Record the assistant response and decode it into an execution plan.

        Shared by the sync and async iteration paths — which differ *only* in how the
        response is fetched and how the code is executed; everything around that is
        identical and lives here (and in :meth:`_finalize_iteration`) so the two paths
        cannot drift. Appends the assistant message, then parses, and returns the
        ``enter_event``. The parsed code rides on ``enter_event.code`` (its edit target); the code
        that actually *runs* is ``span.committed_code`` — the enter code after any ``InsertCode``
        / ``DeleteCode`` — so the bare parsed string is not returned separately (it would be dead
        at both call sites).

        There is exactly one outcome: ``parse`` has no failure mode, so there is no second
        return shape for a caller to discriminate against.
        """
        assistant_message: MessageDict = {
            "role": "assistant",
            "content": response_content,
        }
        messages.append(assistant_message)
        set_provenance(
            assistant_message,
            MessageProvenance(MessageKind.ASSISTANT, iteration=iteration),
        )
        self._stamp_back_tokens(messages, assistant_message)

        # No try/except: ``parse`` has no recoverable failure mode (see ``BaseProtocol.parse``).
        # It returns its best reading — the message verbatim when it has none, empty code when
        # the message is empty — and malformed input surfaces at exec time as a ``SyntaxError``
        # inside a recoverable ``Continue`` instead. Anything this raises is a bug in a custom
        # protocol and is terminal, which is the same treatment every non-parse exception here
        # always had.
        # ``parse`` takes the whole LLMResponse (#1146), not the extracted content — the same
        # canonical object the history record uses (``_last_llm_response`` on the real query path,
        # a synthesized one on the mock path), so a protocol can read ``.content is None`` /
        # ``.raw_response`` (refusal / finish_reason / tool_calls) rather than a coerced string.
        code = self.protocol.parse(self._llm_response_to_record(response_content))

        enter_event = REPLExecEnter(
            config=self.config,
            invoke_id=invoke_id,
            iteration=iteration,
            code=code,
            # The namespace BEFORE this turn's AddToNamespace/DropFromNamespace: the proposal the
            # enter-time effects edit, read off the REPL here because only the loop holds the
            # state. The event copies it.
            namespace=self.repl_template.get_namespace(state),
            depth=depth,
        )
        return enter_event

    def _finalize_iteration(
        self,
        span: REPLExecSpan,
        response_content: str,
        state: object,
    ) -> ExecResult[Any]:
        """Apply exit-time overrides and hand the turn to the REPL's history record.

        Shared by both iteration paths. ``span.get_final_exec_result()`` folds in any
        Complete-time ModifyExecResult / Abort; the REPL is handed that FINAL (post-override)
        result — so a result-override hook (e.g. BudgetForcing) reaches the REPL history without
        ``exec`` knowing, and a turn whose ``exec()`` an enter-time override skipped is handed over
        too.
        """
        # History must not record a REPL result the REPL never saw (#448), which is why the exit-time
        # and enter-time override paths are kept distinct here.
        #
        # Every turn that reaches here is handed to the REPL exactly once, so ``PythonREPL``, which
        # appends an entry per call, holds one entry per iteration (#719), and no guard is needed
        # to say so: ``response_content``
        # is a ``str`` by the time it reaches ``_llm_response_to_record``, so that always returns
        # an object; an empty completion records ``llm_response=""``. The no-response terminal
        # paths (a failed LLM call, or an ``Abort``) *raise* out of ``do_one_repl_iteration`` and
        # never reach here.
        #
        # The one-entry-per-iteration invariant is not something to lean on for addressing:
        # nothing addresses history *by iteration number*, and anything naming the current turn's
        # entry uses ``[-1]``, which holds without it. This runs before the turn's observation is
        # rendered, so ``[-1]`` is that turn — the property the deleted truncation-advice templates
        # relied on when they pointed the agent at ``__history__[-1].repl_output``, and that any
        # host template restating that pointer still relies on.
        exec_result = span.get_final_exec_result()
        self.repl_template.update_repl_history(
            state, self._llm_response_to_record(response_content), exec_result
        )
        return exec_result

    def _llm_response_to_record(self, response_content: str) -> LLMResponse:
        """The ``LLMResponse`` to record for the current turn.

        Precedence is intentional: the whole ``_last_llm_response`` object — stashed by
        ``_record_llm_response`` on the real query path — is the source of truth (its ``.content``
        text AND token/cost/provider ``extra``). ``response_content`` is the mocked-path fallback:
        tests patch ``_query_with_messages`` to return a bare content string, so
        ``_record_llm_response`` never runs and the object stays ``None``; we synthesize a minimal
        ``LLMResponse`` from the string so the default projection is byte-identical. On the real path
        the two agree by construction (``response_content`` IS this object's ``.content``); the one
        way they could diverge — a future hook rewriting ``response_content`` post-query without
        touching the side-channel — resolves to the raw provider object on purpose (history records
        what the model actually returned, so a content-rewriting hook updates the object, not just
        the string).
        """
        # Always returns a response now: content is coerced to ``str`` at
        # ``_record_llm_response``, and the no-response terminal paths (a failed LLM call, an
        # ``Abort`` at ``LLMQueryEnter``) return in ``do_one_repl_iteration`` before reaching here.
        # This used to have a third branch returning ``None`` for "no response to record", which
        # made the caller's history append conditional; that branch was already unreachable in
        # practice (verified: converting it to an assert produced zero hits across the test suite)
        # and is now unreachable by construction.
        if self._last_llm_response is not None:
            return self._last_llm_response
        return LLMResponse(content=response_content)

    def do_one_repl_iteration(
        self,
        iteration: int,
        depth: int,
        messages: list[MessageDict],
        state: object,
        invoke_id: str,
    ) -> ExecResult[Any]:
        # TODO: Fix static types
        # The query snapshot is the buffer itself — prompt additions arrive as message
        # edits folded at query time (see _compose_shown_messages), not baked in ahead.
        self._reset_per_query_state()
        # No catch around the query: a failed LLM call (e.g. a missing API key surfaced as
        # AuthenticationError, or any error still raised after the retry policy) PROPAGATES
        # — LLMQueryExit(Failed) fired from the span CM, the invoke span closes Failed on
        # the unwind, and the caller still receives the exception. The former #687 block
        # here converted it to a terminal Raise result so the exit boundary would fire;
        # the outcome-union Exit made that laundering unnecessary. An Abort's carried
        # exception (iteration / budget hard-stop, committed-input veto, Complete-time
        # stop) propagates the same way — the aborting spans closed Aborted in the CM.
        response_content = self._query_with_messages(
            messages,
            iteration=iteration,
            depth=depth,
            invoke_id=invoke_id,
        )
        # Apply any persistent message edits (compaction) to the carried-forward buffer.
        self._apply_persistent_message_edits(messages, iteration)
        enter_event = self._record_response_and_parse(
            messages, response_content, invoke_id, iteration, depth, state
        )

        with self.dispatcher.span_repl_exec(enter_event) as span:
            # Apply AddToNamespace then DropFromNamespace to the REPL namespace before this turn's
            # code runs. Adds bind names without touching the prompt — the
            # per-turn namespace counterpart of the invoke-level AddInputs; drops unbind names
            # so a dropped name (e.g. __history__) reads as a genuine NameError. Applied
            # whether or not an override skips exec, so the namespace state is consistent for the
            # next turn either way (#568).
            #
            # DROPS RUN FIRST, mirroring ``_apply_input_effects``. Add-first made re-binding a
            # name impossible even though both this effect's docstring and
            # ``PythonREPL.add_to_namespace`` documented the recipe: pairing a ``DropFromNamespace``
            # with an ``AddToNamespace`` hit the already-bound check while the old value was still
            # there and raised ``REPLInputConflictError``, and on an unbound name the pair bound
            # then immediately unbound. Dropping first makes drop-then-add a re-bind, which is
            # the only way to replace a name a hook does not own.
            #
            # The trade is the same one ``AddInputs``/``DropInputs`` makes: a drop can no longer
            # remove a name another effect adds in the same turn (the add now wins). A drop that
            # outlives an add is the same mechanism that lets a stale drop erase one, and the two
            # cannot both hold in a single pass.
            self.repl_template.drop_from_namespace(
                state,
                span.ctx.namespace_drops,
                allow_missing=span.ctx.namespace_drops_allow_missing,
            )
            self.repl_template.add_to_namespace(state, span.ctx.namespace_adds)
            # REPLExecSend — fires iff the code committed. The namespace deltas above just
            # ran, so the execution's commit (code + committed namespace) is final, and the
            # namespace the REPL now reports rides on the event (the loop passes it, as it
            # passes the committed inputs into InvokeSpan.send — only the loop holds the
            # state). Send is the supply boundary: suppliers (ValidateREPLCode, a host's
            # restriction hooks) compose here and the resolved short-circuit lands on span.supplied;
            # an Abort raises its carried exception out of span.send() instead (the span
            # closes Aborted — see span_repl_exec).
            span.send(self.repl_template.get_namespace(state))
            if span.supplied is not None:
                # A Send-composed SupplyExecResult short-circuits execution: skip running
                # the code and use the supplied result instead.
                exec_result = span.supplied
            else:
                # Execute the COMMITTED code — the enter code after any InsertCode/DeleteCode
                # edits (span.committed_code == the original when no hook edited it). This is the
                # same string REPLExecSend observers saw, so what runs matches what was validated.
                # (Invoke inputs are folded in once at InvokeEnter, not here: AddInputs/DropInputs
                # are InvokeEnter-only — #481. Per-turn namespace binds go through
                # AddToNamespace/DropFromNamespace above.)
                exec_result = self.repl_template.exec(
                    state, span.committed_code, str(iteration)
                )
            span.complete(exec_result=exec_result)

        return self._finalize_iteration(span, response_content, state)

    def _append_observation_messages(
        self,
        messages: list[MessageDict],
        exec_result: Continue,
        iteration: int,
    ) -> None:
        """Render the REPL observation via the protocol and append it to ``messages``.

        ``render_observation`` returns the observation as one or more ready messages, which
        append verbatim — persisted history stays pure protocol output, with no per-turn footer
        folded in. (The next-input nudge is unneeded scaffolding for the models JAZ targets, and
        persisting it froze a stale invoke-call count into every turn. The invoke-call budget
        could return as a transient per-query addition.) Each appended message is stamped with
        OBSERVATION provenance.
        """
        # #634 removed the budget-status / "enter your next REPL input" footer from every observation;
        # #657 tracks re-surfacing the budget half as a transient per-query addition.
        new_messages: list[MessageDict] = list(
            self.protocol.render_observation(exec_result)
        )
        for message in new_messages:
            # Stamp each observation message so consumers (TrajectoryRecorder, a
            # context-management hook, …) can identify it by provenance. #566 step C moved the
            # append into this helper (render_observation now returns ready messages), so
            # the OBSERVATION stamp moved here with it.
            set_provenance(
                message, MessageProvenance(MessageKind.OBSERVATION, iteration=iteration)
            )
            messages.append(message)

    @staticmethod
    def _apply_input_effects(
        ctx: "InvokeContext",
        inputs: dict[str, object],
        scope: dict[str, object],
    ) -> dict[str, object]:
        """Fold an invoke's ``AddInputs`` / ``DropInputs`` effects into ``inputs`` *before* the
        REPL and prompt are built. **Drops are applied first, then adds**, so dropping and adding
        the same key replaces it. An added key that collides with a caller-provided top-level
        input/scope entry raises (a hook may not silently override the caller). ``scope`` is never
        edited — a ``jaz.scope`` value is not an input, so naming one is "a name the invoke never
        received" and raises ``MissingDropTargetError`` like any other, unless that ``DropInputs``
        opted into ``allow_missing``; ``scope`` is read only, for the add-collision check. Returns
        the committed ``inputs``, copied only when an effect is present (the common no-effect path
        returns it unchanged)."""
        # The REPL namespace is NOT edited here (review of #1497): these effects are named for
        # ``inputs`` and edit only that, per the event-shape rule that an effect edits the field it
        # is named for. The merged namespace the REPL binds is a DERIVATION of the committed result
        # — the caller rebuilds it as ``{**scope, **inputs}`` right after this returns — so it
        # cannot drift from the commit, and there is no second dict to keep in step. Folding it
        # here used to require filtering by the dropped names that were *actually inputs*, purely
        # so an ``allow_missing`` drop naming a scope value would not strip it from the merged
        # dict; deriving the merge instead makes that structural.
        # There used to be a fourth dict here — the `__jaz_get__`-resolved counterpart of
        # `inputs`, threaded through so `InvokeSend` could be handed committed values without
        # re-resolving (a getter could be side-effecting, so each was resolved exactly once).
        # With the protocol gone, values bind as themselves and `inputs` IS that counterpart.
        added = ctx.added_inputs
        dropped = ctx.dropped_inputs
        if not added and not dropped:
            return inputs
        # Drops run BEFORE adds, so each check reads the state its contract is about:
        # ``MissingDropTargetError`` sees the inputs the invoke was actually called with, and the
        # add-collision check sees what survives the drops.
        #
        # It used to be the other way round, and the two checks then read *opposite* snapshots —
        # drops were validated post-add while adds were validated pre-drop. That inverted both
        # outcomes. A hook whose drop key was a typo landing on a name another hook had added was
        # silently honored (the typo net exists for exactly that mistake, and the added key
        # disappeared from the prompt with no signal to the hook that added it), while a hook
        # legitimately replacing an input — ``DropInputs({"secret"})`` plus
        # ``AddInputs({"secret": "REDACTED"})`` — hit ``REPLInputConflictError`` because the
        # collision check still saw the pre-drop value.
        #
        # What this ordering GIVES UP: a hook can no longer strip an input another hook added.
        # The drop runs while that key is still absent, so the add wins — and ``allow_missing``
        # does not change it (it silences the error, it does not defer the removal). The two
        # properties are mutually exclusive in one pass: a drop that outlives an add is the same
        # mechanism that lets a typo'd drop eat one. Choosing which wins is choosing which
        # mistake to make impossible, and a silently-eaten input is the worse one — it has no
        # signal at all, where a policy hook that needs to strip another hook's key at least
        # fails loudly (``MissingDropTargetError``) and can be rewritten to coordinate.
        # Pinned by ``test_a_drop_cannot_remove_a_key_another_effect_adds``.
        if dropped:
            # Fail loud on a drop of a name this invoke never received — a hook bug (typo, or a
            # "re-drop every turn" pattern), exactly as ``DropFromNamespace`` does at the REPL layer and
            # as ``MissingDropTargetError``'s contract documents for ``DropInputs``. Double-drop
            # stays safe: composition already unioned the keys into one set (dispatcher
            # ``_compose_invoke``), so two hooks dropping the same present name coalesce here rather
            # than tripping this. Keys marked ``allow_missing`` (a ``DropInputs(...,
            # allow_missing=True)``, e.g. a defensive drop that tolerates another effect having
            # already un-passed the input) are exempt from the check.
            #
            # ``inputs`` alone is what counts as present: ``DropInputs`` edits inputs, and a
            # ``jaz.scope`` value is not one (the effect is "remove a ``name=value`` keyword
            # argument", and scope values were never keyword arguments). Reaching into ``scope``
            # used to be an overreach of this fold, not a designed feature: no effect adds to
            # ``scope``, and keeping it out of every effect's reach is what lets ``scope`` be
            # ambient invoke context, like ``config``, rather than a per-invoke proposal.
            present = inputs.keys()
            missing = (dropped - present) - ctx.dropped_inputs_allow_missing
            if missing:
                raise MissingDropTargetError(
                    f"DropInputs target(s) {sorted(missing)!r} were not passed to this invoke; "
                    "dropping an absent input is a hook bug (drop it only when it is present, "
                    "e.g. once on the first InvokeEnter)."
                )
            # Naming the drop set directly is safe GIVEN that ``inputs`` and ``scope`` are
            # disjoint: a name that is not an input is not in ``inputs`` to begin with, so an
            # ``allow_missing`` drop of a scope value is a no-op here and the value stays bound
            # through the caller's re-merge. Where the old shape filtered the merged dict — and
            # so removed a name that was BOTH an input and a scope entry — this leaves the scope
            # copy. That case is unreachable (``_build_invoke_setup`` raises on the intersection,
            # and both entry points into ``Agent.invoke``/``ainvoke`` pass through it), which is
            # why the difference is not a behavior change; it does mean this fold now leans on
            # that check rather than merely agreeing with it.
            inputs = {k: v for k, v in inputs.items() if k not in dropped}
        if added:
            # A hook must not silently override a caller-provided top-level input. An ``AddInputs``
            # key that collides with an existing invoke input (``inputs``) or ``scope`` entry raises,
            # rather than letting ``{**inputs, **added}`` hand the win to the hook value unseen (which
            # would also desync prompt vs REPL and bypass the caller's input validation). Collisions
            # *between* AddInputs hooks are already caught at composition (``_coalesce_add``); this
            # guards the hook-vs-caller case, which composition never sees. A key this invoke also
            # dropped is gone by now, which is what makes drop-then-add a replace.
            collisions = added.keys() & (inputs.keys() | scope.keys())
            if collisions:
                raise REPLInputConflictError(
                    f"AddInputs key(s) {sorted(collisions)!r} already exist as top-level invoke "
                    "input(s)/scope entries; a hook must not silently override a caller-provided "
                    "input. Rename the added key, or stop passing it at the call site."
                )
            inputs = {**inputs, **added}  # renders in the prompt, and binds in the REPL
        return inputs

    def invoke(
        self,
        *,
        subinvoke: Subinvoke,
        depth: int,
        invoke_id: str,
        parent_invoke_id: str | None,
        parent_repl_iteration: int | None,
        scope: dict[str, object] | None = None,
        inputs: dict[str, object],
    ) -> object:
        # The return type is no longer a param (#568): it is declared by a ReturnType(...) hook
        # that renders the <return_type> prompt and enforces the type via effects, so the agent /
        # REPL know nothing about it. Returns whatever the agent RETURNed as an untyped object;
        # the public invoke's typed overloads carry the static -> ReturnT.
        # `inputs` (explicit kwargs) and `scope` (resolved `jaz.scope`) are kept as SEPARATE
        # provenance channels (#727) and merged into the single namespace the REPL actually
        # binds (`bound`) only once the input effects have committed, below. They are disjoint
        # (invoke.py's conflict check guarantees it), so the merge is order-independent and
        # nothing is lost by keeping the two apart — this is what lets the event carry them
        # unmerged with no downstream re-split. `scope` defaults to empty for any direct caller;
        # invoke.py always passes it.
        #
        # There is no payload-substitution step any more. Every value binds as itself, so the
        # dicts the prompt renders and the dicts the REPL binds hold the same objects; the
        # `__jaz_get__` protocol that made them potentially differ is gone.
        scope = scope or {}
        # Copied, not aliased: the effect fold below rebuilds these, but a caller's own mapping
        # must not be observed changing mid-invoke. `resolve_inputs` used to give this copy
        # incidentally by building a new dict; with it gone the copy has to be explicit.
        inputs = dict(inputs)
        scope = dict(scope)
        # Seed the per-invoke blackboard (generation 0) BEFORE any event fires, so
        # InvokeEnter consumers already see seeded keys. seed_blackboard binds it to
        # this invoke's dispatcher, which then stamps it onto every event in emit().
        # Seeds come from active hooks' blackboard_seed (e.g. a MetaData hook
        # carrying `label`), including per-call local hooks — agents can now pass
        # a `MetaData(...)` hook (non-public, `jaz.hooks._builtin.metadata`) positionally to
        # the agent-facing invoke (#545). A direct
        # caller-supplied `metadata=` seed channel is still deferred to #538 (the
        # public invoke-syntax issue; #537's hook-side mechanism landed as
        # `MetaData`, #545).
        self.dispatcher.seed_blackboard()
        # Use dispatcher span for invoke
        enter_event = InvokeEnter(
            config=self.config,
            invoke_id=invoke_id,
            parent_invoke_id=parent_invoke_id,
            parent_repl_iteration=parent_repl_iteration,
            inputs=inputs,
            scope=scope,
            depth=depth,
        )
        # An Abort composed at InvokeEnter raises its carried exception from the `with`
        # statement itself (e.g. an inherited cost-pool budget already exhausted): the
        # span closes Aborted in the CM and the exception propagates out of this invoke
        # bare — the top-level caller contract.
        with self.dispatcher.span_invoke(enter_event) as span:
            # Apply invoke-level input effects (AddInputs / DropInputs) to the invoke's inputs
            # BEFORE the REPL and prompt are built, so an added input renders in the prompt AND
            # binds in the REPL, and a dropped input is un-passed from both — the "inputs"
            # (prompt-level) half of the symmetric effect family. Drops are applied first, then
            # adds, so a key both added and dropped ends added. An added value lands in both the
            # prompt dict and the REPL namespace, exactly as a top-level input does.
            inputs = self._apply_input_effects(span.ctx, inputs, scope)
            # The REPL namespace is DERIVED from the committed inputs, so it reflects the fold by
            # construction rather than by being folded alongside it.
            bound = {**scope, **inputs}
            # InvokeSend — fires iff the input committed. The add/drop fold above IS the
            # commit of this invoke's input set, so Send fires here with the committed
            # resolved inputs (what InvokeEnter.inputs would have been had it carried the
            # post-effect view). It does not fire on an enter-time abort — that raised
            # before the fold, so the input set never commits. An Abort @ InvokeSend
            # raises out of this call (the committed-input veto; the span closed Aborted).
            span.send(inputs)

            # `subinvoke` carries only the framework's recursive sub-invoke primitive (a
            # single callable, bound in the REPL as the bare name `invoke`; user tool
            # namespaces are ordinary inputs). Hooks that need to
            # inject a namespace use AddInputs, not a tool effect.

            # Honor a DisableRecursion effect emitted at InvokeEnter (by RecursionLimit at the
            # cap leaf) by withholding the tool entirely: at the leaf there is nothing left to
            # bind, since the primitive IS the whole surface (#635 moved the former jaz.*
            # helpers out to ordinary inputs). Both consumers below (REPL binding + the prompt
            # that advertises the tool) must see the same value, or they would disagree —
            # a prompt documenting `invoke` over a REPL that has no such name. This is the one
            # read of `recursion_disabled` on this path (`ainvoke` mirrors it), and `subinvoke`
            # itself is required (never `None`), so "the tool is bound" has exactly one cause:
            # the effect was not emitted.
            effective_subinvoke = None if span.ctx.recursion_disabled else subinvoke
            # A WithholdREPLHistory effect withholds `__history__` the same way: the REPL binds no
            # list and the prompt omits its section, both from this one flag. Whether the REPL
            # binds the name at all is the REPL's (`repl_history_description`), read by the
            # opener; the loop holds no history list and knows no entry shape (see
            # `jaz.repl._repl_history`).
            repl_history_withheld = span.ctx.repl_history_withheld

            # Generate unique session ID to prevent linecache collisions between
            # sessions
            session_id = uuid.uuid4().hex[:8]

            # Build the REPL state and the opening prompt only when no result was supplied at
            # Send: a SupplyInvokeResult short-circuits the whole invoke, and `initialize` resolves
            # the invoke's inputs (possibly with I/O) while `render_initial_message_list` can run
            # arbitrary custom-protocol work — neither should run for a supplied invoke, and a
            # failure building them must not discard the supplied result. They stay None on that
            # path; the loop below breaks on `span.supplied` before it would read either.
            state: object | None = None
            messages: list[MessageDict] | None = None
            if span.supplied is None:
                # The configured REPL is a stateless template; `initialize` returns this run's
                # per-invoke state (#1058), threaded into every exec below. The protocol opener still
                # gets the config template (it reads only config: description, history description).
                state = self.repl_template.initialize(
                    inputs=bound,
                    subinvoke=effective_subinvoke,
                    session_id=session_id,
                    repl_history_withheld=repl_history_withheld,
                )

                messages = self.protocol.render_initial_message_list(
                    inputs,
                    scope,
                    self.repl_template,
                    subinvoke=effective_subinvoke,
                    repl_history_withheld=repl_history_withheld,
                )
                # A custom BaseProtocol must return a non-empty opener. Guard it loudly
                # here (the load-bearing half of the old iteration-0 footer assert, #705): an
                # empty list otherwise slips past _stamp_seed_provenance and surfaces later as a
                # muddier empty-message-list error at the first query.
                assert messages, "render_initial_message_list returned an empty opener"

                self._stamp_seed_provenance(messages)

            # (Invoke-level AddInputs/DropInputs were already folded into `inputs` above via
            # _apply_input_effects, and `bound` derived from the result, before the REPL and
            # prompt were built — so added inputs are bound in `repl` and rendered in `messages`,
            # and dropped inputs are absent from both. No post-init REPL mutation needed.)

            # The loop is unbounded; termination is owned by effects — a terminal
            # RETURN/RAISE result, or a governance-emitted Abort. The hard iteration cap
            # lives in IterationLimit (Abort at the always-present LLMQueryEnter, whose
            # carried exception raises out of do_one_repl_iteration — see #481). There is
            # no per-iteration span: the per-turn boundary is now LLMQuery, so the old
            # REPLIterationEnter/Exit events and their span were removed.
            #
            # Iteration numbering is 0-based, everywhere and without exception (#719, executive
            # call): the first turn is iteration 0, and every surface that reports it — the console
            # prefix, the loggers, ATIF ``extra.iteration``, ``iter_N/`` trace dirs, the REPL
            # traceback filename — shows the same number this loop holds. ``itertools.count()``
            # rather than a hand-rolled counter so the 0 start is stated once, structurally.
            #
            # The alternative, and the reason this is worth a comment: most agent frameworks keep a
            # 0-based index internally and add 1 at the display boundary (Keras prints
            # ``f"Epoch {epoch + 1}/{self.epochs}"`` over a 0-based arg; mini-swe-agent renders
            # ``Step {self.i_step + 1}``; SWE-agent logs ``STEP {len(self.trajectory) + 1}``). That
            # split was rejected: it makes ``iteration`` in code and ``iteration`` in the log two
            # different numbers under one name, which is a debugging trap, and it costs a ``+1``
            # that must be remembered at seven display sites to make one internal site read nicer.
            # Keras survives it because its two audiences differ; JAZ's iteration number is read by
            # one person in one sitting — console, then the ATIF trace, then a hook they are
            # writing. One convention everywhere is worth more here than agreement with convention.
            #
            # Consequences to keep straight, since a 0-based *position* now sits next to counts:
            # ``max_iterations`` stays a COUNT (50 means 50 turns, indices 0-49), so IterationLimit
            # hard-stops on ``iteration >= max_iterations`` and its last allowed index is
            # ``max_iterations - 1``; ATIF ``total_steps`` is a count too, so it is
            # ``final index + 1``. IterationLimit's approaching-limit warning reads
            # the remaining turns straight off this 0-based index (``max_iterations - iteration``),
            # an absolute turn count rather than a fraction of the cap (see
            # ``IterationLimit.warn_remaining``, and the sibling ``BudgetPool.warn_*_remaining``).
            #
            # ``PythonREPL``'s ``__history__`` is a plain 0-based list holding exactly one entry
            # per iteration (see ``_finalize_iteration``), so iteration N is now
            # ``__history__[N]``. Note this is
            # alignment, not a contract anything relies on: nothing in-tree addresses history by
            # iteration number, and the heaviest consumers (the oolong/stulife history-search
            # prompts) scan ``prev_repl_history + __history__[:-1]``, a concatenation across
            # agent boundaries whose positions cannot correspond to this agent's turns under any
            # base. ``PythonREPL.repl_history_description`` says ``[-1]`` for that reason.
            #
            # Watch for falsiness when adding a consumer: iteration 0 is falsy, so ``if iteration:``
            # silently means "not the first turn". Use ``is not None``.
            for i in itertools.count():
                # A hook supplied this invoke's terminal result at InvokeSend (span.supplied, set
                # once by span.send() before the loop): complete with it instead of running any
                # turn, so a supplied invoke makes zero LLM calls — and no REPL or opener were built
                # (see the `if span.supplied is None` guard above). Checked at the top of the first
                # pass — itertools.count() always enters, so there is no empty-loop hole. The
                # InvokeComplete transform still fires as the span closes (ReturnType /
                # ValidateReturn / ModifyInvokeResult apply to the supplied result), mirroring
                # SupplyExecResult flowing through REPLExecComplete.
                if span.supplied is not None:
                    span.complete(result=span.supplied)
                    break
                # Past the supplied short-circuit: state/messages are None only on the supplied
                # path (built under the guard above otherwise), and that path broke out one line up.
                assert state is not None and messages is not None
                # Update prehook so nested invokes know which iteration spawned them
                if self.prehook is not None:
                    self.prehook.parent_repl_iteration = i

                exec_result = self.do_one_repl_iteration(
                    i,
                    depth,
                    messages,
                    state,
                    invoke_id,
                )

                match exec_result:
                    case Continue():
                        self._append_observation_messages(
                            messages,
                            exec_result,
                            i,
                        )
                    case Return() | Raise():
                        # Record the terminal result and leave the loop. The InvokeComplete
                        # boundary (fired as the span closes) may still TRANSFORM it — a
                        # ReturnType / ValidateReturn hook downgrades a wrong-typed / invalid
                        # final Return to a Raise there (the backstop for another hook's
                        # REPLExecComplete override reinstating a Return). So we no longer
                        # return/raise from inside the span.
                        span.complete(result=exec_result)
                        break
                    case _:
                        raise _JazInternalError("Unhandled REPL result type")

        # Span closed (InvokeComplete composed, InvokeExit fired with the post-transform
        # result): read the FINAL terminal result and return/raise from it.
        final_result = span.get_result()
        match final_result:
            case Return():
                return final_result.return_value
            case Raise():
                raise final_result.exception
            case _:
                raise _JazInternalError("Unhandled final invoke result type")

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def _aquery_with_messages(
        self,
        messages: list[MessageDict],
        *,
        iteration: int,
        depth: int,
        invoke_id: str,
    ) -> str:
        # Mirrors _query_with_messages exactly except for the awaited LLM call; the edit
        # fold is the shared _compose_shown_messages, so the two paths cannot drift. An
        # Abort at any stage raises out of this method — see the sync path's comment.
        complete_fn, enter_event = self._make_llm_retry_fn(
            self.llm_client._acomplete_with_retry,
            messages,
            invoke_id,
            iteration,
            depth,
        )
        with self.dispatcher.span_llm_query(enter_event) as span:
            shown = self._compose_shown_messages(span, messages)
            # LLMQuerySend — fires iff the input committed (see the sync path's comment):
            # unconditionally after the edit fold, on the supplied path too; never on an
            # enter-time abort (which raised before this body ran).
            span.send(shown)
            # Send-supplied response (e.g. TrajectoryReplay): the provider call is skipped.
            if span.supplied_response is not None:
                llm_response: LLMResponse = span.supplied_response
            else:
                # Provider errors propagate; see the sync path (incl. the TODO(#697)
                # future-policy note).
                llm_response = await complete_fn(messages=self._to_wire(shown))
            result = self._record_llm_response(llm_response, span)
        # Span closed: a Complete-stage ModifyLLMResponse may have replaced the response
        # (authoritative — see _resolve_llm_modify_results). If so, re-read it: its content is what
        # the agent parses next, and the same object is what LLMQueryExit carried and BudgetPool
        # booked, so content / observers / cost all agree. The identity check leaves the common
        # no-transform path (same object) untouched, and covers the supplied-response path too.
        final_response = span.get_response()
        if final_response is not llm_response:
            self._last_llm_response = final_response
            result = final_response.content or ""
        return result

    async def do_one_repl_iteration_async(
        self,
        iteration: int,
        depth: int,
        messages: list[MessageDict],
        state: object,
        invoke_id: str,
    ) -> ExecResult[Any]:
        # Mirrors do_one_repl_iteration exactly except for the two awaited steps — the
        # LLM query above and repl.aexec below. Everything else is the shared
        # _record_response_and_parse / _finalize_iteration, so the paths cannot drift.
        # (The async copy previously diverged: aexec ran *outside* the enter-override
        # `else` and so clobbered enter-time overrides (#592), and it never applied
        # exit-time overrides nor finalized history — all fixed by routing through the
        # shared helpers.)
        self._reset_per_query_state()
        # No catch around the query: LLM-call failures and abort-carried exceptions
        # propagate — see the sync path's comment on the deleted #687 conversion.
        response_content = await self._aquery_with_messages(
            messages,
            iteration=iteration,
            depth=depth,
            invoke_id=invoke_id,
        )
        # Apply any persistent message edits (compaction) to the carried-forward buffer.
        self._apply_persistent_message_edits(messages, iteration)
        enter_event = self._record_response_and_parse(
            messages, response_content, invoke_id, iteration, depth, state
        )

        with self.dispatcher.span_repl_exec(enter_event) as span:
            # Apply DropFromNamespace then AddToNamespace (see the sync path for rationale): unbind the
            # composed names, then bind, before this turn's code runs — drop-first so a
            # drop-then-add pair re-binds; consistent across the override/exec branches.
            self.repl_template.drop_from_namespace(
                state,
                span.ctx.namespace_drops,
                allow_missing=span.ctx.namespace_drops_allow_missing,
            )
            self.repl_template.add_to_namespace(state, span.ctx.namespace_adds)
            # REPLExecSend — fires iff the code committed (see the sync path's comment):
            # after the namespace deltas, carrying the namespace the REPL now reports;
            # suppliers compose here and the resolved short-circuit lands on span.supplied
            # (an Abort raises out of span.send).
            span.send(self.repl_template.get_namespace(state))
            if span.supplied is not None:
                # A Send-composed SupplyExecResult short-circuits execution: skip running
                # the code and use the supplied result instead.
                exec_result = span.supplied
            else:
                # Execute the COMMITTED code — the enter code after any InsertCode/DeleteCode
                # edits (== the original when no hook edited it), the same string REPLExecSend
                # observers saw. (Invoke inputs are folded in once at InvokeEnter, not here:
                # AddInputs/DropInputs are InvokeEnter-only — #481. Per-turn namespace binds go
                # through AddToNamespace/DropFromNamespace above.)
                exec_result = await self.repl_template.aexec(
                    state, span.committed_code, str(iteration)
                )
            span.complete(exec_result=exec_result)

        return self._finalize_iteration(span, response_content, state)

    async def ainvoke(
        self,
        *,
        subinvoke: Subinvoke,
        depth: int,
        invoke_id: str,
        parent_invoke_id: str | None,
        parent_repl_iteration: int | None,
        scope: dict[str, object] | None = None,
        inputs: dict[str, object],
    ) -> object:
        # The return type is a ReturnType(...) hook now (#568), not a param — see `invoke`.
        # Mirrors the sync `invoke` path — including keeping `inputs` (explicit kwargs)
        # and `scope` (resolved `jaz.scope`) as SEPARATE channels (#727), merged into the
        # single REPL namespace (`bound`) only after the input effects commit; see sync
        # `invoke` for the full rationale.
        scope = scope or {}
        # Copied, not aliased: the effect fold below rebuilds these, but a caller's own mapping
        # must not be observed changing mid-invoke. `resolve_inputs` used to give this copy
        # incidentally by building a new dict; with it gone the copy has to be explicit.
        inputs = dict(inputs)
        scope = dict(scope)
        # Seed the per-invoke blackboard (generation 0) before any event fires; see
        # the sync invoke() for rationale and the #538 TODO on caller metadata.
        self.dispatcher.seed_blackboard()
        enter_event = InvokeEnter(
            config=self.config,
            invoke_id=invoke_id,
            parent_invoke_id=parent_invoke_id,
            parent_repl_iteration=parent_repl_iteration,
            inputs=inputs,
            scope=scope,
            depth=depth,
        )
        # An Abort at InvokeEnter raises from the `with` statement itself — see the sync
        # path's comment.
        with self.dispatcher.span_invoke(enter_event) as span:
            # Apply invoke-level input effects (AddInputs / DropInputs) before the REPL and
            # prompt are built — the async mirror of the sync path (see there for the full
            # rationale). Adds render+bind, drops un-pass from both; drops run first, so an add
            # wins on overlap.
            inputs = self._apply_input_effects(span.ctx, inputs, scope)
            # Derived from the committed inputs, as in sync invoke().
            bound = {**scope, **inputs}
            # InvokeSend — fires iff the input committed (see the sync path's comment):
            # right after the add/drop fold, never on an enter-time abort. An Abort here
            # raises out of this call (the span closed Aborted).
            span.send(inputs)

            # `subinvoke` carries only the framework's recursive sub-invoke primitive (a
            # single callable; see the sync path above).

            # Honor DisableRecursion (RecursionLimit at the cap leaf) — same as the sync path:
            # withhold the tool entirely, in the one place the flag is read.
            effective_subinvoke = None if span.ctx.recursion_disabled else subinvoke
            # A WithholdREPLHistory effect withholds `__history__` the same way (see the sync
            # invoke()).
            repl_history_withheld = span.ctx.repl_history_withheld

            session_id = uuid.uuid4().hex[:8]

            # Build the REPL state and the opening prompt only when no result was supplied at Send
            # (the sync path documents why); they stay None for a supplied invoke, which the loop
            # below short-circuits before reading either.
            state: object | None = None
            messages: list[MessageDict] | None = None
            if span.supplied is None:
                # The configured REPL is a stateless template; `initialize` returns this run's
                # per-invoke state (#1058). The protocol opener still gets the config template.
                state = self.repl_template.initialize(
                    inputs=bound,
                    subinvoke=effective_subinvoke,
                    session_id=session_id,
                    repl_history_withheld=repl_history_withheld,
                )

                messages = self.protocol.render_initial_message_list(
                    inputs,
                    scope,
                    self.repl_template,
                    subinvoke=effective_subinvoke,
                    repl_history_withheld=repl_history_withheld,
                )
                # A custom BaseProtocol must return a non-empty opener. Guard it loudly
                # here (the load-bearing half of the old iteration-0 footer assert, #705): an
                # empty list otherwise slips past _stamp_seed_provenance and surfaces later as a
                # muddier empty-message-list error at the first query.
                assert messages, "render_initial_message_list returned an empty opener"

                self._stamp_seed_provenance(messages)

            # (AddInputs/DropInputs already folded into `inputs` above via _apply_input_effects,
            # with `bound` derived from the result — no post-init REPL mutation needed; see sync
            # invoke().)

            # ``i`` is 0-based, as in sync invoke() — see the iteration-numbering note there (#719).
            for i in itertools.count():
                # A hook supplied this invoke's terminal result at InvokeSend (span.supplied, set
                # once by span.send() before the loop): complete with it instead of running any
                # turn, so a supplied invoke makes zero LLM calls — and no REPL or opener were built
                # (see the `if span.supplied is None` guard above). Checked at the top of the first
                # pass — itertools.count() always enters, so there is no empty-loop hole. The
                # InvokeComplete transform still fires as the span closes (ReturnType /
                # ValidateReturn / ModifyInvokeResult apply to the supplied result), mirroring
                # SupplyExecResult flowing through REPLExecComplete.
                if span.supplied is not None:
                    span.complete(result=span.supplied)
                    break
                # Past the supplied short-circuit: state/messages are None only on the supplied
                # path (built under the guard above otherwise), and that path broke out one line up.
                assert state is not None and messages is not None
                # Update prehook so nested invokes know which iteration spawned them
                if self.prehook is not None:
                    self.prehook.parent_repl_iteration = i

                exec_result = await self.do_one_repl_iteration_async(
                    i,
                    depth,
                    messages,
                    state,
                    invoke_id,
                )

                match exec_result:
                    case Continue():
                        self._append_observation_messages(
                            messages,
                            exec_result,
                            i,
                        )
                    case Return() | Raise():
                        # Record the terminal result and leave the loop. The InvokeComplete
                        # boundary (fired as the span closes) may still TRANSFORM it — a
                        # ReturnType / ValidateReturn hook downgrades a wrong-typed / invalid
                        # final Return to a Raise there (the backstop for another hook's
                        # REPLExecComplete override reinstating a Return). So we no longer
                        # return/raise from inside the span.
                        span.complete(result=exec_result)
                        break
                    case _:
                        raise _JazInternalError("Unhandled REPL result type")

        # Span closed (InvokeComplete composed, InvokeExit fired with the post-transform
        # result): read the FINAL terminal result and return/raise from it.
        final_result = span.get_result()
        match final_result:
            case Return():
                return final_result.return_value
            case Raise():
                raise final_result.exception
            case _:
                raise _JazInternalError("Unhandled final invoke result type")
