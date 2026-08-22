import functools
import itertools
import uuid
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from .invoke import Prehook

from tenacity import RetryCallState

from ._invoke_tool import InvokeTool
from ._llm_client import BaseLLM, LLMResponse
from .config import Config
from .exceptions import (
    MissingDropTargetError,
    REPLInputConflictError,
    _JazInternalError,
)
from .hooks import Hook
from .hooks.dispatcher import HookDispatcher
from .hooks.effects import apply_message_edits
from .hooks.events import (
    InvokeContext,
    InvokeEnter,
    LLMQueryEnter,
    LLMQueryRetry,
    LLMQuerySpan,
    REPLExecEnter,
    REPLExecSpan,
)
from .hooks.events.llm_query import MessageEdits
from .inputs import resolve_inputs
from .llm import MessageDict
from .protocol import BaseProtocol
from .provenance import (
    MessageKind,
    MessageProvenance,
    set_provenance,
    to_wire_messages,
)
from .repl.base import BaseREPL, REPLState
from .repl.registry import REPL_LANGUAGE_MAP
from .repl.types import (
    Continue,
    ExecResult,
    Raise,
    Return,
)
from .tokens import set_tokens, tokens_of


class REPLIterationResult[ReturnT](NamedTuple):
    """Result from a single REPL iteration.

    Attributes:
        exec_result: The execution result from the REPL
        state: The per-invoke :class:`REPLState` threaded across iterations (#1058); the REPL
            itself is stateless config, so it is not carried here.
        response_content: The LLM response content. Never None: the no-response cases
            that used to produce one (a failed LLM call, an ``Abort`` at ``LLMQueryEnter``)
            now *raise* out of the query under the abort model, so an iteration result only
            exists when a response does. An *empty* completion is coerced to ""
            (``Agent._record_llm_response``).
        code: The extracted code from the response (None if not found)
    """

    exec_result: ExecResult[ReturnT]
    state: REPLState
    response_content: str
    code: str | None


class Agent:
    def __init__(
        self,
        repl: str,
        prehook: "Prehook | None" = None,
        *,
        config: Config,
        local_hooks: tuple[Hook, ...] = (),
    ) -> None:
        """Construct an agent. ``config`` is REQUIRED — there is no ambient fallback.

        **Why required (#826 §3, executive call).** ``config`` used to default to ``None`` and fall
        back to reading the ambient config, which was the last depth-less ``get_config()`` read left
        in the codebase — the read shape #826 set out to retire, because it silently diverges from
        the config an invoke actually ran under whenever a per-depth override is in scope. Two other
        resolutions were considered and rejected:

        - *Resolve the fallback at depth 1* (treat a bare Agent as a top-level agent). Rejected: an
          Agent has no recursion depth. It carries an optional ``prehook`` and is just an object a
          caller can drive at any level, so "depth 1" is an invented fact — and inventing it widens
          ``configure_by_depth({1: ...})`` to silently reach Agents that are not top-level invokes.
        - *Keep the fallback depth-less* and document it as the one legitimate ambient read.
          Rejected: honest about the semantics (no depth ⇒ no per-depth layer applies), but it
          leaves an implicit ambient read on a public constructor for no ergonomic gain.

        Requiring the argument resolves the tension instead of picking a side of it: the caller
        decides, so no depth has to be invented and no ambient read remains in this path. The
        framework never relied on the fallback anyway — ``_build_invoke_setup`` has always passed
        the depth-resolved ``config`` explicitly (it is the only in-tree construction site), so this
        is unreachable-in-practice surface, not a capability being taken away.

        **Compatibility.** This is a breaking change to the public ``jaz.Agent`` signature; direct
        construction was exported but never documented (no example in README/docs/design). Callers
        that want the old behavior pass ``config=jaz.get_config()`` explicitly — which now says at
        the call site that an ambient, depth-less config is what they meant.

        Keyword-only (rather than reordered ahead of ``prehook``) so existing positional
        ``Agent(repl, prehook)`` calls keep working; only the implicit config default is removed.
        """
        if not repl:
            raise ValueError("A REPL language must be specified")
        self.config = config
        # The components come straight off the config — there is nothing left to build here.
        # This used to be three resolution blocks (a tag-or-instance branch for the backend, the
        # same for the protocol, and a language lookup for the REPL), each turning a
        # `{tag, params}` description into an object and each able to fail at `Agent.__init__`
        # for a config error written much earlier. Config now holds the configured components,
        # so a bad one raised at the `configure`/`ConfigOverride` call site that wrote it.
        if repl not in REPL_LANGUAGE_MAP:
            raise ValueError(f"Unsupported REPL language: {repl}")
        self.repl_language: str = repl
        # The configured REPL is a stateless *template*: `initialize` returns a fresh `REPLState`
        # per invoke and the template holds no per-invoke state, so one config safely serves many
        # concurrent invokes (#1056, #1058).
        self.repl_template: BaseREPL = self.config.repl
        self.llm_client: BaseLLM = self.config.llm
        self.model: str = self.llm_client.get_model()
        # Resolved once here, not per turn: both the client and the model are fixed for this
        # agent's lifetime, so the answer cannot change between iterations.
        self.can_report_cost: bool = self.llm_client.can_report_cost()
        self.protocol: BaseProtocol = self.config.protocol
        self.prehook = prehook
        # Always a per-invoke dispatcher instance (never the shared singleton): the
        # dispatcher now holds this invoke's seeded blackboard (see seed_blackboard()
        # in invoke()/ainvoke()), which is per-call state that a shared singleton
        # could not carry. Local hooks (if any) ride along on the same instance.
        # See HookDispatcher class docstring for why per-instance beats a second
        # contextvar. A HookDispatcher is a trivial object, so per-invoke allocation
        # is cheap; get_dispatcher() remains for any caller that wants the singleton.
        self.dispatcher = HookDispatcher(local_hooks=local_hooks)
        # Per-query side-channel: the message edits composed at the last LLMQueryEnter,
        # stashed by _compose_shown_messages and applied to the carried-forward buffer by
        # _apply_persistent_message_edits (the loop resets it before each query). Carries
        # the composed edits out of the (mock-patched) _query_with_messages without
        # changing its signature.
        self._pending_message_edits: MessageEdits | None = None

        # Per-query side-channel (same idiom as _pending_message_edits): the full LLMResponse
        # from the last *real* query, stashed by _record_llm_response so _record_history can hand
        # the whole object to protocol.build_history_entry (the less-lossy #566 input) without
        # changing what _query_with_messages returns (still the content str — so the many tests
        # that patch _query_with_messages to return a bare string keep working). Reset to None
        # before each query; on the mocked path it stays None and _record_history synthesizes a
        # minimal LLMResponse from the content string, keeping the default projection identical.
        self._last_llm_response: LLMResponse | None = None

        # Per-query side-channel (same idiom as the two above): the last query's *pre-strip*
        # shown list — the buffer dicts (by reference) plus any transient hook additions, as
        # composed by _compose_shown_messages before the wire projection copies stamped
        # messages. _stamp_back_tokens needs it because a token-native backend's
        # TurnRecord.sent is index-aligned with this list, and only these dict objects (not
        # the wire copies) live on in the buffer. Reset to None before each query; stays
        # None on the mocked path.
        # TODO(#1161): this is the third per-query side-channel, each reset in two places (the
        # sync + async query paths). Collapse all three into one record returned by
        # _compose_shown_messages and threaded as a local, removing the paired reset sites.
        self._pending_shown_messages: list[MessageDict] | None = None

    def _make_llm_retry_fn(
        self,
        retry_fn: Any,
        messages: list[MessageDict],
        invoke_id: str,
        iteration: int,
        depth: int,
        can_recurse: bool,
    ) -> tuple[Any, str, LLMQueryEnter]:
        """Bind the Config retry params + a hook-dispatching before-sleep onto
        ``retry_fn`` (``LLM.(a)complete_with_retry``) and return
        (fn, invoke_id, enter_event). All retry logic lives on the LLM; the
        Agent only supplies the parameters and the ``on_retry`` callback."""

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
                    model=self.model,
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
        # No retry settings are threaded. The backend holds its own, and `complete_with_retry`
        # falls back to them for every kwarg left `None`.
        #
        # This used to forward `retry_*` out of `llm.params` as per-call overrides, but only for
        # an instance tag: a tag-built client got them through `create_llm`, while a pre-built
        # one never went through it, so the config keys would otherwise have been silently
        # ignored for instance tags. With the configured backend itself in the config there is
        # no second place for a retry setting to live, so there is nothing to reconcile.
        wrapped = functools.partial(retry_fn, on_retry=log_retry_attempt)

        enter_event = LLMQueryEnter(
            config=self.config,
            invoke_id=invoke_id,
            messages=messages,
            model=self.model,
            iteration=iteration,
            depth=depth,
            can_recurse=can_recurse,
            can_report_cost=self.can_report_cost,
        )
        return wrapped, invoke_id, enter_event

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
        # Stash the whole object for _record_history's projection (see the field's init comment);
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
        can_recurse: bool,
    ) -> str:
        # An Abort composed at any of the query's stages raises its carried exception out
        # of this method (from the `with` statement for an enter-time hard-stop, from
        # span.send for the committed-input veto, from the block's close for a
        # Complete-time one) — the span CM closes the span Aborted first, and the invoke
        # span classifies the same exception Aborted on its own unwind (see
        # HookDispatcher._unwind_outcome). Content is coerced to ``str`` at
        # ``_record_llm_response`` (an empty completion projects to ``""``, #1141), so this
        # returns a plain ``str`` — never ``None``, never a laundered ``Raise``.
        complete_fn, invoke_id, enter_event = self._make_llm_retry_fn(
            self.llm_client.complete_with_retry,
            messages,
            invoke_id,
            iteration,
            depth,
            can_recurse,
        )
        with self.dispatcher.span_llm_query(enter_event) as span:
            shown = self._compose_shown_messages(span, messages)
            # LLMQuerySend — fires iff the input committed. ``shown`` IS the committed
            # input (the post-edit list handed to the result producer), so Send fires here
            # unconditionally: on the live-call path AND the supplied path below (the
            # commit is real on both; only the producer differs — Send is where a supplier
            # like Replay answers, with the committed list in view). It does not fire on
            # an enter-time abort — that raises out of the `with` statement before this
            # body runs, so no input ever commits (the span still closes, Aborted).
            span.send(shown)
            # Send-supplied response (e.g. Replay): the provider call is skipped.
            if span.supplied_response is not None:
                llm_response: LLMResponse = span.supplied_response
            else:
                # A provider error still raised after the retry policy propagates from
                # here: LLMQueryExit(Failed) fires from the span CM, the invoke span
                # closes Failed on the unwind, and the caller receives the exception.
                # TODO(#697): deliberately downgrading *selected* failures to a
                # recoverable Continue remains a legitimate future loop policy (as loop
                # policy — not the blanket Raise-conversion #687 used to do here).
                llm_response = complete_fn(
                    model=self.model,
                    messages=shown,
                    **self.llm_client.request_defaults,
                )
            result = self._record_llm_response(llm_response, span)
        # Span closed: a Complete-stage ModifyLLMResponse may have replaced the response
        # (authoritative — see resolve_llm_modify_results). If so, re-read it: its content is what
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

        This is also the single **provider-egress** point: the returned list is projected
        to wire form (``to_wire_messages``), stripping every reserved sidecar key the
        configured backend has not declared it consumes
        (``BaseLLM.consumes_internal_keys``), so internal per-message keys never reach a model
        that doesn't read them. Both the sync and async query paths route through here, so
        one strip covers both regardless of the underlying LLM client.
        """
        edits = span.ctx.message_edits
        self._pending_message_edits = edits
        shown = apply_message_edits(messages, edits.all_drops, edits.all_adds)
        # Stashed pre-strip: `shown` holds the buffer's own dicts by reference (the fold
        # never copies), while the wire projection below copies every stamped message — so
        # this list, not the returned one, is what stamp-back can mutate meaningfully.
        self._pending_shown_messages = shown
        return to_wire_messages(shown, keep=self.llm_client.consumes_internal_keys)

    def _apply_persistent_message_edits(
        self, messages: list[MessageDict], iteration: int
    ) -> None:
        """Apply the last query's *persistent* message edits to the carried-forward buffer.

        ``messages`` is the loop's real buffer, and the edit indices anchor to exactly it:
        prompt additions now flow through the message-edit fold rather than being baked into
        a longer query snapshot ahead of it (``AddInstructionPrompt`` was removed in favour
        of ``AddMessages`` — #660), so the query snapshot coincides with the buffer and no
        trailing transient instruction has to be stripped. Fold the persistent subset over
        the buffer. A no-op unless a hook emitted a persistent ``DropMessages`` /
        ``AddMessages`` (context compaction).

        Inserted messages are stamped with ``ADDED`` provenance *after* the fold — the
        add's dict objects are now in the buffer by reference, and stamping earlier would
        put a non-serializable provenance value into ``AddMessages.messages`` before the
        fold's ``json.dumps`` sort-key computation runs over it.
        """
        edits = self._pending_message_edits
        if edits is None or not edits.has_persistent:
            return
        messages[:] = apply_message_edits(
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
        # This is the "stamp-back" edge of the token-rollout design (see jaz.tokens'
        # module comment for the rationale): the backend mints one frozen
        # TokenStamp per wire message inside complete(); here the agent attaches each
        # stamp to the buffer dict it came from, so the *next* send reuses the exact ids
        # (no re-render, no drift) and the stamps ride through edits/compaction by
        # reference. The agent treats stamps as opaque blobs and stays token-blind.
        #
        # Positional contract: record.sent[i] describes wire message i, and the wire list
        # is an element-wise projection of the stashed pre-strip shown list — so
        # zip(shown, sent) pairs each stamp with the dict that produced it.
        # strict=True fails loud on a length mismatch — a genuine one-stamp-per-wire-message
        # violation, i.e. a backend bug worth raising on. A subtler misalignment does NOT put
        # wrong ids on the wire: the next send's content-guard re-renders any mismatched stamp,
        # and per-message ids are a pure function of (content, chat template). Its real cost is
        # downstream — _extends (rollout export) stops chaining, so to_flat shatters into pieces:
        # degraded export quality, not corrupted training ground truth.
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
        repl: BaseREPL,
        invoke_id: str,
        iteration: int,
        depth: int,
    ) -> tuple[float | None, REPLExecEnter]:
        """Record the assistant response and decode it into an execution plan.

        Shared by the sync and async iteration paths — which differ *only* in how the
        response is fetched and how the code is executed; everything around that is
        identical and lives here (and in :meth:`_finalize_iteration`) so the two paths
        cannot drift. Appends the assistant message, then parses, and always returns
        ``(exec_timeout_override, enter_event)``. The parsed code rides on ``enter_event.code``
        (its edit target); the code that actually *runs* is ``span.committed_code`` — the enter
        code after any ``InsertCode`` / ``DeleteCode`` — so the bare parsed string is not returned
        separately (it would be dead at both call sites).

        There is exactly one outcome now. This used to return either that tuple or a ready
        ``REPLIterationResult`` carrying a recoverable parse error, which callers told apart with
        ``isinstance`` — a discrimination that only worked because the success branch was a
        *plain* tuple rather than a ``NamedTuple``. ``parse`` no longer has a failure mode, so
        both the second outcome and that latent trap are gone.
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
        code, exec_timeout_override = self.protocol.parse(
            self._llm_response_to_record(response_content), self.repl_language
        )

        enter_event = REPLExecEnter(
            config=self.config,
            invoke_id=invoke_id,
            iteration=iteration,
            code=code,
            depth=depth,
        )
        return exec_timeout_override, enter_event

    def _finalize_iteration(
        self,
        span: REPLExecSpan,
        code: str,
        state: REPLState,
        response_content: str,
        repl_history: list[object] | None,
    ) -> REPLIterationResult[Any]:
        """Apply exit-time overrides and append the iteration's history entry.

        Shared by both iteration paths. ``span.get_final_exec_result()`` folds in any
        Complete-time ModifyExecResult / Abort; the core loop is the single writer
        of ``__history__`` and appends one entry per iteration from that FINAL
        (post-override) result — so a result-override hook (e.g. BudgetForcing) is
        reflected in history without the REPL knowing (#448), and an enter-time override
        that skipped exec() still records an entry. ``repl_history`` is the same list the
        REPL surfaced as ``__history__`` (by reference); None means no history.
        """
        exec_result = span.get_final_exec_result()
        self._record_history(
            repl_history,
            self._llm_response_to_record(response_content),
            exec_result,
        )
        return REPLIterationResult(
            exec_result=exec_result,
            state=state,
            response_content=response_content,
            code=code,
        )

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
        # practice (verified: converting it to an assert produced zero hits across tests + evals)
        # and is now unreachable by construction.
        if self._last_llm_response is not None:
            return self._last_llm_response
        return LLMResponse(content=response_content)

    def _record_history(
        self,
        repl_history: list[object] | None,
        llm_response: LLMResponse,
        exec_result: ExecResult,
    ) -> None:
        """Append one ``__history__`` entry for a turn that produced an LLM response.

        Records **every** turn the LLM actually responded to, storing the FULL response
        (``REPLHistoryEntry.llm_response`` — the raw code incl. its leading comments), not the parsed
        code. This includes a turn whose response **failed to parse**: that is still a real turn the
        LLM took, and its entry carries the unparseable response plus the parse-error exception in
        ``repl_exception`` (``exec_result`` there is the recoverable ``Continue`` the parse error was
        surfaced as). A meta-agent analyzing behavior therefore sees malformed turns too, and the
        per-turn index stays aligned with the loop's iteration count.

        ``llm_response`` (resolved by :meth:`_llm_response_to_record`) is the whole ``LLMResponse``,
        not the bare content string, so the configurable ``self.protocol.build_history_entry`` seam
        can record a richer entry (#566).
        """
        # Every turn that reaches here records an entry, so ``__history__`` holds exactly one per
        # iteration (#719), and no guard is needed to say so: ``response_content`` is a ``str`` by
        # the time it reaches ``_llm_response_to_record``, so that always returns an object. An
        # empty completion is no exception — it projects to ``llm_response=""`` in
        # ``build_history_entry``, as it always has. The no-response terminal paths (a failed LLM
        # call, or an ``Abort``) *raise* out of ``do_one_repl_iteration`` and never reach here.
        #
        # The one-entry-per-iteration invariant is not something to lean on for addressing:
        # nothing addresses history *by iteration number*, and anything naming the current turn's
        # entry uses ``[-1]``, which holds without it. This append runs inside
        # ``_finalize_iteration``, before the turn's observation is rendered, so ``[-1]`` is that
        # turn — hence ``repl_output_truncation_advice.jinja2``.
        #
        # The remaining guard means history is disabled for this REPL, not that there was nothing
        # to record.
        if repl_history is None:
            return
        repl_history.append(
            self.protocol.build_history_entry(llm_response, exec_result)
        )

    def do_one_repl_iteration(
        self,
        iteration: int,
        depth: int,
        can_recurse: bool,
        messages: list[MessageDict],
        state: REPLState,
        invoke_id: str,
        repl_history: list[object] | None = None,
    ) -> REPLIterationResult[Any]:
        # TODO: Fix static types
        # The query snapshot is the buffer itself — prompt additions arrive as message
        # edits folded at query time (see _compose_shown_messages), not baked in ahead.
        self._pending_message_edits = None
        self._last_llm_response = None
        self._pending_shown_messages = None
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
            can_recurse=can_recurse,
        )
        # Apply any persistent message edits (compaction) to the carried-forward buffer.
        self._apply_persistent_message_edits(messages, iteration)
        exec_timeout_override, enter_event = self._record_response_and_parse(
            messages, response_content, self.repl_template, invoke_id, iteration, depth
        )

        with self.dispatcher.span_repl_exec(enter_event) as span:
            # Apply AddVariables then DropVariables to the REPL namespace before this turn's
            # code runs. Adds bind raw names (no ``__jaz_get__``, prompt untouched) — the
            # per-turn namespace counterpart of the invoke-level AddInputs; drops unbind names
            # so a dropped name (e.g. __history__) reads as a genuine NameError. Applied
            # whether or not an override skips exec, so the namespace state is consistent for the
            # next turn either way (#568).
            #
            # DROPS RUN FIRST, mirroring ``_apply_input_effects``. Add-first made re-binding a
            # name impossible even though both this effect's docstring and
            # ``PythonREPL.add_variables`` documented the recipe: pairing a ``DropVariables``
            # with an ``AddVariables`` hit the already-bound check while the old value was still
            # there and raised ``REPLInputConflictError``, and on an unbound name the pair bound
            # then immediately unbound. Dropping first makes drop-then-add a re-bind, which is
            # the only way to replace a name a hook does not own.
            #
            # The trade is the same one ``AddInputs``/``DropInputs`` makes: a drop can no longer
            # remove a name another effect adds in the same turn (the add now wins). A drop that
            # outlives an add is the same mechanism that lets a stale drop erase one, and the two
            # cannot both hold in a single pass.
            self.repl_template.drop_variables(
                state,
                span.ctx.dropped_variables,
                allow_missing=span.ctx.dropped_variables_allow_missing,
            )
            self.repl_template.add_variables(state, span.ctx.added_variables)
            # REPLExecSend — fires iff the code committed. The namespace deltas above just
            # ran, so the execution's commit (code + committed namespace) is final. Send is
            # the supply boundary: suppliers (ValidateREPLCode, evals Restrict*) compose
            # here and the resolved short-circuit lands on span.supplied; an Abort raises
            # its carried exception out of span.send() instead (the span closes Aborted —
            # see span_repl_exec).
            span.send()
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
                # AddVariables/DropVariables above.)
                exec_result = self.repl_template.exec(
                    state, span.committed_code, str(iteration), exec_timeout_override
                )
            span.complete(exec_result=exec_result)

        return self._finalize_iteration(
            span, span.committed_code, state, response_content, repl_history
        )

    def _append_observation_messages(
        self,
        messages: list[MessageDict],
        exec_result: Continue,
        iteration: int,
    ) -> None:
        """Render the REPL observation via the protocol and append it to ``messages``.

        ``render_observation`` returns the observation as one or more ready messages, which
        append verbatim — persisted history stays pure protocol output, with no per-turn
        footer folded in. (#634 removed the budget-status / "enter your next REPL input"
        footer: the next-input nudge is unneeded scaffolding for the models JAZ targets, and
        persisting it froze a stale invoke-call count into every turn. The invoke-call budget
        could return as a transient per-query addition — see #657.) Each appended message is
        stamped with OBSERVATION provenance.
        """
        new_messages: list[MessageDict] = list(
            self.protocol.render_observation(exec_result, iteration)
        )
        for message in new_messages:
            # Stamp each observation message so consumers (ATIFTrace,
            # SlidingWindow, …) can identify it by provenance. #566 step C moved the
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
        resolved_inputs: dict[str, object],
        resolved_bound: dict[str, object],
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        """Fold an invoke's ``AddInputs`` / ``DropInputs`` effects into the input dicts *before*
        the REPL and prompt are built. **Drops are applied first, then adds**, so dropping and
        adding the same key replaces it. Added values are bound raw for the prompt (``inputs``)
        and ``resolve_inputs``-resolved for the REPL namespace (``resolved_bound``) — exactly how
        a top-level input is handled. An added key that collides with a caller-provided top-level
        input/scope entry raises (a hook may not silently override the caller). Dropped names are
        removed from all agent-facing dicts (``inputs``/``scope``/``resolved_inputs``/
        ``resolved_bound``), so the input is un-passed from the prompt and the REPL; dropping a
        name the invoke never received raises ``MissingDropTargetError`` unless that
        ``DropInputs`` opted into ``allow_missing``. Returns the updated ``(inputs, scope,
        resolved_inputs, resolved_bound)``; the inputs are copied only when an effect is present
        (the common no-effect path returns them unchanged)."""
        # ``resolved_inputs`` (the resolved EXPLICIT inputs, scope excluded) is threaded
        # through so the loop can hand ``InvokeSend`` the committed counterpart of
        # ``InvokeEnter.inputs`` without re-running ``resolve_inputs`` over the whole dict
        # (``__jaz_get__`` may be side-effecting, so values are resolved exactly once).
        added = ctx.added_inputs
        dropped = ctx.dropped_inputs
        if not added and not dropped:
            return inputs, scope, resolved_inputs, resolved_bound
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
            # "re-drop every turn" pattern), exactly as ``DropVariables`` does at the REPL layer and
            # as ``MissingDropTargetError``'s contract documents for ``DropInputs``. Double-drop
            # stays safe: composition already unioned the keys into one set (dispatcher
            # ``_compose_invoke``), so two hooks dropping the same present name coalesce here rather
            # than tripping this. Keys marked ``allow_missing`` (a ``DropInputs(...,
            # allow_missing=True)``, e.g. a defensive drop that tolerates another effect having
            # already un-passed the input) are exempt from the check.
            present = inputs.keys() | scope.keys() | resolved_bound.keys()
            missing = (dropped - present) - ctx.dropped_inputs_allow_missing
            if missing:
                raise MissingDropTargetError(
                    f"DropInputs target(s) {sorted(missing)!r} were not passed to this invoke; "
                    "dropping an absent input is a hook bug (drop it only when it is present, "
                    "e.g. once on the first InvokeEnter)."
                )
            inputs = {k: v for k, v in inputs.items() if k not in dropped}
            scope = {k: v for k, v in scope.items() if k not in dropped}
            resolved_inputs = {
                k: v for k, v in resolved_inputs.items() if k not in dropped
            }
            resolved_bound = {
                k: v for k, v in resolved_bound.items() if k not in dropped
            }
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
            inputs = {**inputs, **added}  # raw values render in the prompt
            # Resolve the added values exactly once and feed both resolved views from it
            # (see the resolved_inputs note above — __jaz_get__ may be side-effecting).
            resolved_added = resolve_inputs(added)
            resolved_inputs = {**resolved_inputs, **resolved_added}
            resolved_bound = {
                **resolved_bound,
                **resolved_added,
            }  # resolved for the REPL
        return inputs, scope, resolved_inputs, resolved_bound

    def invoke(
        self,
        *,
        invoke_tool: InvokeTool | None,
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
        # Resolve `__jaz_get__` payload substitution once, up front, via the core
        # `resolve_inputs` helper: the event and every REPL want the value the agent
        # actually binds, not the metadata wrapper (a `jaz.Display` directive, a
        # `Library`). The prompt builder keeps the original `inputs`/`scope` instead — it
        # needs the wrappers to read their descriptions (`__jaz_description__`) — and
        # resolves `__jaz_get__` itself only for the type label / default rendering. This
        # single core resolution replaced both the old display-directive-specific unwrap step and
        # the REPL-level resolution that #510 introduced before this moved into the loop.
        #
        # `inputs` (explicit kwargs) and `scope` (resolved `jaz.scope`) are kept as SEPARATE
        # provenance channels (#727): resolve each independently for the event, and merge
        # into the single namespace the REPL actually binds only here (`resolved_bound`).
        # They are disjoint (invoke.py's conflict check guarantees it), so the merge is
        # order-independent and nothing is lost by keeping the two apart — this is what lets
        # the event carry them unmerged with no downstream re-split. `scope` defaults to
        # empty for any direct caller; invoke.py always passes it.
        scope = scope or {}
        resolved_inputs = resolve_inputs(inputs)
        resolved_scope = resolve_inputs(scope)
        resolved_bound = {**resolved_scope, **resolved_inputs}
        # Seed the per-invoke blackboard (generation 0) BEFORE any event fires, so
        # InvokeEnter consumers already see seeded keys. seed_blackboard binds it to
        # this invoke's dispatcher, which then stamps it onto every event in emit().
        # Seeds come from active hooks' blackboard_seed() (e.g. a MetaData hook
        # carrying `task_name`), including per-call local hooks — agents can now pass
        # `jaz.MetaData(...)` positionally to the agent-facing invoke (#545). A direct
        # caller-supplied `metadata=` seed channel is still deferred to #538 (the
        # public invoke-syntax issue; #537's hook-side mechanism landed as
        # `MetaData`, #545).
        self.dispatcher.seed_blackboard(self.config)
        # Use dispatcher span for invoke
        enter_event = InvokeEnter(
            config=self.config,
            invoke_id=invoke_id,
            parent_invoke_id=parent_invoke_id,
            parent_repl_iteration=parent_repl_iteration,
            inputs=resolved_inputs,
            scope=resolved_scope,
            depth=depth,
            invoke_tool=invoke_tool,
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
            # adds, so a key both added and dropped ends added. Added values go through the
            # same ``resolve_inputs`` (__jaz_get__) resolution as top-level inputs for the REPL
            # binding, while the raw value renders in the prompt (matching top-level inputs).
            inputs, scope, resolved_inputs, resolved_bound = self._apply_input_effects(
                span.ctx, inputs, scope, resolved_inputs, resolved_bound
            )
            # InvokeSend — fires iff the input committed. The add/drop fold above IS the
            # commit of this invoke's input set, so Send fires here with the committed
            # resolved inputs (what InvokeEnter.inputs would have been had it carried the
            # post-effect view). It does not fire on an enter-time abort — that raised
            # before the fold, so the input set never commits. An Abort @ InvokeSend
            # raises out of this call (the committed-input veto; the span closed Aborted).
            span.send(resolved_inputs)

            # `invoke_tool` carries only the framework's recursive sub-invoke primitive (a
            # single callable, bound in the REPL as the bare name `invoke`; user tool
            # namespaces are ordinary inputs — see library_as_input.md). Hooks that need to
            # inject a namespace use AddInputs, not a tool effect.

            # Honor a DisableRecursion effect emitted at InvokeEnter (by RecursionLimit at the
            # cap leaf) by withholding the tool entirely: at the leaf there is nothing left to
            # bind, since the primitive IS the whole surface (#635 moved the former jaz.*
            # helpers out to ordinary inputs). Both consumers below (REPL binding + the prompt
            # that advertises the tool) must see the same value, or they would disagree —
            # a prompt documenting `invoke` over a REPL that has no such name.
            # `can_recurse` is tracked separately from `effective_invoke_tool is None` because
            # it also rides LLMQueryEnter, where the must-exit warnings use it to gate their
            # delegate guidance; with no RecursionLimit installed it is always True.
            effective_invoke_tool = None if span.ctx.recursion_disabled else invoke_tool
            can_recurse = not span.ctx.recursion_disabled

            # Generate unique session ID to prevent linecache collisions between
            # sessions
            session_id = uuid.uuid4().hex[:8]

            # Get REPL configs from Config

            # __history__ is owned by the core agent loop (the single writer): when
            # the REPL maintains history, build the full list here (index-0 init snapshot
            # included, using core-resolved inputs), pass it into initialize() (the REPL
            # surfaces it by reference), and retain the reference to append iteration
            # entries in the loop. A REPL that declares no history support is skipped via
            # the maintains_repl_history capability flag.
            repl_history: list[object] | None = (
                # Empty list = the whole initial history (no index-0 init snapshot, #568). The core
                # loop is the single writer; the REPL surfaces this list as __history__ by ref.
                []
                # __history__ is always maintained now (#568 removed the repl_history
                # flag); only a REPL with maintains_repl_history=False gets None.
                if type(self.repl_template).maintains_repl_history
                else None
            )
            # Build the REPL state and the opening prompt only when no result was supplied at
            # Send: a SupplyInvokeResult short-circuits the whole invoke, and `initialize` resolves
            # the invoke's inputs (possibly with I/O) while `render_initial_message_list` can run
            # arbitrary custom-protocol work — neither should run for a supplied invoke, and a
            # failure building them must not discard the supplied result. They stay None on that
            # path; the loop below breaks on `span.supplied` before it would read either.
            state: REPLState | None = None
            messages: list[MessageDict] | None = None
            if span.supplied is None:
                # The configured REPL is a stateless template; `initialize` returns this run's
                # `REPLState` (#1058), threaded into every exec below. The protocol opener still
                # gets the config template (it reads only config: description, history capability).
                state = self.repl_template.initialize(
                    inputs=resolved_bound,
                    invoke_tool=effective_invoke_tool,
                    session_id=session_id,
                    initial_repl_history=repl_history,
                )

                messages = self.protocol.render_initial_message_list(
                    inputs,
                    scope,
                    self.repl_template,
                    invoke_tool=effective_invoke_tool,
                    depth=depth,
                    recursion_available=can_recurse,
                    repl_language=self.repl_language,
                    input_display_overrides=None,
                )
                # A custom BaseProtocol must return a non-empty opener. Guard it loudly
                # here (the load-bearing half of the old iteration-0 footer assert, #705): an
                # empty list otherwise slips past _stamp_seed_provenance and surfaces later as a
                # muddier empty-message-list error at the first query.
                assert messages, "render_initial_message_list returned an empty opener"

                self._stamp_seed_provenance(messages)

            # (Invoke-level AddInputs/DropInputs were already folded into inputs/scope/
            # resolved_bound above via _apply_input_effects, before the REPL and prompt were
            # built — so added inputs are bound in `repl` and rendered in `messages`, and
            # dropped inputs are absent from both. No post-init REPL mutation needed.)

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
            # ``__history__`` is a plain 0-based list holding exactly one entry per iteration
            # (see ``_record_history``), so iteration N is now ``__history__[N]``. Note this is
            # alignment, not a contract anything relies on: nothing in-tree addresses history by
            # iteration number, and the heaviest consumers (the oolong/stulife history-search
            # prompts) scan ``prev_repl_history + __history__[:-1]``, a concatenation across
            # agent boundaries whose positions cannot correspond to this agent's turns under any
            # base. The truncation advice still says ``[-1]`` for that reason.
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

                repl_iter_result = self.do_one_repl_iteration(
                    i,
                    depth,
                    can_recurse,
                    messages,
                    state,
                    invoke_id,
                    repl_history=repl_history,
                )

                exec_result = repl_iter_result.exec_result
                state = repl_iter_result.state

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
        can_recurse: bool,
    ) -> str:
        # Mirrors _query_with_messages exactly except for the awaited LLM call; the edit
        # fold is the shared _compose_shown_messages, so the two paths cannot drift. An
        # Abort at any stage raises out of this method — see the sync path's comment.
        complete_fn, invoke_id, enter_event = self._make_llm_retry_fn(
            self.llm_client.acomplete_with_retry,
            messages,
            invoke_id,
            iteration,
            depth,
            can_recurse,
        )
        with self.dispatcher.span_llm_query(enter_event) as span:
            shown = self._compose_shown_messages(span, messages)
            # LLMQuerySend — fires iff the input committed (see the sync path's comment):
            # unconditionally after the edit fold, on the supplied path too; never on an
            # enter-time abort (which raised before this body ran).
            span.send(shown)
            # Send-supplied response (e.g. Replay): the provider call is skipped.
            if span.supplied_response is not None:
                llm_response: LLMResponse = span.supplied_response
            else:
                # Provider errors propagate; see the sync path (incl. the TODO(#697)
                # future-policy note).
                llm_response = await complete_fn(
                    model=self.model,
                    messages=shown,
                    **self.llm_client.request_defaults,
                )
            result = self._record_llm_response(llm_response, span)
        # Span closed: a Complete-stage ModifyLLMResponse may have replaced the response
        # (authoritative — see resolve_llm_modify_results). If so, re-read it: its content is what
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
        can_recurse: bool,
        messages: list[MessageDict],
        state: REPLState,
        invoke_id: str,
        repl_history: list[object] | None = None,
    ) -> REPLIterationResult[Any]:
        # Mirrors do_one_repl_iteration exactly except for the two awaited steps — the
        # LLM query above and repl.aexec below. Everything else is the shared
        # _record_response_and_parse / _finalize_iteration, so the paths cannot drift.
        # (The async copy previously diverged: aexec ran *outside* the enter-override
        # `else` and so clobbered enter-time overrides (#592), and it never applied
        # exit-time overrides nor finalized history — all fixed by routing through the
        # shared helpers.)
        self._pending_message_edits = None
        self._last_llm_response = None
        self._pending_shown_messages = None
        # No catch around the query: LLM-call failures and abort-carried exceptions
        # propagate — see the sync path's comment on the deleted #687 conversion.
        response_content = await self._aquery_with_messages(
            messages,
            iteration=iteration,
            depth=depth,
            invoke_id=invoke_id,
            can_recurse=can_recurse,
        )
        # Apply any persistent message edits (compaction) to the carried-forward buffer.
        self._apply_persistent_message_edits(messages, iteration)
        exec_timeout_override, enter_event = self._record_response_and_parse(
            messages, response_content, self.repl_template, invoke_id, iteration, depth
        )

        with self.dispatcher.span_repl_exec(enter_event) as span:
            # Apply DropVariables then AddVariables (see the sync path for rationale): unbind the
            # composed names, then bind, before this turn's code runs — drop-first so a
            # drop-then-add pair re-binds; consistent across the override/exec branches.
            self.repl_template.drop_variables(
                state,
                span.ctx.dropped_variables,
                allow_missing=span.ctx.dropped_variables_allow_missing,
            )
            self.repl_template.add_variables(state, span.ctx.added_variables)
            # REPLExecSend — fires iff the code committed (see the sync path's comment):
            # after the namespace deltas; suppliers compose here and the resolved
            # short-circuit lands on span.supplied (an Abort raises out of span.send).
            span.send()
            if span.supplied is not None:
                # A Send-composed SupplyExecResult short-circuits execution: skip running
                # the code and use the supplied result instead.
                exec_result = span.supplied
            else:
                # Execute the COMMITTED code — the enter code after any InsertCode/DeleteCode
                # edits (== the original when no hook edited it), the same string REPLExecSend
                # observers saw. (Invoke inputs are folded in once at InvokeEnter, not here:
                # AddInputs/DropInputs are InvokeEnter-only — #481. Per-turn namespace binds go
                # through AddVariables/DropVariables above.)
                exec_result = await self.repl_template.aexec(
                    state, span.committed_code, str(iteration), exec_timeout_override
                )
            span.complete(exec_result=exec_result)

        return self._finalize_iteration(
            span, span.committed_code, state, response_content, repl_history
        )

    async def ainvoke(
        self,
        *,
        invoke_tool: InvokeTool | None,
        depth: int,
        invoke_id: str,
        parent_invoke_id: str | None,
        parent_repl_iteration: int | None,
        scope: dict[str, object] | None = None,
        inputs: dict[str, object],
    ) -> object:
        # The return type is a ReturnType(...) hook now (#568), not a param — see `invoke`.
        # Resolve `__jaz_get__` once for the event and REPL (the value the agent
        # binds, not the wrapper) via the core `resolve_inputs` helper; the prompt
        # builder keeps the original `inputs`/`scope` to read wrapper descriptions.
        # Mirrors the sync `invoke` path — including keeping `inputs` (explicit kwargs)
        # and `scope` (resolved `jaz.scope`) as SEPARATE channels (#727), merged into the
        # single REPL namespace only via `resolved_bound`; see sync `invoke` for the full
        # rationale. This is also what now makes `jaz.Display` work here, which the old
        # unwrap step never covered.
        scope = scope or {}
        resolved_inputs = resolve_inputs(inputs)
        resolved_scope = resolve_inputs(scope)
        resolved_bound = {**resolved_scope, **resolved_inputs}
        # Seed the per-invoke blackboard (generation 0) before any event fires; see
        # the sync invoke() for rationale and the #538 TODO on caller metadata.
        self.dispatcher.seed_blackboard(self.config)
        enter_event = InvokeEnter(
            config=self.config,
            invoke_id=invoke_id,
            parent_invoke_id=parent_invoke_id,
            parent_repl_iteration=parent_repl_iteration,
            inputs=resolved_inputs,
            scope=resolved_scope,
            depth=depth,
            invoke_tool=invoke_tool,
        )
        # An Abort at InvokeEnter raises from the `with` statement itself — see the sync
        # path's comment.
        with self.dispatcher.span_invoke(enter_event) as span:
            # Apply invoke-level input effects (AddInputs / DropInputs) before the REPL and
            # prompt are built — the async mirror of the sync path (see there for the full
            # rationale). Adds render+bind, drops un-pass from both; drops run first, so an add
            # wins on overlap.
            inputs, scope, resolved_inputs, resolved_bound = self._apply_input_effects(
                span.ctx, inputs, scope, resolved_inputs, resolved_bound
            )
            # InvokeSend — fires iff the input committed (see the sync path's comment):
            # right after the add/drop fold, never on an enter-time abort. An Abort here
            # raises out of this call (the span closed Aborted).
            span.send(resolved_inputs)

            # `invoke_tool` carries only the framework's recursive sub-invoke primitive (a
            # single callable, or None; see the sync path above and library_as_input.md).

            # Honor DisableRecursion (RecursionLimit at the cap leaf) — same as the sync path:
            # withhold the tool entirely, and track can_recurse separately because it also
            # rides LLMQueryEnter.
            effective_invoke_tool = None if span.ctx.recursion_disabled else invoke_tool
            can_recurse = not span.ctx.recursion_disabled

            session_id = uuid.uuid4().hex[:8]

            # Single-writer __history__ ownership (see the sync invoke()).
            repl_history: list[object] | None = (
                # Empty list = the whole initial history (no index-0 init snapshot, #568). The core
                # loop is the single writer; the REPL surfaces this list as __history__ by ref.
                []
                # __history__ is always maintained now (#568 removed the repl_history
                # flag); only a REPL with maintains_repl_history=False gets None.
                if type(self.repl_template).maintains_repl_history
                else None
            )
            # Build the REPL state and the opening prompt only when no result was supplied at Send
            # (the sync path documents why); they stay None for a supplied invoke, which the loop
            # below short-circuits before reading either.
            state: REPLState | None = None
            messages: list[MessageDict] | None = None
            if span.supplied is None:
                # The configured REPL is a stateless template; `initialize` returns this run's
                # `REPLState` (#1058). The protocol opener still gets the config template.
                state = self.repl_template.initialize(
                    inputs=resolved_bound,
                    invoke_tool=effective_invoke_tool,
                    session_id=session_id,
                    initial_repl_history=repl_history,
                )

                # NOTE: the async path historically does NOT unwrap jaz.Display display
                # overrides (only sync invoke() calls unwrap_display), so it passes
                # input_display_overrides=None — preserved here verbatim. Unifying the
                # sync/async paths is tracked separately (see #566 roadmap).
                messages = self.protocol.render_initial_message_list(
                    inputs,
                    scope,
                    self.repl_template,
                    invoke_tool=effective_invoke_tool,
                    depth=depth,
                    recursion_available=can_recurse,
                    repl_language=self.repl_language,
                    input_display_overrides=None,
                )
                # A custom BaseProtocol must return a non-empty opener. Guard it loudly
                # here (the load-bearing half of the old iteration-0 footer assert, #705): an
                # empty list otherwise slips past _stamp_seed_provenance and surfaces later as a
                # muddier empty-message-list error at the first query.
                assert messages, "render_initial_message_list returned an empty opener"

                self._stamp_seed_provenance(messages)

            # (AddInputs/DropInputs already folded into inputs/resolved_bound above via
            # _apply_input_effects — no post-init REPL mutation needed; see sync invoke().)

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

                repl_iter_result = await self.do_one_repl_iteration_async(
                    i,
                    depth,
                    can_recurse,
                    messages,
                    state,
                    invoke_id,
                    repl_history=repl_history,
                )

                exec_result = repl_iter_result.exec_result
                state = repl_iter_result.state

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
