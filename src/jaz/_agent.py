import functools
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from .invoke import Prehook

from tenacity import RetryCallState

from ._llm_client import LLM, LLMResponse
from .config import Config
from .exceptions import (
    LLMResponseParseError,
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
from .library import Library
from .protocol import InteractionProtocol
from .provenance import (
    MessageKind,
    MessageProvenance,
    set_provenance,
    to_wire_messages,
)
from .providers import MessageDict
from .repl.base import REPL
from .repl.registry import REPL_LANGUAGE_MAP
from .repl.types import (
    Continue,
    ExecResult,
    Raise,
    Return,
)
from .string_utils import summarize_exception


class REPLIterationResult[ReturnT](NamedTuple):
    """Result from a single REPL iteration.

    Attributes:
        exec_result: The execution result from the REPL
        repl: The REPL used for this iteration
        response_content: The LLM response content (None if empty)
        code: The extracted code from the response (None if not found)
    """

    exec_result: ExecResult[ReturnT]
    repl: REPL
    response_content: str | None
    code: str | None


def _parse_error_result(e: LLMResponseParseError) -> Continue:
    """The recoverable ``Continue`` a parse error is surfaced to the model as.

    Centralizes the shape (the raw exception attached — #566 step C) so the sync/async
    parse-catch sites can't drift. ``output`` carries the *rendered* error text (via
    ``summarize_exception``), not an empty string: a parse failure runs no code, so there is no
    ``stdout + traceback`` — populating ``output`` keeps the ``Continue`` contract uniform, that
    ``output`` is the agent-facing text and ``exception`` is the structured metadata alongside it.
    Without this, a parse-failure history entry had an empty ``repl_output``, so the truncation-
    advice pointer at ``__repl_history__[i].repl_output`` led the agent to an empty string.

    Settled by #928: ``render_observation`` now shows ``output`` as the sole rendered text and
    treats ``exception`` as metadata, so populating ``output`` here is what the agent actually
    sees on a parse failure — not a second copy of an ``<error>`` block. The rendering is a
    *protocol* concern; this function's only job is to honor the ``Continue`` contract.
    """
    return Continue(output=summarize_exception(e), exception=e)


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
        # The configured REPL is a *template*: `initialize` returns a fresh instance per invoke
        # and leaves this one clean, so one config can serve many invokes (#1056).
        self.repl_template: REPL = self.config.repl
        self.llm_client: LLM = self.config.llm
        self.model: str = self.llm_client.get_model()
        # Resolved once here, not per turn: both the client and the model are fixed for this
        # agent's lifetime, so the answer cannot change between iterations.
        self.can_report_cost: bool = self.llm_client.can_report_cost()
        self.protocol: InteractionProtocol = self.config.protocol
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
        iteration: int,
        depth: int,
        start_time: datetime,
        end_time: datetime,
    ) -> str | None:
        """Track token usage/cost, complete the span, and return response content."""
        # Complete the span with the response object + query timing. Cost tracking +
        # enforcement is handled by BudgetPool (opt-in) observing LLMQueryExit.
        span.complete(
            response=llm_response,
            iteration=iteration,
            start_time=start_time,
            end_time=end_time,
        )
        # Stash the whole object for _record_history's projection (see the field's init comment);
        # returning only .content keeps _query_with_messages's content-str contract intact.
        self._last_llm_response = llm_response
        return llm_response.content

    def _query_with_messages(
        self,
        messages: list[MessageDict],
        *,
        iteration: int,
        depth: int,
        invoke_id: str,
        can_recurse: bool,
    ) -> str | Raise | None:
        complete_fn, invoke_id, enter_event = self._make_llm_retry_fn(
            self.llm_client.complete_with_retry,
            messages,
            invoke_id,
            iteration,
            depth,
            can_recurse,
        )
        with self.dispatcher.span_llm_query(enter_event) as span:
            # Abort @ LLMQueryEnter (iteration / budget hard-stop): terminate the invoke.
            # Return the terminal Raise; the query never happens and no LLMQueryExit fires.
            # The caller (do_one_repl_iteration) turns it into the iteration's Raise result.
            if span.abort is not None:
                assert isinstance(span.abort, Raise)
                return span.abort
            shown = self._compose_shown_messages(span, messages)
            # Check for override response from hooks (e.g., Replay)
            if span.ctx.override_response is not None:
                llm_response: LLMResponse = span.ctx.override_response
                start_time = end_time = datetime.now()
            else:
                start_time = datetime.now()
                llm_response = complete_fn(
                    model=self.model,
                    messages=shown,
                    **self.llm_client.request_defaults,
                )
                end_time = datetime.now()
            return self._record_llm_response(
                llm_response, span, iteration, depth, start_time, end_time
            )

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
        to wire form (``to_wire_messages``) so the internal per-message provenance key never
        reaches a model. Both the sync and async query paths route through here, so one
        strip covers both regardless of the underlying LLM client.
        """
        edits = span.ctx.message_edits
        self._pending_message_edits = edits
        shown = apply_message_edits(messages, edits.all_drops, edits.all_adds)
        return to_wire_messages(shown)

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
        simpler and correct for any protocol: ``DefaultProtocol`` renders ``[system, user]``,
        but a custom protocol may seed extra context (e.g. few-shot ``assistant`` examples),
        and those are just as much the seed. Role no longer needs to disambiguate a kind
        (that was only for the former split SYSTEM/INPUT_PROMPT kinds); a consumer that wants
        the system prompt vs the user prompt reads the surviving message's ``role``."""
        for message in messages:
            set_provenance(message, MessageProvenance(MessageKind.SEED))

    def _record_response_and_parse(
        self,
        messages: list[MessageDict],
        response_content: str | None,
        repl: REPL,
        invoke_id: str,
        iteration: int,
        depth: int,
        repl_history: list[object] | None,
    ) -> tuple[str, float | None, REPLExecEnter] | REPLIterationResult[Any]:
        """Record the assistant response and decode it into an execution plan.

        Shared by the sync and async iteration paths — which differ *only* in how the
        response is fetched and how the code is executed; everything around that is
        identical and lives here (and in :meth:`_finalize_iteration`) so the two paths
        cannot drift. Appends the assistant message, then parses. On a recoverable
        ``LLMResponseParseError`` returns a ready
        ``REPLIterationResult`` (feedback for the model) — the catch is deliberately
        NARROW: any other exception from a custom protocol is terminal and propagates,
        and ``code`` is left None since the attempt is recorded via ``response_content``
        (#566 step B). Otherwise returns ``(code, exec_timeout_override, enter_event)``.

        Callers discriminate the two outcomes with ``isinstance(parsed, REPLIterationResult)``.
        That only works because the "continue" branch is a *plain* tuple, never a
        ``NamedTuple``: a plain tuple is never an instance of the ``REPLIterationResult``
        subclass, however tuple-shaped it is. If the continue payload is ever upgraded to
        its own ``NamedTuple`` (or otherwise wrapped), the ``isinstance`` check could start
        matching it too — keep it a plain tuple, or switch callers to an explicit tag.
        """
        messages.append({"role": "assistant", "content": response_content})
        set_provenance(
            messages[-1],
            MessageProvenance(MessageKind.ASSISTANT, iteration=iteration),
        )

        try:
            code, exec_timeout_override = self.protocol.parse(
                response_content, self.repl_language
            )
        except LLMResponseParseError as e:
            # A parse failure is still a real turn the LLM took — record it in history (the
            # full unparseable response + the parse error surfaced as a recoverable Continue),
            # so the record is complete and stays index-aligned with the loop's iterations.
            parse_error_result = _parse_error_result(e)
            self._record_history(
                repl_history,
                self._llm_response_to_record(response_content),
                parse_error_result,
            )
            return REPLIterationResult(
                exec_result=parse_error_result,
                repl=repl,
                response_content=response_content,
                code=None,
            )

        enter_event = REPLExecEnter(
            config=self.config,
            invoke_id=invoke_id,
            iteration=iteration,
            code=code,
            depth=depth,
        )
        return code, exec_timeout_override, enter_event

    def _finalize_iteration(
        self,
        span: REPLExecSpan,
        code: str,
        repl: REPL,
        response_content: str | None,
        repl_history: list[object] | None,
    ) -> REPLIterationResult[Any]:
        """Apply exit-time overrides and append the iteration's history entry.

        Shared by both iteration paths. ``span.get_final_exec_result()`` folds in any
        exit-time ModifyResult / Abort; the core loop is the single writer
        of ``__repl_history__`` and appends one entry per iteration from that FINAL
        (post-override) result — so a result-override hook (e.g. BudgetForcing) is
        reflected in history without the REPL knowing (#448), and an enter-time override
        that skipped exec() still records an entry. ``repl_history`` is the same list the
        REPL surfaced as ``__repl_history__`` (by reference); None means no history.
        """
        exec_result = span.get_final_exec_result()
        self._record_history(
            repl_history,
            self._llm_response_to_record(response_content),
            exec_result,
        )
        return REPLIterationResult(
            exec_result=exec_result,
            repl=repl,
            response_content=response_content,
            code=code,
        )

    def _llm_response_to_record(
        self, response_content: str | None
    ) -> LLMResponse | None:
        """The ``LLMResponse`` to record for the current turn, or ``None`` to skip (no response).

        Precedence is intentional: the whole ``_last_llm_response`` object — stashed by
        ``_record_llm_response`` on the real query path — is the source of truth (its ``.content``
        text AND token/cost/provider ``extra``). ``response_content`` is only the no-response signal
        and the mocked-path fallback: it is ``None`` on the no-response terminal paths (a failed LLM
        call, or an ``Abort`` at ``LLMQueryEnter``), so we return ``None`` and the caller skips; and
        on the mocked path (tests patch ``_query_with_messages`` to return a bare content string, so
        ``_record_llm_response`` never runs and the object stays ``None``) we synthesize a minimal
        ``LLMResponse`` from the string so the default projection is byte-identical. On the real path
        the two agree by construction (``response_content`` IS this object's ``.content``); the one
        way they could diverge — a future hook rewriting ``response_content`` post-query without
        touching the side-channel — resolves to the raw provider object on purpose (history records
        what the model actually returned, so a content-rewriting hook updates the object, not just
        the string).
        """
        if self._last_llm_response is not None:
            return self._last_llm_response
        if response_content is not None:
            return LLMResponse(content=response_content)
        return None

    def _record_history(
        self,
        repl_history: list[object] | None,
        llm_response: LLMResponse | None,
        exec_result: ExecResult,
    ) -> None:
        """Append one ``__repl_history__`` entry for a turn that produced an LLM response.

        Records **every** turn the LLM actually responded to, storing the FULL response
        (``REPLHistoryEntry.llm_response`` — the raw code incl. its leading comments), not the parsed
        code. This includes a turn whose response **failed to parse**: that is still a real turn the
        LLM took, and its entry carries the unparseable response plus the parse-error exception in
        ``repl_exception`` (``exec_result`` there is the recoverable ``Continue`` the parse error was
        surfaced as). A meta-agent analyzing behavior therefore sees malformed turns too, and the
        per-turn index stays aligned with the loop's iteration count.

        ``llm_response`` (resolved by :meth:`_llm_response_to_record`) is the whole ``LLMResponse``,
        not the bare content string, so the configurable ``self.protocol.build_history_entry`` seam
        can record a richer entry (#566). It is ``None`` in exactly one case — an **empty LLM
        completion**: ``_query_with_messages`` returns ``None``, which ``_record_response_and_parse``
        routes through its parse-failure branch (``parse(None)`` raises "empty response"), and with no
        response object stashed either, ``_llm_response_to_record`` returns ``None``. We skip it —
        there is nothing to record (so the index skips that one empty-completion turn). The genuinely-
        no-response terminal paths — a failed LLM call, or an ``Abort`` at ``LLMQueryEnter`` — return
        *earlier* in ``do_one_repl_iteration`` and never reach here.
        """
        if repl_history is None or llm_response is None:
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
        repl: REPL,
        invoke_id: str,
        repl_history: list[object] | None = None,
    ) -> REPLIterationResult[Any]:
        # TODO: Fix static types
        # The query snapshot is the buffer itself — prompt additions arrive as message
        # edits folded at query time (see _compose_shown_messages), not baked in ahead.
        self._pending_message_edits = None
        self._last_llm_response = None
        try:
            response_content = self._query_with_messages(
                messages,
                iteration=iteration,
                depth=depth,
                invoke_id=invoke_id,
                can_recurse=can_recurse,
            )
        except Exception as e:
            # #687: a failed LLM call (e.g. a missing API key surfaced as
            # AuthenticationError, or any error still raised after the retry policy) becomes
            # a terminal Raise instead of propagating as an uncaught exception. It then
            # flows through the agent's normal terminal-result path — hooks observe it and
            # the invoke surfaces it — exactly like a code-level `raise`.
            # `except Exception` (NOT BaseException) is deliberate — do not widen it:
            # genuinely-fatal non-Exception signals (KeyboardInterrupt, SystemExit,
            # GeneratorExit, CancelledError) must keep propagating and aborting directly;
            # re-wrapping them into a Raise would be pointless and could interfere with
            # abort semantics.
            # An AbortError arrives here too (it is Exception-rooted) and is deliberately
            # NOT special-cased: it becomes a terminal Raise like any other error, and the
            # invoke surfaces it, which is exactly the semantics wanted — a Raise carrying an
            # AbortError ends this invoke and keeps propagating through enclosing ones. Routing
            # it through this path rather than re-raising is what lets `span.complete()` run, so
            # `InvokeExit` still fires; an earlier `except AbortError: raise` here skipped both.
            # TODO(#697): route selected LLM errors to a recoverable Continue or an
            # AbortError instead of always Raise (slots in here without touching
            # the Exception-only breadth).
            return REPLIterationResult(
                exec_result=Raise(exception=e),
                repl=repl,
                response_content=None,
                code=None,
            )
        # Abort @ LLMQueryEnter (iteration / budget hard-stop) surfaced as a terminal Raise:
        # this is the loop's per-turn termination point, replacing the old RaiseEffect @
        # REPLIterationEnter. Propagate it as the iteration's result; the loop raises it.
        if isinstance(response_content, Raise):
            return REPLIterationResult(
                exec_result=response_content,
                repl=repl,
                response_content=None,
                code=None,
            )
        # Apply any persistent message edits (compaction) to the carried-forward buffer.
        self._apply_persistent_message_edits(messages, iteration)
        parsed = self._record_response_and_parse(
            messages, response_content, repl, invoke_id, iteration, depth, repl_history
        )
        if isinstance(parsed, REPLIterationResult):
            return parsed
        code, exec_timeout_override, enter_event = parsed

        with self.dispatcher.span_repl_exec(enter_event) as span:
            # Apply AddVariables then DropVariables to the REPL namespace before this turn's
            # code runs. Adds bind raw names (no ``__jaz_get__``, prompt untouched) — the
            # per-turn namespace counterpart of the invoke-level AddInputs; drops unbind names
            # so a dropped name (e.g. __repl_history__) reads as a genuine NameError. Applied
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
            repl.drop_variables(
                span.ctx.dropped_variables,
                allow_missing=span.ctx.dropped_variables_allow_missing,
            )
            repl.add_variables(span.ctx.added_variables)
            if span.enter_override is not None:
                # An enter-time OverrideResult (supply) / Abort short-circuits execution:
                # skip running the code and use the composed result instead.
                exec_result = span.enter_override
            else:
                # Execute the code. (Invoke inputs are folded in once at InvokeEnter, not here:
                # AddInputs/DropInputs are InvokeEnter-only — #481. Per-turn namespace binds go
                # through AddVariables/DropVariables above.)
                exec_result = repl.exec(code, str(iteration), exec_timeout_override)
            span.complete(exec_result=exec_result)

        return self._finalize_iteration(
            span, code, repl, response_content, repl_history
        )

    def _append_observation_messages(
        self,
        messages: list[MessageDict],
        exec_result: Continue,
        repl: REPL,
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
            self.protocol.render_observation(exec_result, iteration, repl)
        )
        for message in new_messages:
            # Stamp each observation message so consumers (ConversationHistory,
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
        resolved_bound: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        """Fold an invoke's ``AddInputs`` / ``DropInputs`` effects into the input dicts *before*
        the REPL and prompt are built. **Drops are applied first, then adds**, so dropping and
        adding the same key replaces it. Added values are bound raw for the prompt (``inputs``)
        and ``resolve_inputs``-resolved for the REPL namespace (``resolved_bound``) — exactly how
        a top-level input is handled. An added key that collides with a caller-provided top-level
        input/scope entry raises (a hook may not silently override the caller). Dropped names are
        removed from all three agent-facing dicts (``inputs``/``scope``/``resolved_bound``), so the
        input is un-passed from the prompt and the REPL; dropping a name the invoke never received
        raises ``MissingDropTargetError`` unless that ``DropInputs`` opted into ``allow_missing``.
        Returns the updated ``(inputs, scope, resolved_bound)``; the inputs are copied only when an
        effect is present (the common no-effect path returns them unchanged)."""
        added = ctx.added_inputs
        dropped = ctx.dropped_inputs
        if not added and not dropped:
            return inputs, scope, resolved_bound
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
            resolved_bound = {
                **resolved_bound,
                **resolve_inputs(added),
            }  # resolved for the REPL
        return inputs, scope, resolved_bound

    def invoke(
        self,
        *,
        jaz_library: Library | None,
        jaz_library_no_invoke: Library | None,
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
            jaz_library=jaz_library,
        )
        with self.dispatcher.span_invoke(enter_event) as span:
            # A hook may abort the whole invoke before any execution (e.g. an
            # inherited cost-pool budget is already exhausted). enter_override is
            # a Raise; short-circuit the loop and raise it.
            if span.enter_override is not None:
                assert isinstance(span.enter_override, Raise)
                span.complete(result=span.enter_override)
                raise span.enter_override.exception

            # Apply invoke-level input effects (AddInputs / DropInputs) to the invoke's inputs
            # BEFORE the REPL and prompt are built, so an added input renders in the prompt AND
            # binds in the REPL, and a dropped input is un-passed from both — the "inputs"
            # (prompt-level) half of the symmetric effect family. Drops are applied first, then
            # adds, so a key both added and dropped ends added. Added values go through the
            # same ``resolve_inputs`` (__jaz_get__) resolution as top-level inputs for the REPL
            # binding, while the raw value renders in the prompt (matching top-level inputs).
            inputs, scope, resolved_bound = self._apply_input_effects(
                span.ctx, inputs, scope, resolved_bound
            )

            # `jaz_library` carries only the framework JAZ library (a single
            # Library or None; user tool namespaces are ordinary inputs — see
            # library_as_input.md). Hooks that need to inject a namespace use
            # AddInputs, not a library effect.

            # Honor a DisableRecursion effect emitted at InvokeEnter (by RecursionLimit at the
            # cap leaf): bind the REPL and render the system prompt with the cap-leaf library —
            # the same opted-in jaz.* helpers minus the recursive jaz.invoke tool (#635), or
            # None when nothing is opted in. Both consumers below (REPL binding + the prompt
            # that advertises the tools) must see the same library, or they would disagree.
            # `can_recurse` keys on the DisableRecursion effect, NOT on the library being None:
            # the cap-leaf library can be non-None (it still has the helpers) yet recursion is
            # off. It rides LLMQueryEnter so the must-exit warnings gate their delegate
            # guidance; with no RecursionLimit installed it is always True.
            effective_jaz_library = (
                jaz_library_no_invoke if span.ctx.recursion_disabled else jaz_library
            )
            can_recurse = not span.ctx.recursion_disabled

            # Generate unique session ID to prevent linecache collisions between
            # sessions
            session_id = uuid.uuid4().hex[:8]

            # Get REPL configs from Config

            # __repl_history__ is owned by the core agent loop (the single writer): when
            # the REPL maintains history, build the full list here (index-0 init snapshot
            # included, using core-resolved inputs), pass it into initialize() (the REPL
            # surfaces it by reference), and retain the reference to append iteration
            # entries in the loop. A REPL that declares no history support is skipped via
            # the maintains_repl_history capability flag.
            repl_history: list[object] | None = (
                # Empty list = the whole initial history (no index-0 init snapshot, #568). The core
                # loop is the single writer; the REPL surfaces this list as __repl_history__ by ref.
                []
                # __repl_history__ is always maintained now (#568 removed the repl_history
                # flag); only a REPL with maintains_repl_history=False gets None.
                if type(self.repl_template).maintains_repl_history
                else None
            )
            # The configured REPL comes off the config; `initialize` returns a fresh instance
            # carrying this run's state and leaves the template clean for the next invoke.
            repl = self.repl_template.initialize(
                inputs=resolved_bound,
                jaz_library=effective_jaz_library,
                session_id=session_id,
                initial_repl_history=repl_history,
            )

            messages: list[MessageDict] = self.protocol.render_initial_message_list(
                inputs,
                scope,
                repl,
                jaz_library=effective_jaz_library,
                depth=depth,
                recursion_available=can_recurse,
                repl_language=self.repl_language,
                input_display_overrides=None,
            )
            # A custom InteractionProtocol must return a non-empty opener. Guard it loudly
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
            # lives in IterationLimit (Abort at the always-present LLMQueryEnter, which
            # do_one_repl_iteration surfaces as a terminal Raise result — see #481). There is
            # no per-iteration span: the per-turn boundary is now LLMQuery, so the old
            # REPLIterationEnter/Exit events and their span were removed.
            i = 0
            while True:
                i += 1
                # Update prehook so nested invokes know which iteration spawned them
                if self.prehook is not None:
                    self.prehook.parent_repl_iteration = i

                repl_iter_result = self.do_one_repl_iteration(
                    i,
                    depth,
                    can_recurse,
                    messages,
                    repl,
                    invoke_id,
                    repl_history=repl_history,
                )

                exec_result = repl_iter_result.exec_result
                repl = repl_iter_result.repl

                match exec_result:
                    case Continue():
                        self._append_observation_messages(
                            messages,
                            exec_result,
                            repl,
                            i,
                        )
                    case Return() | Raise():
                        # Record the terminal result and leave the loop. The InvokeExit boundary
                        # (read below, after the span closes) may still TRANSFORM it — a
                        # ReturnType / ValidateReturn hook downgrades a wrong-typed / invalid final
                        # Return to a Raise there (the backstop for another hook's REPLExecExit
                        # override reinstating a Return). So we no longer return/raise from inside
                        # the span.
                        span.complete(result=exec_result)
                        break
                    case _:
                        raise _JazInternalError("Unhandled REPL result type")

        # Span closed and InvokeExit fired: read the FINAL (possibly transformed) terminal result
        # and return/raise from it.
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
    ) -> str | Raise | None:
        # Mirrors _query_with_messages exactly except for the awaited LLM call; the edit
        # fold is the shared _compose_shown_messages, so the two paths cannot drift.
        complete_fn, invoke_id, enter_event = self._make_llm_retry_fn(
            self.llm_client.acomplete_with_retry,
            messages,
            invoke_id,
            iteration,
            depth,
            can_recurse,
        )
        with self.dispatcher.span_llm_query(enter_event) as span:
            # Abort @ LLMQueryEnter (iteration / budget hard-stop): terminate the invoke.
            # Return the terminal Raise; the query never happens and no LLMQueryExit fires.
            if span.abort is not None:
                assert isinstance(span.abort, Raise)
                return span.abort
            shown = self._compose_shown_messages(span, messages)
            # Check for override response from hooks (e.g., Replay)
            if span.ctx.override_response is not None:
                llm_response: LLMResponse = span.ctx.override_response
                start_time = end_time = datetime.now()
            else:
                start_time = datetime.now()
                llm_response = await complete_fn(
                    model=self.model,
                    messages=shown,
                    **self.llm_client.request_defaults,
                )
                end_time = datetime.now()
            return self._record_llm_response(
                llm_response, span, iteration, depth, start_time, end_time
            )

    async def do_one_repl_iteration_async(
        self,
        iteration: int,
        depth: int,
        can_recurse: bool,
        messages: list[MessageDict],
        repl: REPL,
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
        try:
            response_content = await self._aquery_with_messages(
                messages,
                iteration=iteration,
                depth=depth,
                invoke_id=invoke_id,
                can_recurse=can_recurse,
            )
        except Exception as e:
            # #687: a failed LLM call becomes a terminal Raise rather than propagating
            # as an uncaught exception (see the sync path for rationale, including why an
            # AbortError is deliberately not special-cased here). TODO(#697): route selected
            # LLM errors to Continue / an AbortError instead of always Raise.
            return REPLIterationResult(
                exec_result=Raise(exception=e),
                repl=repl,
                response_content=None,
                code=None,
            )
        # Abort @ LLMQueryEnter surfaced as a terminal Raise (see the sync path): propagate
        # it as the iteration's result; the loop raises it.
        if isinstance(response_content, Raise):
            return REPLIterationResult(
                exec_result=response_content,
                repl=repl,
                response_content=None,
                code=None,
            )
        # Apply any persistent message edits (compaction) to the carried-forward buffer.
        self._apply_persistent_message_edits(messages, iteration)
        parsed = self._record_response_and_parse(
            messages, response_content, repl, invoke_id, iteration, depth, repl_history
        )
        if isinstance(parsed, REPLIterationResult):
            return parsed
        code, exec_timeout_override, enter_event = parsed

        with self.dispatcher.span_repl_exec(enter_event) as span:
            # Apply DropVariables then AddVariables (see the sync path for rationale): unbind the
            # composed names, then bind, before this turn's code runs — drop-first so a
            # drop-then-add pair re-binds; consistent across the override/exec branches.
            repl.drop_variables(
                span.ctx.dropped_variables,
                allow_missing=span.ctx.dropped_variables_allow_missing,
            )
            repl.add_variables(span.ctx.added_variables)
            if span.enter_override is not None:
                # An enter-time OverrideResult (supply) / Abort short-circuits execution:
                # skip running the code and use the composed result instead.
                exec_result = span.enter_override
            else:
                # Execute the code. (Invoke inputs are folded in once at InvokeEnter, not here:
                # AddInputs/DropInputs are InvokeEnter-only — #481. Per-turn namespace binds go
                # through AddVariables/DropVariables above.)
                exec_result = await repl.aexec(
                    code, str(iteration), exec_timeout_override
                )
            span.complete(exec_result=exec_result)

        return self._finalize_iteration(
            span, code, repl, response_content, repl_history
        )

    async def ainvoke(
        self,
        *,
        jaz_library: Library | None,
        jaz_library_no_invoke: Library | None,
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
            jaz_library=jaz_library,
        )
        with self.dispatcher.span_invoke(enter_event) as span:
            # A hook may abort the whole invoke before any execution (e.g. an
            # inherited cost-pool budget is already exhausted). enter_override is
            # a Raise; short-circuit the loop and raise it.
            if span.enter_override is not None:
                assert isinstance(span.enter_override, Raise)
                span.complete(result=span.enter_override)
                raise span.enter_override.exception

            # Apply invoke-level input effects (AddInputs / DropInputs) before the REPL and
            # prompt are built — the async mirror of the sync path (see there for the full
            # rationale). Adds render+bind, drops un-pass from both; drops run first, so an add
            # wins on overlap.
            inputs, scope, resolved_bound = self._apply_input_effects(
                span.ctx, inputs, scope, resolved_bound
            )

            # `jaz_library` carries only the framework JAZ library (a single
            # Library or None; see the sync path above and library_as_input.md).

            # Honor DisableRecursion (RecursionLimit at the cap leaf) — same as the sync path:
            # bind the cap-leaf library (helpers minus jaz.invoke, #635) and key can_recurse on
            # the effect rather than library-None-ness.
            effective_jaz_library = (
                jaz_library_no_invoke if span.ctx.recursion_disabled else jaz_library
            )
            can_recurse = not span.ctx.recursion_disabled

            session_id = uuid.uuid4().hex[:8]

            # Single-writer __repl_history__ ownership (see the sync invoke()).
            repl_history: list[object] | None = (
                # Empty list = the whole initial history (no index-0 init snapshot, #568). The core
                # loop is the single writer; the REPL surfaces this list as __repl_history__ by ref.
                []
                # __repl_history__ is always maintained now (#568 removed the repl_history
                # flag); only a REPL with maintains_repl_history=False gets None.
                if type(self.repl_template).maintains_repl_history
                else None
            )
            # The configured REPL comes off the config; `initialize` returns a fresh instance
            # carrying this run's state and leaves the template clean for the next invoke.
            repl = self.repl_template.initialize(
                inputs=resolved_bound,
                jaz_library=effective_jaz_library,
                session_id=session_id,
                initial_repl_history=repl_history,
            )

            # NOTE: the async path historically does NOT unwrap jaz.Display display
            # overrides (only sync invoke() calls unwrap_display), so it passes
            # input_display_overrides=None — preserved here verbatim. Unifying the
            # sync/async paths is tracked separately (see #566 roadmap).
            messages: list[MessageDict] = self.protocol.render_initial_message_list(
                inputs,
                scope,
                repl,
                jaz_library=effective_jaz_library,
                depth=depth,
                recursion_available=can_recurse,
                repl_language=self.repl_language,
                input_display_overrides=None,
            )
            # A custom InteractionProtocol must return a non-empty opener. Guard it loudly
            # here (the load-bearing half of the old iteration-0 footer assert, #705): an
            # empty list otherwise slips past _stamp_seed_provenance and surfaces later as a
            # muddier empty-message-list error at the first query.
            assert messages, "render_initial_message_list returned an empty opener"

            self._stamp_seed_provenance(messages)

            # (AddInputs/DropInputs already folded into inputs/resolved_bound above via
            # _apply_input_effects — no post-init REPL mutation needed; see sync invoke().)

            i = 0
            while True:
                i += 1
                # Update prehook so nested invokes know which iteration spawned them
                if self.prehook is not None:
                    self.prehook.parent_repl_iteration = i

                repl_iter_result = await self.do_one_repl_iteration_async(
                    i,
                    depth,
                    can_recurse,
                    messages,
                    repl,
                    invoke_id,
                    repl_history=repl_history,
                )

                exec_result = repl_iter_result.exec_result
                repl = repl_iter_result.repl

                match exec_result:
                    case Continue():
                        self._append_observation_messages(
                            messages,
                            exec_result,
                            repl,
                            i,
                        )
                    case Return() | Raise():
                        # Record the terminal result and leave the loop. The InvokeExit boundary
                        # (read below, after the span closes) may still TRANSFORM it — a
                        # ReturnType / ValidateReturn hook downgrades a wrong-typed / invalid final
                        # Return to a Raise there (the backstop for another hook's REPLExecExit
                        # override reinstating a Return). So we no longer return/raise from inside
                        # the span.
                        span.complete(result=exec_result)
                        break
                    case _:
                        raise _JazInternalError("Unhandled REPL result type")

        # Span closed and InvokeExit fired: read the FINAL (possibly transformed) terminal result
        # and return/raise from it.
        final_result = span.get_result()
        match final_result:
            case Return():
                return final_result.return_value
            case Raise():
                raise final_result.exception
            case _:
                raise _JazInternalError("Unhandled final invoke result type")
