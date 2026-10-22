"""Generic OpenTelemetry tracing hook for JAZ hook events.

This hook translates JAZ execution events into OpenTelemetry spans and exports
them via OTLP HTTP. Backend-specific hooks (e.g., Jaeger, Langfuse) should
compose this hook with backend-specific defaults.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from typing import Any, cast

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context
from opentelemetry.util.types import AttributeValue

from jaz._string_utils import summarize_exception
from jaz.hooks._base import Event
from jaz.hooks._dispatcher import Hook

# Event and payload fields that do not become span attributes. Everything else does, under
# its own name, so this set is the whole list of exceptions:
#   - `config`, `blackboard`: objects the span cannot hold (the blackboard's `label` is read
#     separately, as the trace name);
#   - `timestamp`: the span's own start time;
#   - `raw_response`: the provider's native response object, which `LLMResponse` itself keeps
#     out of its repr;
#   - `tokens`: per-token records, bulky and provider-specific, kept for ATIF traces;
#   - `namespace`: the REPL's live namespace, which both `REPLExecEnter` and `REPLExecSend`
#     carry in full. It is unbounded — every name the agent has bound, `__history__` and the
#     imported modules included — and it would be stamped twice per turn, so a long run's
#     trace would be mostly namespace dumps. What the turn *did* to it is kept:
#     `REPLExecSend.namespace_edits` records WHICH bindings hooks added and dropped, by name;
#     `NamespaceAddRecord.value` / `NamespaceDropRecord.value` are excluded with it, or the
#     objects would come back through the edit records for every name a hook touched.
#
# A member spelled `Type.field` excludes that field of that record only; a bare name excludes
# it wherever it appears.
_NOT_ATTRIBUTES = frozenset(
    {
        "config",
        "blackboard",
        "timestamp",
        "raw_response",
        "tokens",
        "namespace",
        "NamespaceAddRecord.value",
        "NamespaceDropRecord.value",
    }
)

#: Ambient invoke context: resolved once per invoke and identical on every event of it, so it
#: is stamped on the `invoke` span and left off that invoke's `repl_exec` / `llm_query` spans,
#: which carry `jaz.invoke_id` to correlate. `scope` is the reason — a scoped API client or
#: dataframe reprs to kilobytes, and copying it onto every span multiplies the trace with no
#: information — and `hooks` rides along so the rule is one rule rather than a size cutoff.
_AMBIENT_ONCE = frozenset({"scope", "hooks"})


def _is_record(value: object) -> bool:
    """One of the framework's own records — a dataclass or NamedTuple defined under ``jaz`` —
    whose fields are flattened by path. A user value of the same shapes (a dataclass returned
    by the agent, say) is not: it is recorded as text like any other value, so its fields are
    neither exploded nor filtered by the exclusion set."""
    if not type(value).__module__.startswith("jaz."):
        return False
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return True
    return isinstance(value, tuple) and hasattr(type(value), "_fields")


def _render_record(record: object) -> str:
    """One framework record as ``Type(field=value, ...)``, minus its excluded fields."""
    shown = [
        f"{name}={getattr(record, name)!r}"
        for name in _field_names(record)
        if name not in _NOT_ATTRIBUTES
        and f"{type(record).__name__}.{name}" not in _NOT_ATTRIBUTES
    ]
    return f"{type(record).__name__}({', '.join(shown)})"


def _field_names(value: object) -> tuple[str, ...]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple(f.name for f in dataclasses.fields(value))
    # A NamedTuple: `_is_record` checked `_fields` exists, which the static type cannot see.
    return tuple(cast(Any, value)._fields)


class OTelTracing(Hook):
    """Export the run as OpenTelemetry spans over OTLP HTTP.

    Under ``with`` the trace covers invokes nested inside. Passed positionally, only that
    invoke is traced and its sub-invokes contribute no spans to this trace.

    One span per span event family, named after it: ``invoke`` (``InvokeEnter`` to
    ``InvokeExit``), ``repl_exec`` (``REPLExecEnter`` to ``REPLExecExit``) and ``llm_query``
    (``LLMQueryEnter`` to ``LLMQueryExit``); a turn's ``repl_exec`` and the ``llm_query`` that
    chose its code are both children of the ``invoke`` span, siblings in call order rather than
    nested, because the query closes before the turn's execution opens. An ``LLMQueryRetry`` is a
    span event named ``llm_query_retry`` on the open ``llm_query`` span. Every field of the
    events that open, commit and close a span is an attribute under ``jaz.<field>``, with a
    nested payload flattened by path: ``jaz.outcome`` holds the outcome variant's name
    (``completed`` / ``aborted`` / ``failed``), ``jaz.outcome.result`` the result's type
    (``return`` / ``raise`` / ``continue`` / ``llmresponse``), and
    ``jaz.outcome.result.<field>`` its fields (every record node carries its type name, so a
    plain record such as ``LLMQuerySend.message_edits`` appears as ``jaz.message_edits`` too).
    Only the framework's own records are flattened; a value of the agent's, a returned
    dataclass say, is recorded as text. ``None`` and empty collections are omitted; strings are
    truncated to ``max_attribute_length``. ``jaz.messages`` is the whole committed message list
    of every query, each message carrying its provenance stamp, so a long run should set
    ``max_attribute_length``. The REPL namespace is the one event field left out as a matter of
    size rather than type: which names the turn bound and unbound ride ``jaz.namespace_edits``,
    without their values. ``jaz.scope`` and ``jaz.hooks`` are resolved once per invoke and stamped
    on the ``invoke`` span alone, since copying them onto its every child span would multiply the
    trace without adding to it; ``jaz.invoke_id`` and ``jaz.depth`` are on every span, to correlate
    them. A ``*Send`` overwrites what its ``*Enter`` stamped, writing an empty collection rather
    than omitting it, so a committed empty value replaces the proposal instead of leaving it
    standing. Beside those, the viewer-facing
    conventions: ``input.value`` / ``output.value`` and their ``langfuse.*`` twins,
    ``gen_ai.request.model``, and ``error.type`` / ``error.message`` with ERROR status on a
    raise, an abort or a failure.

    Args:
        endpoint: OTLP HTTP traces endpoint (e.g., ``.../v1/traces``)
        service_name: OTel ``service.name`` resource attribute
        headers: Optional OTLP exporter headers (auth, tenant, etc.)
        max_attribute_length: Maximum length for string attributes. If None,
            strings are not truncated.

    Reads one blackboard key: ``label``, a human name for the run. Seed it with a
    ``MetaData(label=...)`` hook; absent, it is ``"main"``. Spans are named after their events
    (``invoke``, ``llm_query``, ``repl_exec``), so the label does not name one — it rides the
    root span as the ``langfuse.trace.name`` attribute, which is what Langfuse titles a trace by.
    """

    # EXECUTIVE CALL (user, 2026-09-19): the attribute vocabulary is DERIVED from the events,
    # not authored beside them. Before this the hook named spans and attributes by hand
    # (`repl_iteration_<n>` for a `REPLExec*` span; `jaz.exec_result_type` = "EXECUTE" for a
    # `Continue`; `jaz._inputs`, `jaz.hook_added_inputs`, `jaz.last_last_message`), so every
    # name could drift from the event it came from and the handlers were mostly relabelling.
    # Now the span is the event family and the attribute is the field path, so a reader who
    # knows `InvokeExit.outcome.result.return_value` knows the attribute, a field added to an
    # event reaches the trace with no change here, and what is left in the handlers is
    # serialization — the values a span cannot hold as they are — plus the viewer conventions.

    def __init__(
        self,
        *,
        endpoint: str,
        service_name: str = "jaz-agent",
        headers: dict[str, str] | None = None,
        max_attribute_length: int | None = None,
    ):
        self._max_attribute_length = max_attribute_length
        self._active_spans: ContextVar[dict[str | tuple[str, str], Span] | None] = (
            ContextVar(f"otel_active_spans_{id(self)}", default=None)
        )
        self._invoke_roots: ContextVar[dict[str, bool] | None] = ContextVar(
            f"otel_invoke_roots_{id(self)}", default=None
        )

        resource = Resource.create({"service.name": service_name})
        self.provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or {})
        self.provider.add_span_processor(BatchSpanProcessor(exporter))
        self.tracer = self.provider.get_tracer(__name__)

    def _get_active_spans(self) -> dict[str | tuple[str, str], Span]:
        spans = self._active_spans.get()
        if spans is None:
            spans = {}
            self._active_spans.set(spans)
        return spans

    def _get_invoke_roots(self) -> dict[str, bool]:
        roots = self._invoke_roots.get()
        if roots is None:
            roots = {}
            self._invoke_roots.set(roots)
        return roots

    def _truncate(self, text: str) -> str:
        if (
            self._max_attribute_length is None
            or len(text) <= self._max_attribute_length
        ):
            return text
        return text[: self._max_attribute_length] + "..."

    # ---- serialization: an event's fields, as span attributes -------------------------------

    def _serialize(self, value: object) -> AttributeValue | None:
        """One value as a span attribute, or ``None`` for one that should be omitted.

        Scalars pass through; everything else becomes text (truncated), because a span holds
        only scalars and lists of them. An exception renders through ``summarize_exception``
        so an ``ExceptionGroup`` shows its children rather than a count. Empty collections
        are omitted like ``None``: an attribute saying ``{}`` says nothing.
        """
        if value is None:
            return None
        if isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return self._truncate(value)
        if isinstance(value, BaseException):
            return self._truncate(summarize_exception(value))
        if isinstance(value, bytes | bytearray):
            # Text, not a Sequence of ints — which is what the Sequence arm below would make it.
            return self._truncate(repr(value))
        if isinstance(value, Mapping):
            return self._truncate(str(dict(value))) if value else None
        if isinstance(value, set | frozenset):
            return self._truncate(str(sorted(value, key=str))) if value else None
        if isinstance(value, Sequence):
            if not value:
                return None
            if all(_is_record(item) for item in value):
                # A sequence of the framework's own records is rendered record by record so
                # the exclusion set reaches inside it. Without this the whole sequence is one
                # repr, and a field excluded on the record — `NamespaceAddRecord.value`, the
                # object a hook bound — comes back in it, which is how excluding `namespace`
                # alone would fail to keep the namespace out.
                return self._truncate(", ".join(_render_record(item) for item in value))
            return self._truncate(str(list(value)))
        return self._truncate(repr(value))

    def _flatten(
        self, prefix: str, record: object, attributes: dict[str, AttributeValue]
    ) -> None:
        """Write ``record``'s fields under ``prefix.<field>``, recursing into nested records.

        The record's own type name goes under ``prefix`` itself, lowercased: for a union
        payload (``Return`` / ``Raise`` / ``Continue``, ``Completed`` / ``Aborted`` /
        ``Failed``) the variant IS the fact a reader wants, and this is where the path to
        the fields below it is spelled.
        """
        attributes[prefix] = type(record).__name__.lower()
        for name in _field_names(record):
            if name in _NOT_ATTRIBUTES:
                continue
            if f"{type(record).__name__}.{name}" in _NOT_ATTRIBUTES:
                continue
            value = getattr(record, name)
            key = f"{prefix}.{name}"
            if _is_record(value):
                self._flatten(key, value, attributes)
            else:
                serialized = self._serialize(value)
                if serialized is not None:
                    attributes[key] = serialized

    def _attributes(
        self,
        event: Event,
        *,
        exclude: frozenset[str] = frozenset(),
        restamp: bool = False,
    ) -> dict[str, AttributeValue]:
        """Every field of ``event`` as ``jaz.<field>``, nested payloads flattened by path.

        ``restamp`` writes an empty collection as its own text instead of omitting it, for a
        ``*Send`` that overwrites what its ``*Enter`` already stamped.
        """
        # Omitting an empty collection is right when the attribute is being set for the first
        # time and wrong when it overwrites: a span whose Enter stamped
        # ``jaz.inputs = "{'task': ...}"`` and whose Send committed ``{}`` would otherwise keep
        # naming an input the invoke never received. An empty ``None`` stays omitted either
        # way — no field is a value at Enter and ``None`` at Send.
        attributes: dict[str, AttributeValue] = {}
        for f in dataclasses.fields(event):
            if f.name in _NOT_ATTRIBUTES or f.name in exclude:
                continue
            value = getattr(event, f.name)
            key = f"jaz.{f.name}"
            if _is_record(value):
                self._flatten(key, value, attributes)
            else:
                serialized = self._serialize(value)
                if serialized is None and restamp and value is not None:
                    serialized = self._truncate(str(value))
                if serialized is not None:
                    attributes[key] = serialized
        return attributes

    # ---- the viewer-facing conventions ------------------------------------------------------

    def _format_message(self, message: object) -> str:
        if isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                return str(content)
        return str(message)

    def _set_input_attributes(
        self, span: Span, value: object, *, include_trace_fields: bool
    ) -> None:
        text = self._truncate(str(value))
        span.set_attribute("input.value", text)
        span.set_attribute("langfuse.observation.input", text)
        if include_trace_fields:
            span.set_attribute("langfuse.trace.input", text)

    def _set_output_attributes(
        self, span: Span, value: object, *, include_trace_fields: bool
    ) -> None:
        text = self._truncate(str(value))
        span.set_attribute("output.value", text)
        span.set_attribute("langfuse.observation.output", text)
        if include_trace_fields:
            span.set_attribute("langfuse.trace.output", text)

    def _set_error(self, span: Span, exception: BaseException) -> None:
        # OTel's own error conventions beside the `jaz.*` path that carries the same
        # exception; `summarize_exception` for the reason `_serialize` gives.
        span.set_status(Status(StatusCode.ERROR))
        span.set_attribute("error.type", type(exception).__name__)
        span.set_attribute(
            "error.message", self._truncate(summarize_exception(exception))
        )

    def _end_span_abnormal(self, span: Span, event: Any) -> None:
        """Close ``span`` for a non-completed ``*Exit`` (an ``Aborted``/``Failed`` outcome).

        The payoff: the span context managers fire the exit events on the
        exceptional path too (bottom-up as the exception unwinds), so this hook closes its
        OTel spans at their natural point, in order, with ERROR status and the terminating
        exception — instead of leaking them to ``teardown()`` with inflated, out-of-order
        durations. ``teardown()`` is retained as the backstop for the machinery gaps the
        design doc documents (a failure during the exit-emit itself).
        """
        # The abnormal arm exists because #892 made the span context managers fire exit events on
        # every path, including the one where an exception is already unwinding. Ambient context
        # is left off here as everywhere below the invoke span: this closes a span that was
        # opened, so whatever ambient it carries it already has.
        span.set_attributes(self._attributes(event, exclude=_AMBIENT_ONCE))
        self._set_error(span, event.outcome.exception)
        span.end()

    def on_any(self, event: Event) -> list:
        from jaz.hooks.events import (
            Completed,
            InvokeEnter,
            InvokeExit,
            InvokeSend,
            LLMQueryEnter,
            LLMQueryExit,
            LLMQueryRetry,
            LLMQuerySend,
            REPLExecEnter,
            REPLExecExit,
            REPLExecSend,
        )

        spans = self._get_active_spans()
        invoke_roots = self._get_invoke_roots()

        match event:
            # Non-completed arms first (#892 outcome union): close the interval with
            # ERROR + the exception rather than matching the Completed payload arms below.
            # See _end_span_abnormal.
            case LLMQueryExit(invoke_id=invoke_id) if not isinstance(
                event.outcome, Completed
            ):
                span = spans.pop((invoke_id, "llm_query"), None)
                if span:
                    self._end_span_abnormal(span, event)

            case REPLExecExit(invoke_id=invoke_id) if not isinstance(
                event.outcome, Completed
            ):
                span = spans.pop((invoke_id, "repl_exec"), None)
                if span:
                    self._end_span_abnormal(span, event)

            case InvokeExit(invoke_id=invoke_id) if not isinstance(
                event.outcome, Completed
            ):
                # No orphan sweep on this arm (nor anywhere): every opened child span
                # closes at its OWN exit event now — the abort carve-out that skipped
                # LLMQueryExit is gone, and abnormal unwinds fire the child exits
                # bottom-up before this one. teardown() remains the backstop for the
                # machinery gaps the design doc documents.
                span = spans.pop(invoke_id, None)
                invoke_roots.pop(invoke_id, None)
                if span:
                    self._end_span_abnormal(span, event)

            case InvokeEnter(invoke_id=invoke_id, parent_invoke_id=parent_invoke_id):
                # Per-invoke label carried on the blackboard (seeded by a MetaData
                # hook), not a core event field. Absent → "main".
                label = str(event.blackboard.get("label", "main"))
                parent_span = spans.get(parent_invoke_id) if parent_invoke_id else None
                is_root_invoke = parent_span is None
                span = self.tracer.start_span(
                    "invoke",
                    attributes=self._attributes(event),
                    context=(set_span_in_context(parent_span) if parent_span else None),
                )
                # The span's input is the invoke's inputs: there is no separate prompt
                # string — the task rides in `inputs` like any other input.
                self._set_input_attributes(
                    span, dict(event.inputs), include_trace_fields=is_root_invoke
                )
                if is_root_invoke:
                    span.set_attribute("langfuse.trace.name", label)
                invoke_roots[invoke_id] = is_root_invoke
                spans[invoke_id] = span

            case InvokeSend(invoke_id=invoke_id):
                # Re-stamp the invoke span with the COMMITTED input set (post
                # AddInputs/DropInputs) — the enter-time stamp above is the proposal,
                # which misses hook-injected inputs (the input-side #906 pattern, e.g.
                # ultrahorizon's per-child `env`; TrajectoryRecorder made the same Enter→Send
                # move). The enter stamp is kept as the early value for spans whose
                # Send never fires (an enter-time abort). `input_edits` rides along as the
                # event carries it.
                span = spans.get(invoke_id)
                if span:
                    span.set_attributes(self._attributes(event, restamp=True))
                    self._set_input_attributes(
                        span,
                        dict(event.inputs),
                        include_trace_fields=invoke_roots.get(invoke_id, False),
                    )

            case InvokeExit(invoke_id=invoke_id, outcome=Completed(result=result)):
                span = spans.pop(invoke_id, None)
                if span:
                    from jaz.repl._types import Raise, Return

                    is_root_invoke = invoke_roots.pop(invoke_id, False)
                    span.set_attributes(self._attributes(event))
                    match result:
                        case Raise(exception=exception):
                            # The exception is the payload and `jaz.outcome.result.exception`
                            # its single carrier: with no output on a Raise, the output slots
                            # could only repeat it, and an empty Langfuse output column is the
                            # accurate rendering of "this invoke produced no result".
                            self._set_error(span, exception)
                        case Return(return_value=return_value):
                            self._set_output_attributes(
                                span, return_value, include_trace_fields=is_root_invoke
                            )
                            span.set_status(Status(StatusCode.OK))
                        case _:
                            # An invoke only ever exits on a *terminal* result, so there is no
                            # Continue arm here (unlike the repl_exec span, where Continue is
                            # the common case). The loop completes the invoke span solely
                            # from its ``case Return() | Raise()`` arm — a Continue appends an
                            # observation and iterates — and the enter-time abort path is
                            # asserted to be a Raise in ``Agent.invoke``. The ATIF trace hook
                            # encodes the same invariant by matching only Return/Raise.
                            #
                            # Raised rather than `assert`-ed so `python -O` cannot silently turn
                            # a leaked non-terminal result into an untraced invoke.
                            raise AssertionError(
                                "InvokeExit carried a non-terminal result: "
                                f"{type(result).__name__}"
                            )
                    # No orphan sweep here anymore: the enter/send-abort carve-out died
                    # with the abort-propagation model (span_event_lifecycle.md, final
                    # stage) — an aborted query now fires its own LLMQueryExit(Aborted),
                    # closed by the abnormal arm above, so a completing invoke can have
                    # no dangling per-turn children.
                    span.end()

            case REPLExecEnter(invoke_id=invoke_id):
                parent_span = spans.get(invoke_id)
                span = self.tracer.start_span(
                    "repl_exec",
                    attributes=self._attributes(event, exclude=_AMBIENT_ONCE),
                    context=(set_span_in_context(parent_span) if parent_span else None),
                )
                self._set_input_attributes(span, event.code, include_trace_fields=False)
                spans[(invoke_id, "repl_exec")] = span

            case REPLExecSend(invoke_id=invoke_id, code=code):
                # Re-stamp the turn's span with the COMMITTED code — the enter-time stamp is
                # the proposal, and `InsertCode` / `DeleteCode` may have rewritten it. This is
                # also the stage carrying `code_edits` / `namespace_edits`: which hook changed
                # the code, and what it bound before the code ran.
                span = spans.get((invoke_id, "repl_exec"))
                if span:
                    span.set_attributes(
                        self._attributes(event, exclude=_AMBIENT_ONCE, restamp=True)
                    )
                    self._set_input_attributes(span, code, include_trace_fields=False)

            case REPLExecExit(invoke_id=invoke_id, outcome=Completed(result=result)):
                span = spans.pop((invoke_id, "repl_exec"), None)
                if span:
                    from jaz.repl._types import Continue, Raise, Return

                    span.set_attributes(self._attributes(event, exclude=_AMBIENT_ONCE))
                    match result:
                        case Raise(exception=exception):
                            # Mirrors the invoke-level arm: the exception is the payload, the
                            # output slots stay unset.
                            self._set_error(span, exception)
                        case Return(return_value=return_value):
                            # The generic OTel/Langfuse output slot takes the return value: for
                            # a successful finish that *is* the observation's output, and the
                            # invoke-level arm fills it the same way.
                            self._set_output_attributes(
                                span, return_value, include_trace_fields=False
                            )
                        case Continue(output=output, exception=exception):
                            # `jaz.outcome.result.output` and `.exception` are the event's own
                            # fields; the generic output slot is the viewer's single "what
                            # happened this turn" column, where a readable placeholder beats an
                            # empty cell. A recoverable error is not ERROR status: the turn ran
                            # and the agent continues.
                            output_text = str(output).strip()
                            if output_text:
                                output_value = output_text
                            elif exception is not None:
                                output_value = (
                                    f"ERROR: {summarize_exception(exception)}"
                                )
                            else:
                                output_value = "[no repl output]"
                            self._set_output_attributes(
                                span, output_value, include_trace_fields=False
                            )
                    span.end()

            case LLMQueryEnter(invoke_id=invoke_id):
                model = event.config.llm.model_info.name
                # Enter's `messages` are the pre-edit proposal; LLMQuerySend overwrites them
                # below with the committed post-edit list, which is what the model is shown
                # (#906's observation rule for inputs). Stamped here all the same, as the
                # invoke span stamps its enter-time inputs: a query aborted at Enter has no
                # Send, and the proposal is the only record of what it was about to ask.
                attributes: dict[str, AttributeValue] = self._attributes(
                    event, exclude=_AMBIENT_ONCE
                )
                attributes["langfuse.observation.type"] = "generation"
                # The GenAI semantic conventions make the model "Conditionally Required: if
                # available", and Langfuse reads the same key (plus its own); a backend that
                # names no model gets no label rather than a fake one.
                if model is not None:
                    attributes["gen_ai.request.model"] = model
                    attributes["langfuse.observation.model.name"] = model
                # The invoke span, not this turn's `repl_exec`: the query runs before the
                # turn's exec span opens (`Agent`'s loop closes `span_llm_query` and then
                # enters `span_repl_exec`), so a same-invoke exec span is never open here.
                parent_span = spans.get(invoke_id)
                span = self.tracer.start_span(
                    "llm_query",
                    attributes=attributes,
                    context=(set_span_in_context(parent_span) if parent_span else None),
                )
                spans[(invoke_id, "llm_query")] = span

            case LLMQuerySend(invoke_id=invoke_id, messages=messages):
                # Stamp the open generation span from the COMMITTED message list — exactly
                # what a Send-time supplier saw; the result producer gets its wire projection
                # (``Agent._to_wire``).
                span = spans.get((invoke_id, "llm_query"))
                if span:
                    span.set_attributes(
                        self._attributes(event, exclude=_AMBIENT_ONCE, restamp=True)
                    )
                    if messages:
                        self._set_input_attributes(
                            span,
                            self._format_message(messages[-1]),
                            include_trace_fields=False,
                        )

            case LLMQueryRetry(invoke_id=invoke_id, exception=exception):
                # Recorded as a span event (OTel's native shape for a point-in-time
                # occurrence inside an interval) on the open generation span, so a call
                # that failed and retried leaves its failed attempts, errors, and backoff
                # in the trace.
                span = spans.get((invoke_id, "llm_query"))
                if span:
                    attributes = self._attributes(event, exclude=_AMBIENT_ONCE)
                    attributes["error.type"] = type(exception).__name__
                    attributes["error.message"] = self._truncate(
                        summarize_exception(exception)
                    )
                    span.add_event("llm_query_retry", attributes)

            # Matching the Completed variant both narrows and unwraps the payload — the
            # non-completed arms matched above.
            case LLMQueryExit(invoke_id=invoke_id, outcome=Completed(result=response)):
                span = spans.pop((invoke_id, "llm_query"), None)
                if span:
                    span.set_attributes(self._attributes(event, exclude=_AMBIENT_ONCE))
                    if response.content:
                        self._set_output_attributes(
                            span, response.content, include_trace_fields=False
                        )
                    span.set_status(Status(StatusCode.OK))
                    span.end()

        return []

    def teardown(self, exc: BaseException | None = None) -> None:
        # End any spans still open, mark them errored if the scope raised, then
        # force-flush the exporter so batched spans are actually sent.
        spans = self._get_active_spans()
        invoke_roots = self._get_invoke_roots()
        for span_key in list(spans.keys()):
            span = spans.pop(span_key)
            if exc is not None:
                # Same rendering as `_set_error` — one exception format across every failure
                # surface of this hook.
                self._set_error(span, exc)
                if isinstance(span_key, str):
                    self._set_output_attributes(
                        span,
                        summarize_exception(exc),
                        include_trace_fields=invoke_roots.get(span_key, False),
                    )
            span.end()
            if isinstance(span_key, str):
                invoke_roots.pop(span_key, None)

        self.provider.force_flush()
