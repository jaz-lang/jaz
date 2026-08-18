"""Generic OpenTelemetry tracing hook for JAZ hook events.

This hook translates JAZ execution events into OpenTelemetry spans and exports
them via OTLP HTTP. Backend-specific hooks (e.g., Jaeger, Langfuse) should
compose this hook with backend-specific defaults.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.string_utils import summarize_exception


class OTelTracing(Hook):
    """Export the run as OpenTelemetry spans over OTLP HTTP.

    Under ``with`` the trace covers invokes nested inside. Passed positionally, only that
    invoke is traced and its sub-invokes contribute no spans to this trace.

    Args:
        endpoint: OTLP HTTP traces endpoint (e.g., ``.../v1/traces``)
        service_name: OTel ``service.name`` resource attribute
        headers: Optional OTLP exporter headers (auth, tenant, etc.)
        max_attribute_length: Maximum length for string attributes. If None,
            strings are not truncated.
    """

    # Reads the per-invoke ``task_name`` off the blackboard (set by a ``MetaData``
    # hook) to name the root trace; declaring it lets that seed validate. Absent →
    # ``"main"``.
    blackboard_consumes = {"task_name": "Human label used as the root trace name."}

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

    def _end_span_abnormal(self, span: Span, event: Any) -> None:
        """Close ``span`` for a non-completed ``*Exit`` (an ``Aborted``/``Failed`` outcome).

        The #892 payoff: the span context managers now fire the exit events on the
        exceptional path too (bottom-up as the exception unwinds), so this hook closes its
        OTel spans at their natural point, in order, with ERROR status and the terminating
        exception — instead of leaking them to ``teardown()`` with inflated, out-of-order
        durations. ``teardown()`` is retained as the backstop for the machinery gaps the
        design doc documents (a failure during the exit-emit itself).
        """
        outcome = event.outcome
        span.set_status(Status(StatusCode.ERROR))
        # The variant IS the status; its class name (lowercased) is the serialized form.
        span.set_attribute("jaz.span_outcome", type(outcome).__name__.lower())
        span.set_attribute("error.type", type(outcome.exception).__name__)
        # summarize_exception, not str(): multiple aborts at one boundary group into an
        # ExceptionGroup by law, and str() on a group renders only a sub-exception count,
        # dropping every child's text — on exactly the failure arms this attribute exists
        # for. Matches the jaz.exception result attributes below.
        span.set_attribute(
            "error.message", self._truncate(summarize_exception(outcome.exception))
        )
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
                span = spans.pop((invoke_id, "repl_iteration"), None)
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
            case InvokeEnter(
                invoke_id=invoke_id,
                parent_invoke_id=parent_invoke_id,
                inputs=inputs,
                depth=cur_depth,
            ):
                # Per-invoke label carried on the blackboard (seeded by a MetaData
                # hook), not a core event field. Absent → "main".
                task_name = str(event.blackboard.get("task_name", "main"))
                attributes: dict[str, Any] = {
                    "jaz.span_type": "invoke",
                    "jaz.invoke_id": invoke_id,
                    # The former `jaz.prompt` attribute is gone (#538): `task` is now an
                    # ordinary input, so the prompt lives in `jaz.inputs` below rather than
                    # in a separate distinguished field.
                    "jaz.depth": cur_depth,
                    # Explicit `**inputs` kwargs and resolved ambient `jaz.scope` are logged as
                    # SEPARATE attributes (#727): they are distinct provenance channels. `jaz.scope`
                    # is omitted entirely when no scope is active (the common case) to avoid noise.
                    "jaz.inputs": str(dict(inputs)),
                    # The governance active for this invoke: each active hook serialized via
                    # rendered at this edge (`event.hooks` carries the LIVE set, #727), incl. this
                    # tracer. Emitted UNCONDITIONALLY (unlike `jaz.scope` above, omitted-when-empty):
                    # the active-hook set is the governance record — always worth pinning on the
                    # span, and never empty for an event this tracer observes (the tracer is in it).
                    "jaz.hooks": str(list(event.hooks)),
                }
                if event.scope:
                    attributes["jaz.scope"] = str(dict(event.scope))
                parent_span = spans.get(parent_invoke_id) if parent_invoke_id else None
                is_root_invoke = parent_span is None
                span = self.tracer.start_span(
                    "invoke",
                    attributes=attributes,
                    context=(set_span_in_context(parent_span) if parent_span else None),
                )
                # The span's input is the invoke's inputs (#538): there is no separate
                # prompt string anymore — the task rides in `inputs` like any other input.
                self._set_input_attributes(
                    span, dict(inputs), include_trace_fields=is_root_invoke
                )
                if is_root_invoke:
                    span.set_attribute("langfuse.trace.name", task_name)
                invoke_roots[invoke_id] = is_root_invoke
                spans[invoke_id] = span

            case InvokeSend(
                invoke_id=invoke_id,
                inputs=inputs,
                added_inputs=added_inputs,
                dropped_inputs=dropped_inputs,
            ):
                # Re-stamp the invoke span's inputs with the COMMITTED set (post
                # AddInputs/DropInputs) — the enter-time stamp above is the proposal,
                # which misses hook-injected inputs (the input-side #906 pattern, e.g.
                # ultrahorizon's per-child `env`; ATIFTrace made the same Enter→Send
                # move). The enter stamp is kept as the early value for spans whose
                # Send never fires (an enter-time abort). Hook-added/dropped names are
                # recorded only when hooks edited the set, mirroring ATIF's
                # hook_added_inputs/hook_dropped_inputs provenance split.
                span = spans.get(invoke_id)
                if span:
                    span.set_attribute("jaz.inputs", str(dict(inputs)))
                    if added_inputs:
                        span.set_attribute(
                            "jaz.hook_added_inputs", str(sorted(added_inputs))
                        )
                    if dropped_inputs:
                        span.set_attribute(
                            "jaz.hook_dropped_inputs", str(sorted(dropped_inputs))
                        )
                    self._set_input_attributes(
                        span,
                        dict(inputs),
                        include_trace_fields=invoke_roots.get(invoke_id, False),
                    )

            case InvokeExit(invoke_id=invoke_id, outcome=Completed(result=result)):
                span = spans.pop(invoke_id, None)
                if span:
                    from jaz.repl.types import (
                        Raise,
                        Return,
                    )

                    is_root_invoke = invoke_roots.pop(invoke_id, False)

                    match result:
                        case Raise():
                            span.set_attribute("jaz.result_type", "RAISE")
                            # A ``Raise`` carries no output (#903); its exception is the payload.
                            # ``jaz.exception`` is left as the *single* carrier rather than also
                            # feeding the output attributes: before #903 the two were
                            # complementary (``jaz.output`` held the invoke's captured stdout),
                            # but with no output field the output attributes could only repeat
                            # the error verbatim. A failed invoke reads as ERROR status plus
                            # ``jaz.exception``; an empty Langfuse output column is the accurate
                            # rendering of "this invoke produced no result".
                            span.set_attribute(
                                "jaz.exception",
                                self._truncate(summarize_exception(result.exception)),
                            )
                            span.set_status(Status(StatusCode.ERROR))
                        case Return(return_value=return_value):
                            span.set_attribute("jaz.result_type", "RETURN")
                            # Same `jaz.*` payload naming as the repl_iteration RETURN arm, so
                            # the two span levels are queryable the same way: the returned value
                            # under ``jaz.return_value``, plus the generic output slot below.
                            span.set_attribute(
                                "jaz.return_value", self._truncate(str(return_value))
                            )
                            self._set_output_attributes(
                                span,
                                return_value,
                                include_trace_fields=is_root_invoke,
                            )
                            span.set_status(Status(StatusCode.OK))
                        case _:
                            # An invoke only ever exits on a *terminal* result, so there is no
                            # Continue arm here (unlike the repl_iteration span, where Continue
                            # is the common case). The loop completes the invoke span solely
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

            case REPLExecEnter(
                invoke_id=invoke_id,
                iteration=iteration,
                code=code,
                depth=depth,
            ):
                attributes = {
                    "jaz.span_type": "repl_iteration",
                    "jaz.invoke_id": invoke_id,
                    "jaz.iteration": iteration,
                    "jaz.code": self._truncate(code),
                    "jaz.depth": depth,
                }
                parent_span = spans.get(invoke_id)
                span = self.tracer.start_span(
                    f"repl_iteration_{iteration}",
                    attributes=attributes,
                    context=(set_span_in_context(parent_span) if parent_span else None),
                )
                self._set_input_attributes(span, code, include_trace_fields=False)
                spans[(invoke_id, "repl_iteration")] = span

            case REPLExecExit(invoke_id=invoke_id, outcome=Completed(result=result)):
                span = spans.pop((invoke_id, "repl_iteration"), None)
                if span:
                    from jaz.repl.types import (
                        Continue,
                        Raise,
                        Return,
                    )

                    match result:
                        case Raise(exception=exception):
                            span.set_attribute("jaz.exec_result_type", "RAISE")
                            # Mirrors the invoke-level RAISE arm exactly: a ``Raise`` carries no
                            # output (#903), so the exception goes to ``jaz.exception`` and the
                            # output attributes stay unset. Stuffing ``RAISE: <exception>`` into
                            # ``jaz.output`` would file an error under a name that means
                            # "what this turn produced", which is what it no longer is.
                            span.set_attribute(
                                "jaz.exception",
                                self._truncate(summarize_exception(exception)),
                            )
                            span.set_status(Status(StatusCode.ERROR))
                        case Return(return_value=return_value):
                            span.set_attribute("jaz.exec_result_type", "RETURN")
                            # No output on a ``Return`` (#903) — the return value *is* the
                            # payload, so record it under its own name rather than under
                            # ``jaz.output``: a return value is not the turn's printed output,
                            # and after #903 the two can no longer coincide. This leaves each
                            # `jaz.*` payload attribute meaning exactly one thing —
                            # ``jaz.exception`` on RAISE, ``jaz.return_value`` on RETURN,
                            # ``jaz.output`` only on EXECUTE/ERROR, where it really is stdout.
                            # A constant "RETURN" label here would drop the finishing turn's
                            # only real content from the trace.
                            span.set_attribute(
                                "jaz.return_value", self._truncate(str(return_value))
                            )
                            # The generic OTel/Langfuse output slot still takes the return
                            # value: for a successful finish that *is* the observation's
                            # output, and the invoke-level RETURN arm fills it the same way.
                            self._set_output_attributes(
                                span, return_value, include_trace_fields=False
                            )
                        case Continue(output=output, exception=exception):
                            # One arm for both continue outcomes (#566 C): a clean
                            # success and a recoverable error differ only in the label
                            # and whether ``jaz.exception`` is set.
                            is_error = exception is not None
                            span.set_attribute(
                                "jaz.exec_result_type",
                                "ERROR" if is_error else "EXECUTE",
                            )
                            output_text = str(output).strip()
                            # Each `jaz.*` payload attribute names exactly one thing, and neither
                            # stands in for the other: ``jaz.output`` is this turn's actual stdout,
                            # ``jaz.exception`` the recoverable exception — the same name and
                            # rendering the RAISE arms use, so "the exception for this turn" is one
                            # query regardless of whether it was recoverable. An ERROR turn that
                            # printed nothing therefore leaves ``jaz.output`` unset rather than
                            # filing the exception text under it (which is what the old
                            # ``ERROR: <exc>`` fallback did, and is the same duplication that was
                            # removed from the invoke-level RAISE arm).
                            #
                            # Note the overlap this does NOT try to remove: a did-not-execute
                            # Continue (e.g. a REPL timeout-pragma error) deliberately carries the
                            # rendered error as its ``output``, so both attributes legitimately hold
                            # that text — ``output`` is what the agent saw, ``exception`` is the
                            # structured metadata beside it.
                            if output_text:
                                span.set_attribute(
                                    "jaz.output", self._truncate(output_text)
                                )
                            if exception is not None:
                                span.set_attribute(
                                    "jaz.exception",
                                    self._truncate(summarize_exception(exception)),
                                )
                            # The generic OTel/Langfuse output slot keeps a human-readable
                            # fallback: unlike the `jaz.*` attributes it is the viewer's single
                            # "what happened this turn" column, where a placeholder reads better
                            # than an empty cell.
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

            case LLMQueryEnter(
                invoke_id=invoke_id,
                model=model,
                iteration=iteration,
                depth=depth,
            ):
                # The span opens here (Enter is when the interval starts), but the
                # message-derived generation labels are stamped at LLMQuerySend below: the
                # committed post-edit list is what the model is actually shown, so labelling
                # from Enter's pre-edit proposal would record messages a compaction dropped
                # or miss ones a hook added (#906's observation rule for inputs).
                attributes: dict[str, Any] = {
                    "jaz.span_type": "llm_query",
                    "jaz.invoke_id": invoke_id,
                    "jaz.model": model,
                    "langfuse.observation.type": "generation",
                    "langfuse.generation.model": model,
                    "jaz.iteration": iteration,
                    "jaz.depth": depth,
                }
                parent_span = spans.get((invoke_id, "repl_iteration")) or spans.get(
                    invoke_id
                )
                span = self.tracer.start_span(
                    "llm_query",
                    attributes=attributes,
                    context=(set_span_in_context(parent_span) if parent_span else None),
                )
                spans[(invoke_id, "llm_query")] = span

            case LLMQuerySend(invoke_id=invoke_id, messages=messages):
                # Label the open generation span from the COMMITTED message list — exactly
                # what is handed to the result producer (and what a Send-time supplier saw).
                span = spans.get((invoke_id, "llm_query"))
                if span and messages:
                    span.set_attribute("jaz.message_count", len(messages))
                    # Positional tail labels are exact here: this list is final for the
                    # query, so no length/role hedging against in-flight edits is needed —
                    # only the role check that decides whether the second-to-last message
                    # is a user turn worth labelling.
                    if len(messages) >= 2 and messages[-2]["role"] == "user":
                        span.set_attribute(
                            "jaz.last_last_message",
                            self._truncate(self._format_message(messages[-2])),
                        )
                    span.set_attribute(
                        "jaz.last_message",
                        self._truncate(self._format_message(messages[-1])),
                    )
                    self._set_input_attributes(
                        span,
                        self._format_message(messages[-1]),
                        include_trace_fields=False,
                    )

            case LLMQueryRetry(
                invoke_id=invoke_id,
                attempt_number=attempt_number,
                exception=exception,
                wait_seconds=wait_seconds,
            ):
                # Recorded as a span event (OTel's native shape for a point-in-time
                # occurrence inside an interval) on the open generation span, so a call
                # that failed and retried leaves its failed attempts, errors, and backoff
                # in the trace — previously only the loggers observed this event and the
                # trace showed one clean call however many attempts it took.
                span = spans.get((invoke_id, "llm_query"))
                if span:
                    span.add_event(
                        "llm_retry",
                        {
                            "attempt_number": attempt_number,
                            "error.type": type(exception).__name__,
                            "error.message": self._truncate(
                                summarize_exception(exception)
                            ),
                            "wait_seconds": wait_seconds,
                        },
                    )

            # Matching the Completed variant both narrows and unwraps the payload — the
            # non-completed arms matched above.
            case LLMQueryExit(invoke_id=invoke_id, outcome=Completed(result=response)):
                content = response.content
                span = spans.pop((invoke_id, "llm_query"), None)
                if span:
                    if content:
                        span.set_attribute("jaz.response", self._truncate(str(content)))
                        self._set_output_attributes(
                            span, content, include_trace_fields=False
                        )
                    if response.prompt_tokens is not None:
                        span.set_attribute("jaz.prompt_tokens", response.prompt_tokens)
                    if response.completion_tokens is not None:
                        span.set_attribute(
                            "jaz.completion_tokens", response.completion_tokens
                        )
                    if response.cost_usd is not None:
                        span.set_attribute("jaz.cost_usd", response.cost_usd)
                    span.set_status(Status(StatusCode.OK))
                    span.end()

        return []

    def teardown(self, exc: BaseException | None = None) -> None:
        # End any spans still open, mark them errored if the scope raised, then
        # force-flush the exporter so batched spans are actually sent.
        exc_type = type(exc) if exc is not None else None
        spans = self._get_active_spans()
        invoke_roots = self._get_invoke_roots()
        for span_key in list(spans.keys()):
            span = spans.pop(span_key)
            if exc_type is not None and exc is not None:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", exc_type.__name__)
                # Same group-expanding rendering as _end_span_abnormal — one exception
                # format across every failure surface of this hook.
                summary = summarize_exception(exc)
                span.set_attribute("error.message", self._truncate(summary))
                if isinstance(span_key, str):
                    self._set_output_attributes(
                        span,
                        summary,
                        include_trace_fields=invoke_roots.get(span_key, False),
                    )
            span.end()
            if isinstance(span_key, str):
                invoke_roots.pop(span_key, None)

        self.provider.force_flush()


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_legacy_hook_names_still_importable``). This hook is absent from every ``__all__``,
#: so that deep-path import is the only way it was ever reachable.
OTelTracingHook = OTelTracing
