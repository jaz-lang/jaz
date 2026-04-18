"""Generic OpenTelemetry tracing hook for Jaz hook events.

This hook translates Jaz execution events into OpenTelemetry spans and exports
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


class OTelTracingHook(Hook):
    """Hook that exports Jaz hook spans via OTLP HTTP.

    Args:
        endpoint: OTLP HTTP traces endpoint (e.g., ``.../v1/traces``)
        service_name: OTel ``service.name`` resource attribute
        headers: Optional OTLP exporter headers (auth, tenant, etc.)
        max_attribute_length: Maximum length for string attributes. If None,
            strings are not truncated.
    """

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

    def on_event(self, event: Event) -> list:
        from jaz.hooks.events import (
            InvokeEnter,
            InvokeExit,
            LLMQueryEnter,
            LLMQueryExit,
            ReplExecutionEnter,
            ReplExecutionExit,
        )

        spans = self._get_active_spans()
        invoke_roots = self._get_invoke_roots()

        match event:
            case InvokeEnter(
                invoke_id=invoke_id,
                parent_invoke_id=parent_invoke_id,
                prompt=prompt,
                task_name=task_name,
                return_type=return_type,
                inputs=inputs,
                max_iterations=max_iterations,
                cur_recursion_depth=cur_depth,
                max_recursion_depth=max_depth,
            ):
                attributes: dict[str, Any] = {
                    "jaz.span_type": "invoke",
                    "jaz.invoke_id": invoke_id,
                    "jaz.prompt": self._truncate(
                        prompt if isinstance(prompt, str) else str(prompt)
                    ),
                    "jaz.return_type": str(return_type) if return_type else "None",
                    "jaz.max_iterations": max_iterations,
                    "jaz.cur_recursion_depth": cur_depth,
                    "jaz.max_recursion_depth": max_depth,
                    "jaz.inputs": str(inputs),
                }
                parent_span = spans.get(parent_invoke_id) if parent_invoke_id else None
                is_root_invoke = parent_span is None
                span = self.tracer.start_span(
                    "invoke",
                    attributes=attributes,
                    context=(set_span_in_context(parent_span) if parent_span else None),
                )
                self._set_input_attributes(
                    span, prompt, include_trace_fields=is_root_invoke
                )
                if is_root_invoke:
                    span.set_attribute("langfuse.trace.name", task_name)
                invoke_roots[invoke_id] = is_root_invoke
                spans[invoke_id] = span

            case InvokeExit(invoke_id=invoke_id, result=result):
                span = spans.pop(invoke_id, None)
                if span:
                    from jaz.repl.types import (
                        ErrorResult,
                        ExecuteResult,
                        RaiseResult,
                        ReturnResult,
                    )

                    is_root_invoke = invoke_roots.pop(invoke_id, False)

                    if isinstance(result, RaiseResult):
                        span.set_attribute("jaz.result_type", "RAISE")
                        span.set_attribute(
                            "jaz.error", self._truncate(str(result.output))
                        )
                        self._set_output_attributes(
                            span, result.output, include_trace_fields=is_root_invoke
                        )
                        span.set_status(Status(StatusCode.ERROR))
                    elif isinstance(result, ErrorResult):
                        span.set_attribute("jaz.result_type", "ERROR")
                        self._set_output_attributes(
                            span, result.output, include_trace_fields=is_root_invoke
                        )
                        span.set_status(Status(StatusCode.OK))
                    elif isinstance(result, ReturnResult):
                        span.set_attribute("jaz.result_type", "RETURN")
                        self._set_output_attributes(
                            span,
                            result.return_value,
                            include_trace_fields=is_root_invoke,
                        )
                        span.set_status(Status(StatusCode.OK))
                    elif isinstance(result, ExecuteResult):
                        span.set_attribute("jaz.result_type", "EXECUTE")
                        self._set_output_attributes(
                            span, result.output, include_trace_fields=is_root_invoke
                        )
                        span.set_status(Status(StatusCode.OK))
                    span.end()

            case ReplExecutionEnter(
                invoke_id=invoke_id,
                iteration=iteration,
                max_iterations=max_iterations,
                code=code,
                cur_recursion_depth=depth,
            ):
                attributes = {
                    "jaz.span_type": "repl_iteration",
                    "jaz.invoke_id": invoke_id,
                    "jaz.iteration": iteration,
                    "jaz.max_iterations": max_iterations,
                    "jaz.code": self._truncate(code),
                    "jaz.cur_recursion_depth": depth,
                }
                parent_span = spans.get(invoke_id)
                span = self.tracer.start_span(
                    f"repl_iteration_{iteration}",
                    attributes=attributes,
                    context=(set_span_in_context(parent_span) if parent_span else None),
                )
                self._set_input_attributes(span, code, include_trace_fields=False)
                spans[(invoke_id, "repl_iteration")] = span

            case ReplExecutionExit(invoke_id=invoke_id, exec_result=result):
                span = spans.pop((invoke_id, "repl_iteration"), None)
                if span:
                    from jaz.repl.types import (
                        ErrorResult,
                        ExecuteResult,
                        RaiseResult,
                        ReturnResult,
                    )

                    match result:
                        case RaiseResult(output=output, exception=exception):
                            span.set_attribute("jaz.exec_result_type", "RAISE")
                            output_text = str(output).strip()
                            output_value = (
                                output_text if output_text else f"RAISE: {exception!r}"
                            )
                            span.set_attribute(
                                "jaz.output", self._truncate(output_value)
                            )
                            self._set_output_attributes(
                                span, output_value, include_trace_fields=False
                            )
                            span.set_status(Status(StatusCode.ERROR))
                        case ErrorResult(output=output, error_summary=error_summary):
                            span.set_attribute("jaz.exec_result_type", "ERROR")
                            output_text = str(output).strip()
                            output_value = (
                                output_text
                                if output_text
                                else f"ERROR: {error_summary}"
                            )
                            span.set_attribute(
                                "jaz.output", self._truncate(output_value)
                            )
                            self._set_output_attributes(
                                span, output_value, include_trace_fields=False
                            )
                        case ReturnResult(output=output):
                            span.set_attribute("jaz.exec_result_type", "RETURN")
                            output_text = str(output).strip()
                            output_value = output_text if output_text else "RETURN"
                            span.set_attribute(
                                "jaz.output", self._truncate(output_value)
                            )
                            self._set_output_attributes(
                                span, output_value, include_trace_fields=False
                            )
                        case ExecuteResult(output=output):
                            span.set_attribute("jaz.exec_result_type", "EXECUTE")
                            output_text = str(output).strip()
                            output_value = (
                                output_text if output_text else "[no repl output]"
                            )
                            span.set_attribute(
                                "jaz.output", self._truncate(output_value)
                            )
                            self._set_output_attributes(
                                span, output_value, include_trace_fields=False
                            )
                    span.end()

            case LLMQueryEnter(
                invoke_id=invoke_id,
                messages=messages,
                model=model,
                iteration=iteration,
                cur_recursion_depth=depth,
            ):
                attributes: dict[str, Any] = {
                    "jaz.span_type": "llm_query",
                    "jaz.invoke_id": invoke_id,
                    "jaz.model": model,
                    "jaz.message_count": len(messages),
                    "langfuse.observation.type": "generation",
                    "langfuse.generation.model": model,
                }
                if messages:
                    if len(messages) >= 2 and messages[-2]["role"] == "user":
                        attributes["jaz.last_last_message"] = self._truncate(
                            self._format_message(messages[-2])
                        )
                    last_message = self._truncate(self._format_message(messages[-1]))
                    attributes["jaz.last_message"] = last_message
                if iteration is not None:
                    attributes["jaz.iteration"] = iteration
                if depth is not None:
                    attributes["jaz.cur_recursion_depth"] = depth

                parent_span = spans.get((invoke_id, "repl_iteration")) or spans.get(
                    invoke_id
                )
                span = self.tracer.start_span(
                    "llm_query",
                    attributes=attributes,
                    context=(set_span_in_context(parent_span) if parent_span else None),
                )
                if messages:
                    self._set_input_attributes(
                        span,
                        self._format_message(messages[-1]),
                        include_trace_fields=False,
                    )
                spans[(invoke_id, "llm_query")] = span

            case LLMQueryExit(
                invoke_id=invoke_id,
                response_content=resp,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            ):
                span = spans.pop((invoke_id, "llm_query"), None)
                if span:
                    if resp:
                        span.set_attribute("jaz.response", self._truncate(str(resp)))
                        self._set_output_attributes(
                            span, resp, include_trace_fields=False
                        )
                    if prompt_tokens is not None:
                        span.set_attribute("jaz.prompt_tokens", prompt_tokens)
                    if completion_tokens is not None:
                        span.set_attribute("jaz.completion_tokens", completion_tokens)
                    if cost is not None:
                        span.set_attribute("jaz.cost_usd", cost)
                    span.set_status(Status(StatusCode.OK))
                    span.end()

        return []

    def __exit__(self, exc_type, exc_value, traceback):
        spans = self._get_active_spans()
        invoke_roots = self._get_invoke_roots()
        for span_key in list(spans.keys()):
            span = spans.pop(span_key)
            if exc_type is not None:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", exc_type.__name__)
                span.set_attribute("error.message", str(exc_value))
                if isinstance(span_key, str):
                    self._set_output_attributes(
                        span,
                        f"{exc_type.__name__}: {exc_value}",
                        include_trace_fields=invoke_roots.get(span_key, False),
                    )
            span.end()
            if isinstance(span_key, str):
                invoke_roots.pop(span_key, None)

        self.provider.force_flush()
        return super().__exit__(exc_type, exc_value, traceback)
