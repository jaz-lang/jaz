"""Jaeger tracing hook that exports spans via OpenTelemetry OTLP HTTP."""

from jaz.hooks.builtin.otel_tracing import OTelTracingHook


class JaegerTracingHook(OTelTracingHook):
    """Hook preset for exporting Jaz traces to Jaeger.

    Args:
        endpoint: OTLP HTTP endpoint URL.
            Defaults to ``http://localhost:4318/v1/traces``.
        service_name: Service name shown in Jaeger. Defaults to ``jaz-agent``.
        max_attribute_length: Maximum length for string attributes.
            If None (default), no truncation is applied.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        service_name: str = "jaz-agent",
        max_attribute_length: int | None = None,
    ):
        super().__init__(
            endpoint=endpoint,
            service_name=service_name,
            max_attribute_length=max_attribute_length,
        )
