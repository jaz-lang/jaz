"""Jaeger tracing hook that exports spans via OpenTelemetry OTLP HTTP."""

from jaz.hooks.builtin.otel_tracing import OTelTracing


class JaegerTracing(OTelTracing):
    """Hook preset for exporting JAZ traces to Jaeger.

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


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_legacy_hook_names_still_importable``).
JaegerTracingHook = JaegerTracing
