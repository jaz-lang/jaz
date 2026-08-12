"""Langfuse Cloud tracing hook via OpenTelemetry OTLP HTTP export."""

from __future__ import annotations

import base64
import os

from jaz.hooks.builtin.otel_tracing import OTelTracing


def _normalize_langfuse_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _build_langfuse_headers(public_key: str, secret_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("utf-8")
    return {"Authorization": f"Basic {token}"}


class LangfuseTracing(OTelTracing):
    """Hook preset for exporting Jaz traces to Langfuse Cloud.

    By default, credentials are read from environment:
    - ``LANGFUSE_PUBLIC_KEY``
    - ``LANGFUSE_SECRET_KEY``
    - ``LANGFUSE_HOST`` (optional, defaults to ``https://cloud.langfuse.com``)
      ``LANGFUSE_BASE_URL`` is also supported as an alias.

    Args:
        public_key: Langfuse public key. Falls back to env if omitted.
        secret_key: Langfuse secret key. Falls back to env if omitted.
        base_url: Langfuse base URL. Falls back to env/default if omitted.
        service_name: OTel service name attribute.
        max_attribute_length: Maximum length for string attributes.
            If None (default), no truncation is applied.
    """

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        service_name: str = "jaz-agent",
        max_attribute_length: int | None = None,
    ):
        resolved_public = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        resolved_secret = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        resolved_base = (
            base_url
            or os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com"
        )

        if not resolved_public or not resolved_secret:
            raise ValueError(
                "LangfuseTracing requires credentials. Provide public_key/secret_key "
                "or set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."
            )

        endpoint = (
            f"{_normalize_langfuse_base_url(resolved_base)}/api/public/otel/v1/traces"
        )
        headers = _build_langfuse_headers(resolved_public, resolved_secret)
        super().__init__(
            endpoint=endpoint,
            service_name=service_name,
            headers=headers,
            max_attribute_length=max_attribute_length,
        )


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``).
LangfuseTracingHook = LangfuseTracing
