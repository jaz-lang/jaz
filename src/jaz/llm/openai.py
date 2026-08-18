"""OpenAI provider implementation.

Handles direct API calls to OpenAI's chat completions endpoint.

API reference: https://platform.openai.com/docs/api-reference/chat/create
Error codes: https://platform.openai.com/docs/guides/error-codes
"""

from __future__ import annotations

import logging
import os
import time as _time
from typing import Any

import httpx

from ..credentials import resolve_credential
from .base import Choice, CompletionResponse, Message, MessageDict, Usage
from .exceptions import (
    APIError,
    AuthenticationError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnsupportedParamsError,
    is_content_policy_violation,
)
from .llm import BaseLLM, LLMResponse

# https://platform.openai.com/docs/api-reference/chat/create
OPENAI_API_BASE = "https://api.openai.com/v1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("openai_provider.log")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_fh)


# Dormant for v1: LiteLLM is the sole registered backend
# (design/design_features/litellm_sole_backend_v1.md, #1082). This class stays in-tree and
# importable — `OpenAILLM(...)` still works if constructed directly — but carries no
# `@register_llm` tag, so `backend: openai` no longer resolves; reach OpenAI via `openai/…` on the
# litellm backend. Revive the built-in (for the deferred lean-core split) by restoring the
# `@register_llm("openai")` decorator (the class stays exported from `llm/__init__.py`).
class OpenAILLM(BaseLLM):
    """The OpenAI chat-completions backend.

    Resolves its key from ``api_key``, else the ``OPENAI_API_KEY`` environment variable,
    else the stored credential for ``"openai"`` in ``~/.jaz/credentials.json``.
    """

    # Resolving the key at construction (see `__init__`) is what keeps it off a serialized
    # config: `LLM.from_dict` rebuilds the backend from its authored params, and the key is
    # read from the environment or the credentials store at that point — never carried in
    # config, so it cannot leak into a persisted config record.

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        **retry: Any,
    ) -> None:
        # `**retry` forwards the base's retry_* settings. They stay discoverable by
        # `declared_init_keys`'s MRO walk despite the **kwargs, because they are declared on
        # `LLM.__init__` itself — which is what keeps `declared_init_keys` correct here.
        super().__init__(**retry)
        # The stored credential is the LAST resort, below the environment variable: an
        # exported key is the one-off override for a single run, while the stored file is the
        # persistent baseline. Resolution happens here, at construction, rather than per
        # request: `Agent.__init__` builds a fresh client per invoke, so a key stored mid-
        # session is picked up by the next invoke without a restart, and the request path
        # stays free of file I/O.
        self.api_key = (
            api_key or os.environ.get("OPENAI_API_KEY") or resolve_credential("openai")
        )
        self.base_url = (
            base_url or os.environ.get("OPENAI_API_BASE") or OPENAI_API_BASE
        ).rstrip("/")
        self.timeout = timeout

    def complete(
        self, model: str, messages: list[MessageDict], **kwargs: Any
    ) -> LLMResponse:
        service_tier = kwargs.get("service_tier")
        return self.finalize(
            self._request(model, messages, **kwargs), model, service_tier=service_tier
        )

    async def acomplete(
        self, model: str, messages: list[MessageDict], **kwargs: Any
    ) -> LLMResponse:
        service_tier = kwargs.get("service_tier")
        return self.finalize(
            await self._arequest(model, messages, **kwargs),
            model,
            service_tier=service_tier,
        )

    def _request(
        self,
        model: str,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> CompletionResponse:
        if not self.api_key:
            raise AuthenticationError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable, "
                "or store one from the jaz console with set_credential('openai').",
                llm_provider="openai",
            )

        # Build request payload — model and messages are required.
        # https://platform.openai.com/docs/api-reference/chat/create
        # Sent verbatim, so the id is checked rather than converted.
        self.validate_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # A per-call `timeout` (e.g. from model_config) sizes the HTTP transport, not the
        # request body — pop it before forwarding the rest as OpenAI params. (#623)
        request_timeout = self._resolve_request_timeout(kwargs, self.timeout)

        # Forward optional parameters (only if not None)
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        _start = _time.time()
        _prompt_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)
        logger.debug(
            "POST %s/chat/completions | model=%s | ~%d prompt tokens | timeout=%ss",
            self.base_url,
            model,
            _prompt_tokens,
            request_timeout,
        )

        try:
            with httpx.Client(timeout=request_timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as e:
            _elapsed = _time.time() - _start
            logger.error("TIMEOUT after %.1fs: %s: %s", _elapsed, type(e).__name__, e)
            raise APIError(
                f"Request timed out after {request_timeout}s",
                llm_provider="openai",
            ) from e
        except httpx.RequestError as e:
            _elapsed = _time.time() - _start
            logger.error(
                "REQUEST ERROR after %.1fs: %s: %s", _elapsed, type(e).__name__, e
            )
            raise APIError(
                f"Request failed: {e}",
                llm_provider="openai",
            ) from e
        except Exception as e:
            _elapsed = _time.time() - _start
            logger.error(
                "UNEXPECTED ERROR after %.1fs: %s: %s", _elapsed, type(e).__name__, e
            )
            raise

        _elapsed = _time.time() - _start
        logger.debug(
            "Response: status=%d | %.1fs | %d bytes",
            response.status_code,
            _elapsed,
            len(response.content),
        )

        if response.status_code != 200:
            self._handle_error_response(response)

        data = response.json()
        return self._parse_response(data)

    async def _arequest(
        self,
        model: str,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> CompletionResponse:
        if not self.api_key:
            raise AuthenticationError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable, "
                "or store one from the jaz console with set_credential('openai').",
                llm_provider="openai",
            )

        # Sent verbatim, so the id is checked rather than converted.

        self.validate_model(model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # See complete(): a per-call `timeout` sizes the transport, not the payload. (#623)
        request_timeout = self._resolve_request_timeout(kwargs, self.timeout)

        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        _prompt_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)
        logger.debug(
            "POST %s/chat/completions | model=%s | ~%d prompt tokens | timeout=%ss",
            self.base_url,
            model,
            _prompt_tokens,
            request_timeout,
        )

        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as e:
            logger.error("TIMEOUT: %s: %s", type(e).__name__, e)
            raise APIError(
                f"Request timed out after {request_timeout}s",
                llm_provider="openai",
            ) from e
        except httpx.RequestError as e:
            logger.error("REQUEST ERROR: %s: %s", type(e).__name__, e)
            raise APIError(
                f"Request failed: {e}",
                llm_provider="openai",
            ) from e

        logger.debug(
            "Response: status=%d | %d bytes",
            response.status_code,
            len(response.content),
        )

        if response.status_code != 200:
            self._handle_error_response(response)

        data = response.json()
        return self._parse_response(data)

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Map HTTP error responses to exception types.

        Status code mapping per https://platform.openai.com/docs/guides/error-codes:
          400 -> invalid_request_error (or context_length_exceeded)
          401 -> AuthenticationError
          403 -> PermissionDeniedError
          404 -> NotFoundError
          429 -> RateLimitError
          500 -> APIError (server error)
          503 -> APIError (engine overloaded)
        """
        status_code = response.status_code
        try:
            # Error envelope: {"error": {"message": "...", "type": "...", "code": "..."}}
            # https://platform.openai.com/docs/guides/error-codes
            error_data = response.json()
            error = error_data.get("error", {})
            message = error.get("message", response.text)
            error_code = error.get("code", "")
        except Exception:
            message = response.text
            error_code = ""

        if status_code == 401:
            raise AuthenticationError(
                message, status_code=status_code, llm_provider="openai"
            )
        elif status_code == 403:
            raise PermissionDeniedError(
                message, status_code=status_code, llm_provider="openai"
            )
        elif status_code == 404:
            raise NotFoundError(message, status_code=status_code, llm_provider="openai")
        elif status_code == 429:
            raise RateLimitError(
                message, status_code=status_code, llm_provider="openai"
            )
        elif status_code == 400:
            # "context_length_exceeded" is the error code for context window issues.
            if error_code == "context_length_exceeded":
                raise ContextWindowExceededError(
                    message, status_code=status_code, llm_provider="openai"
                )
            # Content/usage-policy rejection (e.g. code "invalid_prompt" or
            # "...flagged as potentially violating our usage policy"). Distinct
            # from a malformed-params 400 — the request is well-formed.
            if is_content_policy_violation(error_code, message):
                raise ContentPolicyViolationError(
                    message, status_code=status_code, llm_provider="openai"
                )
            # "Could not parse the JSON body" is a transient error from network
            # corruption — the same request succeeds on retry.
            # https://github.com/HKUDS/LightRAG/issues/2794
            if "could not parse the json body" in message.lower():
                raise APIError(message, status_code=status_code, llm_provider="openai")
            raise UnsupportedParamsError(
                message, status_code=status_code, llm_provider="openai"
            )
        else:
            # 500 (internal server error), 503 (engine overloaded), and others
            raise APIError(message, status_code=status_code, llm_provider="openai")

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        """Parse the API response JSON into a CompletionResponse.

        Response schema: https://platform.openai.com/docs/api-reference/chat/object
        """
        choices: list[Choice] = []
        for choice_data in data.get("choices", []):
            message_data = choice_data.get("message", {})
            message = Message(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content"),
            )
            choices.append(
                Choice(
                    message=message,
                    index=choice_data.get("index", 0),
                    # TODO(#669): finish_reason == "content_filter" (200-path
                    # content block) is recorded but not raised as
                    # ContentPolicyViolationError.
                    finish_reason=choice_data.get("finish_reason"),
                )
            )

        usage_data = data.get("usage")
        usage: Usage | None = None
        if usage_data:
            # OpenAI reports cached tokens in prompt_tokens_details.cached_tokens
            # (subset of prompt_tokens). Reasoning/prediction sub-counts live in
            # completion_tokens_details — they're already inside completion_tokens
            # for billing, so we surface them via Usage.extra for observability.
            #
            # cache_creation_input_tokens is deliberately left at its 0 default:
            # OpenAI's prompt caching writes are automatic and UNBILLED (only
            # reads are discounted), and the usage payload has no cache-creation/
            # write field to read. The 0 is therefore honest for *cost* — OpenAI's
            # true cache-creation cost is $0, so it contributes $0 in compute_cost.
            # It is NOT a claim that "no tokens were cached": tokens are physically
            # written to OpenAI's cache, OpenAI just doesn't expose or charge for
            # that count. So for OpenAI this 0 means "not reported / not billed",
            # which coincides with a genuine zero cost (unlike Anthropic, which
            # reports a real cache_creation_input_tokens billed at the write rate).
            prompt_details = usage_data.get("prompt_tokens_details") or {}
            completion_details = usage_data.get("completion_tokens_details") or {}
            extra: dict[str, Any] = {}
            for key in (
                "reasoning_tokens",
                "accepted_prediction_tokens",
                "rejected_prediction_tokens",
                "audio_tokens",
            ):
                value = completion_details.get(key)
                if value:
                    extra[key] = value
            prompt_audio = prompt_details.get("audio_tokens")
            if prompt_audio:
                extra["prompt_audio_tokens"] = prompt_audio
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                cache_read_input_tokens=prompt_details.get("cached_tokens", 0),
                extra=extra,
            )

        return CompletionResponse(
            choices=choices,
            usage=usage,
            model=data.get("model", ""),
            id=data.get("id", ""),
        )
