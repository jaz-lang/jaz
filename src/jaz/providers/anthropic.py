"""Anthropic provider implementation.

Handles direct API calls to Anthropic's messages endpoint.
Converts OpenAI-style messages to Anthropic's format.

API reference: https://docs.anthropic.com/en/api/messages
Errors: https://docs.anthropic.com/en/api/errors
"""

from __future__ import annotations

import os
from typing import Any

import httpx

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
from .llm import LLM, LLMResponse
from .registry import register_llm

# https://docs.anthropic.com/en/api/messages
ANTHROPIC_API_BASE = "https://api.anthropic.com"

# https://docs.anthropic.com/en/api/versioning
ANTHROPIC_API_VERSION = "2023-06-01"

# Anthropic requires max_tokens; use this when the caller doesn't provide one.
DEFAULT_MAX_TOKENS = 8192

# OpenAI parameters that have no Anthropic equivalent and should be silently dropped.
_OPENAI_ONLY_PARAMS = frozenset(
    {
        "n",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "response_format",
        "parallel_tool_calls",
        "service_tier",
        "store",
    }
)

# Map OpenAI parameter names to their Anthropic equivalents.
_PARAM_MAPPING = {
    "stop": "stop_sequences",
    "max_completion_tokens": "max_tokens",
}


@register_llm("anthropic")
class AnthropicLLM(LLM):
    """The Anthropic messages backend.

    Reads ``ANTHROPIC_API_KEY`` from the environment when no ``api_key`` is configured, so a
    recorded config never has to carry the key. Converts OpenAI-style messages to Anthropic's
    format automatically.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        **retry: Any,
    ) -> None:
        # See OpenAILLM.__init__ on why `**retry` keeps `declared_init_keys` correct.
        super().__init__(**retry)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (
            base_url or os.environ.get("ANTHROPIC_API_BASE") or ANTHROPIC_API_BASE
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
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.",
                llm_provider="anthropic",
            )

        # Convert OpenAI-style messages to Anthropic format.
        # System messages become a top-level "system" field; user/assistant
        # messages go in the "messages" array.
        # https://docs.anthropic.com/en/api/messages
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Sent verbatim, so the id is checked rather than converted.

        self.validate_model(model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
        }

        if system_prompt is not None:
            payload["system"] = system_prompt

        # Anthropic uses "max_tokens" where OpenAI uses "max_completion_tokens".
        # Pop both and let the mapped name go into the payload via _PARAM_MAPPING.
        max_tokens = (
            kwargs.pop("max_tokens", None)
            or kwargs.pop("max_completion_tokens", None)
            or DEFAULT_MAX_TOKENS
        )
        payload["max_tokens"] = max_tokens

        # A per-call `timeout` (e.g. from model_config) sizes the HTTP transport, not the
        # request body — pop it before forwarding the rest as Anthropic params. (#623)
        request_timeout = self._resolve_request_timeout(kwargs, self.timeout)

        # Map remaining kwargs, dropping OpenAI-only params.
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in _OPENAI_ONLY_PARAMS:
                continue
            mapped_key = _PARAM_MAPPING.get(key, key)
            payload[mapped_key] = value

        # https://docs.anthropic.com/en/api/getting-started#authentication
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_API_VERSION,
        }

        try:
            with httpx.Client(timeout=request_timeout) as client:
                response = client.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as e:
            raise APIError(
                f"Request timed out after {request_timeout}s",
                llm_provider="anthropic",
            ) from e
        except httpx.RequestError as e:
            raise APIError(
                f"Request failed: {e}",
                llm_provider="anthropic",
            ) from e

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
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.",
                llm_provider="anthropic",
            )

        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Sent verbatim, so the id is checked rather than converted.

        self.validate_model(model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
        }

        if system_prompt is not None:
            payload["system"] = system_prompt

        max_tokens = kwargs.pop("max_tokens", None) or kwargs.pop(
            "max_completion_tokens", None
        )
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # See complete(): a per-call `timeout` sizes the transport, not the payload. (#623)
        request_timeout = self._resolve_request_timeout(kwargs, self.timeout)

        for key, value in kwargs.items():
            if value is None:
                continue
            if key in _OPENAI_ONLY_PARAMS:
                continue
            mapped_key = _PARAM_MAPPING.get(key, key)
            payload[mapped_key] = value

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_API_VERSION,
        }

        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as e:
            raise APIError(
                f"Request timed out after {request_timeout}s",
                llm_provider="anthropic",
            ) from e
        except httpx.RequestError as e:
            raise APIError(
                f"Request failed: {e}",
                llm_provider="anthropic",
            ) from e

        if response.status_code != 200:
            self._handle_error_response(response)

        data = response.json()
        return self._parse_response(data)

    def _convert_messages(
        self, messages: list[MessageDict]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert OpenAI-style messages to Anthropic format.

        Key differences from OpenAI (https://docs.anthropic.com/en/api/messages):
        - System messages are a top-level "system" field, not in the messages array.
        - Only "user" and "assistant" roles are allowed in messages.
        - Messages must alternate between user and assistant; consecutive
          same-role messages are merged.
        - Content can be a string or an array of content blocks.
        """
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role in ("system", "developer"):
                # Anthropic uses a top-level "system" field.
                if isinstance(content, list):
                    text = self._extract_text_from_blocks(content)
                else:
                    text = content
                system_parts.append(text)
            elif role in ("user", "assistant"):
                # Pass content through as-is (string or list of content blocks).
                # Anthropic natively supports content block arrays.
                if anthropic_messages and anthropic_messages[-1]["role"] == role:
                    # Merge consecutive same-role messages.
                    prev = anthropic_messages[-1]["content"]
                    if isinstance(prev, str) and isinstance(content, str):
                        anthropic_messages[-1]["content"] = f"{prev}\n\n{content}"
                    else:
                        # Convert both to block lists and concatenate.
                        prev_blocks = (
                            [{"type": "text", "text": prev}]
                            if isinstance(prev, str)
                            else prev
                        )
                        new_blocks = (
                            [{"type": "text", "text": content}]
                            if isinstance(content, str)
                            else content
                        )
                        anthropic_messages[-1]["content"] = prev_blocks + new_blocks
                else:
                    anthropic_messages.append({"role": role, "content": content})

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, anthropic_messages

    @staticmethod
    def _extract_text_from_blocks(blocks: list[Any]) -> str:
        """Extract text from an OpenAI content block array."""
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Map HTTP error responses to exception types.

        Status code mapping per https://docs.anthropic.com/en/api/errors:
          400 -> invalid_request_error
          401 -> authentication_error
          403 -> permission_error
          404 -> not_found_error
          413 -> request_too_large
          429 -> rate_limit_error
          500 -> api_error
          529 -> overloaded_error
        """
        status_code = response.status_code
        try:
            # Error envelope: {"type": "error", "error": {"type": "...", "message": "..."}}
            # https://docs.anthropic.com/en/api/errors
            error_data = response.json()
            error = error_data.get("error", {})
            message = error.get("message", response.text)
        except Exception:
            message = response.text

        if status_code == 401:
            raise AuthenticationError(
                message, status_code=status_code, llm_provider="anthropic"
            )
        elif status_code == 403:
            raise PermissionDeniedError(
                message, status_code=status_code, llm_provider="anthropic"
            )
        elif status_code == 404:
            raise NotFoundError(
                message, status_code=status_code, llm_provider="anthropic"
            )
        elif status_code == 429:
            raise RateLimitError(
                message, status_code=status_code, llm_provider="anthropic"
            )
        elif status_code == 400:
            # Anthropic returns error_type "invalid_request_error" for all 400s.
            # The actual context-window error message is "prompt is too long: N tokens > M maximum".
            # https://docs.anthropic.com/en/api/errors
            if "prompt is too long" in message.lower():
                raise ContextWindowExceededError(
                    message, status_code=status_code, llm_provider="anthropic"
                )
            # Content/usage-policy rejection (well-formed request, blocked content).
            # Pass code=None: Anthropic 400s carry only error_type
            # "invalid_request_error" (never a granular content-policy code), so
            # detection here is necessarily message-only.
            if is_content_policy_violation(None, message):
                raise ContentPolicyViolationError(
                    message, status_code=status_code, llm_provider="anthropic"
                )
            raise UnsupportedParamsError(
                message, status_code=status_code, llm_provider="anthropic"
            )
        elif status_code == 413:
            raise ContextWindowExceededError(
                message, status_code=status_code, llm_provider="anthropic"
            )
        elif status_code in (500, 529):
            # 529 is Anthropic-specific: "overloaded_error"
            raise APIError(message, status_code=status_code, llm_provider="anthropic")
        else:
            raise APIError(message, status_code=status_code, llm_provider="anthropic")

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        """Parse the API response JSON into a CompletionResponse.

        Response schema: https://platform.claude.com/docs/en/api/messages/create#message
        - content is an array of blocks: [{"type": "text", "text": "..."}]
        - usage has input_tokens and output_tokens (not prompt/completion)
        - stop reason is "end_turn", "stop_sequence", "max_tokens", or "tool_use"
        """
        content_blocks = data.get("content", [])
        text_content = ""
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_content += block.get("text", "")

        message = Message(
            role="assistant",
            content=text_content if text_content else None,
        )

        choice = Choice(
            message=message,
            index=0,
            # TODO(#669): stop_reason == "refusal" (Claude 4+ blocks content on a
            # 200 here) is recorded but not raised as ContentPolicyViolationError.
            finish_reason=data.get("stop_reason"),
        )

        # Anthropic uses input_tokens/output_tokens, not prompt_tokens/completion_tokens.
        # Cache tokens are reported separately and are NOT included in input_tokens
        # (unlike OpenAI's prompt_tokens, which is a superset that includes cached_tokens).
        # We normalize to the OpenAI/ATIF convention: Usage.prompt_tokens is the total
        # of non-cached input + cache creation + cache read. This keeps cost computation
        # in providers/pricing.py correct (it subtracts cache buckets from prompt_tokens)
        # and matches ATIF v1.7 MetricsSchema semantics.
        # See https://platform.claude.com/docs/en/api/messages/create#message.usage
        usage_data = data.get("usage")
        usage: Usage | None = None
        if usage_data:
            input_tokens = usage_data.get("input_tokens", 0)
            output_tokens = usage_data.get("output_tokens", 0)
            cache_creation = usage_data.get("cache_creation_input_tokens", 0)
            cache_read = usage_data.get("cache_read_input_tokens", 0)
            prompt_tokens = input_tokens + cache_creation + cache_read
            extra: dict[str, Any] = {}
            cache_creation_detail = usage_data.get("cache_creation")
            if cache_creation_detail:
                extra["cache_creation"] = cache_creation_detail
            service_tier = usage_data.get("service_tier")
            if service_tier is not None:
                extra["service_tier"] = service_tier
            server_tool_use = usage_data.get("server_tool_use")
            if server_tool_use is not None:
                extra["server_tool_use"] = server_tool_use
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=output_tokens,
                total_tokens=prompt_tokens + output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                extra=extra,
            )

        return CompletionResponse(
            choices=[choice],
            usage=usage,
            model=data.get("model", ""),
            id=data.get("id", ""),
        )
