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

from .base import Choice, CompletionResponse, Message, MessageDict, Provider, Usage
from .exceptions import (
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnsupportedParamsError,
)

# https://platform.openai.com/docs/api-reference/chat/create
OPENAI_API_BASE = "https://api.openai.com/v1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("openai_provider.log")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_fh)


class OpenAIProvider(Provider):
    """OpenAI API provider.

    Uses the OPENAI_API_KEY environment variable for authentication.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url or os.environ.get("OPENAI_API_BASE") or OPENAI_API_BASE
        ).rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        model: str,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> CompletionResponse:
        if not self.api_key:
            raise AuthenticationError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable.",
                llm_provider="openai",
            )

        # Build request payload — model and messages are required.
        # https://platform.openai.com/docs/api-reference/chat/create
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

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
            self.timeout,
        )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as e:
            _elapsed = _time.time() - _start
            logger.error("TIMEOUT after %.1fs: %s: %s", _elapsed, type(e).__name__, e)
            raise APIError(
                f"Request timed out after {self.timeout}s",
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
                    finish_reason=choice_data.get("finish_reason"),
                )
            )

        usage_data = data.get("usage")
        usage: Usage | None = None
        if usage_data:
            # OpenAI reports cached tokens in prompt_tokens_details.cached_tokens
            prompt_details = usage_data.get("prompt_tokens_details") or {}
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                cache_read_input_tokens=prompt_details.get("cached_tokens", 0),
            )

        return CompletionResponse(
            choices=choices,
            usage=usage,
            model=data.get("model", ""),
            id=data.get("id", ""),
        )
