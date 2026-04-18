"""Custom exception hierarchy for LLM provider errors.

Unified error handling interface across different LLM providers.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        llm_provider: str | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.llm_provider = llm_provider
        super().__init__(message)


class RateLimitError(LLMError):
    """Raised when the API rate limit is exceeded.

    This is a retryable error - the request can be retried after a delay.
    """

    pass


class AuthenticationError(LLMError):
    """Raised when API authentication fails (invalid API key, etc.).

    This is NOT retryable - the API key or credentials need to be fixed.
    """

    pass


class ContextWindowExceededError(LLMError):
    """Raised when the input exceeds the model's context window.

    This is NOT retryable with the same input.
    """

    pass


class NotFoundError(LLMError):
    """Raised when the requested model or resource is not found.

    This is NOT retryable - the model name or resource needs to be corrected.
    """

    pass


class PermissionDeniedError(LLMError):
    """Raised when the API key lacks permission for the requested operation.

    This is NOT retryable - permissions need to be granted.
    """

    pass


class UnsupportedParamsError(LLMError):
    """Raised when unsupported parameters are passed to the API.

    This is NOT retryable - the parameters need to be corrected.
    """

    pass


class APIError(LLMError):
    """Generic API error for other server-side issues.

    May or may not be retryable depending on the specific error.
    """

    pass
