"""Base types and abstract provider interface.

Provider-agnostic types for LLM completion requests and responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Type alias for OpenAI-style message format (replaces AllMessageValues)
MessageDict = dict[str, Any]


@dataclass
class Usage:
    """Token usage information from a completion call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class Message:
    """A message in the completion response."""

    role: str
    content: str | None


@dataclass
class Choice:
    """A single choice/completion from the LLM response."""

    message: Message
    index: int = 0
    finish_reason: str | None = None


@dataclass
class CompletionResponse:
    """Normalized response from an LLM completion call.

    This structure is provider-agnostic and mirrors the OpenAI response format.
    """

    choices: list[Choice] = field(default_factory=list)
    usage: Usage | None = None
    model: str = ""
    id: str = ""


class Provider(ABC):
    """Abstract base class for LLM providers.

    Each provider implementation handles the specifics of its API:
    - Request formatting
    - Authentication
    - Error mapping
    - Response parsing
    """

    @abstractmethod
    def complete(
        self,
        model: str,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a completion call against the provider's API.

        Args:
            model: The model identifier (e.g., "gpt-4o", "claude-3-opus").
            messages: The conversation messages in OpenAI format.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A CompletionResponse with the completion content and usage info.

        Raises:
            LLMError: Or one of its subclasses for API errors.
        """
        ...
