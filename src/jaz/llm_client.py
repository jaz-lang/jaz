from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .providers import (
    AnthropicProvider,
    AuthenticationError,
    CompletionResponse,
    ContextWindowExceededError,
    NotFoundError,
    OpenAIProvider,
    PermissionDeniedError,
    Provider,
    UnsupportedParamsError,
    compute_cost,
)
from .providers import (
    get_model_info as _get_model_info,
)


@dataclass
class LLMResponse:
    """Normalized response from an LLM completion call."""

    content: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    raw_response: Any = field(default=None, repr=False)


class LLMClient(ABC):
    """Abstract base class for LLM completion providers.

    Agent owns retry logic, hook dispatcher integration, and cost tracking.
    Implementations only need to provide the raw completion call.
    """

    @abstractmethod
    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        """Execute a completion call against the LLM.

        Args:
            model: The model identifier (e.g., "openai/gpt-4").
            messages: The conversation messages in OpenAI format.
            **kwargs: Additional provider-specific parameters (temperature, max_tokens, etc.).

        Returns:
            An LLMResponse with the completion content and usage information.
        """
        ...

    @property
    def non_retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        """Exception types that should NOT be retried.

        Returns a tuple of exception types. If the completion call raises one
        of these, the retry logic will immediately re-raise instead of retrying.
        """
        return (KeyboardInterrupt,)

    @abstractmethod
    def get_model(self, model_config: dict[str, Any]) -> str:
        """Extract the model name from model_config.

        Args:
            model_config: Configuration dict containing a 'model' key.

        Returns:
            The model identifier string, or empty string if not set.
        """
        ...

    @abstractmethod
    def get_model_info(self, model_config: dict[str, object]) -> dict[str, Any]:
        """Get model metadata.

        Args:
            model_config: Configuration dict containing a 'model' key.

        Returns:
            A dictionary of model metadata.
        """
        ...


def create_llm_client(client_type: str) -> LLMClient:
    """Factory function to create an LLM client from a string identifier.

    Args:
        client_type: The type of client to create ("litellm" or "rlm").

    Returns:
        An LLMClient instance.
    """
    if client_type == "litellm":
        # For backward compatibility, "litellm" now uses DirectLLMClient
        return DirectLLMClient()
    elif client_type == "rlm":
        from .llm_client_rlm import RLMClient

        return RLMClient()
    elif client_type == "tail_call_delegation":
        from .llm_client_delegate import TailCallDelegationClient

        return TailCallDelegationClient()
    else:
        raise ValueError(f"Unknown LLM client type: {client_type!r}")


class DirectLLMClient(LLMClient):
    """Default LLM client that makes direct API calls to providers.

    Supports OpenAI and Anthropic APIs. Model strings must use the format
    "provider/model" (e.g., "openai/gpt-4o", "anthropic/claude-3-opus-20240229").
    """

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
        }

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DirectLLMClient)

    def __hash__(self) -> int:
        return hash(type(self))

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        provider_name, model_name = self._parse_model(model)
        provider = self._providers.get(provider_name)

        if provider is None:
            raise ValueError(
                f"Unknown provider: {provider_name!r}. "
                f"Supported providers: {list(self._providers.keys())}"
            )

        response: CompletionResponse = provider.complete(model_name, messages, **kwargs)

        # Extract usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None

        # Compute cost from bundled pricing data (accounting for cached token rates)
        cost: float | None = None
        if prompt_tokens is not None and completion_tokens is not None:
            cost = compute_cost(
                model_name,
                prompt_tokens,
                completion_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens
                if usage
                else 0,
                cache_read_input_tokens=usage.cache_read_input_tokens if usage else 0,
            )

        # Extract content
        content: str | None = None
        if response.choices:
            content = response.choices[0].message.content

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            raw_response=response,
        )

    @property
    def non_retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        return (
            UnsupportedParamsError,
            NotFoundError,
            PermissionDeniedError,
            ContextWindowExceededError,
            AuthenticationError,
            KeyboardInterrupt,
        )

    def get_model(self, model_config: dict[str, object]) -> str:
        model = model_config["model"]
        assert isinstance(model, str)
        return model

    def get_model_info(self, model_config: dict[str, object]) -> dict[str, Any]:
        model = self.get_model(model_config)
        _, model_name = self._parse_model(model)
        return _get_model_info(model_name)

    def _parse_model(self, model: str) -> tuple[str, str]:
        """Parse a model string into (provider, model_name).

        Args:
            model: Model string like "openai/gpt-4o" or "anthropic/claude-3-opus".

        Returns:
            Tuple of (provider_name, model_name).

        Raises:
            ValueError: If model string doesn't include a provider prefix.
        """
        if "/" not in model:
            raise ValueError(
                f"Model string must include provider prefix "
                f"(e.g., 'openai/{model}' or 'anthropic/{model}')"
            )
        provider, model_name = model.split("/", 1)
        return provider.lower(), model_name


class MockLLMClient(LLMClient):
    """LLM client that delegates to a user-provided callback.

    The callback receives ``(model, messages, **kwargs)`` and returns either:

    * A ``str`` (convenience) — wrapped into an ``LLMResponse`` with zero
      token counts and zero cost.
    * An ``LLMResponse`` (full control) — returned as-is.

    Example::

        def my_llm(model, messages, **kwargs):
            last = messages[-1]["content"]
            if "summarize" in last:
                return "<repl_input lang='python'>\\nRETURN 'summary'\\n</repl_input>"
            return "<repl_input lang='python'>\\nRETURN 42\\n</repl_input>"

        mock = MockLLMClient(fn=my_llm)
        jaz.configure(llm_client=mock)
    """

    def __init__(self, fn: Callable[..., str | LLMResponse]) -> None:
        self._fn = fn

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        result = self._fn(model, messages, **kwargs)
        if isinstance(result, str):
            return LLMResponse(
                content=result,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost=0.0,
            )
        return result

    def get_model(self, model_config: dict[str, Any]) -> str:
        return str(model_config.get("model", "mock"))

    def get_model_info(self, model_config: dict[str, object]) -> dict[str, Any]:
        return {}

    def __deepcopy__(self, memo: dict[int, Any]) -> MockLLMClient:
        # Mock clients are shared external dependencies, not owned state.
        # config_override() deepcopies the config; we want the same mock instance.
        return self


# Alias for backward compatibility
LiteLLMClient = DirectLLMClient
