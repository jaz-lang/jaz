"""LLM provider implementations.

This package provides direct API access to LLM providers,
providing direct API access without third-party LLM wrapper dependencies.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import Choice, CompletionResponse, Message, MessageDict, Provider, Usage
from .exceptions import (
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    LLMError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnsupportedParamsError,
)
from .openai import OpenAIProvider
from .pricing import compute_cost, cost_per_token, get_all_models, get_model_info

__all__ = [
    # Types
    "Choice",
    "CompletionResponse",
    "Message",
    "MessageDict",
    "Provider",
    "Usage",
    # Exceptions
    "APIError",
    "AuthenticationError",
    "ContextWindowExceededError",
    "LLMError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "UnsupportedParamsError",
    # Pricing
    "compute_cost",
    "cost_per_token",
    "get_all_models",
    "get_model_info",
    # Providers
    "AnthropicProvider",
    "OpenAIProvider",
]
