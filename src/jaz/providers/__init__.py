"""LLM backends — direct API access, with no third-party LLM wrapper dependency.

One class per backend (:class:`LLM`), selected by one tag. Add your own with
``@register_llm("mybackend")``; see :mod:`jaz.providers.llm` for what a backend owns and why
the former ``LLMClient``/``Provider`` split was merged away.
"""

from __future__ import annotations

from .anthropic import AnthropicLLM
from .base import (
    Choice,
    CompletionResponse,
    Message,
    MessageDict,
    Usage,
    declared_init_keys,
)
from .exceptions import (
    APIError,
    AuthenticationError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    LLMError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnsupportedParamsError,
)
from .extras_merge import (
    MergeRule,
    get_extra_merge_rule,
    merge_extras,
    register_extra_merge_rule,
)
from .llm import LLM, LLMResponse
from .openai import OpenAILLM
from .pricing import compute_cost, cost_per_token, get_all_models, get_model_info
from .registry import LLM_REGISTRY, register_llm

__all__ = [
    # Types
    "Choice",
    "CompletionResponse",
    "Message",
    "MessageDict",
    "LLM",
    "LLMResponse",
    "declared_init_keys",
    "Usage",
    # Registration
    "LLM_REGISTRY",
    "register_llm",
    # Exceptions
    "APIError",
    "AuthenticationError",
    "ContentPolicyViolationError",
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
    # Backends
    "AnthropicLLM",
    "OpenAILLM",
    # Extras merging (per-field rule registry for LLMResponse.extra)
    "MergeRule",
    "get_extra_merge_rule",
    "merge_extras",
    "register_extra_merge_rule",
]
