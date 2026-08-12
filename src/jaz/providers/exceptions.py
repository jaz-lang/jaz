"""LLM provider error taxonomy — re-exported from the central ``jaz.exceptions``.

The provider error classes are **defined in ``jaz.exceptions``** (so they share the
``JazException`` base — one catch-all — and are importable from the single public home
``jaz.exceptions`` without exposing the internal ``jaz.providers`` package). They are
re-exported here for back-compat: ``from jaz.providers.exceptions import X`` still works,
and this remains the module the provider implementations import from. This module also
hosts the provider-only ``is_content_policy_violation`` helper used by the OpenAI /
Anthropic error mappers.
"""

from __future__ import annotations

# Absolute rather than the relative form used elsewhere in src/jaz, deliberately: this
# module is a shim onto a *fixed canonical home*, so the import should name that home.
# `..exceptions` resolves relative to this module's own package, so moving this file to
# another depth (e.g. providers/llm/exceptions.py) would silently repoint it at a path
# that doesn't exist; `jaz.exceptions` survives any move. The usual argument for relative
# imports — surviving a rename or vendoring of the `jaz` package itself — is the less
# likely event, and would break this shim's premise anyway.
from jaz.exceptions import (
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

__all__ = [
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "NotFoundError",
    "PermissionDeniedError",
    "UnsupportedParamsError",
    "APIError",
    "is_content_policy_violation",
]


# Provider error codes / message markers that indicate a content/usage-policy
# rejection (a 400 whose *content* was blocked, not a malformed request).
#
# Source: OpenAI/Azure/Anthropic 400 responses observed in practice, kept at
# parity with litellm's cross-provider mapping (the broadest maintained reference):
# https://github.com/BerriAI/litellm/blob/main/litellm/litellm_core_utils/exception_mapping_utils.py
# Anthropic's real 400 envelope is confirmed as
#   {"type":"error","error":{"type":"invalid_request_error",
#    "message":"Output blocked by content filtering policy"}}
# (widely reported, e.g. anthropics/claude-code#6195) — matched by the
# `content filtering policy` marker below; its "...violate our Usage Policy..."
# variant is matched by `usage policy`.
#
# Detection is two-tier and intentionally code-first: the structured error
# `code` is stable, so it is authoritative; the message markers are a best-effort
# fallback for responses that carry only prose. Prose matching is inherently
# drift-prone — if a provider rewords a message we simply miss it and the 400
# falls through to UnsupportedParamsError (non-retryable): mislabeled, but no
# retry loop. Prefer curating `code`s over markers when a stable code exists.
_CONTENT_POLICY_CODES = frozenset(
    {
        "content_policy_violation",  # OpenAI / Azure
        "invalid_prompt",  # OpenAI
        "responsibleaipolicyviolation",  # Azure Responsible AI (matched lowercased)
    }
)
_CONTENT_POLICY_MARKERS = (
    "usage policy",  # OpenAI "...violating our usage policy"; Anthropic "...violate our Usage Policy"
    "content management policy",  # Azure OpenAI content filter
    "content filtering policy",  # Anthropic 400 "Output blocked by content filtering policy" (claude-code#6195)
    "content_filter_policy",  # Azure underscore variant
    "flagged as potentially violating",
    "rejected as a result of the safety system",  # OpenAI safety system
    "rejected as a result of our safety system",  # Azure safety system
    "failed as a result of our safety system",  # Azure: "your task failed as a result..."
    "responsibleaipolicyviolation",  # Azure, when surfaced in the message body
)


def is_content_policy_violation(code: str | None, message: str | None) -> bool:
    """Heuristic: does a 400 (code, message) denote a content-policy rejection?"""
    if code and code.lower() in _CONTENT_POLICY_CODES:
        return True
    if message:
        m = message.lower()
        return any(marker in m for marker in _CONTENT_POLICY_MARKERS)
    return False
