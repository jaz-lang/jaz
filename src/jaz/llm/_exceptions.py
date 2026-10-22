"""The LLM error taxonomy, as the backends import it, and the HTTP-status rule they share.

The classes are defined in ``jaz.exceptions`` (the comment above them there says why); this
module is the in-package import path the backend implementations use.
"""

from __future__ import annotations

import email.utils
import math
import time
from collections.abc import Mapping
from typing import Any

# Absolute rather than the relative form used elsewhere in src/jaz, deliberately: this
# module is a shim onto a *fixed canonical home*, so the import should name that home.
# `..exceptions` resolves relative to this module's own package, so moving this file to
# another depth (e.g. llm/llm/exceptions.py) would silently repoint it at a path
# that doesn't exist; `jaz.exceptions` survives any move. The usual argument for relative
# imports — surviving a rename or vendoring of the `jaz` package itself — is the less
# likely event, and would break this shim's premise anyway.
from jaz.exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    LLMError,
    NotFoundError,
    OverloadedError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    RequestAbortedError,
    ServerError,
    UnsupportedParamsError,
)

__all__ = [
    "LLMError",
    "RateLimitError",
    "QuotaExceededError",
    "AuthenticationError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "NotFoundError",
    "PermissionDeniedError",
    "BadRequestError",
    "UnsupportedParamsError",
    "ServerError",
    "InternalServerError",
    "OverloadedError",
    "APIResponseValidationError",
    "APITimeoutError",
    "APIConnectionError",
    "RequestAbortedError",
]


# Structured error codes that name a cause. A backend that reads one decides by it before the
# status, a transient status included (the user's call, 2026-10-17): the code says what failed,
# while a status can be relayed or reused for other things. The Copilot CLI passes a provider's
# status through unchanged, so its policy error came as a 429 or 500 too (see `_copilot`), and a
# context limit reported with a 503 would otherwise be retried as an overloaded server, though
# the same request fails the same way every time. The rejected alternative was the status rule
# alone, which spends every retry on a request that cannot succeed.
# - `misalignment_policy_violation`: one of OpenAI's Responses error codes
#   (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/types/responses/response_error.py#L47-L53).
# - `context_length_exceeded`: OpenAI's code for an input too long for the model, which LiteLLM
#   maps to a 400 (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/responses/streaming_iterator.py#L214).
# - `model_max_prompt_tokens_exceeded`: the Copilot API's (see `_copilot._CONTEXT_LENGTH_MARKERS`).
# - `bio_policy`: OpenAI's flag for possible biological risk, in the same list. Codex reads it as
#   that ("This content was flagged for possible biological risk",
#   https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/codex-api/src/api_bridge.rs#L290-L291)
#   and gives it no retry delay, as a terminal error
#   (https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/protocol/src/error.rs#L410-L437);
#   a real body says the same (community-sourced: UKGovernmentBEIS/inspect_ai#5626).
# - `content_policy_violation`: OpenAI's image API rejecting a prompt "as a result of our safety
#   system", with type `invalid_request_error`, seen as a 400 (community-sourced:
#   songquanpeng/one-api#737) and relayed as a 401 (BerriAI/litellm#6313), which LiteLLM matches
#   by text only on its non-streaming path
#   (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L341).
# OpenAI's `invalid_prompt` is left out: OpenRouter sends it for a request that fails schema
# validation ("Invalid Responses API request", community-sourced: openai/codex#43740,
# can1357/oh-my-pi#8020), which is no policy block, and as a code that beats the status it would
# make one a `ContentPolicyViolationError`. It already fails fast as a `BadRequestError`.
_POLICY_CODES = frozenset(
    {"misalignment_policy_violation", "bio_policy", "content_policy_violation"}
)
_CONTEXT_LENGTH_CODES = frozenset(
    {"context_length_exceeded", "model_max_prompt_tokens_exceeded"}
)
# - `model_not_found`: OpenAI's code for a model id the project cannot use, whether it does not
#   exist or the project is not granted it. LiteLLM reads the pair `invalid_request_error` +
#   `model_not_found` as its `NotFoundError` at any status
#   (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L325-L331),
#   so a server that sends it with a 400 is covered too (TODO(#1744) for the providers that send
#   no code at all: Bedrock's 400, the Copilot CLI). Either way the fix is the model name, which
#   is what `NotFoundError` means. OpenRouter's `not_found` is an `error_type` in its metadata,
#   not a `code`, and is left to the status.
_NOT_FOUND_CODES = frozenset({"model_not_found"})

# The text `LiteLLM` puts on the `NotFoundError` for a redirect loop, which has no status; the
# console reads it back to point the user at the address rather than the model id.
REDIRECT_LOOP_HINT = "The server kept redirecting; check api_base."


def _class_for_code(code: object) -> type[LLMError] | None:
    """JAZ's class for a structured error code that names a cause, or ``None``."""
    # A provider's `code` can be any JSON value, and a dict or list is not hashable.
    if not isinstance(code, str):
        return None
    if code in _CONTEXT_LENGTH_CODES:
        return ContextWindowExceededError
    if code in _POLICY_CODES:
        return ContentPolicyViolationError
    if code in _NOT_FOUND_CODES:
        return NotFoundError
    return None


def _class_for_status(status: int) -> type[LLMError] | None:
    """JAZ's class for an HTTP error status, or ``None`` outside 3xx, 4xx and 5xx."""
    # The rule the `openai` SDK and LiteLLM's Router both retry by
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L907-L922,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L7165-L7188):
    # 408, 409, 429 and 5xx are transient, any other 4xx is the request's fault, registered in
    # HTTP or not. An unregistered 4xx fails fast as a `BadRequestError` as in every peer checked
    # (the user's call, 2026-10-14, to follow them): the `openai`, `anthropic`, `groq`, Together
    # and Fireworks SDKs give it their generic `APIStatusError` and retry only 408/409/429/5xx
    # (https://github.com/groq/groq-python/blob/v1.7.0/src/groq/_base_client.py#L775-L806);
    # google-genai only 408, 429, 500 and 502-504, and only when asked
    # (https://github.com/googleapis/python-genai/blob/v2.30.1/google/genai/_api_client.py#L571-L600);
    # DSPy turns any 4xx it does not name into its generic `ProviderError`, which it does not
    # retry
    # (https://github.com/stanfordnlp/dspy/blob/3.4.1/dspy/_vendor/lm15/errors.py#L546-L558,
    # https://github.com/stanfordnlp/dspy/blob/3.4.1/dspy/utils/exceptions.py#L263). None
    # special-cases Groq's 498 ("Flex Tier Capacity Exceeded", which Groq documents as worth
    # retrying: https://console.groq.com/docs/errors#client-error-codes), Groq's own SDK
    # included, and Cohere uses 498 for an invalid token
    # (https://github.com/cohere-ai/cohere-python/blob/7.1.0/src/cohere/errors/invalid_token_error.py),
    # so one unregistered code means different things per provider. Neither `SGLangLLM` nor
    # `GitHubCopilotLLM` documents one besides a 499 (SGLang 0.5.17 to 0.5.21 sends only 432,
    # for a corrupt video jaz's text-only `/generate` cannot send,
    # https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/multimodal/processors/mimo_v2.py#L1635;
    # nothing for Copilot, found by a search of its SDK, CLI and issue trackers). A server that
    # wants a retry says so with `x-should-retry`, which jaz honours on `LiteLLM` and `SGLangLLM`
    # as the openai, anthropic and groq SDKs do (`LLMError.is_retryable`); the Copilot runtime
    # passes no response headers, so on `GitHubCopilotLLM` an unregistered 4xx always fails fast.
    # A 499 is the exception (its branch below).
    # A 5xx is the server's whether registered or not (Anthropic's 529, "overloaded";
    # https://platform.claude.com/docs/en/api/errors#http-errors).
    # A 3xx that reaches here is a `NotFoundError` (the user's call, 2026-10-17): it is one the
    # transport did not follow (`SGLangLLM` follows none; LiteLLM follows any 3xx with a
    # `Location` but a 300 or 304, a 301, 302 or 303 as a GET, so it hands back a 300, a 304 or a
    # redirect without a `Location`), so every retry gets the same answer.
    # A class names what the caller must fix (see `NotFoundError`), and a redirect means the
    # address is wrong, as a 404 does. Rejected: the retried `ServerError` it was, which spent
    # every retry on the same redirect, and `BadRequestError`, whose fix is the request's
    # content. A 1xx is `None`.
    # A 402 is Payment Required, which every provider that sends one uses for exhausted credit or
    # billing, so it is a `QuotaExceededError`:
    # - Anthropic `billing_error`: https://platform.claude.com/docs/en/api/errors
    # - Gemini `payment_required`: https://ai.google.dev/gemini-api/docs/api-errors
    # - OpenRouter: https://openrouter.ai/docs/api/reference/errors-and-debugging
    # - DeepSeek "Insufficient Balance": https://api-docs.deepseek.com/quick_start/error_codes
    # - Together, monthly spend limit: https://docs.together.ai/docs/error-codes
    # - Fireworks: https://docs.fireworks.ai/guides/inference-error-codes
    # - Cohere: https://docs.cohere.com/reference/errors
    # - Copilot CLI `quota_exceeded` (a user report):
    #   https://github.com/github/copilot-cli/issues/3431
    if status == 402:
        return QuotaExceededError
    if status == 401:
        return AuthenticationError
    if status == 403:
        return PermissionDeniedError
    if status == 404 or 300 <= status <= 399:
        return NotFoundError
    if status == 429:
        return RateLimitError
    if status == 529:
        return OverloadedError
    if 500 <= status <= 599:
        return InternalServerError
    if status in (408, 409):
        return ServerError
    # A 499 ("client closed request") is the retryable `ServerError` here; `SGLangLLM` alone
    # makes its own a `RequestAbortedError` (the user's call, 2026-10-16, reversing their
    # 2026-10-14 call that it was a `RequestAbortedError` on every backend). A client that reads
    # a response did not close the request, so what a 499 means is whatever its sender means:
    # - GitHub Copilot: transient. Users saw "499 status code (no body)" fail sessions at random,
    #   and the CLI now retries it
    #   (https://github.com/github/copilot-cli/issues/595#issuecomment-3960080131, also #665);
    #   CLI 1.0.97, pointed at a local server answering 499, sent 6 requests in about 15 s
    #   before reporting it. What GitHub's own API sends outside BYOK is not public (#1726).
    # - Cohere: "try the request again" (https://docs.cohere.com/reference/errors).
    # - Gemini: documented as the client's own cancel, "No action needed"
    #   (https://ai.google.dev/gemini-api/docs/api-errors), but users get 499s they did not
    #   cause (googleapis/python-genai#2506), which Google acknowledged without a cause
    #   (https://discuss.ai.google.dev/t/anyone-experiencing-google-api-core-exceptions-cancelled-499-the-operation-was-cancelled/80743).
    # - SGLang (0.5.17 alone sends one): a controller aborted it on purpose, which a retry would
    #   undo (see `SGLangLLM`).
    # The SDKs that fail a 499 fast show nothing about what it means: `openai`'s by the
    # catch-all rule above, Cohere's by the same rule against its own docs
    # (https://github.com/cohere-ai/cohere-python/blob/5.20.1/src/cohere/core/http_client.py#L87-L89),
    # and Google's by gRPC's, where CANCELLED is usually the caller's own cancel, raised by the
    # client library itself
    # (https://github.com/grpc/grpc/blob/39d7a4eaeb6d2a09feedfbc69f9b29a172938eae/doc/statuscodes.md#L45).
    # Rejected: failing it fast everywhere and documenting Gemini as a limitation (Cohere's own
    # advice and Copilot's CLI both retry it); excepting providers one by one (that leaves the
    # fail-fast rule with SGLang alone, which is this rule spelled backwards); and retrying it on
    # SGLang too. A retry is wasted work when an RL framework resumes aborted generation itself,
    # which it does from SGLang's 200 "abort" reply, after the weight update
    # (https://github.com/inclusionAI/AReaL/blob/v2.1.0/areal/infra/remote_inf_engine.py#L954-L1036).
    # It is wrong when the framework cancelled because it had enough samples
    # (https://github.com/THUDM/slime/blob/v0.4.0/slime/rollout/sglang_rollout.py#L426-L447).
    # A 499 has its own branch, so that a change to the rule for other unregistered codes leaves
    # it where it is.
    if status == 499:
        return ServerError
    if 400 <= status <= 499:
        return BadRequestError
    return None


def _response_details(response: Any) -> dict[str, Any]:
    """The :class:`LLMError` keyword arguments a provider's HTTP error ``response`` fills."""
    # Duck-typed on `httpx.Response` (`headers`, `json()`, `text`): the backends that pass one
    # (`LiteLLM`, `SGLangLLM`) use httpx, but this module imports nothing a backend does not
    # already need.
    headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    try:
        body: Any = response.json()
    except Exception:
        # Not JSON, or a streamed body nobody read.
        try:
            body = response.text or None
        except Exception:
            body = None
    details = _header_details(headers)
    if details["retry_after"] is None:
        details["retry_after"] = _google_retry_delay(body)
    return {
        **details,
        "body": body,
        "code": _error_code(body) or details["code"],
    }


def _has_retry_after(response: Any) -> bool:
    """Whether the provider's ``response`` says how long to wait."""
    # A 402 that does is a rate limit rather than an exhausted quota on every backend that reads
    # a response (`_litellm._classified`, `_sglang._classified`): waiting clears it.
    return (
        response is not None and _response_details(response)["retry_after"] is not None
    )


def _header_details(headers: Mapping[str, str]) -> dict[str, Any]:
    """The :class:`LLMError` keyword arguments a provider's response ``headers`` fill alone."""
    headers = {str(k).lower(): str(v) for k, v in headers.items()}
    # Bedrock's error type, as AWS's restJson1 protocol sends it in a header
    # ("ThrottlingException:http://internal.amazon.com/..."), with the namespace some services
    # put before a `#` dropped as that protocol's clients do
    # (https://smithy.io/2.0/aws/protocols/aws-restjson1-protocol.html). Its body here is
    # `{"message": ...}`, though the protocol allows the type there too.
    amazon_code = (
        headers.get("x-amzn-errortype", "").partition(":")[0].rpartition("#")[2] or None
    )
    return {
        # No headers is "not seen", as `LLMError` documents it: `None`, not `{}`.
        "headers": headers or None,
        "code": amazon_code,
        "request_id": next(
            (headers[h] for h in _REQUEST_ID_HEADERS if headers.get(h)), None
        ),
        "retry_after": _retry_after(headers),
        # The `openai`/`anthropic` SDKs' override header: only the two literal values count
        # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L896-L904,
        # https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_base_client.py#L865-L873).
        "is_retryable": {"true": True, "false": False}.get(
            headers.get("x-should-retry", "")
        ),
    }


# OpenAI, Anthropic and Bedrock, in that order
# (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_exceptions.py#L103;
# https://platform.claude.com/docs/en/api/errors#request-id,
# which also documents `x-amzn-requestid`).
_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-amzn-requestid")


def _google_retry_delay(body: Any) -> float | None:
    """The ``retryDelay`` of a ``google.rpc.RetryInfo`` in a Gemini or Vertex error body."""
    # Gemini 429s carry the wait in the body
    # (`"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}]`),
    # as observed in real bodies (huggingface/smolagents#1134, google-gemini/gemini-cli#9248), and
    # Vertex's do too per gemini-cli#9248; no Google doc says whether they also send a
    # `retry-after` header.
    if isinstance(body, list) and len(body) == 1:
        body = body[0]
    error = body.get("error") if isinstance(body, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    for detail in details if isinstance(details, list) else ():
        if not isinstance(detail, dict):
            continue
        if not str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
            continue
        delay = detail.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                seconds = float(delay[:-1])
            except ValueError:
                return None
            return seconds if math.isfinite(seconds) and seconds >= 0 else None
    return None


def _error_code(body: Any) -> str | None:
    """The first string among the error object's ``code``, ``type``, ``status`` and
    ``metadata.error_type``."""
    # Gemini's OpenAI-compatible endpoint wraps its body in a one-element list (community-sourced,
    # e.g. https://github.com/livekit/agents/issues/1896);
    # most providers nest the error object under `error`, and some (Anthropic's
    # `{"type": "error", "error": {...}}`;
    # https://platform.claude.com/docs/en/api/errors#error-shapes)
    # put a `type` beside it, so the nested object is read first.
    if isinstance(body, list) and len(body) == 1:
        body = body[0]
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    error = error if isinstance(error, dict) else body
    # OpenRouter's code is an int, and its typed code is `metadata.error_type`
    # (https://openrouter.ai/docs/api/reference/errors-and-debugging).
    metadata = error.get("metadata")
    fallback = metadata.get("error_type") if isinstance(metadata, dict) else None
    return next(
        (
            v
            for v in (*(error.get(k) for k in ("code", "type", "status")), fallback)
            if isinstance(v, str)
        ),
        None,
    )


def _retry_after(headers: dict[str, str]) -> float | None:
    """Seconds the provider asked to wait, from ``retry-after-ms`` or ``retry-after``."""
    # The `openai`/`anthropic` SDKs' precedence
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L812-L834):
    # the millisecond header first (Azure OpenAI
    # documents it, https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/quota), then
    # `retry-after` as seconds or an HTTP date. A value that is not a finite, non-negative
    # number of seconds is ignored rather than guessed at.
    seconds: float | None = None
    if "retry-after-ms" in headers:
        try:
            seconds = float(headers["retry-after-ms"]) / 1000
        except ValueError:
            seconds = None
    if seconds is None and "retry-after" in headers:
        value = headers["retry-after"]
        try:
            seconds = float(value)
        except ValueError:
            try:
                date = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            # A date already past means "now".
            seconds = max(0.0, date.timestamp() - time.time())
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds
