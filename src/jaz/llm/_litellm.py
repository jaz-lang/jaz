"""The LiteLLM backend — route completions through the LiteLLM library.

:class:`LiteLLM` hands each call to :func:`litellm.completion`, so one backend reaches every
provider LiteLLM routes to and a model released after this JAZ version is usable without a JAZ
release. See the class for usage, the model-id convention, how cost is computed, and the
limitations.
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
import functools
import json
import math
import os
import re
import sys
import warnings

# Module name is ``_litellm`` (private, like ``_agent``), deliberately NOT
# ``litellm.py``: a sibling module named ``litellm`` would shadow the ``litellm`` package under a
# relative import and confuse readers. The public name is the class ``LiteLLM``, imported from
# ``jaz.llm``; nothing imports this module directly.
from collections.abc import Mapping
from typing import Any, NamedTuple

from .._component import repr_call
from ..credentials import resolve_credential
from ..exceptions import UnknownRequestParamWarning
from ._base import BaseLLM, _require_model_id
from ._exceptions import (
    REDIRECT_LOOP_HINT,
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
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
    UnsupportedParamsError,
    _class_for_code,
    _class_for_status,
    _has_retry_after,
    _header_details,
    _response_details,
)
from ._types import LLMResponse, ModelInfo, RetryPolicy

# Retry settings that would otherwise ride the open tail as request params, refused by name. A
# caller who set one would believe retries were configured, so failing fast and naming
# `RetryPolicy` lets them fix it. LiteLLM's own `num_retries` (the user's call, 2026-10-13) and
# `max_retries` do nothing, since this backend forces `num_retries=0`, which overrides both
# (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/main.py#L5458-L5459); the
# `retry_wait_*` keywords, which LiteLLM does not know, would reach the provider's request body
# (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L5057-L5082; observed: found
# by sending them through `openai/` and `hosted_vllm/` to a local server on LiteLLM 1.105.0).
# All but `num_retries` are the pre-`RetryPolicy` constructor keywords (kept on the user's
# objection, 2026-10-13, to an audit proposing to delete them). They share the retired keywords'
# message rather than `_REFUSED_QUERY_PARAMS`' "not supported" because each has a replacement to
# name (implementation choice, not decided by the user).
_RETRY_KEYWORDS = (
    "max_retries",
    "retry_wait_multiplier",
    "retry_wait_min",
    "retry_wait_max",
    "num_retries",
)

# Query params this backend refuses. Each lets LiteLLM answer a call as a model other than the
# backend's own: `fallbacks` re-sends a failed call to other models, `model_list` races the
# deployments listed under the model name (`batch_completion_models`), and
# `context_window_fallback_dict` re-sends a call that overflowed the context window to the mapped
# model (LiteLLM's `client` wrapper). Refused, not supported, as the user decided: `BaseLLM`'s
# contract is that a backend serves one model, fixed for its lifetime (`model_info` is reported
# once), and jaz has no notion of one invoke running on several models. So there is no
# alternative to offer, and the errors only say the param is unsupported. Fallbacks would also
# run a second retry loop under jaz's own. Only `litellm.model_fallbacks` has a process-wide twin
# that `completion` reads, so it alone is checked per call: `litellm.model_list` is LiteLLM's
# registry of model names, and `litellm.context_window_fallbacks` is read only by its Router
# (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/router.py#L1116).
_REFUSED_QUERY_PARAMS = ("fallbacks", "model_list", "context_window_fallback_dict")

# Common cross-provider request parameters — the reference the typo check matches against. NOT an
# allow-list: an unrecognized key is still forwarded (the ``**query_params`` tail is an open
# passthrough for provider-specific extras a JAZ release cannot enumerate). This set exists only so
# a *near-miss* of one of these can be flagged as a likely misspelling. Kept deliberately generous
# so a real, correctly-spelled param never trips the warning.
_KNOWN_REQUEST_PARAMS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "top_a",
        "min_p",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "stop",
        "stream",
        "stream_options",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "verbosity",
        "thinking",
        "service_tier",
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "metadata",
        "store",
        "modalities",
        "prediction",
        "audio",
        "timeout",
        "extra_headers",
        "extra_body",
        "extra_query",
    }
)


# The ``src/jaz`` root (this file is ``src/jaz/llm/_litellm.py``). Used to skip ALL internal jaz frames
# when attributing the misspelled-param warning, so it points at the caller's construction /
# ``jaz.configure`` site regardless of how many backend ``__init__``\\ s forwarded via
# ``super().__init__`` — a fixed ``stacklevel`` can't, since real backends add a forwarding frame
# and every one would then share a single internal lineno (collapsing distinct call sites under
# Python's once-per-location dedup).
_JAZ_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _warn_on_misspelled_request_params(
    request_params: Mapping[str, Any], backend: str
) -> None:
    """Warn once per kwarg that looks like a misspelled request parameter.

    A key is flagged only when it is a *close* match to a known param (``difflib`` ratio ≥ 0.8)
    and is not itself known — the "looks like a typo of `temperature`" case. Unknown keys with no
    close match are assumed to be intentional provider extras and pass silently. The kwarg is
    forwarded either way; this only surfaces the likely mistake.
    """
    known = _KNOWN_REQUEST_PARAMS
    for key in request_params:
        if key in known:
            continue
        match = difflib.get_close_matches(key, known, n=1, cutoff=0.8)
        if match:
            warnings.warn(
                f"{backend} received request parameter {key!r}, which is not a recognized "
                f"parameter — did you mean {match[0]!r}? It will be forwarded to the provider "
                f"as-is (which typically ignores unknown fields).",
                UnknownRequestParamWarning,
                # Skip jaz's own frames so the warning lands on the caller's construction site,
                # not on a backend's internal ``super().__init__`` (see the note on _JAZ_PKG_ROOT).
                skip_file_prefixes=(_JAZ_PKG_ROOT,),
            )


def _reserved_model_keys() -> frozenset[str]:
    """LiteLLM's pseudo-entry ids — map keys that document the schema, not models.

    Read from LiteLLM rather than hardcoded: it declares two, and JAZ pins `litellm>=1.105.0`
    with no ceiling. Imported inside the function, not at module scope, because importing
    `litellm` eagerly is exactly what `test_import_jaz_does_not_import_litellm` forbids.
    """
    # The two declared keys:
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/get_model_cost_map.py#L52
    try:
        from litellm.litellm_core_utils.get_model_cost_map import (
            RESERVED_TOP_LEVEL_KEYS,
        )

        return frozenset(RESERVED_TOP_LEVEL_KEYS)
    except ImportError:
        # The one reserved key actually reachable through `_model_info` today.
        return frozenset({"sample_spec"})


def _litellm() -> Any:
    """Import ``litellm`` on demand and return the module.

    Kept out of module scope so importing this file — which happens on ``import jaz``, since
    ``jaz.llm`` imports this module eagerly — does not pay LiteLLM's ~2 s import + heavy
    dependency tree. The cost is deferred to the first call/pricing check instead.
    """
    import litellm

    # Cosmetic: silence LiteLLM's per-call "Give Feedback / Provider List" banner on stderr.
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2373-L2381
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/get_llm_provider_logic.py#L533-L537
    litellm.suppress_debug_info = True
    return litellm


#: Appended to every mapped :class:`AuthenticationError`.
#
# `~/.jaz/credentials.json` has no public SDK writer — the supported way to fill it is the console
# `set_credential(...)`, so an auth failure has to name that spelling or the store is unreachable
# from the message. Provider-agnostic (`<provider>`, not a hardcoded vendor) because this one
# backend serves every vendor.
_CREDENTIAL_HINT = "\n\nTo store a key for this provider, run set_credential('<provider>') in the jaz console."

# The class name LiteLLM puts in front of every message ("litellm.RateLimitError: "). It can be
# nested and repeated: "litellm.InternalServerError: InternalServerError: OpenAIException - ..."
# or, for its `BadRequestError` subclasses, "litellm.ContextWindowExceededError:
# litellm.BadRequestError: ContextWindowExceededError: OpenAIException - ...".
# https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L471, #L557, #L773
_LITELLM_CLASS_PREFIX = re.compile(r"litellm\.(\w+): ")
# What a handler's missing-key error says: "API key", "api_key", "FIREWORKS_API_KEY" (see
# `_mapped_by_class`).
_NAMES_API_KEY = re.compile(r"api[ _]?key", re.IGNORECASE)
# LiteLLM's provider tag, sometimes with its class name inside ("AzureException APIError - ").
# https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2122
_PROVIDER_TAG = re.compile(r"(\w+Exception)(?: \w*Error)? - ")
# LiteLLM's advice appended to an unmapped status ("... Handle with `litellm.InternalServerError`.").
# https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L612
_HANDLE_WITH = re.compile(r"\s*Handle with `litellm\.\w+`\.?$")


def _without_litellm_prefix(message: str) -> str:
    """``message`` without the LiteLLM class names around the provider's text."""
    # Stripped because the class it names is LiteLLM's, not the one the caller catches (a
    # `litellm.BadRequestError` can surface as a `PermissionDeniedError`); the original stays
    # chained as `__cause__`. The provider tag ("OpenAIException - ") is kept, without a class
    # name LiteLLM put inside it: it says which provider answered. A LiteLLM proxy reached on an
    # `openai/` route relays the upstream's message after its own tag, so the names are
    # stripped again after each leading tag. Through `litellm_proxy/` or `hosted_vllm/` the
    # relayed message sits inside a JSON body instead, and its names stay (LiteLLM 1.97.0 and
    # 1.105.0): rewriting inside the body would alter the provider's answer. A bare `<Name>: ` is dropped only when it repeats a name just stripped, so a wrapped
    # Python error ("ValueError: bad") keeps its class.
    done = ""
    while True:
        names = set()
        while match := _LITELLM_CLASS_PREFIX.match(message):
            names.add(match.group(1))
            message = message[match.end() :]
        head, sep, rest = message.partition(": ")
        if sep and head in names:
            message = rest
        tag = _PROVIDER_TAG.match(message)
        if tag is None:
            break
        done += f"{tag.group(1)} - "
        message = message[tag.end() :]
    return done + _HANDLE_WITH.sub("", message)


class _Chain(NamedTuple):
    """What the exception chain under LiteLLM's error says, read in one walk (`_chain`)."""

    # LiteLLM's error.
    exc: BaseException
    # `exc`, then each `__cause__`/`__context__` below it, up to the exception the caller was
    # handling when it made the call, the chain's end, or a link seen before.
    links: tuple[BaseException, ...]
    # The provider handler's error LiteLLM translated, or `None` (see `_chain`).
    provider_error: Any
    # The provider's HTTP response, or `None` when none came back.
    response: Any
    # The first link under LiteLLM's and the SDK's wrappers, which is what failed (on a chain
    # that cycles through wrappers alone, the repeated wrapper), and whether a wrapper above it
    # said timeout (see `_transport_failure`).
    failed: BaseException | None
    said_timeout: bool


def _classes(module: Any, *names: str) -> tuple[type, ...]:
    """The classes ``module`` defines under ``names``, leaving out any it does not."""
    # Every LiteLLM class the mapping reads is looked up through this. JAZ pins `litellm` with no
    # ceiling, and LiteLLM adds and drops exception classes between versions, so one name looked
    # up as a plain attribute would raise `AttributeError` from the mapping, in place of every
    # error it was mapping, a plain 429 included. A class missing here matches nothing, and the
    # status rule decides (`_mapped_by_class`). LiteLLM 1.105.0 has `SensitiveDataRouteException`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L1236), which
    # 1.87.0's `exceptions.py` lacks, and 1.87.0 has `GuardrailInterventionNormalStringError`
    # (https://github.com/BerriAI/litellm/blob/v1.87.0/litellm/exceptions.py#L1053), which
    # 1.105.0's lacks. DSPy looks up LiteLLM's classes the same way
    # (https://github.com/stanfordnlp/dspy/blob/3.4.1/dspy/clients/engines/litellm_errors.py#L46-L71).
    return tuple(
        found
        for name in names
        if isinstance(found := getattr(module, name, None), type)
    )


def _base_llm_exception() -> tuple[type, ...]:
    """LiteLLM's ``BaseLLMException``, or nothing if it has moved."""
    # A private module path, so it is optional in the same way as `_classes`.
    try:
        from litellm.llms.base_llm.chat.transformation import BaseLLMException
    except ImportError:
        return ()
    return (BaseLLMException,)


def _chain(
    litellm: Any, exc: BaseException, handling: BaseException | None = None
) -> _Chain:
    """Walk the chain under ``exc`` once and read off what the mapping needs."""
    import httpx

    # `handling` is the exception the caller was handling when it made the call: LiteLLM's
    # error chains onto it when nothing of its own lies below, and it says nothing about this
    # call, so the walk ends there as at the chain's end. A link seen before ends it too.
    links: list[BaseException] = []
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and link is not handling and id(link) not in seen:
        seen.add(id(link))
        links.append(link)
        link = link.__cause__ or link.__context__
    # The link that ended the walk on a cycle, if one did.
    repeated = link if link is not None and link is not handling else None

    le = litellm.exceptions
    oai = getattr(le, "openai", None)
    base_llm_exception = _base_llm_exception()
    # LiteLLM raises its exception while handling its provider handler's error (`OpenAIError`,
    # `AnthropicError`, `BedrockError`, ...), so that error is `exc`'s immediate
    # `__cause__`/`__context__`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/main.py#L6076-L6078).
    # Only that one link, and only when it is a provider handler's error (they share
    # `BaseLLMException`,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/base_llm/chat/transformation.py#L41):
    # it is the one LiteLLM translated. One further down (under a bug of LiteLLM's, say) was not
    # seen on the routes tried (inferred: absent from #1691's Sagemaker chains and its 11-route
    # sweep against a local server on LiteLLM 1.105.0, which compared statuses, not chain depth),
    # and what it would tell, the provider's status, comes from the HTTP
    # response, which is read at any depth (below). An implementation choice, not decided by the
    # user: the reasons the response can be trusted at depth would cover it too.
    # The `openai` SDK's own status error counts too: LiteLLM's async Azure handler re-raises
    # it as is rather than wrapping it in a provider handler's error
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/azure/azure.py#L495-L496).
    # Since a caller can be handling one of those itself, `handling` is never read as this
    # call's.
    below = exc.__cause__ or exc.__context__
    provider_error = (
        below
        if below is not handling
        and isinstance(below, base_llm_exception + _classes(oai, "APIStatusError"))
        else None
    )

    # Every LiteLLM route keeps the provider's HTTP response at the bottom of the chain, as the
    # `httpx.HTTPStatusError` its handler raised (LiteLLM's `MaskedHTTPStatusError` is one) or
    # under the `openai` SDK's error; LiteLLM's own exception can carry a response it built, with
    # its class's fixed status and a placeholder request, keeping at most the provider's headers
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L779-L787).
    # Read at any depth, unlike the provider error above (an implementation choice, not decided by
    # the user): Sagemaker's handler leaves the real response under a bug of its own (see
    # `_provider_status`), and an earlier request's response could sit in this chain only if
    # LiteLLM sent a second one while handling it, which its retries and fallbacks would do, and
    # this backend forces `num_retries=0` and refuses fallbacks (`_REFUSED_QUERY_PARAMS`). The
    # caller's own error is `handling`, where the walk stops.
    response = next(
        (link.response for link in links if isinstance(link, httpx.HTTPStatusError)),
        None,
    )

    # Any exception class LiteLLM or the `openai` SDK defines is a wrapper (some provider
    # handlers, such as `OpenAILikeError` for watsonx and Vertex partner models, share no base
    # with the rest:
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/openai_like/common_utils.py#L6).
    wrappers = _classes(oai, "APIError") + base_llm_exception

    def is_wrapper(error: BaseException) -> bool:
        if isinstance(error, wrappers):
            return True
        return type(error).__module__.partition(".")[0] in ("litellm", "openai")

    timeouts = _classes(le, "Timeout") + _classes(oai, "APITimeoutError")
    failed: BaseException | None = repeated
    said_timeout = False
    for link in links:
        if not is_wrapper(link):
            failed = link
            break
        said_timeout = said_timeout or isinstance(link, timeouts)
    return _Chain(exc, tuple(links), provider_error, response, failed, said_timeout)


def _provider_status(chain: _Chain) -> int | None:
    """The HTTP status the provider returned: the response's, else the provider error's, else
    LiteLLM's."""
    # The response is what the provider sent; the two below it can both be made up. LiteLLM's
    # mapper rewrites its own `status_code` (a Vertex 429 whose text contains "403" comes out as
    # 403:
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L1199-L1210),
    # and a provider handler that fails while handling the provider's error reports a status of
    # its own: Sagemaker's answers a 403 with a 500, from the `AttributeError` it raises reading
    # the httpx error on the sync path and outright on the async one
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/sagemaker/completion/handler.py#L291-L297,
    # #L516-L520). The provider error's status still decides when no response is in the chain.
    if chain.response is not None:
        return chain.response.status_code
    if chain.provider_error is not None:
        status = getattr(chain.provider_error, "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
    status = getattr(chain.exc, "status_code", None)
    return status if isinstance(status, int) else None


def _transport_failure(chain: _Chain) -> type[LLMError] | None:
    """:class:`APITimeoutError`, :class:`APIConnectionError` or, for a redirect loop,
    :class:`NotFoundError` if the chain is a transport failure."""
    # LiteLLM's class does not say: a refused connection arrives as `InternalServerError` from most
    # providers and as `APIConnectionError` from Ollama and Gemini, a timeout as `Timeout`, and
    # `APIConnectionError` is also LiteLLM's catch-all for a bug in its own code (a `KeyError`
    # from a malformed Anthropic `tools` entry, no request sent)
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2670-L2709).
    # So the chain decides: the first link that is not a wrapper (`_chain`) is what failed. An
    # httpx timeout there is an `APITimeoutError`, and any other failure httpx reports sending
    # the request or reading the response (`_transport_class`) an `APIConnectionError`.
    # A LiteLLM or SDK timeout class among the wrappers is an `APITimeoutError` on its own word
    # only when nothing lies below it: LiteLLM's own `Timeout` class also names a real HTTP 408
    # or 504 that way (an `httpx.HTTPStatusError` below), which the status rule decides, with
    # its status.
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2286, #L2325
    # Anything else is `None`, and the error stays a plain `ServerError`. Stopping at the first
    # non-wrapper, or at the exception the caller was handling when it made the call, keeps a
    # builtin raised inside LiteLLM or the caller's own exception from deciding.
    if chain.failed is None and chain.said_timeout:
        return APITimeoutError
    return _transport_class(chain.failed)


def _transport_class(error: BaseException | None) -> type[LLMError] | None:
    """The jaz class for ``error`` if it is an httpx timeout, redirect loop or connection
    failure."""
    # The rule `SGLangLLM` uses (`_sglang._classified`): every `httpx.RequestError` but a
    # timeout is the connection failing, as the `openai` SDK classifies whatever its client's
    # `send` raises
    # (https://github.com/openai/openai-python/blob/v2.20.0/src/openai/_base_client.py#L1024-L1037).
    # That covers refused, reset, TLS handshake, closed mid-response and an unreachable proxy,
    # and also a body httpx cannot decode (`DecodingError`, a gzip header over plain bytes),
    # LiteLLM's async transport's bare `httpx.RequestError` for a malformed status line
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/custom_httpx/aiohttp_transport.py#L43-L67),
    # which the sync transport reports as `httpx.RemoteProtocolError` (each seen beneath
    # LiteLLM's wrappers on four routes, LiteLLM 1.105.0).
    # The exceptions are the setup failures that pass through (`_local_failures`), and a redirect
    # loop (`TooManyRedirects`), a `NotFoundError` as a 3xx is (`_class_for_status`, the user's
    # call, 2026-10-17): LiteLLM follows redirects, its sync client with `follow_redirects=True`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/custom_httpx/http_handler.py#L1413)
    # and its async one by default (#L618 and #L697 at the same tag), and a server that redirects
    # to itself does so on every attempt (21 requests per attempt, sync and async, against a local
    # server). The `openai` SDK files it as a connection error with the rest of `send`'s failures;
    # jaz assigns by what the caller must fix, and a loop means the address is wrong.
    import httpx

    if isinstance(error, httpx.TimeoutException):
        return APITimeoutError
    if isinstance(error, httpx.TooManyRedirects):
        return NotFoundError
    if isinstance(error, httpx.RequestError) and not _is_local_failure(error):
        return APIConnectionError
    return None


# What marks an exhausted quota, credit or spend limit. The rule: an error is a
# `QuotaExceededError` if the provider says so in a structured field (the error's `code`/`type`, a
# quota id), or, where a provider documents only prose, in the narrow wording its error message
# begins with. Read whatever class LiteLLM chose and whatever the status, since providers send
# these as 400, 403, 412 and 429 alike; a 402 is one by the shared status rule
# (`_class_for_status`). Provider-agnostic: a marker is matched on
# every provider, so an OpenAI-compatible endpoint sending one is recognized. Through a LiteLLM
# proxy, only what the proxy's message still carries is: the status, and the upstream's own body
# where LiteLLM kept it (Anthropic, Gemini); a code or type that LiteLLM read off an OpenAI-SDK
# upstream is gone before the proxy relays it.
#
# A per-day limit counts only when it resets on a calendar day, as Gemini's does (detected by its
# quota id). Groq's and OpenAI's per-day limits stay `RateLimitError`: their 429 names a wait
# ("Please try again in …"), which we read as a rolling window that frees up like any rate limit
# (inferred; neither provider's rate-limit guide states the window, though
# https://console.groq.com/docs/rate-limits shows a per-day reset header of minutes, which fits a
# rolling window). Bedrock's ("Too many tokens per day", a `ThrottlingException`) is left out until
# its body through LiteLLM can be checked: TODO(#1638).
#
# What stays out, deliberately: Gemini's 429 message ("You exceeded your current quota, please check
# your plan and billing details") is OpenAI's prose but is sent for per-minute limits too, so only
# the quota id decides there (community-sourced:
# https://github.com/huggingface/smolagents/issues/1134); and Gemini's FAILED_PRECONDITION 400 also
# covers the free tier being unavailable in the caller's country, which enabling billing fixes
# rather than topping up a credit (per the 2025 version of
# https://ai.google.dev/gemini-api/docs/troubleshooting; the current page no longer lists it).

# Values of the error object's `code` or `type`, or of Anthropic's `details.error_code`.
_QUOTA_CODES = frozenset(
    {
        # OpenAI (429): `code` names the cause, and `type` can still be `insufficient_quota`.
        # https://developers.openai.com/api/docs/guides/error-codes
        # Azure OpenAI sends it too when the model quota for the subscription's region is used up
        # (community-sourced: a volunteer moderator's answer,
        # https://learn.microsoft.com/en-in/answers/questions/5759907/get-this-429-insufficient-quota-error-for-azure-op);
        # the fix given there is raising the allocation, which needs someone to act as a billing
        # quota does, so it fails fast too.
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        # OpenAI (400, `invalid_request_error`), "Billing hard limit has been reached"
        # (community-sourced; not in OpenAI's docs):
        # https://community.openai.com/t/when-create-batch-job-we-got-error-code-400-error-message-billing-hard-limit-has-been-reached/811204
        "billing_hard_limit_reached",
        # OpenAI (429), "Your account is not active, please check your billing details"
        # (community-sourced; not in OpenAI's docs):
        # https://community.openai.com/t/error-failed-to-send-message-http-429-error-message-your-account-is-not-active-please-check-your-billing-details-on-our-website-type-billing-not-active-param-null-code-null/244898
        "billing_not_active",
        # Anthropic (429, `rate_limit_error`): the usage tier's monthly spend cap, which has no
        # `retry-after`. https://platform.claude.com/docs/en/api/rate-limits#reaching-your-spend-cap
        "enforced_spend_limit_reached",
        # Gemini's Interactions API string code (429): "You have exceeded your daily quota."
        # https://ai.google.dev/gemini-api/docs/api-errors scopes these codes to that API;
        # `generateContent`, which LiteLLM calls, sends `RESOURCE_EXHAUSTED` and is matched by
        # quota id below. Kept for an Interactions-API body relayed through a proxy.
        "quota_exceeded",
        # Moonshot / Kimi (429), insufficient balance or token quota:
        # https://platform.kimi.ai/docs/api/errors.md. Its calendar-day TPD limit shares
        # `rate_limit_reached_error` with its per-minute limits, so it stays a `RateLimitError`
        # (TODO(#1753): a quota by the rule beside `RateLimitError`, once its wording is known).
        "exceeded_current_quota_error",
        # A LiteLLM proxy's key, user or team budget: `ProxyErrorTypes.budget_exceeded`
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/proxy/_types.py#L4411).
        # The status has moved between versions (400, then 429 in 1.97.0,
        # https://github.com/BerriAI/litellm/blob/v1.97.0/litellm/proxy/auth/auth_exception_handler.py#L149-L151,
        # 422 by default from 1.105.0's `litellm.budget_exceeded_status_code`,
        # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/__init__.py#L409),
        # so only the code decides.
        "budget_exceeded",
        # xAI (403), a personal team out of credits (community-sourced: a real body in
        # https://github.com/cvolkernick/personal-workspace/issues/1077; not in xAI's docs)
        "personal-team-blocked:spending-limit",
    }
)

# Matched at the start of the error object's message, never anywhere in the raw response: a
# request-validation 400 (vLLM's, for one) echoes the offending input inside its message, so a
# substring match would read a prompt quoting one of these as a quota error.
# https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/api_server.py#L1540-L1555
_QUOTA_MESSAGES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Anthropic (400, `invalid_request_error`), whose messages begin with these and carry no
        # structured field: prepaid credit at zero (community-sourced, a real body in
        # https://github.com/BerriAI/litellm/issues/24320; Anthropic documents the 402
        # `billing_error`, which the status rule covers), and an organization or workspace
        # spend limit the user set
        # (https://platform.claude.com/docs/en/api/rate-limits#setting-your-own-spend-limit).
        r"your credit balance is too low",
        r"you have reached your specified api usage limits",
        r"you have reached your specified workspace api usage limits",
        # xAI (429), a team out of credits or at its monthly spending limit; the message is the
        # body's `error` string (community-sourced: a real 429 in
        # https://github.com/continuedev/continue/issues/10373). A 403 with this message is
        # matched too (community-sourced: a paraphrased 403 `permission-denied` body in
        # https://github.com/cvolkernick/personal-workspace/issues/1077).
        r"your team \S+ has either used all available credits or reached its monthly spending limit",
        # OpenRouter (403), a key's own spend cap or a budget, for bodies without
        # `metadata.limit_source` (real messages in the title of
        # https://github.com/RooCodeInc/Roo-Code/issues/515 and in
        # https://github.com/NousResearch/hermes-agent/issues/107166); a body with it is
        # decided by `_OPENROUTER_QUOTA_SOURCES`.
        r"key limit exceeded\b",
        r"budget limit exceeded\b",
        # Fireworks (412 `PRECONDITION_FAILED`), an account suspended at its monthly spending
        # limit or for unpaid invoices ("412 Precondition Failed | Account is suspended",
        # https://docs.fireworks.ai/guides/inference-error-codes; a real body in
        # https://github.com/mozilla-ai/any-llm/issues/1168)
        r"account \S+ is suspended",
    )
)

# Gemini / Vertex (429 RESOURCE_EXHAUSTED): a `google.rpc.QuotaFailure` violation whose
# `quotaId` is a per-day quota (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), which
# resets at midnight Pacific (https://ai.google.dev/gemini-api/docs/rate-limits). A per-minute
# one (`...PerMinute...`) stays a `RateLimitError`. Google's own gemini-cli treats a quota id
# containing `PerDay` or `Daily` as terminal
# (https://github.com/google-gemini/gemini-cli/blob/v0.63.0/packages/core/src/utils/googleQuotaErrors.ts#L330);
# the body shape is community-sourced (a per-minute body in
# https://github.com/huggingface/smolagents/issues/1134, a per-day one in
# https://github.com/google-gemini/gemini-cli/issues/9248).
_PER_DAY_QUOTA_IDS = ("PerDay", "Daily")

# OpenRouter's `error.metadata.limit_source` on a budget 403
# (https://openrouter.ai/docs/api/reference/limits, "Handling 403 budget errors"), whatever its
# message says. Its 402 sources need no entry, since a 402 is a quota by status already, or a
# `RateLimitError` when it carries a `Retry-After`, as only its transient
# `openrouter_in_flight_budget` does (see `_classified`).
_OPENROUTER_QUOTA_SOURCES = frozenset(
    {
        "openrouter_key_limit",
        "openrouter_workspace_budget",
        "openrouter_guardrail_budget",
    }
)


def _error_object(chain: _Chain) -> dict[str, Any]:
    """The provider's ``error`` object: from its HTTP response's body, else from the provider
    error LiteLLM translated."""
    # The body is what the provider sent, and is read the same way on every route; the provider
    # error is LiteLLM's copy of it, in a shape each handler chooses, and is read only when no
    # response came back (an implementation choice, not decided by the user).
    if chain.response is not None:
        return _error_in(_response_details(chain.response)["body"])
    link = chain.provider_error
    if link is None:
        return {}
    # Two shapes: a handler built on the `openai` SDK keeps the body's `error` value as `body`
    # (https://github.com/openai/openai-python/blob/v2.20.0/src/openai/_client.py#L416),
    # while the others keep the raw response text as `message`. xAI's `error` is a bare string
    # (community-sourced: a real 403 body in
    # https://github.com/cvolkernick/personal-workspace/issues/1077; not in xAI's docs), which
    # the SDK then keeps as `body` on its own.
    body = getattr(link, "body", None)
    if isinstance(body, str):
        return {"error": body}
    if not isinstance(body, dict | list):
        try:
            body = json.loads(getattr(link, "message", None) or "")
        except (TypeError, ValueError):
            return {}
    return _error_in(body)


def _error_in(body: Any) -> dict[str, Any]:
    """The ``error`` object in a provider's parsed error ``body``, or ``{}``."""
    # Gemini's OpenAI-compatible endpoint wraps the body in a one-element list (community-sourced,
    # e.g. https://github.com/livekit/agents/issues/1896; no Google doc shows it, and Google's
    # error model, https://google.aip.dev/193, is a bare object). A body whose `error` is not an
    # object (xAI's is a string) is itself the error object.
    if isinstance(body, list) and len(body) == 1:
        body = body[0]
    if not isinstance(body, dict):
        return {}
    error = body.get("error")
    return error if isinstance(error, dict) else body


def _is_quota_exceeded(chain: _Chain) -> bool:
    """Whether the provider's error says the quota, credit or spend limit is exhausted."""
    # A 402 is one by the shared status rule, but that rule overrules a specific class only when
    # the response is in the chain (`_mapped_by_class`); a 402 LiteLLM gives a specific class
    # without one (NLP Cloud's comes out as its `RateLimitError`) is caught here.
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L1684-L1689
    if _provider_status(chain) == 402:
        return True
    return _error_says_quota(_error_object(chain), relayed=True)


def _is_openrouter_content_block(chain: _Chain) -> bool:
    """Whether the provider's 403 is OpenRouter's moderation flag or a model's refusal, rather
    than a permission."""
    # OpenRouter documents its 403 as "insufficient permissions, guardrail block, or moderation
    # flag" (https://openrouter.ai/docs/api/reference/errors-and-debugging). A moderation flag
    # carries `ModerationErrorMetadata` (`reasons`, `flagged_input`, ...), and its typed
    # `error_type` is `content_policy_violation` where the response has one. A model refusal the
    # provider reported as an error is a 403 with `error_type` `refusal`. A guardrail block is
    # `permission_denied`, the same `error_type` as a key's missing permission ("the key is valid
    # but lacks the required permission OR the request was blocked by a guardrail"), and
    # OpenRouter's guardrails include content filters and prompt-injection detection
    # (https://openrouter.ai/docs/guides/features/guardrails), so a content block run as a
    # guardrail arrives as a `PermissionDeniedError`; the body does not say which it was, so it
    # is left there and documented on the class. Not gated on the route: these
    # fields are OpenRouter's, and no other provider's 403 is known to carry them.
    # A LiteLLM proxy relaying one is read the same way, from the body it relays.
    error = _error_object(chain)
    for candidate in (error, _relayed_error(error)):
        metadata = candidate.get("metadata") if candidate else None
        if isinstance(metadata, dict) and (
            metadata.get("error_type") in ("content_policy_violation", "refusal")
            or "flagged_input" in metadata
        ):
            return True
    return False


# Google's `ErrorInfo` reasons for a missing, invalid or expired credential, as opposed to one
# that is valid but lacks access (`ACCESS_TOKEN_SCOPE_INSUFFICIENT`, `API_KEY_SERVICE_BLOCKED`, ...)
# or an account problem (`ACCOUNT_STATE_INVALID`):
# https://github.com/googleapis/googleapis/blob/94bc991b3f1df58bc841801631715894a7ac5e28/google/api/error_reason.proto#L86,
# #L340, #L396, #L413, #L720, #L735.
_GOOGLE_CREDENTIAL_REASONS = frozenset(
    {
        "API_KEY_INVALID",
        "ACCESS_TOKEN_EXPIRED",
        "ACCESS_TOKEN_TYPE_UNSUPPORTED",
        "CREDENTIALS_MISSING",
        "JWT_TOKEN_INVALID",
        "CREDENTIAL_TYPE_UNSUPPORTED",
    }
)

# AWS's error types for credentials it does not recognise, all sent with a 403, as opposed to
# `AccessDeniedException`, a valid credential without permission:
# https://docs.aws.amazon.com/bedrock/latest/APIReference/CommonErrors.html
_AWS_CREDENTIAL_ERRORS = frozenset(
    {"UnrecognizedClientException", "ExpiredTokenException", "IncompleteSignature"}
)

# AWS's error type for an account that has not submitted Anthropic's use-case form, sent with a
# 404 ("Model use case details have not been submitted for this account"):
# https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html#ts-ftu-form.
# The status would make it a `NotFoundError`, whose fix is the model name or the URL; the fix is
# the account's standing, which is `PermissionDeniedError`'s.
_AWS_ACCOUNT_STANDING_ERRORS = frozenset({"FTUFormNotFilled"})


def _is_bad_credential(chain: _Chain) -> bool:
    """Whether the provider's structured error says the credential itself is bad."""
    # A bad credential is a 401 on 10 of the 14 providers probed with a made-up one, and the 401
    # names `AuthenticationError` by status. Two others send another status and say why in a
    # field of their own rather than in the message: Gemini a 400 with `ErrorInfo` `reason:
    # API_KEY_INVALID`, and Bedrock a 403 with `x-amzn-errortype: UnrecognizedClientException:...`.
    # LiteLLM finds both by matching the message text, which the status rule overrules
    # (`_mapped_by_class`); these fields, unlike the text, cannot be a prompt echoed back.
    # Providers whose bad-key answer has no such field are not covered: TODO(#1646) xAI's 400 says
    # so only in its message, and TODO(#1736) Fireworks' 404 reads the same as an unknown model.
    # TODO(#1738): Bedrock most likely answers a real key ID with a wrong secret with
    # `InvalidSignatureException`, which AWS also sends for clock skew, so it is not in the set.
    response = chain.response
    if response is None:
        return False
    if _header_details(response.headers)["code"] in _AWS_CREDENTIAL_ERRORS:
        return True
    details = _error_object(chain).get("details")
    return any(
        isinstance(detail, dict)
        and detail.get("domain") == "googleapis.com"
        and detail.get("reason") in _GOOGLE_CREDENTIAL_REASONS
        for detail in (details if isinstance(details, list) else ())
    )


# Where a LiteLLM proxy's message starts the upstream body it relays.
_RELAYED_BODY = re.compile(r"Exception - \[?(?=\{)")


def _error_message(error: dict[str, Any]) -> str:
    """A provider ``error`` object's message."""
    # xAI puts its message in a string `error` field rather than `message`
    # (community-sourced: https://github.com/continuedev/continue/issues/10373).
    return next(
        (m for m in (error.get("message"), error.get("error")) if isinstance(m, str)),
        "",
    )


def _relayed_error(error: dict[str, Any]) -> dict[str, Any] | None:
    """The upstream ``error`` object a LiteLLM proxy relays inside its message, if any."""
    # A LiteLLM proxy relays an upstream error inside its own message
    # ("litellm.BadRequestError: AnthropicException - {...}"), so the upstream's fields are no
    # longer the object's own; the body right after LiteLLM's "<Provider>Exception - " (a list
    # for Gemini) is decoded, once. Only there: JSON elsewhere in a message can be the caller's
    # own input echoed back.
    message = _error_message(error)
    seam = _RELAYED_BODY.search(message)
    if seam is None:
        return None
    try:
        nested, _ = json.JSONDecoder().raw_decode(message, seam.end())
    except ValueError:
        return None
    if not isinstance(nested, dict):
        return None
    inner = nested.get("error")
    return inner if isinstance(inner, dict) else nested


def _error_says_quota(error: dict[str, Any], *, relayed: bool) -> bool:
    """Whether a provider ``error`` object carries a quota marker."""
    message = _error_message(error)
    if any(pattern.match(message) for pattern in _QUOTA_MESSAGES):
        return True
    # The relayed body is checked by the same rules, so a proxy's message no longer hides the
    # upstream's own marker.
    nested = _relayed_error(error) if relayed else None
    if nested is not None and _error_says_quota(nested, relayed=False):
        return True
    details = error.get("details")
    codes = [error.get("code"), error.get("type")]
    if isinstance(details, dict):
        codes.append(details.get("error_code"))
    if any(isinstance(code, str) and code in _QUOTA_CODES for code in codes):
        return True
    violations = [
        violation
        for detail in (details if isinstance(details, list) else ())
        if isinstance(detail, dict)
        for violation in detail.get("violations") or ()
        if isinstance(violation, dict)
    ]
    if any(
        marker in str(violation.get("quotaId", ""))
        for violation in violations
        for marker in _PER_DAY_QUOTA_IDS
    ):
        return True
    metadata = error.get("metadata")
    return (
        isinstance(metadata, dict)
        and metadata.get("limit_source") in _OPENROUTER_QUOTA_SOURCES
    )


def _mapped_exception(
    litellm: Any, exc: BaseException, handling: BaseException | None = None
) -> Exception | None:
    """Translate a LiteLLM exception into JAZ's error taxonomy, or ``None`` to re-raise as-is."""
    chain = _chain(litellm, exc, handling)
    mapped = _classified(litellm, chain)
    if (
        isinstance(mapped, LLMError)
        and mapped.status_code is not None
        and chain.response is not None
    ):
        for name, value in _response_details(chain.response).items():
            setattr(mapped, name, value)
    return mapped


def _decoded_a_response(error: BaseException) -> bool:
    """Whether ``error`` is a ``JSONDecodeError`` raised decoding an HTTP response's body."""
    # LiteLLM also decodes the caller's own input while it builds a request, before anything is
    # sent: a tool call's `arguments` on `anthropic/`, `gemini/` and `ollama_chat/` (found by
    # sending `"{not json"` through each, LiteLLM 1.105.0). That fails the same way on every
    # attempt, and the provider never saw it, so it must not read as a garbled response. Those
    # decode with `json.loads` directly; every garbled response on the 22 routes probed was
    # decoded by `httpx.Response.json`, which is what tells them apart here (an implementation
    # choice, not decided by the user). Neither the chain's shape nor LiteLLM's class can: a
    # tool call's decode on `gemini/` chains as `APIConnectionError<Exception<JSONDecodeError`,
    # as a garbled 200 does on `openai/`. A route that decodes a response some other way is not
    # matched, and keeps the class LiteLLM's error maps to.
    import httpx

    if not isinstance(error, json.JSONDecodeError):
        return False
    decode = httpx.Response.json.__code__
    frame = error.__traceback__
    while frame is not None:
        if frame.tb_frame.f_code is decode:
            return True
        frame = frame.tb_next
    return False


def _classified(litellm: Any, chain: _Chain) -> Exception | None:
    """The jaz error for LiteLLM's error by class, status and the provider's markers."""
    mapped = _mapped_by_class(litellm, chain)
    # A body that is not JSON, with no error status, is a garbled response, raised as the other
    # backends raise one whatever class LiteLLM chose: its handlers call it a 422 on Gemini,
    # Vertex and Bedrock, which would fail fast, and a 500 or a connection error elsewhere. All
    # 22 routes tried (sync and async, LiteLLM 1.105.0) kept the `JSONDecodeError` in the chain,
    # raised in `httpx.Response.json`, and no real response. Chosen by the user on Claude's
    # recommendation (2026-10-17) over detecting garbled JSON route by route, which would
    # reimplement LiteLLM's parsing (#1661), so JSON of the wrong shape still fails as whatever
    # LiteLLM's parser raises (TODO(#1641)). See `_decoded_a_response` for what it leaves out.
    if (
        isinstance(mapped, LLMError)
        and chain.response is None
        and any(_decoded_a_response(link) for link in chain.links)
    ):
        return APIResponseValidationError(
            f"The provider's response is not valid JSON: {mapped.message}",
            status_code=None,
        )
    # A 402 that says how long to wait is a rate limit, not an exhausted quota: waiting clears
    # it. OpenRouter's is the documented case: its 402 for a key's in-flight budget (spend
    # reserved by requests still running or recently settled) is "transient", and "the
    # response includes a Retry-After header; wait for it and retry"
    # (https://openrouter.ai/docs/api/reference/limits), while "a 402 without the header is
    # not a wait-and-retry case"
    # (https://openrouter.ai/docs/api/reference/errors-and-debugging). Read off the header
    # rather than OpenRouter's `limit_source`, so it holds for any provider that sends one.
    # "Says how long" is the wait jaz reads into
    # `LLMError.retry_after`, Google's `RetryInfo` in the body included, which the retry then
    # honours (implementation choice, not decided by the user). It must run before the quota
    # markers, which count every 402 as a quota (`_is_quota_exceeded`). A quota code or message
    # beats the wait (the user's call, 2026-10-17): it names a limit that needs someone to act or
    # a calendar reset, which waiting does not clear, as it already decides on a 429. Google's
    # gemini-cli treats a per-day quota id as terminal whatever its retry delay says
    # (https://github.com/google-gemini/gemini-cli/blob/v0.63.0/packages/core/src/utils/googleQuotaErrors.ts#L326-L337).
    # No provider is known to send both on a 402.
    if (
        isinstance(mapped, LLMError)
        and _provider_status(chain) == 402
        and _has_retry_after(chain.response)
        and not _error_says_quota(_error_object(chain), relayed=True)
    ):
        return RateLimitError(mapped.message, status_code=mapped.status_code)
    # A quota marker overrides the class, whichever LiteLLM chose: see `_is_quota_exceeded`.
    if isinstance(mapped, LLMError) and _is_quota_exceeded(chain):
        return QuotaExceededError(mapped.message, status_code=mapped.status_code)
    # So does a code that names a cause, whatever the status (see `_exceptions._class_for_code`).
    by_code = _class_for_code(_error_object(chain).get("code"))
    if (
        isinstance(mapped, LLMError)
        and by_code is not None
        and type(mapped) is not by_code
    ):
        return by_code(mapped.message, status_code=mapped.status_code)
    # A provider's structured "bad credential" field overrides the class, as a quota marker does:
    # see `_is_bad_credential`.
    if (
        isinstance(mapped, LLMError)
        and type(mapped) is not AuthenticationError
        and _is_bad_credential(chain)
    ):
        return AuthenticationError(
            f"{mapped.message}{_CREDENTIAL_HINT}", status_code=mapped.status_code
        )
    # OpenRouter answers a moderation flag or a model's refusal with a 403, the status it also
    # uses for a key's missing permission; its body tells them apart (see
    # `_is_openrouter_content_block`).
    if type(mapped) is PermissionDeniedError and _is_openrouter_content_block(chain):
        return ContentPolicyViolationError(
            mapped.message, status_code=mapped.status_code
        )
    # Bedrock's use-case-form 404 is the account's standing, not the model name (see
    # `_AWS_ACCOUNT_STANDING_ERRORS`).
    if (
        isinstance(mapped, LLMError)
        and type(mapped) is not PermissionDeniedError
        and chain.response is not None
        and _header_details(chain.response.headers)["code"]
        in _AWS_ACCOUNT_STANDING_ERRORS
    ):
        return PermissionDeniedError(mapped.message, status_code=mapped.status_code)
    # Only what would otherwise be a plain `ServerError` is narrowed: a status-bearing class such
    # as `PermissionDeniedError` is LiteLLM's or the provider's verdict and stands. No HTTP
    # response came back, so `status_code` is `None`, as on the other backends, rather than the
    # one LiteLLM makes up (500 for a refused connection, 408 for a timeout:
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/custom_httpx/llm_http_handler.py#L6080,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L356).
    if isinstance(mapped, ServerError) and type(mapped) is ServerError:
        transport = _transport_failure(chain)
        if transport is NotFoundError:
            # httpx's own text: LiteLLM's above it says "Connection error."
            return NotFoundError(
                f"{chain.failed} {REDIRECT_LOOP_HINT}",
                status_code=None,
            )
        if transport is not None:
            return transport(mapped.message, status_code=None)
        # LiteLLM's 5xx classes (`InternalServerError`, `ServiceUnavailableError`, ...) land
        # here too. A 5xx the provider really sent names the subclass, as on the other
        # backends; the response's own status decides, since LiteLLM's can be one it made up
        # (it reports an Anthropic 529 as a 500:
        # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L547-L551).
        response = chain.response
        if response is not None and 500 <= response.status_code <= 599:
            status = response.status_code
            # Never `None` for a 5xx; the fallback is for the type checker.
            return (_class_for_status(status) or ServerError)(
                mapped.message, status_code=status
            )
        # No response and no transport failure: if a programming error is what failed, it is a
        # bug, which passes through rather than being retried (see `_BUG_CLASSES`).
        if response is None and (
            isinstance(chain.failed, _BUG_CLASSES) or _is_local_failure(chain.failed)
        ):
            return None
    return mapped


def _local_failures() -> tuple[type[BaseException], ...]:
    """Failures of the caller's setup that LiteLLM wraps like a server failure."""
    # TODO(#1757): a `JSONDecodeError` of the caller's own input (`vertex_credentials`, a tool
    # call's `arguments`) is one too, but is retried as a `ServerError`.
    #
    # An `api_base` without a usable scheme (`httpx.UnsupportedProtocol` on the sync path,
    # `httpx.InvalidURL` on the async one; a scheme that is not http(s) is `_is_local_failure`'s
    # other case there), a request httpx refuses to send
    # (`LocalProtocolError`, client-side by httpx's own definition:
    # https://github.com/encode/httpx/blob/0.28.1/httpx/_exceptions.py#L222-L228), and a
    # missing optional dependency (`ImportError`, e.g. the Vertex route without google-auth)
    # fail the same way on every attempt, so they pass through as a bug does rather than being
    # retried for minutes (found by running each through LiteLLM 1.97.0 and 1.105.0).
    import httpx

    return (
        httpx.UnsupportedProtocol,
        httpx.InvalidURL,
        httpx.LocalProtocolError,
        ImportError,
    )


def _is_local_failure(error: BaseException | None) -> bool:
    """Whether ``error`` is a failure of the caller's setup (`_local_failures`)."""
    if isinstance(error, _local_failures()):
        return True
    # An `api_base` whose scheme is not http(s) fails on the async path as aiohttp's
    # `NonHttpUrlClientError`, which LiteLLM's transport maps to a bare `httpx.RequestError`
    # (its `ClientError` row,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/custom_httpx/aiohttp_transport.py#L43-L67),
    # where the sync path raises `httpx.UnsupportedProtocol`; aiohttp's error beneath tells it
    # apart. Its redirect variant is left out, but a server's redirect to such a URL arrives as
    # the plain error too (LiteLLM 1.105.0, aiohttp 3.14.3), so on both paths it passes through as
    # if `api_base` were at fault: TODO(#1719).
    try:
        from aiohttp import NonHttpUrlClientError, RedirectClientError
    except ImportError:
        return False
    cause = error.__cause__ or error.__context__ if error is not None else None
    return isinstance(cause, NonHttpUrlClientError) and not isinstance(
        cause, RedirectClientError
    )


# What LiteLLM raises when a bug fails, beneath the `APIConnectionError` it wraps every exception it
# does not recognise in (`exception_type`'s catch-all,
# https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2670-L2709),
# or beneath its `InternalServerError` on routes built on the `openai` SDK: Python's programming
# errors. One of these as the failed link (the first under LiteLLM's wrappers, `_chain`), with no
# provider response in the chain, is let through as LiteLLM raised it, so it is not retried and
# keeps its traceback, as the translation rule beside `LLMError` says of a bug. `ValueError` is
# left out: LiteLLM raises it for real conditions too, so it stays a retried `ServerError`, except
# a handler's missing-key `ValueError` (see `_mapped_by_class`); a body that is not JSON
# (`json.JSONDecodeError`) is a garbled response (see `_classified`).
#
# Whatever lies below the bug does not change that, a connection failure included (the user's
# call, 2026-10-13). LiteLLM's mapper can fail while it maps a connection failure, and then its
# catch-all raises `APIConnectionError` over the mapper's error with the httpx error beneath
# (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2711-L2737).
# No route was seen
# producing it (a refused port through 37 routes, sync and async, LiteLLM 1.105.0), and a bug
# that fires there fires on every such failure, so it is reported like any other bug rather
# than kept as a second rule.
#
# Claude's call, delegated by the user (2026-10-12, #1661), weighing simplicity, how easy the
# behaviour is to predict, correctness and what a user expects. The chain does not tell a
# LiteLLM bug from a provider 200 whose body lacks the fields LiteLLM reads: on `anthropic/` a
# `{}` or `{"choices": null}` body fails with the same `KeyError`. Three options:
#   * Retry both, as before. A bug is deterministic, so it was retried until the policy gave
#     up (11 calls and minutes of backoff by default) and then reported as a `ServerError` with
#     a status 500 no server sent; a call that reached the provider is billed each time.
#   * Detect a garbled response route by route first (#1641), then pass the rest through. The
#     most precise, but it reimplements LiteLLM's per-provider parsing and drifts with it.
#   * Pass both through (chosen). One rule a user can predict from the traceback, matching the
#     translation rule. The cost: a garbled 200 that LiteLLM fails on this way is not retried,
#     where `SGLangLLM`, which parses its own responses, retries one (#1622). Garbled 200s from
#     a hosted provider are rare, and on `openai/` routes LiteLLM catches them itself (a body
#     with no `choices` is its `APIError`, a malformed choice a bare `Exception` it raises), so
#     those stay a retried `ServerError` (LiteLLM 1.97.0 and 1.105.0), and a body that is not
#     JSON is a retried `APIResponseValidationError` on every route probed (see
#     `_classified`). The `anthropic/` `KeyError`:
#     https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/anthropic/chat/transformation.py#L2182;
#     the `openai/` ones:
#     https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/openai/openai.py#L875,
#     https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py#L192-L199
#     and #L878.
_BUG_CLASSES = (
    AttributeError,
    LookupError,
    NameError,
    TypeError,
    AssertionError,
    NotImplementedError,
)


_GUARDRAIL_BLOCK_STATUSES = frozenset({400, 403, 422})


def _guardrail_block(le: Any, chain: _Chain) -> str | None:
    """The message of a guardrail block run by LiteLLM somewhere in the chain, else ``None``."""
    # A guardrail LiteLLM runs blocked the prompt or the response: a content-policy block,
    # decided in-process rather than by the provider (the user's call, 2026-10-12; these used to
    # escape as classes outside jaz's taxonomy). LiteLLM's guardrails block in two ways, and both
    # count: its own `GuardrailRaisedException` / `BlockedPiiEntityError`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/integrations/custom_guardrail.py#L77-L109),
    # and, for most of its built-in integrations (Lakera, Aporia, Azure, Model Armor, ...; 75
    # raises with a 400 in 1.105.0's `proxy/guardrails/guardrail_hooks/`), a
    # `fastapi.HTTPException`. Those run on the SDK path too, through `CustomGuardrail`'s
    # deployment hooks, when the call passes `guardrails=[...]` (`custom_guardrail.py#L828-L833`
    # at the same tag). An `HTTPException` is a block only with a 400, 403 or 422, the set
    # LiteLLM's own classifier uses (`_GUARDRAIL_BLOCK_STATUS_CODES`, `custom_guardrail.py`,
    # 1.97.0 and 1.105.0; mirrored rather than imported, as it is private). Any other status is
    # the guardrail's own service failing, such as a 429 Purview passes through from the Graph
    # API
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/proxy/guardrails/guardrail_hooks/microsoft_purview/purview_dlp.py#L140-L160),
    # a 408 timeout or a 5xx, and is left alone (TODO(#1663)). The statuses they carry are the
    # ones a LiteLLM proxy would answer with, not a provider's, so `status_code` stays `None`.
    #
    # The chain is walked rather than only the top checked, as a defence against LiteLLM wrapping
    # the block: on a sync call to a model on its Responses API bridge (`gpt-5-pro`, the `-codex`
    # models) it runs the pre-call hook itself (`_run_sync_responses_pre_call_deployment_hook`,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/custom_httpx/llm_http_handler.py#L2414-L2450),
    # and 1.97.0's `completion` wrapped what it raised in LiteLLM's `BadRequestError`; 1.105.0
    # raises it unwrapped. The walk stops at the exception the caller was handling, so one of
    # those is not read as this call's block. A non-guardrail `CustomLogger` hook that raises an
    # `HTTPException` with a block status is read as one, since it is not told apart.
    #
    # A guardrail in passthrough mode blocks with `ModifyResponseException`, which tells a
    # LiteLLM proxy to answer with the violation message instead of an error
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L1201-L1212,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/integrations/custom_guardrail.py#L300-L341);
    # LiteLLM's own classifier counts it as an intervention (`#L77-L109` above), and so does jaz:
    # outside a proxy there is nothing to substitute, so it is a block like the rest (the
    # user's call, 2026-10-12). `SensitiveDataRouteException` asks a proxy to reroute the call
    # to another model (`litellm/exceptions.py#L1236-L1263` at the same tag); jaz does not
    # reroute, as it refuses LiteLLM's `fallbacks`, so `_mapped_by_class` makes it a fail-fast
    # `BadRequestError` (the user's call, 2026-10-12).
    # Limits, each a consequence of the mapping rather than a choice:
    #   * LiteLLM's sync wrapper does not run the deployment hooks (`wrapper` vs
    #     `wrapper_async`, 1.97.0 and 1.105.0:
    #     https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L1678 vs #L2006),
    #     so a sync call is guarded only on the Responses bridge.
    #   * Some integrations signal their own outage, when set to fail closed, or a
    #     misconfiguration as a block (`GuardrailRaisedException` from Singulr, PromptGuard and
    #     Vigil Guard; a 400 from XecGuard and Microsoft Purview, 1.105.0), so those fail fast as
    #     content blocks too:
    #     https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/proxy/guardrails/guardrail_hooks/singulr/singulr.py#L192-L200
    #     https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/proxy/guardrails/guardrail_hooks/promptguard/promptguard.py#L176-L182
    #     https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/proxy/guardrails/guardrail_hooks/vigil_guard/vigil_guard.py#L200-L216
    #     https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/proxy/guardrails/guardrail_hooks/xecguard/xecguard.py#L367-L373
    #     https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/proxy/guardrails/guardrail_hooks/microsoft_purview/purview_dlp.py#L170-L175
    #     LiteLLM's `GuardrailRaisedException.blocked_content` tells the two apart, but it
    #     defaults to `False`, so reading it would make a block from an integration that never
    #     sets it look like an outage (TODO(#1740)).
    #   * A post-call block comes after the provider answered, so the call can be billed
    #     (TODO(#1639)).
    #   * A block relayed over HTTP, by a LiteLLM proxy or by OpenRouter's guardrails, carries
    #     no marker that covers every block, and keeps its status's class (TODO(#1660)).
    blocks = _classes(
        le,
        "GuardrailRaisedException",
        "BlockedPiiEntityError",
        "ModifyResponseException",
    )
    for link in chain.links:
        if isinstance(link, blocks):
            return getattr(link, "message", None) or str(link)
        status = getattr(link, "status_code", None)
        # By name, since `fastapi` is only LiteLLM's proxy extra
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/pyproject.toml#L50-L57);
        # its `HTTPException` subclasses Starlette's
        # (https://github.com/fastapi/fastapi/blob/0.143.0/fastapi/exceptions.py#L17).
        if (
            any(
                klass.__name__ == "HTTPException"
                and klass.__module__.partition(".")[0] in ("fastapi", "starlette")
                for klass in type(link).__mro__
            )
            and status in _GUARDRAIL_BLOCK_STATUSES
        ):
            return str(getattr(link, "detail", None) or link)
    return None


def _openai_setup_error(oai: Any, chain: _Chain) -> BaseException | None:
    """The ``openai`` SDK's bare ``OpenAIError`` in a chain with no response, else ``None``."""
    if chain.response is not None:
        return None
    base = _classes(oai, "OpenAIError")
    sent = _classes(oai, "APIError")
    return next(
        (
            link
            for link in chain.links
            if isinstance(link, base) and not isinstance(link, sent)
        ),
        None,
    )


def _together_overflow(exc: BaseException, status: int | None) -> bool:
    """Whether ``exc`` is Together's 403, which Together documents as a prompt over the model's
    context length."""
    # Together documents its 403 as "Input token count + `max_tokens` parameter must be less
    # than the context length of the model being queried", and nothing else
    # (https://docs.together.ai/docs/error-codes), where providers otherwise use the status for
    # a permission. So on this route the status itself names the overflow, and the class is
    # decided by status and route rather than by LiteLLM's class: LiteLLM reads the overflow
    # only from the legacy wording "`inputs` tokens + `max_new_tokens` must be <="
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L1716-L1735)
    # and gives the documented wording its generic class, which the status rule would make a
    # `PermissionDeniedError` telling the caller to fix a permission when the prompt has to
    # shrink. A Together 403 for anything else is undocumented and would be misread here. A
    # LiteLLM proxy relaying Together reports `litellm_proxy`, not this route, and keeps the
    # status rule.
    return status == 403 and getattr(exc, "llm_provider", None) == "together_ai"


def _mapped_by_class(litellm: Any, chain: _Chain) -> Exception | None:
    """Translate a LiteLLM exception by its class and the provider's status.

    LiteLLM raises its *own* exception classes (which subclass the ``openai`` SDK's, not JAZ's).
    The base retries only its ``retryable_exceptions``, so without this translation a rate
    limit or 5xx would fail fast as a LiteLLM class instead of being retried, and downstream
    ``except jaz.exceptions.X`` would never see a bad API key, a content-policy rejection or a
    context overflow either. Checked most-specific first: LiteLLM's
    ``ContextWindowExceeded`` / ``ContentPolicyViolation`` / ``UnsupportedParams`` all subclass
    ``BadRequestError``, so order is load-bearing. Every error in the ``openai`` SDK's ``APIError``
    family is mapped, and so are LiteLLM's ``BudgetExceededError``, the errors its guardrails
    raise to block a call, a guardrail's request to reroute the call, and a missing ``openai``
    key; any other error outside it, and any other ``openai`` client setup error, is re-raised
    as-is.
    """
    # LiteLLM's `BadRequestError` subclasses:
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L535, #L619, #L945
    exc = chain.exc
    le = litellm.exceptions
    # LiteLLM's exceptions subclass the `openai` SDK's, and jaz does not depend on `openai`
    # directly, so its classes are reached through LiteLLM's own import of it.
    oai = getattr(le, "openai", None)
    # `litellm.max_budget`, LiteLLM's own spend cap for the process, raised before any request
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L1736-L1741
    # and #L2020-L2025).
    # Not an `openai` SDK error, so the rows below would let it through; DSPy treats it as a
    # billing error too
    # (https://github.com/stanfordnlp/dspy/blob/3.4.1/dspy/clients/engines/litellm_errors.py#L49).
    if isinstance(exc, _classes(le, "BudgetExceededError")):
        return QuotaExceededError(str(exc))
    block = _guardrail_block(le, chain)
    if block is not None:
        return ContentPolicyViolationError(block)
    if isinstance(exc, _classes(le, "SensitiveDataRouteException")):
        target = getattr(exc, "route_to_model", None) or "another model"
        return BadRequestError(
            f"A LiteLLM guardrail asked to reroute this call to {target} "
            f"({getattr(exc, 'message', exc)}). Rerouting is a LiteLLM proxy feature and jaz "
            "does not reroute, so remove the guardrail or call that model directly."
        )
    status = _provider_status(chain)
    message = getattr(exc, "message", None) or str(exc)
    message = _without_litellm_prefix(message)
    # The `openai` SDK raises a bare `OpenAIError`, outside its `APIError` family as nothing was
    # sent, when it cannot build a client, a missing key among them. LiteLLM recognises the
    # missing key only by openai 2.20's wording
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2031);
    # openai 2.54 words it differently
    # (https://github.com/openai/openai-python/blob/v2.54.0/src/openai/_client.py#L232), and
    # LiteLLM then files it under its catch-all `APIConnectionError`, a 500, which would be
    # retried. So the SDK's own error decides: one naming `api_key` (both wordings do) is an
    # `AuthenticationError`, and any other bare `OpenAIError` (such as `provider` combined with
    # top-level auth options, #L185-L188 at the same tag) passes through as LiteLLM raised it,
    # failing fast like the other local failures (`_local_failures`).
    # Across releases: the class test rests on the SDK's public hierarchy and LiteLLM keeping the
    # SDK error in the chain, the text test on the parameter name both wordings share. If a release
    # rewords the error without `api_key`, it still fails fast, as LiteLLM's `APIConnectionError`
    # without the hint; if it breaks the class test, the error is a retried `ServerError` again.
    # Either way the dependency canary's real missing-key call
    # (`test_auth_failure_points_at_this_console_affordance`) goes red. Checking for a key before
    # the call instead would mean re-implementing how LiteLLM and the SDK resolve credentials, and
    # openai 2.54 accepts others (`workload_identity`, `admin_api_key`; #L232 at the same tag).
    # TODO(#1709): the Azure client's own setup errors (no endpoint, no `api_version`) are
    # `ValueError`s, which this does not catch, so they are still retried.
    setup = _openai_setup_error(oai, chain)
    if setup is not None:
        if "api_key" not in str(setup):
            return None
        return AuthenticationError(f"{message}{_CREDENTIAL_HINT}")
    # LiteLLM's own provider handlers raise a plain `ValueError` when they have no key to send,
    # before any request: Gemini's "Missing Gemini API key"
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/vertex_ai/vertex_llm_base.py#L725-L728),
    # Fireworks' "FIREWORKS_API_KEY is not set"
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/fireworks_ai/chat/transformation.py#L212-L213).
    # LiteLLM wraps it in its catch-all `APIConnectionError`, which would be retried as a
    # `ServerError`. So a `ValueError` that failed (`_chain`) and names an API key is an
    # `AuthenticationError` too, read off its text as the SDK's error is above, since nothing
    # else marks it, when nothing else is in the chain: neither a response nor a handler's
    # error. The handler's error is what tells a request that was answered: Gemini's transform
    # raises a `ValueError` quoting a 200 body it cannot use, under its `VertexAIError`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L1745-L1746
    # and #L2504-L2508), and that body can mention an API key. A `ValueError` for any other
    # setup problem (Predibase's missing tenant id,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/predibase/chat/transformation.py#L330-L333,
    # an `api_base` Triton cannot parse,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/triton/completion/transformation.py#L163-L168)
    # is still retried: TODO(#1709).
    if (
        chain.response is None
        and chain.provider_error is None
        and isinstance(chain.failed, ValueError)
        and _NAMES_API_KEY.search(str(chain.failed))
    ):
        # The handler's own text: LiteLLM's catch-all message around it carries its traceback
        # and a "This looks like a bug in LiteLLM" footer
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2691-L2706).
        return AuthenticationError(f"{chain.failed}{_CREDENTIAL_HINT}")
    rows: list[tuple[Any, str, type[LLMError]]] = [
        (le, "ContextWindowExceededError", ContextWindowExceededError),
        (le, "ContentPolicyViolationError", ContentPolicyViolationError),
        (le, "AuthenticationError", AuthenticationError),
        (le, "PermissionDeniedError", PermissionDeniedError),
        (le, "NotFoundError", NotFoundError),
        (le, "RateLimitError", RateLimitError),
        (le, "UnsupportedParamsError", UnsupportedParamsError),
        # The SDK's bases, not LiteLLM's: `le.InvalidRequestError` is an `openai.BadRequestError`
        # but not an `le.BadRequestError`
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L1028).
        # A 400 or 422 is the request's fault, so both fail fast.
        (oai, "BadRequestError", BadRequestError),
        (oai, "UnprocessableEntityError", BadRequestError),
        # LiteLLM's `JSONSchemaValidationError` is a response that failed the schema the caller
        # asked LiteLLM to enforce (`litellm.enable_json_schema_validation`, or
        # `response_format.enforce_validation`): an invalid response, so ours. Its base,
        # LiteLLM's `APIResponseValidationError`, is not mapped: LiteLLM raises it bare only when
        # a user's own `pre_call_rules`/`post_call_rules` reject a call, not for a garbled
        # response, so it stays a plain `ServerError` (TODO(#1641): a garbled 200 is not
        # detected on this backend).
        (le, "JSONSchemaValidationError", APIResponseValidationError),
        # The catch-all is the SDK's `APIError` because LiteLLM's is not a base of anything: its
        # connection, timeout and 5xx errors subclass the SDK's classes directly
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L811, #L853, #L356, #L760).
        # They map to the retryable `ServerError`, which `_classified` then narrows to
        # `APITimeoutError` or `APIConnectionError` when the chain shows a transport failure, or to
        # `InternalServerError` or `OverloadedError` when a real 5xx response is in it.
        (oai, "APIError", ServerError),
    ]
    jaz_type = next(
        (jaz for module, name, jaz in rows if isinstance(exc, _classes(module, name))),
        None,
    )
    if jaz_type is None:
        return None
    # The status decides the class, and LiteLLM's class stands only where it refines the status's
    # by subclassing it: a 400 `ContextWindowExceededError` stays one (a `BadRequestError`), a 403
    # or 503 one becomes the status's class. A subclass retries as the class it refines
    # (`test_a_refinement_retries_as_the_status_s_class`), so a class contradicting the status's
    # verdict never stands, which is the user's call (2026-10-13). The subclass test itself is the
    # user's call (2026-10-16), over two looser rules that let a class refine without
    # subclassing, provided the two agreed on retrying:
    #   * any class naming a cause no status names: it kept a context overflow LiteLLM read from
    #     the text of a 401, 403 or 404, so `except AuthenticationError` missed a bad key;
    #   * any class at all on a plain `BadRequestError`: it kept Gemini's bad-key 400 an
    #     `AuthenticationError` with the credential hint
    #     (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L1192-L1193),
    #     but LiteLLM reads that from the text, so a 400 merely quoting "Invalid API Key" told the
    #     user to fix a key that works.
    # The rule alone loses Gemini's bad-key 400 and Bedrock's bad-credential 403; `_classified`
    # wins them back from the providers' structured fields (`_is_bad_credential`). LiteLLM's class
    # goes wrong in several ways, which is why the status decides:
    #   * Its mapper falls back to `APIError`, `APIConnectionError` or `BadRequestError` when it
    #     has no branch for an error: an OpenAI 403 or 413 arrives as `APIError`, and from 1.100.0
    #     any 4xx a route's mapper does not handle as `BadRequestError`
    #     (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2245
    #     and #L2333-L2340, absent in 1.99.0).
    #   * It picks some classes from the message text whatever the status: a rate limit when
    #     "rate limit" appears anywhere in it ("the phrase branches stay ungated", #L45-L75 at
    #     the same tag), a context overflow on a list of phrases (#L82-L114, checked before any
    #     status in its OpenAI mapper, #L317), and an Azure content-policy violation on another
    #     (#L117-L134, checked before any status in its Azure mapper, #L1989). Providers echo
    #     the request in validation errors (vLLM's 400 includes the offending input,
    #     https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/api_server.py#L1540-L1555),
    #     so a 400 quoting a prompt about rate limits would be
    #     a retried rate limit, and a 503 whose body mentions the context length a fail-fast
    #     overflow.
    #   * A per-provider mapper can disagree with the status: 1.92.0 to 1.99.0 report a watsonx
    #     403 as `AuthenticationError`
    #     (https://github.com/BerriAI/litellm/blob/v1.97.0/litellm/litellm_core_utils/exception_mapping_utils.py#L751-L752).
    # LiteLLM's own Router decides by status too
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/router.py#L7947-L7951,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L7165).
    # The status is trusted over the class when the provider's response is in the chain, and for
    # a generic class whatever its source, since that class says nothing: from the provider
    # error, or LiteLLM's own when no request reached the provider. A specific class with no
    # response stands, its status being one LiteLLM or its handler chose alongside it. The
    # overrides in `_classified` still apply to its result, such as a quota marker making any
    # class a `QuotaExceededError`; a guardrail block (above) has no response status and is
    # decided before it.
    generic = _classes(le, "APIError", "APIConnectionError", "BadRequestError")
    if _together_overflow(exc, status):
        jaz_type = ContextWindowExceededError
    elif status is not None and (chain.response is not None or type(exc) in generic):
        by_status = _class_for_status(status)
        if by_status is not None and not issubclass(jaz_type, by_status):
            # A 5xx stays a plain `ServerError` here: LiteLLM makes one up for a refused
            # connection or a bug of its own, so `_classified` narrows it to a transport
            # failure, lets a bug through, or makes it an `InternalServerError` once a real
            # response shows the provider sent it.
            jaz_type = (
                ServerError if issubclass(by_status, InternalServerError) else by_status
            )
    if jaz_type is AuthenticationError:
        # TODO(#1747): a 401 after a cross-origin 307/308, which strips `Authorization`, blames
        # the key here though the address is what is wrong. TODO(#1771): so does OpenAI's 401
        # for an IP outside the project's allow-list, which carries no code to tell it apart.
        message = f"{message}{_CREDENTIAL_HINT}"
    if jaz_type is NotFoundError and status is not None and 300 <= status <= 399:
        # LiteLLM's message for one is empty on most routes and only "Error code: 300" on
        # `openai/`, so the status goes in the hint.
        message = f"{message} (a {status} LiteLLM could not follow; check api_base)"
    # `LLMError.status_code` is the provider's answer, so with no HTTP response in the chain it
    # is `None`, whatever status LiteLLM or its handler put on the exception: theirs is made up,
    # before any request (a missing Anthropic key is LiteLLM's `AuthenticationError`, 401,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/anthropic/common_utils.py#L1070-L1075
    # and https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L143;
    # unsupported params its `UnsupportedParamsError`, 400, #L957 there; OCI's handler raises a
    # 401 for missing credentials,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/oci/chat/transformation.py#L424-L426)
    # or for a 200 it could not use (the 422 of the `OpenAIError` LiteLLM builds over a garbled
    # 200 on `openai/` and the other OpenAI-compatible routes, #1641; the 500 on a response
    # failing `JSONSchemaValidationError`,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L906). The status
    # still classifies the error above. A local server answering 400, 401, 429 and 503 was
    # reached by 51 of 63 routes, sync and async (LiteLLM 1.105.0), and each left the response
    # in the chain, under a handler's error where there was one. Reading the response alone,
    # rather than also a handler's error without one, is an implementation choice, not decided
    # by the user: every handler error seen without a response carried a status nobody sent.
    if chain.response is None:
        status = None
    return jaz_type(message, status_code=status)


# The keys LiteLLM's Gemini transform puts in `_hidden_params` of a response it parses, side by
# side (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L2472-L2482).
# Either one marks it: LiteLLM keeps `vertex_ai_safety_results` there only "to prevent
# regressions", so it is the likelier to go, while its own Langfuse integration reads
# `vertex_ai_grounding_metadata`
# (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/integrations/langfuse/langfuse.py#L1233);
# no other route sets either (observed: grepping LiteLLM 1.105.0's package for both keys finds
# them written only in that Gemini transform module).
_GEMINI_TRANSFORM_MARKS = ("vertex_ai_grounding_metadata", "vertex_ai_safety_results")


def _parsed_by_gemini_transform(response: Any) -> bool:
    """Whether LiteLLM parsed ``response`` with its Gemini transform."""
    # Read off the response rather than worked out from the route: which routes use the
    # transform depends on LiteLLM's routing (`gemini/`, `vertex_ai_beta/`, and `vertex_ai/`
    # by model name, `base_model` and `api_base`), which the response already reflects. The
    # transform skips the marks only on the flagged stubs it returns early (a blocked prompt, a
    # flagged `finishReason`, #L2431-L2443 at the same tag), which have no output, so this is
    # never asked of them. The transform also sets each mark as an attribute, which is what
    # survives LiteLLM's cache: a cache hit is rebuilt from `model_dump_json()`, which leaves out
    # `_hidden_params`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/caching/caching.py#L694-L695,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/caching/caching_handler.py#L883-L886).
    hidden = getattr(response, "_hidden_params", None)
    return any(
        (isinstance(hidden, dict) and mark in hidden)
        or getattr(response, mark, None) is not None
        for mark in _GEMINI_TRANSFORM_MARKS
    )


def _raise_if_refused(response: Any, model: str) -> None:
    """Raise :class:`ContentPolicyViolationError` if a successful response says the provider
    refused or filtered it, or :class:`PermissionDeniedError` if it says the account is
    restricted."""
    # The rule (the user's call on #669): a refusal or content filter the provider signals
    # explicitly on a 200 raises, like the 4xx a backend maps to the same class, so it is not
    # retried by default. Anything else stays a normal response, an empty or length-cut one
    # included: that carries no verdict, and routing it is #697's.
    #
    # Two fields carry the signal through LiteLLM 1.105.0, checked against each provider's real
    # 200 body (tests/llm/test_litellm.py, `TestRefusalOnSuccess`):
    # - `finish_reason == "content_filter"`, which LiteLLM normalizes from Anthropic's
    #   `stop_reason: "refusal"`, Gemini's flagged `finishReason`s and its prompt block
    #   (`promptFeedback.blockReason`, which it answers with one empty choice rather than
    #   raising), and passes through from OpenAI and Azure. The provider's own reason can sit in
    #   the choice's `provider_specific_fields["native_finish_reason"]`: `Choices` puts it there
    #   when a transform hands it the unmapped one, as Gemini's does and Anthropic's does not
    #   (https://docs.litellm.ai/docs/completion/output,
    #   https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/types/utils.py#L1662-L1667), and
    #   OpenRouter's body carries a `native_finish_reason` of its own
    #   (https://openrouter.ai/docs/api_reference/overview#finish-reason) that lands there as well
    #   (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py#L702);
    #   the message names it when it says more than `content_filter`.
    #   https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/core_helpers.py#L203, #L244
    #   https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L1630-L1660
    #   and #L2431-L2443
    # - OpenAI's `message.refusal`, which LiteLLM moves to `provider_specific_fields` and
    #   leaves `finish_reason` at "stop"
    #   (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py#L667-L669).
    #   It lands there only because LiteLLM's `Message` has no `refusal` field, so the attribute
    #   is read too: were LiteLLM to add the field, the refusal would no longer be in
    #   `provider_specific_fields`.
    #
    # Content alongside `content_filter` still raises: it is output the filter cut short (Azure,
    # https://learn.microsoft.com/en-us/azure/foundry-classic/foundry-models/concepts/content-filter)
    # or a refusal's preamble (Anthropic,
    # https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback),
    # and returned as a normal turn it would reach the REPL as truncated code with nothing
    # marking it so. LiteLLM already drops Gemini's partial text here, so raising also keeps the
    # providers alike.
    #
    # Gemini's RECITATION (a recitation filter: output too close to source material) counts: the
    # provider blocked the output, the model did not choose to stop, so it is not a usable answer
    # — and LiteLLM folds it into `content_filter` anyway, so it could not be told apart
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/core_helpers.py#L215-L231,
    # https://ai.google.dev/api/generate-content#FinishReason).
    #
    # LiteLLM folds more Gemini reasons into `content_filter` than are policy blocks, so on Gemini
    # the native reason decides: a stop whose reason is `OTHER`, `LANGUAGE`, `NO_IMAGE` or
    # `IMAGE_OTHER` is returned, even with no output, since none is a policy verdict (the user's
    # call, 2026-10-13: "NO_IMAGE and IMAGE_OTHER aren't policy blocks", chosen over keeping them
    # raised); any other (`ESCALATION`, `IMAGE_RECITATION`, ...) raises, with or without text.
    # The transform keeps the reason on each candidate it parses
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L2349-L2360).
    # Not gated on the route (an implementation choice, not decided by the user): the names are
    # Gemini's, and mean the same through any route that keeps them.
    #
    # Where the reason is absent, the response's shape decides. For SAFETY, PROHIBITED_CONTENT,
    # BLOCKLIST, SPII, RECITATION, IMAGE_SAFETY, IMAGE_PROHIBITED_CONTENT and a blocked prompt,
    # LiteLLM discards whatever Gemini sent and returns a bare stub, with no content, reasoning,
    # tool calls or native reason
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L1313-L1327
    # and #L1664-L1690), so it raises. A LiteLLM cache hit has lost the reason too: the rebuild
    # keeps only a choice's keys outside `Choices`'s fields
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py#L702),
    # and hands `Choices` the already-mapped `finish_reason`. There a Gemini-transform response
    # with output is returned and one without raises. This rests on the stub having no text: if
    # LiteLLM made it keep the blocked text but not the native reason, a block would be returned
    # (TODO(#1742)). Gated on the transform
    # (`_parsed_by_gemini_transform`), since only there is a block's text discarded: a Gemini
    # model reached another way (OpenRouter, Google's OpenAI-compatible endpoint, a proxy) keeps
    # a filtered response's text, and the `vertex_ai` route also serves Claude and other partner
    # models. Limitation (TODO(#1659)): on a cache hit, a no-verdict stop with no output raises,
    # and an `ESCALATION` or `IMAGE_RECITATION` block with text, or a `PUP_LIMITED_DISABLED`
    # stop (below), is returned.
    #
    # Gemini's `PUP_LIMITED_DISABLED` raises whatever `finish_reason` LiteLLM gives it: "token
    # generation stopped because the user account is limited or disabled due to Prohibited Use
    # Policy (PUP) violations" (https://ai.google.dev/api/generate-content#FinishReason), and
    # LiteLLM maps it to "stop", as it does any reason it does not know
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/core_helpers.py#L253-L257),
    # so it would otherwise be returned as a normal turn. The user's call (2026-10-14), made on
    # the condition that it does not clear on retry: Google ties it to the account, not the
    # request, so resending cannot change it. Google publishes no duration for the limit, and it
    # was not tried against a limited account. It is a `PermissionDeniedError` (the user's call,
    # 2026-10-17): what has to change is the account's standing, not the request, and
    # `ContentPolicyViolationError`'s docstring invites opting into retries, which cannot clear
    # it.
    #
    # TODO(#1642): models LiteLLM routes through its Responses-API bridge (`gpt-5-pro`,
    # `*-codex`, `o3-pro`, ...) lose a refusal before it reaches here: LiteLLM 1.105.0 drops the
    # `refusal` content part and reports `finish_reason="stop"`. Their content filter does
    # arrive, as `finish_reason="content_filter"`.
    #
    # TODO(#1639): a refused call can be billed (Anthropic bills a mid-stream refusal, and one
    # before any output in some categories; Azure OpenAI bills a `content_filter` stop and a
    # content-filter 400, https://learn.microsoft.com/en-us/azure/foundry-classic/openai/faq),
    # but its cost is not reported, since cost reaches hooks only on a returned `LLMResponse` and
    # an `LLMError` carries none.
    #
    # `status_code` is left `None`: LiteLLM's response object does not carry the status. Its
    # headers it keeps on most routes, so the request id and the rest come from there: as is
    # under `_hidden_params["headers"]`, or under `_hidden_params["additional_headers"]` with an
    # `llm_provider-` prefix
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/llm_response_utils/get_headers.py#L37-L45,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/openai/openai.py#L293).
    # Its Gemini route keeps none (LiteLLM 1.105.0), so there they are
    # `None`. The body is the parsed response, not an error body, so it stays `None`.
    choices = getattr(response, "choices", None)
    if not choices:
        return
    details = _header_details(_response_headers(response))
    choice = choices[0]
    message = getattr(choice, "message", None)
    fields = getattr(message, "provider_specific_fields", None)
    refusal = getattr(message, "refusal", None) or (
        fields.get("refusal") if isinstance(fields, dict) else None
    )
    if isinstance(refusal, str) and refusal:
        raise ContentPolicyViolationError(
            f"{model} refused the request: {refusal}", **details
        )
    if _native_finish_reason(choice) == "PUP_LIMITED_DISABLED":
        raise PermissionDeniedError(
            f"{model} stopped because the Google account is limited or disabled for violating "
            "the Prohibited Use Policy (finish reason 'PUP_LIMITED_DISABLED'); resending the "
            "request will not help.",
            **details,
        )
    if getattr(choice, "finish_reason", None) == "content_filter":
        native = _native_finish_reason(choice)
        content = getattr(message, "content", None)
        if native is not None:
            if native in _GEMINI_STOPS_WITHOUT_VERDICT:
                return
        elif _parsed_by_gemini_transform(response) and (
            content
            or getattr(message, "reasoning_content", None)
            or getattr(message, "tool_calls", None)
        ):
            return
        reason = (
            f" (provider reason {native!r})"
            if native is not None and native != "content_filter"
            else ""
        )
        # The text that came with the stop (a refusal's preamble, a guardrail's message, the
        # output cut short) is the provider's explanation when there is one.
        said = f" The provider returned: {content}" if content else ""
        raise ContentPolicyViolationError(
            f"{model} stopped with finish_reason 'content_filter'{reason}: the provider's "
            f"content filter blocked the prompt or the response, or the model refused it.{said}",
            **details,
        )


# The Gemini finish reasons LiteLLM reports as `content_filter` that are no policy verdict
# (https://ai.google.dev/api/generate-content#FinishReason).
_GEMINI_STOPS_WITHOUT_VERDICT = frozenset(
    {"OTHER", "LANGUAGE", "NO_IMAGE", "IMAGE_OTHER"}
)


def _native_finish_reason(choice: Any) -> str | None:
    """The provider's own finish reason, where LiteLLM kept it beside the mapped one."""
    fields = getattr(choice, "provider_specific_fields", None)
    native = fields.get("native_finish_reason") if isinstance(fields, dict) else None
    return native if isinstance(native, str) and native else None


def _response_headers(response: Any) -> dict[str, str]:
    """The provider's headers on a response LiteLLM parsed."""
    hidden = getattr(response, "_hidden_params", None)
    if not isinstance(hidden, dict):
        return {}
    headers = hidden.get("headers")
    if isinstance(headers, Mapping) and headers:
        return dict(headers)
    additional = hidden.get("additional_headers")
    if not isinstance(additional, Mapping):
        return {}
    return {
        str(k).removeprefix("llm_provider-"): v
        for k, v in additional.items()
        if str(k).startswith("llm_provider-")
    }


def _refuse_query_params(params: Mapping[str, Any], caller: str) -> None:
    """Raise ``TypeError`` if ``params`` holds a key this backend refuses."""
    retry = [name for name in _RETRY_KEYWORDS if name in params]
    if retry:
        # Read from the class, so a field added to `RetryPolicy` is named here too.
        fields = ", ".join(field.name for field in dataclasses.fields(RetryPolicy))
        raise TypeError(
            f"{caller}() got {', '.join(retry)}: retry settings are one value — pass "
            f"retry_policy=RetryPolicy(...) (fields: {fields})."
        )
    for name in _REFUSED_QUERY_PARAMS:
        if name in params:
            raise TypeError(f"{caller}() got {name}, which is not supported.")
    # A truthy `stream` makes LiteLLM return a chunk iterator, which `_to_response` cannot
    # read: the turn came back with `content=None`, and mid-stream errors and refusals were never
    # seen. Refused rather than consumed (implementation choice, not decided by the user): a jaz
    # turn is the whole response, so assembling chunks would buy nothing but a second path
    # through error mapping. A falsy `stream` is let through: LiteLLM's default is `None`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/main.py#L5153).
    if params.get("stream"):
        raise TypeError(
            f"{caller}() got stream={params['stream']!r}, which is not supported: the backend "
            "reads each response whole. Leave stream unset."
        )


# LiteLLM's reader of the billed amount a provider's transform keeps (`LiteLLM._cost`).
_BILLED_COST_HELPER = "get_response_cost_from_hidden_params"


def _usd(value: Any) -> float | None:
    """``value`` as a cost in USD if it is a finite, non-negative number, else ``None``."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return float(value)
    return None


class LiteLLM(BaseLLM):
    """LLM backend that routes completions through LiteLLM — the default, reaching every
    provider LiteLLM routes to (OpenAI, Anthropic, Gemini, Bedrock, Vertex, …)::

        jaz.configure(llm=LiteLLM(model="openai/gpt-5-mini"))

    The model id is LiteLLM's routing key and is sent verbatim, so it carries a ``provider/``
    prefix (``openai/gpt-5-mini``, ``anthropic/claude-sonnet-5``, ``vertex_ai/gemini-2.5-pro``)
    written exactly as LiteLLM's own documentation writes it. The provider LiteLLM routes to also
    selects the credential: a key stored for ``"openai"`` in ``~/.jaz/credentials.json`` is used
    for ``openai/…`` models — and for a bare ``gpt-5-mini``, since LiteLLM infers the provider
    when there is no prefix — behind an explicitly passed ``api_key`` and behind the provider's
    environment variable. Cost is computed by :func:`litellm.completion_cost` from LiteLLM's
    price map, at the rates of the model that answered (and, for a region passed as
    ``aws_region_name``, that region's, where the map lists one), and reported only when ``model_info.can_report_cost`` (see :class:`ModelInfo`), read
    from the configured model's entry, says it can be. For a model id the map does not know, such
    as an Azure deployment name, pass LiteLLM's ``base_model`` to name the model it serves; both
    cost and ``model_info`` then read that model's entry::

        LiteLLM(model="azure/my-deployment", base_model="azure/gpt-4o")

    On OpenRouter, the cost is the amount OpenRouter reports charging, when it reports one, plus
    the provider's charge to your own key when you bring one.

    Args:
        model: The LiteLLM route id, ``provider/model``.
        allowed_openai_params: Request params LiteLLM would reject for this provider but should
            forward anyway (an escape hatch for a model newer than the installed LiteLLM).
        drop_params: Silently drop request params the provider does not support instead of
            raising.
        retry_policy: How failed calls are retried; see :class:`RetryPolicy`.
        **query_params: Per-request defaults sent on every call (``temperature``,
            ``max_tokens``, ``reasoning_effort``, ``api_base``, ...).

    Limitations:
        * Importing ``litellm`` is slow (~2 s) and pulls a heavy dependency tree; the import is
          deferred until the first call, so ``import jaz`` does not pay it.
        * LiteLLM fetches its price map live, so reported cost can vary over time, and a model
          LiteLLM has not mapped reports no cost (``model_info.can_report_cost`` is ``False``
          for it).
        * A response that carries no token usage (an OpenAI-compatible server that omits
          ``usage``, which LiteLLM fills in as zeros) reports its token counts and ``cost_usd``
          as ``None``. So does one reporting zero prompt tokens, which no real call has.
        * On ``anthropic/`` models, ``extra["reasoning_tokens"]`` is LiteLLM's estimate from
          the thinking text when Anthropic reports no count of its own.
        * A parameter LiteLLM recognizes is always transformed; ``allowed_openai_params`` forces
          it past validation but not past LiteLLM's mapping.
        * LiteLLM's ``fallbacks``, ``model_list`` and ``context_window_fallback_dict`` are not
          supported, and neither is ``stream=True``: passing any of them raises ``TypeError``,
          as does LiteLLM's ``num_retries`` (set retries with ``retry_policy``),
          and a call made while ``litellm.model_fallbacks`` or ``litellm.num_retries`` is set
          raises :class:`~jaz.exceptions.UnsupportedParamsError`.
        * A refused or filtered call raises
          :class:`~jaz.exceptions.ContentPolicyViolationError`, and its cost is not reported.
        * A programming error inside LiteLLM (a ``KeyError``, ``AttributeError``, ``TypeError``,
          ...) is raised as LiteLLM's own exception and is not retried, whether a bug in LiteLLM
          caused it or a response LiteLLM could not read, and even when LiteLLM hit it while
          handling a dropped connection. To retry it, add that exception's
          class, such as ``litellm.APIConnectionError``, to
          ``RetryPolicy.extra_retryable_exceptions``.
    """

    # The live price map the docstring mentions:
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/get_model_cost_map.py#L618
    # The Anthropic reasoning-token estimate (`token_counter` over the thinking text, used only
    # when the usage block carries no thinking count):
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/anthropic/chat/transformation.py#L2296-L2303

    # Why a whole backend around what looks like one `litellm.completion` call: the executive
    # decision recorded in PR #1104 (design doc "LiteLLM as the sole backend for v1", never
    # merged) and its parent #1082 makes LiteLLM the DEFAULT v1 backend and the only one that
    # reaches vendor APIs — there are no hand-written vendor-direct transports. `GitHubCopilotLLM`
    # and `SGLangLLM` are a CLI-backed credential path and a self-hosted server, neither of which
    # LiteLLM covers. Each rule below (verbatim dispatch, num_retries=0, price via
    # completion_cost, carry the cache buckets, map exceptions into JAZ's taxonomy) is
    # transcribed from that doc.

    def __init__(
        self,
        *,
        model: str,
        allowed_openai_params: list[str] | None = None,
        drop_params: bool | None = None,
        retry_policy: RetryPolicy | Mapping[str, Any] = RetryPolicy(),  # noqa: B008
        **query_params: Any,
    ) -> None:
        # `retry_policy` is named here, not forwarded from the tail: with an open tail of its own
        # a backend must say which keywords are the base's, or they would land in the tail.
        super().__init__(retry_policy=retry_policy)
        # This backend's own setting, not the base's: the route id LiteLLM dispatches on.
        self._model = _require_model_id("LiteLLM", model)
        _refuse_query_params(query_params, "LiteLLM")
        _warn_on_misspelled_request_params(query_params, "LiteLLM")
        # The open tail (`temperature`, `reasoning_effort`, `max_tokens`, `api_base`, ...): the
        # params this backend puts on every `litellm.completion` call. Its own, stored here and
        # sent by its own `complete`; the base never sees them. A plain dict — a proxy would
        # not pickle, and a backend must stay copyable — that nothing outside reads.
        self._query_params: dict[str, Any] = dict(query_params)
        # Config-file escape hatches for a model too new for the *installed* LiteLLM: unblock a
        # param LiteLLM would reject (`allowed_openai_params`) or silently drop unknown ones
        # (`drop_params`) from a YAML/settings file rather than a code change. Held as attributes,
        # not folded into the query params, because they are LiteLLM-call controls — not model
        # params that belong on the wire.
        # Private, like every other setting a backend is built with: nothing reads them back
        # (the console describes a component from its recorded settings, not its attributes).
        # ``tuple(...)`` copies the list, so a caller mutating theirs afterwards cannot reach a
        # backend that a ``ConfigOverride`` shares by reference with the base config (#741).
        self._allowed_openai_params: tuple[str, ...] | None = (
            None if allowed_openai_params is None else tuple(allowed_openai_params)
        )
        self._drop_params = drop_params

    def __repr__(self) -> str:
        return repr_call(
            "LiteLLM",
            ("model", self._model),
            (
                "allowed_openai_params",
                None
                if self._allowed_openai_params is None
                else list(self._allowed_openai_params),
            ),
            ("drop_params", self._drop_params),
            ("retry_policy", self._retry_policy),
            *self._query_params.items(),
        )

    def _with_query_params(self, **extra: Any) -> LiteLLM:
        """A shallow copy of this backend with ``extra`` merged into its query params.

        Private and deprecated — not API; do not add callers. The copy shares every handle with
        the original; only the params dict is replaced, so the original (shared by reference with
        the base config) is never touched.

        Raises:
            TypeError: ``extra`` carries a query param the constructor refuses.
        """
        # TODO(#1313): delete this once `evals/` leaves the repo (#1289). Its only caller is that
        # tree's per-task `prompt_cache_key` path; jaz-evals-internal stamps the key into the
        # authored spec before the backend is built, so nothing live needs it.
        #
        # The refusal repeats `__init__`'s because this is the only other way into
        # `_query_params`, and it is a `TypeError` here too so the same mistake raises the same
        # error whichever way it came in. It is deleted with this method; nothing else depends on
        # it, since `_build_params`' per-call check covers only the process-wide global.
        _refuse_query_params(extra, "_with_query_params")
        clone = copy.copy(self)
        clone._query_params = {**self._query_params, **extra}
        return clone

    def _build_params(self, messages: list[Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self._model, "messages": messages}
        if self._allowed_openai_params is not None:
            # A fresh list per call: the stored value is a frozen tuple (#741), and handing the
            # shared container to litellm would let a callee mutating it corrupt the base config.
            params["allowed_openai_params"] = list(self._allowed_openai_params)
        if self._drop_params is not None:
            params["drop_params"] = self._drop_params
        params.update(self._query_params)
        # LiteLLM's own retries off, and the provider SDK's with them (`_RETRY_KEYWORDS`): JAZ's
        # retryer is hook- and budget-aware, and a second retry loop underneath it would silently
        # multiply attempts and spend. No query param can carry `num_retries`; the constructor
        # refuses it.
        params["num_retries"] = 0
        # The forced 0 does not cover the process-wide `litellm.num_retries` on the sync path:
        # `completion`'s wrapper reads `num_retries or litellm.num_retries`, and 0 is falsy, so a
        # set global wins (LiteLLM 1.105.0; `acompletion`'s wrapper honours a per-request 0:
        # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L1899 and #L1436-L1439).
        # Refused on both paths, like `model_fallbacks` below, so the setting is never silently
        # half-applied, and rather than reset because it is the caller's. Truthiness, because that
        # is the sync wrapper's test: 0 and `None` both leave it off. Removable once LiteLLM's sync
        # wrapper lets an explicit 0 win.
        if _litellm().num_retries:
            raise UnsupportedParamsError(
                "litellm.num_retries is set, which is not supported. Set "
                "litellm.num_retries = None, and set retries with RetryPolicy."
            )
        # Checked per call because the process-wide `litellm.model_fallbacks` can be set at any time
        # and cannot be switched off per request -- LiteLLM reads `fallbacks or model_fallbacks`, so
        # neither `None` nor `[]` overrides it
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/main.py#L5461-L5464). (A
        # `fallbacks` query param never gets this far: `__init__` and `_with_query_params` refuse
        # it.) `is not None` because that is LiteLLM's own test: an empty `model_fallbacks` still
        # enters its fallback runner, which re-reads the global and recurses until every call fails,
        # a successful one included
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/main.py#L644,
        # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/fallback_utils.py#L84).
        # `UnsupportedParamsError`, so the refusal fails fast and is still an `LLMError` a caller of
        # `jaz.invoke` can catch with the rest.
        if _litellm().model_fallbacks is not None:
            raise UnsupportedParamsError(
                "litellm.model_fallbacks is set, which is not supported. Set "
                "litellm.model_fallbacks = None to use this backend."
            )
        stored = self._stored_api_key(params)
        if stored is not None:
            params["api_key"] = stored
        return params

    def _stored_api_key(self, params: dict[str, Any]) -> str | None:
        """The ``~/.jaz/credentials.json`` key for this call's provider, or ``None``.

        Returns ``None`` whenever an explicit key or the provider's environment variable
        already supplies one, so the documented precedence (explicit → environment → stored
        file) holds.
        """
        # Resolved PER REQUEST rather than once in `__init__`, for FRESHNESS — not because the
        # provider varies. It does not vary in practice: `model` is
        # fixed per instance and `Agent` caches it for its lifetime, so a given backend serves one
        # provider. What varies is the FILE. `Agent.__init__` does
        # `self.llm_client = self.config.llm`, reusing the configured instance rather than
        # rebuilding it, so a key resolved at construction is the only key that instance will ever
        # have. Per-request resolution is therefore the sole reason `set_credential(...)` followed
        # by an `invoke(...)` works without restarting the console — the "takes effect from the
        # next agent run" promise in `jaz._console.set_credential`. A backend that resolves at
        # construction would not honour it (see `jaz/credentials.py`).
        #
        # Cost is not a consideration and should not become one: this runs immediately before a
        # network round-trip to an LLM, which is four to five orders of magnitude slower. Do not
        # add a cache for speed — a cache would reintroduce exactly the staleness this avoids.
        #
        # It is also why the key is injected here instead of riding this backend's open
        # `**query_params` tail, which would work mechanically (LiteLLM forwards `api_key`
        # verbatim). Putting it there would make the
        # secret ordinary construction config on the component — precisely what the credentials
        # store exists to prevent (see the executive call at the top of `jaz/credentials.py`:
        # the key must never be config *input*). Here it is read off disk at call time and dies
        # with the params dict.
        #
        # `params` already carries a configured `api_key` (`_build_params` merges the query
        # params first), so one check covers a constructed key and the precedence holds.
        if params.get("api_key"):
            return None
        litellm = _litellm()
        model = params.get("model")
        if not isinstance(model, str) or not model:
            return None
        try:
            provider = litellm.get_llm_provider(model=model)[1]
        except Exception:
            # An unroutable/unknown model is LiteLLM's error to raise on the call itself, with
            # a far better message than anything this resolution step could produce. Declining
            # to supply a key is the only sane reading here.
            #
            # This MUST stay ahead of the `validate_environment` check below: that call returns
            # `keys_in_environment: True` for a model it cannot route (nothing is missing when
            # nothing is required:
            # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L6733-L6737 and #L7122-L7123),
            # so ordering it first swallowed the unroutable case in a
            # `return None` and left this handler dead.
            return None
        if not provider:
            return None
        # What keeps the environment ahead of the file: LiteLLM reads `OPENAI_API_KEY` &c.
        # itself, so injecting unconditionally would silently DEMOTE an exported key below the
        # stored one and invert the documented order.
        #
        # KNOWN LIMITATION — this is exact only for providers authenticated by a single API-key
        # variable (openai, anthropic, gemini, …). `validate_environment` ANDs over every
        # variable a provider needs (this, and the environment read above:
        # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L6740-L6748),
        # so a multi-variable provider reports "not satisfied" while its key IS exported: with
        # `AZURE_API_KEY` set but `AZURE_API_BASE` unset, this still
        # falls through and injects the stored key, demoting the exported one. Providers whose
        # auth is not an API key at all (bedrock → AWS credentials, vertex_ai → project/location)
        # are the same shape, and there an injected `api_key` is merely inert. Tracked in #1076
        # with the rest of the multi-backend credential model; the single-key providers the store
        # is actually used for are correct.
        try:
            if litellm.validate_environment(model).get("keys_in_environment"):
                return None
        except Exception:
            return None
        # Left OUTSIDE the exception mapping in `complete`/`acomplete` on purpose: a malformed
        # `~/.jaz/credentials.json` raises `ValueError` straight out of the completion call rather
        # than becoming a `jaz.exceptions` LLM error. That is the right shape — it is a
        # configuration error, not an API failure, so mapping it to `AuthenticationError` would
        # misdescribe it, and the message already names the offending path and what is wrong with
        # it. Failing loudly here matches `load_credentials`, which refuses to treat an unparseable
        # store as "no credential".
        return resolve_credential(str(provider))

    def complete(self, messages: list[Any]) -> LLMResponse:
        litellm = _litellm()
        params = self._build_params(messages)
        handling = sys.exception()
        try:
            response = litellm.completion(**params)
        except Exception as exc:
            mapped = _mapped_exception(litellm, exc, handling)
            if mapped is not None:
                raise mapped from exc
            raise
        return self._to_response(response)

    async def acomplete(self, messages: list[Any]) -> LLMResponse:
        # Native async via litellm.acompletion, not the base's asyncio.to_thread(complete)
        # fallback — that fallback burns one blocking thread per in-flight call, which the
        # sole/default backend should not do. Shares _build_params and the exception mapping.
        litellm = _litellm()
        params = self._build_params(messages)
        handling = sys.exception()
        try:
            response = await litellm.acompletion(**params)
        except Exception as exc:
            mapped = _mapped_exception(litellm, exc, handling)
            if mapped is not None:
                raise mapped from exc
            raise
        return self._to_response(response)

    def _to_response(self, response: Any) -> LLMResponse:
        _raise_if_refused(response, self._model)

        usage = getattr(response, "usage", None)
        # LiteLLM seeds every response with an all-zero `Usage` before the provider answers
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/main.py#L5475-L5476), so a
        # body without `usage` comes back as zeros that look exactly like a reported count: the
        # response keeps no trace of the raw body to tell them apart. Zero prompt tokens is the
        # tell, since every real call sends some. Unknown, not zero, so a `cost_budget` aborts the
        # turn (BudgetPool's uncosted backstop) instead of booking it as free — the rule the user
        # set for `GitHubCopilotLLM`'s missing cost figure (the user's call, 2026-10-13).
        # TODO(#1704): a server reporting zero prompt tokens is read as reporting no usage.
        if not getattr(usage, "prompt_tokens", None):
            usage = None
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)

        extra: dict[str, Any] = {}
        # Cache *writes* go into the ATIF `cache_creation_input_tokens` bucket (Anthropic's name
        # for the count, https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
        # so downstream usage reporting is uniform across backends. LiteLLM documents the count
        # as `usage.cache_creation_input_tokens`
        # (https://docs.litellm.ai/docs/completion/prompt_caching); 1.105.0 mirrors it into
        # `prompt_tokens_details.cache_write_tokens`
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/types/utils.py#L1946-L1952),
        # and the documented field is read when that one is missing.
        cache_write = getattr(prompt_details, "cache_write_tokens", None) or getattr(
            usage, "cache_creation_input_tokens", None
        )
        if cache_write:
            extra["cache_creation_input_tokens"] = cache_write
        reasoning = getattr(completion_details, "reasoning_tokens", None)
        if reasoning:
            extra["reasoning_tokens"] = reasoning

        content = None
        choices = getattr(response, "choices", None)
        if choices:
            content = getattr(getattr(choices[0], "message", None), "content", None)
            # LiteLLM's normalized reason, in OpenAI's vocabulary, under the key
            # `GitHubCopilotLLM` uses for the same.
            finish_reason = getattr(choices[0], "finish_reason", None)
            if isinstance(finish_reason, str) and finish_reason:
                extra["finish_reason"] = finish_reason

        return LLMResponse(
            content=content,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            cached_tokens=getattr(prompt_details, "cached_tokens", None),
            cost_usd=None if usage is None else self._cost(response, usage),
            extra=extra,
            raw_response=response,
        )

    def _cost(self, response: Any, usage: Any) -> float | None:
        """The call's cost in USD, or ``None`` when it cannot be priced."""
        # LiteLLM's own cost for the call, which its `client` wrapper puts on every response it
        # returns as `_hidden_params["response_cost"]`
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/llm_response_utils/response_metadata.py#L128):
        # the amount the provider reports charging where LiteLLM's transform keeps one, else
        # `completion_cost` with the call's own params, so a user's `base_model` and a region's
        # rates apply as LiteLLM applies them
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/cost_calculator.py#L2046-L2060).
        # TODO(#1732): custom per-token prices LiteLLM registers globally, repricing other
        # backends on the same model. TODO(#1733): a LiteLLM cache hit keeps the original call's
        # cost.
        # Read rather than recomputed: calling `completion_cost` directly ignores a figure a
        # provider's transform keeps aside and reprices from the map, which on OpenRouter is not
        # what it billed. OpenRouter
        # reports its bill as `usage.cost` ("The total amount charged to your account",
        # https://openrouter.ai/docs/use-cases/usage-accounting), which its LiteLLM transform
        # keeps for this
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/openrouter/chat/transformation.py#L202-L216).
        # The helper that reads the kept figure is LiteLLM's internal, and `litellm` has no upper
        # pin, so a release without it costs only the billed amount (the map gate below then
        # applies to every call) rather than failing every call that succeeded.
        read_billed = getattr(
            getattr(_litellm(), "cost_calculator", None), _BILLED_COST_HELPER, None
        )
        hidden = getattr(response, "_hidden_params", None)
        hidden = dict(hidden) if isinstance(hidden, Mapping) else {}
        billed = _usd(read_billed(hidden)) if callable(read_billed) else None
        # Priced from the map only when `can_report_cost` said it would be, so the pre-flight
        # check and the cost agree: an Azure deployment name the map does not know was priced
        # every call (from the model its response names) while a `cost_budget` was refused
        # before the first. A billed amount needs no map, so it stands either way.
        # TODO(#1705): an unmapped model's cost is dropped where the provider reported it in a
        # field LiteLLM prices from rather than keeps (xAI's, Perplexity's).
        if billed is None and not self.model_info.can_report_cost:
            return None
        # `None`, not 0.0, when LiteLLM could not price it, so BudgetPool records the call as
        # uncosted rather than free.
        cost = _usd(hidden.get("response_cost"))
        # With a key of the user's own, which OpenRouter's usage flags as `is_byok`
        # (https://openrouter.ai/openapi.yaml, the chat usage schema), the provider bills that key
        # too, reported as `cost_details.upstream_inference_cost`; the usage-accounting page's
        # example puts a 0.95 `cost` beside a 19 upstream cost, so `cost` is then OpenRouter's
        # fee alone, and LiteLLM's figure is that fee. The upstream cost is added only then: the
        # spec also shows one on a request that is not BYOK, already inside its `cost`
        # (implementation choice, not decided by the user). Scoped to OpenRouter, the one
        # provider documenting `is_byok`.
        if (
            cost is not None
            and billed is not None
            and hidden.get("custom_llm_provider") == "openrouter"
            and getattr(usage, "is_byok", None) is True
        ):
            details = getattr(usage, "cost_details", None)
            upstream = _usd(
                details.get("upstream_inference_cost")
                if isinstance(details, Mapping)
                else getattr(details, "upstream_inference_cost", None)
            )
            cost += upstream or 0.0
        return cost

    @property
    def _base_model(self) -> str | None:
        """LiteLLM's ``base_model``, if the user set one."""
        # Read as LiteLLM reads it, from a `base_model` param or `model_info["base_model"]`
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/main.py#L5381-L5383); it is
        # LiteLLM's documented way to name the model behind an Azure deployment for cost
        # tracking (https://docs.litellm.ai/docs/providers/azure#spend-tracking-for-azure-openai-models-proxy).
        model_info = self._query_params.get("model_info")
        base_model = self._query_params.get("base_model") or (
            model_info.get("base_model") if isinstance(model_info, Mapping) else None
        )
        return base_model if isinstance(base_model, str) and base_model else None

    @property
    def _cost_model(self) -> str:
        """The price-map key ``model_info`` reads: ``base_model`` if set, else the route."""
        return self._base_model or self._model

    def _model_info(self) -> dict[str, Any]:
        """The configured model's entry in LiteLLM's map (``base_model``'s, if set) — the one
        guarded lookup.

        ``{}`` for a model LiteLLM has not mapped, and also for its ``sample_spec``
        pseudo-entry, which documents the map's schema rather than describing a model. The
        ``provider/`` prefix needs no special handling: LiteLLM resolves its own route names.
        """
        # The one map read: both non-name fields of `model_info` come from it, and `_cost` prices
        # at the same key.
        litellm = _litellm()
        model = self._cost_model
        try:
            info = dict(litellm.get_model_info(model))
        except Exception:
            return {}
        # LiteLLM's map carries pseudo-entries that document the schema *in their values* —
        # `sample_spec`'s `max_input_tokens` is the sentence describing the key, not a number
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/model_prices_and_context_window.json#L12),
        # so returning one hands a `str` to `_compute_max_input_tokens`, which divides by it.
        #
        # Keyed on the RESOLVED entry, not on `model`: LiteLLM resolves its own route names and
        # matches case-insensitively, so `litellm_proxy/sample_spec` and `SAMPLE_SPEC` both land
        # on this entry too, and a check against the configured string misses both. `info["key"]`
        # is what the lookup actually resolved to.
        #
        # The reserved set comes from LiteLLM rather than a literal, because it declares two
        # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/get_model_cost_map.py#L52;
        # `fallback_generalizations` is consumed at load and so is absent from the map today)
        # and JAZ pins `litellm>=1.105.0` with no ceiling. Falls back to the one name that is
        # reachable today if the constant moves.
        #
        # `{}` rather than raising (EXECUTIVE CALL, user, 2026-09-14). The objection is real —
        # `{}` now carries two meanings, "LiteLLM has not mapped this" and "JAZ refuses this
        # entry" — and it is accepted because raising is worse on every axis that matters here:
        #
        #   * `{}` is what this method is documented to return for a model it cannot describe,
        #     and `sample_spec` is exactly that. A raise would be a new failure mode in a getter.
        #   * A raise would not arrive early. Nothing calls `_model_info()` at configure time;
        #     it is read through `model_info` (by `ContextWindowWarning`, `BudgetPool`), so a raise fires
        #     whenever a hook happens to ask and crashes the hook rather than the model call.
        #   * The misconfiguration is already loud twice over: `model_info.can_report_cost` -> `False`
        #     means a `cost_budget` on it is REJECTED PRE-FLIGHT (the entry's `0.0` rate would
        #     otherwise have made the budget silently never trip), and the first real completion
        #     fails at the provider, since `sample_spec` is not a model anywhere.
        #
        # Making a bad model id loud at configure time is a separate feature — it would have to
        # decide what to do about every unmapped-but-real model too — not a change to this getter.
        return {} if info.get("key") in _reserved_model_keys() else info

    @functools.cached_property
    def model_info(self) -> ModelInfo:
        """The route id plus both facts from LiteLLM's map (``None``/``False`` if unmapped),
        read from ``base_model``'s entry when it is set.

        ``can_report_cost`` is ``True`` only when LiteLLM's map publishes a per-token input
        rate for the model; a model priced another way (per second or per image, such as
        ``openai/sora-2``) answers ``False``, so a ``cost_budget`` on it is rejected up front.

        Limitations: a published rate of ``0`` counts as one, so such a call is reported as
        free and a ``cost_budget`` on it never trips. And the first read is cached for the
        object's lifetime: a model registered with LiteLLM after that read is not seen by this
        backend instance.
        """
        # TODO(#1034): a published rate of 0 is taken at its word, so such a call is reported as
        # free. Requiring a published rate rather than a non-zero one is an implementation
        # choice, not decided by the user: a free model (a local `ollama/` one) is real. The
        # `openai/sora-2` entry has no per-token rate:
        # https://github.com/BerriAI/litellm/blob/v1.105.0/model_prices_and_context_window.json#L52728-L52740
        info = self._model_info()
        # Type-checked rather than trusted: the map is external data. `_model_info` already drops
        # the one entry known to carry a sentence here (`sample_spec`); this is for the next one.
        # Integral floats are real windows (LiteLLM 1.97.0's bundled map carried seven
        # `xai/grok-4-*fast*` entries as `2000000.0`), so the test is "an integral number", not "an
        # int" — and `bool` is excluded explicitly because it passes an `int` check.
        value = info.get("max_input_tokens")
        if isinstance(value, bool) or not isinstance(value, int | float):
            window = None
        else:
            window = int(value) if value == int(value) else None
        # A model LiteLLM has not mapped, or prices other than per token, reports no rate →
        # False, so a `cost_budget` set on it is rejected pre-flight rather than found
        # unenforceable mid-run. Read from the raw map entry: `get_model_info` fills a missing
        # rate in as 0 ("Defaulting to 0",
        # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L6108-L6119), so its
        # answer cannot tell an unpriced model from a free one.
        # A rate in a `tiered_pricing` tier counts too: LiteLLM prices such models per token
        # by context length (dashscope, qwen, volcengine), with no top-level rate.
        raw = _litellm().model_cost.get(info.get("key")) if info else None
        return ModelInfo(
            name=self._model,
            max_input_tokens=window,
            can_report_cost=isinstance(raw, Mapping) and _has_token_rate(raw),
        )


def _has_token_rate(entry: Mapping[str, Any]) -> bool:
    """Whether a LiteLLM price-map entry publishes a per-token input rate, tiered or not."""
    if entry.get("input_cost_per_token") is not None:
        return True
    tiers = entry.get("tiered_pricing")
    return isinstance(tiers, list) and any(
        isinstance(tier, Mapping) and tier.get("input_cost_per_token") is not None
        for tier in tiers
    )
