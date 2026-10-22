"""The SGLang backend — token-in/token-out completions over SGLang's native ``/generate``.

Where the text backends send OpenAI-format messages and let the server own the chat
template, :class:`SGLangLLM` renders the conversation to token ids itself (via a
:class:`~jaz.llm.ChatFormat`) and posts ``input_ids`` to ``/generate``,
receiving the sampled ids and per-token logprobs back. Every response carries a
:class:`~jaz.llm.TurnRecord` on ``LLMResponse.tokens`` — the exact tokens sent and
sampled — which is what makes agent runs recordable as training-grade rollouts
(see :class:`~jaz.hooks.RolloutRecorder`).

Usage::

    jaz.configure(llm=SGLangLLM(model="Qwen/Qwen3-8B",
                                base_url="http://localhost:30000"))

``chat_format`` defaults to ``HFChatFormat(model)``, which needs the ``jaz-lang[tokens]``
extra (HuggingFace ``transformers``); pass a constructed :class:`ChatFormat` to avoid the
dependency.

Reasoning models (hidden thinking traces):
    For models that emit an inline reasoning trace (Qwen3's ``<think>…</think>``), the
    trace is treated the way reasoning-model APIs treat it — it is **hidden from the
    visible message**: ``LLMResponse.content`` carries only the text after the trace, so
    the REPL never executes reasoning prose and later turns' context does not replay it.
    The token sidecar (``LLMResponse.tokens``) is unaffected: the sampled ids, logprobs,
    and ``content_text`` keep the full generation, trace included, so training still
    optimizes the reasoning tokens. Configure with ``reasoning_tags=(open, close)``
    (default ``("<think>", "</think>")``); pass ``reasoning_tags=None`` to disable and
    expose raw generations. An *unterminated* trace (generation truncated mid-thought)
    strips to an empty visible message rather than leaking a partial trace.

    Only a *leading* trace is stripped — one opening at the start of the generation,
    matching how reasoning models emit them. Known limitations of textual matching: a
    generation that literally *starts* with the open-tag string is indistinguishable from
    a trace (a mid-text literal, e.g. agent code printing ``"<think>"``, is safe and
    stays visible); a hypothetical mid-text trace is left visible; a nested trace
    (``<think>a<think>b</think>c</think>``) leaks its tail past the first close tag; and
    a template that pre-appends the open tag to the *prompt* (DeepSeek-R1 style), so the
    generation begins mid-trace with no open tag of its own, is not detected — this class
    supports open-tag-emitting families (Qwen3). If any of these bite, pass
    ``reasoning_tags=None`` and strip caller-side.

Limitations:
    * No cost reporting: local/self-hosted serving has no meaningful per-token price, so
      ``cost_usd`` is ``None`` and ``model_info.can_report_cost`` is ``False`` (a
      ``cost_budget`` on this backend is rejected pre-flight).
    * Non-streaming only: ``/generate`` is called in one shot — fine for rollout
      production; revisit if interactive use matters.
    * One event loop per instance: the async client's pooled connections bind to the loop
      they were opened on, so reusing an instance across separate ``asyncio.run`` calls can
      raise "event loop is closed" — construct a fresh instance per loop (training's single
      long-lived loop is the design point). The pools are released when the backend is collected.
"""

# Design rationale (maintainers; token-native rollouts design "Option C — messages own
# their tokens, turns own their sends", PR #1099; executive call, user, 2026-08-13). Key
# properties this backend is built around:
#
# * Fully stateless and mode-less: one instance serves all invokes concurrently, holds no
#   per-invoke state, reads no ambient context, and behaves identically whether or not a
#   RolloutRecorder is listening. ("Stateless" = no per-invoke state; the pooled HTTP
#   clients below are connection reuse, not state.)
# * Retry safety is structural: the agent calls through _complete_with_retry, so
#   complete() may run several times for one logical turn. Each attempt builds its stamps
#   and TurnRecord from its own server response and mutates nothing shared; a failed
#   attempt's objects are dropped with it. (The rejected ledger design could double-append
#   a sampled segment on retry; that hazard does not exist here.)
# * Incremental append aligns with SGLang's radix prefix cache, so the stamped-reuse path
#   is also the server's fast path. First calls and un-stamped runs full-render every
#   message — correct, just cache-cold.

from __future__ import annotations

import asyncio
import difflib
import functools
import re
import uuid
import weakref
from collections.abc import Mapping
from typing import Any

import httpx

from jaz._tokens import TOKENS_KEY, TokenStamp, TurnRecord, tokens_of

from .._component import repr_call
from ._base import BaseLLM, _require_model_id
from ._chat_format import ChatFormat, HFChatFormat
from ._exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    ContextWindowExceededError,
    LLMError,
    NotFoundError,
    RateLimitError,
    RequestAbortedError,
    ServerError,
    _class_for_status,
    _has_retry_after,
    _response_details,
)
from ._types import LLMResponse, ModelInfo, RetryPolicy

# Body markers of SGLang's over-length 400s, matched lowercased. In 0.5.17 the tokenizer manager's
# ("The input (N tokens) is longer than the model's context length", "Requested token count exceeds
# the model's maximum context length") carry the first; the scheduler's ("Input length (N tokens)
# exceeds the maximum allowed length", raised when the input exceeds the smaller of the window and
# the KV pool, less a few reserved tokens) carries the second
# (https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/tokenizer_manager.py#L1150-L1186,
# https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/utils.py#L216-L219,
# https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/schedule_batch.py#L1747-L1749).
# Prose matching is drift-prone; a missed marker falls through to BadRequestError
# (non-retryable) — mislabeled, but no retry loop.
_CONTEXT_LENGTH_MARKERS = ("context length", "exceeds the maximum allowed length")

# The API fields of SGLang 0.5.17's `SamplingParams` (srt/sampling/sampling_params.py), which
# `/generate` builds as `SamplingParams(**sampling_params)`. It takes no `**kwargs`
# (https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/sampling/sampling_params.py#L45-L82),
# and the TypeError an unknown key raises escapes the handler, which catches only ValueError,
# as a generic 500, which would be retried
# (https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/entrypoints/http_server.py#L917-L919;
# the server's only exception handlers are at #L524 and #L581). So an unknown key is refused
# at construction, where LiteLLM only warns on a near-miss: its open tail reaches providers
# jaz cannot enumerate, while this one reaches one server that
# rejects every key it does not know. Its internal fields (`stop_strs`, `is_normalized`, ...)
# are left out: the server sets them while normalizing. `n` is too: with `n > 1` the server
# returns a list of generations, which this backend does not read (`_check_sampling_params`;
# https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/io_struct.py#L445-L451).
# TODO(#1633): a field a later SGLang adds is refused until it is listed here, and a value of
# the wrong type is not checked, so it still comes back as a retried 500.
_SAMPLING_PARAMS = frozenset(
    {
        "max_new_tokens",
        "stop",
        "stop_token_ids",
        "stop_regex",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "min_new_tokens",
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "no_stop_trim",
        "stream_interval",
        "logit_bias",
        "sampling_seed",
        "custom_params",
    }
)

_DEFAULT_TIMEOUT_SECONDS = 600.0  # generation can legitimately take minutes


def _release_clients(http: httpx.Client, ahttp: httpx.AsyncClient) -> None:
    """Close both pooled clients — the finalizer's target, so it must not raise."""
    try:
        http.close()
    except Exception:
        pass
    try:
        # An async client closes on a loop; at collection there may be none running.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(ahttp.aclose())
        else:
            loop.create_task(ahttp.aclose())
    except Exception:
        pass


def _mapped_exception(exc: BaseException) -> Exception | None:
    """Translate an httpx failure into jaz's error taxonomy, or ``None`` to re-raise."""
    error = _classified(exc)
    if isinstance(exc, httpx.HTTPStatusError) and isinstance(error, LLMError):
        for name, value in _response_details(exc.response).items():
            setattr(error, name, value)
    return error


def _classified(exc: BaseException) -> Exception | None:
    """The jaz error for an httpx failure, or ``None`` to re-raise.

    The base retries only its ``retryable_exceptions``, so without this a 429, 5xx or
    transport error would fail fast as a raw httpx error, and a bad request, auth failure or
    unknown route would not reach ``except jaz.exceptions.X``. A transient failure maps to a
    retryable class, the request's or the configuration's fault to one that fails fast.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = exc.response.text
        message = f"SGLang server returned {status}: {body[:500]}"
        if status == 400 and any(m in body.lower() for m in _CONTEXT_LENGTH_MARKERS):
            return ContextWindowExceededError(message, status_code=status)
        if 300 <= status <= 399:
            # Redirects are not followed (the user's call, 2026-10-13). SGLang's `/generate` never
            # redirects: it is a plain route for POST and PUT
            # (https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/entrypoints/http_server.py#L867-L871),
            # so a 3xx comes from a proxy or a `base_url` that points at something other than the
            # server. NotFoundError, because that is a 404's diagnosis too: the route is not at
            # this URL. It fails fast, since every retry gets the same answer.
            #
            # The trade-off: a 307/308 to a working server (a proxy's http -> https upgrade) now
            # fails where the `openai` SDK and LiteLLM would follow it
            # (https://github.com/openai/openai-python/blob/v2.20.0/src/openai/_base_client.py#L816,
            # https://github.com/BerriAI/litellm/blob/v1.97.0/litellm/llms/custom_httpx/http_handler.py#L601);
            # the fix is to point `base_url` at the target, which the message names. Following
            # needs a special case for each way a followed redirect ends in an error: a
            # 301/302/303 turns the POST into a GET whose answer (a 405, a 200 login page) says
            # nothing about the redirect
            # (https://github.com/encode/httpx/blob/0.28.1/httpx/_client.py#L494-L515), and loops
            # and non-http targets fail in httpx, as `TooManyRedirects` and `UnsupportedProtocol`
            # (https://github.com/encode/httpx/blob/0.28.1/httpx/_client.py#L1686-L1689,
            # https://github.com/encode/httpcore/blob/1.0.9/httpcore/_async/connection_pool.py#L205-L213).
            # TODO(#1698): a 3xx whose `Location` httpx cannot parse never gets here: httpx raises
            # `RemoteProtocolError` while building the next request, which it does even without
            # following (https://github.com/encode/httpx/blob/0.28.1/httpx/_client.py#L985-L994),
            # so it is retried as a transport failure.
            location = exc.response.headers.get("location")
            target = f" to {str(exc.response.url.join(location))!r}" if location else ""
            return NotFoundError(
                f"SGLang server redirected the request ({status}{target}); redirects are "
                "not followed, so check base_url.",
                status_code=status,
            )
        # A 499 is a request SGLang aborted on purpose. Only SGLang 0.5.17 sends one for
        # `/generate`, for a request aborted before it reached the scheduler
        # (https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/tokenizer_manager.py#L198-L199),
        # which happens only through `abort_request`: `pause_generation(mode="abort")` or a
        # weight update with `abort_all_requests`, as an RL trainer does. SGLang added it in
        # sgl-project/sglang#32588 and reverted it in #34160; on every other release such a
        # request is not aborted at all and runs once generation resumes (TODO(#1727)), and the
        # Rust server sends 499 only on its control endpoints
        # (https://github.com/sgl-project/sglang/blob/v0.5.21/rust/sglang-server/src/api_server/common.rs#L55-L56).
        # A trainer can tell that such a request waited out the update and ran on new weights by
        # `weight_version`, which SGLang changes only when the update passes one
        # (https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/tokenizer_manager.py#L2190-L2191).
        # Found by reading the source at the tags, not by running a server (#1726). A retry
        # would keep generating after the trainer cancelled, so it is jaz's `RequestAbortedError`,
        # which is not retried by default but can be opted in by class (the user's calls,
        # 2026-10-12 and 2026-10-13). A request aborted inside the scheduler comes back as a 200
        # instead, on every release, and raises the same class (`_parsed_body`). Other backends'
        # 499 is the shared rule's retryable `ServerError`, as nothing says anyone meant it to
        # stop (see `_class_for_status`).
        if status == 499:
            return RequestAbortedError(message, status_code=status)
        # A 402 that says how long to wait is a rate limit, as on `LiteLLM` (its `_classified`
        # has the evidence): waiting clears it, where a plain 402 is an exhausted quota.
        if status == 402 and _has_retry_after(exc.response):
            return RateLimitError(message, status_code=status)
        jaz_type = _class_for_status(status) or ServerError
        return jaz_type(message, status_code=status)
    # Connect/read failures are the routine retryable case for a local server that is
    # (re)starting or briefly saturated. Every other failure to send the request or read the
    # response is the connection failing, as the `openai` SDK classifies everything but a
    # timeout that its client's `send` raises
    # (https://github.com/openai/openai-python/blob/v2.20.0/src/openai/_base_client.py#L1024-L1037);
    # `send` reads the body there as `post` does here, so that includes a body httpx cannot
    # decode. One class rather than a `ServerError` for an undecodable body is an implementation
    # choice, not decided by the user: both classes are retried, so it decides only which class a
    # `RetryPolicy.limits` key names. Nor is it an `APIResponseValidationError`: that class is for
    # a success status whose body is not a generation, and httpx raises `DecodingError` without
    # the response, whatever its status. The exception is a request httpx refuses to send
    # (`LocalProtocolError`, client-side by httpx's own definition:
    # https://github.com/encode/httpx/blob/0.28.1/httpx/_exceptions.py#L222-L228): it fails the
    # same way on every attempt, so it passes through unretried, as on `LiteLLM`
    # (`_local_failures`; implementation choice, not decided by the user).
    if isinstance(exc, httpx.LocalProtocolError):
        return None
    if isinstance(exc, httpx.TimeoutException):
        # A pause in `in_place`/`retract` mode holds the request open until the trainer resumes,
        # so a `timeout` shorter than the pause lands here and the retry re-samples the turn
        # from scratch after the pause while SGLang aborts the original on the disconnect
        # (TODO(#1770)).
        return APITimeoutError(f"SGLang request timed out: {exc}")
    if isinstance(exc, httpx.RequestError):
        return APIConnectionError(f"SGLang connection failed: {exc}")
    return None


def _check_sampling_params(params: Mapping[str, Any]) -> None:
    """Raise ``TypeError`` for the first key that is not an SGLang sampling param."""
    for key in params:
        if key == "max_tokens" or key in _SAMPLING_PARAMS:
            continue
        if key == "n":
            if params[key] == 1:
                continue
            raise TypeError(
                f"SGLangLLM() got n={params[key]!r}; it reads one generation per call, so only "
                "n=1 is supported."
            )
        # `seed` is the OpenAI spelling, too far from `sampling_seed` for difflib to suggest.
        # SGLang 0.5.17 ignores `sampling_seed` unless the server runs with
        # --enable-deterministic-inference
        # (https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/sampling/sampling_batch_info.py#L89, #L118-L124),
        # so the hint says so.
        match = (
            ["sampling_seed"]
            if key == "seed"
            else difflib.get_close_matches(key, _SAMPLING_PARAMS, n=1)
        )
        hint = f" Did you mean {match[0]!r}?" if match else ""
        if key == "seed":
            hint += " (It takes effect only on a server run with --enable-deterministic-inference.)"
        raise TypeError(
            f"SGLangLLM() got {key!r}, which is not an SGLang sampling param.{hint}"
        )


# A `/generate` response, parsed: the body, its `meta_info`, the sampled ids and their logprobs.
_Body = tuple[dict[str, Any], dict[str, Any], list[int], tuple[float, ...]]


def _parsed_body(response: httpx.Response) -> _Body:
    """Parse a 200 from ``/generate``, raising ``RequestAbortedError`` for an aborted request
    and ``APIResponseValidationError`` for one that is not a generation."""

    # A malformed 200 is retryable (an `APIResponseValidationError` is a `ServerError`), as the
    # user decided (#1622): a body truncated by a proxy can be transient. The cost is that a
    # persistent mismatch (an HTML proxy page, the wrong server) spends every retry on a full
    # generation. Real SGLang always sends `meta_info` with `return_logprob=True`
    # (https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/tokenizer_manager.py#L2215-L2219, #L2509),
    # so only a non-conforming server or proxy reaches these.
    def unparseable(reason: str) -> APIResponseValidationError:
        return APIResponseValidationError(
            f"SGLang returned an unparseable response: {reason}",
            status_code=response.status_code,
            **_response_details(response),
        )

    try:
        data = response.json()
    except ValueError as exc:  # JSONDecodeError, or a body that is not UTF-8
        raise unparseable(f"not JSON: {response.text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise unparseable(f"expected a JSON object, got {type(data).__name__}")
    if data.get("error") is not None:
        raise unparseable(f"it carries an error: {str(data['error'])[:500]}")
    meta = data.get("meta_info")
    if not isinstance(meta, dict):
        raise unparseable("no meta_info")
    # A request aborted inside the scheduler comes back as a 200 whose finish reason is
    # "abort": `/abort_request` without a message, which a pause with mode="abort" and a weight
    # update's `abort_all_requests` send, sets one with no status, and the tokenizer manager
    # raises only an abort carrying 400, 500 or 503 and returns the rest as a normal result
    # (v0.5.21: https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler.py#L5574-L5584,
    # https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/tokenizer_manager.py#L1790-L1830;
    # v0.5.17: https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/scheduler.py#L4456-L4460,
    # https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/tokenizer_manager.py#L1618-L1660).
    # For a text-only request like this backend's, every other abort in either release carries
    # one of those three statuses, so a 200 abort has none; the exception is multimodal
    # disaggregation, whose encoder aborts with its own status (a 408 on an embedding timeout,
    # https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler_components/request_receiver.py#L241-L248),
    # which an `input_ids`-only request never reaches. It raises `RequestAbortedError`, the class
    # of the 499 SGLang 0.5.17 sends for the same abort landing before the scheduler, by the
    # rule the user set on #669 for a 200 that signals failure: it raises the class a status
    # would map to rather than reach the REPL as truncated code with nothing marking it. The
    # user's rule for #1654 (2026-10-14) was to follow precedent unless the case is rare; it is
    # not, since slime aborts every unfinished request at the end of each rollout step
    # (https://github.com/THUDM/slime/blob/05e3b017fed9e02de418c2fbac9d75602a534be5/slime/rollout/sglang_rollout.py#L570-L571).
    #
    # Unlike #669's refusals, whose `body` stays `None`, the body and the 200 are kept, as
    # `unparseable` above keeps them: the body is SGLang's raw JSON with the partial generation.
    # `RolloutRecorder` records only a completed query, so the aborted turn's tokens are not in
    # the rollout, and the body alone cannot rebuild the turn: it has the partial `output_ids`
    # and logprobs but not the `input_ids` this backend sent, and the turn's stamps are recorded
    # nowhere once the query fails (TODO(#1746)); the abort `message` tells "Abort in waiting
    # queue", no partial at all, from a mid-generation abort. Every RL framework that reuses a
    # partial re-sends prompt plus partial ids after the pause lifts, which the `RetryPolicy`
    # opt-in does not do (it re-sends the original prompt on its backoff; TODO(#1768)).
    # Choosing #669 as the precedent and keeping the body are implementation choices, not
    # decided by the user.
    #
    # A grammar that rejects a sampled token aborts with the same no-status "Aborted"
    # (https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L700-L709),
    # so it raises `RequestAbortedError` too, though nobody cancelled it: TODO(#1729).
    finish = meta.get("finish_reason")
    if isinstance(finish, dict) and finish.get("type") == "abort":
        raise RequestAbortedError(
            f"SGLang aborted the request ({finish.get('message') or 'Aborted'}): a pause "
            "with mode='abort', a weight update or /abort_request cancelled it, or a grammar "
            "rejected a sampled token. The partial generation is in this error's body. Not "
            "retried by default; to retry it, add RequestAbortedError to "
            "RetryPolicy.extra_retryable_exceptions.",
            status_code=response.status_code,
            **_response_details(response),
        )
    # The output_token_logprobs triplets [logprob, token_id, text|None] are the *portable*
    # source of sampled ids + logprobs — top-level `output_ids` is newer and absent in older
    # servers; slime and AReaL parse the triplets (verl reads top-level `output_ids`).
    triplets = meta.get("output_token_logprobs")
    if not isinstance(triplets, list):
        raise unparseable("no meta_info.output_token_logprobs")
    try:
        output_ids = [int(t[1]) for t in triplets]
        logprobs = tuple(float(t[0]) for t in triplets)
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise unparseable(f"malformed output_token_logprobs: {exc}") from exc
    return data, meta, output_ids, logprobs


class SGLangLLM(BaseLLM):
    """Token-native LLM backend for SGLang's ``/generate`` API.

    Where the text backends send OpenAI-format messages and let the server own the chat
    template, this one renders the conversation to token ids itself and posts ``input_ids``,
    receiving the sampled ids and per-token logprobs back::

        jaz.configure(llm=SGLangLLM(model="Qwen/Qwen3-8B",
                                    base_url="http://localhost:30000"))

    Every response carries the exact tokens sent and sampled on ``LLMResponse.tokens``, which is
    what makes agent runs recordable as training-grade rollouts.

    ``chat_format`` defaults to :class:`HFChatFormat` for ``model``, which needs the
    ``jaz-lang[tokens]`` extra (HuggingFace ``transformers``); pass a constructed
    :class:`ChatFormat` to avoid that dependency. For models that emit an inline reasoning
    trace the trace is hidden from ``LLMResponse.content`` but kept in the token sidecar — see
    the module documentation for that and the remaining limitations.

    Args:
        model: The HuggingFace model name/path. Never sent — ``/generate`` serves whatever the
            server was launched with — it picks the default ``chat_format`` and labels the
            token record.
        base_url: Root URL of the SGLang server.
        chat_format: How to render messages to token ids. A :class:`ChatFormat`, a model
            name/path for :class:`HFChatFormat`, or ``None`` to derive one from ``model``.
        reasoning_tags: ``(open, close)`` delimiters of the hidden reasoning trace, or
            ``None`` to expose raw generations.
        retry_policy: How failed calls are retried; see :class:`RetryPolicy`.
        **query_params: SGLang sampling params (``temperature``, ``max_new_tokens``, ...),
            ``max_tokens`` as an alias of ``max_new_tokens``, and ``timeout``, the HTTP timeout
            in seconds.

    A request the server aborts (through ``/abort_request``, a pause with ``mode="abort"``, or
    a weight update that aborts in-flight requests) raises
    :class:`~jaz.exceptions.RequestAbortedError`, not retried by default, with whatever was
    generated in its ``body``. Opting the class back in through ``RetryPolicy`` re-sends the
    original prompt from scratch on the policy's backoff; it does not continue from the partial
    generation. SGLang answers a grammar failure the same way, so that raises it
    too. A request still waiting to reach the scheduler is aborted only on SGLang 0.5.17; on
    other releases it runs once generation resumes, possibly on new weights, so after a weight
    update check the response's ``tokens.sampled.weight_version``, which changes only when the
    update passes ``weight_version`` and names one version for the whole turn. A pause in
    ``mode="in_place"`` or ``mode="retract"`` holds the request open instead of aborting it; a
    ``timeout`` shorter than the pause turns it into an ``APITimeoutError``, retried as a fresh
    request after the pause.

    Raises:
        TypeError: A key in ``query_params`` is not a sampling param of SGLang 0.5.17; one
            added in a later SGLang is refused too.
        ValueError: ``base_url`` is not an ``http://`` or ``https://`` URL with a host.
    """

    # The model id is never sent: `/generate` takes `input_ids` and serves whatever the server
    # was launched with. It only picks the default chat format (`HFChatFormat(model)`) and labels
    # the record, so a wrong id is not a request that fails — with `chat_format` set explicitly
    # it is a mislabelled `TurnRecord` and nothing else. Nothing here can detect that: the server
    # holds the only ground truth about which model is loaded, and it does not report it on this
    # path.

    # Declares the token sidecar so to_wire_messages leaves TokenStamps on this backend's
    # wire list — the stamped-reuse path below reads them. Provenance stays stripped.
    extra_message_keys = frozenset({TOKENS_KEY})

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:30000",
        chat_format: ChatFormat | str | None = None,
        reasoning_tags: tuple[str, str] | list[str] | None = ("<think>", "</think>"),
        retry_policy: RetryPolicy | Mapping[str, Any] = RetryPolicy(),  # noqa: B008
        **query_params: Any,
    ) -> None:
        super().__init__(retry_policy=retry_policy)
        # The params this backend maps onto `sampling_params` on every `/generate` — its own.
        # `timeout` is the one key that is not a sampling param: it is the HTTP transport
        # timeout (sent as an API field it is ignored or rejected, #623), so it is split off here.
        params = dict(query_params)
        self._timeout: float = params.pop("timeout", _DEFAULT_TIMEOUT_SECONDS)
        _check_sampling_params(params)
        self._query_params: dict[str, Any] = params
        # This backend's own setting: the chat-format default, and the label a rollout carries.
        self._model = _require_model_id("SGLangLLM", model)
        # Checked here for a clear error: httpx reports a missing scheme as `UnsupportedProtocol`
        # (https://github.com/encode/httpx/blob/0.28.1/httpx/_exceptions.py#L208), a transport
        # error that every call would retry rather than one that names the mistake. Redirects are
        # not followed, so `base_url` is the only URL a request goes to.
        try:
            url = httpx.URL(base_url)
        except httpx.InvalidURL:
            url = None
        if url is None or url.scheme not in ("http", "https") or not url.host:
            schemeless = "://" not in base_url and not base_url.lower().startswith(
                "http"
            )
            hint = f" Did you mean 'http://{base_url}'?" if schemeless else ""
            raise ValueError(
                "SGLangLLM base_url must be an http:// or https:// URL with a host, got "
                f"{base_url!r}.{hint}"
            )
        self._base_url = base_url.rstrip("/")
        # Hidden-reasoning interface (executive call, user, 2026-08-15): the standard
        # contract for thinking models is that the trace is hidden from the output —
        # reasoning stays ON and gets gradient, but only the post-trace text is the
        # message. Three distinct surfaces, split deliberately:
        #   * execution + cross-turn context: LLMResponse.content is stripped, so the
        #     agent's REPL parse and the history it appends never see the trace (the
        #     stale-stamp guard in _prepare_stamps then re-renders the stripped text
        #     fresh, keeping trace ids out of later input_ids);
        #   * loss: the sampled TokenStamp keeps the FULL ids/logprobs/content_text, so
        #     recorded rollouts still train the reasoning tokens (mask 1).
        # Consequence: multi-turn hidden-reasoning rollouts are non-monotone BY DESIGN
        # (the next turn's context is not an extension of prev sent+sampled), so
        # Rollout.to_spans() splits them at every turn — one Sample per turn, nothing to choose.
        # Default ON: shipping the raw trace as .content fed `<think>` prose to the REPL
        # (SyntaxError every thinking turn) and tanked Qwen3-0.6B GSM8K greedy pass@1
        # 0.27→0.147 with 69% aborts (measured 2026-08-15, N=100×3); stamped-full/
        # visible-stripped restored it. Alternatives rejected: enable_thinking=False at
        # the template (kills the reasoning that helps: 0.220 but degenerate loops) and
        # /no_think (Qwen3-0.6B ignores the soft switch).
        self._reasoning_tags_spec = tuple(reasoning_tags) if reasoning_tags else None
        if reasoning_tags is not None:
            open_tag, close_tag = reasoning_tags  # 2-tuple contract; raises on misuse
            if not open_tag or not close_tag:
                raise ValueError("reasoning_tags must be two non-empty strings or None")
            # LEADING trace only (executive call, user, 2026-08-15, on review of #1177):
            # matches how reasoning models actually emit traces (Qwen3: start of the
            # generation, nowhere else — mid-text traces were never observed). Bounds the
            # literal-tag false positive to generations that *start* with the open tag; a
            # mid-text occurrence (agent code printing "<think>") stays visible. A closed
            # leading trace strips through the close tag (+ whitespace); an unterminated
            # leading open tag (truncation mid-thought) strips to end-of-text so a
            # partial trace never leaks into the visible message.
            #
            # Precedent: hiding the trace from the message is the serving-stack standard —
            # SGLang --reasoning-parser and vLLM reasoning outputs both split the
            # trace out of content (SGLang's reasoning_content, vLLM's reasoning)
            # (https://docs.sglang.ai/advanced_features/separate_reasoning.html,
            # https://docs.vllm.ai/en/latest/features/reasoning_outputs/), and the Qwen3
            # model card recommends history without thinking content
            # (https://huggingface.co/Qwen/Qwen3-0.6B). Those parsers are strictly MORE
            # permissive than this leading-tag match: they also detect a trace whose open
            # tag lives in the *prompt* (DeepSeek-R1-style templates pre-append <think>,
            # so the generation starts mid-trace) — documented here as an unsupported
            # family (module docstring) rather than guessed at.
            self._reasoning_closed: re.Pattern[str] | None = re.compile(
                r"\A\s*" + re.escape(open_tag) + r".*?" + re.escape(close_tag) + r"\s*",
                re.DOTALL,
            )
            self._reasoning_open: re.Pattern[str] | None = re.compile(
                r"\A\s*" + re.escape(open_tag) + r".*\Z", re.DOTALL
            )
        else:
            self._reasoning_closed = None
            self._reasoning_open = None
        # `chat_format` may be a ChatFormat, a tokenizer name, or None (= the model id).
        # String/None resolve lazily on first call so constructing the backend (registry,
        # config files, `import jaz`) never pays the transformers import or tokenizer
        # download — and never *requires* jaz-lang[tokens] unless the default is actually used.
        self._chat_format: ChatFormat | None = (
            chat_format if isinstance(chat_format, ChatFormat) else None
        )
        self._chat_format_spec: str | None = (
            chat_format if isinstance(chat_format, str) else None
        )
        # Instance-held pooled clients (connection reuse at hundreds of concurrent
        # rollouts). Construction is cheap and opens no sockets. Tests replace these with
        # httpx.MockTransport-backed clients. `_ahttp`'s pooled connections bind to the
        # event loop they are first used on, so one instance assumes one long-lived loop
        # (training's design point); a per-loop client keyed by asyncio.get_running_loop()
        # would lift that but adds loop-bookkeeping state to a deliberately stateless
        # backend, so the single-loop assumption is documented instead; the finalizer below
        # releases both pools when the backend is collected.
        self._http = httpx.Client(follow_redirects=False)
        self._ahttp = httpx.AsyncClient(follow_redirects=False)
        # Released when this backend is collected or at interpreter exit, so nothing has to
        # close a backend: CPython frees it the moment its last reference drops, which is when
        # an explicit close would have run. The finalizer holds the clients, not `self`.
        weakref.finalize(self, _release_clients, self._http, self._ahttp)

    def _format(self) -> ChatFormat:
        if self._chat_format is None:
            self._chat_format = HFChatFormat(self._chat_format_spec or self._model)
        return self._chat_format

    # ------------------------------------------------------------------
    # complete / acomplete
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return repr_call(
            "SGLangLLM",
            ("model", self._model),
            ("base_url", self._base_url),
            ("chat_format", self._chat_format_spec),
            ("reasoning_tags", self._reasoning_tags_spec),
            ("timeout", self._timeout),
            ("retry_policy", self._retry_policy),
            *self._query_params.items(),
        )

    def complete(self, messages: list[Any]) -> LLMResponse:
        fmt = self._format()
        stamps = self._prepare_stamps(messages, fmt)
        timeout = self._timeout
        payload = self._build_payload(stamps, fmt, self._query_params)
        try:
            response = self._http.post(
                f"{self._base_url}/generate", json=payload, timeout=timeout
            )
            response.raise_for_status()
        except Exception as exc:
            mapped = _mapped_exception(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        return self._finish(_parsed_body(response), stamps, fmt)

    async def acomplete(self, messages: list[Any]) -> LLMResponse:
        # Native async, not the base's asyncio.to_thread(complete) fallback: a blocking
        # thread per in-flight generation is untenable at training-batch scale (hundreds
        # of concurrent rollouts). Shares every non-transport helper with complete().
        fmt = self._format()
        stamps = self._prepare_stamps(messages, fmt)
        timeout = self._timeout
        payload = self._build_payload(stamps, fmt, self._query_params)
        try:
            response = await self._ahttp.post(
                f"{self._base_url}/generate", json=payload, timeout=timeout
            )
            response.raise_for_status()
        except Exception as exc:
            mapped = _mapped_exception(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        return self._finish(_parsed_body(response), stamps, fmt)

    # ------------------------------------------------------------------
    # Shared request/response helpers (sync and async cannot drift)
    # ------------------------------------------------------------------

    def _prepare_stamps(self, messages: list[Any], fmt: ChatFormat) -> list[TokenStamp]:
        """One TokenStamp per wire message, in order: reuse or fresh-render."""
        stamps: list[TokenStamp] = []
        for message in messages:
            content = message.get("content")
            stamp = tokens_of(message)
            # Stamped-reuse iff the content still matches the stamp — the stale-stamp
            # guard (runtime guard 2): a hook that edited a stamped message's content
            # after stamping must not have the old ids silently shadow the edit. The
            # mismatch re-renders under a fresh message_id, which surfaces at export as a
            # monotonicity break — the same representable event as any other history edit
            # — instead of a silent divergence between hook intent and what the model saw.
            if stamp is not None and stamp.content_text == content:
                stamps.append(stamp)
            else:
                stamps.append(
                    TokenStamp(
                        role=str(message.get("role", "")),
                        ids=tuple(fmt.render_message(message)),
                        content_text="" if content is None else str(content),
                        message_id=uuid.uuid4().hex,
                    )
                )
        return stamps

    def _build_payload(
        self, stamps: list[TokenStamp], fmt: ChatFormat, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        input_ids = list(fmt.initial_ids)
        for stamp in stamps:
            input_ids.extend(stamp.ids)
        input_ids.extend(fmt.generation_prompt_ids)
        return {
            "input_ids": input_ids,
            "sampling_params": self._sampling_params(kwargs),
            "return_logprob": True,
        }

    def _sampling_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Map jaz/OpenAI-style request kwargs onto SGLang ``sampling_params``."""
        params = dict(kwargs)
        # OpenAI spelling → SGLang spelling; an explicit max_new_tokens wins. Every other
        # key is a SamplingParams field (checked at construction) and passes through verbatim.
        if "max_tokens" in params:
            params.setdefault("max_new_tokens", params.pop("max_tokens"))
        # Forced after the merge so no config can undo them (LiteLLM's num_retries=0 precedent): the
        # RL sampling-correctness convention. no_stop_trim and skip_special_tokens=False make the
        # server's decoded text match the sampled ids (the ids carry the stop token either way, in
        # 0.5.17). The *text* the protocol parses is decoded separately with specials stripped
        # (ChatFormat.decode) — recorded ids and parsed text serve different masters.
        params["no_stop_trim"] = True
        params["skip_special_tokens"] = False
        return params

    def _finish(
        self,
        body: _Body,
        stamps: list[TokenStamp],
        fmt: ChatFormat,
    ) -> LLMResponse:
        data, meta, output_ids, logprobs = body
        finish_reason = meta.get("finish_reason")
        # Structured ({"type": "stop", ...}) in current servers, a bare string in older
        # ones — normalize to the string.
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type")
        # One version for the whole turn. A turn that straddles an update has tokens from two
        # policies; SGLang main reports them as `meta_info.weight_versions` spans, not read
        # yet (TODO(#1769)).
        weight_version = meta.get("weight_version")
        if weight_version is not None:
            weight_version = str(weight_version)

        gen_prompt = fmt.generation_prompt_ids
        content = fmt.decode(output_ids)
        # Visible message vs sidecar (see the reasoning_tags rationale in __init__):
        # .content hides the reasoning trace; the sampled stamp below keeps the full
        # decode as content_text, both because the stamp documents what its ids decode to
        # and because a stripped content_text would make the stale-stamp guard *match*
        # the stripped history text and replay the trace ids into the next context.
        visible = content
        if self._reasoning_closed is not None and self._reasoning_open is not None:
            visible = self._reasoning_closed.sub("", visible)
            visible = self._reasoning_open.sub("", visible)
        # The sampled stamp's ids are the turn's *full* contribution to a future context:
        # the generation-prompt tokens that opened it, the verbatim sampled ids (including
        # the stop token — no_stop_trim), and any trailing template separator the format
        # appends after the stop token (the "\n" after <|im_end|> in ChatML-family
        # templates; #1134). Capturing the framing here, from the call that produced it, is
        # what lets later contexts replay the turn byte-exactly, matching a full-template
        # render. The separator sits *outside* sampled_span, so the loss mask still covers
        # only the sampled ids.
        #
        # This "separator in the sequence but *outside* the loss span" split is the convention of
        # the trainers this backend exports to, so a JAZ rollout replayed there is byte-exact with
        # correct masking. verl inserts it as a non-loss boundary token ("boundary tokens ... must
        # not carry loss") in verl/utils/tokenizer/continuous_token.py
        # (https://github.com/volcengine/verl); slime and AReaL reinsert it from a full-template
        # re-render and mask it out, in slime/agent/trajectory.py (https://github.com/THUDM/slime)
        # and areal/experimental/openai/client.py (https://github.com/inclusionAI/AReaL).
        #
        # Only a natural stop leaves the end-of-turn token in output_ids; a length or
        # content-filter finish truncates mid-turn with no stop token, so the *post-stop*
        # separator does not belong — appending it would inject a phantom token matching
        # neither what was sampled nor a real full-template render of the (unfinished) turn.
        suffix = fmt.sampled_suffix_ids if finish_reason == "stop" else ()
        sampled = TokenStamp(
            role="assistant",
            ids=tuple(gen_prompt) + tuple(output_ids) + tuple(suffix),
            content_text=content,
            sampled_span=(len(gen_prompt), len(gen_prompt) + len(output_ids)),
            logprobs=logprobs,
            weight_version=weight_version,
            finish_reason=finish_reason,
        )
        record = TurnRecord(
            sent=tuple(stamps),
            sampled=sampled,
            initial=tuple(fmt.initial_ids),
        )
        extra: dict[str, Any] = {}
        if weight_version is not None:
            extra["weight_version"] = weight_version
        # Built directly: SGLang's envelope is not OpenAI-shaped, and pricing does not apply
        # to a self-hosted server.
        return LLMResponse(
            content=visible,
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            cached_tokens=meta.get("cached_tokens"),
            cost_usd=None,
            extra=extra,
            raw_response=data,
            tokens=record,
        )

    @functools.cached_property
    def model_info(self) -> ModelInfo:
        """The configured id only: no window, no cost."""
        # Local/self-hosted serving has no meaningful per-token price, and `/generate` does not
        # report the loaded model's window — so a cost_budget is rejected pre-flight and the
        # context-window hook stands down, rather than either guessing.
        return ModelInfo(name=self._model)
