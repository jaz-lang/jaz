"""The GitHub Copilot backend — completions billed to GitHub AI credits.

:class:`GitHubCopilotLLM` reaches every model GitHub Copilot serves (``gpt-5-mini``,
``claude-opus-5``, …) through the official Copilot SDK, so usage is charged to a Copilot
subscription's **GitHub AI credits** rather than to a per-vendor API key::

    jaz.configure(llm=GitHubCopilotLLM(model="gpt-5-mini"))

Model ids are the ones ``copilot`` itself lists (the console's setup wizard asks the account); they
carry no ``provider/`` prefix.

Authentication is the SDK's, in its own order: an explicit ``github_token``, then
``GITHUB_COPILOT_API_TOKEN`` (with ``COPILOT_API_URL``), then ``COPILOT_GITHUB_TOKEN`` /
``GH_TOKEN`` / ``GITHUB_TOKEN``, then a stored ``copilot`` CLI login, then ``gh``. jaz's
credential store is not consulted — the token is GitHub's, not a model provider's.

Requires the ``jaz-lang[copilot]`` extra (``github-copilot-sdk``) and a GitHub Copilot subscription.

Limitations:
    * **Sampling parameters are not available.** The Copilot runtime owns the request body, so
      ``temperature``, ``top_p``, ``max_tokens`` and friends are rejected rather than silently
      dropped. ``reasoning_effort`` is the one knob that maps through.
    * **The conversation is flattened into a single prompt.** The SDK's unit of input is one
      prompt string per turn, so every message after the system message is rendered into a
      ``[[role]]``-delimited transcript. Content that itself contains a line reading
      ``[[assistant]]`` is indistinguishable from a role marker — on the way in it is read as
      one, and on the way out the completion is cut short there, because a model that has been
      shown a transcript sometimes carries on writing it past the end of its own turn.
    * **No streaming and no token sidecar** — one prompt in, one final message out.
    * Cost is GitHub's own per-call AI-credit figure at a cent a credit, whether or not the call
      falls within the plan's included allowance. It arrives *after* the call, so a
      ``cost_budget`` is enforced on completed calls like any other backend's. A turn whose
      figure is missing, or that GitHub marks as incomplete, reports ``cost_usd=None``
      (unknown), which a ``cost_budget`` aborts on.
    * On a legacy request-based plan (an annual Copilot Pro or Pro+ plan billed in premium
      requests), the AI-credit figure is not what the call is billed, even when it is reported.
"""

# Legacy request-based billing for annual Pro and Pro+ plans:
# https://docs.github.com/en/copilot/concepts/billing/copilot-requests

# The authentication order in the docstring is the SDK's
# (https://github.com/github/copilot-sdk/blob/v1.0.11/docs/auth/authenticate.md#L302-L310).

# Why the SDK rather than an HTTP endpoint (the shape of this backend follows from it): GitHub
# publishes no OpenAI-compatible chat endpoint that spends AI credits. The credit-billed surface is
# the Copilot CLI runtime, which the SDK drives over JSON-RPC:
#
#     jaz Agent -> GitHubCopilotLLM -> copilot SDK -> (JSON-RPC) -> Copilot CLI -> model
#
# The CLI is an *agent* runtime, not an inference API, so the work here is subtracting the agent:
# `mode="empty"` + `available_tools=[]` + a `replace` system message leave a bare completion
# primitive. jaz owns the loop, the tools (the REPL) and the prompt; Copilot supplies only the
# model call. Leaving the CLI's own tools on would run a second agent loop inside every jaz turn.
#
# LiteLLM's `github_copilot/` provider was the alternative and was rejected: it authenticates by
# device flow against the Copilot *editor* token
# (https://github.com/BerriAI/litellm/blob/v1.97.0/litellm/llms/github_copilot/authenticator.py#L22-L24)
# and reports at most a price-map estimate (for a few models), never the credits GitHub charges,
# where the SDK reuses an existing `gh`/`copilot` login and returns GitHub's own per-call credit
# charge.

from __future__ import annotations

import asyncio
import functools
import importlib
import re
import threading
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._component import repr_call
from .._paths import ensure_config_dir
from ._base import BaseLLM, _require_model_id
from ._exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AuthenticationError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    LLMError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
    _class_for_code,
    _class_for_status,
)
from ._types import LLMResponse, ModelInfo, RetryPolicy

#: GitHub prices every model in AI credits and fixes one credit at a cent, so the credit charge
#: the runtime reports converts to dollars with no pricing table to drift.
#: https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
USD_PER_AI_CREDIT = 0.01

#: Credits are reported in nano-credits (``totalNanoAiu``), 1e9 to the credit by the SDK's own
#: convention (its usage-and-billing doc asks to confirm it against current billing).
#: https://github.com/github/copilot-sdk/blob/v1.0.11/docs/features/usage-and-billing.md#L666
_NANO = 1e9

#: The only per-request parameter jaz forwards to the SDK (the SDK takes others, such as
#: ``reasoning_summary`` and ``context_tier``, that jaz does not expose); any other query param
#: would be silently discarded, so it is rejected instead.
#: https://github.com/github/copilot-sdk/blob/v1.0.11/python/copilot/client.py#L2093-L2103
_SUPPORTED_REQUEST_PARAMS = frozenset({"reasoning_effort"})

# Copilot's own overflow 400 has code `model_max_prompt_tokens_exceeded`, the SDK's example of a
# CAPI 400 error code
# (https://github.com/github/copilot-sdk/blob/v1.0.21/nodejs/src/generated/session-events.ts#L7375-L7378).
# That example is documented on a telemetry event, not on `session.error`, whose `errorCode` is the
# "Fine-grained error code from the upstream provider, when available"
# (https://github.com/github/copilot-sdk/blob/v1.0.21/nodejs/src/generated/session-events.ts#L2187-L2190),
# so the code is matched in that field and in the text. User reports also show the text "prompt
# token count of N exceeds the limit of M" (microsoft/copilot-intellij-feedback#94), and Claude's
# relayed "prompt is too long: N tokens > M maximum" and "Input is too long."
# (microsoft/vscode#295987). Narrow phrases only: a bare "too long" or "max_tokens" also matched
# "tool name is too long" and "max_tokens is not supported", and a bare "exceeds the limit of"
# would match an image size limit.
_CONTEXT_LENGTH_MARKERS = (
    "context length",
    "context_length",
    "model_max_prompt_tokens_exceeded",
    "token count of",
    "prompt is too long",
    "input is too long",
)


def _import_sdk() -> tuple[Any, Any]:
    """Import the Copilot SDK, returning ``(copilot, copilot.session_events)``."""
    # Deferred and via importlib on purpose: the SDK is an optional extra, so neither the import
    # nor its types may be reachable at module-import or type-check time.
    try:
        return (
            importlib.import_module("copilot"),
            importlib.import_module("copilot.session_events"),
        )
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(
            "The copilot backend needs the GitHub Copilot SDK. Install it with "
            "`pip install jaz-lang[copilot]` (or `pip install github-copilot-sdk`)."
        ) from exc


def _text_of(content: Any) -> str:
    """Flatten one message's ``content`` to text, tolerating the multimodal list form."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI's content-parts form: keep the text parts, drop what the SDK cannot carry.
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


#: A line that is *exactly* a ``[[role]]`` transcript marker. Used both to build the flattened
#: prompt and to cut the completion short at the first marker the model emits itself.
_ROLE_MARKER_LINE = re.compile(
    r"^\[\[(?:user|assistant|system|developer|tool)\]\][ \t]*$", re.M
)

#: The same leak wearing a different costume, for reused sessions.
_SESSION_MARKER_LINE = re.compile(
    r"^(?:(?:user|assistant)<current_datetime>|(?:\[exit=-?\d+\]|\*\(\d+\.\d+s\)\*)[ \t]*$)",
    re.M,
)
# Under `reuse_session` the model is never shown jaz's `[[role]]` transcript — each send is one
# bare turn, and the conversation it reads is the session's *own* history. So when it runs past
# the end of its turn it imitates that history's format instead: a user turn opens with the
# caller's `<current_datetime>` preamble, and REPL output arrives as an `[exit=N]` line with a
# `*(1.2s)*` latency footer. `_ROLE_MARKER_LINE` never matches any of those, so without a second
# pattern the reuse path silently loses the trim that the fresh-session path has.
#
# The alternatives are anchored differently on purpose. `[exit=N]` and `*(1.2s)*` stand alone on
# their line in the observed history, so they are matched as whole lines like `_ROLE_MARKER_LINE`
# is: a bare `^` prefix would cut any completion containing a line that merely *starts* with
# `[exit=0]` — an agent echoing shell output in a heredoc or docstring, which on this benchmark
# family it does constantly. The `<current_datetime>` preamble cannot be anchored, since its line
# continues with the timestamp and closing tag, but it is a long enough literal that a line
# starting with it is not plausibly anything else. The residual cost of anchoring is a missed trim
# if a real `[exit=N]` line ever carries trailing text; that fails safe, leaving the pre-existing
# behaviour, whereas a false cut silently destroys good output.


def _conversation_turns(messages: list[Any]) -> tuple[str, list[tuple[str, str]]]:
    """Split an OpenAI-format conversation into ``(system_text, [(role, text), ...])``."""
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        text = _text_of(message.get("content"))
        if role == "system" or role == "developer":
            system_parts.append(text)
        else:
            turns.append((role, text))
    return "\n\n".join(p for p in system_parts if p), turns


def _render_turns(turns: list[tuple[str, str]]) -> tuple[str, bool]:
    """Render turns into one prompt, returning ``(prompt, flattened)``."""
    # A single user turn is sent verbatim: the transcript markers below are pure overhead for
    # the one-shot case, and every token of framing is a token the model has to read past.
    flattened = not (len(turns) == 1 and turns[0][0] == "user")
    if not flattened:
        return turns[0][1], False
    return "\n\n".join(f"[[{role}]]\n{text}" for role, text in turns), True


def _split_conversation(messages: list[Any]) -> tuple[str, str, bool]:
    """Split an OpenAI-format conversation into ``(system_text, prompt_text, flattened)``.

    System messages become the session's system message; the rest are rendered into one prompt.
    ``flattened`` is whether that rendering used ``[[role]]`` markers.
    """
    system_text, turns = _conversation_turns(messages)
    prompt, flattened = _render_turns(turns)
    return system_text, prompt, flattened


def _trim_transcript_continuation(
    content: str | None, pattern: re.Pattern[str] = _ROLE_MARKER_LINE
) -> tuple[str | None, int]:
    """Cut ``content`` at the first line matching ``pattern``, a transcript-format marker.

    Returns the trimmed text and the number of characters dropped.
    """
    # Why this exists: the flattened prompt IS a transcript, so a long conversation teaches the
    # model the transcript's own format, and it sometimes keeps writing past the end of its turn —
    # emitting `[[user]]`, a *hallucinated* next user message, and then a reply to it. Observed on
    # claude-opus-4.8 in a Terminal-Bench run: a turn whose real content was fine continued with
    # `[[user]]\n[Note: this is an automatic nudge ...]` and two invented turns. Under a code-only
    # protocol the invented prose is not valid Python, so the whole turn dies as a SyntaxError and
    # is spent for nothing.
    #
    # This is the output-side counterpart of the input-side ambiguity in the module docstring, and
    # it inherits that ambiguity: a completion that legitimately contains a line reading
    # `[[assistant]]` is truncated there. That is accepted — the marker vocabulary is fixed and
    # narrow, a whole line matching one exactly is overwhelmingly the leak, and the alternative
    # (leave it in) corrupts the turn outright. The fresh-session path applies this only when the
    # prompt actually used markers, so a single-turn call — sent verbatim, with nothing to
    # imitate — is never cut there; the reuse path calls it unconditionally, for the reason given
    # at that call site.
    #
    # A stop sequence would be the right mechanism; the Copilot SDK exposes none, so the cut is
    # made here on the returned text.
    if not content:
        return content, 0
    match = pattern.search(content)
    if match is None:
        return content, 0
    # Counted against the trimmed text, not `match.start()`: the `rstrip` drops whitespace before
    # the marker too, and that is text the caller no longer receives.
    trimmed = content[: match.start()].rstrip()
    return trimmed, len(content) - len(trimmed)


def _mapped_error(
    message: str,
    status_code: int | None,
    *,
    error_type: str | None = None,
    error_code: str | None = None,
    remediation: str | None = None,
    request_id: str | None = None,
    session_has_history: bool = False,
) -> LLMError:
    """Translate a runtime error into jaz's taxonomy so retry knows what is transient."""
    # Fed only from `SessionErrorData`, the sole channel that carries an HTTP status. The setup
    # calls (`start`, `get_auth_status`, `create_session`) instead fail with the SDK's
    # `JsonRpcError`, whose `code` is a JSON-RPC code and not a status
    # (https://github.com/github/copilot-sdk/blob/v1.0.11/python/copilot/_jsonrpc.py#L23-L30, #L385),
    # so there is nothing to map: they propagate unmapped and, not being retryable, fail fast.
    # TODO(#1361): a bad model id also lands there (it is handed to `create_session`), so it
    # fails as a raw `JsonRpcError` rather than `NotFoundError`, or reaches here as a 400 and is
    # labelled `BadRequestError`. SDK 1.0.11 defines no error code or JSON-RPC code for
    # an unknown model, so the only discriminator would be the CLI's prose.
    #
    # The SDK documents the error's category and code (`ErrorData` in SDK 1.0.11's
    # `session-events.ts`,
    # https://github.com/github/copilot-sdk/blob/v1.0.11/nodejs/src/generated/session-events.ts#L1194-L1212):
    # `error_type` is a category, its examples "authentication", "authorization", "quota",
    # "rate_limit", "context_limit" and "query", and for a rate limit or quota `error_code` is a
    # `RateLimitErrorCode` ("user_weekly_rate_limited", "rate_limited", ...) or a quota code
    # ("quota_exceeded", "billing_not_configured", ...). A category or code decides only where it
    # says more than the status: a status decides over the category, by the rule the other
    # backends share. (As of SDK 1.0.11 (CLI 1.0.79) the two never disagree: the CLI's error
    # planner, `sessionPlanRunErrorJson` in its bundled `app.js`, run across every 4xx and 5xx,
    # pairs "quota" only with 402, "rate_limit" only with 429, and "authentication" and
    # "authorization" only with 401 or 403.) With no status the category decides, and that case
    # does occur: as of SDK 1.0.11 (CLI 1.0.79) the CLI's catch around its agent loop
    # (`runAgenticLoop`) emits "authentication" with no status for a session "not created with
    # authentication info", and that must fail fast. A weekly rate limit can last days (users
    # quote messages naming a reset date, in comments on github/copilot-cli#2769; GitHub
    # documents no weekly limit), so it is a quota, as Gemini's per-day quota is; the other
    # rate-limit codes are not documented to outlast a retry, and stay rate limits.
    details: dict[str, Any] = {"code": error_code, "request_id": request_id}
    # 413 is a context overflow on the SDK's own word too: its
    # `CompactionTrigger.CONTEXT_LIMIT_RETRY` is "compaction forced by a context-limit model
    # response (e.g. HTTP 413)"
    # (https://github.com/github/copilot-sdk/blob/v1.0.11/nodejs/src/generated/session-events.ts#L253).
    if (
        error_type == "context_limit"
        or _class_for_code(error_code) is ContextWindowExceededError
        or status_code == 413
        # TODO(#1752): the CLI's no-status error for a prompt it could not compact is missed.
        or (
            status_code == 400
            and any(marker in message.lower() for marker in _CONTEXT_LENGTH_MARKERS)
        )
    ):
        return ContextWindowExceededError(message, status_code=status_code, **details)
    if (
        error_type == "policy"
        or _class_for_code(error_code) is ContentPolicyViolationError
    ):
        return ContentPolicyViolationError(message, status_code=status_code, **details)
    if error_code in _NEW_SESSION_CODES and session_has_history:
        return RateLimitError(message, status_code=status_code, **details)
    if error_code in _QUOTA_CODES:
        return QuotaExceededError(message, status_code=status_code, **details)
    # Everything else follows the rule the other backends share, a 402 included (the CLI's
    # `quota_exceeded`, premium requests used up, seen as a 402 in a user report:
    # github/copilot-cli#3431). With no error status, the SDK's `error_type` decides, then its
    # `remediation`, else the retryable `ServerError`: nothing says the request was at fault. An
    # error status in 3xx, 4xx or 5xx decides whatever the type or remediation. The type is asked
    # before the remediation (implementation choice, not decided by the user) because it names
    # what failed while the remediation names what a person could do about it: a rate limit that
    # also suggests switching account is still worth a retry.
    jaz_type = (
        (_class_for_status(status_code) if status_code is not None else None)
        or _CLASS_FOR_ERROR_TYPE.get(error_type or "")
        or _CLASS_FOR_REMEDIATION.get(remediation or "")
        or ServerError
    )
    return jaz_type(message, status_code=status_code, **details)


# Quota codes a new session can clear: `session_quota_exceeded` is the CLI's per-session
# spending cap, a 402 (as of SDK 1.0.11 (CLI 1.0.79); the code, type and status checked by running
# the CLI's error planner). When the service sends no message of its own, the CLI's fallback text
# says "You've reached the spending limit for this session. Start a new session to continue." A
# retry starts a new session, but that helps only when the session that hit the cap had spent
# across earlier turns: on the default fresh-session path the session was new and hit the cap on
# its one turn, so a new one would hit it again. So it is a retryable `RateLimitError` only for a
# reused session with history, and the fail-fast quota otherwise (the user's calls, 2026-10-12
# and 2026-10-13). Each new session can spend up to the cap again, within the retry policy's caps.
_NEW_SESSION_CODES = frozenset({"session_quota_exceeded"})

# The one "policy" error observed is the provider refusing what the request asks for, whatever
# status it came with, so it is checked before the status, as the context limit is. Were an
# org-policy block ever reported this way it would get the wrong class, though one that fails
# fast just the same. The SDK lists "policy" among the
# categories without defining it
# (https://github.com/github/copilot-sdk/blob/v1.0.21/nodejs/src/generated/session-events.ts#L2191-L2194;
# added, with `remediation` "rewind_conversation", in the snapshot for CLI 1.0.97-0,
# https://github.com/github/copilot-sdk/commit/a72e9dda2b1269e6d684bdb7a1de43f5c0c539df). Driven
# through SDK 1.0.21 with a bring-your-own-key OpenAI provider against a fake server, CLI 1.0.97
# raises it, with code `misalignment_policy_violation` and remediation "rewind_conversation", for
# that OpenAI error code, passes the provider's status through unchanged (400, 401, 403, 422, 429
# and 500 were tried), and sends the request once without retrying it. So the status says
# nothing about the error's kind: mapped by status, a 429 or 500 one would be retried. The code
# is one of OpenAI's documented Responses error codes
# (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/types/responses/response_error.py#L47-L53),
# and Codex treats it as terminal
# (https://github.com/openai/codex/blob/rust-v0.162.1/codex-rs/protocol/src/error.rs#L393-L423).
# The user asked (2026-10-13) that the class follow the error's actual semantics; this class is
# an implementation choice made on that evidence, not one the user picked. The code is in the
# shared table that `LiteLLM` and this backend read before the status
# (`_exceptions._class_for_code`); `SGLangLLM` receives no such codes.

# Codes for a limit that needs someone to act or a calendar reset, which beat the status as the
# shared cause codes do (`_exceptions._POLICY_CODES`; the user's calls, 2026-10-17): the SDK's
# `RateLimitErrorCode` for a limit that can last days (see `_mapped_error`), and its quota codes.
# `quota_exceeded` is the premium-request allowance, which resets "on the 1st of each month at
# 00:00:00 UTC" unless someone sets a budget or upgrades
# (https://docs.github.com/en/copilot/concepts/billing/copilot-requests); the one real body found
# is a 402 that cleared only once the organization raised its budget
# (github/copilot-cli#3431). `billing_not_configured` needs billing set up.
# `session_quota_exceeded` is a quota unless a new session can clear it (`_NEW_SESSION_CODES`,
# checked first). The CLI's error planner pairs the "quota" type only with a 402 (see
# `_mapped_error`), where the status rule gives the same class; the code decides if one comes
# with a 429 or no status.
_QUOTA_CODES = frozenset(
    {
        "user_weekly_rate_limited",
        "quota_exceeded",
        "billing_not_configured",
        "session_quota_exceeded",
    }
)

# The class for the SDK's `error_type` when the status gives no class: none came, or a
# non-error status did.
_CLASS_FOR_ERROR_TYPE: dict[str, type[LLMError]] = {
    "authentication": AuthenticationError,
    "authorization": PermissionDeniedError,
    "quota": QuotaExceededError,
    "rate_limit": RateLimitError,
}

# The class for the SDK's `remediation` (SDK 1.0.21's `RemediationAction`, absent before it),
# asked when neither the status nor the type gives one. Only the actions that name the failure
# unambiguously are mapped: `sign_in` is for a credential "absent, expired, or rejected" and
# `switch_account` for an account that "lacks access to the requested resource"
# (https://github.com/github/copilot-sdk/blob/v1.0.21/nodejs/src/generated/session-events.ts#L207-L222).
# `show_account` names no failure, the sandbox actions are about tools (which jaz turns off), and
# `rewind_conversation` names no cause for the block.
_CLASS_FOR_REMEDIATION: dict[str, type[LLMError]] = {
    "sign_in": AuthenticationError,
    "switch_account": PermissionDeniedError,
}


# The `AssistantTurnRetryData.reason` values a resend of the failed model call was seen with (see
# `_send`'s `collected_error`).
_RESEND_REASONS = frozenset(
    {"api_error", "rate_limit", "streaming_error", "transient_auth_error"}
)


def _session_error(error: Any, *, session_has_history: bool = False) -> LLMError:
    """A ``SessionErrorData`` in jaz's taxonomy."""
    # The request id is GitHub's own (`provider_call_id`, the `x-github-request-id` header, which
    # the SDK documents as "for correlating with server-side logs",
    # https://github.com/github/copilot-sdk/blob/v1.0.11/nodejs/src/generated/session-events.ts#L1210-L1212),
    # the one to quote to GitHub; the Copilot service's (`x-copilot-service-request-id`) is the
    # fallback when it is absent. `remediation` is read with `getattr` because SDK 1.0.11, the
    # oldest the extra allows, has no such field.
    return _mapped_error(
        f"Copilot runtime error ({error.error_type}): {error.message}",
        error.status_code,
        error_type=error.error_type,
        error_code=error.error_code,
        remediation=_enum_value(getattr(error, "remediation", None)),
        request_id=error.provider_call_id or error.service_request_id,
        session_has_history=session_has_history,
    )


def _enum_value(member: Any) -> Any:
    """The wire value of one of the SDK's ``Enum`` fields, which compare unequal to it."""
    return getattr(member, "value", member)


def _credits_known(usage: Any) -> bool:
    """Whether one ``assistant.usage`` event's AI-credit figure is the call's whole charge."""
    # SDK 1.0.21 says so itself with `ai_credits_status`: "complete" when "every observed call
    # supplied an AI-credit amount, including an explicit zero", while for "unavailable"
    # "numeric zero is not a zero-cost claim"
    # (https://github.com/github/copilot-sdk/blob/v1.0.21/nodejs/src/generated/session-events.ts#L522-L531).
    # So only "complete" counts, whatever `copilot_usage` holds: the user's call (2026-10-13)
    # that a missing figure is unknown, not $0, extended to a figure GitHub marks incomplete.
    # Without the field (SDK 1.0.11, the oldest the extra allows, or a CLI that leaves it out)
    # the figure's presence decides.
    status = _enum_value(getattr(usage, "ai_credits_status", None))
    if status is not None and status != "complete":
        return False
    return usage.copilot_usage is not None


def _filtered_error(
    usages: list[Any], model: str
) -> ContentPolicyViolationError | None:
    """``ContentPolicyViolationError`` if the turn's last model call was filtered."""
    # TODO(#1639): the filtered call's usage events report tokens that can be billed (GitHub does
    # not document whether a filtered turn costs credits), but that cost is lost once this
    # raises, since cost reaches hooks only on a returned `LLMResponse`.
    # The SDK documents `content_filter_triggered` as "whether the model response was blocked or
    # truncated by content filtering (finish_reason === 'content_filter'). For Anthropic models
    # this corresponds to a 'refusal' stop reason" (`AssistantUsageData` in SDK 1.0.11), and
    # `finish_reason` as normalized to OpenAI's vocabulary
    # (https://github.com/github/copilot-sdk/blob/v1.0.11/nodejs/src/generated/session-events.ts#L4160, #L4175).
    # Raised as `LiteLLM` raises its `content_filter`, so a filtered turn does not reach the REPL as
    # a normal one.
    if not usages:
        return None
    last = usages[-1]
    if not (last.content_filter_triggered or last.finish_reason == "content_filter"):
        return None
    return ContentPolicyViolationError(
        f"{model} stopped with finish_reason 'content_filter': the content filter blocked "
        "the prompt or the response, or the model refused it.",
        request_id=last.provider_call_id or last.service_request_id,
    )


async def _abort_quietly(session: Any) -> None:
    """Cancel the message ``session`` is still processing, if it can be."""
    # On a timeout `send_and_wait` stops waiting, and by the SDK's own docstring it "does not
    # abort in-flight agent work", so the model call could keep spending credits while the
    # retry spends again. `disconnect` sends `session.detach`, `session.destroy` before SDK 1.0.13
    # (https://github.com/github/copilot-sdk/blob/v1.0.21/CHANGELOG.md#L191,
    # https://github.com/github/copilot-sdk/blob/v1.0.21/python/copilot/session.py#L3657),
    # and no source says either cancels a message in flight. The SDK's `abort` is "Abort the
    # currently processing message in this session"
    # (https://github.com/github/copilot-sdk/blob/v1.0.21/python/copilot/session.py#L3681-L3683),
    # so it is sent first. Best-effort: on a dead CLI it fails, and that must not replace the
    # timeout. Bounded, because the SDK waits on the request with no timeout of its own and a
    # CLI that is alive but not answering would hold the call past jaz's `timeout` (the 10 s
    # is an implementation choice, not decided by the user). TODO(#1680): the `disconnect`
    # that follows is unbounded the same way.
    try:
        await asyncio.wait_for(session.abort(), timeout=_ABORT_TIMEOUT_S)
    except Exception:
        pass


_ABORT_TIMEOUT_S = 10.0


def _sdk_transport_errors() -> tuple[type[BaseException], ...]:
    """The SDK's own "the CLI is gone" exception types, or ``()`` without the SDK."""
    # `ProcessExitedError` is defined in the SDK's private `_jsonrpc` module and not exported
    # (https://github.com/github/copilot-sdk/blob/v1.0.11/python/copilot/_jsonrpc.py#L33),
    # so it is looked up rather than imported, and a later SDK that moves it degrades to the
    # other two signals in `_transport_lost` instead of breaking the backend.
    try:
        error = importlib.import_module("copilot._jsonrpc").ProcessExitedError
    except (ImportError, AttributeError):
        return ()
    return (error,)


def _transport_lost(client: Any, exc: BaseException) -> bool:
    """Whether ``exc`` means ``client``'s CLI can serve no further call."""
    # No one signal covers every way the CLI goes. The SDK fails requests in flight with its
    # `ProcessExitedError` when the CLI's stdout closes
    # (https://github.com/github/copilot-sdk/blob/v1.0.11/python/copilot/_jsonrpc.py#L290-L320);
    # a request written after that hits the closed pipe as a `ConnectionError`
    # (`BrokenPipeError`); and a CLI that dies between `send` and the idle event surfaces only as
    # a `send_and_wait` timeout, so the process itself is polled too. TODO(#1636): that turn
    # still waits out the whole timeout first. jaz's own `APITimeoutError` is a `ConnectionError`
    # too, as in the SDKs, but says nothing about the CLI, so only the poll decides it.
    if isinstance(exc, (ConnectionError, *_sdk_transport_errors())) and not isinstance(
        exc, APITimeoutError
    ):
        return True
    poll = getattr(getattr(client, "_process", None), "poll", None)
    return poll is not None and poll() is not None


def _terminate(client: Any) -> None:
    """Kill the CLI subprocess directly, for when the RPC channel is no longer usable."""
    # Reaches for the SDK's private Popen handle rather than calling its `force_stop`, which in
    # SDK 1.0.11 has two effects a discard must not have: it clears the client's connection, so
    # a concurrent `create_session` on the discarded client silently starts a new CLI that
    # nothing tracks; and on a CLI still running it stops the reader before the requests in
    # flight are failed, so they wait forever. Killing the process lets the SDK's reader fail
    # them with `ProcessExitedError`. Leaving a live child for garbage collection is how
    # orphans happen. `force_stop`:
    # https://github.com/github/copilot-sdk/blob/v1.0.11/python/copilot/client.py#L2025
    process = getattr(client, "_process", None)
    if process is not None and process.poll() is None:
        process.terminate()


class _Runtime:
    """A background event loop thread plus the one started ``CopilotClient`` living on it.

    Held apart from :class:`GitHubCopilotLLM` so a ``weakref.finalize`` can shut it down without
    keeping the backend itself alive.
    """

    # The loop is dedicated rather than the caller's, for two reasons that both bite in practice:
    # `complete()` is a *sync* method on the ABC (so it needs a loop of its own to drive), and the
    # SDK's JSON-RPC transport binds its connection to the loop it was opened on — reusing an
    # instance across separate `asyncio.run` calls would fail with "event loop is closed". One
    # long-lived loop serves the sync path, the async path and any number of concurrent invokes.

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.client: Any = None
        self._thread = threading.Thread(
            target=self._run, name="jaz-copilot-runtime", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: Any) -> Any:
        """Schedule ``coro`` on the runtime loop, returning a ``concurrent.futures.Future``."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self, *, graceful: bool = True) -> None:
        """Stop the CLI subprocess and the loop thread. Idempotent.

        ``graceful=False`` kills the subprocess instead of asking it to stop — see the comment
        on why the finalizer cannot afford the polite version.
        """
        if self.loop.is_closed():
            return
        client, self.client = self.client, None
        if client is not None:
            if graceful:
                try:
                    self.submit(client.stop()).result(timeout=30)
                except Exception:
                    # Best effort: a lingering subprocess beats an exception nobody can act on.
                    _terminate(client)
            else:
                _terminate(client)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)
        if not self.loop.is_closed():
            self.loop.close()

    def shutdown_at_exit(self) -> None:
        """Shut down without the graceful stop — the finalizer's entry point."""
        # A graceful stop is an RPC, and the SDK writes RPCs through asyncio's DEFAULT executor
        # (https://github.com/github/copilot-sdk/blob/v1.0.11/python/copilot/_jsonrpc.py#L269),
        # which CPython has already torn down by the time a `weakref.finalize` atexit callback
        # runs (threads are joined before atexit). The call cannot succeed there; it can only
        # log a traceback out of the SDK's internals. So the finalizer kills the child instead,
        # and `_close()` is the deterministic way to end a session (tests use it).
        self.shutdown(graceful=False)


class GitHubCopilotLLM(BaseLLM):
    """LLM backend that routes completions through GitHub Copilot.

    Usage is charged to a Copilot subscription's **GitHub AI credits** rather than to a
    per-vendor API key::

        jaz.configure(llm=GitHubCopilotLLM(model="gpt-5-mini"))

    Requires the ``jaz-lang[copilot]`` extra (``github-copilot-sdk``) and a GitHub Copilot
    subscription. Model ids are the ones ``copilot`` itself lists
    (the console's setup wizard asks the account) and carry no ``provider/`` prefix.

    Sampling parameters (``temperature``, ``top_p``, ``max_tokens``) are rejected rather than
    silently dropped — the Copilot runtime owns the request body, and ``reasoning_effort`` is
    the one knob that maps through. See the module documentation for authentication order and
    the remaining limitations.

    Args:
        model: The Copilot model id, as ``copilot`` lists it (``gpt-5-mini``); no
            ``provider/`` prefix.
        base_directory: Where the Copilot runtime keeps its per-session state.
            Defaults to ``<jaz config dir>/copilot``.
        github_token: GitHub token to authenticate with. Defaults to the SDK's own
            resolution (environment variables, then a signed-in CLI).
        timeout: Seconds to wait for one completion.
        reuse_session: Keep one Copilot session alive across calls instead of starting a
            fresh one per completion.
        retry_policy: How failed calls are retried; see :class:`RetryPolicy`. The Copilot CLI
            retries a failed model call itself first, so an error from a call it already
            retried is not retried again.
        **query_params: per-request params the Copilot runtime accepts (``reasoning_effort``).

    Raises:
        TypeError: A query param other than ``reasoning_effort`` is passed.
        ValueError: ``model`` is empty, or ``timeout`` is not positive.
    """

    def __init__(
        self,
        *,
        model: str,
        base_directory: str | None = None,
        github_token: str | None = None,
        timeout: float = 600.0,
        reuse_session: bool = False,
        retry_policy: RetryPolicy | Mapping[str, Any] = RetryPolicy(),  # noqa: B008
        **query_params: Any,
    ) -> None:
        super().__init__(retry_policy=retry_policy)
        # This backend's own setting: the id handed to `create_session`.
        self._model = _require_model_id("GitHubCopilotLLM", model)
        # The params this backend puts on every session it creates — its own, never the base's.
        self._query_params: dict[str, Any] = dict(query_params)
        # Rejected at construction, not per call, so a config that can never be honoured fails
        # once at `jaz.configure` rather than mid-run on the first query. `TypeError` because it
        # is a configuration error, which `jaz.exceptions` reserves the builtins for, and the
        # same refusal on `LiteLLM` raises it.
        unsupported = sorted(set(self._query_params) - _SUPPORTED_REQUEST_PARAMS)
        if unsupported:
            raise TypeError(
                f"GitHubCopilotLLM cannot forward {', '.join(unsupported)}: the Copilot runtime builds "
                f"the request body itself. Supported: {', '.join(sorted(_SUPPORTED_REQUEST_PARAMS))}.",
            )
        if not timeout > 0:
            # Written `not > 0` so a NaN is refused too. Without this every call would time out
            # at once and be retried as an `APITimeoutError`.
            raise ValueError(
                f"GitHubCopilotLLM timeout must be a positive number of seconds, got {timeout!r}."
            )
        self._base_directory = base_directory
        self._github_token = github_token
        self._timeout = timeout
        # Keep one Copilot session alive across `complete()` calls and send only the turns that
        # are new, instead of re-sending the whole conversation to a fresh session every time.
        #
        # This exists because a fresh session per call throws away the provider's prompt cache.
        # Measured on the Terminal-Bench timeout group: `cache_read_tokens` was pinned at exactly
        # 2145 on every call however long the conversation got — the fixed preamble and nothing
        # else — so by turn 15 about 20,000 tokens of history were re-prefilled uncached on every
        # turn, and median call latency rose from 3.8s at turn 0 to 20.6s past turn 15. A direct
        # probe confirmed the cause rather than inferring it: replaying one recorded conversation
        # turn by turn both ways, fresh-per-turn reads 0 cached tokens on every turn while writing
        # thousands, and a reused session reaches 75% cache reads by turn 3.
        #
        # Off by default because it trades away the strict statelessness that `complete()`
        # promises: the session, not the caller, then holds the history. The reuse path below
        # only takes effect when the incoming conversation is exactly the previous one plus new
        # turns, and falls back to a fresh session otherwise, so a hook that rewrites history
        # still gets what it asked for.
        self._reuse_session = reuse_session
        self._runtime: _Runtime | None = None
        self._runtime_lock = threading.Lock()
        self._client_lock: asyncio.Lock | None = None
        self._session_lock: asyncio.Lock | None = None
        #: The live reusable session, and the state needed to decide whether it still applies.
        self._session: Any = None
        # The system message `_session` was created with. The model and query params are fixed
        # at construction, so the system message is the only setting that can differ per call.
        self._session_key: str | None = None
        # The client `_session` was created on, so discarding that client drops the session too.
        self._session_client: Any = None
        #: Turns the live session has already seen, including its own replies as jaz recorded
        #: them. The next call must extend this exactly, or the session is discarded.
        self._session_turns: list[tuple[str, str]] = []
        self._usages: list[Any] = []
        self._errors: list[Any] = []
        self._retries: list[Any] = []

    # ------------------------------------------------------------------
    # Runtime lifecycle
    # ------------------------------------------------------------------

    def _ensure_runtime(self) -> _Runtime:
        with self._runtime_lock:
            if self._runtime is None:
                runtime = _Runtime()
                self._client_lock = asyncio.Lock()
                self._session_lock = asyncio.Lock()
                self._runtime = runtime
                # Tied to this backend's lifetime rather than to `atexit`: an eval that builds a
                # backend per config would otherwise pin every CLI subprocess it ever started
                # until the process exits.
                weakref.finalize(self, runtime.shutdown_at_exit)
            return self._runtime

    async def _ensure_client(self, runtime: _Runtime) -> Any:
        assert self._client_lock is not None
        async with self._client_lock:
            if runtime.client is not None:
                return runtime.client
            copilot, _ = _import_sdk()
            base = Path(self._base_directory or ensure_config_dir() / "copilot")
            # The runtime chdirs into it and stores session state there, so it must exist.
            base.mkdir(parents=True, exist_ok=True)
            client = copilot.CopilotClient(
                mode="empty",
                base_directory=str(base),
                working_directory=str(base),
                github_token=self._github_token,
                log_level="error",
            )
            try:
                await client.start()
                status = await client.get_auth_status()
                if not status.isAuthenticated:
                    raise AuthenticationError(
                        "Not signed in to GitHub Copilot. Run `gh auth login` (or `copilot` and "
                        "`/login`), or set COPILOT_GITHUB_TOKEN / GH_TOKEN. "
                        f"Runtime reported: {status.statusMessage!r}.",
                    )
            except BaseException:
                # The client is never cached on this path, so this is the last reference to a
                # CLI subprocess that `start` may already have spawned (it does not stop one it
                # fails after spawning).
                try:
                    await client.stop()
                except Exception:
                    _terminate(client)
                raise
            runtime.client = client
            return client

    async def _discard_client(self, runtime: _Runtime, client: Any) -> None:
        """Kill ``client``'s CLI and uncache it, so the next call starts a new one."""
        assert self._client_lock is not None
        async with self._client_lock:
            # Another call may already have replaced it; that one is live and stays.
            if runtime.client is client:
                runtime.client = None
        # A reused session lives on the client that created it, so it dies with it; left in
        # place, the next call would send on it through a new client and blame that one.
        if self._session_client is client:
            self._session = None
            self._session_key = None
            self._session_turns = []
            self._session_client = None
        _terminate(client)

    def _close(self) -> None:
        """Stop the Copilot CLI subprocess and the background loop now. Idempotent.

        Private: the finalizer registered on the runtime does this at collection or exit, so
        nothing needs to call it; tests do, to end a runtime deterministically.
        """
        with self._runtime_lock:
            runtime, self._runtime = self._runtime, None
        # Dropped without disconnecting: the session lives on the runtime's loop, which is about
        # to be torn down along with the CLI process that hosts it, so there is nothing left to
        # close politely and no loop to await the close on.
        self._session = None
        self._session_key = None
        self._session_turns = []
        if runtime is not None:
            runtime.shutdown()

    def __repr__(self) -> str:
        return repr_call(
            "GitHubCopilotLLM",
            ("model", self._model),
            ("base_directory", self._base_directory),
            ("github_token", self._github_token),
            ("timeout", self._timeout),
            ("reuse_session", self._reuse_session),
            ("retry_policy", self._retry_policy),
            *self._query_params.items(),
        )

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------

    def complete(self, messages: list[Any]) -> LLMResponse:
        runtime = self._ensure_runtime()
        return runtime.submit(self._complete(runtime, messages)).result()

    async def acomplete(self, messages: list[Any]) -> LLMResponse:
        runtime = self._ensure_runtime()
        # Bridged onto the caller's loop rather than run in a thread (the base's default), so a
        # concurrent async invoke does not consume a thread-pool slot per in-flight query.
        return await asyncio.wrap_future(
            runtime.submit(self._complete(runtime, messages))
        )

    async def _complete(self, runtime: _Runtime, messages: list[Any]) -> LLMResponse:
        client = await self._ensure_client(runtime)
        try:
            return await self._complete_on(client, messages)
        except Exception as exc:
            # The client is cached across calls, so a CLI that died would otherwise fail every
            # later call the same way. Once it is discarded the next attempt starts a new CLI, so
            # the loss is retryable, as a dropped connection is on the other backends; a CLI
            # that dies every time costs `max_retries` restarts. A timeout here is the CLI
            # dying mid-turn (the process poll is what found it), so it is a lost connection
            # too; any other error already in jaz's taxonomy is the runtime's answer and keeps
            # its class.
            if not _transport_lost(client, exc):
                raise
            await self._discard_client(runtime, client)
            if isinstance(exc, LLMError) and not isinstance(exc, APITimeoutError):
                raise
            raise APIConnectionError(f"Copilot CLI connection lost: {exc}") from exc

    async def _complete_on(self, client: Any, messages: list[Any]) -> LLMResponse:
        _, events = _import_sdk()

        if self._reuse_session and self._session_lock is not None:
            # Skipped when another call already holds the session: two conversations sharing one
            # session would interleave their histories. The loser takes the fresh path below and
            # is merely uncached, not wrong.
            if not self._session_lock.locked():
                async with self._session_lock:
                    return await self._complete_reusing(client, events, messages)

        system_text, prompt, flattened = _split_conversation(messages)

        usages: list[Any] = []
        errors: list[Any] = []
        retries: list[Any] = []

        def collect(event: Any) -> None:
            if isinstance(event.data, events.AssistantUsageData):
                usages.append(event.data)
            elif isinstance(event.data, events.SessionErrorData):
                errors.append(event.data)
            elif isinstance(event.data, events.AssistantTurnRetryData):
                retries.append(event.data)

        # A fresh session per call by default, which is what makes this backend obey `complete()`'s
        # stateless contract: jaz hands the full conversation every turn and hooks may rewrite
        # history arbitrarily, so a session that remembered its own turns would answer from a
        # history the caller had already edited. `reuse_session` opts out of that above, and pays
        # for it with the prefix checks in `_complete_reusing` that detect exactly this case.
        # Session creation is local to the CLI runtime — no model call, so no credits (as
        # observed; GitHub does not document it).
        session = await self._new_session(client, system_text, collect)
        try:
            content, response = await self._send(
                session, events, prompt, errors, usages, retries
            )
        finally:
            try:
                await session.disconnect()
            except Exception:
                # Best-effort, as in `_drop_session`: the session is abandoned either way, and
                # one on a dead CLI cannot close, so the error from trying must not replace the
                # one that ended the turn.
                pass

        filtered = _filtered_error(usages, self._model)
        if filtered is not None:
            raise filtered
        dropped = 0
        if flattened:
            content, dropped = _trim_transcript_continuation(content)
        return self._to_llm_response(content, usages, response, dropped)

    async def _new_session(self, client: Any, system_text: str, on_event: Any) -> Any:
        """A bare-completion Copilot session for ``system_text``, reporting to ``on_event``."""
        return await client.create_session(
            model=self._model,
            available_tools=[],
            system_message={"mode": "replace", "content": system_text},
            streaming=False,
            enable_session_telemetry=False,
            on_event=on_event,
            # Validated against `_SUPPORTED_REQUEST_PARAMS` at construction.
            **{k: v for k, v in self._query_params.items() if v is not None},
        )

    async def _send(
        self,
        session: Any,
        events: Any,
        prompt: str,
        errors: list[Any],
        usages: list[Any],
        retries: list[Any],
        *,
        session_has_history: bool = False,
    ) -> tuple[str | None, Any]:
        """Send ``prompt`` on ``session``, returning ``(content, response)``.

        ``errors``, ``usages`` and ``retries`` are the lists the session's event handler collects
        ``SessionErrorData``, ``AssistantUsageData`` and ``AssistantTurnRetryData`` into. The
        first error, in jaz's taxonomy, is what a failed turn raises, not retried by jaz if the
        CLI already retried the turn; a turn with no reply that the content filter does not
        explain raises ``ServerError``, and one whose model calls a tool raises
        ``APIResponseValidationError``.
        """

        def collected_error() -> Exception | None:
            if not errors:
                return None
            mapped = _session_error(errors[0], session_has_history=session_has_history)
            # The Copilot CLI retries a failed model call itself before it reports the error, and
            # jaz's own retries would multiply that: driven through a bring-your-own-key provider
            # against a local fake server (CLI 1.0.97, offline), it sent 6 requests for a 500, 502,
            # 503, 408, 499, 429, a garbled 200 and a dropped or refused connection, and 3 for a
            # 401, so one `complete()` under jaz's default `max_retries=10` could cost 66. Nothing
            # turns its retries off: no session or client option in SDK 1.0.11, no flag or setting
            # in the CLI's help, and its undocumented `COPILOT_AGENT_ERROR_CODES_TO_RETRY` changed
            # nothing when set. It reports each retry as an `assistant.turn_retry` event
            # (https://github.com/github/copilot-sdk/blob/v1.0.11/python/copilot/generated/session_events.py#L1705-L1710),
            # so an error from a turn that has one is not retried by jaz (the user's call,
            # 2026-10-17): `LLMError.is_retryable` is `False`, which stops even a class the caller
            # opted in, as the provider's own "do not retry" does. Rejected: `max_retries=0` by
            # default, which also drops the retries the CLI never made (jaz's own timeout, a lost
            # event, a tool-call turn), and reading the error's "retried N times" text, which a
            # refused connection and a 401 lack.
            # Only a retry that resent the failed call counts (the user's call, 2026-10-17): the
            # event also fires when the CLI compacts the conversation instead (`reason`
            # "forced_compaction", seen on a 413 with one request sent), which is no retry of
            # the call. The reasons are the ones seen on resends in the runs above; a reason the
            # CLI adds later is not counted, so jaz retries on top of the CLI again there, the
            # accepted cost of not mistaking other events for a resend. Rejected: counting every
            # `turn_retry`, which stops jaz's retry after a compaction the CLI never resent.
            if any(retry.reason in _RESEND_REASONS for retry in retries):
                mapped.is_retryable = False
            return mapped

        # A tool call ends the turn. jaz offers the model no tools (`available_tools=[]`; the
        # CLI then sends none), but the CLI still runs its agent loop: a model that answers with
        # a tool call anyway has it executed, fail, and fed back, and the CLI asks the model
        # again for as long as it keeps calling, the conversation growing every round. Driven
        # through SDK 1.0.21 against a fake provider that always calls a tool, CLI 1.0.97 sent
        # 319 requests in 8 s, stopped only by the timeout (#1710). So the first message that
        # carries a tool request aborts the turn, which then raises `APIResponseValidationError`
        # (a success status, but not a completion jaz asked for): retried like any garbled
        # response, so a runaway costs one request per attempt (an implementation choice, not
        # decided by the user). Text that came with the call is dropped with it rather than
        # returned: it is usually the preamble to the call, not an answer (also an
        # implementation choice). The aborted turn ends at `session.idle` with no error, so
        # `send_and_wait` returns the tool-call message and the check after it raises. The SDK's
        # own per-session credit cap (`session_limits`) was rejected as the bound: its minimum is 30
        # AI credits and it adds a `<session_limits_status>` budget note to the prompt (observed:
        # found by running Copilot CLI 1.0.97 through SDK 1.0.21 with `session_limits` set, #1711),
        # and its exhausted state waits for a user decision
        # (https://github.com/github/copilot-sdk/blob/v1.0.21/python/copilot/generated/session_events.py#L10184-L10186,
        # #L16117-L16126).
        loop = asyncio.get_running_loop()
        tool_calls: list[Any] = []
        aborts: list[asyncio.Task[None]] = []

        def watch(event: Any) -> None:
            data = event.data
            if (
                not tool_calls
                and isinstance(data, events.AssistantMessageData)
                and getattr(data, "tool_requests", None)
            ):
                tool_calls.append(data)
                # Handlers run on the loop the client started on, which `_send` runs on too
                # (https://github.com/github/copilot-sdk/blob/v1.0.21/python/copilot/_jsonrpc.py#L528-L530;
                # v1.0.11's #L421-L423), so the abort is scheduled here directly.
                aborts.append(loop.create_task(_abort_quietly(session)))

        def unrequested_tool_call() -> APIResponseValidationError:
            names = sorted(
                {
                    str(getattr(request, "name", "?"))
                    for request in tool_calls[0].tool_requests
                }
            )
            return APIResponseValidationError(
                f"{self._model} answered with a tool call ({', '.join(names)}), though jaz "
                "offers it no tools; the turn was aborted."
            )

        unsubscribe = session.on(watch)
        timeout = self._timeout
        try:
            response = await session.send_and_wait(prompt, timeout=timeout)
        except TimeoutError as exc:
            if tool_calls:
                raise unrequested_tool_call() from exc
            # A CLI that dies after `send` also lands here (see `_transport_lost`), so it is
            # reported as a timeout too; `_complete` still restarts the CLI.
            await _abort_quietly(session)
            raise APITimeoutError(f"Copilot did not answer within {timeout}s.") from exc
        except Exception as exc:
            if tool_calls:
                raise unrequested_tool_call() from exc
            # A model/HTTP failure reaches us twice: as the `SessionErrorData` event the
            # session's handler saw, and again as the bare `Exception("Session error: ...")` that
            # `send_and_wait` re-raises from it — status code discarded
            # (https://github.com/github/copilot-sdk/blob/v1.0.11/python/copilot/session.py#L1794-L1797).
            # Re-deriving the taxonomy from the event is what keeps a 429 or 5xx retryable (the
            # bare `Exception` is not) and gives a 401/403/400 its class. TODO(#1741): this
            # runs before `_filtered_error` reads the usage, so a filtered turn that also sent a
            # `session.error` jaz does not recognise as a filter would take that error's class: a
            # retried `ServerError` with no status, or its status's class.
            mapped = collected_error()
            if mapped is None:
                raise
            raise mapped from exc
        finally:
            unsubscribe()
            if aborts:
                await asyncio.gather(*aborts, return_exceptions=True)
        if tool_calls:
            raise unrequested_tool_call()
        # Reachable only for an error event that arrived outside `send_and_wait`'s own
        # subscription window (it registers its handler after `create_session` installed the
        # session's, and raises for every error it does see;
        # https://github.com/github/copilot-sdk/blob/v1.0.11/python/copilot/session.py#L1794-L1797).
        mapped = collected_error()
        if mapped is not None:
            raise mapped
        # The SDK's docs show a typical turn's `assistant.message` and `assistant.usage`, ended by
        # `session.idle`
        # (https://github.com/github/copilot-sdk/blob/v1.0.21/docs/features/streaming-events.md#L981-L1003,
        # https://github.com/github/copilot-sdk/blob/v1.0.21/docs/features/agent-loop.md#L112-L114).
        # Observed with CLI 1.0.97 through SDK 1.0.21 against a fake provider: every turn that
        # ended without an error had an `assistant.message`, an empty reply included (which also
        # gets a `session.error`, "No response was returned"), though a model call inside it may
        # have none (the CLI retried an empty cut-off reply, and those calls emitted usage alone);
        # a failed turn has no message, only its `session.error`. So
        # a turn with no reply, unless a filtered call's usage explains it (raised as
        # `ContentPolicyViolationError` by the caller), is one whose `session.error` or
        # `assistant.message` never reached jaz: the SDK parses
        # each event in a callback on its loop
        # (https://github.com/github/copilot-sdk/blob/v1.0.21/python/copilot/_jsonrpc.py#L528-L530,
        # https://github.com/github/copilot-sdk/blob/v1.0.21/python/copilot/client.py#L4951),
        # and its parser rejects an enum value it does not know
        # (https://github.com/github/copilot-sdk/blob/v1.0.21/python/copilot/generated/session_events.py#L75-L81,
        # #L110-L112), so an event carrying an enum value newer than the SDK (such as a new
        # `remediation`) is logged by asyncio and dropped, and `send_and_wait` returns `None` at
        # the idle event. Retryable, as nothing says the request was at fault (implementation
        # choice, not decided by the user).
        # TODO(#1706): the lost error's content is gone, and a request that loses it every time
        # spends `max_retries` attempts.
        if response is None and _filtered_error(usages, self._model) is None:
            raise ServerError(
                "Copilot ended the turn without a reply; the event that would have explained it "
                "was probably lost (see the asyncio log)."
            )
        content = (
            response.data.content
            if response is not None
            and isinstance(response.data, events.AssistantMessageData)
            else None
        )
        return content, response

    async def _complete_reusing(
        self, client: Any, events: Any, messages: list[Any]
    ) -> LLMResponse:
        """Answer from the live session when the conversation merely extends what it has seen."""
        # Callers must hold `self._session_lock`: the usage buffers below are per-instance, not
        # per-call, because a reused session's event handler is registered once at creation and
        # cannot be re-bound per turn.
        system_text, turns = _conversation_turns(messages)

        # The reuse test. `_session_turns` holds what this session was actually told, including
        # the replies exactly as they were handed back to jaz, so an edited or truncated history
        # fails the prefix check and gets a fresh session — which is how `complete()`'s stateless
        # contract survives being answered from session-held state.
        extends = (
            self._session is not None
            and self._session_key == system_text
            and len(turns) > len(self._session_turns)
            and turns[: len(self._session_turns)] == self._session_turns
        )
        # Reported on the response because a reuse path that rebuilds every turn is otherwise
        # invisible: it returns the same answers and only shows up in the bill. Recorded from the
        # second call on — the first has no session to discard, so it is not a degradation.
        restarted = self._session is not None and not extends
        if not extends:
            await self._drop_session()
            self._session_key = system_text

        if self._session is None:

            def collect(event: Any) -> None:
                if isinstance(event.data, events.AssistantUsageData):
                    self._usages.append(event.data)
                elif isinstance(event.data, events.SessionErrorData):
                    self._errors.append(event.data)
                elif isinstance(event.data, events.AssistantTurnRetryData):
                    self._retries.append(event.data)

            self._session_client = client
            self._session = await self._new_session(client, system_text, collect)
            self._session_turns = []

        new_turns = turns[len(self._session_turns) :]
        prompt, flattened = _render_turns(new_turns)
        self._usages.clear()
        self._errors.clear()
        self._retries.clear()

        try:
            content, response = await self._send(
                self._session,
                events,
                prompt,
                self._errors,
                self._usages,
                self._retries,
                # Whether this session already spent on earlier turns: only then can a new one
                # clear its spending cap (see `_NEW_SESSION_CODES`).
                session_has_history=bool(self._session_turns),
            )
        except BaseException:
            # The session's history is now unknown — the send may have landed, may not have — so
            # it cannot be trusted to prefix-match anything. Drop it and let the next call rebuild.
            await self._drop_session()
            raise

        filtered = _filtered_error(self._usages, self._model)
        if filtered is not None:
            # The session remembers a turn jaz will not record, so it is rebuilt next time.
            await self._drop_session()
            raise filtered

        dropped = 0
        # Session markers are checked unconditionally, unlike the fresh-session path's
        # `flattened` gate: a reused send is usually a single bare turn, but the model can read
        # the whole conversation in the session's own history, so there is always a format for it
        # to continue. `[[role]]` is still checked as a fallback for the first send, which does
        # carry the flattened transcript.
        content, dropped = _trim_transcript_continuation(content, _SESSION_MARKER_LINE)
        if not dropped and flattened:
            content, dropped = _trim_transcript_continuation(content)

        if dropped:
            # The session remembers the untrimmed reply while jaz records the trimmed one, so the
            # two histories have diverged and no later prefix check could detect it. Rebuilding is
            # the only safe move; it costs one uncached turn.
            await self._drop_session()
        else:
            self._session_turns = [*turns, ("assistant", content or "")]

        return self._to_llm_response(
            content, list(self._usages), response, dropped, session_restarted=restarted
        )

    async def _drop_session(self) -> None:
        """Disconnect the reusable session, if any, and forget its history."""
        session, self._session = self._session, None
        self._session_turns = []
        if session is not None:
            try:
                await session.disconnect()
            except Exception:
                # Best-effort: the session is being abandoned either way, and a failure to close
                # a dying session must not mask the error that led here.
                pass

    @staticmethod
    def _to_llm_response(
        content: str | None,
        usages: list[Any],
        raw: Any = None,
        dropped_chars: int = 0,
        session_restarted: bool = False,
    ) -> LLMResponse:
        """Fold the turn's ``assistant.usage`` events into one :class:`LLMResponse`.

        Summed rather than taken from the last event because one prompt can span several model
        calls (a server-side retry, an auto model switch), and every one of them is billed.
        """
        # One usage event per model call:
        # https://github.com/github/copilot-sdk/blob/v1.0.11/docs/features/usage-and-billing.md#L33
        prompt_tokens = sum(u.input_tokens or 0 for u in usages)
        completion_tokens = sum(u.output_tokens or 0 for u in usages)
        cached_tokens = sum(u.cache_read_tokens or 0 for u in usages)
        cache_write_tokens = sum(u.cache_write_tokens or 0 for u in usages)
        reasoning_tokens = sum(u.reasoning_tokens or 0 for u in usages)
        # Unknown unless every model call in the turn reported its charge: a missing figure may
        # be a real charge or a dropped one, so the turn's cost is `None` rather than a sum that
        # leaves it out.
        credits = (
            sum(u.copilot_usage.total_nano_aiu for u in usages) / _NANO
            if usages and all(_credits_known(u) for u in usages)
            else None
        )
        # The user's call (2026-10-13): no usage event, or one without `copilot_usage`, is an
        # unknown cost, as `LiteLLM` reports a call it cannot price, so a `cost_budget` aborts the
        # turn. As of SDK 1.0.11 (CLI 1.0.79) the CLI itself counts a missing figure as 0 AI
        # credits wherever it charges usage (its `assistant.usage` and compaction handlers in the
        # bundled `app.js`, which feed its session spending cap). jaz differs because a missing
        # figure does not show the call was free, and booking $0 would let the budget silently
        # never charge a call that was billed. SDK 1.0.21 later documented the same reading for
        # its `ai_credits_status` (see `_credits_known`). Requiring every event rather than any
        # (so a turn with one figure missing is unknown too) is an implementation choice, not
        # decided by the user: a partial sum understates the turn the same way.

        extra: dict[str, Any] = {}
        if credits is not None:
            extra["copilot_ai_credits"] = credits
        if dropped_chars:
            # Recorded rather than logged: it is the one thing a trajectory cannot show after the
            # fact, since the dropped text never reaches the protocol or the history.
            extra["copilot_dropped_continuation_chars"] = dropped_chars
        if session_restarted:
            # Only recorded when it happened, so a healthy reuse run adds nothing to every step
            # while a run that rebuilds on every turn is visible as a flag on every turn.
            extra["copilot_session_restarted"] = True
        if cache_write_tokens:
            extra["cache_creation_input_tokens"] = cache_write_tokens
        if reasoning_tokens:
            extra["reasoning_tokens"] = reasoning_tokens
        if len(usages) > 1:
            # How many billed model calls one `complete()` actually cost. The sums above already
            # fold them together, which makes a retry storm invisible: a Terminal-Bench turn
            # reported 32,295 completion tokens against 377 characters of returned content and
            # took 274s, and without this count there is no way to tell an enormous single
            # response from a dozen silently retried ones. Only recorded when it exceeds 1, so
            # the common case adds nothing to every trajectory step.
            extra["copilot_model_calls"] = len(usages)
        if usages:
            # The id actually served, which `model="auto"` and auto-switch make differ from the
            # one requested.
            extra["copilot_model"] = usages[-1].model
            if usages[-1].finish_reason:
                extra["finish_reason"] = usages[-1].finish_reason

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=None if credits is None else credits * USD_PER_AI_CREDIT,
            extra=extra,
            # The assistant-message event, not the usage event: `raw_response` is defined as the
            # payload a protocol reads for refusal / finish_reason / tool_calls, so it has to be
            # the one carrying the message. The event SDK has no single "raw response" object,
            # and nothing is lost by the choice — the usage events' billing and served-model
            # fields are already folded into `extra` and the token counts above.
            raw_response=raw,
        )

    @functools.cached_property
    def model_info(self) -> ModelInfo:
        """The configured id; cost is reportable, as GitHub reports each call's cost.

        A turn whose figure is missing still reports ``cost_usd=None``.
        """
        # `can_report_cost` stays `True` (an implementation choice, not decided by the user). It
        # answers "will this backend set `cost_usd` for this model", which BudgetPool asks before
        # the first call to reject an unenforceable `cost_budget` outright. For Copilot the answer
        # does not depend on the model: the figure comes per call from the service, so `False`
        # would forbid a `cost_budget` on every Copilot run. A call that does arrive without one
        # is caught per turn instead, by BudgetPool's abort on `cost_usd is None`.
        # No input-token limit: a bare `gpt-5-mini` would resolve in LiteLLM's map to OpenAI's
        # window, which is not what this backend serves.
        return ModelInfo(name=self._model, can_report_cost=True)
