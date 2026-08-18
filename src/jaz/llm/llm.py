"""The LLM backend: one class per backend, doing the whole job.

A :class:`BaseLLM` owns everything about talking to one model service — the HTTP call, error
mapping, retry, cost accounting and model metadata. It is selected by a single tag
(``"litellm"``, ``"sglang"``, ``"rlm"``, plus anything ``@register_llm``\\ ed) and built from
that tag's params: ``LLM_REGISTRY[tag].from_dict(params)``.

A backend declares its ``__init__`` (whose signature declares its config surface) and implements
``complete``; :meth:`BaseLLM.finalize` gives it cost accounting and the base gives it retry.
"""

# History (executive call, user, 2026-08-07): this class is a merge of two. JAZ used to split the
# job across an ``LLMClient`` (retry, cost, model info; returned an ``LLMResponse``) and a
# ``Provider`` beneath it (the HTTP call; returned a ``CompletionResponse``), with
# ``DefaultLLMClient`` delegating between them. The split was never a real seam: one ``llm`` config
# field resolved into two constructor layers (``create_llm`` hardcoded which keys went where);
# backends outside that pairing got no retry or cost accounting; and response normalization was
# duplicated across ``complete``/``acomplete`` and had drifted (the async copy dropped
# ``cached_tokens`` and the ``extra`` bag). ``Provider``, ``register_provider``, ``PROVIDER_REGISTRY``
# and ``DefaultLLMClient`` were removed outright rather than aliased — keeping a second name for one
# concept is what let the layers drift — and the client seam was explicitly not yet public API (#984).

from __future__ import annotations

import asyncio
import difflib
import logging
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from jaz.exceptions import UnknownRequestParamWarning
from jaz.tokens import TurnRecord

from .base import CompletionResponse
from .exceptions import (
    AuthenticationError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
    UnsupportedParamsError,
)
from .pricing import compute_cost
from .pricing import get_model_info as _get_model_info

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Normalized response from an LLM completion call.

    The metric fields are named and defined as in ATIF v1.7 ``MetricsSchema``:
      * ``prompt_tokens`` — total input tokens INCLUDING cached tokens
      * ``cached_tokens`` — subset of ``prompt_tokens`` that hit the cache
      * ``completion_tokens`` — total output tokens INCLUDING reasoning
        and tool-call tokens (matches OpenAI's billing definition; for
        reasoning models the ``reasoning_tokens`` sub-count is surfaced
        in ``extra``)
      * ``cost_usd`` — cost of this call in US dollars
      * ``extra`` — provider-specific metrics (e.g. ``reasoning_tokens``,
        ``cache_creation_input_tokens`` for Anthropic), and where a backend
        reports them, ATIF's optional ``prompt_token_ids``,
        ``completion_token_ids`` and ``logprobs``

    ``content`` and ``raw_response`` are not metrics and have no ATIF counterpart.
    There is no ``total_tokens``: sum ``prompt_tokens`` and ``completion_tokens``.
    """

    # Aligned field-for-field with ATIF v1.7 MetricsSchema (executive call, user,
    # 2026-08-12) rather than either adopting the format's schema wholesale or letting the
    # two drift. Full unification was never reachable — this is a *response*, not a metrics
    # record, so `content`/`raw_response` keep it a strict superset of MetricsSchema.
    #
    # `cost` became `cost_usd` to match ATIF: the old name never said what currency it was
    # in, and both emitters already renamed it on the way out.
    #
    # `total_tokens` was REMOVED. It carried nothing its parts did not — every backend
    # either forwarded a provider number that the OpenAI usage contract defines as
    # `prompt + completion` (openai, litellm) or derived exactly that sum itself (anthropic,
    # rlm) — and ATIF has no counterpart. It could also contradict its own parts: OpenAI's
    # read zero-filled a missing count while LiteLLM's gave None, so the same unreported
    # usage surfaced as `total_tokens=0` beside nonzero parts on one backend and None on the
    # other. Consumers now sum at the point of use. `Budget` still records a total because
    # its cost records are an on-disk format the eval tooling reads.
    #
    # ATIF's remaining three metrics (`prompt_token_ids`, `completion_token_ids`,
    # `logprobs`) are deliberately NOT fields here. They are optional, provider-rare, and by
    # far the largest payload in a response — per-token arrays — so promoting them would
    # write a retention cost into the contract every backend nominally implements. They ride
    # in `extra` under their ATIF names (`extras_merge` already standardizes on those
    # spellings), and the two ATIF emitters hoist them into top-level `metrics`, which is
    # the layer that actually owes the format conformance.

    content: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = field(default=None, repr=False)
    # Per-turn token record from a token-native backend; None for text backends.
    #
    # A first-class field rather than `extra` keys — unlike the per-token ATIF arrays above,
    # this is a cross-provider *contract* (any token-native backend returns one, and the
    # agent's stamp-back and RolloutRecorder key off it), and `extras_merge`'s additive merge
    # rules are wrong for token sequences (extras_merge.py's own docstring anticipates
    # exactly this). Pointer cost only: the record shares the per-message stamp objects.
    tokens: TurnRecord | None = None


# Retry defaults — the class-attribute fallbacks used when a caller constructs a backend
# without passing retry params (e.g. ace_baseline.py). Retry config otherwise rides in the
# backend's ``llm.params`` (``max_retries``, ``retry_wait_*``, ``retry_policy``); there is no
# separate config-layer copy to keep in sync.
#
# ``max_retries`` is EXTRA retries after the first call (the LLM-SDK convention — OpenAI /
# Anthropic / LangChain / … all count this way), NOT total attempts: ``max_retries=2`` → 3
# calls. Default 10 (→ 11 attempts): bare client SDKs default to 2, but an AGENT framework
# running long autonomous loops wants more per-call resilience (a transient blip mid-run is
# expensive), so this matches the Claude Agent SDK / Claude Code default of 10 — its closest
# peer — rather than the client-SDK 2 or the more centrist agent default of ~3-4 (OpenHands/DSPy).
# Near-identical to the prior behaviour too (was ``retry_max_attempts=10`` total = 9 retries;
# rounded up to a clean 10).
_DEFAULT_MAX_RETRIES = 10
_DEFAULT_WAIT_MULTIPLIER = 1.0
_DEFAULT_WAIT_MIN = 4.0
_DEFAULT_WAIT_MAX = 60.0


# Common cross-provider request parameters — the reference the typo check matches against. NOT an
# allow-list: an unrecognized key is still forwarded (the ``**request_defaults`` tail is an open
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


# The ``src/jaz`` root (this file is ``src/jaz/llm/llm.py``). Used to skip ALL internal jaz frames
# when attributing the misspelled-param warning, so it points at the caller's construction /
# ``jaz.configure`` site regardless of how many backend ``__init__``\\ s forwarded via
# ``super().__init__`` — a fixed ``stacklevel`` can't, since real backends add a forwarding frame
# and every one would then share a single internal lineno (collapsing distinct call sites under
# Python's once-per-location dedup).
_JAZ_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _warn_on_misspelled_request_params(
    request_defaults: Mapping[str, Any], backend: str
) -> None:
    """Warn once per kwarg that looks like a misspelled request parameter.

    A key is flagged only when it is a *close* match to a known param (``difflib`` ratio ≥ 0.8)
    and is not itself known — the "looks like a typo of `temperature`" case. Unknown keys with no
    close match are assumed to be intentional provider extras and pass silently. The kwarg is
    forwarded either way; this only surfaces the likely mistake.
    """
    known = _KNOWN_REQUEST_PARAMS
    for key in request_defaults:
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


def _default_retry_logger(max_attempts: int) -> Callable[[RetryCallState], None]:
    """A ``before_sleep`` callback that logs each retry at WARNING — the default
    for the ``*_with_retry`` wrappers when the caller supplies no ``on_retry``."""

    def _log(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "LLM call failed (%s: %s), retrying (attempt %d/%d)...",
            type(exc).__name__,
            str(exc)[:200],
            retry_state.attempt_number,
            max_attempts,
        )

    return _log


class BaseLLM(ABC):
    """One LLM backend: the API call, retry, cost accounting and model metadata.

    Subclass this and register it to add a backend::

        @register_llm("mybackend")
        class MyLLM(BaseLLM):
            def __init__(self, base_url: str | None = None, **retry):
                super().__init__(**retry)
                self.base_url = base_url or os.environ["MYBACKEND_API_BASE"]

            def complete(self, model, messages, **kwargs) -> LLMResponse:
                ...  # call the API, then: return self.finalize(response, model)

    Declaring ``__init__`` is all the config wiring a backend needs: its named parameters are
    what authored config can set, and the ``**request_defaults`` tail takes everything else as a
    per-request default that rides on the wire.

    To read a key from the shared credentials store (``~/.jaz/credentials.json``), call
    :func:`jaz.credentials.resolve_credential` with the provider's tag — the default
    :class:`LiteLLM` backend does this per request, keyed by the provider it routes the model to,
    while the in-tree :class:`OpenAILLM` / :class:`AnthropicLLM` classes do it once in ``__init__``
    (one vendor each). A backend that never calls it does not read the store, so a key stored under
    its tag is inert. Resolve it *below* an explicitly passed ``api_key`` and below the provider's
    environment variable, and hold it outside ``request_defaults`` so the secret never becomes
    construction config.

    **Retry** is not a framework layer wrapped around backends — ``complete_with_retry`` /
    ``acomplete_with_retry`` are ordinary methods here, and the retry settings are ordinary
    construction fields of this base class, being the defaults its own ``complete_with_retry``
    reads. That is what lets the config layer route them with no special knowledge: they are
    discovered from this ``__init__`` signature exactly as a backend's ``base_url`` is discovered
    from its own. ``max_retries`` carries the LLM-SDK name unprefixed (the count is the headline
    knob every caller recognises); the backoff-tuning siblings stay ``retry_``-prefixed
    (``retry_wait_multiplier``/``retry_wait_min``/``retry_wait_max``/``retry_policy``) so they
    read as retry settings among flat siblings like ``model`` and ``temperature``.

    **Cost accounting** is :meth:`~BaseLLM.finalize`, which every backend calls on its wire response —
    so pricing is written once rather than per backend, and a new backend is priced for free.

    The one part that cannot be construction-time is the retry *observer*: ``Agent`` passes an
    ``on_retry`` callback carrying ``invoke_id``/``iteration`` to dispatch ``LLMQueryRetry``,
    which is per-call observability rather than policy.
    """

    # Declared as *class* attributes, not only assigned in ``__init__``, so a subclass that
    # defines its own ``__init__`` without calling ``super().__init__()`` still resolves
    # them. Subclassing an ABC and forgetting the super call is common enough — and the
    # failure would be an ``AttributeError`` deep inside a retry, far from the cause — that
    # the settings must not depend on cooperative construction.
    #: Retries after the first call (the LLM-SDK convention: OpenAI/Anthropic/… all count
    #: *extra* retries, not total attempts). ``max_retries=2`` → up to 3 calls; ``0`` = no retry.
    #: Must be ``>= 0``. (``retry_wait_*`` / ``retry_policy`` tune the backoff and per-error budget.)
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_wait_multiplier: float = _DEFAULT_WAIT_MULTIPLIER
    retry_wait_min: float = _DEFAULT_WAIT_MIN
    retry_wait_max: float = _DEFAULT_WAIT_MAX
    retry_policy: dict[str, int] | None = None

    #: The model id this backend serves. Required — there is no default; ``get_model`` raises
    #: when it is unset rather than letting ``"model": ""`` reach the API.
    model: str = ""

    # No default model, by executive call (#1086): "an LLM with no model" is underspecified, and a
    # silent ``gpt-5-mini`` fallback would pin a model nobody wrote and price/describe runs as it.
    # (The *backend* now does default — to ``litellm``, v1's sole backend
    # (design/design_features/litellm_sole_backend_v1.md) — because a single registered backend
    # removes the "rides on whichever is first" ambiguity #1086 fought; the *model* has no such safe
    # default, so it stays required.) The unset value is ``""`` (not a `Config`-supplied default:
    # groups replace rather than merge, so
    # a config-layer default would apply only to a config nobody touched and come out unset the
    # moment `configure(llm={...})` builds a new backend). The empty string is caught by the
    # existing `get_model` guard at `Agent.__init__`; the authored-data path (`build_component`)
    # additionally requires `model` up front so a YAML/settings block fails at build time. This
    # matches DSPy ("No LM is loaded") and LangChain (no auto-picked model) — a bare
    # `jaz.invoke("hi")` with nothing configured now raises rather than quietly calling gpt-5-mini.
    # A backend that genuinely does serve a canonical model may still override this with a literal.

    #: The tag ``@register_llm`` registered this class under; ``None`` for an unregistered
    #: one. Declared with the other class attributes so the class surface reads in one place.
    registry_tag: ClassVar[str | None] = None

    #: Reserved sidecar keys (see ``jaz.provenance.RESERVED_KEYS``) this backend reads and
    #: therefore receives on the wire; every other reserved key is stripped at egress.
    #: Default: none — a text backend's wire list carries no internal keys.
    #
    # A per-class declaration like ``registry_tag`` (and a class attribute for the same
    # reason as the retry fields above: it must resolve even when a subclass skips
    # ``super().__init__()``). Text backends drop ``messages`` straight into their request
    # payloads, so nothing internal may survive egress for them; a token-native backend
    # (``SGLangLLM``) declares the token key so its stamps reach ``complete()``. Keying the
    # strip by backend declaration keeps the single egress choke point
    # (``to_wire_messages``) while letting each backend opt in to exactly the sidecars it
    # consumes. Provenance is declared by no backend, so it is stripped for all.
    consumes_internal_keys: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        model: str | None = None,
        max_retries: int | None = None,
        retry_wait_multiplier: float | None = None,
        retry_wait_min: float | None = None,
        retry_wait_max: float | None = None,
        retry_policy: dict[str, int] | None = None,
        **request_defaults: Any,
    ) -> None:
        # `model` and the open `**request_defaults` tail (`temperature`, `reasoning_effort`,
        # `max_tokens`, `service_tier`, ...) live on the backend now rather than being re-derived
        # per call from a config bag.
        #
        # This is what replaced `split_llm_params`, which partitioned one flat `llm.params` map
        # into a construction half and a request half by introspecting `__init__`. That existed
        # only because the config held a *tag plus a bag*, so something had to decide which keys
        # reached the constructor and which rode the wire. It carried an explicit exception for
        # `model` — "always a request param even if a backend names it in `__init__`", because
        # binding it at construction would freeze it — and that exception is what made the split
        # impossible to remove incrementally: adding `model` here while the split still existed
        # put it in `construction_keys()`, and the instance-tag branch then filtered it out of
        # the request, so no model reached the wire at all.
        #
        # With the configured backend itself in the config, changing the model means constructing
        # a backend, which is what an override now does anyway — so the exception has nothing
        # left to protect.
        #
        # `complete`'s signature is deliberately UNCHANGED: it still takes `model` per call.
        # Freezing it into the instance would break every out-of-tree backend and the README's
        # worked example for no gain. The caller reads `llm.model` and passes it, exactly as it
        # passed a bag-derived one before.
        # Retry values are checked here rather than by the config layer, which used to do it
        # on this class's behalf. A bad one fails far from its cause otherwise. ``max_retries``
        # counts EXTRA retries, so ``0`` is valid (no retry); only a negative — which would abort
        # before the first call even completes — is rejected.
        if max_retries is not None and (
            not isinstance(max_retries, numbers.Integral) or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        for name, value, inclusive in (
            ("retry_wait_multiplier", retry_wait_multiplier, False),
            ("retry_wait_min", retry_wait_min, True),
            ("retry_wait_max", retry_wait_max, False),
        ):
            if value is None:
                continue
            if not isinstance(value, numbers.Real):
                raise ValueError(f"{name} must be a number")
            if float(value) < 0 or (not inclusive and float(value) == 0):
                raise ValueError(
                    f"{name} must be " + ("non-negative" if inclusive else "positive")
                )
        if retry_policy is not None:
            if not isinstance(retry_policy, Mapping):
                raise ValueError("retry_policy must be a dict mapping str -> int")
            for exc_name, limit in retry_policy.items():
                if not isinstance(exc_name, str):
                    raise ValueError(
                        "retry_policy keys must be exception class names (str)"
                    )
                # A per-error *retry* count (extra retries), consistent with ``max_retries``, so
                # ``0`` is valid (that error type never retries); only a negative is rejected.
                if not isinstance(limit, numbers.Integral) or limit < 0:
                    raise ValueError(
                        f"retry_policy[{exc_name!r}] must be a non-negative integer"
                    )
        self.request_defaults: dict[str, Any] = request_defaults
        _warn_on_misspelled_request_params(request_defaults, type(self).__name__)
        if model is not None:
            self.model = model
        # Only set what was actually passed, so an unset kwarg falls through to the class
        # default rather than shadowing it with ``None``.
        for name, value in (
            ("max_retries", max_retries),
            ("retry_wait_multiplier", retry_wait_multiplier),
            ("retry_wait_min", retry_wait_min),
            ("retry_wait_max", retry_wait_max),
            ("retry_policy", retry_policy),
        ):
            if value is not None:
                setattr(self, name, value)

    @abstractmethod
    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        """Execute a completion call against the LLM.

        Args:
            model: The model identifier, as this backend's API expects it
                (e.g. ``"gpt-4"``) — see :meth:`~BaseLLM.validate_model`.
            messages: The conversation messages in OpenAI format.
            **kwargs: Additional provider-specific parameters (temperature, max_tokens, etc.).

        Returns:
            An LLMResponse with the completion content and usage information.
        """
        ...

    async def acomplete(
        self, model: str, messages: list[Any], **kwargs: Any
    ) -> LLMResponse:
        """Async version of complete(). Default: runs complete() in a thread.

        Subclasses should override this for native async I/O.
        """
        return await asyncio.to_thread(self.complete, model, messages, **kwargs)

    @property
    def non_retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        """Exception types that must NOT be retried — re-raised immediately.

        The default is the set no amount of backoff can fix: a malformed request, a rejected
        prompt, a missing model, a bad key, an over-long context — plus ``KeyboardInterrupt``,
        so a retry loop never swallows a user's interrupt.
        """
        # Promoted from the old DefaultLLMClient, where it was reachable only by backends that
        # happened to be routed through that one client. Every HTTP backend wants exactly this
        # set, and a backend that raises none of these is unaffected (the types simply never
        # match), so the base is the right home. Override to add backend-specific types.
        return (
            UnsupportedParamsError,
            ContentPolicyViolationError,
            NotFoundError,
            PermissionDeniedError,
            ContextWindowExceededError,
            AuthenticationError,
            KeyboardInterrupt,
        )

    def _retry_default(self, name: str, override: Any) -> Any:
        """Resolve one retry setting: an explicit per-call value wins, else the field.

        The fields are the configured source of truth (they come from ``llm.params``);
        the per-call kwargs remain as an escape hatch for direct callers that wrap
        ``complete_with_retry`` themselves, which is why they default to ``None`` rather
        than to a literal — ``None`` means "unspecified", not "zero attempts".
        """
        return getattr(self, name) if override is None else override

    def _retryer(
        self,
        *,
        max_retries: int,
        wait_multiplier: float,
        wait_min: float,
        wait_max: float,
        on_retry: Callable[[RetryCallState], None] | None,
        retry_policy: dict[str, int] | None = None,
    ) -> Callable[[Any], Any]:
        """Build the tenacity retry decorator shared by ``complete_with_retry``
        and its async mirror ``acomplete_with_retry`` — tenacity wraps sync and
        async callables transparently. Falls back to the default WARNING logger
        when the caller supplies no ``on_retry``.

        ``retry_policy`` assigns a *per-category retry budget* keyed by exception
        class name. Every failing attempt is bucketed by its error's category and
        counted independently; the call is retried only while the *current*
        error's category is still under its own budget:

        - a type named in ``retry_policy`` -> its own retry budget (that value),
          overriding both ``max_retries`` and ``non_retryable_exceptions``
          (set 0 to force no retry for that type);
        - any other retryable type -> a shared default budget of ``max_retries``
          (so an empty policy reduces exactly to the plain behavior:
          ``max_retries`` retries, non-retryable types failing fast);
        - any other non-retryable type -> budget 0 (fail fast).

        Counting each category independently is the semantics LiteLLM's
        ``RetryPolicy`` docstring *promises* ("custom number of retries per
        exception type") but its code does not deliver: LiteLLM computes the
        count once from the *first* exception and never re-derives (see litellm
        ``router.py::async_function_with_retries``), so under mixed error types a
        listed type's number is silently governed by whatever failed first. Here
        a burst of one error type can never consume another's allowance. (An
        earlier version of this code compared against tenacity's *global*
        ``attempt_number``, which had the same cross-contamination bug in a
        different guise; the per-category counters below fix it.)

        Values are *retry* counts (extra retries, N -> N+1 calls), consistent with
        ``max_retries`` and with the LLM-SDK convention. Matching is by *exact*
        runtime class name (``type(exc).__name__``) over an open dict, not LiteLLM's
        subclass-aware ``isinstance`` over a fixed set of typed fields — so a
        base-class key like "LLMError" will NOT catch subclasses.
        """
        policy = retry_policy or {}
        non_retryable = self.non_retryable_exceptions

        # One counter per error category, tracking attempts already spent on it.
        # tenacity invokes the ``retry`` predicate exactly once per completed
        # attempt (see BaseRetrying._begin_iter), so incrementing here counts
        # each attempt exactly once.
        default_bucket = object()
        counts: dict[Any, int] = {}

        def retries_for(exc: BaseException) -> tuple[Any, int]:
            name = type(exc).__name__
            if name in policy:
                return name, policy[name]
            if isinstance(exc, non_retryable):
                return name, 0  # fail fast — no retries
            return default_bucket, max_retries

        def should_retry(retry_state: RetryCallState) -> bool:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if exc is None:
                return False
            key, retries = retries_for(exc)
            counts[key] = counts.get(key, 0) + 1
            # ``counts`` is attempts completed on this category; retry while that does not exceed
            # the allowed *retries*. So ``retries`` extra calls follow the first: after attempt
            # ``retries + 1`` this is ``> retries`` and stops. (0 retries -> stop after attempt 1.)
            return counts[key] <= retries

        # Liveness backstop only — the per-category counters above are the real stop condition.
        # Total attempts are bounded by (1 first call + max_retries) plus each policy type's
        # extra retries. Equals max_retries + 1 when no policy is set.
        attempt_cap = max_retries + 1 + sum(policy.values())

        return retry(
            reraise=True,
            stop=stop_after_attempt(attempt_cap),
            wait=wait_exponential(
                multiplier=wait_multiplier, min=wait_min, max=wait_max
            ),
            before_sleep=on_retry
            if on_retry is not None
            else _default_retry_logger(attempt_cap),
            retry=should_retry,
        )

    def complete_with_retry(
        self,
        model: str,
        messages: list[Any],
        *,
        max_retries: int | None = None,
        retry_wait_multiplier: float | None = None,
        retry_wait_min: float | None = None,
        retry_wait_max: float | None = None,
        on_retry: Callable[[RetryCallState], None] | None = None,
        retry_policy: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute ``complete`` with automatic retry on transient errors.

        Retries on any exception not in ``non_retryable_exceptions`` using
        exponential backoff. ``acomplete_with_retry`` is the async mirror.

        Args:
            model: The model identifier.
            messages: The conversation messages.
            max_retries: Per-call override; defaults to the field of the same name — extra
                retries after the first call (``2`` → up to 3 calls; ``0`` = no retry). The call
                kwargs, the constructor parameters and the ``llm.params`` keys share one
                vocabulary, so a setting is spelled identically wherever it is written.
            retry_wait_multiplier: Per-call override; defaults to the field.
            retry_wait_min: Per-call override; defaults to the field.
            retry_wait_max: Per-call override; defaults to the field.
            on_retry: Optional callback invoked before each retry sleep.
                Receives a tenacity ``RetryCallState``. If not provided, retries
                are logged at WARNING level.
            retry_policy: Per-call override; defaults to ``self.retry_policy``.
                LiteLLM-style per-exception attempt budgets (see ``_retryer``).
            **kwargs: Passed through to ``complete()``.

        Returns:
            An LLMResponse with the completion content and usage information.
        """
        retryer = self._retryer(
            max_retries=self._retry_default("max_retries", max_retries),
            wait_multiplier=self._retry_default(
                "retry_wait_multiplier", retry_wait_multiplier
            ),
            wait_min=self._retry_default("retry_wait_min", retry_wait_min),
            wait_max=self._retry_default("retry_wait_max", retry_wait_max),
            on_retry=on_retry,
            retry_policy=self._retry_default("retry_policy", retry_policy),
        )
        return retryer(self.complete)(model, messages, **kwargs)

    async def acomplete_with_retry(
        self,
        model: str,
        messages: list[Any],
        *,
        max_retries: int | None = None,
        retry_wait_multiplier: float | None = None,
        retry_wait_min: float | None = None,
        retry_wait_max: float | None = None,
        on_retry: Callable[[RetryCallState], None] | None = None,
        retry_policy: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async mirror of ``complete_with_retry`` — wraps ``acomplete``.

        Same retry policy and parameters, so async callers get the same
        transient-error resilience as sync callers. The Agent's async path calls
        this via ``_make_llm_retry_fn``, passing its hook-dispatching ``on_retry``.
        """
        retryer = self._retryer(
            max_retries=self._retry_default("max_retries", max_retries),
            wait_multiplier=self._retry_default(
                "retry_wait_multiplier", retry_wait_multiplier
            ),
            wait_min=self._retry_default("retry_wait_min", retry_wait_min),
            wait_max=self._retry_default("retry_wait_max", retry_wait_max),
            on_retry=on_retry,
            retry_policy=self._retry_default("retry_policy", retry_policy),
        )
        return await retryer(self.acomplete)(model, messages, **kwargs)

    def get_model(self) -> str:
        """The model id this backend serves.

        Raises ``ValueError`` when none is configured, or when the id carries a backend prefix
        this backend would not send (see :meth:`~BaseLLM.validate_model`). Override only for a backend whose
        model is not the ``model`` constructor argument (``MockLLMClient`` does).
        """
        # Read from the instance rather than from a passed-in config bag: the backend now holds
        # its own `model`, so a caller cannot supply a different one here and silently disagree
        # with what `request_defaults` was built for.
        #
        # Raises rather than defaulting to "". Returning the empty string kept this docstring's
        # promise only vacuously: the "" travelled all the way to the API as `"model": ""`.
        # A missing model is a config error, and the useful place to say so is before any
        # request is built.
        if not isinstance(self.model, str) or not self.model:
            raise ValueError(
                f"{type(self).__name__} has no model configured (model={self.model!r}). "
                f"Set it, e.g. jaz.configure(llm={type(self).__name__}(model='gpt-5-mini'))."
            )
        # Validated here rather than where the request is built, for the same reason the check
        # above lives here: this is the seam every entry point passes through — the Python API,
        # a settings file, an eval YAML — so a bad id surfaces once, at `Agent.__init__`, rather
        # than once per call site that remembers to look. `validate_model` is the per-backend
        # rule, so a transport that legitimately keeps the prefix overrides it and inherits the
        # right behaviour here for free.
        self.validate_model(self.model)
        return self.model

    def validate_model(self, model: str) -> None:
        """Reject a model id this backend would not send verbatim.

        The configured id **is** the wire id: nothing is stripped, rewritten or inferred, so
        this only says yes or no. A leading ``segment/`` naming a registered backend tag is an
        error, and the message names the id to use instead::

            OpenAILLM().validate_model("gpt-5-mini")                       -> ok (bare id)
            OpenAILLM().validate_model("meta-llama/Llama-3.1-8B-Instruct") -> ok (slash, not a tag)
            OpenAILLM().validate_model("litellm/gpt-5-mini")               -> ValueError (registered tag)

        The rule that makes this predictable: **write the id the backend's own documentation
        uses.** OpenAI documents ``gpt-5-mini`` and Anthropic documents ``claude-sonnet-4-5`` —
        bare. A leading ``provider/`` is an *aggregator* convention.

        Override to a no-op where that convention is the real one: a transport-style backend
        routing by provider name (LiteLLM) documents ``openai/gpt-5-mini``, so the prefix is its
        routing key and there is nothing to reject. That is the same rule, not an exception.
        """
        # Returns nothing, and is named for what it does. It began as a `strip_tag` that
        # rewrote the id, and kept a `-> str` through the move to literal ids even though every
        # path — here and in the anticipated transport override — returns its input unchanged.
        # A signature promising a transformation that never happens made two call sites read as
        # conversions (`"model": self.wire_model(model)`) and two others read as no-ops.
        #
        # Per-backend rather than the module-level `strip_tag` this replaced: that function
        # consulted the *global* registry, so it stripped any leading segment naming any
        # registered tag no matter which backend was running — `anthropic/claude-…` on an
        # OpenAI backend silently became a request for `claude-…` against OpenAI. Only the
        # backend knows what its own prefix means, so only the backend can judge it.
        from .registry import registered_llm_tags

        head, sep, rest = model.partition("/")
        # Only a *registered tag* is rejected. Any other slash belongs to the id, so
        # Hugging-Face-style names survive — the norm against a local OpenAI-compatible server,
        # which is the deployment the README leads with.
        if sep and head in registered_llm_tags():
            if self.registry_tag is not None and head == self.registry_tag:
                hint = f"drop the prefix and use {rest!r}"
            else:
                hint = f"configure the {head!r} backend and use {rest!r}, or drop the prefix"
            raise ValueError(
                f"model {model!r} carries the backend prefix {head + '/'!r}. The model id is "
                f"sent verbatim, so write it the way the backend's own docs do: {hint}."
            )

    def _price_key(self, model: str) -> str:
        """The price-table key for ``model``: the full id if the table has it, else unprefixed.

        Full id **first**, because a slash is not always a provider prefix — the bundled table
        keys image variants that way (``1024-x-1024/dall-e-2``, ``openai/sora-2``), and
        stripping first would price those under the wrong key.

        The fallback exists for transport-style backends, where the *literal* model id carries a
        provider prefix by design (``openai/gpt-5-mini``, ``vertex_ai/gemini-2.5-pro``) while the
        table keys the model bare. It is deliberately not restricted to registered jaz tags:
        those prefixes are the transport's provider names, not jaz's. Only a fallback, so a
        table hit on the full id always wins.
        """
        if _get_model_info(model):
            return model
        _, sep, rest = model.partition("/")
        return rest if sep and _get_model_info(rest) else model

    def get_model_info(self) -> dict[str, Any]:
        """Metadata for the configured model, from the bundled pricing table."""
        return _get_model_info(self._price_key(self.get_model()))

    def can_report_cost(self) -> bool:
        """Whether a per-call cost will be reported for the configured model.

        Answered *before* any query, so a cost budget that could never be enforced can be
        rejected without spending tokens. ``True`` promises only that a cost will be
        populated — not that it is nonzero or accurate.

        Limitation: this tests for a pricing-table *entry*, not for usable token rates. An
        entry that publishes no per-token rate prices at ``0.0`` rather than ``None``, so this
        returns ``True`` and the cost is reported as free.
        """
        # Concrete rather than abstract because `finalize` prices every response from the same
        # bundled table, so the base can answer for any backend that uses it. A backend that
        # computes cost some other way (or is handed one) overrides — MockLLMClient does.
        return bool(self.get_model_info())

    # ------------------------------------------------------------------
    # Construction from config — the Layer 0/1 entry point.
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, params: Mapping[str, Any] | None = None) -> BaseLLM:
        """Build this backend from its construction params — ``REGISTRY[tag].from_dict(params)``.

        The default maps params to constructor kwargs. Override for a backend whose config
        shape does not match its ``__init__``.
        """
        return cls(**(params or {}))

    # ------------------------------------------------------------------
    # Shared response normalization — what every HTTP backend gets for free.
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_request_timeout(kwargs: dict[str, Any], default: float) -> float:
        """Resolve the HTTP transport timeout for one request and strip it from ``kwargs``.

        A per-call ``timeout`` applies to the HTTP *transport*, not the request body — so it
        must be popped out of ``kwargs`` before the rest are forwarded into the payload,
        otherwise it is sent as an API field and ignored or rejected (#623).
        """
        override = kwargs.pop("timeout", None)
        return default if override is None else override

    def finalize(
        self,
        response: CompletionResponse,
        model: str,
        *,
        service_tier: str | None = None,
    ) -> LLMResponse:
        """Normalize a wire-shaped :class:`CompletionResponse` into an :class:`LLMResponse`.

        Extracts the content and token counts, prices the call from the bundled table, and
        builds the ATIF-style ``extra`` bag (provider-specific token buckets plus whatever the
        backend reported). ``model`` may carry a ``tag/`` prefix; pricing uses the bare name.

        ``service_tier`` picks the matching priority/flex/batch rate keys when published. A
        tier the backend *reports* wins over the one requested, in case the API downgraded it.
        """
        # This is the half of the old two-class design that was worth keeping shared. It used to
        # live inside `DefaultLLMClient.complete` (and again, subtly diverged, in `.acomplete`)
        # — so a backend that was not routed through that client got no cost accounting at all,
        # and the async path silently omitted `cached_tokens` and the `extra` bag. Merging the
        # client and provider layers made it a base-class method every backend can call, which
        # is what removes the duplication rather than relocating it.
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None

        cost: float | None = None
        if prompt_tokens is not None and completion_tokens is not None:
            served_tier = (
                usage.extra.get("service_tier") if usage else None
            ) or service_tier
            cost = compute_cost(
                self._price_key(model),
                prompt_tokens,
                completion_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens
                if usage
                else 0,
                cache_read_input_tokens=usage.cache_read_input_tokens if usage else 0,
                service_tier=served_tier,
            )

        extra: dict[str, Any] = {}
        if usage:
            if usage.cache_creation_input_tokens:
                extra["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
            extra.update(usage.extra)

        return LLMResponse(
            content=response.choices[0].message.content if response.choices else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=usage.cache_read_input_tokens if usage else None,
            cost_usd=cost,
            extra=extra,
            raw_response=response,
            tokens=response.tokens,
        )
