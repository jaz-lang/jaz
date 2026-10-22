"""The LLM backend's result type — what :meth:`BaseLLM.complete` hands back.

Separate from ``_base`` for the reason ``jaz.repl._types`` is separate from ``jaz.repl._base``:
the result type is what every caller of the seam touches, while the ABC is what only an
implementer touches, and the hook and agent layers want the former without the latter.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from jaz import exceptions
from jaz._tokens import TurnRecord


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
        ``completion_token_ids`` and ``logprobs``, and ``finish_reason`` (why
        the response stopped, in OpenAI's vocabulary: ``"stop"``,
        ``"length"``, ``"tool_calls"``, ...)

    ``content`` and ``raw_response`` are not metrics and have no ATIF counterpart.
    There is no ``total_tokens``: sum ``prompt_tokens`` and ``completion_tokens``.
    """

    # Aligned field-for-field with ATIF v1.7 MetricsSchema (executive call, user,
    # 2026-08-12) rather than either adopting the format's schema wholesale or letting the
    # two drift. Full unification was never reachable — this is a *response*, not a metrics
    # record, so `content`/`raw_response` keep it a strict superset of MetricsSchema.
    #
    # `cost_usd`, not `cost`, to match ATIF: a bare `cost` never says what currency it is in.
    #
    # There is no `total_tokens` field. It would carry nothing its parts do not — the OpenAI
    # usage contract defines it as `prompt + completion`, and a backend reports those two parts
    # however it gets them (forwarded from the provider, summed over a turn's usage events, or
    # read off a local server's meta) — and ATIF has
    # no counterpart. It could also contradict its own parts: a backend that zero-fills a
    # missing count beside one that leaves it `None` surfaces the same unreported usage as
    # `total_tokens=0` next to nonzero parts on one and `None` on the other. Consumers sum at
    # the point of use. `Budget` still records a total because its cost records are an
    # on-disk format the eval tooling reads.
    #
    # ATIF's remaining three metrics (`prompt_token_ids`, `completion_token_ids`,
    # `logprobs`) are deliberately NOT fields here. They are optional, provider-rare, and by
    # far the largest payload in a response — per-token arrays — so promoting them would
    # write a retention cost into the contract every backend nominally implements. They ride
    # in `extra` under their ATIF names, and the two ATIF emitters hoist them into top-level
    # `metrics`, which is the layer that actually owes the format conformance.

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
    # agent's stamp-back and RolloutRecorder key off it), and `extra` is a bag of additive
    # counts: anything that combines two responses' extras sums like-typed values, which is
    # exactly wrong for token sequences (they concatenate, and a summed one describes no turn
    # at all). Pointer cost only: the record shares the per-message stamp objects.
    tokens: TurnRecord | None = None


@dataclass(frozen=True, eq=False)
class RetryPolicy:
    """How a backend retries a failed call: the backoff schedule and the retry caps.

    Every count is a cap on *extra* retries, the LLM-SDK convention: ``max_retries=2`` is up to
    three calls, ``0`` is no retry. A failed call is retried, with exponential backoff, while
    it is retryable and under every cap it falls under. It is retryable if it is an instance of
    a class in the backend's ``retryable_exceptions`` (by default
    :class:`~jaz.exceptions.RateLimitError`, :class:`~jaz.exceptions.ServerError` and
    :class:`~jaz.exceptions.APIConnectionError`, timeouts included) or in
    ``extra_retryable_exceptions``. The provider can override that for one error
    (:attr:`~jaz.exceptions.LLMError.is_retryable`, from its ``x-should-retry`` header), in
    either direction, so its "do not retry" stops even a class in
    ``extra_retryable_exceptions``.
    ``max_retries`` caps every error, and each ``limits`` key caps the errors that are
    instances of its class.

    Attributes:
        max_retries: Cap on one call's retries, in total across every error.
        wait_multiplier: Exponential-backoff multiplier, in seconds.
        wait_min: Shortest backoff between attempts, in seconds; jitter lengthens it at random
            by up to a quarter.
        wait_max: Cap on the exponential backoff, in seconds, before jitter lengthens it by
            up to a quarter. It does not cap a provider's ``Retry-After``; see
            ``retry_after_max``.
        retry_after_max: Longest ``Retry-After`` a retry waits, in seconds. A provider's
            ``Retry-After`` is waited instead of the backoff, plus up to a quarter of jitter;
            one longer than this stops the retries, and the call raises the error, whose
            ``retry_after`` says when to try again. ``0`` never waits one out.
        limits: Caps on the retries of particular errors, keyed by exception class
            (``RateLimitError``, ``LLMError``, a backend's own class, ...); any other key
            raises ``TypeError``. Every key an error is an instance of caps it, and all errors
            under one key share its cap, so ``{LLMError: 3, RateLimitError: 1}`` allows rate
            limits 1 retry and all LLM errors 3 in total. A value above ``max_retries``, or
            above the value of a key for one of its superclasses, raises ``ValueError``, since
            it could never apply. A key only caps: an error that is not retryable fails at
            once, so a key on such a class applies only to errors the provider asked to retry.
            To give one error more retries than the rest of its family, raise
            ``max_retries`` and cap the siblings, not the family:
            ``max_retries=8, limits={ServerError: 2, APIConnectionError: 2}`` lets rate limits
            retry 8 times and every other retried error twice. A key on a class the backend
            maps to another before retrying never matches.
        extra_retryable_exceptions: Exception classes to retry in addition to the backend's
            ``retryable_exceptions``, subclasses included, as a list, tuple or set (held as a
            tuple): ``extra_retryable_exceptions=[ContentPolicyViolationError]`` retries
            content-policy rejections, which can pass when resent, and
            ``[httpx.ReadTimeout]`` retries a timeout a backend did not map. A broad class
            retries everything under it: ``[Exception]`` retries every error a backend
            raises, bugs included, and ``[BaseException]`` also retries a task
            cancellation, Ctrl-C and ``sys.exit()``, so none of them stops a call until its
            retries run out. Anything but exception classes raises ``TypeError``. Like
            ``limits``, it holds classes, so neither can be written in a config file.
    """

    # ``max_retries`` is EXTRA retries after the first call (the LLM-SDK convention — OpenAI /
    # Anthropic / LangChain / … all count this way), NOT total attempts: ``max_retries=2`` → 3
    # calls. Default 10 (→ 11 attempts): bare client SDKs default to 2
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_constants.py#L8),
    # but an AGENT framework running long autonomous loops wants more per-call resilience (a
    # transient blip mid-run is expensive), so this matches the Claude Agent SDK / Claude Code
    # default of 10 (https://code.claude.com/docs/en/env-vars) — its closest peer — rather than the
    # client-SDK 2 or the more centrist agent default of ~3-5 (DSPy 3,
    # https://github.com/stanfordnlp/dspy/blob/3.4.0/dspy/clients/lm.py#L211;
    # OpenHands 5,
    # https://github.com/OpenHands/software-agent-sdk/blob/v1.54.0/openhands-sdk/openhands/sdk/llm/llm.py#L379).
    # These are the only retry defaults; there is no config-layer copy.
    max_retries: int = 10
    # The wait defaults (1x backoff from 4 s, capped at 60 s) are an
    # implementation choice, not decided by the user. They sit between the peers: the
    # `openai`/`anthropic` SDKs and LiteLLM wait 0.5 s growing to 8 s
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_constants.py#L11-L12,
    # https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_constants.py#L14-L15,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/constants.py#L495-L496),
    # OpenHands 8 s growing to 64 s
    # (https://github.com/OpenHands/software-agent-sdk/blob/v1.54.0/openhands-sdk/openhands/sdk/llm/llm.py#L381-L382),
    # and Claude Code retries a 529 overloaded error from a 500 ms base delay
    # (https://code.claude.com/docs/en/env-vars).
    wait_multiplier: float = 1.0
    wait_min: float = 4.0
    wait_max: float = 60.0
    # The threshold for a provider's `Retry-After`, separate from `wait_max` (the user's call,
    # 2026-10-13): one setting doing both jobs made raising the backoff cap also accept longer
    # server delays, and lowering it refuse them. Above it the call stops rather than waits
    # (the user's call, 2026-10-12; the reasoning and peers are in `BaseLLM._retryer`); at or
    # below it the delay is waited in full, not capped by `wait_max`, as the `openai` SDK
    # honours a delay up to `MAX_RETRY_AFTER_DELAY` while `MAX_RETRY_DELAY` caps only its
    # backoff
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L869-L878).
    # The default 120 s is the `openai` SDK's `MAX_RETRY_AFTER_DELAY = 2 * 60`
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_constants.py#L13), an
    # implementation choice, not decided by the user; pydantic-ai's `wait_retry_after` defaults
    # to 300 s, though it caps the wait there rather than stopping
    # (https://github.com/pydantic/pydantic-ai/blob/v2.55.0/pydantic_ai_slim/pydantic_ai/retries.py#L514-L515).
    # No ordering against `wait_max` is required (an implementation choice, not decided by the
    # user): the two bound different waits, and either order is a coherent policy, such as a
    # long backoff for blips with a refusal to sit out a server's long delay.
    retry_after_max: float = 120.0
    # Keys typed `Any`, checked at runtime in `__post_init__`: `Mapping` is invariant in its key,
    # so `Mapping[type[BaseException], int]` would reject `policy = {RateLimitError: 3}` passed as a
    # variable (pyright infers `dict[type[RateLimitError], int]`), the commonest way to write one.
    # TODO(#1590): class keys can't be written in a config file; dotted import paths would be.
    limits: Mapping[Any, int] = field(default_factory=dict)
    # Its own field rather than a `limits` key that also made errors retryable (the user's
    # call). That lift rule had to decide what a key nested with a retryable class meant
    # (`{LLMError: 3}`: a cap, or "retry every LLM error"?) and had no obvious winner; the one
    # tried, "a key lifts unless it covers a retryable class", made a key's meaning depend on
    # classes it does not name. With retrying and capping separate, `limits` is only caps and
    # this only adds to the backend's set, whose name it extends. A `limits` key on a class
    # that is never retried is allowed, not refused: whether it can apply depends on whether
    # some error class inherits from both it and a retryable class, which may not be defined
    # yet, so a check would accept or refuse the same policy depending on what is imported.
    # Typed as any collection, not `tuple` (the user's call): a caller can then write
    # `[ContentPolicyViolationError]`, avoiding the one-element tuple's trailing comma, or a
    # set, since order means nothing to `isinstance`. Held as a tuple, as `limits` is held as a
    # `mappingproxy`: abstract in, immutable stored.
    # TODO(#1590): like `limits`, classes can't be written in a config file.
    extra_retryable_exceptions: Collection[type[BaseException]] = ()

    # No value equality or hashing (`eq=False`, the user's call): nothing compares two policies,
    # and a by-value `==` would call `(A, B)` and `(B, A)` different although they retry the
    # same errors. Two policies are equal only if they are the same object.
    #
    # One frozen object rather than four scalar attributes plus a mapping on the backend: a
    # backend author sets one thing, authored config carries one `retry_policy` key, and the settings
    # cannot be spelled inconsistently across the constructor, the class defaults and the
    # retry call. Validation lives here so every path that builds one gets it.
    def __post_init__(self) -> None:
        # A `bool` is an `Integral` and a `Real`, and NaN passes every comparison below (a NaN
        # `wait_max` silently removed the cap), so both are refused by name.
        if (
            not isinstance(self.max_retries, numbers.Integral)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        for name, value, inclusive in (
            ("wait_multiplier", self.wait_multiplier, False),
            ("wait_min", self.wait_min, True),
            ("wait_max", self.wait_max, False),
            ("retry_after_max", self.retry_after_max, True),
        ):
            if (
                not isinstance(value, numbers.Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            if float(value) < 0 or (not inclusive and float(value) == 0):
                raise ValueError(
                    f"{name} must be " + ("non-negative" if inclusive else "positive")
                )
        # tenacity's `wait_exponential(min, max)` clamps to `max` first
        # (https://github.com/jd/tenacity/blob/9.1.4/tenacity/wait.py#L208),
        # so an inverted pair silently waits `min` on every attempt with no backoff and no cap;
        # refused instead.
        if float(self.wait_min) > float(self.wait_max):
            raise ValueError("wait_min must not exceed wait_max")
        if not isinstance(self.limits, Mapping):
            raise ValueError("limits must be a mapping of exception class -> retries")
        # Keys are the classes themselves (an executive call by the user, after #1579 and
        # #1588). Name strings had to be resolved through a table jaz could build only from its
        # own public classes, so a backend's own subclass could never be named, while
        # `retryable_exceptions`, which the retry loop combines with `limits`, holds classes.
        # Any exception class is accepted, the same domain as `retryable_exceptions`.
        # TODO(#1591): a key on a class the backend maps to another never matches.
        for key, limit in self.limits.items():
            if not (isinstance(key, type) and issubclass(key, BaseException)):
                raise TypeError(
                    f"limits keys must be exception classes, got {key!r}."
                    f"{_bad_key_hint(key, 'limits={{{}: ...}}')}"
                )
            # A `bool` refused as for `max_retries`: `{RateLimitError: True}` would cap at 1.
            if (
                not isinstance(limit, numbers.Integral)
                or isinstance(limit, bool)
                or limit < 0
            ):
                raise ValueError(
                    f"limits[{key.__name__}] must be a non-negative integer"
                )
            # Refused rather than held to `max_retries` (the user's call): every count is a cap,
            # so such a value could never bind, and accepting it silently is exactly the
            # LiteLLM-style config ("8 retries for rate limits") that would then quietly get
            # `max_retries`. Nothing expressible is lost: the value behaves as `max_retries`.
            if limit > self.max_retries:
                raise ValueError(
                    f"limits[{key.__name__}] = {limit} exceeds max_retries = {self.max_retries}: "
                    "every limit is a cap within max_retries, so it could never apply. Raise "
                    "max_retries to allow more retries, and cap the other errors with limits."
                )
        # A cap under a broader key can never exceed it: every error under the narrower key also
        # spends the broader key's allowance. The same reason as the check against `max_retries`,
        # the allowance of every error; refused for the same reason (the user's call).
        for sub, sub_limit in self.limits.items():
            for sup, sup_limit in self.limits.items():
                if sub is not sup and issubclass(sub, sup) and sub_limit > sup_limit:
                    raise ValueError(
                        f"limits[{sub.__name__}] = {sub_limit} exceeds limits[{sup.__name__}] = "
                        f"{sup_limit}, which also caps every {sub.__name__}, so it could never "
                        f"apply. Lower it, or to give {sub.__name__} more retries than the rest of "
                        f"{sup.__name__}, raise {sup.__name__}'s cap and cap its siblings instead."
                    )
        extra = self.extra_retryable_exceptions
        # A mapping is a `Collection` of its keys, so it would pass and silently drop its
        # values; refused because `{Cls: 2}` is the shape of `limits` and easy to write here.
        if isinstance(extra, Mapping):
            raise TypeError(
                "extra_retryable_exceptions takes exception classes, not a mapping, got "
                f"{extra!r}. Put the classes here and their retry caps in limits: "
                "extra_retryable_exceptions=[cls], limits={cls: n}."
            )
        if isinstance(extra, (str, type)) or not isinstance(extra, Collection):
            fix = (
                f" Wrap it in a list: [{extra.__name__}]."
                if isinstance(extra, type) and issubclass(extra, BaseException)
                else _bad_key_hint(extra, "extra_retryable_exceptions=[{}, ...]")
            )
            raise TypeError(
                "extra_retryable_exceptions must be a collection of exception classes, got "
                f"{extra!r}.{fix}"
            )
        extra = tuple(extra)
        for cls in extra:
            if not (isinstance(cls, type) and issubclass(cls, BaseException)):
                raise TypeError(
                    "extra_retryable_exceptions must hold exception classes, got "
                    f"{cls!r}.{_bad_key_hint(cls, 'extra_retryable_exceptions=[{}, ...]')}"
                )
        object.__setattr__(self, "extra_retryable_exceptions", extra)
        # Frozen all the way down: the field cannot be rebound, and the mapping cannot be
        # written through — a policy is shared as `BaseLLM`'s class default, so a write-through
        # would reach every backend. A `mappingproxy` does not pickle, so `__reduce__` rebuilds
        # from the plain values, which keeps a backend holding one copyable. `dataclasses.asdict` still cannot deep-copy it; read
        # `dict(policy.limits)` instead.
        object.__setattr__(self, "limits", MappingProxyType(dict(self.limits)))

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            RetryPolicy,
            (
                self.max_retries,
                self.wait_multiplier,
                self.wait_min,
                self.wait_max,
                self.retry_after_max,
                dict(self.limits),
                self.extra_retryable_exceptions,
            ),
        )


def _bad_key_hint(key: object, example: str) -> str:
    """What to pass instead of `key`, which is not an exception class; `example` shows the
    field with one ``{}`` for the class."""
    if isinstance(key, str):
        name = key.rsplit(".", 1)[-1]
        cls = getattr(exceptions, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            return (
                f" Pass the class itself: {example.format(f'jaz.exceptions.{name}')}."
            )
        return " Pass the class itself, not its name."
    if isinstance(key, BaseException):
        return f" Pass a class, not an instance: {type(key).__name__}."
    return ""


@dataclass(frozen=True)
class ModelInfo:
    """What a backend knows about the model it serves; every field has an "unknown" value.

    Attributes:
        name: The model id, when the backend serves a nameable one — a routed id such as
            ``"openai/gpt-5-mini"`` for :class:`LiteLLM`. ``None`` for a backend with no such
            notion (a mock, a server that serves whatever it was launched with). Observability
            hooks label spans with it when present and omit the label when not.
        max_input_tokens: The model's input-token limit, or ``None`` when the backend cannot
            say. ``None`` means unknown, not unlimited — :class:`jaz.hooks.ContextWindowWarning`
            disables itself on it rather than sizing a window from a guess.
        can_report_cost: Whether this backend will set ``LLMResponse.cost_usd`` for this
            model. ``False`` is the honest default: the framework prices nothing itself. A cost
            budget on a backend that answers ``False`` is rejected before any token is spent.
    """

    # A typed value rather than a `model` attribute plus two getters on the base class: the
    # base mandates nothing about the model — not even that there is one to name — and asks
    # instead what the backend can say about it. A field is added here when something in the
    # framework reads it, not when a backend could report it.
    name: str | None = None
    max_input_tokens: int | None = None
    can_report_cost: bool = False
