"""The LLM backend: one class per backend, doing the whole job.

A :class:`BaseLLM` owns everything about talking to one model service — the HTTP call, error
mapping, retry, cost accounting and model metadata. It is selected by a single tag
(``"litellm"``, ``"copilot"``, ``"sglang"``) and built from that tag's params, by calling the
class the consumer's resolver returns. Core holds no tag -> class mapping; see
``jaz._instantiate.build_config``'s ``resolvers``.

A backend declares its ``__init__`` (whose signature declares its config surface) and implements
``complete``, returning an :class:`LLMResponse`; the base gives it retry.
"""

# One class does the whole job (executive call, user, 2026-08-07): no client/provider split with
# a delegating layer between them. A split is not a real seam here — one ``llm`` config field
# would resolve into two constructor layers, a backend outside the pairing would get no retry or
# cost accounting, and response normalization duplicated across the layers drifts — it did: the
# async copy dropped `cached_tokens` and the `extra` bag. One name per
# concept, no aliases, for the same reason.

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from jaz._property_contract import require_properties

from ._exceptions import (
    APIConnectionError,
    LLMError,
    RateLimitError,
    ServerError,
)
from ._types import LLMResponse, ModelInfo, RetryPolicy

logger = logging.getLogger(__name__)


def _require_model_id(backend: str, model: object) -> str:
    """The model id a backend was constructed with, or a ``ValueError`` naming the backend.

    For the backends that take ``model=``: an empty id is a configuration error, and the useful
    place to say so is construction, not the first request (where ``""`` would travel to the
    API as ``"model": ""``).
    """
    if not isinstance(model, str) or not model:
        raise ValueError(
            f"{backend} needs a model id (got {model!r}): pass "
            f"model=<the id this backend's API expects>."
        )
    return model


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
    """One LLM backend: the API call, retry, and the cost and model metadata it reports.

    Subclass this and pass an instance to add a backend::

        class MyLLM(BaseLLM):
            def __init__(self, model, base_url, retry_policy=RetryPolicy(), **query_params):
                # retry_policy is named above, or the tail would take it
                super().__init__(retry_policy=retry_policy)
                self._model = model
                self._base_url = base_url
                self._query_params = query_params  # yours; the base never sees them

            def complete(self, messages) -> LLMResponse:
                wire = ...  # call the API with self._model, messages, **self._query_params
                return LLMResponse(content=wire.text, prompt_tokens=..., extra={})

    :class:`LLMResponse` is the whole return contract. With this class, and the two value types
    :class:`RetryPolicy` and :class:`ModelInfo`, it is everything a backend author needs from
    ``jaz.llm``.

    Declaring ``__init__`` is all the config wiring a backend needs: its parameters are what
    authored config can set, and a ``**query_params`` tail of its own is how it takes anything
    it does not name. Give the class a ``__repr__`` that renders that constructor call, with
    every secret as ``'***'``: it is how the console's settings view and a trace describe a
    configured backend, shown verbatim and filtered by nothing else — so a dataclass-generated
    ``repr`` would print an ``api_key`` field as is. Without a ``__repr__`` they show the class
    name and ``model_info.name``. The backend stores what it was given and puts it on the request itself;
    ``complete`` receives the messages and nothing else. Nothing passes through this class: a
    backend that takes a tail names ``retry_policy`` explicitly and forwards only that.

    To read a key from the shared credentials store (``~/.jaz/credentials.json``), call
    :func:`jaz.credentials.resolve_credential` with the provider's tag — the default
    :class:`LiteLLM` backend does this per request, keyed by the provider it routes the model to.
    A backend that never calls it does not read the store, so a key stored under its tag is
    inert. Resolve it *below* an explicitly passed ``api_key`` and below the provider's
    environment variable, and hold it as your own attribute so the secret never becomes
    construction config.

    **Retry** is not a framework layer wrapped around backends: the policy is one construction
    field of this base class, ``retry_policy`` (a :class:`RetryPolicy`), and the framework runs
    ``complete`` under it. That is what lets the config layer route it with no special
    knowledge — it is discovered from this ``__init__`` signature exactly as a backend's
    ``base_url`` is discovered from its own.

    **What is retried** is declared by :attr:`~BaseLLM.retryable_exceptions`: a failed call is
    retried only if it raised one of those classes, by default
    :class:`~jaz.exceptions.RateLimitError`, :class:`~jaz.exceptions.ServerError` and
    :class:`~jaz.exceptions.APIConnectionError` (which includes
    :class:`~jaz.exceptions.APITimeoutError`). So raise a transient provider failure as one of
    them, or add your own class to the tuple; anything else (a rejected request, a bug, a
    cancellation) propagates at once, unless the ``RetryPolicy`` adds its class with
    ``extra_retryable_exceptions``. When the provider says how long to wait or whether to
    retry, pass that on the error as ``retry_after`` (seconds) and ``is_retryable``: the retry
    follows the verdict and waits ``retry_after`` plus up to a quarter of jitter, except that a
    ``retry_after`` longer than ``RetryPolicy.retry_after_max`` means no retry at all, whatever
    the verdict.

    **The model is yours too.** The base has no ``model``: a backend that serves a nameable
    model takes ``model=`` as its own constructor parameter and reports it through
    :attr:`~BaseLLM.model_info`; one that does not (a fixed local server) has nothing to
    declare.

    **What you know about the model** is reported once, through :attr:`~BaseLLM.model_info`
    (a :class:`ModelInfo`): the name observability hooks label spans with, the input-token
    limit :class:`ContextWindowWarning` sizes its window from, and whether you set
    ``LLMResponse.cost_usd`` — the framework prices nothing itself. Every field has an "unknown"
    default, and each is honoured: a ``cost_budget`` on a backend that reports no cost is
    rejected before any call rather than failing partway through, and an unknown window leaves
    the context-window hook disabled rather than sized from a guess.

    The one part that cannot be construction-time is the retry *observer*: the core loop passes an
    ``on_retry`` callback carrying ``invoke_id``/``iteration`` to dispatch ``LLMQueryRetry``,
    which is per-call observability rather than policy.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        require_properties(cls, ("model_info", "retryable_exceptions"))
        # The old deny-list is refused rather than ignored: a backend still overriding it would
        # otherwise lose the fail-fast it declared there, with nothing to say so.
        if any("non_retryable_exceptions" in vars(klass) for klass in cls.__mro__):
            raise TypeError(
                f"{cls.__name__} defines non_retryable_exceptions, which BaseLLM no longer "
                "reads. Override retryable_exceptions instead: the exception classes a failed "
                "call is retried on (by default RateLimitError, ServerError and "
                "APIConnectionError); anything else already fails at once."
            )

    # Declared as *class* attributes, not only assigned in ``__init__``, so a subclass that
    # defines its own ``__init__`` without calling ``super().__init__()`` still resolves
    # them. Subclassing an ABC and forgetting the super call is common enough — and the
    # failure would be an ``AttributeError`` deep inside a retry, far from the cause — that
    # the settings must not depend on cooperative construction.
    # Private, like the request params: the retry wrapper is the only reader, so a property
    # would be surface nothing reads. (A frozen value regardless, since a backend instance is
    # shared by every config that holds it, #741.) Class-level default so it resolves for a
    # subclass that skips ``super().__init__()``.
    _retry_policy: RetryPolicy = RetryPolicy()

    # The base has no `model`: not as an attribute, not as a constructor parameter, not as an
    # argument to `complete`. Nor any request param: the base merging a mapping it does not
    # understand into `complete(**kwargs)` made three parties share one dict with nothing in
    # `complete`'s signature saying so (executive call, user, 2026-09-17: the implementing class
    # is in charge, not the base or the caller). A backend that serves a nameable model takes `model=` itself
    # (`LiteLLM`, `GitHubCopilotLLM`, `SGLangLLM` do), stores it privately, and reports it through
    # `model_info.name`; one that does not (a mock, a fixed local server) has nothing to
    # pretend. That keeps #1086's rule — no silent default model — without the base mandating
    # a concept every backend then had to satisfy, and it is why "nothing configured" is the
    # absence of a configured backend rather than a placeholder one — the same call DSPy
    # (`settings.lm = None`) and Pydantic AI (`Agent.model = None`) make.

    #: Reserved sidecar keys (see ``jaz._provenance.RESERVED_KEYS``) this backend reads and
    #: therefore receives on the wire; every other reserved key is stripped at egress.
    #: Default: none — a text backend's wire list carries no internal keys.
    #
    # A per-class declaration (and a class attribute for the same reason as the retry fields
    # above: it must resolve even when a subclass skips ``super().__init__()``). Text backends drop ``messages`` straight into their request
    # payloads, so nothing internal may survive egress for them; a token-native backend
    # (``SGLangLLM``) lists the token key so its stamps reach ``complete()``. Keying the strip
    # by backend declaration keeps the single egress choke point (``to_wire_messages``) while
    # letting each backend opt in to exactly the sidecars it consumes. Provenance is listed by
    # no backend, so it is stripped for all. A key outside ``RESERVED_KEYS`` is inert here —
    # only reserved keys are ever stripped. Named for what the value *is* (a set of keys)
    # rather than what the backend does with them: a verb-phrase name reads as a boolean flag.
    extra_message_keys: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | Mapping[str, Any] = RetryPolicy(),  # noqa: B008
    ) -> None:
        # Retry is the one thing the base holds: it is what `_complete_with_retry` reads. There
        # is no open tail here — the params a backend sends are the backend's own, stored by its
        # own `__init__` and put on its own `complete` call — so nothing passes *through* this
        # class, and `super().__init__(model=...)` or `super().__init__(temperature=...)` is an
        # ordinary "unexpected keyword argument".
        #
        # The default is the policy itself (frozen, so a safe default argument; B008's target is
        # the mutable case), spelled here rather than as a private module constant because the
        # published signature prints the default's source expression, and `RetryPolicy()` is a
        # name a reader can import. A mapping is accepted so authored config
        # (`retry_policy: {max_retries: 2}`) reaches the same kind of policy a Python caller
        # builds; validation is the object's own.
        if isinstance(retry_policy, RetryPolicy):
            self._retry_policy = retry_policy
        elif isinstance(retry_policy, Mapping):
            fields = [f.name for f in dataclasses.fields(RetryPolicy)]
            unknown = [key for key in retry_policy if key not in fields]
            if unknown:
                raise TypeError(
                    f"retry_policy: unknown key(s) {', '.join(map(repr, unknown))}. A "
                    f"retry_policy mapping takes the RetryPolicy fields: {', '.join(fields)} — "
                    f"per-error retry counts go under limits={{RateLimitError: 3}}, keyed by "
                    f"the class."
                )
            try:
                self._retry_policy = RetryPolicy(**retry_policy)
            except TypeError as exc:
                # With the keys checked above, a TypeError is `RetryPolicy`'s own, and each says
                # a class is missing (a `limits` key, an `extra_retryable_exceptions` entry). A
                # mapping is how authored config arrives, and a config file can only write names,
                # so it gets the config note. Added for every mapping rather than guessed from
                # the keys (an implementation choice, not decided by the user): a Python caller
                # who passed a mapping can use the same fix.
                raise TypeError(
                    f"retry_policy: {exc} A config file can't hold classes, so set limits"
                    " and extra_retryable_exceptions in Python, with"
                    " retry_policy=RetryPolicy(...)."
                ) from exc
        else:
            raise TypeError(
                f"retry_policy must be a RetryPolicy or a mapping of its fields, got "
                f"{type(retry_policy).__name__}"
            )

    @abstractmethod
    def complete(self, messages: list[Any]) -> LLMResponse:
        """Execute a completion call against the LLM.

        The messages are the only input. Everything else the call needs — the model, the
        sampling parameters, credentials — is the backend's own state, set at construction and
        put on the request by this method.

        Args:
            messages: The conversation in OpenAI chat format, in *wire* form — the term
                used throughout ``jaz`` for what actually goes to the provider: the agent's
                message buffer with every reserved sidecar key stripped except those this
                backend lists in ``extra_message_keys``.

        Returns:
            An LLMResponse with the completion content and usage information.
        """
        ...

    async def acomplete(self, messages: list[Any]) -> LLMResponse:
        """Async version of complete(). Default: runs complete() in a thread.

        Subclasses should override this for native async I/O.
        """
        return await asyncio.to_thread(self.complete, messages)

    @property
    def retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        """Exception types a failed call is retried on by default; anything else is re-raised
        at once, unless ``RetryPolicy.extra_retryable_exceptions`` adds it. The provider can
        override this for one error (``LLMError.is_retryable``, from its ``x-should-retry``
        header).

        The default is jaz's transient classes: :class:`~jaz.exceptions.RateLimitError`,
        :class:`~jaz.exceptions.ServerError` (a server error or invalid response) and
        :class:`~jaz.exceptions.APIConnectionError` (a failed transport, timeouts
        included).
        Override to add a class your backend raises for a transient failure, for example
        ``(*super().retryable_exceptions, httpx.ReadTimeout)``.
        """
        # An allow-list, not a list of hopeless errors (the user's call): an error nobody
        # declared fails fast, so a bug, a deterministic setup failure or a cancellation is never
        # retried, while a backend that does not map a transient failure into jaz's classes can
        # still declare it here. The provider SDKs and agent frameworks retry the same way, by
        # their own transient classes
        # (https://github.com/stanfordnlp/dspy/blob/3.4.0/dspy/utils/exceptions.py#L263;
        # the `openai` SDK by a status allow-list,
        # https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L907-L922).
        # A subclass of a retryable class is retryable too; a
        # backend that wants one to fail fast gives it a different base.
        #
        # A tuple, while `RetryPolicy.extra_retryable_exceptions` is any `Collection` (the
        # user's call): the type follows the role. This one is mostly read, by the retry loop
        # and possibly by user code, and a tuple is what `isinstance` and `except` take; it is
        # written by spreading `super()`'s value, so the one-element trailing comma rarely bites.
        # The policy field is written by users, often with one class, and only jaz reads it.
        return (RateLimitError, ServerError, APIConnectionError)

    def _retryer(
        self, on_retry: Callable[[RetryCallState], None] | None
    ) -> Callable[[Any], Any]:
        """Build the tenacity retry decorator shared by ``_complete_with_retry``
        and its async mirror ``_acomplete_with_retry`` — tenacity wraps sync and
        async callables transparently. Falls back to the default WARNING logger
        when the caller supplies no ``on_retry``.

        Every count is a cap on failures that may be retried (N -> up to N retries, N+1
        calls): ``max_retries`` caps every error, and each ``retry_policy.limits`` key caps
        the errors that are instances of its class. A failure counts against every cap it
        falls under, and the call raises as soon as any is exceeded. A failure that is not an
        instance of :attr:`retryable_exceptions` or ``retry_policy.extra_retryable_exceptions``
        raises at once, and one that is retries, unless it is an
        :class:`~jaz.exceptions.LLMError` whose ``is_retryable`` says otherwise.

        The wait before a retry is the ``retry_after`` the provider asked for, else the
        exponential backoff capped at ``wait_max``, lengthened at random by up to a quarter. A
        ``retry_after`` longer than ``retry_after_max`` is not shortened: the call raises the
        error at once, and its ``retry_after`` says when to try again.
        """
        # Two rules, both the user's calls, and both "every rule that applies must permit it".
        # Retryable: a failure is retried if it is an instance of the backend's
        # `retryable_exceptions` or the policy's `extra_retryable_exceptions`, unless the
        # provider's verdict (`is_retryable`, below) overrides that. Caps:
        # `max_retries` is the allowance of "every error" (enforced as `stop_after_attempt`) and
        # each key's value is its own; each failure spends one from every allowance it falls
        # under, and the call raises as soon as one is overdrawn. Matched with `isinstance`, so a
        # key reaches subclasses and a provider class that merely shares a jaz name
        # (`openai.RateLimitError`) does not.
        settings = self._retry_policy
        max_retries = settings.max_retries
        retryable = (*self.retryable_exceptions, *settings.extra_retryable_exceptions)
        remaining = dict(settings.limits)

        def should_retry(retry_state: RetryCallState) -> bool:
            # tenacity calls this exactly once per completed attempt (BaseRetrying._begin_iter),
            # so each failure is spent exactly once.
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if exc is None:
                return False
            # The provider's own verdict (`x-should-retry`) beats the class, as in the `openai`
            # and `anthropic` SDKs
            # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L896-L904,
            # https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_base_client.py#L865-L873),
            # for every class alike, `QuotaExceededError` included; the caps below still apply.
            # The user's call (2026-10-13): an earlier version ignored the verdict on a quota
            # error in both directions, which no provider's behaviour motivated and which the
            # user had never decided.
            verdict = exc.is_retryable if isinstance(exc, LLMError) else None
            if verdict is False:
                return False
            if verdict is None and not isinstance(exc, retryable):
                return False
            # A wait longer than the caller allows is not shortened: the provider asked not to
            # be called sooner, so the call fails with the error, whose `retry_after` says when
            # to try again (see `wait` below).
            retry_after = exc.retry_after if isinstance(exc, LLMError) else None
            if retry_after is not None and retry_after > settings.retry_after_max:
                # Logged only while `max_retries` allows another attempt, as the `openai` SDK
                # logs its own stop only when retries remain
                # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L887-L893,
                # reached only when `remaining_retries > 0`:
                # https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L1197),
                # so `max_retries=0` stays quiet. A spent `limits` key may still log it, pointing
                # at `retry_after_max` when raising it alone would not bring the retry back.
                # Gating on the caps too (`all(remaining[cls] > 0 for cls in covering)`) would
                # be one condition, but a second copy of the cap rule below that could drift
                # from it; the warning stays true either way and needs both a spent cap and an
                # over-long wait on one error (an implementation choice, not decided by the
                # user).
                if retry_state.attempt_number < attempt_cap:
                    logger.warning(
                        "Not retrying %s: the provider asked to wait %gs, longer than "
                        "RetryPolicy.retry_after_max (%gs).",
                        type(exc).__name__,
                        retry_after,
                        settings.retry_after_max,
                    )
                return False
            covering = [cls for cls in remaining if isinstance(exc, cls)]
            for cls in covering:
                remaining[cls] -= 1
            return all(remaining[cls] >= 0 for cls in covering)

        # The "every error" allowance: at most `max_retries` retries in all. Also the default
        # retry log's denominator.
        # TODO(#1589): as the denominator it is only an upper bound, since a smaller allowance
        # can stop the call sooner.
        attempt_cap = max_retries + 1

        backoff = wait_exponential(
            multiplier=settings.wait_multiplier,
            min=settings.wait_min,
            max=settings.wait_max,
        )

        def wait(retry_state: RetryCallState) -> float:
            # The provider's `Retry-After` when it gave one, else the backoff capped at `wait_max`,
            # plus jitter of up to a quarter so clients that failed together do not retry together.
            # A `Retry-After` is not capped by `wait_max`, which bounds only the backoff; it is
            # bounded by `retry_after_max` instead (see `RetryPolicy`). Jitter is added after the
            # cap, as LiteLLM's Router does (it adds a fixed 0-0.75 s, `_calculate_retry_after`;
            # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L7237-L7250,
            # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/constants.py#L497), not
            # subtracted, as the `openai`/`anthropic` SDKs do from their backoff
            # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L881,
            # https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_base_client.py#L859):
            # below the cap a retry never comes sooner than the provider or `wait_min` asked, and
            # clients at the cap still spread out.
            #
            # A `Retry-After` beyond `retry_after_max` stops the retries: `should_retry` fails the
            # call with the error instead (the user's call, 2026-10-12, reversing the 2026-10-11
            # call to wait the cap and retry). OpenAI's rate-limits guide says "If a valid server
            # delay exceeds the supported or configured maximum retry delay, stop retrying and
            # defer the request rather than retrying sooner"
            # (https://developers.openai.com/api/docs/guides/rate-limits), and the `openai` SDK
            # does so from 2.52 above 120 s (openai-python #3555,
            # https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L887).
            # Capping and retrying, the earlier choice, sent requests the provider had asked not
            # to receive (up to 10 over about 11 minutes for a one-hour wait). The other peers:
            # LiteLLM's Router
            # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L7240),
            # the Vercel AI SDK
            # (https://github.com/vercel/ai/blob/ai%407.0.137/packages/ai/src/util/retry-with-exponential-backoff.ts#L52)
            # and the `openai` SDK through 2.51
            # (https://github.com/openai/openai-python/blob/v2.51.0/src/openai/_base_client.py#L801)
            # ignore a long value and use their short backoff; the `anthropic` (1.13)
            # (https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_base_client.py#L847)
            # and Mistral
            # (https://github.com/mistralai/client-python/blob/v3.2.0/src/mistralai/client/utils/retries.py#L186-L191)
            # SDKs wait whatever it says; pydantic-ai caps it at 300 s and keeps retrying
            # (https://github.com/pydantic/pydantic-ai/blob/v2.55.0/pydantic_ai_slim/pydantic_ai/retries.py#L514-L515).
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            retry_after = exc.retry_after if isinstance(exc, LLMError) else None
            base = (
                retry_after
                if retry_after is not None and retry_after > 0
                else backoff(retry_state)
            )
            return base * (1 + 0.25 * random.random())

        return retry(
            reraise=True,
            stop=stop_after_attempt(attempt_cap),
            wait=wait,
            before_sleep=on_retry
            if on_retry is not None
            else _default_retry_logger(attempt_cap),
            retry=should_retry,
        )

    def _complete_with_retry(
        self,
        messages: list[Any],
        *,
        on_retry: Callable[[RetryCallState], None] | None = None,
    ) -> LLMResponse:
        """``complete`` under this backend's retry policy (the agent's call path)."""
        # Framework-only: the agent calls this, a backend author never does and never overrides
        # it (the retry *policy* is the override point). It passes nothing but the messages: a
        # backend's request params are its own, put on the call by its own `complete`.
        return self._retryer(on_retry)(self.complete)(messages)

    async def _acomplete_with_retry(
        self,
        messages: list[Any],
        *,
        on_retry: Callable[[RetryCallState], None] | None = None,
    ) -> LLMResponse:
        """Async mirror of ``_complete_with_retry`` — wraps ``acomplete``."""
        return await self._retryer(on_retry)(self.acomplete)(messages)

    @functools.cached_property
    def model_info(self) -> ModelInfo:
        """What this backend knows about its model — see :class:`ModelInfo`.

        The base knows nothing: no name, no input-token limit, no cost. Override to report what
        yours can; :class:`LiteLLM` answers from LiteLLM's own model map.
        """
        # A cached property, not a `get_` method: what a backend knows about its model is fixed
        # for the object's lifetime, so it is computed once and read like the static fact it is
        # (`llm.model_info.max_input_tokens`). A getter earns its verb by validating and
        # raising; nothing here does.
        return ModelInfo()
