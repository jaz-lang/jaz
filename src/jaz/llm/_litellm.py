"""The LiteLLM backend — route completions through the LiteLLM library.

Where a single-vendor backend class posts directly to one vendor's HTTP API,
:class:`LiteLLM` hands the call to :func:`litellm.completion`, so a single backend reaches
every provider LiteLLM routes to (OpenAI, Anthropic, Gemini, Bedrock, Vertex, …) and a model
released after this JAZ version is usable without a JAZ release.

Usage::

    jaz.configure(llm=LiteLLM(model="openai/gpt-5-mini"))

The model id is LiteLLM's routing key and is sent verbatim, so it carries a ``provider/`` prefix
(``openai/gpt-5-mini``, ``anthropic/claude-sonnet-5``, ``vertex_ai/gemini-2.5-pro``) — written
exactly as LiteLLM's own documentation writes it. The provider LiteLLM routes to also selects the
credential: a key stored for ``"openai"`` in ``~/.jaz/credentials.json`` is used for ``openai/…``
models — and for a bare ``gpt-5-mini``, since LiteLLM infers the provider when there is no prefix
— behind an explicitly passed ``api_key`` and behind the provider's environment variable. Cost is
computed by LiteLLM's calculator
(:func:`litellm.completion_cost`), not the bundled :mod:`jaz.llm.pricing` table, and
:meth:`~LiteLLM.can_report_cost` reflects LiteLLM's price map.

Limitations:
    * Importing ``litellm`` is slow (~2 s) and pulls a heavy dependency tree; this backend defers
      that import until the first call, so ``import jaz`` does not pay it.
    * Cache-*write* pricing needs ``litellm>=1.87.0`` (the release that began reporting
      ``cache_write_tokens``); an older LiteLLM prices writes at $0.
    * LiteLLM fetches its price map live, so reported cost can vary over time, and a model LiteLLM
      has not mapped reports no cost (``can_report_cost`` returns ``False`` for it).
    * A parameter LiteLLM recognizes is always transformed; ``allowed_openai_params`` forces it
      past validation but not past LiteLLM's mapping.
"""

from __future__ import annotations

# Module name is ``_litellm`` (private, like ``_llm_client``/``_agent``), deliberately NOT
# ``litellm.py``: a sibling module named ``litellm`` would shadow the ``litellm`` package under a
# relative import and confuse readers. The public name is the class ``LiteLLM``, imported from
# ``jaz.llm``; nothing imports this module directly.
from typing import Any

from ..credentials import resolve_credential
from .exceptions import (
    APIError,
    AuthenticationError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnsupportedParamsError,
)
from .llm import BaseLLM, LLMResponse
from .registry import register_llm


def _litellm() -> Any:
    """Import ``litellm`` on demand and return the module.

    Kept out of module scope so importing this file — which happens on ``import jaz``, since
    ``jaz.llm`` imports this module eagerly — does not pay LiteLLM's ~2 s import + heavy
    dependency tree. The cost is deferred to the first call/pricing check instead.
    """
    import litellm

    # Cosmetic: silence LiteLLM's per-call "Give Feedback / Provider List" banner on stderr.
    litellm.suppress_debug_info = True
    return litellm


def _mapped_exception(litellm: Any, exc: BaseException) -> Exception | None:
    """Translate a LiteLLM exception into JAZ's error taxonomy, or ``None`` to re-raise as-is.

    LiteLLM raises its *own* exception classes (which subclass the ``openai`` SDK's, not JAZ's).
    Without this translation the base retryer's ``isinstance(exc, non_retryable_exceptions)`` never
    matches, so exactly the failures the taxonomy exists to fail fast on — bad API key, content
    policy, context overflow, unknown model — get retried with backoff, and downstream
    ``except jaz.exceptions.X`` never sees the error either. Checked most-specific first: LiteLLM's
    ``ContextWindowExceeded`` / ``ContentPolicyViolation`` / ``UnsupportedParams`` all subclass
    ``BadRequestError``, which (with the rest) subclasses ``APIError``, so order is load-bearing.
    """
    le = litellm.exceptions
    status = getattr(exc, "status_code", None)
    provider = getattr(exc, "llm_provider", None) or "litellm"
    message = getattr(exc, "message", None) or str(exc)
    mapping: list[tuple[type, type]] = [
        (le.ContextWindowExceededError, ContextWindowExceededError),
        (le.ContentPolicyViolationError, ContentPolicyViolationError),
        (le.AuthenticationError, AuthenticationError),
        (le.PermissionDeniedError, PermissionDeniedError),
        (le.NotFoundError, NotFoundError),
        (le.RateLimitError, RateLimitError),
        (le.UnsupportedParamsError, UnsupportedParamsError),
        (le.BadRequestError, UnsupportedParamsError),
        (le.APIError, APIError),  # catch-all for the retryable server/connection errors
    ]
    for litellm_type, jaz_type in mapping:
        if isinstance(exc, litellm_type):
            return jaz_type(message, status_code=status, llm_provider=provider)
    return None


@register_llm("litellm")
class LiteLLM(BaseLLM):
    """LLM backend that routes completions through LiteLLM. See the module docstring for usage,
    the model-id convention, how cost is computed, and the limitations."""

    # Why a whole backend around what looks like one `litellm.completion` call: the executive
    # decision in design/design_features/litellm_sole_backend_v1.md makes LiteLLM the sole v1
    # backend, retiring the hand-written openai/anthropic transports. Each rule below (verbatim
    # dispatch, num_retries=0, price via completion_cost — never pricing.py, carry the cache
    # buckets, map exceptions into JAZ's taxonomy) is transcribed from that doc / its parent #1082.

    def __init__(
        self,
        *,
        allowed_openai_params: list[str] | None = None,
        drop_params: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Config-file escape hatches for a model too new for the *installed* LiteLLM: unblock a
        # param LiteLLM would reject (`allowed_openai_params`) or silently drop unknown ones
        # (`drop_params`) from a YAML/settings file rather than a code change. Held as attributes,
        # not folded into `request_defaults`, because they are LiteLLM-call controls — not model
        # params that belong on the wire.
        self.allowed_openai_params = allowed_openai_params
        self.drop_params = drop_params

    def validate_model(self, model: str) -> None:
        # No-op override: the `provider/` prefix is LiteLLM's routing key, not a JAZ tag to
        # reject. The base method's docstring documents exactly this transport-backend case.
        return None

    def _build_params(
        self, model: str, messages: list[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"model": model, "messages": messages}
        if self.allowed_openai_params is not None:
            params["allowed_openai_params"] = self.allowed_openai_params
        if self.drop_params is not None:
            params["drop_params"] = self.drop_params
        params.update(kwargs)
        # Forced after the merge so no config can re-enable LiteLLM's own retries: JAZ's retryer
        # is hook- and budget-aware, and a second retry loop underneath it would silently multiply
        # attempts and spend.
        params["num_retries"] = 0
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
        # Resolved PER REQUEST rather than in `__init__` like the single-vendor backends, for
        # FRESHNESS — not because the provider varies. It does not vary in practice: `get_model`
        # reads the instance's own `model`, and `Agent` caches it as fixed for its lifetime, so a
        # given backend serves one provider. What varies is the FILE. `Agent.__init__` does
        # `self.llm_client = self.config.llm`, reusing the configured instance rather than
        # rebuilding it, so a key resolved at construction is the only key that instance will ever
        # have. Per-request resolution is therefore the sole reason `set_credential(...)` followed
        # by an `invoke(...)` works without restarting the console — the "takes effect from the
        # next agent run" promise in `jaz.console.set_credential`. The single-vendor backends
        # resolve in `__init__` and consequently do NOT honour it (see `jaz/credentials.py`).
        #
        # Cost is not a consideration and should not become one: this runs immediately before a
        # network round-trip to an LLM, which is four to five orders of magnitude slower. Do not
        # add a cache for speed — a cache would reintroduce exactly the staleness this avoids.
        #
        # It is also why the key is injected here instead of riding the base class's open
        # `**request_defaults` tail, which would work mechanically (`_agent` splats those into
        # the call and LiteLLM forwards `api_key` verbatim). Putting it there would make the
        # secret ordinary construction config on the component — precisely what the credentials
        # store exists to prevent (see the executive call at the top of `jaz/credentials.py`:
        # the key must never be config *input*). Here it is read off disk at call time and dies
        # with the params dict.
        #
        # `request_defaults` is consulted as well as `params` because the two carry a
        # configured key on different paths: the agent splats `request_defaults` into the call
        # (so it arrives in `params`), but a direct `backend.complete(...)` does not. Checking
        # only `params` inverted the precedence for direct callers — their `LiteLLM(api_key=…)`
        # lost to the stored file.
        if params.get("api_key") or self.request_defaults.get("api_key"):
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
            # nothing is required), so ordering it first swallowed the unroutable case in a
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
        # variable a provider needs, so a multi-variable provider reports "not satisfied" while
        # its key IS exported: with `AZURE_API_KEY` set but `AZURE_API_BASE` unset, this still
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
        # than becoming a `jaz.llm.exceptions` type. That is the right shape — it is a
        # configuration error, not an API failure, so mapping it to `AuthenticationError` would
        # misdescribe it, and the message already names the offending path and what is wrong with
        # it. Failing loudly here matches `load_credentials`, which refuses to treat an unparseable
        # store as "no credential".
        return resolve_credential(str(provider))

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        litellm = _litellm()
        params = self._build_params(model, messages, kwargs)
        try:
            response = litellm.completion(**params)
        except Exception as exc:
            mapped = _mapped_exception(litellm, exc)
            if mapped is not None:
                raise mapped from exc
            raise
        return self._to_response(response)

    async def acomplete(
        self, model: str, messages: list[Any], **kwargs: Any
    ) -> LLMResponse:
        # Native async via litellm.acompletion, not the base's asyncio.to_thread(complete)
        # fallback — that fallback burns one blocking thread per in-flight call, which the
        # sole/default backend should not do. Shares _build_params and the exception mapping.
        litellm = _litellm()
        params = self._build_params(model, messages, kwargs)
        try:
            response = await litellm.acompletion(**params)
        except Exception as exc:
            mapped = _mapped_exception(litellm, exc)
            if mapped is not None:
                raise mapped from exc
            raise
        return self._to_response(response)

    def _to_response(self, response: Any) -> LLMResponse:
        litellm = _litellm()

        usage = getattr(response, "usage", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)

        extra: dict[str, Any] = {}
        # Cache *writes* go into the ATIF `cache_creation_input_tokens` bucket — the same place the
        # anthropic backend puts them — so downstream usage reporting is uniform across backends.
        cache_write = getattr(prompt_details, "cache_write_tokens", None)
        if cache_write:
            extra["cache_creation_input_tokens"] = cache_write
        reasoning = getattr(completion_details, "reasoning_tokens", None)
        if reasoning:
            extra["reasoning_tokens"] = reasoning

        # Price via LiteLLM's calculator, never `jaz.llm.pricing`: pricing a model too new
        # for the bundled snapshot is the whole reason this backend exists, so routing its cost
        # back through that snapshot would defeat it. `None` when LiteLLM cannot price the call, so
        # BudgetPool records it as uncosted rather than charging a false $0.
        try:
            cost: float | None = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = None

        content = None
        choices = getattr(response, "choices", None)
        if choices:
            content = getattr(getattr(choices[0], "message", None), "content", None)

        return LLMResponse(
            content=content,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            cached_tokens=getattr(prompt_details, "cached_tokens", None),
            cost_usd=cost,
            extra=extra,
            raw_response=response,
        )

    def can_report_cost(self) -> bool:
        # Ask LiteLLM's (live) price map, not the base's bundled table, since this backend prices
        # via `completion_cost`. A model LiteLLM has not mapped raises there → False, so a
        # `cost_budget` set on it is rejected pre-flight rather than found unenforceable mid-run.
        litellm = _litellm()
        # Resolve the model outside the try: an unconfigured model is a config error that must
        # surface as "no model configured", not be swallowed here and read as "unpriced model".
        model = self.get_model()
        try:
            info = litellm.get_model_info(model)
        except Exception:
            return False
        return info.get("input_cost_per_token") is not None
