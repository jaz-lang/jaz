"""A backend that answers from a callable: deterministic tests without a provider."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Mapping
from typing import Any

from .._component import repr_call
from ._base import BaseLLM
from ._types import LLMResponse, ModelInfo, RetryPolicy


class MockLLM(BaseLLM):
    """A backend that answers from a callable you supply — for tests and scripted runs.

    Every other backend talks to a provider; this one calls ``fn`` with the wire-form
    messages and returns what it answers, so an agent can be exercised with no API key, no
    network, and no nondeterminism. It is a full backend: it goes through ``jaz.configure``
    and ``jaz.ConfigOverride`` like any other and reports cost, so ``BudgetPool`` and the other
    hooks see it exactly as they would a real one.

    Retries are off by default (a raising callback is a failing test, not a flaky provider);
    pass ``retry_policy=`` to exercise the retry path itself.

    ``fn`` receives the messages (the conversation in OpenAI chat format, in wire form) and
    returns either a ``str`` — the assistant's reply, wrapped into an :class:`LLMResponse`
    with zero token counts and zero cost — or an :class:`LLMResponse`, used as is (set
    ``cost_usd`` yourself if a cost budget should count it). ``fn`` may be a coroutine
    function; ``acomplete`` awaits its result. It may raise: a ``jaz.exceptions`` LLM error
    is classified for retry like a provider's, anything else propagates.

    Example::

        from jaz.llm import MockLLM

        def scripted(messages):
            last = messages[-1]["content"]
            if "summarize" in last:
                return "return 'summary'"
            return "return 42"

        with jaz.ConfigOverride(llm=MockLLM(scripted)):
            assert jaz.invoke(task="add 40 and 2") == 42

    A canned sequence is one lambda: ``replies = iter([...]); MockLLM(lambda m: next(replies))``.

    It serves no model, so ``model_info.name`` is ``None``: observability hooks label nothing.
    No authored config can name it — it is constructed with a callable, which no config file
    can express — so passing the instance is the only way to select it.
    """

    def __init__(
        self,
        fn: Callable[[list[Any]], str | LLMResponse],
        *,
        # Spelled inline, not as a private module constant: the published signature prints
        # the default's source expression, and `RetryPolicy(max_retries=0)` is a name a reader
        # can import. Frozen, so B008's mutable-default target does not apply.
        retry_policy: RetryPolicy | Mapping[str, Any] = RetryPolicy(max_retries=0),  # noqa: B008
    ) -> None:
        # No retries by default, unlike a provider backend: a callback that raises a retryable
        # error does so because the test says so, and retrying it eleven times behind
        # exponential backoff turns a failed test into a five-minute hang. A test that is *about* retry
        # passes `retry_policy=` explicitly; everything else wants the first answer.
        super().__init__(retry_policy=retry_policy)
        if not callable(fn):
            raise TypeError(
                f"MockLLM(fn=...) takes a callable that maps the messages to a reply, got "
                f"{type(fn).__name__}. For a canned sequence: "
                f"replies = iter([...]); MockLLM(lambda messages: next(replies))."
            )
        self._fn = fn

    def __repr__(self) -> str:
        return repr_call(
            "MockLLM", ("fn", self._fn), ("retry_policy", self._retry_policy)
        )

    @staticmethod
    def _as_response(result: Any) -> LLMResponse:
        if isinstance(result, LLMResponse):
            return result
        if isinstance(result, str):
            return LLMResponse(
                content=result,
                prompt_tokens=0,
                completion_tokens=0,
                cached_tokens=0,
                cost_usd=0.0,
            )
        raise TypeError(
            f"MockLLM's fn returned {type(result).__name__}; it must return the reply as a "
            f"str or an LLMResponse."
        )

    def complete(self, messages: list[Any]) -> LLMResponse:
        return self._as_response(self._fn(messages))

    async def acomplete(self, messages: list[Any]) -> LLMResponse:
        # Call, then look at what came back: that also covers an async callable object, a
        # `functools.partial` over a coroutine function, and an `async def` decorated without
        # `@wraps` — every case `inspect.iscoroutinefunction` would miss.
        result = self._fn(messages)
        if asyncio.iscoroutine(result):
            result = await result
        return self._as_response(result)

    @functools.cached_property
    def model_info(self) -> ModelInfo:
        # No name — nothing is served — but the str path always sets `cost_usd`, so a cost
        # budget is not rejected pre-flight; an LLMResponse without a cost trips the hook's
        # backstop after one turn, which is the honest outcome for a reply that priced nothing.
        return ModelInfo(can_report_cost=True)

    def __deepcopy__(self, memo: dict[int, Any]) -> MockLLM:
        # `ConfigOverride` deep-copies the config it derives from. A mock is shared test
        # state (call counts, a scripted iterator), so the copy must be the same object.
        return self
