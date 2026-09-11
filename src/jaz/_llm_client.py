"""The mock LLM backend, plus the pre-merge import path for ``BaseLLM``/``LLMResponse``.

The backend classes themselves live in :mod:`jaz.llm.llm`; this module holds the callback-driven
:class:`MockLLMClient` and nothing else.

``BaseLLM`` and ``LLMResponse`` are re-exported here because this is where they lived before the
``LLMClient``/``Provider`` merge, and internal call sites import them from this path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from .llm.llm import (
    BaseLLM,
    LLMResponse,
)

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "MockLLMClient",
]


class MockLLMClient(BaseLLM):
    """LLM backend that delegates to a user-provided callback.

    The callback receives ``(model, messages, **kwargs)`` and returns either:

    * A ``str`` (convenience) — wrapped into an ``LLMResponse`` with zero
      token counts and zero cost.
    * An ``LLMResponse`` (full control) — returned as-is.

    Example::

        def my_llm(model, messages, **kwargs):
            last = messages[-1]["content"]
            if "summarize" in last:
                return "return 'summary'"
            return "return 42"

        mock = MockLLMClient(fn=my_llm)
        jaz.configure(llm=mock)

    No authored config can name it: it is constructed with a *callable*, so no config file
    could express one. Passing the instance is the only way to select it — which is
    the ordinary way to select any backend.
    """

    #: Its own default, rather than the real one it would otherwise inherit from ``BaseLLM``. A mock
    #: reporting ``gpt-5-mini`` would be priced and described as that model.
    model: str = "mock"

    def __init__(self, fn: Callable[..., str | LLMResponse], **retry: Any) -> None:
        super().__init__(**retry)
        self._fn = fn

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        result = self._fn(model, messages, **kwargs)
        if isinstance(result, str):
            return LLMResponse(
                content=result,
                prompt_tokens=0,
                completion_tokens=0,
                cached_tokens=0,
                cost_usd=0.0,
            )
        return result

    async def acomplete(
        self, model: str, messages: list[Any], **kwargs: Any
    ) -> LLMResponse:
        # Call _fn first, then check what we got back. This correctly handles
        # async callable objects, functools.partial wrapping a coroutine fn,
        # and async defs decorated without @wraps — all cases where
        # inspect.iscoroutinefunction() would return False but the call
        # still produces a coroutine.
        result = self._fn(model, messages, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, str):
            return LLMResponse(
                content=result,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
            )
        return result

    def get_model(self) -> str:
        return self.model or "mock"

    def get_model_info(self) -> dict[str, Any]:
        # No pricing table entry: the callback's cost is whatever it reports (0.0 for the
        # str convenience form), never computed from published rates.
        return {}

    def can_report_cost(self) -> bool:
        # The base answers from the pricing table, which has nothing for a mock; but a mock
        # always populates `cost`, so it can report one. Overridden rather than left to the
        # base so a cost budget under a mock is not rejected before the first call.
        return True

    def __deepcopy__(self, memo: dict[int, Any]) -> MockLLMClient:
        # Mock clients are shared external dependencies, not owned state.
        # ConfigOverride() deepcopies the config; we want the same mock instance.
        return self
