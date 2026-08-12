"""Selecting and building the configured LLM backend.

The backend classes themselves live in :mod:`jaz.providers.llm` — this module is the thin layer
between ``Config.llm`` and them: it validates the tag, routes ``llm.params`` to the right
destination, and builds the instance.

``LLM`` and ``LLMResponse`` are re-exported here because this is where they lived before the
``LLMClient``/``Provider`` merge, and internal call sites import them from this path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from .providers.llm import (
    LLM,
    LLMResponse,
)
from .providers.registry import (
    register_llm,
    registered_llm_tags,
    resolve_llm,
)

__all__ = [
    "LLM",
    "LLMResponse",
    "MockLLMClient",
    "create_llm",
    "known_llm_tags",
    "register_llm",
    "validate_llm_tag",
]


def known_llm_tags() -> frozenset[str]:
    """Every tag :func:`create_llm` accepts.

    Computed rather than a constant because the registry is **open** — ``@register_llm`` lets a
    user add a backend without forking, which the README documents as the extension point. A
    frozen set here would silently exclude them.
    """
    return registered_llm_tags()


def validate_llm_tag(name: str, *, field: str = "llm.tag") -> None:
    """Raise ``ValueError`` if ``name`` is not a registered backend tag.

    Single source of truth for the message so the ``configure`` / ``ConfigOverride`` /
    depth-override / factory call sites can't drift. ``field`` names the offending setting
    (a depth override passes its indexed field path).
    """
    if name not in known_llm_tags():
        raise ValueError(
            f"Unknown {field} {name!r}. "
            f"Known backends: {', '.join(sorted(known_llm_tags()))}. "
            "Register one with @register_llm, or pass an LLM instance."
        )


def create_llm(tag: str, params: Mapping[str, Any] | None = None) -> LLM:
    """Build the backend named by ``tag``, configured from ``params``.

    The whole ``params`` mapping is handed to the backend's constructor, whose
    ``**request_defaults`` tail absorbs whatever it does not name.
    """
    # `REGISTRY[tag].from_dict(params)` with NO per-field special-casing at all — the whole bag
    # goes to the constructor, whose `**request_defaults` tail absorbs whatever is not a named
    # parameter.
    #
    # The pre-merge version carried four exceptions; merging the two class layers removed three,
    # and the last (`model` excluded by name, because it "rides per call") lived in
    # `split_llm_params` until the backend started holding its own `model`. That split existed
    # only to decide which keys of a tag-plus-bag config reached the constructor; with the
    # configured backend itself in the config there is no bag to partition.
    validate_llm_tag(tag)
    cls = resolve_llm(tag)
    assert cls is not None  # validate_llm_tag above guarantees it
    return cls.from_dict(params)


class MockLLMClient(LLM):
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

    Not registered under a tag, and deliberately so: it is constructed with a *callable*, so no
    authored config could name it. Passing the instance is the only way to select it — which is
    the ordinary way to select any backend.
    """

    #: Its own default, rather than the real one it would otherwise inherit from ``LLM``. A mock
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
                total_tokens=0,
                cached_tokens=0,
                cost=0.0,
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
                total_tokens=0,
                cost=0.0,
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
