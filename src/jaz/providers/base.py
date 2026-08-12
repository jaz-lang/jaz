"""Base types and abstract provider interface.

Provider-agnostic types for LLM completion requests and responses.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

# Type alias for OpenAI-style message format (replaces AllMessageValues)
MessageDict = dict[str, Any]


@dataclass
class Usage:
    """Token usage information from a completion call.

    `prompt_tokens` is the total input count INCLUDING cached tokens
    (OpenAI/ATIF convention). `cache_read_input_tokens` is a subset of
    `prompt_tokens`. Anthropic's `cache_creation_input_tokens` is also
    folded into `prompt_tokens` so `compute_cost` can subtract it out
    and price it at the cache-creation rate.

    `extra` carries provider-specific fields that aren't part of the
    normalized shape (e.g. OpenAI `reasoning_tokens`, Anthropic
    `cache_creation` 5m/1h sub-buckets, `service_tier`).
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A message in the completion response."""

    role: str
    content: str | None


@dataclass
class Choice:
    """A single choice/completion from the LLM response."""

    message: Message
    index: int = 0
    finish_reason: str | None = None


@dataclass
class CompletionResponse:
    """Normalized response from an LLM completion call.

    This structure is provider-agnostic and mirrors the OpenAI response format.
    """

    choices: list[Choice] = field(default_factory=list)
    usage: Usage | None = None
    model: str = ""
    id: str = ""


def declared_init_keys(cls: type) -> frozenset[str]:
    """The named (introspectable) parameters of ``cls.__init__``, excluding ``self``.

    Backs :meth:`jaz.repl.base.REPL.construction_keys`, so a component classifies authored keys by the
    same rule.

    Walks the MRO and unions each class's own declared parameters, because the common
    forwarding idiom (``def __init__(self, ..., **retry): super().__init__(**retry)``)
    hides the base's parameters behind a ``**kwargs`` that cannot be introspected. Without
    the walk, settings declared on a base class — such as ``LLM``'s ``retry_*``
    fields — would be invisible on every subclass that forwards them.

    ``*args``/``**kwargs`` are themselves never reported: a class whose settings exist
    *only* behind ``**kwargs`` (declared on no class in the MRO) must name them explicitly.
    """
    keys: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        keys.update(
            name
            for name, param in inspect.signature(init).parameters.items()
            if name != "self"
            and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        )
    return frozenset(keys)


# The `Provider` ABC that used to live here was merged into `LLM` (see `jaz.providers.llm`):
# one class per backend now owns the HTTP call, retry, cost accounting and model metadata,
# instead of splitting them across a client and a provider that only ever paired with each
# other. This module keeps the wire-shaped types both halves already shared.
