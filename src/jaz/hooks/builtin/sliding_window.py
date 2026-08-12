"""Sliding window hook for managing LLM context window usage.

Drops old messages when the conversation exceeds a token budget,
keeping the system prompt and most recent messages.
"""

from __future__ import annotations

from typing import Any

from jaz.provenance import MessageKind, provenance_of

from ..base import Event
from ..dispatcher import Effect, Hook
from ..effects import DropMessages
from ..events import LLMQueryEnter


class SlidingWindow(Hook):
    """Drop old messages when context exceeds a fraction of the model's window.

    The system prompt and the initial user prompt (task + inputs) are always kept, as are the
    most recent messages that fit in the budget. Everything between them is dropped from the
    conversation — unlike :class:`~jaz.hooks.builtin.compaction.Compaction`, nothing is
    summarized first, so this hook leaves nothing in its place.

    Useful for long-running agents that do not delegate, where the conversation would
    otherwise outgrow the model's context window.

    Under ``with``, every invoke in scope trims its own conversation. Passed positionally,
    only that invoke's is trimmed.

    Args:
        fraction: Fraction of ``max_input_tokens`` to use as the message
            budget (e.g., 0.7 means keep messages fitting in 70% of the
            context window).
        max_input_tokens: The model's maximum input tokens (context window
            size). Obtain from ``get_model_info(model)["max_input_tokens"]``.

    Example::

        from jaz.hooks.builtin.sliding_window import SlidingWindow

        with SlidingWindow(fraction=0.7, max_input_tokens=272000):
            jaz.invoke(...)

    Raises:
        ValueError: If ``fraction`` is outside ``(0, 1]`` or ``max_input_tokens`` is not
            positive.
    """

    # The kept prefix is identified by *provenance* (``MessageKind.SEED``, stamped at seed
    # time — #599) rather than by role or position, so it stays pinned even if a persistent
    # message edit inserts a summary before the task or otherwise reorders the prefix. Matches
    # ``Compaction._compaction_span``.

    def __init__(self, fraction: float, max_input_tokens: int) -> None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if max_input_tokens <= 0:
            raise ValueError(
                f"max_input_tokens must be positive, got {max_input_tokens}"
            )
        self._budget = int(fraction * max_input_tokens)

    def on_any(self, event: Event) -> list[Effect]:
        match event:
            case LLMQueryEnter(messages=messages):
                estimated = _estimate_tokens(messages)
                if estimated > self._budget:
                    drop = _indices_to_drop(messages, self._budget)
                    if drop:
                        return [DropMessages(indices=drop)]
        return []


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Fast token estimate: ~4 chars per token."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            # Multimodal content parts
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += len(part["text"])
        total += 4  # Role/metadata overhead per message
    return total // 4


def _indices_to_drop(messages: list[dict[str, Any]], budget: int) -> set[int]:
    """Indices to drop to fit ``budget``.

    Pins the seed (system prompt + initial user prompt, plus any protocol-seeded prefix) by
    *provenance* (``MessageKind.SEED``, stamped at seed time — #599), not by role or fixed
    positions, so a persistent message edit that reorders the prefix (e.g. compaction
    inserting a summary before the task) can't push the seed into the droppable region.
    Keeps the pinned seed plus the most recent messages that fit in the remaining budget;
    everything in between is dropped. Returns positions into ``messages`` (so the result
    composes by union with other hooks' drops).

    No role-based fallback for un-stamped buffers: the core agent loop always stamps the
    seed before any query (``Agent._stamp_seed_provenance`` in ``invoke``/``ainvoke``), so
    every buffer that reaches this hook via the loop is stamped. An empty ``pinned`` set is
    therefore either a buffer whose seed a prior hook deliberately dropped — where
    fabricating a pin by role would resurrect content the edit removed — or an unstamped
    buffer a hook built by hand, for which pinning nothing and letting the recency window
    apply is acceptable. Either way, pin only what provenance marks.
    """
    n = len(messages)
    pinned = {
        i
        for i, m in enumerate(messages)
        if (p := provenance_of(m)) is not None and p.kind is MessageKind.SEED
    }
    if n - len(pinned) <= 0:
        return set()  # nothing droppable beyond the pinned prefix

    keep = set(pinned)
    remaining_budget = budget - _estimate_tokens([messages[i] for i in sorted(pinned)])
    if remaining_budget > 0:
        # Keep messages from the end until the budget is exceeded.
        tokens_used = 0
        for i in range(n - 1, -1, -1):
            if i in pinned:
                continue
            msg_tokens = _estimate_tokens([messages[i]])
            if tokens_used + msg_tokens > remaining_budget:
                break
            keep.add(i)
            tokens_used += msg_tokens

    return set(range(n)) - keep


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``). This hook is absent from every ``__all__``,
#: so that deep-path import is the only way it was ever reachable.
SlidingWindowHook = SlidingWindow
