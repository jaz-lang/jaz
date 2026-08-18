"""Context-window hook: warn the agent as its prompt fills the window.

This is a **level** signal (each invoke's own loop), the counterpart to the **pool**
budgets owned by :class:`~BudgetPool`. Activate it via a
propagating channel — ``with ContextWindowWarning(warn_fraction=0.8, warning_text=...):
jaz.invoke(...)`` — to guard an invoke and its nested invokes. It is advisory-only, and both
fields are required (there is no "disabled" state — you omit the hook to not warn).

This hook is **advisory only** — it emits a per-turn warning prompt and nothing else:

- ``warn_fraction`` — once ``prompt_tokens / max_input_tokens`` reaches this fraction the hook
  surfaces the caller's ``warning_text`` VERBATIM (via a transient ``AddMessages`` at
  ``on_llm_query_enter``) — no hook-authored framing, no token counts.

There is **no** ``Abort`` and **no** ``ModifyExecResult`` here: the hook never
terminates the invoke. The real hard stop is the *literal* context window — once the
prompt exceeds ``max_input_tokens`` the LLM API call itself fails. The warning just
gives the agent a chance to wrap up before hitting that wall.

The context window is sized from the invoke's **effective** config carried on
``InvokeEnter`` (not ambient ``get_config()``), so the threshold is correct under
``ConfigOverride(llm={...})`` / ``resolve_for_depth``. If
``max_input_tokens`` can't be determined for the configured model the hook simply
can't warn (the ratio is undefined); it degrades silently rather than raising, since
the model's own context limit is still the backstop.

This hook was split out of the former ``GovernanceHook``; the iteration cap
(which owns termination) now lives in
:class:`~IterationLimit`. The two are
independent: on a turn that is both context-near-full and at the iteration warning,
the agent receives both hooks' ``warning_text`` (union). ``BudgetPool``'s approaching-budget
warning is a *third* emitter of a caller-supplied ``warning_text``. Each hook emits ONLY the
text its caller wired in (no hook-authored framing), so if the same string is wired into all
three, the rare turn that trips all three thresholds (context fraction + iteration threshold +
budget ``warn_*_remaining``) shows it up to three times. Deliberately accepted, not deduped:
prompt composition stays purely additive and the hooks share no state, so global dedup would be
the wrong layer — a caller that minds the repetition wires distinct strings (or one hook).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import AddMessages, Effect
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
)
from jaz.hooks.events.base import Completed

if TYPE_CHECKING:
    from jaz.config import Config


@dataclass(eq=False)
class ContextWindowWarning(Hook):
    """Warn the agent as its prompt approaches the model's context window.

    Advisory only: it emits a warning into the prompt and never terminates the invoke. The
    model's own context-window limit remains the hard stop — the LLM call fails once the
    prompt exceeds it.

    Under ``with``, every invoke in scope is watched, each against its own prompt. Passed
    positionally, only that invoke is — this hook does not warn the invokes it nests.

    Args:
        warn_fraction: The warning threshold — warn once ``prompt_tokens / max_input_tokens``
            reaches this fraction. Required. Deliberately kept a FRACTION, unlike the absolute
            thresholds on ``IterationLimit`` / ``BudgetPool``: context windows span ~100x across
            models, so only a scale-free threshold is meaningful across models.
        warning_text: Agent-facing warning text emitted alongside the context-window warning.
            Required; used verbatim. An empty string emits nothing.
    """

    # Handles ``InvokeEnter``/``InvokeExit`` (per-invoke window + token state) and
    # ``LLMQueryEnter``/``LLMQueryExit`` (threshold check and token capture); emits
    # ``AddMessages``.

    # A ``@dataclass(eq=False)``: ``eq=False`` keeps identity semantics (hooks are deduped by
    # ``is``), and the generated ``__repr__`` is what observability consumers render. The
    # per-invoke token dicts carry ``repr=False`` as well as ``init=False``: ``init=False`` alone
    # keeps a field out of the constructor but NOT out of the generated ``__repr__``, so without
    # it every trace line would carry live per-invoke token counts.

    # Both fields are REQUIRED (no defaults), an executive call: this hook is opt-in (the
    # library's default ``baseline_hooks`` is empty), and a construction always states intent —
    # at what fraction to warn and what to say — rather than a bare ``ContextWindowWarning()``
    # silently either doing nothing or warning at some baked-in fraction. To not warn, omit the
    # hook. ``warning_text`` is a plain ``str`` (the callable ``(can_delegate: bool) -> str`` form
    # was dropped across all three governance hooks — its only in-tree user, the eval harness's
    # delegate-aware template, now renders to a fixed string at config time).
    warn_fraction: float
    warning_text: str

    # Per-invoke state (keyed by invoke_id; safe to share one instance). init=False so it is
    # runtime state, not a construction param, and is excluded from serialization.
    _last_prompt_tokens: dict[str, int] = field(
        init=False, default_factory=dict, repr=False
    )
    _max_input_tokens: dict[str, int] = field(
        init=False, default_factory=dict, repr=False
    )

    # --- lifecycle ---

    def on_invoke_enter(self, event: InvokeEnter) -> list[Effect]:
        # Capture the context window from the invoke's EFFECTIVE config (#463). Emits
        # no effects — if the window can't be determined we simply can't warn for this
        # invoke (the LLM API's own context limit remains the hard backstop).
        max_input_tokens = self._compute_max_input_tokens(event.config)
        if max_input_tokens is not None:
            self._max_input_tokens[event.invoke_id] = max_input_tokens
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        # ``warn_fraction`` is required now, so the hook is always active when constructed —
        # there is no ``fraction is None`` fast-path to skip tracking on.
        # Non-completed arms (#892 outcome union) carry no response — nothing to track.
        if not isinstance(event.outcome, Completed):
            return []
        response = event.outcome.result
        if response.prompt_tokens is not None:
            self._last_prompt_tokens[event.invoke_id] = response.prompt_tokens
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        self._last_prompt_tokens.pop(event.invoke_id, None)
        self._max_input_tokens.pop(event.invoke_id, None)
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Advisory warning only (no result override, no termination): once the prompt reaches
        # ``warn_fraction`` of the window, emit the caller's ``warning_text`` VERBATIM. There is no
        # hook-authored framing — no "you have used N%…" line; the message is exactly what the
        # caller supplied (an empty string emits nothing). The warning is an AddMessages, which
        # only composes at the LLM query.
        ratio = self._context_ratio(event.invoke_id)
        if ratio is None or ratio < self.warn_fraction or not self.warning_text:
            return []
        return [AddMessages([{"role": "user", "content": self.warning_text}])]

    # --- helpers ---

    def _context_ratio(self, invoke_id: str) -> float | None:
        """Current prompt-tokens / max-input-tokens, or None if not applicable."""
        tokens = self._last_prompt_tokens.get(invoke_id)
        max_input = self._max_input_tokens.get(invoke_id)
        if tokens is None or not max_input:
            return None
        return tokens / max_input

    @staticmethod
    def _compute_max_input_tokens(config: Config) -> int | None:
        # Sizes the context window from the invoke's EFFECTIVE config (carried on
        # InvokeEnter), not ambient get_config() — so warn_fraction is
        # correct under ConfigOverride(llm={...}) / resolve_for_depth (#463).
        # No construction: the config holds the configured backend. This used to build one per
        # hook fire from `{tag, params}` — an HTTP-capable object created and thrown away on
        # every context-window check, just to read a pricing-table entry.
        info = config.llm.get_model_info()
        if not info:
            return None
        return info.get("max_input_tokens")


#: Deprecated aliases for the pre-rename spellings — see the rationale block in
#: ``jaz/hooks/__init__.py``. ``ContextWindowHook`` is the original ``*Hook`` alias; ``ContextWindow``
#: is the spelling this hook carried before it gained a required ``warning_text`` and was renamed to
#: ``ContextWindowWarning``. Both keep the deep-path import working (the alias map is checked by
#: ``test_legacy_hook_names_still_importable``).
ContextWindowHook = ContextWindowWarning
ContextWindow = ContextWindowWarning
