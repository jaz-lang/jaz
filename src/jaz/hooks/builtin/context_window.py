"""Context-window hook: warn the agent as its prompt fills the window.

This is a **level** signal (each invoke's own loop), the counterpart to the **pool**
budgets owned by :class:`~jaz.hooks.builtin.budget_pool.BudgetPool`. Activate it via a
propagating channel — ``with ContextWindowHook(context_window_fraction=...):
jaz.invoke(...)`` — to guard an invoke and its nested invokes. It is advisory-only and
does nothing until ``context_window_fraction`` is set.

This hook is **advisory only** — it emits a per-turn warning prompt and nothing else:

- ``context_window_fraction`` — once ``prompt_tokens / max_input_tokens`` reaches this
  fraction the hook surfaces (via a transient ``AddMessages`` at ``on_llm_query_enter``)
  a warning (plus the shared
  ``must_exit_warning`` text) nudging the agent to finish before its prompt outgrows
  the model's context window.

There is **no** ``Abort`` and **no** ``ModifyResult`` here: the hook never
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
:class:`~jaz.hooks.builtin.iteration_limit.IterationLimit`. The two are
independent: on a turn that is both context-near-full and at the iteration warning,
the agent receives both warnings (union). ``BudgetPool``'s approaching-budget
warning is a *third* emitter of the same shared ``must_exit_warning`` text, so in the
rare turn that trips all three (context fraction + iteration threshold + budget
``warn_fraction``) the byte-identical default line can appear up to three times.
Deliberately accepted, not deduped — prompt composition stays purely additive;
the only byte-identical collision is the generic default warning (every other baseline
line embeds variable data and is already unique), so global dedup would be the wrong
layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jaz.hooks.builtin._must_exit import (
    deserialize_must_exit_warning,
    resolve_must_exit_warning,
    serialize_must_exit_warning,
)
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import AddMessages, Effect
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
)

if TYPE_CHECKING:
    from jaz.config import Config


@dataclass(eq=False, kw_only=True)
class ContextWindow(Hook):
    """Warn the agent as its prompt approaches the model's context window.

    Advisory only: it emits a warning into the prompt and never terminates the invoke. The
    model's own context-window limit remains the hard stop — the LLM call fails once the
    prompt exceeds it.

    Under ``with``, every invoke in scope is watched, each against its own prompt. Passed
    positionally, only that invoke is — this hook does not warn the invokes it nests.

    Args:
        context_window_fraction: The warning threshold. If set, warn once
            ``prompt_tokens / max_input_tokens`` reaches this fraction. ``None`` (the
            default) disables the hook.
        must_exit_warning: Agent-facing warning text emitted alongside the
            context-window warning. ``None`` (default) renders the stock warning, which
            tells the agent to finish because it is nearing its context-window limit.
            Otherwise a plain string, or a callable ``(can_delegate: bool) -> str`` invoked
            at emission time with the one per-invoke fact. An empty string / empty
            render emits nothing.
    """

    # Handles ``InvokeEnter``/``InvokeExit`` (per-invoke window + token state) and
    # ``LLMQueryEnter``/``LLMQueryExit`` (threshold check and token capture); emits
    # ``AddMessages``.
    #
    # The stock warning is deliberately finish-method-agnostic — it says to finish, not *how*,
    # because the finish mechanism is the REPL's concern, not this hook's.

    # A ``@dataclass(eq=False)`` — params derive from the fields; ``must_exit_warning`` carries
    # the shared (de)serializer as field metadata (see ``IterationLimit``). The per-invoke token
    # dicts are ``field(init=False)`` runtime state, excluded from ``to_dict``.

    context_window_fraction: float | None = None
    must_exit_warning: str | Callable[[bool], str] | None = field(
        default=None,
        metadata={
            "to_dict": serialize_must_exit_warning,
            "from_dict": deserialize_must_exit_warning,
        },
    )
    # Per-invoke state (keyed by invoke_id; safe to share one instance). init=False so it is
    # runtime state, not a construction param, and is excluded from serialization.
    _last_prompt_tokens: dict[str, int] = field(init=False, default_factory=dict)
    _max_input_tokens: dict[str, int] = field(init=False, default_factory=dict)

    # --- lifecycle ---

    def on_invoke_enter(self, event: InvokeEnter) -> list[Effect]:
        # Capture the context window from the invoke's EFFECTIVE config (#463). Emits
        # no effects — if the window can't be determined we simply can't warn for this
        # invoke (the LLM API's own context limit remains the hard backstop).
        if self.context_window_fraction is None:
            return []
        max_input_tokens = self._compute_max_input_tokens(event.config)
        if max_input_tokens is not None:
            self._max_input_tokens[event.invoke_id] = max_input_tokens
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        # Only track when the warning is on. This skips dead work on the default
        # (context_window_fraction=None) hot path AND bounds the per-invoke-state
        # leak: the dict entries are freed on InvokeExit, which the dispatcher emits
        # only when the span completed — so an invoke that dies before span.complete()
        # (an exception escaping the loop body) would otherwise leak an entry on this
        # process-global default hook. With nothing stored when fraction is None, that
        # common path can't leak at all.
        if self.context_window_fraction is None:
            return []
        if event.response.prompt_tokens is not None:
            self._last_prompt_tokens[event.invoke_id] = event.response.prompt_tokens
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        self._last_prompt_tokens.pop(event.invoke_id, None)
        self._max_input_tokens.pop(event.invoke_id, None)
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Advisory warning only (no result override, no termination): nudge the agent to
        # finish before its prompt outgrows the model's context window. The warning is an
        # AddMessages, which only composes at the LLM query.
        ratio = self._context_ratio(event.invoke_id)
        if (
            ratio is None
            or self.context_window_fraction is None
            or ratio < self.context_window_fraction
        ):
            return []

        tokens = self._last_prompt_tokens[event.invoke_id]
        max_input = self._max_input_tokens[event.invoke_id]
        warning = resolve_must_exit_warning(
            self.must_exit_warning,
            can_delegate=event.can_recurse,
        )
        # One trailing user message per hook (this hook's fragments joined, blanks dropped),
        # appended to the query (index defaults to the end). Different hooks' messages are
        # not coalesced with each other — see AddMessages.
        fragments = [
            f"WARNING: You have used {ratio * 100:.0f}% of the model's "
            f"context window ({tokens:,} / {max_input:,} tokens). Finish soon "
            "before the prompt exceeds the window.",
            warning,
        ]
        content = "\n".join(t for t in fragments if t)
        return [AddMessages([{"role": "user", "content": content}])]

    # --- helpers ---

    def _context_ratio(self, invoke_id: str) -> float | None:
        """Current prompt-tokens / max-input-tokens, or None if not applicable."""
        if self.context_window_fraction is None:
            return None
        tokens = self._last_prompt_tokens.get(invoke_id)
        max_input = self._max_input_tokens.get(invoke_id)
        if tokens is None or not max_input:
            return None
        return tokens / max_input

    @staticmethod
    def _compute_max_input_tokens(config: Config) -> int | None:
        # Sizes the context window from the invoke's EFFECTIVE config (carried on
        # InvokeEnter), not ambient get_config() — so context_window_fraction is
        # correct under ConfigOverride(llm={...}) / resolve_for_depth (#463).
        # No construction: the config holds the configured backend. This used to build one per
        # hook fire from `{tag, params}` — an HTTP-capable object created and thrown away on
        # every context-window check, just to read a pricing-table entry.
        info = config.llm.get_model_info()
        if not info:
            return None
        return info.get("max_input_tokens")


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``).
ContextWindowHook = ContextWindow
