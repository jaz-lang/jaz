"""Event definitions for the hook system.

Every event has a matching handler on :class:`jaz.hooks.Hook`, named after it
(:class:`LLMQueryEnter` → ``on_llm_query_enter``). A handler observes the event and returns a
list of effects; returning effects is the only way a hook influences the run. Which effects
are accepted depends on the event — each event class documents its own allowed set, and an
effect returned at an event outside that set raises
:class:`~jaz.exceptions.InvalidEffectError` (out-of-stage effects are hook bugs and fail
loudly, never silently no-op). A hook that treats every event alike overrides ``on_any``
instead::

    from jaz.hooks import Hook
    from jaz.hooks.effects import Abort, Effect
    from jaz.hooks.events import LLMQueryEnter

    class StopAfterTenTurns(Hook):
        def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
            if event.iteration >= 10:  # iterations are 0-based
                return [Abort(error=RuntimeError("too many turns"))]
            return []

Each span's events follow one pipeline of value states::

    proposal ──[edits]──▸ committed input ──[work or supply]──▸ raw result ──[modify]──▸ outcome

    *Enter     observes the proposal          controls edits + abort     fires always
    *Send      observes the committed input   controls supply + abort    fires iff the input committed
    *Complete  observes the raw result        controls modify + abort    fires iff a raw result exists
    *Exit      observes the outcome union     controls nothing (terminal) fires whenever the span opened

Events mark **value states becoming fixed**, not phases of the machinery: an event's
payload is the commit of the previous stage's effects, so no event can observe its own
composition's output — only the next event can carry it. ``*Send`` therefore fires on the
supplied/replayed path too (the input commit is real even when no provider is called), and
``*Exit`` carries a tagged outcome union (``Completed[...] | Aborted | Failed`` — the variant
IS the span status, its payload the **post-transform** result or the terminating exception),
so an interval that opened always closes, with the true outcome.

The events, by span:

- :class:`InvokeEnter` / :class:`InvokeSend` / :class:`InvokeComplete` / :class:`InvokeExit`
  — one quadruple per invoke.
- :class:`LLMQueryEnter` / :class:`LLMQuerySend` / :class:`LLMQueryComplete` /
  :class:`LLMQueryExit` — the per-turn LLM call.
- :class:`LLMQueryRetry` — fired per retry attempt of that call.
- :class:`REPLExecEnter` / :class:`REPLExecSend` / :class:`REPLExecComplete` /
  :class:`REPLExecExit` — the turn's REPL execution, which happens only when the response
  parsed into runnable code.

**Experimental.** The hook system is an experimental feature; its interfaces may change
in a future release.
"""

# The pipeline framing is design/design_features/span_event_lifecycle.md, fully landed:
# the four-stage lattice is live (suppliers compose at *Send, transformers at *Complete,
# *Exit is observation-only and always fires), and the abort model propagates an Abort's
# carried exception from the span CMs (the aborting invoke's spans close Aborted).

from ..base import Event
from .base import Aborted, Completed, Failed
from .invoke import (
    InvokeComplete,
    InvokeCompleteContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    InvokeContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    InvokeEnter,
    InvokeExit,
    InvokeOutcome,
    InvokeSend,
    InvokeSendContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    InvokeSpan,  # noqa: F401  # internal span wrapper; not in __all__
)
from .llm_query import (
    LLMQueryComplete,
    LLMQueryCompleteContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    LLMQueryContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    LLMQueryEnter,
    LLMQueryExit,
    LLMQueryOutcome,
    LLMQueryRetry,
    LLMQueryRetryContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    LLMQuerySend,
    LLMQuerySendContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    LLMQuerySpan,  # noqa: F401  # internal span wrapper; not in __all__
    MessageAddRecord,
    MessageDropRecord,
    ResolvedMessageEdits,
)
from .repl_execution import (
    REPLExecComplete,
    REPLExecCompleteContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    REPLExecContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    REPLExecEnter,
    REPLExecExit,
    REPLExecOutcome,
    REPLExecSend,
    REPLExecSendContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    REPLExecSpan,  # noqa: F401  # internal span wrapper (imported by core); not in __all__
)

# Public event surface (`jaz.hooks.events`) — the typed `event` a hook handler receives.
# `Event` is the base. The `*Context`/`*Span` variants are the dispatcher's internal
# effect-composition contracts (zero consumer usage) — imported above for reachability but
# deliberately kept OUT of __all__.
__all__ = [
    "Event",
    # Span-outcome variants (every *Exit's ``outcome`` is Completed[...] | Aborted | Failed)
    "Completed",
    "Aborted",
    "Failed",
    # Per-span outcome-union aliases
    "LLMQueryOutcome",
    "REPLExecOutcome",
    "InvokeOutcome",
    # REPL execution
    "REPLExecEnter",
    "REPLExecSend",
    "REPLExecComplete",
    "REPLExecExit",
    # LLM query
    "LLMQueryEnter",
    "LLMQuerySend",
    "LLMQueryComplete",
    "LLMQueryExit",
    # Resolved-edit records carried by LLMQuerySend
    "ResolvedMessageEdits",
    "MessageDropRecord",
    "MessageAddRecord",
    # Invoke
    "InvokeEnter",
    "InvokeSend",
    "InvokeComplete",
    "InvokeExit",
    # LLM retry
    "LLMQueryRetry",
]
