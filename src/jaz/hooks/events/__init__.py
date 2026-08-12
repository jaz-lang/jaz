"""Event definitions for the hook system.

Every event has a matching handler on :class:`jaz.hooks.Hook`, named after it
(:class:`LLMQueryEnter` → ``on_llm_query_enter``). A handler observes the event and returns a
list of effects; returning effects is the only way a hook influences the run. Which effects
are honored depends on the event — each event class documents its own allowed set, and
anything else is ignored. A hook that treats every event alike overrides ``on_any`` instead::

    from jaz.hooks import Hook
    from jaz.hooks.effects import Abort, Effect
    from jaz.hooks.events import LLMQueryEnter

    class StopAfterTenTurns(Hook):
        def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
            if event.iteration > 10:
                return [Abort(error=RuntimeError("too many turns"))]
            return []

The events, by span:

- :class:`InvokeEnter` / :class:`InvokeExit` — one pair per invoke.
- :class:`LLMQueryEnter` / :class:`LLMQueryExit` — the per-turn LLM call.
- :class:`LLMQueryRetry` — fired per retry attempt of that call.
- :class:`REPLExecEnter` / :class:`REPLExecExit` — the turn's REPL execution, which happens
  only when the response parsed into runnable code.

**Experimental.** The hook system is an experimental feature; its interfaces may change
in a future release.
"""

from ..base import Event
from .invoke import (
    InvokeContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    InvokeEnter,
    InvokeExit,
    InvokeExitContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    InvokeSpan,  # noqa: F401  # internal span wrapper; not in __all__
)
from .llm_query import (
    LLMQueryContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    LLMQueryEnter,
    LLMQueryExit,
    LLMQueryRetry,
    LLMQueryRetryContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    LLMQuerySpan,  # noqa: F401  # internal span wrapper; not in __all__
)
from .repl_execution import (
    REPLExecContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    REPLExecEnter,
    REPLExecExit,
    REPLExecExitContext,  # noqa: F401  # internal dispatcher contract; not in __all__
    REPLExecSpan,  # noqa: F401  # internal span wrapper (imported by core); not in __all__
)

# Public event surface (`jaz.hooks.events`) — the typed `event` a hook handler receives.
# `Event` is the base. The `*Context`/`*Span`/`*ExitContext` variants are the dispatcher's
# internal effect-composition contracts (zero consumer usage) — imported above for
# reachability but deliberately kept OUT of __all__.
__all__ = [
    "Event",
    # REPL execution
    "REPLExecEnter",
    "REPLExecExit",
    # LLM query
    "LLMQueryEnter",
    "LLMQueryExit",
    # Invoke
    "InvokeEnter",
    "InvokeExit",
    # LLM retry
    "LLMQueryRetry",
]
