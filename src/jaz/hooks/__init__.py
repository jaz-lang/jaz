"""Hook system: observe and influence a running agent through events and effects.

The mental model in three sentences. As an agent runs, the framework fires typed
**events** at fixed points: once around the whole invoke, and per turn around the LLM
query and the REPL execution — each of those three spans walks the same four stages,
``Enter → Send → Complete → Exit`` (see :mod:`jaz.hooks.events` for the pipeline). A
**hook** is a class with a method per event it cares about (:class:`LLMQueryEnter` →
``on_llm_query_enter``); each handler receives the event and returns a list of
**effects** — the only way a hook changes anything (:mod:`jaz.hooks.effects` says which
effect is valid at which stage). Effects from all active hooks are composed
order-independently and applied by the framework; a hook that returns ``[]`` everywhere
is a pure observer.

Public surface (three parts):

- ``jaz.hooks`` — the authoring primitive :class:`Hook` and the ready-made
  "battery" hooks (:class:`PrintLogger`, :class:`BudgetPool`, :class:`RecursionLimit`,
  :class:`ReturnType`, ...).
- ``jaz.hooks.events`` — the typed events a hook handler *receives* (:class:`InvokeEnter`,
  :class:`REPLExecExit`, ...).
- ``jaz.hooks.effects`` — the typed effects a hook handler *returns* (:class:`ModifyExecResult`,
  :class:`AddMessages`, :class:`Abort`, ...).

A complete first hook — count the turns, nudge the agent, then stop it::

    from jaz import invoke
    from jaz.hooks import Hook
    from jaz.hooks.effects import Abort, AddMessages, Effect
    from jaz.hooks.events import LLMQueryEnter

    class TurnLimiter(Hook):
        def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
            if event.iteration >= 10:
                return [Abort(error=RuntimeError("out of turns"))]
            if event.iteration >= 8:
                return [AddMessages([{"role": "user", "content": "Finish soon."}])]
            return []

    # `with` activation propagates to nested invokes:
    with TurnLimiter():
        result = invoke(task="...")

    # Passing a hook positionally scopes it to that one invoke — nested invokes
    # do NOT see it. See the Hook class for the difference.
    result = invoke(TurnLimiter(), task="...")

Before writing your own, check the battery hooks — limits, budgets, logging, tracing and
replay are already covered.

**Experimental.** The hook system is an experimental feature; its interfaces may change
in a future release.
"""

# The event/effect *vocabulary* deliberately lives in the ``events`` / ``effects``
# sub-namespaces, not flat on ``jaz.hooks`` — an Event or Effect only has meaning inside a
# hook, so it belongs to the hook system's grammar, scoped under it. (The names are still
# re-exported flat for back-compat, but the blessed path is ``jaz.hooks.events.X`` /
# ``jaz.hooks.effects.X``.)

from typing import TYPE_CHECKING

from .._warnings import make_lazy_getattr

# Public sub-namespaces — the hook vocabulary (events you receive, effects you return),
# advertised in __all__ (the docs generator discovers them there).
from . import effects, events

# Public authoring surface: the `Hook` primitive + the ready-made "battery" hooks. The
# hook *vocabulary* lives in the events/effects sub-namespaces (jaz.hooks.events /
# jaz.hooks.effects). The dispatch engine (`HookDispatcher`/`get_dispatcher`), the
# internal contract base (`ExecutionContext`), the `Blackboard` store type, and
# `MetaData`/`WorkflowReplay` are demoted (see `_DEMOTED`:
# reachable via `__getattr__`, with a NonPublicAPIWarning).
#
# `TrajectoryRecorder`/`TrajectoryReplay` are public here as of 2026-08-15 (user call),
# superseding the review-of-#962 demotion of the replay hook and the recorder's
# deep-path-only status: ATIF is the canonical trace/replay/cost format (#813, #1159), so its
# write/read pair belongs on the offered surface rather than behind deep imports every consumer
# had to know.
#
# The pair has been renamed twice. At promotion the replay hook went `Replay` -> `ATIFReplay`
# (user call, same review): `RolloutRecorder` already makes token-native rollouts a second
# recorded format, so the bare `Replay` would collide with a future token-native one, and a
# shared prefix keeps the write/read pair adjacent in `__all__` and on the docs page. Later
# `ATIFTrace`/`ATIFReplay` became `TrajectoryRecorder`/`TrajectoryReplay`: the prefix argument
# survives (they still share one), but ATIF is the *format* these read and write, not what they
# are, and `Trace` named the artifact where an actor was meant. `Trajectory` rather than `Trace`
# because `OTelTracing` already owns "trace" in this namespace, in the industry sense.
from .builtin import (
    BudgetForcing,
    BudgetPool,
    ContextWindowWarning,
    FileLogger,
    IterationLimit,
    PrintLogger,
    RecursionLimit,
    ReturnType,
    RolloutRecorder,
    TrajectoryDirectoryRecorder,
    TrajectoryRecorder,
    TrajectoryReplay,
    ValidateREPLCode,
    ValidateReturn,
)
from .dispatcher import Hook

if TYPE_CHECKING:
    # Static types for the reachable-but-demoted names (resolved lazily at runtime via
    # __getattr__, which warns). Not bound at runtime; type checkers read this block.
    from .base import ExecutionContext as ExecutionContext
    from .blackboard import Blackboard as Blackboard
    from .builtin import MetaData as MetaData
    from .builtin import WorkflowReplay as WorkflowReplay
    from .dispatcher import HookDispatcher as HookDispatcher
    from .dispatcher import get_dispatcher as get_dispatcher
    from .effects import UNSET as UNSET
    from .effects import Abort as Abort
    from .effects import AddInputs as AddInputs
    from .effects import AddMessages as AddMessages
    from .effects import AddVariables as AddVariables
    from .effects import BlackboardWrite as BlackboardWrite
    from .effects import DeleteCode as DeleteCode
    from .effects import DisableRecursion as DisableRecursion
    from .effects import DropInputs as DropInputs
    from .effects import DropMessages as DropMessages
    from .effects import DropVariables as DropVariables
    from .effects import Effect as Effect
    from .effects import InsertCode as InsertCode
    from .effects import ModifyExecResult as ModifyExecResult
    from .effects import ModifyInvokeResult as ModifyInvokeResult
    from .effects import ModifyLLMResponse as ModifyLLMResponse
    from .effects import SupplyExecResult as SupplyExecResult
    from .effects import SupplyInvokeResult as SupplyInvokeResult
    from .effects import SupplyLLMResponse as SupplyLLMResponse
    from .events import Aborted as Aborted
    from .events import Completed as Completed
    from .events import Event as Event
    from .events import Failed as Failed
    from .events import InvokeComplete as InvokeComplete
    from .events import InvokeCompleteContext as InvokeCompleteContext
    from .events import InvokeContext as InvokeContext
    from .events import InvokeEnter as InvokeEnter
    from .events import InvokeExit as InvokeExit
    from .events import InvokeOutcome as InvokeOutcome
    from .events import InvokeSend as InvokeSend
    from .events import InvokeSendContext as InvokeSendContext
    from .events import InvokeSpan as InvokeSpan
    from .events import LLMQueryComplete as LLMQueryComplete
    from .events import LLMQueryCompleteContext as LLMQueryCompleteContext
    from .events import LLMQueryContext as LLMQueryContext
    from .events import LLMQueryEnter as LLMQueryEnter
    from .events import LLMQueryExit as LLMQueryExit
    from .events import LLMQueryOutcome as LLMQueryOutcome
    from .events import LLMQueryRetry as LLMQueryRetry
    from .events import LLMQueryRetryContext as LLMQueryRetryContext
    from .events import LLMQuerySend as LLMQuerySend
    from .events import LLMQuerySendContext as LLMQuerySendContext
    from .events import LLMQuerySpan as LLMQuerySpan
    from .events import REPLExecComplete as REPLExecComplete
    from .events import REPLExecCompleteContext as REPLExecCompleteContext
    from .events import REPLExecContext as REPLExecContext
    from .events import REPLExecEnter as REPLExecEnter
    from .events import REPLExecExit as REPLExecExit
    from .events import REPLExecOutcome as REPLExecOutcome
    from .events import REPLExecSend as REPLExecSend
    from .events import REPLExecSendContext as REPLExecSendContext
    from .events import REPLExecSpan as REPLExecSpan

# Conditionally import tracing hooks
try:
    from .builtin import JaegerTracing, LangfuseTracing, OTelTracing  # noqa: F401

    _TRACING_AVAILABLE = True
except ImportError:
    _TRACING_AVAILABLE = False

__all__ = [
    # Public sub-namespaces (the hook vocabulary)
    "events",
    "effects",
    # Authoring primitive
    "Hook",
    # Concrete built-in hooks ("batteries")
    # Writers before the reader: the two hooks that WRITE a trajectory, then the one that
    # reads one back, which is the order the docs page renders. (This block was alphabetical
    # when the pair was `TrajectoryRecorder`/`TrajectoryReplay` and the two rules agreed;
    # `TrajectoryDirectoryRecorder` sorts before both, so only the write-before-read rule
    # survives.)
    "TrajectoryRecorder",
    "TrajectoryDirectoryRecorder",
    "TrajectoryReplay",
    "BudgetForcing",
    "BudgetPool",
    "ContextWindowWarning",
    "IterationLimit",
    "RecursionLimit",
    "PrintLogger",
    "FileLogger",
    "ReturnType",
    "RolloutRecorder",
    "ValidateReturn",
    "ValidateREPLCode",
]

if _TRACING_AVAILABLE:
    # `OTelTracing` alongside its two presets: it is the base a caller subclasses to export to
    # any other OTLP collector, so the extension seam has to be reachable by the same route as
    # the batteries. It was previously deep-path-only, which meant the one tracing name a user
    # is expected to subclass was the only one they could not import from here — and silently,
    # since a module read warns nowhere (see `jaz._warnings.warn_on_nonpublic_submodules`,
    # installed on the top-level package only).
    __all__.append("OTelTracing")
    __all__.append("JaegerTracing")
    __all__.append("LangfuseTracing")

# Reachable-but-unsupported names (warn on access; blessed homes noted). The hook
# vocabulary is public under jaz.hooks.events / jaz.hooks.effects — the flat
# `jaz.hooks.<name>` spelling of it is demoted; the dispatch engine / internal machinery
# has no public home.
_DEMOTED = {
    "ExecutionContext": ("jaz.hooks.base", "ExecutionContext"),
    "Blackboard": ("jaz.hooks.blackboard", "Blackboard"),
    "MetaData": ("jaz.hooks.builtin", "MetaData"),
    "WorkflowReplay": ("jaz.hooks.builtin", "WorkflowReplay"),
    "HookDispatcher": ("jaz.hooks.dispatcher", "HookDispatcher"),
    "get_dispatcher": ("jaz.hooks.dispatcher", "get_dispatcher"),
    # Effect vocabulary — blessed path: jaz.hooks.effects.<name>
    "Effect": ("jaz.hooks.effects", "Effect"),
    "Abort": ("jaz.hooks.effects", "Abort"),
    "SupplyExecResult": ("jaz.hooks.effects", "SupplyExecResult"),
    "ModifyExecResult": ("jaz.hooks.effects", "ModifyExecResult"),
    "ModifyInvokeResult": ("jaz.hooks.effects", "ModifyInvokeResult"),
    "SupplyInvokeResult": ("jaz.hooks.effects", "SupplyInvokeResult"),
    "DisableRecursion": ("jaz.hooks.effects", "DisableRecursion"),
    "AddInputs": ("jaz.hooks.effects", "AddInputs"),
    "DropInputs": ("jaz.hooks.effects", "DropInputs"),
    "AddVariables": ("jaz.hooks.effects", "AddVariables"),
    "DropVariables": ("jaz.hooks.effects", "DropVariables"),
    "InsertCode": ("jaz.hooks.effects", "InsertCode"),
    "DeleteCode": ("jaz.hooks.effects", "DeleteCode"),
    "DropMessages": ("jaz.hooks.effects", "DropMessages"),
    "AddMessages": ("jaz.hooks.effects", "AddMessages"),
    "SupplyLLMResponse": ("jaz.hooks.effects", "SupplyLLMResponse"),
    "ModifyLLMResponse": ("jaz.hooks.effects", "ModifyLLMResponse"),
    "UNSET": ("jaz.hooks.effects", "UNSET"),
    "BlackboardWrite": ("jaz.hooks.effects", "BlackboardWrite"),
    # Event vocabulary — blessed path: jaz.hooks.events.<name>
    "Event": ("jaz.hooks.events", "Event"),
    "Completed": ("jaz.hooks.events", "Completed"),
    "Aborted": ("jaz.hooks.events", "Aborted"),
    "Failed": ("jaz.hooks.events", "Failed"),
    "LLMQueryOutcome": ("jaz.hooks.events", "LLMQueryOutcome"),
    "REPLExecOutcome": ("jaz.hooks.events", "REPLExecOutcome"),
    "InvokeOutcome": ("jaz.hooks.events", "InvokeOutcome"),
    "REPLExecEnter": ("jaz.hooks.events", "REPLExecEnter"),
    "REPLExecSend": ("jaz.hooks.events", "REPLExecSend"),
    "REPLExecComplete": ("jaz.hooks.events", "REPLExecComplete"),
    "REPLExecExit": ("jaz.hooks.events", "REPLExecExit"),
    "LLMQueryEnter": ("jaz.hooks.events", "LLMQueryEnter"),
    "LLMQuerySend": ("jaz.hooks.events", "LLMQuerySend"),
    "LLMQueryComplete": ("jaz.hooks.events", "LLMQueryComplete"),
    "LLMQueryExit": ("jaz.hooks.events", "LLMQueryExit"),
    "InvokeEnter": ("jaz.hooks.events", "InvokeEnter"),
    "InvokeSend": ("jaz.hooks.events", "InvokeSend"),
    "InvokeComplete": ("jaz.hooks.events", "InvokeComplete"),
    "InvokeExit": ("jaz.hooks.events", "InvokeExit"),
    "LLMQueryRetry": ("jaz.hooks.events", "LLMQueryRetry"),
    # internal dispatcher contracts / span wrappers
    "InvokeContext": ("jaz.hooks.events", "InvokeContext"),
    "InvokeSendContext": ("jaz.hooks.events", "InvokeSendContext"),
    "InvokeCompleteContext": ("jaz.hooks.events", "InvokeCompleteContext"),
    "InvokeSpan": ("jaz.hooks.events", "InvokeSpan"),
    "LLMQueryContext": ("jaz.hooks.events", "LLMQueryContext"),
    "LLMQuerySendContext": ("jaz.hooks.events", "LLMQuerySendContext"),
    "LLMQueryCompleteContext": ("jaz.hooks.events", "LLMQueryCompleteContext"),
    "LLMQuerySpan": ("jaz.hooks.events", "LLMQuerySpan"),
    "LLMQueryRetryContext": ("jaz.hooks.events", "LLMQueryRetryContext"),
    "REPLExecContext": ("jaz.hooks.events", "REPLExecContext"),
    "REPLExecSendContext": ("jaz.hooks.events", "REPLExecSendContext"),
    "REPLExecCompleteContext": ("jaz.hooks.events", "REPLExecCompleteContext"),
    "REPLExecSpan": ("jaz.hooks.events", "REPLExecSpan"),
}

# No pre-rename hook spellings are kept. #806 question 1 dropped the ``Hook`` suffix across the
# set, later renames added more (``ContextWindow`` -> ``ContextWindowWarning``,
# ``ValidateREPLInput`` -> ``ValidateREPLCode``, ``Replay``/``ATIFReplay`` ->
# ``TrajectoryReplay``, ``ATIFTrace`` -> ``TrajectoryRecorder``), and each left an alias here so
# an out-of-tree import kept working with a warning. They are all gone: the set had grown to
# sixteen dead spellings across four renames, which made the public surface read as larger and
# more settled than it is — ``dir(jaz.hooks)`` advertised three names for one replay hook. An old
# name is now an ``AttributeError`` naming the module, the same failure a typo gives. Pre-1.0 is
# when that trade is cheapest; ``ATIFTrace``/``ATIFReplay`` were announced in 0.2.0a2 and so do
# have released users, which is what the changelog entry is for rather than a reason to keep them.
#
# Why the surviving ``_DEMOTED`` entries above are routed through ``__getattr__`` rather than
# bound as plain aliases (``X = Y``), which still matters for the ~30 that remain: a plain alias
# forces its target to be imported eagerly just to have something to alias, and an eagerly-bound
# attribute never reaches ``__getattr__`` — silently un-demoting the very name this module means
# to demote. That is why the block below stays ``TYPE_CHECKING``-only.

__getattr__, __dir__ = make_lazy_getattr(__name__, __all__, _DEMOTED)
