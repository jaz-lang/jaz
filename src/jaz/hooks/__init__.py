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

Before writing your own, check the battery hooks — limits, budgets, logging, tracing,
compaction, and replay are already covered.

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
# `MetaData`/`WorkflowReplayHook` are demoted (see `_DEMOTED`:
# reachable via `__getattr__`, with a NonPublicAPIWarning).
#
# `ATIFTrace`/`ATIFReplay` are public here as of 2026-08-15 (user call), superseding the
# review-of-#962 demotion of the replay hook and `ATIFTrace`'s deep-path-only status:
# ATIF is the canonical trace/replay/cost format (#813, #1159), so its write/read pair
# belongs on the offered surface rather than behind deep imports every consumer had to
# know. The replay hook was renamed `Replay` → `ATIFReplay` at promotion (user call,
# same review): `RolloutRecorder` already makes token-native rollouts a second recorded
# format, so the bare name would collide with a future token-native replay, and the
# shared prefix keeps the write/read pair adjacent in `__all__` and on the docs page.
from .builtin import (
    ATIFReplay,
    ATIFTrace,
    BudgetForcing,
    BudgetPool,
    Compaction,
    ContextWindowWarning,
    FileLogger,
    IterationLimit,
    PrintLogger,
    RecursionLimit,
    ReturnType,
    RolloutRecorder,
    ValidateREPLCode,
    ValidateReturn,
)
from .dispatcher import Hook

if TYPE_CHECKING:
    # Static types for the reachable-but-demoted names (resolved lazily at runtime via
    # __getattr__, which warns). Not bound at runtime; type checkers read this block.
    from .base import ExecutionContext as ExecutionContext
    from .blackboard import Blackboard as Blackboard

    # Pre-rename spellings, kept importable via _DEMOTED (see below). Taken from
    # ``.builtin``, which binds them as plain aliases of the renamed classes.
    from .builtin import BudgetForcingHook as BudgetForcingHook
    from .builtin import CompactionHook as CompactionHook
    from .builtin import ContextWindow as ContextWindow
    from .builtin import ContextWindowHook as ContextWindowHook
    from .builtin import IterationLimitHook as IterationLimitHook
    from .builtin import JaegerTracingHook as JaegerTracingHook
    from .builtin import LangfuseTracingHook as LangfuseTracingHook
    from .builtin import MetaData as MetaData
    from .builtin import Replay as Replay
    from .builtin import ReplayHook as ReplayHook
    from .builtin import ValidateREPLInput as ValidateREPLInput
    from .builtin import WorkflowReplay as WorkflowReplay
    from .builtin import WorkflowReplayHook as WorkflowReplayHook
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
    from .builtin import JaegerTracing, LangfuseTracing  # noqa: F401

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
    "ATIFReplay",
    "ATIFTrace",
    "BudgetForcing",
    "BudgetPool",
    "Compaction",
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
    __all__.append("JaegerTracing")
    __all__.append("LangfuseTracing")

# Reachable-but-unsupported names (warn on access; blessed homes noted). The hook
# vocabulary is public under jaz.hooks.events / jaz.hooks.effects — the flat
# `jaz.hooks.<name>` spelling of it is demoted; the dispatch engine / internal machinery
# has no public home.
_DEMOTED = {
    # Pre-promotion spelling of ATIFReplay: this name was reachable-with-warning here
    # before the rename, so it keeps exactly that status rather than becoming an
    # AttributeError.
    "Replay": ("jaz.hooks.builtin", "ATIFReplay"),
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

# Pre-rename spellings (#806 question 1 dropped the ``Hook`` suffix across the whole set;
# the REPL-input → REPL-code rename added one more that is not a suffix drop).
# Every one was in ``__all__`` before the rename — documented surface, not merely reachable —
# so dropping it outright would turn `from jaz.hooks import WorkflowReplayHook` in
# out-of-tree code into an ImportError with no hint of the replacement.
#
# Routed through ``_DEMOTED`` rather than bound as plain aliases (`XHook = X`), which is what
# the naming-audit PR used while this module still lacked a ``__getattr__`` shim — its comment
# said so explicitly and expected the unwind once ``make_lazy_getattr`` landed here. Plain
# aliases would also force ``WorkflowReplay`` to be imported eagerly just to have something to
# alias, and an eagerly-bound attribute never reaches ``__getattr__`` — silently un-demoting
# the very name this module means to demote.
#
# These are *import*-compatibility only: they keep the old names importable. An
# entry here registers nothing and has no bearing on serialized ``qualified_name`` reconstruction.
_DEMOTED.update(
    {
        "BudgetForcingHook": ("jaz.hooks.builtin", "BudgetForcing"),
        "CompactionHook": ("jaz.hooks.builtin", "Compaction"),
        "ContextWindow": ("jaz.hooks.builtin", "ContextWindowWarning"),
        "ContextWindowHook": ("jaz.hooks.builtin", "ContextWindowWarning"),
        "IterationLimitHook": ("jaz.hooks.builtin", "IterationLimit"),
        "ReplayHook": ("jaz.hooks.builtin", "ATIFReplay"),
        "WorkflowReplayHook": ("jaz.hooks.builtin", "WorkflowReplay"),
        # Not a ``Hook``-suffix drop: "REPL input" was renamed to "REPL code" because
        # "input" already means an *invoke input* everywhere else in the vocabulary
        # (``AddInputs``, ``DropInputs``, ``__inputs__``), so the old name read as
        # "validate the invoke's inputs" — the opposite of what it does.
        "ValidateREPLInput": ("jaz.hooks.builtin", "ValidateREPLCode"),
    }
)

if _TRACING_AVAILABLE:
    # Conditional: the tracing hooks only exist when their optional deps are installed, so
    # listing them unconditionally would advertise them in ``__dir__`` and then fail on access.
    _DEMOTED["JaegerTracingHook"] = ("jaz.hooks.builtin", "JaegerTracing")
    _DEMOTED["LangfuseTracingHook"] = ("jaz.hooks.builtin", "LangfuseTracing")

__getattr__, __dir__ = make_lazy_getattr(__name__, __all__, _DEMOTED)
