"""Hook system for orchestrating hooks and composing execution contexts.

The hook system provides a type-safe, composable way to observe and influence
agent execution through events, effects, and contexts.

Public surface (three parts):

- ``jaz.hooks`` — the authoring primitive :class:`Hook` and the ready-made
  "battery" hooks (:class:`PrintLogger`, :class:`BudgetPool`, :class:`RecursionLimit`,
  :class:`ReturnType`, ...).
- ``jaz.hooks.events`` — the typed events a hook handler *receives* (:class:`InvokeEnter`,
  :class:`REPLExecExit`, ...).
- ``jaz.hooks.effects`` — the typed effects a hook handler *returns* (:class:`ModifyResult`,
  :class:`AddMessages`, :class:`Abort`, ...).

Example usage::

    from jaz.hooks import Hook, PrintLogger
    from jaz.hooks.events import REPLExecExit
    from jaz.hooks.effects import ModifyResult

    # `with` activation propagates to nested invokes:
    with BudgetPool(cost_budget=1.0, calls_budget=50), PrintLogger():
        result = invoke(...)

    # Passing a hook positionally scopes it to that one invoke — nested invokes
    # do NOT see it. See ``Hook`` for the difference.
    result = invoke(PrintLogger(), task="...")

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
# `MetaData`/`WorkflowReplayHook`/`TemplatedMustExitWarning` are demoted (see `_DEMOTED`:
# reachable via `__getattr__`, with a NonPublicAPIWarning).
#
# `ConversationHistory` and `Replay` are demoted too, and are deliberately absent from this
# eager import: an eagerly-bound attribute is found by normal lookup and never reaches
# `__getattr__`, so importing them here would silently re-promote the two names.
from .builtin import (
    BudgetForcing,
    BudgetPool,
    Compaction,
    ContextWindow,
    FileLogger,
    IterationLimit,
    PrintLogger,
    RecursionLimit,
    ReturnType,
    ValidateREPLInput,
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
    from .builtin import ContextWindowHook as ContextWindowHook
    from .builtin import ConversationHistory as ConversationHistory
    from .builtin import ConversationHistoryHook as ConversationHistoryHook
    from .builtin import IterationLimitHook as IterationLimitHook
    from .builtin import JaegerTracingHook as JaegerTracingHook
    from .builtin import LangfuseTracingHook as LangfuseTracingHook
    from .builtin import MetaData as MetaData
    from .builtin import Replay as Replay
    from .builtin import ReplayHook as ReplayHook
    from .builtin import TemplatedMustExitWarning as TemplatedMustExitWarning
    from .builtin import WorkflowReplay as WorkflowReplay
    from .builtin import WorkflowReplayHook as WorkflowReplayHook
    from .dispatcher import HookDispatcher as HookDispatcher
    from .dispatcher import get_dispatcher as get_dispatcher
    from .effects import Abort as Abort
    from .effects import AddInputs as AddInputs
    from .effects import AddMessages as AddMessages
    from .effects import AddVariables as AddVariables
    from .effects import BlackboardWrite as BlackboardWrite
    from .effects import DisableRecursion as DisableRecursion
    from .effects import DropInputs as DropInputs
    from .effects import DropMessages as DropMessages
    from .effects import DropVariables as DropVariables
    from .effects import Effect as Effect
    from .effects import ModifyResult as ModifyResult
    from .effects import OverrideResponse as OverrideResponse
    from .effects import OverrideResult as OverrideResult
    from .events import Event as Event
    from .events import InvokeContext as InvokeContext
    from .events import InvokeEnter as InvokeEnter
    from .events import InvokeExit as InvokeExit
    from .events import InvokeSpan as InvokeSpan
    from .events import LLMQueryContext as LLMQueryContext
    from .events import LLMQueryEnter as LLMQueryEnter
    from .events import LLMQueryExit as LLMQueryExit
    from .events import LLMQueryRetry as LLMQueryRetry
    from .events import LLMQueryRetryContext as LLMQueryRetryContext
    from .events import LLMQuerySpan as LLMQuerySpan
    from .events import REPLExecContext as REPLExecContext
    from .events import REPLExecEnter as REPLExecEnter
    from .events import REPLExecExit as REPLExecExit
    from .events import REPLExecExitContext as REPLExecExitContext
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
    "BudgetForcing",
    "BudgetPool",
    "Compaction",
    "ContextWindow",
    "IterationLimit",
    "RecursionLimit",
    "PrintLogger",
    "FileLogger",
    "ReturnType",
    "ValidateReturn",
    "ValidateREPLInput",
]

if _TRACING_AVAILABLE:
    __all__.append("JaegerTracing")
    __all__.append("LangfuseTracing")

# Reachable-but-unsupported names (warn on access; blessed homes noted). The hook
# vocabulary is public under jaz.hooks.events / jaz.hooks.effects — the flat
# `jaz.hooks.<name>` spelling of it is demoted; the dispatch engine / internal machinery
# has no public home.
_DEMOTED = {
    "ConversationHistory": ("jaz.hooks.builtin", "ConversationHistory"),
    "Replay": ("jaz.hooks.builtin", "Replay"),
    "ExecutionContext": ("jaz.hooks.base", "ExecutionContext"),
    "Blackboard": ("jaz.hooks.blackboard", "Blackboard"),
    "MetaData": ("jaz.hooks.builtin", "MetaData"),
    "TemplatedMustExitWarning": ("jaz.hooks.builtin", "TemplatedMustExitWarning"),
    "WorkflowReplay": ("jaz.hooks.builtin", "WorkflowReplay"),
    "HookDispatcher": ("jaz.hooks.dispatcher", "HookDispatcher"),
    "get_dispatcher": ("jaz.hooks.dispatcher", "get_dispatcher"),
    # Effect vocabulary — blessed path: jaz.hooks.effects.<name>
    "Effect": ("jaz.hooks.effects", "Effect"),
    "Abort": ("jaz.hooks.effects", "Abort"),
    "OverrideResult": ("jaz.hooks.effects", "OverrideResult"),
    "ModifyResult": ("jaz.hooks.effects", "ModifyResult"),
    "DisableRecursion": ("jaz.hooks.effects", "DisableRecursion"),
    "AddInputs": ("jaz.hooks.effects", "AddInputs"),
    "DropInputs": ("jaz.hooks.effects", "DropInputs"),
    "AddVariables": ("jaz.hooks.effects", "AddVariables"),
    "DropVariables": ("jaz.hooks.effects", "DropVariables"),
    "DropMessages": ("jaz.hooks.effects", "DropMessages"),
    "AddMessages": ("jaz.hooks.effects", "AddMessages"),
    "OverrideResponse": ("jaz.hooks.effects", "OverrideResponse"),
    "BlackboardWrite": ("jaz.hooks.effects", "BlackboardWrite"),
    # Event vocabulary — blessed path: jaz.hooks.events.<name>
    "Event": ("jaz.hooks.events", "Event"),
    "REPLExecEnter": ("jaz.hooks.events", "REPLExecEnter"),
    "REPLExecExit": ("jaz.hooks.events", "REPLExecExit"),
    "LLMQueryEnter": ("jaz.hooks.events", "LLMQueryEnter"),
    "LLMQueryExit": ("jaz.hooks.events", "LLMQueryExit"),
    "InvokeEnter": ("jaz.hooks.events", "InvokeEnter"),
    "InvokeExit": ("jaz.hooks.events", "InvokeExit"),
    "LLMQueryRetry": ("jaz.hooks.events", "LLMQueryRetry"),
    # internal dispatcher contracts / span wrappers
    "InvokeContext": ("jaz.hooks.events", "InvokeContext"),
    "InvokeSpan": ("jaz.hooks.events", "InvokeSpan"),
    "LLMQueryContext": ("jaz.hooks.events", "LLMQueryContext"),
    "LLMQuerySpan": ("jaz.hooks.events", "LLMQuerySpan"),
    "LLMQueryRetryContext": ("jaz.hooks.events", "LLMQueryRetryContext"),
    "REPLExecContext": ("jaz.hooks.events", "REPLExecContext"),
    "REPLExecExitContext": ("jaz.hooks.events", "REPLExecExitContext"),
    "REPLExecSpan": ("jaz.hooks.events", "REPLExecSpan"),
}

# Pre-rename spellings (#806 question 1 dropped the ``Hook`` suffix across the whole set).
# Every one was in ``__all__`` before the rename — documented surface, not merely reachable —
# so dropping it outright would turn `from jaz.hooks import ConversationHistoryHook` in
# out-of-tree code into an ImportError with no hint of the replacement.
#
# Routed through ``_DEMOTED`` rather than bound as plain aliases (`XHook = X`), which is what
# the naming-audit PR used while this module still lacked a ``__getattr__`` shim — its comment
# said so explicitly and expected the unwind once ``make_lazy_getattr`` landed here. Plain
# aliases would also force ``WorkflowReplay`` to be imported eagerly just to have something to
# alias, and an eagerly-bound attribute never reaches ``__getattr__`` — silently un-demoting
# the very name this module means to demote.
#
# These are *import*-compatibility only. The serialized ``qualified_name`` in existing ATIF
# traces / conversation-history records is handled separately by ``_LEGACY_HOOK_QUALNAMES``
# in ``dispatcher.py`` — an entry here registers nothing.
_DEMOTED.update(
    {
        "BudgetForcingHook": ("jaz.hooks.builtin", "BudgetForcing"),
        "CompactionHook": ("jaz.hooks.builtin", "Compaction"),
        "ContextWindowHook": ("jaz.hooks.builtin", "ContextWindow"),
        "ConversationHistoryHook": ("jaz.hooks.builtin", "ConversationHistory"),
        "IterationLimitHook": ("jaz.hooks.builtin", "IterationLimit"),
        "ReplayHook": ("jaz.hooks.builtin", "Replay"),
        "WorkflowReplayHook": ("jaz.hooks.builtin", "WorkflowReplay"),
    }
)

if _TRACING_AVAILABLE:
    # Conditional: the tracing hooks only exist when their optional deps are installed, so
    # listing them unconditionally would advertise them in ``__dir__`` and then fail on access.
    _DEMOTED["JaegerTracingHook"] = ("jaz.hooks.builtin", "JaegerTracing")
    _DEMOTED["LangfuseTracingHook"] = ("jaz.hooks.builtin", "LangfuseTracing")

__getattr__, __dir__ = make_lazy_getattr(__name__, __all__, _DEMOTED)
