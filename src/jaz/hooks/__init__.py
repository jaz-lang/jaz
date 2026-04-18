"""Hook system for orchestrating hooks and composing execution contexts.

The hook system provides a type-safe, composable way to observe and influence
agent execution through events, effects, and contexts.

Key concepts:
- Events: Discrete points in execution lifecycle (e.g., ReplIterationEnter)
- Effects: Typed outputs from hooks expressing desired influence (e.g., HaltExecution)
- Contexts: Typed contracts between dispatcher and agent with composed effects
- Spans: Context manager wrappers that track completion and fire exit events
- Hooks: Extension points that process events and return effects (context-aware)
- HookDispatcher: Stateless orchestrator that reads hooks from context variables

Example usage:
    # Use hooks as context managers - they automatically propagate to nested invokes
    with BudgetControlHook(cost_tracker):
        with SafetyFilterHook():
            result = invoke(...)  # Both hooks are active
            # If invoke calls another invoke, hooks propagate automatically!

    # Disable specific hook types
    with PrintLogger(level=logging.INFO):
        invoke(...)  # Has PrintLogger

        with disable_hook(PrintLogger):
            invoke(...)  # PrintLogger disabled

    # Isolated context (no parent hooks)
    with isolated_hooks():
        invoke(...)  # No hooks at all
"""

from .base import Event, ExecutionContext, HaltableContext
from .builtin import (
    ConversationHistoryHook,
    FileLogger,
    IdentifyWorkflowHook,
    MemoryStoreHook,
    MultipleChoiceHook,
    ParentUpdatesHook,
    PrintLogger,
    WorkflowReplayHook,
    WorkflowStrategyHook,
)
from .context import disable_hook, isolated_hooks
from .dispatcher import Hook, HookDispatcher, get_dispatcher
from .effects import (
    AddInstructionPrompt,
    AddLibrary,
    AddReplInput,
    AddSystemPrompt,
    Effect,
    HaltExecution,
    ModifyIterationLimit,
)
from .events import (
    BudgetUpdate,
    BudgetUpdateContext,
    InvokeContext,
    InvokeEnter,
    InvokeExit,
    InvokeSpan,
    LLMQueryContext,
    LLMQueryEnter,
    LLMQueryExit,
    LLMQuerySpan,
    LLMRetry,
    LLMRetryContext,
    Message,
    MessageContext,
    ReplExecutionContext,
    ReplExecutionEnter,
    ReplExecutionExit,
    ReplExecutionSpan,
    ReplIterationContext,
    ReplIterationEnter,
    ReplIterationExit,
    ReplIterationSpan,
)

# Conditionally import tracing hooks
try:
    from .builtin import JaegerTracingHook, LangfuseTracingHook  # noqa: F401

    _TRACING_AVAILABLE = True
except ImportError:
    _TRACING_AVAILABLE = False

__all__ = [
    # Core
    "HookDispatcher",
    "Hook",
    "get_dispatcher",
    # Context management
    "disable_hook",
    "isolated_hooks",
    # Base
    "Event",
    "ExecutionContext",
    "HaltableContext",
    # Events - REPL iteration
    "ReplIterationEnter",
    "ReplIterationExit",
    "ReplIterationContext",
    "ReplIterationSpan",
    # Events - REPL execution
    "ReplExecutionEnter",
    "ReplExecutionExit",
    "ReplExecutionContext",
    "ReplExecutionSpan",
    # Events - LLM query
    "LLMQueryEnter",
    "LLMQueryExit",
    "LLMQueryContext",
    "LLMQuerySpan",
    # Events - Invoke
    "InvokeEnter",
    "InvokeExit",
    "InvokeContext",
    "InvokeSpan",
    # Events - Message
    "Message",
    "MessageContext",
    # Events - Budget
    "BudgetUpdate",
    "BudgetUpdateContext",
    # Events - LLM retry
    "LLMRetry",
    "LLMRetryContext",
    # Effects
    "Effect",
    "HaltExecution",
    "AddInstructionPrompt",
    "AddSystemPrompt",
    "AddReplInput",
    "ModifyIterationLimit",
    "AddLibrary",
    # Built-in hooks
    "ConversationHistoryHook",
    "MemoryStoreHook",
    "PrintLogger",
    "FileLogger",
    "WorkflowReplayHook",
    "MultipleChoiceHook",
    "IdentifyWorkflowHook",
    "WorkflowStrategyHook",
    "ParentUpdatesHook",
]

if _TRACING_AVAILABLE:
    __all__.append("JaegerTracingHook")
    __all__.append("LangfuseTracingHook")
