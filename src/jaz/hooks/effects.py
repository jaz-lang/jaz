"""Effect definitions for the hook system.

Effects are the typed outputs from hooks that express how they want to influence
execution. The dispatcher composes these effects into an ExecutionContext.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaz.library import Library


@dataclass
class Effect:
    """Base class for all effects."""

    pass


# Universal effects (apply to all ExecutionContext types)


@dataclass
class HaltExecution(Effect):
    """Request to halt execution immediately.

    Any hook can halt execution by returning this effect.
    If multiple hooks return HaltExecution, all errors are collected and raised.
    A single error is raised directly; multiple errors are raised as an
    ExceptionGroup.
    """

    error: Exception


# Prompt modification effects


@dataclass
class AddInstructionPrompt(Effect):
    """Add instruction text to the next user prompt sent to the agent.

    Only valid for contexts that support prompt modification (ReplIterationContext,
    InvokeContext). Multiple AddInstructionPrompt effects are sorted lexicographically
    for deterministic ordering regardless of hook order (commutative).
    """

    text: str


@dataclass
class AddSystemPrompt(Effect):
    """Add text to the system prompt for the agent.

    Only valid for InvokeContext. Multiple AddSystemPrompt effects are sorted
    lexicographically for deterministic ordering regardless of hook order
    (commutative).
    """

    text: str


# REPL iteration effects


@dataclass
class ModifyIterationLimit(Effect):
    """Change the maximum number of REPL iterations.

    Only valid for ReplIterationContext. If multiple hooks return this effect,
    the dispatcher uses the minimum value.
    """

    new_max: int
    reason: str


# Invoke effects


@dataclass
class AddLibrary(Effect):
    """Add a library to the agent's environment.

    Only valid for InvokeContext.
    """

    library: Library


# REPL input effects


@dataclass
class AddReplInput(Effect):
    """Add an arbitrary Python object to the REPL environment.

    This effect allows hooks to inject variables, functions, classes, or any
    other Python object into the REPL state. The injected objects become
    available as variables in the agent's code execution environment.

    Valid contexts: InvokeContext, ReplIterationContext, ReplExecutionContext

    Composition rule: All inputs are added; last write wins for duplicate keys.
    Inputs accumulate across context levels (invoke → iteration → execution).

    Example:
        # Add a configuration object
        AddReplInput(key="config", value={"api_key": "abc"}, reason="API config")

        # Add a helper function
        def my_helper(x): return x * 2
        AddReplInput(key="helper", value=my_helper, reason="Utility function")
    """

    key: str  # Variable name in REPL
    value: object  # Python object to add
    reason: str = ""  # Optional explanation for debugging/logging
