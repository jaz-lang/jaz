"""Base classes for the hook system.

This module defines the foundational types used throughout the hook system:
- Event: Base class for all events
- ExecutionContext: Base context for all events (read-only, no capabilities)
- HaltableContext: Context that supports halting execution
"""

from dataclasses import dataclass, field


@dataclass
class Event:
    """Base class for all events.

    Events represent discrete points in the agent's execution lifecycle where
    hooks can observe and influence behavior.
    """

    pass


@dataclass
class ExecutionContext:
    """Base context for all events.

    This is a read-only context with no capabilities. Use HaltableContext
    for events that support halting execution.
    """

    pass


@dataclass
class HaltableContext(ExecutionContext):
    """Context that supports halting execution.

    Use this as the base class for contexts where hooks should be able
    to halt the agent's execution (e.g., REPL iteration, LLM query, invoke).
    """

    halt_errors: list[Exception] = field(default_factory=list)

    @property
    def should_halt(self) -> bool:
        return bool(self.halt_errors)
