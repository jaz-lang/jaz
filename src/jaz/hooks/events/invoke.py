"""Invoke event, contexts, and span."""

from dataclasses import dataclass, field

from typing_extensions import TypeForm

from jaz.library import Library
from jaz.repl.types import ExecResult

from ..base import Event, HaltableContext


@dataclass
class InvokeEnter[ReturnT](Event):
    """Fired when Agent.invoke is called, before any execution begins.

    Hooks can halt execution, add to prompts, or modify the environment.
    """

    invoke_id: str
    parent_invoke_id: str | None
    parent_repl_iteration: int | None
    prompt: str
    return_type: TypeForm[ReturnT] | None
    inputs: dict[str, object]
    max_iterations: int
    cur_recursion_depth: int
    max_recursion_depth: int
    libraries: list[Library]
    task_name: str


@dataclass
class InvokeExit(Event):
    """Fired when Agent.invoke completes or raises.

    This is observational only - execution is complete.
    Hooks can record metrics but cannot influence execution.
    """

    invoke_id: str
    result: ExecResult
    cur_recursion_depth: int
    max_recursion_depth: int


@dataclass
class InvokeContext(HaltableContext):
    """Context for invoke enter events.

    Hooks can:
    - Halt execution (prevent the invocation)
    - Add to the user prompt (instruction_prompt_additions)
    - Add to the system prompt (system_prompt_additions)
    - Add libraries
    - Add REPL inputs
    - Record metrics
    """

    # Prompt modifications
    instruction_prompt_additions: list[str] = field(default_factory=list)
    system_prompt_additions: list[str] = field(default_factory=list)

    # Environment modifications
    additional_libraries: list[Library] = field(default_factory=list)

    # REPL inputs
    repl_inputs: dict[str, object] = field(default_factory=dict)


class InvokeSpan:
    """Span for invoke.

    Usage:
        with dispatcher.span_invoke(...) as span:
            if span.ctx.should_halt:
                _raise_halt_errors(span.ctx.halt_errors)  # ExceptionGroup if multiple

            # Add libraries, modify prompt, etc.
            result = agent._invoke_internal(...)
            span.complete(result=result)
    """

    def __init__(self, ctx: InvokeContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        self._result: ExecResult | None = None

    def complete(self, *, result: ExecResult) -> None:
        """Complete span with invocation result.

        Args:
            result: The result of the invocation (RETURN or RAISE)

        Raises:
            RuntimeError: If span was already completed
        """
        if self._completed:
            raise RuntimeError("Span already completed")
        self._result = result
        self._completed = True

    def is_completed(self) -> bool:
        """Check if span was completed."""
        return self._completed

    def get_result(self) -> ExecResult:
        """Get the invocation result.

        Returns:
            The result provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        assert self._result is not None
        return self._result
