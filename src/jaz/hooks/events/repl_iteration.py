"""REPL iteration event, context, and span."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaz.agent import ReplIterationResult

from ..base import Event, HaltableContext


@dataclass
class ReplIterationEnter(Event):
    """Fired before a complete REPL iteration (LLM query + REPL execution).

    This event is fired before the agent queries the LLM and executes code
    in a single REPL iteration. Hooks can halt the entire invoke by raising
    an exception, or record metrics.
    """

    invoke_id: str
    iteration: int
    max_iterations: int
    cur_recursion_depth: int
    max_recursion_depth: int


@dataclass
class ReplIterationExit(Event):
    """Fired after a complete REPL iteration (LLM query + REPL execution).

    This is observational only - the iteration is complete.
    Hooks can record metrics but cannot influence execution.
    """

    invoke_id: str
    iteration: int
    max_iterations: int
    cur_recursion_depth: int
    max_recursion_depth: int
    result: "ReplIterationResult"


@dataclass
class ReplIterationContext(HaltableContext):
    """Context for REPL iteration events.

    Hooks can:
    - Halt the entire invoke by raising an exception (via HaltExecution effect)
    - Add text to the prompt (via AddInstructionPrompt effect)
    - Add REPL inputs
    - Record metrics

    The halt functionality is inherited from HaltableContext:
    - should_halt: bool (property, True when halt_errors is non-empty)
    - halt_errors: list[Exception]
    """

    # Prompt modifications
    instruction_prompt_additions: list[str] = field(default_factory=list)

    # REPL inputs
    repl_inputs: dict[str, object] = field(default_factory=dict)


class ReplIterationSpan:
    """Span for REPL iteration.

    Usage:
        with dispatcher.span_repl_iteration(...) as span:
            if span.ctx.should_halt:
                _raise_halt_errors(span.ctx.halt_errors)  # ExceptionGroup if multiple

            result = do_one_repl_iteration(...)
            span.complete(result=result)
    """

    def __init__(self, ctx: ReplIterationContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        self._result: ReplIterationResult | None = None

    def complete(self, *, result: "ReplIterationResult") -> None:
        """Complete span with iteration result.

        Args:
            result: The ReplIterationResult from do_one_repl_iteration

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

    def get_result(self) -> "ReplIterationResult":
        """Get the iteration result.

        Returns:
            The ReplIterationResult provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        assert self._result is not None
        return self._result
