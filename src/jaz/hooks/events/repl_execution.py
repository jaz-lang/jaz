"""REPL execution event, context, and span."""

from dataclasses import dataclass, field

from jaz.repl.types import ExecResult

from ..base import Event, HaltableContext


@dataclass
class ReplExecutionEnter(Event):
    """Fired before executing a REPL execution.

    This is after the LLM has generated code but before it's executed.
    Hooks can halt execution, modify prompts, or adjust execution limits.
    """

    invoke_id: str
    iteration: int
    max_iterations: int
    code: str
    cur_recursion_depth: int


@dataclass
class ReplExecutionExit(Event):
    """Fired after executing a REPL execution.

    This is observational only - execution is complete.
    Hooks can record metrics but cannot influence execution.
    """

    invoke_id: str
    iteration: int
    max_iterations: int
    exec_result: ExecResult
    cur_recursion_depth: int


@dataclass
class ReplExecutionContext(HaltableContext):
    """Context for REPL execution events.

    Hooks can:
    - Halt execution
    - Override the max iteration limit
    - Add REPL inputs
    - Record metrics
    """

    # Iteration control
    max_iterations_override: int | None = None

    # REPL inputs
    repl_inputs: dict[str, object] = field(default_factory=dict)


class ReplExecutionSpan:
    """Span for REPL execution.

    Usage:
        with dispatcher.span_repl_execution(...) as span:
            if span.ctx.should_halt:
                _raise_halt_errors(span.ctx.halt_errors)  # ExceptionGroup if multiple

            exec_result = repl.exec(...)
            span.complete(exec_result=exec_result)
    """

    def __init__(self, ctx: ReplExecutionContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        self._exec_result: ExecResult | None = None

    def complete(self, *, exec_result: ExecResult) -> None:
        """Complete span with execution result.

        Args:
            exec_result: The result of executing the REPL code

        Raises:
            RuntimeError: If span was already completed
        """
        if self._completed:
            raise RuntimeError("Span already completed")
        self._exec_result = exec_result
        self._completed = True

    def is_completed(self) -> bool:
        """Check if span was completed."""
        return self._completed

    def get_exec_result(self) -> ExecResult:
        """Get the execution result.

        Returns:
            The execution result provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        assert self._exec_result is not None
        return self._exec_result
