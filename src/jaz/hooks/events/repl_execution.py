"""REPL execution event, context, and span (the conditional exec span).

Named ``REPLExec*`` (shortened from the former ``REPLExecution*``) per #481 — the span fires
**only when the LLM response parsed to runnable code**, so it is conditional; a parse-failure
turn opens no exec span. Loop/budget control belongs on the always-present :class:`LLMQueryEnter`
via :class:`Abort`, not here (see the effects module + agent loop docstrings)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jaz.repl.types import ExecResult

from ..base import Event, ExecutionContext

if TYPE_CHECKING:
    from ..effects import ModifyResult, OverrideResult


@dataclass(frozen=True)
class REPLExecEnter(Event):
    """Fired before executing parsed REPL code.

    **Not every turn.** It fires only when the LLM's response parsed into runnable code, so a
    turn is skipped entirely when the response fails to parse, when a hook emits an
    :class:`Abort` at :class:`LLMQueryEnter`, or when the LLM call itself errors — none of
    those reach an execution.

    Allowed effects:

    - :class:`OverrideResult` — supply a result in place of running the code.
    - :class:`Abort` — terminate the invoke.
    - :class:`AddVariables` — bind names into the REPL namespace before the code runs.
    - :class:`DropVariables` — unbind names from the REPL namespace before the code runs.

    Attributes:
        iteration: The agent-loop iteration this execution belongs to.
        code: The REPL source about to be executed.
    """

    iteration: int
    code: str


@dataclass(frozen=True)
class REPLExecExit(Event):
    """Fired after executing parsed REPL code.

    Fires on the turns :class:`REPLExecEnter` did, once the turn produces a result — including
    when an enter-time :class:`OverrideResult` or :class:`Abort` skipped the execution, in which case
    ``exec_result`` is the supplied result rather than one the code produced. It does *not*
    fire if an error escapes before a result exists, which is logged rather than reported here.

    Allowed effects:

    - :class:`ModifyResult` — replace the execution result with a different one.
    - :class:`Abort` — terminate the invoke.

    Attributes:
        iteration: The agent-loop iteration this execution belongs to.
        exec_result: The result the code produced — a :class:`Continue`, :class:`Return`, or :class:`Raise`.
    """

    iteration: int
    exec_result: ExecResult


@dataclass
class REPLExecContext(ExecutionContext):
    """Context for REPL execution *enter* events.

    Hooks can:
    - **Supply** a result via :class:`OverrideResult` (execution is skipped and the supplied
      result is used) or terminate via :class:`Abort` — collected in ``override_effects`` /
      ``abort_errors`` and resolved by ``resolve_override_results``.
    - **Bind namespace names** via :class:`AddVariables` — names to write into the REPL namespace
      before this turn's code runs, merged into ``added_variables`` (raw values, no ``__jaz_get__``;
      the prompt is untouched). The per-turn namespace counterpart of the invoke-level :class:`AddInputs`.
    - **Drop namespace names** via :class:`DropVariables` — the names to delete from the REPL
      namespace before this turn's code runs, unioned into ``dropped_variables``.
    - Record metrics

    Inputs (:class:`AddInputs`/:class:`DropInputs`) are NOT here: they are :class:`InvokeEnter`-only (#481) — fixed
    at invoke start, before the prompt. The per-turn namespace analogues are :class:`AddVariables` /
    :class:`DropVariables` (per-turn: a hook may want a name re-added/re-dropped each turn; see their
    docstrings). ``added_variables`` is applied before ``dropped_variables``, so a name in both
    ends dropped.
    """

    # Supply / terminate effects: the OverrideResults (each carrying a full ExecResult) and
    # the exceptions from Aborts (``abort_errors`` — they resolve to a ``Raise``).
    override_effects: list[OverrideResult] = field(default_factory=list)
    abort_errors: list[Exception] = field(default_factory=list)
    # Namespace names to bind before this turn's code runs (union of all AddVariables
    # ``variables`` via the shared conflict-error rule; ``__builtins__`` is refused).
    added_variables: dict[str, object] = field(default_factory=dict)
    # Namespace names to delete before this turn's code runs (union of all DropVariables
    # ``names``; ``__builtins__`` is refused at composition time — see the dispatcher).
    dropped_variables: set[str] = field(default_factory=set)

    # Subset of ``dropped_variables`` whose ``DropVariables`` set ``allow_missing=True`` — exempt
    # from the "dropping an unbound name raises ``MissingDropTargetError``" check (a defensive drop
    # that tolerates the name already being absent). Union across hooks: one opt-in exempts the name.
    dropped_variables_allow_missing: set[str] = field(default_factory=set)


@dataclass
class REPLExecExitContext(ExecutionContext):
    """Context for REPL execution *exit* events.

    Hooks can **transform** the execution result via :class:`ModifyResult` or terminate via
    :class:`Abort`; the effects are collected here and resolved against the actual ``exec_result``
    (see ``resolve_modify_results``).
    """

    # Transform / terminate effects: the ModifyResults (each carrying a full ExecResult) and
    # the exceptions from Aborts (``abort_errors`` — they resolve to a ``Raise``).
    modify_effects: list[ModifyResult] = field(default_factory=list)
    abort_errors: list[Exception] = field(default_factory=list)


class REPLExecSpan:
    """Span for REPL execution.

    Usage:
        with dispatcher.span_repl_exec(...) as span:
            if span.enter_override is not None:
                exec_result = span.enter_override  # execution skipped by a hook
            else:
                exec_result = repl.exec(...)
            span.complete(exec_result=exec_result)
        # exit-time ModifyResult / Abort is applied here:
        exec_result = span.get_final_exec_result()

    ``enter_override`` is set by the dispatcher from enter-time effects (an :class:`OverrideResult`
    supply or an :class:`Abort`). ``get_final_exec_result()`` returns the completed result after any
    exit-time override has been applied.
    """

    def __init__(self, ctx: REPLExecContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        self._exec_result: ExecResult | None = None
        self._final_exec_result: ExecResult | None = None
        # ExecResult produced by an enter-time OverrideResult / Abort, if any.
        # When set, the caller should skip executing the REPL code.
        self.enter_override: ExecResult | None = None

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

    def set_final_exec_result(self, exec_result: ExecResult) -> None:
        """Record the result after exit-time override composition."""
        self._final_exec_result = exec_result

    def get_final_exec_result(self) -> ExecResult:
        """Get the result after any exit-time override has been applied.

        Returns:
            The final execution result (the exit override if one was produced,
            otherwise the result provided to complete()).

        Raises:
            RuntimeError: If the final result has not been set (span not
                completed, e.g. an exception propagated out of the span).
        """
        if self._final_exec_result is None:
            raise RuntimeError("Final exec result not set")
        return self._final_exec_result
