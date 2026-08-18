"""REPL execution events, contexts, and span (the conditional exec span).

The exec span fires **only when the LLM response parsed to runnable code**, so it is
conditional; a parse-failure turn opens no exec span. Loop/budget control belongs on the
always-present :class:`LLMQueryEnter` via :class:`Abort`, not here (see the effects module +
agent loop docstrings)."""

# These events are named ``REPLExec*`` (shortened from the former ``REPLExecution*``) per #481.

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from jaz.repl.types import ExecResult

from ..base import Event, ExecutionContext
from .base import Aborted, Completed, Failed

#: The REPL-execution span's outcome union — what :attr:`REPLExecExit.outcome` holds.
type REPLExecOutcome = Completed[ExecResult] | Aborted | Failed

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..effects import ModifyExecResult, SupplyExecResult


@dataclass(frozen=True)
class REPLExecEnter(Event):
    """Fired before executing parsed REPL code.

    **Not every turn.** It fires only when the LLM's response parsed into runnable code, so a
    turn is skipped entirely when the response fails to parse, when a hook emits a
    :class:`Abort` at :class:`LLMQueryEnter`, or when the LLM call itself errors — none
    of those reach an execution.

    Allowed effects:

    - :class:`Abort` — abort the invoke.
    - :class:`AddVariables` — bind names into the REPL namespace before the code runs.
    - :class:`DropVariables` — unbind names from the REPL namespace before the code runs.

    Suppliers do not compose here: :class:`SupplyExecResult` is a :class:`REPLExecSend`
    effect — a supplier decides on the basis of the committed input, which at this event
    composition may still rewrite.

    Attributes:
        iteration: The agent-loop iteration this execution belongs to.
        code: The REPL source about to be executed.
    """

    iteration: int
    code: str


@dataclass(frozen=True)
class REPLExecSend(Event):
    """Fired when an execution's input has committed — code + namespace deltas are final.

    Fires after :class:`REPLExecEnter`'s :class:`AddVariables` / :class:`DropVariables` have
    been applied to the REPL namespace, before the code runs — including when a supply then
    short-circuits the execution (the namespace deltas committed either way).

    Allowed effects:

    - :class:`SupplyExecResult` — supply a result in place of running the code; the code is
      not executed and the supplied result becomes the turn's raw result.
    - :class:`Abort` — abort the invoke, declining the committed input.

    Attributes:
        iteration: The agent-loop iteration this execution belongs to.
        code: The REPL source about to be executed (or short-circuited).
        added_variables: Namespace names bound before this turn's code, per the composed
            :class:`AddVariables` effects — a read-only mapping.
        dropped_variables: Namespace names unbound before this turn's code, per the
            composed :class:`DropVariables` effects — a frozenset.
    """

    # Send marks the *input commit*, not the dispatch to ``repl.exec`` — which is why it
    # still fires when a supply short-circuits execution (span_event_lifecycle.md, the
    # default-action framing on LLMQuerySend). Suppliers live HERE, not at Enter: a
    # supplier/validator decides on the basis of the input, and the input it must see is the
    # committed one (``ValidateREPLInput`` and the evals suppliers read ``event.code``).

    iteration: int
    code: str
    # ``Mapping``/``AbstractSet`` in the annotations, ``MappingProxyType``/``frozenset`` at
    # runtime (see __post_init__): the constructor accepts the dispatcher's plain dict/set,
    # the types forbid mutation through the fields.
    added_variables: Mapping[str, object] = field(default_factory=dict)
    dropped_variables: AbstractSet[str] = frozenset()

    def __post_init__(self) -> None:
        # Same read-only coercion as InvokeSend (see events/invoke.py): one shared event
        # instance is dispatched to every hook, so a hook must not be able to rebind or
        # alias-mutate what a later-dispatched hook observes. Before this the invoke span
        # was the only one whose Send payloads were structurally immutable — the same
        # concept carried a mutable dict/set here (type-level asymmetry, consistency
        # audit N1/C1).
        object.__setattr__(
            self, "added_variables", MappingProxyType(dict(self.added_variables))
        )
        object.__setattr__(self, "dropped_variables", frozenset(self.dropped_variables))


@dataclass(frozen=True)
class REPLExecComplete(Event):
    """Fired when an execution's raw result exists — the transform boundary.

    Fires with the result the producer made: the value ``repl.exec`` returned, or the result
    a :class:`SupplyExecResult` supplied at :class:`REPLExecSend` (a supplied result is a
    result; only aborts divert). It does not fire when the span unwinds on an
    in-flight exception (no raw result exists — transformers never run during unwind).

    Allowed effects:

    - :class:`ModifyExecResult` — replace the execution result with a different one.
    - :class:`Abort` — abort the invoke.

    Attributes:
        iteration: The agent-loop iteration this execution belongs to.
        exec_result: The raw result the producer made — a :class:`Continue`,
            :class:`Return`, or :class:`Raise` — before any transform composes.
    """

    # This is the transform boundary that used to live on REPLExecExit: hooks here see the
    # RAW result and their ModifyExecResult effects compose into the final one, which the
    # subsequent REPLExecExit then carries (span_event_lifecycle.md — no event can observe
    # its own composition's output; only the next event can).

    iteration: int
    exec_result: ExecResult


@dataclass(frozen=True)
class REPLExecExit(Event):
    """Fired when the execution span closes, with the turn's final outcome.

    Fires on the turns :class:`REPLExecEnter` did. ``outcome`` is the tagged union
    :data:`REPLExecOutcome`: :class:`Completed` carries the turn's **final,
    post-transform** result (``outcome.result`` — any :class:`ModifyExecResult` composed at
    :class:`REPLExecComplete` already applied); :class:`Aborted` means a hook's
    :class:`Abort` at one of this execution's control stages ended the invoke
    (``outcome.exception`` carries the abort's error, which propagates after the event
    fires); :class:`Failed` carries every other error that escaped before a result existed
    (``outcome.exception``; it still propagates after the event fires).

    Observation-only: no effect is valid here — the outcome is already final (an effect
    returned here raises :class:`~jaz.exceptions.InvalidEffectError`). To change a
    result, transform it at :class:`REPLExecComplete`.

    Timing: like every event, this exit carries only :attr:`~jaz.hooks.events.Event.timestamp`
    (its emission time); the span's duration is ``Exit.timestamp - Enter.timestamp``.

    Attributes:
        iteration: The agent-loop iteration this execution belongs to.
        outcome: How the span ended, with its payload — match on the variant
            (``Completed[ExecResult] | Aborted | Failed``).
    """

    # No time fields: the former start_time/end_time (interval-on-record, #1011) were
    # replaced by the base ``Event.timestamp`` — see its field comment in hooks/base.py
    # for the decision record.
    #
    # ``outcome`` has no default — see the rationale on ``LLMQueryExit``.

    iteration: int
    outcome: REPLExecOutcome


@dataclass
class REPLExecContext(ExecutionContext):
    """Context for REPL execution *enter* events.

    Hooks can:
    - **Abort** the invoke via :class:`Abort` — collected in ``abort_errors``;
      ``span_repl_exec`` raises the combined exception (the supply/veto slot is
      :class:`REPLExecSend`, not here).
    - **Bind namespace names** via :class:`AddVariables` — names to write into the REPL namespace
      before this turn's code runs, merged into ``added_variables`` (raw values, no ``__jaz_get__``;
      the prompt is untouched). The per-turn namespace counterpart of the invoke-level :class:`AddInputs`.
    - **Drop namespace names** via :class:`DropVariables` — the names to delete from the REPL
      namespace before this turn's code runs, unioned into ``dropped_variables``.
    - Record metrics

    Inputs (:class:`AddInputs`/:class:`DropInputs`) are NOT here: they are :class:`InvokeEnter`-only (#481) — fixed
    at invoke start, before the prompt. The per-turn namespace analogues are :class:`AddVariables` /
    :class:`DropVariables` (per-turn: a hook may want a name re-added/re-dropped each turn; see their
    docstrings). ``dropped_variables`` is applied before ``added_variables``, so a name in both
    ends *added* — which is what makes the documented drop-then-add re-bind recipe work (see
    the ``AddVariables`` docstring for the ordering rationale).
    """

    # Abort effects: the exceptions from Aborts (``abort_errors`` — ``span_repl_exec``
    # raises their combination; the span closes ``Aborted``). ``SupplyExecResult`` no longer
    # composes at enter (it moved to Send with the suppliers); the internal ``abort_errors``
    # name is kept across contexts.
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
class REPLExecSendContext(ExecutionContext):
    """Context for REPL execution *send* events — the supply boundary.

    Hooks can **supply** a result via :class:`SupplyExecResult` (execution is skipped and the
    supplied result becomes the raw result) or abort via :class:`Abort`; collected in
    ``supply_effects`` / ``abort_errors``. Supplies resolve via ``resolve_supply_results``;
    an abort supersedes them — ``span_repl_exec`` raises the combined exception out of
    ``span.send()`` and the span closes ``Aborted``.
    """

    supply_effects: list[SupplyExecResult] = field(default_factory=list)
    abort_errors: list[Exception] = field(default_factory=list)


@dataclass
class REPLExecCompleteContext(ExecutionContext):
    """Context for REPL execution *complete* events — the transform boundary.

    Hooks can **transform** the raw result via :class:`ModifyExecResult` or abort via
    :class:`Abort`; the effects are collected here and resolved against the actual
    ``exec_result`` (see ``resolve_modify_results``).
    """

    # Transform / abort effects: the ModifyExecResults (each carrying a full ExecResult) and
    # the exceptions from Aborts (``abort_errors`` — ``span_repl_exec`` raises their
    # combination, superseding the transform fold).
    modify_effects: list[ModifyExecResult] = field(default_factory=list)
    abort_errors: list[Exception] = field(default_factory=list)


class REPLExecSpan:
    """Span for REPL execution.

    Usage:
        with dispatcher.span_repl_exec(enter_event) as span:
            # ...apply span.ctx namespace deltas to the REPL...
            span.send()  # fires REPLExecSend; suppliers compose here (an Abort raises)
            if span.supplied is not None:
                exec_result = span.supplied  # a hook supplied the result at Send
            else:
                exec_result = repl.exec(...)
            span.complete(exec_result=exec_result)
        # REPLExecComplete-composed ModifyExecResult is applied here:
        exec_result = span.get_final_exec_result()

    ``supplied`` is resolved by :meth:`send` from the Send-composed supply effects — pure
    supply: an ``Abort`` at any of this span's control stages raises its carried exception
    from the ``span_repl_exec`` context manager instead (the span closes ``Aborted``).
    ``get_final_exec_result()`` returns the completed result after any Complete-time
    transform has been applied.
    """

    def __init__(self, ctx: REPLExecContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        self._exec_result: ExecResult | None = None
        self._final_exec_result: ExecResult | None = None
        # ExecResult short-circuiting the producer, resolved at Send: a Send-composed
        # SupplyExecResult supply — pure supply; the legacy semantics that also laundered
        # an Enter/Send-time Abort into a terminal Raise here are gone (aborts raise from
        # the CM). When set, the caller should skip executing the REPL code. None until
        # ``send()`` runs.
        self.supplied: ExecResult | None = None
        # Emitter for REPLExecSend, installed by ``span_repl_exec`` (the span is pure data
        # and must not import the dispatcher; the CM closes over the enter event + composed
        # namespace deltas, and stores the Send-composed supply on ``supplied``). None on a
        # hand-built span.
        self._emit_send: Callable[[], None] | None = None

    def send(self) -> None:
        """Fire :class:`REPLExecSend` for this execution's committed input.

        Called by the agent loop once the composed namespace deltas have been applied to
        the REPL — i.e. exactly when the input commits, before the code runs. Send-composed
        :class:`SupplyExecResult` effects are resolved and stored on :attr:`supplied`,
        which the caller reads to decide whether to run the code; a Send-composed
        :class:`Abort` raises its carried exception from this call (the span closes
        ``Aborted``).

        Raises:
            RuntimeError: If the span was not created by ``span_repl_exec``.
        """
        if self._emit_send is None:
            raise RuntimeError(
                "REPLExecSpan.send() requires a span created by span_repl_exec"
            )
        self._emit_send()

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
        """Record the result after Complete-time transform composition."""
        self._final_exec_result = exec_result

    def get_final_exec_result(self) -> ExecResult:
        """Get the result after any Complete-time transform has been applied.

        Returns:
            The final execution result (the composed override if one was produced,
            otherwise the result provided to complete()).

        Raises:
            RuntimeError: If the final result has not been set (span not
                completed, e.g. an exception propagated out of the span).
        """
        if self._final_exec_result is None:
            raise RuntimeError("Final exec result not set")
        return self._final_exec_result
