"""REPL execution events, contexts, and span (the conditional exec span).

The exec span opens **only on a turn that reaches execution**, so it is conditional: a query
that errored, or one a hook aborted, opens no exec span. What does not make it conditional is
the response's content — there is no parse-failure turn. Loop/budget control belongs on the
always-present :class:`LLMQueryEnter` via :class:`Abort`, not here (see the effects module +
agent loop docstrings)."""

# These events are named ``REPLExec*`` (shortened from the former ``REPLExecution*``) per #481.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from jaz.repl._types import ExecResult

from .._base import Event, ExecutionContext
from ._base import Aborted, Completed, Failed

#: The REPL-execution span's outcome union — what :attr:`REPLExecExit.outcome` holds.
type REPLExecOutcome = Completed[ExecResult] | Aborted | Failed

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaz._sentinel import UnsetType

    from ..effects import (
        DeleteCode,
        InsertCode,
        ModifyExecResult,
        SupplyExecResult,
    )


@dataclass(frozen=True)
class REPLExecEnter(Event):
    """Fired before executing parsed REPL code.

    **Not every turn.** It fires only on a turn that reaches execution, so a turn is skipped
    entirely when the LLM call itself errors, or when a hook emits an :class:`Abort` at any of
    the ``LLMQuery`` events — including :class:`LLMQueryComplete` and :class:`LLMQueryExit`,
    which abort after a response exists. The response's *content* never skips it: an empty or
    unparseable response is empty or nonsense *code*, which the REPL judges.

    Allowed effects:

    - :class:`Abort` — abort the invoke.
    - :class:`AddToNamespace` — bind names into the REPL namespace before the code runs.
    - :class:`DropFromNamespace` — unbind names from the REPL namespace before the code runs.
    - :class:`InsertCode` — insert source into the proposed code before it runs.
    - :class:`DeleteCode` — delete a character range from the proposed code before it runs.

    ``InsertCode`` / ``DeleteCode`` edit the *code* (the LLM-message twins are ``AddMessages`` /
    ``DropMessages``); ``AddToNamespace`` / ``DropFromNamespace`` edit the *namespace*. Both edit this
    event's proposal — ``code`` here is the LLM's original, and the composed edits produce the
    committed code that :class:`REPLExecSend` carries and the REPL runs.

    Suppliers do not compose here: :class:`SupplyExecResult` is a :class:`REPLExecSend`
    effect — a supplier decides on the basis of the committed code, which at this event
    composition may still rewrite.

    Attributes:
        iteration: The agent-loop iteration this execution belongs to.
        code: The REPL source proposed by the LLM, *before* any enter-time ``InsertCode`` /
            ``DeleteCode`` edits (which produce the committed code carried by
            :class:`REPLExecSend`).
        namespace: The REPL namespace as it stands *before* this turn's ``AddToNamespace`` /
            ``DropFromNamespace`` edits — the agent's ``globals()`` (inputs, hook bindings, the
            agent's own assignments from earlier turns, and ``__builtins__`` as a read-only copy,
            which holds the allowed builtins and the framework's ``invoke`` and ``__history__``),
            as a read-only snapshot whose variables' values are the live objects. What the REPL
            reports via :meth:`BaseREPL.get_namespace`, so a REPL whose namespace it cannot read
            may report less. The edited-then-committed namespace rides
            :attr:`REPLExecSend.namespace`.
    """

    # ``code`` and ``namespace`` are the two things this event's effects edit, and both are
    # here as the PROPOSAL — the same relation ``InvokeEnter.inputs`` and
    # ``LLMQueryEnter.messages`` have to their Sends (user's executive call, 2026-09-18: every
    # ``*Enter`` carries what its effects can edit; every ``*Send`` carries the same fields
    # committed). A hook deciding an ``AddToNamespace`` can read what is already bound here
    # instead of tracking it from ``InvokeEnter``.

    # Required, like ``InvokeEnter.inputs`` and ``LLMQueryEnter.messages``: the proposal is
    # not optional. Coerced in ``__post_init__`` for the same reasons as ``REPLExecSend``.
    iteration: int
    code: str
    namespace: Mapping[str, object]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "namespace", MappingProxyType(dict(self.namespace)))


@dataclass(frozen=True)
class CodeInsertRecord:
    """One :class:`InsertCode` edit folded into this execution's committed code.

    Attributes:
        text: The inserted source.
        position: The resolved offset into the enter-time code the text was inserted at
            (a negative ``at`` counts from the end; this is the absolute result).
    """

    text: str
    position: int


@dataclass(frozen=True)
class CodeDeleteRecord:
    """One run of source the :class:`DeleteCode` edits removed from this execution's
    enter-time code.

    A record is what was removed, not which effect asked: overlapping and adjacent
    ``DeleteCode`` ranges are unioned into disjoint runs, one record each, so the records
    never overlap and their ``text`` sums to exactly what the committed code lost; an empty
    range removes nothing and leaves no record.

    Attributes:
        start: The resolved start offset of the run into the enter-time code.
        end: The resolved end offset (exclusive).
        text: The deleted source — ``code[start:end]`` of the enter-time code.
    """

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class NamespaceAddRecord:
    """One name an :class:`AddToNamespace` bound into the namespace before this execution.

    Attributes:
        name: The bound name.
        value: The bound object, verbatim.
    """

    name: str
    value: object


@dataclass(frozen=True)
class NamespaceDropRecord:
    """One name a :class:`DropFromNamespace` unbound from the namespace before this execution.

    Attributes:
        name: The dropped name.
        value: The object that was unbound, or ``UNSET`` when nothing was — the name was
            absent (an ``allow_missing`` drop) or the REPL cannot report values (see
            :meth:`BaseREPL.get_namespace`).
    """

    name: str
    # No default, like ``NamespaceAddRecord.value``: ``UNSET`` is a positive statement that
    # nothing was bound, and a defaulted field would let a hand-built record make it by
    # omission — the one wrong reading ``UNSET`` exists to rule out. The union spelling stays
    # as documentation of the two cases, though ``object`` already covers both.
    value: object | UnsetType


@dataclass(frozen=True)
class ResolvedCodeEdits:
    """This execution's enter-time *code* edits as self-contained, pre-resolved records.

    ``inserts`` records :class:`InsertCode`, ``deletes`` records :class:`DeleteCode` — each
    sequence named by the verb of the effect it records, the rule every ``<thing>_edits``
    field follows. A record carries what the edit did, including what it removed: a delete
    record is one run of removed source after the ``DeleteCode`` ranges are unioned, so the
    delete records never overlap (see :class:`CodeDeleteRecord`).

    Ordering: ``inserts`` in application order (by resolved offset, hook order within one);
    ``deletes`` by resolved ``start`` (disjoint, so that is a total order).
    """

    # Why records rather than the raw effects (the shape ``LLMQuerySend.message_edits`` set):
    # an observer wants what happened, not what was asked for. A raw ``InsertCode`` still
    # carries its negative offset, so a reader would have to re-run the fold's offset
    # resolution to learn where its text landed; a raw ``DeleteCode`` says which range went,
    # not what was in it. There is no ``persistent``
    # flag here, unlike the message records: the code edits apply to this turn's proposal
    # alone. That flag is a property of the message buffer, not of edits in general.
    #
    # One ``Resolved<Thing>Edits`` per edited field — ``code`` here, ``namespace`` below —
    # rather than one bag for the span: the record type names the field it applies to, the
    # way ``ResolvedMessageEdits`` does, so a reader of ``REPLExecSend.code_edits`` never has
    # to learn that "exec edits" means two things.

    inserts: Sequence[CodeInsertRecord] = ()
    deletes: Sequence[CodeDeleteRecord] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "inserts", tuple(self.inserts))
        object.__setattr__(self, "deletes", tuple(self.deletes))


@dataclass(frozen=True)
class ResolvedNamespaceEdits:
    """This execution's enter-time *namespace* edits as self-contained records.

    ``adds`` records :class:`AddToNamespace`, ``drops`` records :class:`DropFromNamespace` — each
    sequence named by the verb of the effect it records. A drop record carries the value it
    unbound, read off :attr:`REPLExecEnter.namespace` (the namespace before the drops).

    Ordering: both sequences sorted by name — a record's coordinate, the rule every
    ``<thing>_edits`` sequence follows.
    """

    # No ``persistent`` flag: the namespace outlives the turn by construction (an add stays
    # bound until something drops it).

    adds: Sequence[NamespaceAddRecord] = ()
    drops: Sequence[NamespaceDropRecord] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "adds", tuple(self.adds))
        object.__setattr__(self, "drops", tuple(self.drops))


@dataclass(frozen=True)
class REPLExecSend(Event):
    """Fired when an execution's code has committed — code + namespace deltas are final.

    Fires after :class:`REPLExecEnter`'s :class:`AddToNamespace` / :class:`DropFromNamespace` have
    been applied to the REPL namespace, before the code runs — including when a supply then
    short-circuits the execution (the namespace deltas committed either way).

    Allowed effects:

    - :class:`SupplyExecResult` — supply a result in place of running the code; the code is
      not executed and the supplied result becomes the turn's raw result.
    - :class:`Abort` — abort the invoke, declining the committed code.

    Attributes:
        iteration: The agent-loop iteration this execution belongs to.
        code: The REPL source about to be executed (or short-circuited) — the enter-time
            proposal after any :class:`InsertCode` / :class:`DeleteCode` edits were folded in.
        code_edits: The enter-time edits that produced ``code``, as self-contained records
            (see :class:`ResolvedCodeEdits`).
        namespace_edits: The enter-time edits that produced ``namespace`` from
            :attr:`REPLExecEnter.namespace`, as self-contained records (see
            :class:`ResolvedNamespaceEdits`).
        namespace: The REPL namespace the committed code runs against — the agent's
            ``globals()`` (inputs, hook bindings, the agent's own assignments, and
            ``__builtins__`` as a read-only copy, which holds the allowed builtins and the
            framework's ``invoke`` and ``__history__``) after this turn's deltas. The mapping is
            read-only; the variables' values are the live objects. What the REPL reports via
            :meth:`BaseREPL.get_namespace`, so a REPL whose namespace it cannot read may report
            less.
    """

    # Send marks the *code commit*, not the dispatch to ``repl.exec`` — which is why it
    # still fires when a supply short-circuits execution (span_event_lifecycle.md, the
    # default-action framing on LLMQuerySend). Suppliers live HERE, not at Enter: a
    # supplier/validator decides on the basis of the code, and the code it must see is the
    # committed one (``ValidateREPLCode`` and a host's suppliers read ``event.code``).
    #
    # Every ``*Send`` event carries the enter-time input edits AND the result of committing
    # them (user's executive call, 2026-09-18): ``InvokeSend`` has the committed ``inputs``,
    # ``LLMQuerySend`` the committed ``messages`` plus ``message_edits``, and this one the
    # committed ``code`` plus ``code_edits`` and the committed ``namespace`` plus
    # ``namespace_edits`` — because for the REPL the committed input is not only the code but
    # the namespace it runs against, and each edited field gets its own ``<thing>_edits``. A
    # REPL's state is opaque to the framework (``BaseREPL``'s ``StateT``), so the namespace is
    # not derivable here: the loop
    # asks the REPL (``BaseREPL.get_namespace``) and passes the answer into ``span.send``, the way
    # it passes the committed ``inputs`` into ``InvokeSpan.send``. Snapshotted (copied +
    # frozen) because the ``BaseREPL.get_namespace`` contract PERMITS a live view — a custom REPL
    # may hand back its exec dict itself (``PythonREPL`` happens to build a fresh dict per
    # call) — and the agent keeps executing after Send.
    #
    # ``namespace_edits`` knowingly reverses the ``REPLExecSend`` half of #1474, which removed
    # the raw ``namespace_adds``/``namespace_drops`` from here under a different criterion — "a
    # field ships when a consumer needs it and cannot get it from another field" — because no
    # hook read them. The symmetry call replaces that criterion for the ``*Send`` events: their
    # payload is defined by the stage (edits + result), not by the current reader count, so an
    # observer written against one Send can be written against any of them. Without it an
    # observer could see WHAT ran but not which hook rewrote it or what it bound — the
    # exec-side instance of the ``InvokeEnter``-records-the-proposal problem that
    # ``InvokeSend`` was added to fix.

    # ``Mapping`` in the annotation, ``MappingProxyType`` at runtime (see __post_init__): the
    # constructor accepts the REPL's live view, the type forbids mutation through the field.
    iteration: int
    code: str
    # ``namespace`` is required, like ``InvokeSend.inputs``: the committed namespace is not
    # optional, and the ``span.send`` argument that feeds it is required for the same reason.
    # The two edit-record fields default to empty records, as ``LLMQuerySend.message_edits``
    # does — an event with no edits is the common case, and a required field must precede the
    # defaulted ones in a non-kw-only dataclass.
    namespace: Mapping[str, object]
    code_edits: ResolvedCodeEdits = field(default_factory=ResolvedCodeEdits)
    namespace_edits: ResolvedNamespaceEdits = field(
        default_factory=ResolvedNamespaceEdits
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Same read-only coercion as InvokeSend (see its __post_init__): one shared event
        # instance is dispatched to every hook, so the field must be a frozen view of a COPY —
        # a hook can neither rebind a sandbox name through it nor watch turn N's snapshot
        # change under it when turn N's code runs. The copy is shallow: the values ARE the
        # sandbox's objects, so mutating one in place (``event.namespace["xs"].append(...)``)
        # does reach the agent — deep-copying arbitrary values is neither safe nor cheap.
        object.__setattr__(self, "namespace", MappingProxyType(dict(self.namespace)))


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

    Allowed effects — narrow: a hook may return :class:`Abort` (and only ``Abort``), on **every**
    arm. On the :class:`Completed` arm this stops the run on the basis of the **final,
    post-transform** result (the thing :class:`REPLExecComplete` can't see, since a
    ``ModifyExecResult`` there composes after every Complete hook has run). On the
    :class:`Aborted` / :class:`Failed` arms a terminating exception is already unwinding, so an abort
    here cannot *replace* it: it is folded *into* it via an ``ExceptionGroup`` (the in-flight
    exception stays present, so ``is_fatal`` still governs how far the combined error travels). Any
    effect other than ``Abort`` raises :class:`~jaz.exceptions.InvalidEffectError` on the ``Completed``
    arm and is dropped with an error log on the abnormal arms (a loud raise there would replace the
    exception that must win). To *transform* a result rather than abort on it, use
    :class:`ModifyExecResult` at :class:`REPLExecComplete`.

    Timing: like every event, this exit carries only :attr:`~jaz.hooks.events.Event.timestamp`
    (its emission time); the span's duration is ``Exit.timestamp - Enter.timestamp``.

    Attributes:
        iteration: The agent-loop iteration this execution belongs to.
        outcome: How the span ended, with its payload — match on the variant
            (``Completed[ExecResult] | Aborted | Failed``).
    """

    # #892 settled the abnormal-arm rule: an abort raised while an exception is already
    # unwinding folds into that exception rather than replacing it.
    #
    # No time fields: the former start_time/end_time (interval-on-record, #1011) were
    # replaced by the base ``Event.timestamp`` — see its field comment in hooks/_base.py
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
    - **Bind namespace names** via :class:`AddToNamespace` — names to write into the REPL namespace
      before this turn's code runs, merged into ``namespace_adds`` (raw values;
      the prompt is untouched). The per-turn namespace counterpart of the invoke-level :class:`AddInputs`.
    - **Drop namespace names** via :class:`DropFromNamespace` — the names to delete from the REPL
      namespace before this turn's code runs, unioned into ``namespace_drops``.
    - **Edit the proposed code** via :class:`InsertCode` / :class:`DeleteCode` — collected into
      ``code_inserts`` / ``code_deletes`` and folded against the enter-time code by
      ``span_repl_exec`` into the committed code (the code-string counterpart of the namespace
      edits above; the LLM-message twins are ``AddMessages`` / ``DropMessages``).
    - Record metrics

    Inputs (:class:`AddInputs`/:class:`DropInputs`) are NOT here: they are :class:`InvokeEnter`-only — fixed
    at invoke start, before the prompt. The per-turn namespace analogues are :class:`AddToNamespace` /
    :class:`DropFromNamespace` (per-turn: a hook may want a name re-added/re-dropped each turn; see their
    docstrings). ``namespace_drops`` is applied before ``namespace_adds``, so a name in both
    ends *added* — which is what makes the documented drop-then-add re-bind recipe work (see
    the ``AddToNamespace`` docstring for the ordering rationale).
    """

    # The inputs/namespace split is #481: inputs are per-invoke setup applied once before the
    # prompt renders, so they belong at InvokeEnter rather than here.

    # Abort effects: the exceptions from Aborts (``abort_errors`` — ``span_repl_exec``
    # raises their combination; the span closes ``Aborted``). ``SupplyExecResult`` no longer
    # composes at enter (it moved to Send with the suppliers); the internal ``abort_errors``
    # name is kept across contexts.
    abort_errors: list[Exception] = field(default_factory=list)
    # Namespace names to bind before this turn's code runs (union of all AddToNamespace
    # ``entries`` via the shared conflict-error rule; ``__builtins__`` is refused).
    namespace_adds: dict[str, object] = field(default_factory=dict)
    # Namespace names to delete before this turn's code runs (union of all DropFromNamespace
    # ``names``; ``__builtins__`` is refused at composition time — see the dispatcher).
    namespace_drops: set[str] = field(default_factory=set)

    # Subset of ``namespace_drops`` whose ``DropFromNamespace`` set ``allow_missing=True`` — exempt
    # from the "dropping an unbound name raises ``MissingDropTargetError``" check (a defensive drop
    # that tolerates the name already being absent). Union across hooks: one opt-in exempts the name.
    namespace_drops_allow_missing: set[str] = field(default_factory=set)

    # InsertCode / DeleteCode effects editing the proposed code (accumulated, not conflict-checked
    # at collection — like AddMessages/DropMessages; the fold in ``_apply_code_edits`` unions deletes
    # and orders same-offset inserts). ``_compose_repl_exec`` is the only writer.
    code_inserts: list[InsertCode] = field(default_factory=list)
    code_deletes: list[DeleteCode] = field(default_factory=list)


@dataclass
class REPLExecSendContext(ExecutionContext):
    """Context for REPL execution *send* events — the supply boundary.

    Hooks can **supply** a result via :class:`SupplyExecResult` (execution is skipped and the
    supplied result becomes the raw result) or abort via :class:`Abort`; collected in
    ``supply_effects`` / ``abort_errors``. Supplies resolve via ``_resolve_supply_results``;
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
    ``exec_result`` (see ``_resolve_modify_results``).
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
            span.send(repl.get_namespace(state))  # fires REPLExecSend
            if span.supplied is not None:
                exec_result = span.supplied  # a hook supplied the result at Send
            else:
                exec_result = repl.exec(span.committed_code, ...)  # the post-edit code
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
        # The committed REPL source the agent should run — the enter code after enter-time
        # InsertCode / DeleteCode edits are folded in (``_apply_code_edits``). Set by
        # ``span_repl_exec`` before ``yield`` (== the enter code when no hook edits), and read by
        # the agent for ``repl.exec`` in place of the local code, so what runs is what
        # ``REPLExecSend`` observers (e.g. ``ValidateREPLCode``) saw. Mirrors ``supplied`` — a
        # resolved value the CM computes and the agent consumes. ``""`` on a hand-built span until
        # the CM sets it.
        self.committed_code: str = ""
        # Emitter for REPLExecSend, installed by ``span_repl_exec`` (the span is pure data
        # and must not import the dispatcher; the CM closes over the enter event + composed
        # namespace deltas, and stores the Send-composed supply on ``supplied``). None on a
        # hand-built span.
        self._emit_send: Callable[[Mapping[str, object]], None] | None = None

    def send(self, namespace: Mapping[str, object]) -> None:
        """Fire :class:`REPLExecSend` for this execution's committed code and namespace.

        Called by the agent loop once the composed namespace deltas have been applied to
        the REPL — i.e. exactly when the code commits, before it runs — with the namespace
        the REPL now reports (:meth:`BaseREPL.get_namespace`), which the event snapshots. The
        drop records read the values they unbound off the Enter event's snapshot, so nothing
        else is passed. Positional and required, like ``InvokeSpan.send(inputs)``. Send-composed
        :class:`SupplyExecResult` effects are resolved and stored on :attr:`supplied`,
        which the caller reads to decide whether to run the code; a Send-composed
        :class:`Abort` raises its carried exception from this call (the span closes
        ``Aborted``).

        Raises:
            RuntimeError: If the span was not created by ``span_repl_exec``.
        """
        # No default for ``namespace``: the loop is the only caller, and a default would let
        # it silently commit without the namespace — the event's own field default exists for
        # hand-built test instances, not for this path.
        # TODO(#1776): this and the three lifecycle checks below are framework invariant
        # breaches rooted at ``RuntimeError``, which ``is_fatal`` treats as recoverable.
        if self._emit_send is None:
            raise RuntimeError(
                "REPLExecSpan.send() requires a span created by span_repl_exec"
            )
        self._emit_send(namespace)

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

    @property
    def is_completed(self) -> bool:
        """Whether the span has been completed."""
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
