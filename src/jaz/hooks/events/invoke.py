"""Invoke events, contexts, and span."""

from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import MappingProxyType

from jaz._invoke_tool import InvokeTool
from jaz.repl.types import Raise, Return, TerminalExecResult

from ..base import Event, ExecutionContext
from .base import Aborted, Completed, Failed

#: The invoke span's outcome union — what :attr:`InvokeExit.outcome` holds. The payload is
#: ``Return | Raise`` (never ``Continue``): an invoke only ever exits on a terminal result.
type InvokeOutcome = Completed[Return | Raise] | Aborted | Failed


@dataclass(frozen=True)
class InvokeEnter(Event):
    """Fired when Agent.invoke is called, before any execution begins.

    **Fires once per invoke**, not per turn — one of these opens every invoke, including each
    nested sub-invoke, before the first LLM query.

    Allowed effects:

    - :class:`Abort` — abort the invoke before its first turn.
    - :class:`AddInputs` — add an input, visible in the prompt and bound in the REPL.
    - :class:`DropInputs` — remove an input from both the prompt and the REPL.
    - :class:`DisableRecursion` — withhold the recursive ``invoke`` tool from this invoke.

    Attributes:
        parent_invoke_id: The enclosing invoke's id, or ``None`` at the top level.
        parent_repl_iteration: The enclosing invoke's iteration that spawned this one, or
            ``None`` at the top level.
        inputs: The explicit ``**inputs`` passed to ``jaz.invoke``, as a read-only mapping.
        invoke_tool: The recursive sub-invoke primitive bound into this invoke (as the bare
            REPL name ``invoke``), or ``None`` when recursion is disabled.
        scope: The resolved ambient ``jaz.scope`` values, as a read-only mapping. Disjoint
            from ``inputs``.
    """

    # Identity vs. label: ``invoke_id`` (a per-invoke uuid4, carried on every event) is the
    # stable identity used to correlate an invocation across logs / traces / replay.
    # ``task_name`` is deliberately not a field here — it is a purely *descriptive* human
    # label, optional and read off the blackboard via the ``MetaData`` hook (see
    # ``hooks/builtin/metadata.py``). It was kept off-core because core never reads it; an
    # unlabeled invoke defaulting to ``"main"`` is fine precisely because uniqueness and
    # addressability come from ``invoke_id``, not the name — so a generated/mandatory name
    # would duplicate identity that already exists. (Design decision, reviewed on #546.)

    parent_invoke_id: str | None
    parent_repl_iteration: int | None
    # The invoke's EXPLICIT input variables — the ``**inputs`` kwargs passed to ``jaz.invoke``
    # (including user-supplied tool namespaces, which are ordinary inputs — see
    # ``library_as_input.md``). Values are already resolved via the ``__jaz_get__`` protocol.
    # Ambient ``jaz.scope`` values are NOT here — they are a separate provenance channel, on
    # ``scope`` below. The two are DISJOINT: a name defined both ways raises at invoke time
    # (see the conflict check in ``invoke._build_invoke_setup``), so ``{**scope, **inputs}``
    # (which is what the agent's REPL binds as globals) is a plain union with no shadowing.
    # Exposed as a read-only mapping (see ``__post_init__``) — like the whole event, treat it as
    # immutable; a hook must not rebind keys here (it would perturb later-dispatched hooks).
    inputs: Mapping[str, object]
    # The framework's recursive sub-invoke primitive bound into this invoke — a single
    # callable the agent reaches as the bare REPL name ``invoke`` — or None. (It was a whole
    # ``Library`` named ``jaz`` until the namespace's last other member left in #635.) A
    # ``DisableRecursion`` effect emitted in response to THIS event (by ``RecursionLimit`` at
    # the cap leaf) causes the primitive to bind the REPL with this nullified — the
    # recursion-cap affordance-removal lives in the effect layer, not a framework
    # ``max_depth`` field.
    invoke_tool: InvokeTool | None
    # The RESOLVED ambient scope for this invoke — the ``jaz.scope`` values in effect
    # (``{**parent_scope, **local_scope}``, mirroring how ``config`` is the resolved effective
    # config). A separate observability channel from ``inputs`` because ``jaz.scope`` and the
    # ``**inputs`` kwargs are distinct user-side APIs and are rendered in distinct sections of
    # the agent's initial prompt. Defaulted so tests can construct the event without it; the
    # real invoke path always supplies it. Disjoint from ``inputs`` (see ``inputs`` above).
    # Read-only mapping, same as ``inputs`` (see ``__post_init__``).
    scope: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Event is frozen and must be treated as fully immutable (see the ``Event`` docstring).
        # Expose inputs/scope as read-only mappings over a private shallow copy, so a hook can
        # neither rebind a key nor alias-mutate the dict the agent still holds — either would
        # silently change what a later-dispatched hook at the same event observes. ``dict()``
        # is a shallow copy (cheap; it does not copy the values), and ``object.__setattr__`` is
        # the frozen-dataclass escape hatch for this framework-internal coercion.
        #
        # The inner ``dict()`` is LOAD-BEARING, not redundant with the proxy:
        # ``MappingProxyType`` is a read-only *view*, not a copy — wrapping the caller's dict
        # directly would block writes through the field while still showing every later
        # mutation of the caller's dict. The copy severs; the proxy freezes. Construction
        # sites therefore pass their containers uncopied (the event owns the severing) — do
        # not "simplify" the inner copy away, and do not re-add call-site copies.
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "scope", MappingProxyType(dict(self.scope)))


@dataclass(frozen=True)
class InvokeSend(Event):
    """Fired when an invoke's input set has committed — post :class:`AddInputs` / :class:`DropInputs`.

    Fires once per invoke, after :class:`InvokeEnter`'s input effects have been folded into
    the inputs the prompt renders and the REPL binds, before the first turn. It does not
    fire when a hook aborts the invoke at :class:`InvokeEnter` (the input never commits).

    Allowed effects:

    - :class:`SupplyInvokeResult` — supply the invoke's terminal result and skip the whole agent
      loop (the invoke-boundary twin of :class:`SupplyExecResult` at :class:`REPLExecSend`).
    - :class:`Abort` — abort the invoke, declining the committed input set.

    Attributes:
        inputs: The invoke's committed explicit inputs (resolved values, post add/drop), as
            a read-only mapping — the counterpart of :class:`InvokeEnter`'s pre-commit
            ``inputs``.
        added_inputs: The inputs hooks added via :class:`AddInputs` (raw effect values).
        dropped_inputs: The input names hooks dropped via :class:`DropInputs`.
    """

    # Observers of InvokeEnter record the *proposal*: an input a hook injects via AddInputs
    # (e.g. ultrahorizon's per-child ``env`` library) is absent from InvokeEnter.inputs, so
    # traces built from Enter mis-record what the child actually received — the input-side
    # instance of the #906 pattern. This event is the commit those observers should read.
    # ``scope`` is not repeated here: it is not editable at InvokeEnter, so InvokeEnter's
    # copy is already the committed one.

    # ``Mapping``/``AbstractSet`` in the annotations, ``MappingProxyType``/``frozenset`` at
    # runtime (see __post_init__) — the constructor accepts plain containers, the types
    # forbid mutation through the fields.
    inputs: Mapping[str, object]
    added_inputs: Mapping[str, object] = field(default_factory=dict)
    dropped_inputs: AbstractSet[str] = frozenset()

    def __post_init__(self) -> None:
        # Same read-only coercion as InvokeEnter (see its __post_init__): one shared event
        # instance is dispatched to every hook, so a hook must not be able to rebind or
        # alias-mutate what a later-dispatched hook observes.
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(
            self, "added_inputs", MappingProxyType(dict(self.added_inputs))
        )
        object.__setattr__(self, "dropped_inputs", frozenset(self.dropped_inputs))


@dataclass(frozen=True)
class InvokeComplete(Event):
    """Fired when the invoke's raw terminal result exists — the transform boundary.

    **Fires once per invoke**, with the terminal result the loop produced. It does not
    fire when the invoke unwinds on an in-flight exception before a terminal result
    exists (transformers never run during unwind), nor when a hook's :class:`Abort`
    ended the invoke at an earlier stage (an abort skips the pipeline's remaining
    stages — its stickiness is structural).

    Allowed effects:

    - :class:`ModifyInvokeResult` — replace the invoke's terminal result.
    - :class:`Abort` — abort the invoke with an error.

    Attributes:
        result: The invoke's raw terminal result — a :class:`Return` or a :class:`Raise` —
            before any transform composes.
    """

    # This is the transform boundary that used to live on InvokeExit (#568): the last line
    # of defense for a return contract. A ``ReturnType`` / ``ValidateReturn`` hook re-checks
    # the *final* ``Return`` here and downgrades it to a ``Raise``, so a wrong-typed /
    # invalid return can't escape even when another hook's ``REPLExecComplete`` override
    # reinstated a ``Return`` past the per-turn check. The transformed result is what
    # ``InvokeExit`` then carries, and what the invoke actually returns.
    #
    # Typed ``TerminalExecResult`` (not the full ``ExecResult``): an invoke completes only on a
    # terminal result — the loop breaks solely on its Return/Raise arm — so the raw result a hook
    # sees here is never a ``Continue``. This is the same narrowing ``ModifyInvokeResult`` carries.

    result: TerminalExecResult


@dataclass(frozen=True)
class InvokeExit(Event):
    """Fired when the invoke span closes, with the invoke's final outcome.

    **Fires once per invoke**, not per turn. ``outcome`` is the tagged union
    :data:`InvokeOutcome`: :class:`Completed` carries the invoke's **final, post-transform**
    terminal result (``outcome.result`` — any :class:`ModifyInvokeResult` composed at
    :class:`InvokeComplete` already applied), exactly what the invoke returns or raises to
    its caller. :class:`Aborted` means this invoke's own hook control plane ended it via
    :class:`Abort` — at whichever stage the effect composed (``outcome.exception``
    carries the abort's error, which the invoke raises to its caller). :class:`Failed`
    carries every other error that escaped before the invoke produced a result — a REPL
    that fails to initialize, a ``FatalError`` propagating from below, a child invoke's
    abort passing through; the error still propagates after the event fires.

    Observation-only: no effect is valid here — the outcome is already final (an effect
    returned here raises :class:`~jaz.exceptions.InvalidEffectError`). To change the terminal
    result, transform it at :class:`InvokeComplete`. (Unlike the *per-turn* ``*Exit`` events —
    :class:`LLMQueryExit` / :class:`REPLExecExit`, which accept an :class:`Abort` on **every** arm —
    ``InvokeExit`` stays fully observation-only: an invoke aborting at its own
    terminal Exit would make this event's ``Completed`` payload diverge from the exception the
    caller then receives, with no enclosing span to record the abort, and :class:`InvokeComplete`
    already gives full control over the final result.)

    Timing: like every event, this exit carries only :attr:`~jaz.hooks.events.Event.timestamp`
    (its emission time); the span's duration is ``Exit.timestamp - Enter.timestamp``.

    Attributes:
        outcome: How the span ended, with its payload — match on the variant
            (``Completed[Return | Raise] | Aborted | Failed``).
    """

    # The Completed payload being the POST-transform value is the #906 fix
    # (span_event_lifecycle.md): observers of this event finally record what the invoke
    # actually returned. See the Complete→Exit ordering comment in ``span_invoke``.
    #
    # No time fields: the former start_time/end_time (interval-on-record, #1011) were
    # replaced by the base ``Event.timestamp`` — see its field comment in hooks/base.py
    # for the decision record.
    #
    # ``outcome`` has no default — see the rationale on ``LLMQueryExit``.

    outcome: InvokeOutcome


@dataclass
class InvokeSendContext(ExecutionContext):
    """Context for invoke *send* events.

    The invoke-boundary twin of ``REPLExecSendContext``: hooks can supply the invoke's terminal
    result via :class:`SupplyInvokeResult` (``supply_effects`` — ``span_invoke``'s send emitter
    resolves them to ``span.supplied`` and the agent skips the whole loop), or abort via
    :class:`Abort` (``abort_errors`` — the send emitter raises their combination; the span closes
    ``Aborted``).
    """

    # ``supply_effects`` holds the ``SupplyInvokeResult``s (each carrying a terminal ExecResult);
    # ``abort_errors`` the Abort exceptions. Left untyped-element (plain ``list``) for the same
    # reason as ``InvokeCompleteContext.modify_effects``: importing the effect class here creates
    # an events→effects cycle that mis-resolves the frozen generic ``Event`` dataclasses in this
    # module. The dispatcher's ``_compose_invoke_send`` is the only writer and appends the right type.
    supply_effects: list = field(default_factory=list)
    abort_errors: list[Exception] = field(default_factory=list)


@dataclass
class InvokeCompleteContext(ExecutionContext):
    """Context for invoke *complete* events (#568).

    The invoke-boundary twin of ``REPLExecCompleteContext``: the invoke's terminal result is a
    **transform** boundary, so a hook may replace it via :class:`ModifyInvokeResult` (or abort via
    :class:`Abort`). The effects are collected here and resolved against the actual terminal result by
    ``resolve_invoke_modify_results`` in ``span_invoke``. Used by :class:`ReturnType` / :class:`ValidateReturn` to
    downgrade a wrong-typed / invalid final :class:`Return` to a :class:`Raise` — the backstop that survives
    another hook's :class:`REPLExecComplete` override reinstating a :class:`Return`. Unlike the REPL
    boundary its effect is :class:`ModifyInvokeResult`, not :class:`ModifyExecResult`: an invoke
    completes only on a terminal result, so a :class:`Continue` cannot be composed here (the effect
    rejects one at construction).
    """

    # Transform / abort effects, mirroring REPLExecCompleteContext: ``modify_effects`` holds the
    # ``ModifyInvokeResult``s (each carrying a terminal ExecResult) and ``abort_errors`` the Abort
    # exceptions (resolve to a Raise). Left untyped-element (plain ``list``) deliberately: importing
    # ``ModifyInvokeResult`` (in the effects layer) here creates an events→effects cycle that makes pyright
    # mis-resolve the frozen generic ``Event`` dataclasses in this module. The dispatcher's
    # ``_compose_invoke_complete`` is the only writer and appends the right types.
    modify_effects: list = field(default_factory=list)
    abort_errors: list[Exception] = field(default_factory=list)


@dataclass
class InvokeContext(ExecutionContext):
    """Context for invoke enter events.

    Hooks can:
    - Short-circuit the whole invoke with an Abort (abort before the first
      iteration): the loop is skipped and the combined carried exception is raised
      instead (see ``abort_errors``).
    - Add invoke inputs (``added_inputs``, via :class:`AddInputs`) — including tool namespaces /
      libraries, now ordinary inputs (the dedicated AddLibrary effect was removed with the
      privileged libraries= path, see ``library_as_input.md``). Added inputs render in the
      prompt and bind in the REPL, applied before the prompt is built.
    - Drop invoke inputs (``dropped_inputs``, via :class:`DropInputs`) — un-pass an input so it is
      gone from the prompt *and* the REPL (contrast :class:`DropVariables`, which unbinds a REPL
      name only). Applied *before* any :class:`AddInputs`, so dropping and adding the same key
      replaces it (provided the caller passed that key), and a drop never removes an input
      another hook added.
    - Suppress the recursive ``invoke`` tool for this invoke with a
      :class:`DisableRecursion` effect (``recursion_disabled``): the primitive binds
      the REPL with ``invoke_tool=None``, so the agent never sees ``invoke``.
      This is the hook-driven successor to the framework's former ``max_depth``
      structural affordance-removal (see :class:`RecursionLimit`).
    - Record metrics

    The result-scoped effects (SupplyExecResult / ModifyExecResult / ModifyInvokeResult) are not
    valid here (an invoke has no execution result to supply or transform yet), and ``Finish`` (a graceful
    Return-terminate) is deliberately excluded (#481, YAGNI) — so Abort is the only
    terminating effect at invoke enter.
    """

    # Inputs added by ``AddInputs`` (name -> value). Applied to the invoke's inputs before the
    # prompt renders, so they show in the prompt and bind in the REPL.
    added_inputs: dict[str, object] = field(default_factory=dict)

    # Input names dropped by ``DropInputs`` (union). Un-passed before the prompt renders — gone
    # from the prompt and the REPL. Applied BEFORE ``added_inputs``, so an add of the same key
    # wins (the drop runs while that key is still absent) and drop-then-add is a replacement.
    dropped_inputs: set[str] = field(default_factory=set)

    # Subset of ``dropped_inputs`` whose ``DropInputs`` set ``allow_missing=True`` — exempt from the
    # "dropping an input the invoke never received raises" check (a defensive drop that tolerates the
    # key already being un-passed by another effect). Union across hooks: one opt-in exempts the key.
    dropped_inputs_allow_missing: set[str] = field(default_factory=set)

    # Abort effects: the exceptions from Aborts (``abort_errors`` — they resolve to
    # a ``Raise`` that short-circuits the invoke before its first iteration).
    abort_errors: list[Exception] = field(default_factory=list)

    # Set True by a ``DisableRecursion`` effect: the primitive binds this invoke's REPL with
    # ``invoke_tool=None`` (no recursive ``invoke`` tool). Idempotent — any number of
    # emitters collapse to this one boolean. Its inverse rides ``LLMQueryEnter`` as
    # ``can_recurse`` so the must-exit warnings gate their delegate guidance correctly.
    recursion_disabled: bool = False


class InvokeSpan:
    """Span for invoke.

    Usage:
        with dispatcher.span_invoke(...) as span:
            result = agent._invoke_internal(...)
            span.complete(result=result)

    An :class:`Abort` never surfaces on the span: the ``span_invoke`` context manager
    raises the carried exception itself (from the ``with`` statement at enter, from
    ``send()`` at the committed-input veto), after closing the span with an ``Aborted``
    outcome. The former ``abort`` / ``send_abort`` fields were the legacy laundering
    channel and are gone.
    """

    def __init__(self, ctx: InvokeContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        # An invoke completes only on a terminal result (see ``InvokeComplete.result``), so the
        # span holds a ``TerminalExecResult`` throughout — set at ``complete`` and possibly
        # replaced by a ``ModifyInvokeResult`` override at ``set_final_result``.
        self._result: TerminalExecResult | None = None
        # A Send-time ``SupplyInvokeResult`` short-circuit, resolved by ``span_invoke``'s send
        # emitter and read by the agent after ``send()``: when non-None the agent skips the whole
        # loop and completes with it (mirrors ``REPLExecSpan.supplied``). Terminal by the effect's
        # own ``__post_init__``, so it is a valid ``complete(result=...)`` argument.
        self.supplied: TerminalExecResult | None = None
        # Emitter for InvokeSend, installed by ``span_invoke`` (the span is pure data and
        # must not import the dispatcher; the CM closes over the enter event + composed
        # input effects so ``send()`` needs only the committed inputs). None on a
        # hand-built span.
        self._emit_send: Callable[[Mapping[str, object]], None] | None = None

    def send(self, inputs: Mapping[str, object]) -> None:
        """Fire :class:`InvokeSend` with this invoke's committed inputs.

        Called by the agent loop once the enter-time :class:`AddInputs` / :class:`DropInputs`
        have been folded into the invoke's inputs — i.e. exactly when the input set commits,
        before the prompt renders its first turn. A Send-composed :class:`Abort` raises its
        carried exception from this call (the span closes ``Aborted``).

        Raises:
            RuntimeError: If the span was not created by ``span_invoke``.
        """
        if self._emit_send is None:
            raise RuntimeError(
                "InvokeSpan.send() requires a span created by span_invoke"
            )
        self._emit_send(inputs)

    def complete(self, *, result: TerminalExecResult) -> None:
        """Complete span with invocation result.

        Args:
            result: The invocation's terminal result — a ``Return`` or a ``Raise`` (an invoke
                never completes on a ``Continue``).

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

    def get_result(self) -> TerminalExecResult:
        """Get the invocation result.

        Returns:
            The terminal result provided to complete() (or the ``set_final_result`` override) —
            a ``Return`` or a ``Raise``.

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        assert self._result is not None
        return self._result

    def set_final_result(self, result: TerminalExecResult) -> None:
        """Replace the completed result with an ``InvokeComplete``-composed override.

        Called only by ``span_invoke`` after emitting ``InvokeComplete`` and resolving any
        ``ModifyInvokeResult`` (mirrors ``REPLExecSpan.set_final_exec_result``). The caller then reads
        the (possibly overridden) terminal result via :meth:`get_result` *after* the span closes.
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        self._result = result
