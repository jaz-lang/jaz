"""Invoke event, contexts, and span."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from jaz.library import Library
from jaz.repl.types import ExecResult

from ..base import Event, ExecutionContext


@dataclass(frozen=True)
class InvokeEnter(Event):
    """Fired when Agent.invoke is called, before any execution begins.

    **Fires once per invoke**, not per turn — one of these opens every invoke, including each
    nested sub-invoke, before the first LLM query.

    Allowed effects:

    - :class:`Abort` — terminate the invoke before its first turn.
    - :class:`AddInputs` — add an input, visible in the prompt and bound in the REPL.
    - :class:`DropInputs` — remove an input from both the prompt and the REPL.
    - :class:`DisableRecursion` — withhold the recursive ``jaz.invoke`` tool from this invoke.

    Attributes:
        parent_invoke_id: The enclosing invoke's id, or ``None`` at the top level.
        parent_repl_iteration: The enclosing invoke's iteration that spawned this one, or
            ``None`` at the top level.
        inputs: The explicit ``**inputs`` passed to ``jaz.invoke``, as a read-only mapping.
        jaz_library: The JAZ library bound into this invoke, or ``None`` when recursion is
            disabled.
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
    # The framework JAZ library bound into this invoke (jaz.invoke + opted-in
    # jaz.* exports), or None. A ``DisableRecursion`` effect emitted in response to THIS
    # event (by ``RecursionLimit`` at the cap leaf) causes the primitive to bind the REPL
    # with this nullified — the recursion-cap affordance-removal now lives in the effect
    # layer, not a framework ``max_depth`` field.
    jaz_library: Library | None
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
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "scope", MappingProxyType(dict(self.scope)))


@dataclass(frozen=True)
class InvokeExit(Event):
    """Fired when Agent.invoke completes or raises.

    **Fires once per invoke**, not per turn, on every path that reaches a terminal result —
    including when a hook emits an :class:`Abort` at :class:`InvokeEnter`, which still completes
    with that :class:`Raise`. It does *not* fire if an error escapes before the invoke produces a
    result at all (a REPL that fails to initialize, say); that case is logged, not reported as
    an exit.

    Allowed effects:

    - :class:`ModifyResult` — replace the invoke's terminal result.
    - :class:`Abort` — terminate the invoke with an error.

    Attributes:
        result: The invoke's terminal result — a :class:`Return` or a :class:`Raise`.
    """

    # This is a **transform** boundary, unlike the enter-time ``InvokeContext``: it is the last
    # line of defense for a return contract. A ``ReturnType`` / ``ValidateReturn`` hook re-checks
    # the *final* ``Return`` here and downgrades it to a ``Raise``, so a wrong-typed / invalid
    # return can't escape even when another hook's ``REPLExecExit`` override reinstated a
    # ``Return`` past the per-turn check.
    #
    # Known limitation (#906): the event carries the PRE-transform result, while the invoke
    # returns the POST-transform one — so a passive observer keyed on ``event.result`` misses an
    # InvokeExit downgrade. Inherent to the event being the transform's *input*.

    result: ExecResult


@dataclass
class InvokeExitContext(ExecutionContext):
    """Context for invoke *exit* events (#568).

    Symmetric with ``REPLExecExitContext``: the invoke's terminal result is a **transform**
    boundary, so a hook may replace it via :class:`ModifyResult` (or terminate via :class:`Abort`). The
    effects are collected here and resolved against the actual terminal result by
    ``resolve_modify_results`` in ``span_invoke``. Used by :class:`ReturnType` / :class:`ValidateReturn` to
    downgrade a wrong-typed / invalid final :class:`Return` to a :class:`Raise` — the backstop that survives
    another hook's :class:`REPLExecExit` override reinstating a :class:`Return`.
    """

    # Transform / terminate effects, mirroring REPLExecExitContext: ``modify_effects`` holds the
    # exit-time ``ModifyResult``s (each carrying a full ExecResult) and ``abort_errors`` the Abort
    # exceptions (resolve to a Raise). Left untyped-element (plain ``list``) deliberately: importing
    # ``ModifyResult`` (in the effects layer) here creates an events→effects cycle that makes pyright
    # mis-resolve the frozen generic ``Event`` dataclasses in this module. The dispatcher's
    # ``_compose_invoke_exit`` is the only writer and appends the right types.
    modify_effects: list = field(default_factory=list)
    abort_errors: list[Exception] = field(default_factory=list)


@dataclass
class InvokeContext(ExecutionContext):
    """Context for invoke enter events.

    Hooks can:
    - Short-circuit the whole invoke with an Abort (terminate before the first
      iteration): the loop is skipped and the composed Raise is raised instead
      (see ``abort_errors``).
    - Add invoke inputs (``added_inputs``, via :class:`AddInputs`) — including tool namespaces /
      libraries, now ordinary inputs (the dedicated AddLibrary effect was removed with the
      privileged libraries= path, see ``library_as_input.md``). Added inputs render in the
      prompt and bind in the REPL, applied before the prompt is built.
    - Drop invoke inputs (``dropped_inputs``, via :class:`DropInputs`) — un-pass an input so it is
      gone from the prompt *and* the REPL (contrast :class:`DropVariables`, which unbinds a REPL
      name only). Applied *before* any :class:`AddInputs`, so dropping and adding the same key
      replaces it (provided the caller passed that key), and a drop never removes an input
      another hook added.
    - Suppress the recursive ``jaz.invoke`` tool for this invoke with a
      :class:`DisableRecursion` effect (``recursion_disabled``): the primitive binds
      the REPL with ``jaz_library=None``, so the agent never sees ``jaz.invoke``.
      This is the hook-driven successor to the framework's former ``max_depth``
      structural affordance-removal (see :class:`RecursionLimit`).
    - Record metrics

    The result-scoped effects (OverrideResult / ModifyResult) are not valid here (an invoke
    has no execution result to supply or transform), and ``Finish`` (a graceful
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

    # Terminate effects: the exceptions from Aborts (``abort_errors`` — they resolve to a
    # ``Raise`` that short-circuits the invoke before its first iteration).
    abort_errors: list[Exception] = field(default_factory=list)

    # Set True by a ``DisableRecursion`` effect: the primitive binds this invoke's REPL with
    # ``jaz_library=None`` (no recursive ``jaz.invoke`` tool). Idempotent — any number of
    # emitters collapse to this one boolean. Its inverse rides ``LLMQueryEnter`` as
    # ``can_recurse`` so the must-exit warnings gate their delegate guidance correctly.
    recursion_disabled: bool = False


class InvokeSpan:
    """Span for invoke.

    Usage:
        with dispatcher.span_invoke(...) as span:
            if span.enter_override is not None:
                result = span.enter_override  # a hook aborted the invoke
            else:
                result = agent._invoke_internal(...)
            span.complete(result=result)

    ``enter_override`` is set by the dispatcher from enter-time Aborts.
    """

    def __init__(self, ctx: InvokeContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        self._result: ExecResult | None = None
        # ExecResult produced by an enter-time Abort, if any. When set, the
        # caller should skip the invoke body.
        self.enter_override: ExecResult | None = None

    def complete(self, *, result: ExecResult) -> None:
        """Complete span with invocation result.

        Args:
            result: The result of the invocation (a ``return`` or a ``raise``)

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

    def set_final_result(self, result: ExecResult) -> None:
        """Replace the completed result with an ``InvokeExit``-composed override.

        Called only by ``span_invoke`` after emitting ``InvokeExit`` and resolving any exit-time
        ``ModifyResult`` (mirrors ``REPLExecSpan.set_final_exec_result``). The caller then reads
        the (possibly overridden) terminal result via :meth:`get_result` *after* the span closes.
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        self._result = result
