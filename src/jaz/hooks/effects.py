"""Effect definitions for the hook system.

An effect is what a hook handler returns to influence the run — see :class:`jaz.hooks.Hook`.
Each class below documents the events it is valid at and how multiple instances of it compose;
an effect returned at an event that does not accept it is ignored. The effects every hook
returns at an event are composed together before they are applied.

**Experimental.** The hook system is an experimental feature; its interfaces may change
in a future release.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from jaz.exceptions import (
    MessageEditIndexError,
    ReturnValueConflictError,
)
from jaz.providers import MessageDict
from jaz.repl.types import Continue, ExecResult, Raise, Return

# The public effect surface (`jaz.hooks.effects`) — the typed values a hook handler
# returns to influence execution. `Effect` is the base for annotating handler returns.
__all__ = [
    "Effect",
    "OverrideResult",
    "ModifyResult",
    "Abort",
    "DisableRecursion",
    "AddInputs",
    "DropInputs",
    "AddVariables",
    "DropVariables",
    "DropMessages",
    "AddMessages",
    "OverrideResponse",
]


@dataclass(frozen=True)
class Effect:
    """Base class for all effects — the values a hook handler returns to influence the run.

    Each subclass documents the events it is valid at and how multiple instances of it
    compose. An effect returned at an event that does not accept it is ignored.

    Effects are frozen: a hook constructs one to *declare* an intent, and it is only ever read
    from there on.
    """

    # Freezing makes the read-only contract structural — an effect can't be aliased and edited
    # out from under the compositor. Frozen inheritance is all-or-nothing, so every subclass is
    # frozen too.

    pass


# Exec-result effects — self-scoping supply/transform over the REPL execution span (#481).
# Each carries a *complete* ``ExecResult`` (``Continue`` | ``Return`` | ``Raise``), replacing
# the former field-carrying ``ContinueEffect`` / ``ReturnEffect`` pair:
#
#   ``OverrideResult(result)``  supplies a result in place of running the code — valid at
#     ``REPLExecEnter`` (where work is *pending*). It is the enter-point "supply instead"
#     effect.
#   ``ModifyResult(result)``    transforms the result that ran — valid at ``REPLExecExit``
#     (where a result *exists*). It is the exit-point "transform" effect.
#
# Both **fold** by the same precedence (``_fold_carried_results``): carried ``Continue``s
# concatenate their outputs and group their exceptions; carried ``Return``s cannot merge (two
# distinct values conflict, identical ones coalesce); carried ``Raise``s group their
# exceptions. At the exit
# boundary the carried results fold onto the executed ``original``; at the enter boundary there
# is no original, so the suppliers fold among themselves. This preserves the merge semantics the
# former ``ContinueEffect`` had. ``Override<X>`` supplies and skips the work that would produce
# ``X`` (an *enter* effect); ``Modify<X>`` transforms the ``X`` that already exists (an *exit*
# effect) — the same naming convention as ``OverrideResponse``.
#
# The two boundaries once diverged — supply required its results to be *equal* (a supply "names
# THE result", so two distinct supplies raised ``ResultConflictError``) while transform
# folded. They were unified to fold because independent enter-point vetoes (two hooks each
# rejecting the same input with a recoverable ``Continue``) should compose exactly as independent
# transforms do, not crash on a hard internal error; distinct ``Return`` *values* still conflict
# at both boundaries (they genuinely can't merge).
#
# Termination is a *separate* effect, ``Abort`` (below), NOT one of these: it un-bundles
# "stop the loop" from "supply/transform a result" (#481). ``Abort`` resolves to a terminal
# ``Raise`` and is valid at **every live event** (not just the result boundaries), so
# loop/budget control has an always-present home and never rides a conditional exit. Where it
# co-occurs at a result boundary, ``Abort`` has the highest precedence (termination trumps
# supply/transform). See ``resolve_override_results`` / ``resolve_modify_results``.


@dataclass(frozen=True)
class OverrideResult(Effect):
    """Supply a result in place of running the turn's code.

    Valid at :class:`REPLExecEnter` only — a result must not yet exist to be supplied.
    The code is not executed and the carried ``result`` becomes the turn's outcome.

    Composition: multiple supplies fold. Kind precedence is :class:`Raise` > :class:`Return` >
    :class:`Continue` — when supplies of different kinds meet, the highest present wins and the
    lower ones drop. Within one kind: two :class:`Continue` results concatenate their outputs
    and group their exceptions; two distinct :class:`Return` values raise
    :class:`~jaz.exceptions.ReturnValueConflictError` (identical values coalesce); two
    :class:`Raise` results group their exceptions. An :class:`Abort` supersedes the whole fold.

    Because only the ``output`` field of :class:`Continue` is shown to the agent, a supplied
    :class:`Continue` should render any error into its ``output`` field, otherwise the agent
    does not see it — e.g. ``Continue(output=summarize_exception(exc), exception=exc)``.

    Attributes:
        result: The :class:`Continue` / :class:`Return` / :class:`Raise` to use instead of executing.
    """

    # This is the enter-boundary "supply instead" slot for sandbox/approval hooks generally.
    # ``ValidateREPLInput`` emits it: on a rejected REPL input it supplies a ``Continue``
    # (recoverable), or a ``Raise`` at the failure cap. See ``resolve_override_results``.
    #
    # The ``Continue`` merge is what lets two such hooks reject the SAME input without one of
    # them losing or the fold raising — their outputs concatenate and their exceptions group.
    # Kept out of the docstring: the rule ("two Continue results concatenate their outputs and
    # group their exceptions") already says it without the jargon.

    result: ExecResult


@dataclass(frozen=True)
class ModifyResult(Effect):
    """Replace a result that already exists with the carried one.

    Valid at :class:`REPLExecExit` and :class:`InvokeExit` — both are transform boundaries, so
    a result must exist to replace.

    Composition: multiple transforms fold by carried kind. Kind precedence is :class:`Raise` >
    :class:`Return` > :class:`Continue` — when transforms of different kinds meet, the highest
    present wins and the lower ones drop. Within one kind: a :class:`Continue` concatenates its
    ``output`` onto the original's and groups its exception; two distinct :class:`Return` values
    raise :class:`~jaz.exceptions.ReturnValueConflictError` (identical values coalesce); two
    :class:`Raise` results group theirs. An :class:`Abort` supersedes the fold.

    Because only the ``output`` field of :class:`Continue` is shown to the agent, a carried
    :class:`Continue` should render any error into its ``output`` field, otherwise the agent
    does not see it — e.g. ``Continue(output=summarize_exception(exc), exception=exc)``.

    Attributes:
        result: The :class:`Continue` / :class:`Return` / :class:`Raise` to replace the actual result with.
    """

    # ``output`` is the only agent-facing surface (#928) — ``exception`` is structured metadata
    # with no rendered form of its own, which is why the own-output rule above matters. The fold
    # never renders the grouped exception, so a ``Continue(exception=…, output="")`` contributes
    # an error the agent never sees. ``BudgetForcing`` / ``ReturnType`` / ``ValidateReturn`` all
    # use that shape — ``BudgetForcing``, for instance, turns an early ``Return`` into a
    # ``Continue`` refusal. See ``resolve_modify_results``.
    #
    # Downgrading a terminal result this way shows the agent only the carried text: a ``Return``
    # or ``Raise`` has no ``output`` (#903), so whatever the turn printed before finishing is not
    # carried into the resulting ``Continue``. That is the one thing to know before writing a
    # downgrading hook; ``_fold_carried_results`` documents it at the fold itself.

    result: ExecResult


@dataclass(frozen=True)
class Abort(Effect):
    """Terminate the invoke, surfacing ``error`` as a :class:`Raise` out of ``jaz.invoke()``.

    Valid at every event that accepts effects — :class:`InvokeEnter`, :class:`InvokeExit`,
    :class:`LLMQueryEnter`, :class:`REPLExecEnter`, :class:`REPLExecExit`. Not at
    :class:`LLMQueryRetry`, which accepts nothing.

    Composition: it has the **highest precedence** wherever it co-occurs with the result
    effects, superseding both :class:`OverrideResult` and :class:`ModifyResult`. Multiple
    aborts group their exceptions into an ``ExceptionGroup``.

    An abort resolves to a plain :class:`Raise`, so downstream it is indistinguishable from an agent
    ``raise`` — it carries no separate provenance.

    Attributes:
        error: The exception to terminate the invoke with.
    """

    # Loop and budget hard-stops emit it at the always-present ``LLMQueryEnter`` (once per turn,
    # before the LLM call), so a stop is never tied to the *conditional* ``REPLExecExit``, which
    # is skipped on a parse-failure turn that has no execution.
    #
    # Termination is a first-class, self-scoping effect rather than a mode of the result effects
    # (#481): un-bundling "stop the loop" from "supply/transform a result" is what lets it be
    # valid at events that have no result at all — it references no existing result, which is
    # why it is accepted at every effect-bearing event.
    #
    # It is NOT the only way to stop the loop: an ``OverrideResult`` / ``ModifyResult`` carrying
    # a ``Return`` or a ``Raise`` ends the invoke too. What is unique to ``Abort`` is that it can
    # stop from events that have no result to carry one (``InvokeEnter``, ``LLMQueryEnter``).

    error: Exception


@dataclass(frozen=True)
class DisableRecursion(Effect):
    """Withhold the recursive ``jaz.invoke`` tool from this invoke.

    Valid at :class:`InvokeEnter` only.

    Composition: a pure signal carrying no fields, so emitting it more than once is the same as
    emitting it once.
    """

    # Only coherent at ``InvokeEnter`` because recursion availability is fixed when the REPL is
    # built — there is no later point at which suppressing it means anything.
    #
    # Cut from the docstring: the tool is removed structurally rather than made to fail when
    # called (the agent never sees ``jaz.invoke`` at all), and the invoke still runs to
    # completion — ``Abort`` is the effect that stops it. Both are covered by the summary line
    # plus ``Abort``'s own docstring.
    #
    # Emitted by ``RecursionLimit`` on the cap-leaf invoke (``event.depth == max_depth``) for
    # affordance removal: the primitive honors it by binding the REPL with ``jaz_library=None``.
    # It also publishes the inverse as ``can_recurse`` on ``LLMQueryEnter``, so the must-exit
    # warnings gate their "you may delegate" guidance correctly.
    #
    # The over-cap backstop — a child that enters past the cap anyway, e.g. the public
    # ``jaz.invoke`` reached from a human tool — is handled by ``Abort``, not this effect.

    pass


# Prompt modification effects
#
# Instruction text is no longer a dedicated effect: hooks add it to the model's prompt
# via ``AddMessages`` (below) emitted at ``LLMQueryEnter``, folded into the query with the
# rest of the message edits. This collapsed the old ``AddInstructionPrompt`` (which baked a
# trailing user message into the query snapshot ahead of the fold, at a different index
# base) onto the single message-edit coordinate system — see #660 (and the #596 bug the
# index-base asymmetry caused). System-prompt text is likewise no longer a dedicated
# effect: the former ``AddSystemPrompt`` (invoke-scope text contributed to the *system*
# prompt) was removed together with its sole emitter, ``ParentUpdatesHook`` — with no
# remaining producer, the effect and its ``system_prompt_additions`` plumbing (dispatcher
# composition → ``InvokeContext`` → the protocol signature → the prompt template) were
# dropped rather than kept as dead code.


# REPL input effects


@dataclass(frozen=True)
class AddInputs(Effect):
    """Add inputs to the invoke, equivalent to adding more ``name=value`` pairs as keyword
    arguments to ``jaz.invoke``.

    Valid at :class:`InvokeEnter` only. Each entry both renders in the user prompt and binds as
    a REPL variable. For a namespace-only binding that leaves the prompt alone, use
    :class:`AddVariables`.

    Composition: all ``inputs`` are unioned; the same key with the same value coalesces, and
    with a *different* value raises :class:`~jaz.exceptions.REPLInputConflictError`, so no hook
    silently overwrites another. A key colliding with an input or scope entry passed to the
    invoke also raises — unless a :class:`DropInputs` at the same invoke names that key, which
    clears it first and makes the pair a replacement. (That drop only works on a key the invoke
    was passed; one that another hook merely adds raises
    :class:`~jaz.exceptions.MissingDropTargetError` instead.) Naming ``__builtins__`` raises
    :class:`~jaz.exceptions.SandboxKeyError`. An add always wins over a :class:`DropInputs` of
    the same key: drops are applied first.

    Values resolve through the same ``__jaz_get__`` protocol as a top-level input, so a wrapper
    that defines it binds its payload rather than the wrapper itself.

    Attributes:
        inputs: Name-to-value pairs to add; each renders in the prompt and binds in the REPL.

    Examples:
        AddInputs({"config": {"api_key": "abc"}})
    """

    # The ``__jaz_get__`` wrappers in tree today are ``Library`` and the ``jaz.Display``
    # directive; both are named here rather than in the docstring because they are expected to
    # leave the public API. The protocol sentence is what a caller needs and outlives them.

    # Only valid at ``InvokeEnter`` because inputs are per-invoke setup, applied once before the
    # prompt renders — cut from the docstring, which now just states where it is valid.
    #
    # The prompt-level Add of the symmetric family; its inverse is ``DropInputs``, and the
    # input-level vs namespace-level split mirrors ``DropInputs`` vs ``DropVariables``.
    #
    # Not valid at the conditional ``REPLExecEnter``, which is skipped on a parse-failure turn
    # and so is a poor home for injection; per-turn namespace injection is ``AddVariables``.

    inputs: dict[str, object]


@dataclass(frozen=True)
class DropInputs(Effect):
    """Un-pass invoke inputs by name, equivalent to removing ``name=value`` keyword arguments
    from ``jaz.invoke``.

    Valid at :class:`InvokeEnter` only. Each listed key is removed as if it had never been
    passed to ``jaz.invoke``, so it is removed from both the prompt and the REPL. Contrast
    :class:`DropVariables`, which unbinds a REPL name only and leaves the prompt rendering
    intact.

    Composition: ``keys`` are unioned across all drops, order-independently. Drops are applied
    before any :class:`AddInputs`, so dropping and adding the same key replaces it provided that
    key was passed to the invoke, and a drop can only remove what the invoke was passed — never
    an input another hook added.

    Dropping a key the invoke never received raises
    :class:`~jaz.exceptions.MissingDropTargetError` — a typo or a re-drop-every-turn pattern is
    a hook bug. A key that only an :class:`AddInputs` provides counts as never received. Set
    ``allow_missing=True`` for a defensive drop that tolerates the key already being absent; it
    is tracked per-key, so one tolerant hook suffices. Note that this silences the error without
    making the drop remove an added key — the add still wins.

    Attributes:
        keys: The input names to un-pass.
        allow_missing: Whether dropping an absent key is tolerated rather than an error.

    Examples:
        DropInputs({"secret_token"})
        DropInputs({"maybe_absent"}, allow_missing=True)
    """

    # Only valid at ``InvokeEnter`` because inputs are fixed at invoke setup, before the prompt
    # renders — cut from the docstring, which now just states where it is valid.
    #
    # The add-before-drop ordering is ``Agent._apply_input_effects``: the ``if added:`` branch
    # merges into ``inputs``/``resolved_bound`` first, and the ``if dropped:`` branch then
    # computes ``present`` from that post-add snapshot. Stated in the docstring because the two
    # rules read as contradictory otherwise — "adding and dropping resolves to the drop" against
    # "dropping a key the invoke never received raises".
    #
    # Executive decision, superseding the ``add_drop_effect_family`` doc: that doc proposed
    # ``DropInputs`` as the migration target for ``expose_inputs_in_repl=False`` (withhold the
    # REPL binding but KEEP the inputs in the prompt). Rejected — this effect is kept strictly
    # symmetric with ``AddInputs`` and parallel to ``DropVariables``/``DropMessages``: it
    # literally drops the input, from prompt AND REPL. Making it prompt-preserving to serve one
    # caller would break the family's symmetry (an Add/Drop pair that don't invert each other)
    # for no gain, since ``DropVariables`` already covers withhold-binding-keep-prompt. So
    # ``expose_inputs_in_repl`` stays on a one-shot ``DropVariables`` — see
    # ``evals/repl_config_hooks.py``'s ``WithholdInputsFromREPL``.

    keys: set[str]
    allow_missing: bool = False


@dataclass(frozen=True)
class AddVariables(Effect):
    """Bind names into the agent's REPL namespace before a turn's code runs.

    Valid at :class:`REPLExecEnter` only. Each name is bound as a REPL variable without touching
    the prompt. Values are bound raw, with no ``__jaz_get__`` payload substitution.

    Composition: all ``variables`` are unioned; identical (name, value) coalesces and a
    divergent value for the same name raises :class:`~jaz.exceptions.REPLInputConflictError`.
    Binding a name already present in the *variable* namespace also raises rather than
    clobbering it. A name that lives only in ``__builtins__`` — an allowed builtin such as
    ``len``, or a framework magic such as ``__repl_history__`` — does not count as already
    bound, so adding it succeeds and shadows what was there. Naming ``__builtins__`` itself
    raises :class:`~jaz.exceptions.SandboxKeyError`.

    Drops are applied before adds, so pairing a :class:`DropVariables` with an add re-binds a
    name, and an add wins over a drop of the same name in the same turn. Pair with
    ``DropVariables(..., allow_missing=True)`` whenever the name may not be bound yet — on the
    turn it is still unbound, a strict drop raises
    :class:`~jaz.exceptions.MissingDropTargetError` before the add ever runs.

    Attributes:
        variables: Name-to-value pairs to bind in the REPL namespace.

    Examples:
        AddVariables({"scratch": {}})
    """

    # Only valid at ``REPLExecEnter`` because the namespace matters exactly when code is about
    # to run — cut from the docstring, which now just states where it is valid. Also cut: "for a
    # prompt-rendering input use ``AddInputs``". That reads as an equivalence and is not one —
    # ``AddInputs`` is invoke setup, fixed before the first prompt renders, while this fires per
    # turn and is typically used well after turn 1.
    #
    # The namespace-level Add of the symmetric family and the direct counterpart of
    # ``DropVariables``: where that deletes names from ``repl_state_locals`` before the turn,
    # this writes them. A hook typically binds once (emit on ``event.iteration == 1``).
    #
    # Drops run before adds (``Agent.do_one_repl_iteration``), matching
    # ``_apply_input_effects``. Add-first used to make the documented re-bind recipe impossible:
    # the add hit the already-bound check while the old value was still there
    # (``REPLInputConflictError``), and on an unbound name the pair bound then immediately
    # unbound. The cost of the reorder is that a drop can no longer remove a name another effect
    # adds in the same turn — the same trade the input family makes, for the same reason.
    #
    # The already-present check is ``name in self.repl_state_locals``, which is the *variable*
    # namespace only — ``__builtins__`` is a separate dict, so a name living there is invisible
    # to the guard and gets shadowed rather than rejected. Measured: adding ``len`` or
    # ``__repl_history__`` binds, while adding an existing variable raises. The docstring says
    # "shadows" rather than "is not checked" because the outcome, not the mechanism, is what a
    # hook author is deciding about.

    variables: dict[str, object]


@dataclass(frozen=True)
class DropVariables(Effect):
    """Remove names from the agent's REPL namespace before a turn's code runs.

    Valid at :class:`REPLExecEnter` only. Each listed name is deleted before the turn's code
    executes, so referencing it raises an ordinary ``NameError``. It reaches any name bound in
    the namespace, whatever put it there — an invoke input, a variable an earlier turn assigned,
    or a framework magic such as ``__repl_history__``.

    The prompt is untouched. If the dropped name happened to come from an invoke input, that
    input still renders in the user prompt; un-passing an input is :class:`DropInputs`, which
    acts on a different thing — the invoke's inputs, fixed once at :class:`InvokeEnter`.

    Composition: every drop's ``names`` are combined into one set and applied once, so the
    outcome does not depend on hook order. Two hooks naming the same bound name therefore
    produce a single deletion.

    Dropping a name that is not bound raises :class:`~jaz.exceptions.MissingDropTargetError`.
    Pass ``allow_missing=True`` to ignore names that are not bound rather than raising;
    if an unbound name is dropped by both ``DropVariables(..., allow_missing=False)`` and ``DropVariables(..., allow_missing=False)``,
    then the drop is ignored and will not raise. Naming ``__builtins__`` always raises
    :class:`~jaz.exceptions.SandboxKeyError`, which ``allow_missing`` cannot suppress.

    Drops are applied before any :class:`AddVariables`, so pairing the two re-binds a name, and
    a drop only removes what was bound before the turn — never a name an add supplies. If the
    name may not be bound yet, the drop needs ``allow_missing=True``, or it raises before the
    paired add can bind it.

    Attributes:
        names: The REPL names to unbind.
        allow_missing: Whether dropping an unbound name is tolerated rather than an error.

    Examples:
        # Hide the framework REPL-history list from the agent this turn.
        DropVariables({"__repl_history__"})
        # Defensively unbind an input that another effect may already have un-passed.
        DropVariables({"maybe_unpassed"}, allow_missing=True)
    """

    # The docstring used to say this "unbinds the REPL name only — it does not un-pass an invoke
    # input", pointing at ``DropInputs`` "to remove it everywhere". That framed the two as one
    # operation at two strengths, which they are not: this acts on the REPL namespace (any name,
    # whatever bound it — an input, an agent assignment, ``__repl_history__``), while
    # ``DropInputs`` acts on the invoke's *inputs* and is only valid at ``InvokeEnter``. A
    # per-turn hook cannot reach for it as "the stronger version", and most names this drops were
    # never inputs at all.
    #
    # ``__builtins__`` is refused at composition (``_compose_repl_exec_enter``) rather than in
    # the REPL, which is why ``allow_missing`` cannot reach it — the effect never gets as far as
    # the missing-target check. ``PythonREPL.drop_variables`` also skips the key defensively.
    #
    # ``allow_missing`` composes as a union of names, not a per-effect flag
    # (``ctx.dropped_variables_allow_missing |= names``), so a strict drop of a name another
    # hook opted into does not raise either. That is deliberate: the tolerant hook is the one
    # that knows the name may be absent.
    #
    # Emit on the iteration(s) where the target is actually bound. In practice that is the first
    # iteration for both the ``expose_inputs`` case (drop the binding once, then leave the name
    # free) and the ``repl_history`` case (bound once at init, core appends by reference, so one
    # drop is permanent — re-dropping next turn would raise). The effect is agnostic; the
    # emitting hook chooses when the target is present.
    #
    # ``allow_missing`` exists for ``WithholdInputsFromREPL``, which drops an input's REPL
    # binding on turn one when a ``DropInputs`` may have already un-passed it — without the
    # opt-out that legitimate composition would raise.

    names: set[str]
    allow_missing: bool = False


# Blackboard effects


@dataclass(frozen=True)
class BlackboardWrite(Effect):
    """Write a key/value into the per-invoke blackboard.

    Valid at **any** event — the blackboard spans the whole invoke rather than one context.
    Writes are applied at the event boundary after every hook has run, so a write is invisible
    to the event that produced it and surfaces to the next one. That is what lets a producer
    and a consumer hook coordinate without depending on dispatch order.

    Composition: within one event, identical writes coalesce and two writes to the same key
    with different values raise :class:`~jaz.exceptions.BlackboardWriteConflictError`.

    Attributes:
        key: The blackboard key to write.
        value: The value to publish under that key.

    Examples:
        # A tracking hook publishes accumulated cost for a forcing hook to read next event.
        BlackboardWrite(key="cost", value=0.05)
    """

    key: str
    value: object


# LLM query effects


@dataclass(frozen=True)
class DropMessages(Effect):
    """Drop messages by index from the list sent to the model.

    Valid at :class:`LLMQueryEnter` only. ``indices`` are positions into the message list as
    seen on that event. Indices follow list semantics: a negative index counts from the end,
    and an out-of-range one raises :class:`~jaz.exceptions.MessageEditIndexError`.

    Composition: multiple drops compose by **union** of their indices, so the result is
    independent of hook order and no hook can corrupt another's view — hooks only mark
    positions, and the message list is never mutated mid-composition.

    Attributes:
        indices: Positions to drop from the message list.
        persistent: ``False`` (default) applies the drop to this turn's message list only;
            ``True`` makes the drop survives into later turns.
    """

    # Contrast an arbitrary message transform, which is order-dependent and can act on messages
    # a previous transform already moved or removed — that is why this marks positions instead.
    # ``persistent`` is the only axis on which a transient and a persistent drop differ; see
    # ``apply_message_edits``.

    indices: set[int]
    persistent: bool = False


@dataclass(frozen=True)
class AddMessages(Effect):
    """Insert messages into the conversation before an LLM query.

    Valid at :class:`LLMQueryEnter` only. Messages are inserted before slot ``index`` of the
    list as seen on that event; omitting ``index`` appends.
    An explicit out-of-range index raises :class:`~jaz.exceptions.MessageEditIndexError`.

    Composition is commutative — a pure function of the *set* of adds. Adds at distinct slots
    are independent; adds landing in the same slot are ordered by ``sort_key``, ties broken by
    a canonical serialization of the content. Identical adds from two hooks both appear; they
    are not coalesced, and adjacent same-role adds are not merged.

    Attributes:
        messages: The messages to insert.
        index: Slot to insert before; ``None`` (default) appends. Negative counts from the end.
        persistent: ``False`` (default) applies the add to this turn's message list only;
            ``True`` makes the add survive into later turns.
        sort_key: Used to order adds landing in the same slot; lower comes first.
    """

    # Adjacent same-role adds are deliberately not merged, so a turn where several hooks each
    # add an instruction produces multiple consecutive ``user`` messages. This reverses the old
    # ``_prepare_repl_query_messages`` behavior, which joined all hooks' instruction text into
    # one trailing message to avoid consecutive-user-message confusion: once that text moved
    # onto the message-edit coordinate system, each hook owns its own add and there is no shared
    # join point. Accepted in practice; revisit with a fold-level coalescing step only if the
    # confusion is actually observed.
    #
    # No ``reason`` field: like the rest of the family this effect carries what it does, not a
    # free-text why. The family's ``reason`` was removed after an audit found it unread on every
    # effect except this one, where it only fed a replay log and an error string, never behavior.
    # Mainstream effect systems attach reasons to *denials*, which here is ``Abort``'s exception.

    messages: list[MessageDict]
    index: int | None = None  # None ⇒ append (resolved to len(snapshot) in the fold)
    persistent: bool = False
    sort_key: float = 0.0


def _add_sort_key(add: AddMessages) -> tuple[float, str]:
    """The deterministic within-slot ordering key for an ``AddMessages`` (see its
    docstring): ``sort_key`` first, then a canonical serialization of the content.

    ``default=str`` keeps this total even if a message carries a non-JSON-serializable
    value (e.g. an in-dict provenance stamp from a hook that reuses a message dict) — the
    key stays stable rather than raising. It only affects the *ordering* string, never the
    content that is folded into the buffer."""
    return (
        add.sort_key,
        json.dumps(add.messages, sort_keys=True, ensure_ascii=False, default=str),
    )


def _resolve_edit_index(index: int, n: int, *, upper: int, kind: str) -> int:
    """Resolve a possibly-negative message-edit index against snapshot length ``n``.

    NumPy / ``list`` semantics (#595): a negative index counts from the end (``index + n``,
    so ``-1`` is the last position). The resolved index must lie in ``[0, upper]`` —
    ``upper == n`` for an add slot (``n`` = append), ``upper == n - 1`` for a drop position
    — otherwise :class:`MessageEditIndexError` is raised. Out-of-range is a hook bug (every
    index is computed against the snapshot the hook saw), and silently ignoring it would
    make an add's content vanish; clamping would misplace it. Normalizing here — before the
    fold — keeps composition commutative (a pure function of the resolved index set).
    """
    resolved = index + n if index < 0 else index
    if not (0 <= resolved <= upper):
        raise MessageEditIndexError(
            f"{kind} index {index} is out of range for a {n}-message snapshot"
        )
    return resolved


def apply_message_edits(
    snapshot: list[MessageDict],
    drops: set[int],
    adds: list[AddMessages],
) -> list[MessageDict]:
    """Fold message drop/add edits over ``snapshot``, anchored to its original indices.

    **Pure** — returns a new list, never mutates ``snapshot``. Both the persistent and
    the shown views are produced by calling this with different effect subsets, *both
    folded from the same snapshot* (the single-coordinate-system rule that keeps drops and
    adds commutative): a dropped index removes that original message; an add inserts its
    messages before its ``index`` slot; an add and a drop at overlapping positions commute
    (both reference original indices — an add "at" a dropped slot still inserts, since it
    is new content). Same-slot adds are ordered by :func:`_add_sort_key`.

    Indices follow NumPy / ``list`` semantics (#595): negatives count from the end (``-1``
    = the last message; append is an add ``index`` of ``None`` — the default — or the
    explicit ``len``), and an out-of-range index — after that normalization, ``[-len, len]``
    for an add, ``[-len, len-1]`` for a drop — raises :class:`MessageEditIndexError` rather
    than being ignored or clamped. A ``None`` add index is resolved to the append slot here,
    so it can never be out of range.

    **Blast radius of that raise:** this fold runs on the live query path (via
    ``Agent._compose_shown_messages``), so once a real compaction hook emits edits, an
    out-of-range index aborts the *entire invoke* mid-run — not just that one edit, not just
    that turn. That is deliberate: an OOR index is a hook bug (every index is computed
    against the snapshot the hook saw), and the fail-loud surface is the point (§#595). The
    considered-and-rejected softer alternative was surfacing it as recoverable
    ``ErrorEffect``-style feedback to the offending hook; it's moot while the mechanism is
    dormant (``SlidingWindow`` only drops in-range), and can be revisited if a
    third-party compaction hook taking down the run ever proves too sharp.
    """
    n = len(snapshot)
    normalized_drops = {
        _resolve_edit_index(d, n, upper=n - 1, kind="DropMessages") for d in drops
    }
    adds_by_slot: dict[int, list[AddMessages]] = {}
    for add in adds:
        # index=None ⇒ append; resolve to n here (the append slot) so _resolve_edit_index
        # only ever handles concrete ints — a None append can never be out of range.
        raw_index = add.index if add.index is not None else n
        slot = _resolve_edit_index(raw_index, n, upper=n, kind="AddMessages")
        adds_by_slot.setdefault(slot, []).append(add)
    for slot_adds in adds_by_slot.values():
        slot_adds.sort(key=_add_sort_key)

    result: list[MessageDict] = []
    for i in range(n + 1):
        for add in adds_by_slot.get(i, []):
            result.extend(add.messages)
        if i < n and i not in normalized_drops:
            result.append(snapshot[i])
    return result


@dataclass(frozen=True)
class OverrideResponse(Effect):
    """Supply a pre-computed LLM response, skipping the API call.

    Valid at :class:`LLMQueryEnter` only. The content, token counts, and cost are used exactly
    as if they came from a live call, including cost tracking and budget enforcement — so
    :class:`LLMQueryExit` still fires.

    Composition: identical overrides compose to one; two *distinct* overrides raise
    :class:`~jaz.exceptions.OverrideConflictError`.

    Attributes:
        content: The response text, or ``None``.
        prompt_tokens: Input token count to record, if known.
        completion_tokens: Output token count to record, if known.
        total_tokens: Combined token count to record, if known.
        cost: Cost to record for this call, if known.
    """

    content: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


# Composition helpers for Abort / OverrideResult / ModifyResult


def _combine_exceptions(excs: list[BaseException], message: str) -> BaseException:
    """Combine exceptions into a single one (a group if more than one).

    Uses ``ExceptionGroup`` when every exception derives from ``Exception`` and
    falls back to ``BaseExceptionGroup`` otherwise (e.g. an original
    ``Raise`` carrying a ``BaseException``).
    """
    if len(excs) == 1:
        return excs[0]
    if all(isinstance(e, Exception) for e in excs):
        return ExceptionGroup(message, excs)  # type: ignore[arg-type]
    return BaseExceptionGroup(message, excs)


def _abort_raise(
    abort_errors: list[Exception], original: ExecResult | None
) -> ExecResult:
    """Resolve ``Abort`` effects (+ an original ``Raise``) into a single terminal ``Raise``.

    ``Abort`` supersedes any supply/transform at a result boundary (termination trumps),
    so both resolvers call this first when any abort is present. An original ``Raise``'s exception
    is grouped with the aborts'. (No ``output`` to carry — ``Raise`` has none, #903.)
    """
    excs: list[BaseException] = list(abort_errors)
    if isinstance(original, Raise):
        excs.append(original.exception)
    # ExceptionGroup's message is agent-facing (shows in summarize_exception / str(group));
    # keep it generic and free of internal effect/implementation names.
    return Raise(exception=_combine_exceptions(excs, "Multiple errors occurred"))


def _fold_carried_results(
    carried: list[ExecResult], *, original: ExecResult | None
) -> ExecResult | None:
    """Precedence fold shared by both REPL-exec result boundaries.

    ``carried`` are the full ``ExecResult``s named by this boundary's effects (each
    ``OverrideResult`` / ``ModifyResult`` ``.result``). ``original`` is the executed result at the
    *transform* boundary (``REPLExecExit``), or ``None`` at the *supply* boundary
    (``REPLExecEnter``, where nothing has run yet). The result *kind* is decided by precedence
    ``Raise`` > ``Return`` > ``Continue`` (highest present wins; lower kinds drop):

    - any carried ``Raise`` → ``Raise``, grouping every carried ``Raise``'s exception (plus
      ``original``'s if it too is a ``Raise``);
    - else any carried ``Return`` → ``Return`` — values can't merge, so two *distinct* carried
      values raise ``ReturnValueConflictError`` (identical values coalesce);
    - else any carried ``Continue`` → ``Continue`` — the carried ``Continue``s' outputs are
      concatenated (onto ``original``'s if present) and their non-``None`` exceptions grouped
      (plus ``original``'s if it too is a ``Continue``); a carried ``Continue`` with empty output
      contributes only its exception (the ``BudgetForcing`` shape);
    - else ``None`` (nothing carried — run/keep).

    ``Raise`` / ``Return`` carry no ``output`` (#903), so the fold's ``Raise`` / ``Return`` branches
    simply replace the kind — there is no output to keep or drop, which is exactly why #903 removed
    the field (it erases the old supply-vs-transform asymmetry where a carried terminal's own output
    was silently discarded). Only the ``Continue`` branch has outputs to concatenate, and it prepends
    ``original``'s output only when ``original`` is itself a ``Continue`` (a transform of an executed,
    still-continuing turn).

    The observable consequence, and the one thing to know before adding a downgrading hook: when a
    terminal ``original`` is *downgraded* to a ``Continue`` here (``BudgetForcing``,
    ``ReturnType``), it contributes **no leading output** — the resulting ``Continue`` shows the
    agent only the carried hooks' text, not whatever the turn printed before finishing. See the
    ``Raise`` docstring in ``jaz.repl.types`` for why that loss is accepted.
    """
    raise_results = [r for r in carried if isinstance(r, Raise)]
    return_results = [r for r in carried if isinstance(r, Return)]
    continue_results = [r for r in carried if isinstance(r, Continue)]

    if raise_results:
        excs: list[BaseException] = [r.exception for r in raise_results]
        if isinstance(original, Raise):
            excs.append(original.exception)
        return Raise(exception=_combine_exceptions(excs, "Multiple errors occurred"))

    if return_results:
        # Values can't merge, so two distinct ones are an unresolvable conflict
        # (order-independence forbids a winner). Identical values coalesce.
        values = [r.return_value for r in return_results]
        first = values[0]
        if any(v != first for v in values[1:]):
            raise ReturnValueConflictError(
                f"{len(return_results)} results carrying distinct Return values ({values!r}) "
                "were composed at one REPL-exec boundary; return values cannot be merged, so at "
                "most one Return-carrying result may apply at a boundary."
            )
        return Return(return_value=first)

    if continue_results:
        # Concatenate outputs (original's first — only a Continue original has one — then each
        # carried Continue's) and group the non-None exceptions; a carried Continue may carry
        # exception=None, contributing only output (or, with empty output, nothing but a clean
        # "keep going").
        original_output = original.output if isinstance(original, Continue) else ""
        parts = [original_output] if original_output else []
        parts.extend(c.output for c in continue_results if c.output)
        excs = [c.exception for c in continue_results if c.exception is not None]
        if isinstance(original, Continue) and original.exception is not None:
            excs.append(original.exception)
        exception = (
            _combine_exceptions(excs, "Multiple errors occurred") if excs else None
        )
        # No summary of the combined exception is appended. Each contributing ``Continue`` owes
        # its own rendered exception in its own ``output`` (see ``ModifyResult``), so the joined
        # text already names every composed error; the only thing a summary would add is the
        # synthetic group *wrapper* ("Multiple errors occurred"), which is this fold's bookkeeping
        # rather than anything the agent needs. Appending it would restate every child a second
        # time to surface one invented line.
        return Continue(output="\n".join(parts), exception=exception)

    return None


def resolve_override_results(
    *,
    override_effects: list[OverrideResult],
    abort_errors: list[Exception],
) -> ExecResult | None:
    """Resolve ``OverrideResult`` (+ ``Abort``) at ``REPLExecEnter`` — the *supply* boundary.

    No execution has run yet, so there is no original result; the suppliers' carried results
    fold **among themselves** via ``_fold_carried_results`` (``original=None``) — the same
    precedence/merge the *transform* boundary uses. Two hooks vetoing the same input with
    recoverable ``Continue``s therefore merge (outputs concatenated, exceptions grouped) rather
    than conflicting — e.g. ``RestrictReturnValue`` co-installed with a ``ValidateREPLInput``.
    Distinct carried ``Return`` *values* still can't merge (``ReturnValueConflictError``): a
    supplied return names THE value, and order-independence forbids picking a winner. ``Abort``
    supersedes (termination trumps supply).

    (This folds rather than requiring suppliers to be *equal* — the earlier contract, which raised
    ``ResultConflictError`` on any two distinct supplies. Folding was adopted so that
    independent "veto this input" hooks compose the same way independent transforms do at exit,
    instead of a stack that installs two of them crashing on a hard internal error; see the
    ``resolve_modify_results`` origin-agnostic note for the mirror semantics.)

    Returns the folded/abort result, or ``None`` when nothing was emitted (run the code).
    """
    if abort_errors:
        return _abort_raise(abort_errors, None)
    return _fold_carried_results(
        [eff.result for eff in override_effects], original=None
    )


def resolve_modify_results(
    *,
    modify_effects: list[ModifyResult],
    abort_errors: list[Exception],
    original: ExecResult,
) -> ExecResult | None:
    """Resolve ``ModifyResult`` (+ ``Abort``) at ``REPLExecExit`` — the *transform* boundary.

    ``original`` is the actual ``exec_result``. Each ``ModifyResult`` carries a full
    replacement ``ExecResult``; multiple **fold** by carried-result kind, preserving the merge
    semantics the former ``ContinueEffect`` had. The result *kind* is decided by precedence
    ``Abort`` > carried ``Raise`` > carried ``Return`` > carried ``Continue`` (the highest
    present wins; lower kinds drop):

    - any ``Abort`` → ``Raise`` (see ``_abort_raise``);
    - else any carried ``Raise`` → ``Raise``, grouping every carried ``Raise``'s exception
      (plus ``original``'s if it too is a ``Raise``) into an ``ExceptionGroup``;
    - else any carried ``Return`` → ``Return`` — values can't merge, so two *distinct* carried
      return values raise ``ReturnValueConflictError`` (identical values coalesce);
    - else any carried ``Continue`` → ``Continue`` — the carried ``Continue``s' ``output``s
      are concatenated onto ``original``'s output *when ``original`` is itself a ``Continue``*
      (a terminal one has none to contribute, #903) and their non-``None`` exceptions grouped
      (plus ``original``'s if it too is a ``Continue``); a carried ``Continue`` with empty
      output contributes only its exception (the ``BudgetForcing`` shape);
    - else ``None`` (keep ``original``).

    Origin-agnostic, deliberately: the kind is decided by the effects present at *this*
    boundary, so a carried ``Continue`` composed onto an ``original`` terminal (``Return`` /
    ``Raise`` from an enter override) *downgrades* it to a recoverable ``Continue`` — the exit
    boundary's own effects are the last word (see ``span_repl_exec``).

    Returns the folded override, or ``None`` when nothing was emitted (keep ``original``).
    """
    if abort_errors:
        return _abort_raise(abort_errors, original)
    # Fold the carried replacements onto the executed result — the same precedence/merge the
    # supply boundary uses among its suppliers, but with a real ``original`` to fold onto.
    return _fold_carried_results([m.result for m in modify_effects], original=original)
