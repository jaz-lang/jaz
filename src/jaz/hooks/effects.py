"""Effect definitions for the hook system.

An effect is what a hook handler returns to influence the run — see :class:`jaz.hooks.Hook`.
Each class below documents the events it is valid at and how multiple instances of it compose;
an effect returned at an event that does not accept it raises
:class:`~jaz.exceptions.InvalidEffectError` (out-of-stage effects are hook bugs and fail
loudly, never silently no-op). The effects every hook
returns at an event are composed together before they are applied.

Where each effect composes, per pipeline stage (see :mod:`jaz.hooks.events` for the
``Enter → Send → Complete → Exit`` pipeline itself)::

    Enter    (edit the proposal)     InvokeEnter:   AddInputs, DropInputs, DisableRecursion
                                     LLMQueryEnter: AddMessages, DropMessages
                                     REPLExecEnter: AddVariables, DropVariables, InsertCode, DeleteCode
    Send     (supply the result,     LLMQuerySend:  SupplyLLMResponse
              skipping the work)     REPLExecSend:  SupplyExecResult
                                     InvokeSend:    SupplyInvokeResult
    Complete (transform the raw      REPLExecComplete:  ModifyExecResult
              result)                InvokeComplete:    ModifyInvokeResult
                                     LLMQueryComplete:  ModifyLLMResponse
    Exit     (observe the outcome)   LLMQueryExit / REPLExecExit: Abort (every arm)
                                     InvokeExit: (observation-only)

:class:`Abort` is additionally valid at every ``Enter``/``Send``/``Complete`` above, and — for
the per-turn spans only — at ``LLMQueryExit`` / ``REPLExecExit`` on **every** arm (to stop the run
on the *post-transform* result/response, the one the ``*Complete`` transform can't yet see; on the
``Aborted`` / ``Failed`` arms the abort is folded into the already-unwinding exception via an
``ExceptionGroup`` rather than replacing it). ``InvokeExit`` stays observation-only.
:class:`BlackboardWrite` is valid at every event including every ``Exit``.

**Experimental.** The hook system is an experimental feature; its interfaces may change
in a future release.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from jaz._llm_client import LLMResponse
    from jaz.tokens import TurnRecord

from jaz.exceptions import (
    CodeEditOffsetError,
    MessageEditIndexError,
    ReturnValueConflictError,
)
from jaz.llm import MessageDict
from jaz.repl.types import Continue, ExecResult, Raise, Return, TerminalExecResult

# The public effect surface (`jaz.hooks.effects`) — the typed values a hook handler
# returns to influence execution. `Effect` is the base for annotating handler returns.
__all__ = [
    "Effect",
    "SupplyExecResult",
    "ModifyExecResult",
    "ModifyInvokeResult",
    "SupplyInvokeResult",
    "Abort",
    "BlackboardWrite",
    "DisableRecursion",
    "AddInputs",
    "DropInputs",
    "AddVariables",
    "DropVariables",
    "InsertCode",
    "DeleteCode",
    "DropMessages",
    "AddMessages",
    "SupplyLLMResponse",
    "ModifyLLMResponse",
    "UNSET",
]


@dataclass(frozen=True)
class Effect:
    """Base class for all effects — the values a hook handler returns to influence the run.

    Each subclass documents the events it is valid at and how multiple instances of it
    compose. An effect returned at an event that does not accept it raises
    :class:`~jaz.exceptions.InvalidEffectError` — out-of-stage effects fail loudly rather
    than silently no-op.

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
#   ``SupplyExecResult(result)``  supplies a result in place of running the code — valid at
#     ``REPLExecSend`` (the committed code is in view; work is *pending*). It is the
#     supply-stage "supply instead" effect.
#   ``ModifyExecResult(result)``    transforms the result that ran — valid at
#     ``REPLExecComplete`` (where a raw result *exists*). It is the REPL transform-stage effect.
#   ``ModifyInvokeResult(result)``  the *invoke* transform-stage effect — valid at
#     ``InvokeComplete``, carrying only a ``TerminalExecResult`` (``Return`` | ``Raise``).
#   ``SupplyInvokeResult(result)``  supplies an invoke result in place of running the whole agent
#     loop — valid at ``InvokeSend``, carrying only a ``TerminalExecResult``. The invoke-boundary
#     twin of ``SupplyExecResult``; it completes the supply/modify grid (every span now has a
#     Send supplier and a Complete transform).
#
# ``ModifyExecResult`` and ``ModifyInvokeResult`` are a deliberate split of one former shared
# effect (executive call, see ``TerminalExecResult`` in ``jaz.repl.types``): the REPL boundary
# admits the full ``ExecResult`` (a hook downgrades a terminal to a ``Continue`` to force another
# turn), while the invoke boundary admits only a terminal one (the loop has already broken, so a
# ``Continue`` there is meaningless). The two carry different valid domains, so they are different
# effect types — but the *merge* is identical, so both resolve through the same
# ``_fold_carried_results`` (``resolve_modify_results`` / ``resolve_invoke_modify_results``). What
# is shared is the fold, not the wrapper. ``ModifyInvokeResult`` rejects a ``Continue`` at
# construction (``__post_init__``), moving what was a strippable late ``assert`` in ``span_invoke``
# to the emitting hook's own call site.
#
# All three **fold** by the same precedence (``_fold_carried_results``): carried ``Continue``s
# concatenate their outputs and group their exceptions; carried ``Return``s cannot merge (two
# distinct values conflict, identical ones coalesce); carried ``Raise``s group their
# exceptions. At the transform
# boundary the carried results fold onto the executed ``original``; at the supply boundary there
# is no original, so the suppliers fold among themselves. This preserves the merge semantics the
# former ``ContinueEffect`` had. The naming convention is uniform across the effect surface:
# ``Supply<X>`` supplies ``X`` and skips the work that would produce it (a *Send*-stage effect),
# while ``Modify<X>`` transforms the ``X`` that already exists (a *Complete*-stage effect) — the
# same supply-vs-modify split as ``SupplyLLMResponse`` at the LLM-query boundary.
#
# The two boundaries once diverged — supply required its results to be *equal* (a supply "names
# THE result", so two distinct supplies raised ``ResultConflictError``) while transform
# folded. They were unified to fold because independent enter-point vetoes (two hooks each
# rejecting the same code with a recoverable ``Continue``) should compose exactly as independent
# transforms do, not crash on a hard internal error; distinct ``Return`` *values* still conflict
# at both boundaries (they genuinely can't merge).
#
# Termination is a *separate* effect, ``Abort`` (below), NOT one of these: it
# un-bundles "stop the loop" from "supply/transform a result" (#481). ``Abort``
# raises its carried exception from the span context manager and is valid at **every
# control event** (not just the result boundaries), so loop/budget control has an
# always-present home and never rides a conditional exit. Where it co-occurs at a result
# boundary, ``Abort`` has the highest precedence structurally: the span CM raises before
# the supply/transform fold runs (termination trumps supply/transform), so the resolvers
# below only ever see genuine results.


@dataclass(frozen=True)
class SupplyExecResult(Effect):
    """Supply a result in place of running the turn's code.

    Valid at :class:`REPLExecSend` only — the committed code is in view and a result must
    not yet exist to be supplied. The code is not executed and the carried ``result``
    becomes the turn's outcome.

    Composition: multiple supplies fold. Kind precedence is :class:`Raise` > :class:`Return` >
    :class:`Continue` — when supplies of different kinds meet, the highest present wins and the
    lower ones drop. Within one kind: two :class:`Continue` results concatenate their outputs
    and group their exceptions; two distinct :class:`Return` values raise
    :class:`~jaz.exceptions.ReturnValueConflictError` (identical values coalesce); two
    :class:`Raise` results group their exceptions. A :class:`Abort` supersedes the whole fold.

    Because only the ``output`` field of :class:`Continue` is shown to the agent, a supplied
    :class:`Continue` should render any error into its ``output`` field, otherwise the agent
    does not see it — e.g. ``Continue(output=summarize_exception(exc), exception=exc)``.

    Attributes:
        result: The :class:`Continue` / :class:`Return` / :class:`Raise` to use instead of executing.
    """

    # This is the supply-boundary "supply instead" slot for sandbox/approval hooks generally.
    # ``ValidateREPLCode`` emits it: on rejected REPL code it supplies a ``Continue``
    # (recoverable), or a ``Raise`` at the failure cap. See ``resolve_supply_results``.
    #
    # The ``Continue`` merge is what lets two such hooks reject the SAME code without one of
    # them losing or the fold raising — their outputs concatenate and their exceptions group.
    # Kept out of the docstring: the rule ("two Continue results concatenate their outputs and
    # group their exceptions") already says it without the jargon.

    result: ExecResult


@dataclass(frozen=True)
class ModifyExecResult(Effect):
    """Replace a result that already exists with the carried one.

    Valid at :class:`REPLExecComplete` only — the REPL transform boundary, where a result
    exists to replace. The *invoke* transform boundary (:class:`InvokeComplete`) takes
    :class:`ModifyInvokeResult` instead, since an invoke completes only on a terminal result.

    Composition: multiple transforms fold by carried kind. Kind precedence is :class:`Raise` >
    :class:`Return` > :class:`Continue` — when transforms of different kinds meet, the highest
    present wins and the lower ones drop. Within one kind: a :class:`Continue` concatenates its
    ``output`` onto the original's and groups its exception; two distinct :class:`Return` values
    raise :class:`~jaz.exceptions.ReturnValueConflictError` (identical values coalesce); two
    :class:`Raise` results group theirs. A :class:`Abort` supersedes the fold.

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
class ModifyInvokeResult(Effect):
    """Replace an invoke's terminal result with the carried one.

    Valid at :class:`InvokeComplete` only — the invoke transform boundary. The carried
    ``result`` must be terminal (a :class:`Return` or a :class:`Raise`): the agent loop has
    already finished, so — unlike :class:`ModifyExecResult` at ``REPLExecComplete`` — there is no
    next turn a :class:`Continue` could resume, and one is rejected at construction.

    Composition: multiple transforms fold by carried kind, exactly as :class:`ModifyExecResult`
    does, but the fold can only ever yield a :class:`Return` or
    :class:`Raise` here because no :class:`Continue` can be carried. Kind precedence is
    :class:`Raise` > :class:`Return`; two distinct :class:`Return` values raise
    :class:`~jaz.exceptions.ReturnValueConflictError` (identical values coalesce). A :class:`Abort`
    supersedes the fold.

    Attributes:
        result: The :class:`Return` / :class:`Raise` to replace the invoke's terminal result with.

    Raises:
        TypeError: At construction, if ``result`` is a :class:`Continue`.
    """

    # The fold is shared with ``ModifyExecResult`` via ``_fold_carried_results``.
    #
    # This is the split half of the former single ``ModifyExecResult`` that was valid at both
    # ``*Complete`` boundaries. See the module-level "Exec-result effects" comment and
    # ``TerminalExecResult`` (``jaz.repl.types``) for the executive call: the invoke boundary's
    # valid domain is genuinely narrower than the REPL boundary's, so it gets its own effect whose
    # *field type* states the invariant and whose ``__post_init__`` enforces it — instead of the
    # old strippable ``assert`` in ``span_invoke``, which failed late (at span close, not at the
    # offending hook) and vanished under ``python -O``. Only ``ReturnType`` / ``ValidateReturn``
    # emit it (their terminal ``Raise`` backstop); both already carry a ``Raise``, so the guard
    # never fires for them — it exists for a future hook that mistakes this for ``ModifyExecResult``.

    result: TerminalExecResult

    def __post_init__(self) -> None:
        if isinstance(self.result, Continue):
            raise TypeError(
                "ModifyInvokeResult carries a terminal result (Return | Raise); got a Continue. "
                "An invoke's transform boundary runs after the agent loop has broken, so there is "
                "no next turn a Continue could resume — downgrade-to-Continue is a REPLExecComplete "
                "operation (ModifyExecResult). Emit a Raise to end the invoke with an error."
            )


@dataclass(frozen=True)
class SupplyInvokeResult(Effect):
    """Supply an invoke's terminal result, skipping the entire agent loop.

    Valid at :class:`InvokeSend` only — the supplier sees the invoke's committed inputs and no
    work has run yet. The agent loop (its REPL turns and LLM calls) is not executed and the
    carried ``result`` becomes the invoke's outcome. The invoke transform boundary still fires,
    so a :class:`ModifyInvokeResult` — and the ``ReturnType`` / ``ValidateReturn`` backstops —
    still apply to the supplied result (mirroring how a :class:`SupplyExecResult` still flows
    through :class:`REPLExecComplete`).

    The carried ``result`` must be terminal (a :class:`Return` or a :class:`Raise`): an invoke
    never completes on a :class:`Continue`, so there is no :class:`Continue` to supply in place of
    running it, and one is rejected at construction (as with :class:`ModifyInvokeResult`).

    Composition: multiple supplies fold among themselves, exactly as :class:`SupplyExecResult`
    does at :class:`REPLExecSend` but on terminal results only — a :class:`Raise` supersedes a
    :class:`Return`; two distinct :class:`Return` values raise
    :class:`~jaz.exceptions.ReturnValueConflictError` (identical values coalesce). A :class:`Abort`
    supersedes the whole fold.

    Attributes:
        result: The :class:`Return` / :class:`Raise` to use instead of running the invoke.

    Raises:
        TypeError: At construction, if ``result`` is a :class:`Continue`.
    """

    # The invoke-Send supplier — the invoke-boundary twin of ``SupplyExecResult`` (``REPLExecSend``),
    # which is what fills the last open cell of the supply/modify grid for the invoke span (the
    # ``InvokeSend`` boundary previously had no supplier). It folds like ``SupplyExecResult``
    # (``resolve_invoke_supply_results`` → ``_fold_carried_results`` with ``original=None``) but on
    # the terminal domain, and rejects a ``Continue`` at construction like ``ModifyInvokeResult`` —
    # the same ``TerminalExecResult`` invariant, stated in the field type and enforced here.

    result: TerminalExecResult

    def __post_init__(self) -> None:
        if isinstance(self.result, Continue):
            raise TypeError(
                "SupplyInvokeResult carries a terminal result (Return | Raise); got a Continue. "
                "An invoke completes only on a terminal result, so there is no Continue to supply "
                "in place of running it. Supply a Return or a Raise."
            )


@dataclass(frozen=True)
class Abort(Effect):
    """Abort **this invoke**: ``error`` propagates out of ``jaz.invoke()`` as a raised exception.

    Valid at every control event — the ``*Enter``, ``*Send``, and ``*Complete`` events of all
    three spans — and, for the per-turn spans, at ``LLMQueryExit`` / ``REPLExecExit`` on **every**
    arm (``Completed`` is the post-transform, pre-act point; on the ``Aborted`` / ``Failed`` arms an
    exception is already unwinding, so the abort is folded *into* it via an ``ExceptionGroup`` rather
    than replacing it).
    ``InvokeExit`` stays observation-only.

    At :class:`LLMQueryComplete` the query has already completed and been paid for; the
    abort stops the turn before the agent acts on the response, so the code the model
    proposed never runs.

    ``Abort`` is a *mechanism*, and its scope is exactly the invoke it fires in: the
    invoke ends by raising the carried exception, and the aborting invoke's spans close with
    an ``Aborted`` outcome on their ``*Exit`` events. How far the stop travels is decided
    by the carried exception's *category*, not by this effect: a parent invoke observes an
    aborted child as an ordinary recoverable error in its REPL (a raising call it can catch
    and proceed past) — UNLESS the carried exception is
    :class:`~jaz.exceptions.FatalError`-category, which no agent contains: the framework
    re-raises it past every agent in the tree, and only human-authored code (the top-level
    caller included) catches it normally. The containment boundary is authorship, not tree
    depth.

    Composition: it has the **highest precedence** wherever it co-occurs with the result
    effects, superseding both :class:`SupplyExecResult` and :class:`ModifyExecResult`. Multiple
    aborts group their exceptions into an ``ExceptionGroup``.

    Attributes:
        error: The exception to abort the invoke with.
    """

    # Naming (span_event_lifecycle.md, revised by user decision): the effect KEEPS ``Abort``;
    # the shared-stem conflation with the escalation category was resolved by renaming the
    # *exception* to :class:`~jaz.exceptions.FatalError`, not by renaming the effect. DOM
    # ``AbortController.abort(reason)`` is this effect's exact precedent — an operation-scoped
    # stop with a reason payload, containable by the parent — so the mechanism owns the word
    # "abort"; "fatal" is what the codebase's own ``is_fatal`` predicate already called the
    # category, making the error's new name tautological. (The rejected alternative — an
    # invoke-scoped rename of this effect — moved the wrong name: it is the *category* whose
    # old abort-stem name baked in a claim it couldn't keep; see the rename comment under
    # ``FatalError``.) No back-compat aliases either way — pre-release.
    #
    # Loop and budget hard-stops emit it at the always-present ``LLMQueryEnter`` (once per turn,
    # before the LLM call), so a stop is never tied to the *conditional* ``REPLExecComplete``,
    # which is skipped on a parse-failure turn that has no execution.
    #
    # Termination is a first-class, self-scoping effect rather than a mode of the result effects
    # (#481): un-bundling "stop the loop" from "supply/transform a result" is what lets it be
    # valid at events that have no result at all — it references no existing result, which is
    # why it is accepted at every control event.
    #
    # It is NOT the only way to stop the loop: a ``SupplyExecResult`` / ``ModifyExecResult``
    # carrying a ``Return`` or a ``Raise`` ends the invoke too. What is unique to
    # ``Abort`` is that it can stop from events that have no result to carry one
    # (``InvokeEnter``, ``LLMQueryEnter``, ``LLMQueryComplete``).
    #
    # ``LLMQueryRetry`` is the remaining hold-out. It is split into its own change rather than
    # bundled here because it raises a distinct question — aborting mid-retry pre-empts a
    # retry policy the LLM client owns.

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
    # affordance removal: the primitive honors it by binding the REPL with ``invoke_tool=None``.
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


# Invoke-input effects


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
    ``len``, or a framework magic such as ``__history__`` — does not count as already
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
    # this writes them. A hook typically binds once (emit on ``event.iteration == 0``, the first turn).
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
    # ``__history__`` binds, while adding an existing variable raises. The docstring says
    # "shadows" rather than "is not checked" because the outcome, not the mechanism, is what a
    # hook author is deciding about.

    variables: dict[str, object]


@dataclass(frozen=True)
class DropVariables(Effect):
    """Remove names from the agent's REPL namespace before a turn's code runs.

    Valid at :class:`REPLExecEnter` only. Each listed name is deleted before the turn's code
    executes, so referencing it raises an ordinary ``NameError``. It reaches any name bound in
    the namespace, whatever put it there — an invoke input, a variable an earlier turn assigned,
    or a framework magic such as ``__history__``.

    The prompt is untouched. If the dropped name happened to come from an invoke input, that
    input still renders in the user prompt; un-passing an input is :class:`DropInputs`, which
    acts on a different thing — the invoke's inputs, fixed once at :class:`InvokeEnter`.

    Composition: every drop's ``names`` are combined into one set and applied once, so the
    outcome does not depend on hook order. Two hooks naming the same bound name therefore
    produce a single deletion.

    Dropping a name that is not bound raises :class:`~jaz.exceptions.MissingDropTargetError`.
    Pass ``allow_missing=True`` to ignore names that are not bound rather than raising;
    if an unbound name is dropped by both a ``DropVariables(..., allow_missing=True)`` and a
    ``DropVariables(..., allow_missing=False)``,
    then the drop is ignored and will not raise — one tolerant drop exempts the name for
    every hook naming it. Naming ``__builtins__`` always raises
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
        DropVariables({"__history__"})
        # Defensively unbind an input that another effect may already have un-passed.
        DropVariables({"maybe_unpassed"}, allow_missing=True)
    """

    # The docstring used to say this "unbinds the REPL name only — it does not un-pass an invoke
    # input", pointing at ``DropInputs`` "to remove it everywhere". That framed the two as one
    # operation at two strengths, which they are not: this acts on the REPL namespace (any name,
    # whatever bound it — an input, an agent assignment, ``__history__``), while
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


@dataclass(frozen=True)
class InsertCode(Effect):
    """Insert source into the proposed REPL code, before it runs.

    Valid at :class:`REPLExecEnter` only — the "edit the proposal" stage, alongside
    :class:`AddVariables` / :class:`DropVariables` (which edit the namespace, not the code). The
    LLM-message twin is :class:`AddMessages`.

    The offset is a character position in the ORIGINAL proposed code (:class:`REPLExecEnter`'s
    ``code``), so composition is order-independent — every effect names a position in the code the
    hook saw, not in the post-edit string. Two inserts at the same offset both apply, ordered by
    ``sort_key``; an insert whose offset falls inside a :class:`DeleteCode` range still inserts
    (new content survives a delete — this is how a hook *replaces* a span: delete it and insert at
    the same offset).

    A code edit is observable through the events — :class:`REPLExecEnter` carries the original
    proposal and :class:`REPLExecSend` the committed (edited) code — but the agent's ``__history__``
    records the LLM's raw response, not the parsed/edited code, so the edit is not reflected there.

    Attributes:
        text: The source to insert.
        at: Character offset to insert *before*, in ``[0, len]`` of the original code; a negative
            offset counts from the end (``-1`` = before the last character), matching list
            semantics. An out-of-range offset raises :class:`~jaz.exceptions.CodeEditOffsetError`.
        sort_key: Orders inserts that share an offset (lower first; default ``0.0``), ties broken
            by the inserted text — so same-offset inserts are deterministic without depending on
            hook order. Mirrors :class:`AddMessages`.
    """

    text: str
    at: int
    sort_key: float = 0.0


@dataclass(frozen=True)
class DeleteCode(Effect):
    """Delete a character range from the proposed REPL code, before it runs.

    Valid at :class:`REPLExecEnter` only — the "edit the proposal" stage. The LLM-message twin is
    :class:`DropMessages`.

    The range ``[start, end)`` is resolved against the ORIGINAL proposed code's offsets, so
    composition is order-independent. Overlapping ranges from multiple hooks **union**: each
    resolves to the set of character positions it covers, and the sets union exactly as
    :class:`DropMessages` unions its indices — so no two deletes ever conflict.

    Attributes:
        start: First offset to delete (inclusive). Negative counts from the end.
        end: Offset one past the last character to delete (exclusive). Negative counts from the
            end; ``start == end`` deletes nothing. Both offsets lie in ``[0, len]`` of the original
            code and ``start <= end`` after resolution, else
            :class:`~jaz.exceptions.CodeEditOffsetError` is raised.
    """

    start: int
    end: int


def _code_insert_sort_key(ins: InsertCode) -> tuple[float, str]:
    """Within-offset ordering for an ``InsertCode``: ``sort_key`` first, then the inserted text —
    a total order that keeps same-offset inserts deterministic without depending on hook
    accumulation order (the character-offset twin of :func:`_add_sort_key`)."""
    return (ins.sort_key, ins.text)


def _resolve_code_offset(offset: int, n: int, *, kind: str) -> int:
    """Resolve a possibly-negative code-edit character offset against code length ``n``.

    List / NumPy semantics, mirroring :func:`_resolve_edit_index`: a negative offset counts from
    the end (``offset + n``). Both an insert offset and a delete bound lie in ``[0, n]`` (``n`` =
    end of string), else :class:`~jaz.exceptions.CodeEditOffsetError` is raised — an out-of-range
    offset is a hook bug (it was computed against the code the hook saw), and normalizing here,
    before the fold, keeps composition commutative.

    The upper bound is ``n``, not ``n - 1`` as for a ``DropMessages`` *position*: every offset here
    names a boundary *between* characters (an insert slot, or a half-open delete bound), not a
    character index, and there are ``n + 1`` such boundaries — including ``n``, the end of the
    string (append / delete-to-end). ``DropMessages`` drops a message *at* an index, so its cap is
    ``n - 1``; the append-slot cap ``n`` there is the ``AddMessages`` analogue of this one.
    """
    resolved = offset + n if offset < 0 else offset
    if not (0 <= resolved <= n):
        raise CodeEditOffsetError(
            f"{kind} offset {offset} is out of range for {n}-character code"
        )
    return resolved


def apply_code_edits(
    code: str,
    inserts: list[InsertCode],
    deletes: list[DeleteCode],
) -> str:
    """Fold code insert/delete edits over ``code``, anchored to its original offsets.

    **Pure** — returns a new string, never mutates ``code``. The character-offset twin of
    :func:`apply_message_edits`: every insert offset and delete range is resolved against the
    ORIGINAL code (the single-coordinate-system rule that keeps edits commutative), then a single
    left-to-right pass emits, at each offset, any inserts there (ordered by
    :func:`_code_insert_sort_key`) followed by the original character unless it falls in a deleted
    range. Overlapping deletes union; an insert at a deleted offset still inserts (new content
    survives a delete), which is how a hook replaces a span.

    Offsets follow list semantics (negatives count from the end); an out-of-range offset — or a
    delete whose resolved ``start`` exceeds its ``end`` — raises
    :class:`~jaz.exceptions.CodeEditOffsetError` rather than being ignored or clamped, the same
    fail-loud contract as ``apply_message_edits`` (an out-of-range offset is a hook bug, and the
    raise aborts the invoke).
    """
    n = len(code)
    deleted: set[int] = set()
    for d in deletes:
        start = _resolve_code_offset(d.start, n, kind="DeleteCode start")
        end = _resolve_code_offset(d.end, n, kind="DeleteCode end")
        if start > end:
            raise CodeEditOffsetError(
                f"DeleteCode start {d.start} resolves after end {d.end} "
                f"for {n}-character code"
            )
        deleted.update(range(start, end))

    inserts_by_offset: dict[int, list[InsertCode]] = {}
    for ins in inserts:
        at = _resolve_code_offset(ins.at, n, kind="InsertCode at")
        inserts_by_offset.setdefault(at, []).append(ins)
    for group in inserts_by_offset.values():
        group.sort(key=_code_insert_sort_key)

    parts: list[str] = []
    for i in range(n + 1):
        for ins in inserts_by_offset.get(i, []):
            parts.append(ins.text)
        if i < n and i not in deleted:
            parts.append(code[i])
    return "".join(parts)


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
            ``True`` makes the drop survive into later turns.
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
class SupplyLLMResponse(Effect):
    """Supply a pre-computed LLM response, skipping the API call.

    Valid at :class:`LLMQuerySend` only — the supplier sees the committed post-edit message
    list. The content, token counts, and cost are used exactly as if they came from a live
    call, including cost tracking and budget enforcement — the query completes normally, so
    :class:`LLMQueryComplete` and :class:`LLMQueryExit` still fire.

    Composition: identical overrides compose to one; two *distinct* overrides raise
    :class:`~jaz.exceptions.LLMResponseConflictError`.

    The fields mirror :class:`~jaz.llm.LLMResponse` one-to-one — the supplier constructs a
    whole response, so each field is a direct value (``None`` means the constructed response carries
    ``None`` there, *not* "keep", since a supply has no original). ``extra`` defaults to an empty
    dict.

    Attributes:
        content: The response text, or ``None``.
        prompt_tokens: Input token count to record, if known.
        completion_tokens: Output token count to record, if known.
        cached_tokens: Cached-input token count (subset of ``prompt_tokens``), if known.
        cost_usd: Cost in US dollars to record for this call, if known.
        extra: Provider-specific metrics (e.g. ``reasoning_tokens``); ``None`` ⇒ empty dict.
        raw_response: The raw provider payload a protocol may read (tool calls / finish reason).
        tokens: Per-turn token record for a token-native backend, if any.
    """

    # Mirrors `LLMResponse` one-to-one, so it tracks that type's ATIF v1.7 alignment: `cost` became
    # `cost_usd` and `total_tokens` was dropped (see the comment on `LLMResponse`) — sum
    # `prompt_tokens` + `completion_tokens`. Full coverage (not just the metrics) so a supplied
    # response can carry a `raw_response` a tool-calling protocol reads and the `extra` / `tokens`
    # a live call would have — without it a Replay/mock could not faithfully stand in for the real
    # response. Keeping a second spelling of one concept here is what let jaz's old client/provider
    # split drift apart in the first place.

    content: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    extra: dict[str, Any] | None = None
    raw_response: Any = None
    tokens: TurnRecord | None = None


class _Unset(enum.Enum):
    """The ``ModifyLLMResponse`` "field untouched" sentinel — a keep-marker distinct from ``None``.

    A single-member enum (rather than a bare ``object()``) so it has a real type for annotations
    (``T | None | _Unset``) and narrows under an ``is`` check, and so its ``repr`` is stable.
    """

    UNSET = enum.auto()


#: Sentinel marking a :class:`ModifyLLMResponse` field as "leave the original untouched" — used
#: because ``None`` is a settable value there (see the effect's docstring). Rarely written
#: explicitly (omit the field to get it as the default); exposed so a hook that computes overrides
#: conditionally can assign ``UNSET`` to mean "don't touch this one".
UNSET = _Unset.UNSET


@dataclass(frozen=True)
class ModifyLLMResponse(Effect):
    """Transform the LLM response after the call returned, before the agent acts on it.

    Valid at :class:`LLMQueryComplete` only — the response-transform boundary, the LLM-query twin
    of :class:`ModifyExecResult` at :class:`REPLExecComplete`. The transform is **authoritative**:
    the modified response is what the agent parses into its next REPL turn *and* what
    :class:`LLMQueryExit` observers and cost accounting (``BudgetPool``) see — so a content rewrite
    changes the agent's behavior, not just the record.

    A **partial override** over :class:`~jaz.llm.LLMResponse`'s full field set: each field
    defaults to :data:`UNSET` meaning "leave the original's value untouched"; any *other* value —
    **including ``None``** — replaces that field. The :data:`UNSET` sentinel (not ``None``) marks
    "keep" precisely because ``None`` is a legitimate value for ``content`` / ``raw_response`` /
    ``tokens`` (a refusal has ``content is None``), so this effect *can* set them to ``None``. A
    field left :data:`UNSET` is preserved from the original via ``dataclasses.replace`` — so a
    content rewrite keeps the real provider payload, tokens, and cost intact, and a re-price keeps
    the content.

    Composition: **field-wise merge**. Multiple effects compose by taking, for each field,
    whichever value is not :data:`UNSET`; two effects setting the *same* field to *different*
    values raise
    :class:`~jaz.exceptions.LLMResponseConflictError` (identical values coalesce). This is
    order-independent — a content-redacting hook and a re-pricing hook compose cleanly because they
    touch disjoint fields. A :class:`Abort` supersedes the whole merge.

    Attributes:
        content: Replacement response text (:data:`UNSET` = keep; ``None`` sets a null completion).
        prompt_tokens: Replacement input token count (:data:`UNSET` = keep).
        completion_tokens: Replacement output token count (:data:`UNSET` = keep).
        cached_tokens: Replacement cached-input token count (:data:`UNSET` = keep).
        cost_usd: Replacement cost in US dollars (:data:`UNSET` = keep). Note: this re-prices the
            cost *booked* at ``LLMQueryExit``, but it cannot rescue an uncosted-model turn — the
            ``BudgetPool`` backstop aborts on the *raw* ``cost_usd is None`` at ``LLMQueryComplete``
            before this override resolves (abort supersedes the transform).
        extra: Replacement provider-metrics dict (:data:`UNSET` = keep; replaces wholesale).
        raw_response: Replacement raw provider payload (:data:`UNSET` = keep; ``None`` clears it).
        tokens: Replacement per-turn token record (:data:`UNSET` = keep; ``None`` clears it).
    """

    # The LLM-query Complete-stage transform — the cell the effect grid left empty (the slot was
    # "documented-but-empty by design" on ``LLMQueryComplete`` until a consumer motivated it). It
    # mirrors ``SupplyLLMResponse``'s full field set rather than carrying a whole ``LLMResponse``,
    # so a content-only rewrite can't silently zero the tokens / cost / raw payload — the resolver
    # folds only the set fields onto the original via ``dataclasses.replace``, preserving the rest.
    # The merge is field-wise (not the ``ExecResult`` precedence fold) because an ``LLMResponse``
    # has no kind precedence; conflict-on-same-field keeps it order-independent.
    #
    # ``UNSET`` sentinel (not ``None``) for "keep" (executive call, this conversation): ``None`` is
    # a settable value here — a response legitimately carries ``content`` / ``raw_response`` /
    # ``tokens`` of ``None`` — so overloading it as the keep-marker (as ``SupplyLLMResponse`` can,
    # since a supply has no original to keep) would make those un-settable. The sentinel costs a
    # ``| _Unset`` on each annotation but buys full, unambiguous override coverage.
    #
    # Authoritative feedback (executive call, this conversation): the transformed response is routed
    # back into what the agent parses (``_query_with_messages`` re-reads ``span.get_response()``
    # after the span closes), not merely shown to Exit observers — otherwise "modify the response"
    # would not change behavior. Cost booked at ``LLMQueryExit`` reflects the override; the
    # same-boundary uncosted-model *abort* backstop (``BudgetPool.on_llm_query_complete``) reads the
    # raw pre-transform response deliberately — it guards against a provider reporting no cost, not
    # against what a hook re-prices.

    content: str | None | _Unset = UNSET
    prompt_tokens: int | None | _Unset = UNSET
    completion_tokens: int | None | _Unset = UNSET
    cached_tokens: int | None | _Unset = UNSET
    cost_usd: float | None | _Unset = UNSET
    extra: dict[str, Any] | _Unset = UNSET
    raw_response: Any = UNSET
    tokens: TurnRecord | None | _Unset = UNSET


# Composition helpers for Abort / SupplyExecResult / ModifyExecResult


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


def _fold_carried_results(
    carried: list[ExecResult], *, original: ExecResult | None
) -> ExecResult | None:
    """Precedence fold shared by both REPL-exec result boundaries.

    ``carried`` are the full ``ExecResult``s named by this boundary's effects (each
    ``SupplyExecResult`` / ``ModifyExecResult`` ``.result``). ``original`` is the executed result at the
    *transform* boundary (``REPLExecComplete``), or ``None`` at the *supply* boundary
    (``REPLExecSend``, where nothing has run yet). The result *kind* is decided by precedence
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
        # its own rendered exception in its own ``output`` (see ``ModifyExecResult``), so the joined
        # text already names every composed error; the only thing a summary would add is the
        # synthetic group *wrapper* ("Multiple errors occurred"), which is this fold's bookkeeping
        # rather than anything the agent needs. Appending it would restate every child a second
        # time to surface one invented line.
        return Continue(output="\n".join(parts), exception=exception)

    return None


def resolve_supply_results(
    *,
    supply_effects: list[SupplyExecResult],
) -> ExecResult | None:
    """Resolve ``SupplyExecResult`` at the *supply* boundary (``REPLExecSend``).

    No execution has run yet, so there is no original result; the suppliers' carried results
    fold **among themselves** via ``_fold_carried_results`` (``original=None``) — the same
    precedence/merge the *transform* boundary uses. Two hooks vetoing the same code with
    recoverable ``Continue``s therefore merge (outputs concatenated, exceptions grouped) rather
    than conflicting — e.g. the evals ``RestrictReturnValue`` co-installed with a ``ValidateREPLCode``.
    Distinct carried ``Return`` *values* still can't merge (``ReturnValueConflictError``): a
    supplied return names THE value, and order-independence forbids picking a winner.

    An ``Abort`` composed at the same boundary never reaches this fold: the span context
    manager raises its carried exception first (termination trumps supply — structurally,
    not by a precedence rule inside the fold; the former ``abort_errors`` parameter was the
    legacy laundering channel that turned an abort into a terminal ``Raise`` result).

    (This folds rather than requiring suppliers to be *equal* — the earlier contract, which raised
    ``ResultConflictError`` on any two distinct supplies. Folding was adopted so that
    independent "veto this input" hooks compose the same way independent transforms do at exit,
    instead of a stack that installs two of them crashing on a hard internal error; see the
    ``resolve_modify_results`` origin-agnostic note for the mirror semantics.)

    Returns the folded result, or ``None`` when nothing was emitted (run the code).
    """
    return _fold_carried_results([eff.result for eff in supply_effects], original=None)


def resolve_modify_results(
    *,
    modify_effects: list[ModifyExecResult],
    original: ExecResult,
) -> ExecResult | None:
    """Resolve ``ModifyExecResult`` at the REPL *transform* boundary (``REPLExecComplete``).

    ``original`` is the actual ``exec_result``. Each ``ModifyExecResult`` carries a full
    replacement ``ExecResult``; multiple **fold** by carried-result kind, preserving the merge
    semantics the former ``ContinueEffect`` had. The result *kind* is decided by precedence
    carried ``Raise`` > carried ``Return`` > carried ``Continue`` (the highest
    present wins; lower kinds drop):

    - any carried ``Raise`` → ``Raise``, grouping every carried ``Raise``'s exception
      (plus ``original``'s if it too is a ``Raise``) into an ``ExceptionGroup``;
    - else any carried ``Return`` → ``Return`` — values can't merge, so two *distinct* carried
      return values raise ``ReturnValueConflictError`` (identical values coalesce);
    - else any carried ``Continue`` → ``Continue`` — the carried ``Continue``s' ``output``s
      are concatenated onto ``original``'s output *when ``original`` is itself a ``Continue``*
      (a terminal one has none to contribute, #903) and their non-``None`` exceptions grouped
      (plus ``original``'s if it too is a ``Continue``); a carried ``Continue`` with empty
      output contributes only its exception (the ``BudgetForcing`` shape);
    - else ``None`` (keep ``original``).

    An ``Abort`` at the same boundary supersedes the whole fold structurally: the span
    context manager raises its carried exception before calling this, so an abort can never
    be downgraded by a co-composed transform (the Defect-3 revocation hole of
    span_event_lifecycle.md, closed by construction rather than by a precedence rule here).

    Origin-agnostic, deliberately: the kind is decided by the effects present at *this*
    boundary, so a carried ``Continue`` composed onto an ``original`` terminal (``Return`` /
    ``Raise`` from a Send-time supply) *downgrades* it to a recoverable ``Continue`` — the
    transform boundary's own effects are the last word (see ``span_repl_exec``).

    Returns the folded override, or ``None`` when nothing was emitted (keep ``original``).
    """
    # Fold the carried replacements onto the executed result — the same precedence/merge the
    # supply boundary uses among its suppliers, but with a real ``original`` to fold onto.
    return _fold_carried_results([m.result for m in modify_effects], original=original)


def resolve_invoke_modify_results(
    *,
    modify_effects: list[ModifyInvokeResult],
    original: TerminalExecResult,
) -> TerminalExecResult | None:
    """Resolve ``ModifyInvokeResult`` at the invoke transform boundary (``InvokeComplete``).

    The invoke-boundary twin of ``resolve_modify_results``: it delegates to the *same*
    ``_fold_carried_results``, so the merge semantics are identical. The difference is entirely in
    the domain — ``original`` and every carried result are terminal (``Return`` | ``Raise``), so
    the fold's ``Continue`` branch is unreachable and the override is itself terminal. That is the
    guarantee the shared ``resolve_modify_results`` could *not* make: it accepted a carried
    ``Continue`` and would downgrade the terminal result to it, which at an invoke's completion is
    a leaked non-terminal outcome (previously caught only by a strippable ``assert`` in
    ``span_invoke``). Here it cannot arise, because ``ModifyInvokeResult`` rejects a ``Continue`` at
    construction.

    Returns the folded terminal override, or ``None`` when nothing was emitted (keep ``original``).
    """
    folded = _fold_carried_results(
        [m.result for m in modify_effects], original=original
    )
    # Terminal-in ⇒ terminal-out: no ``Continue`` is carried (the effect forbids it) and
    # ``original`` is terminal, so ``_fold_carried_results`` took its ``Raise`` or ``Return``
    # branch, never ``Continue``. The generic fold is typed ``ExecResult | None``; narrow it here
    # since the branch that would produce a ``Continue`` is unreachable on this path.
    return cast("TerminalExecResult | None", folded)


def resolve_invoke_supply_results(
    *,
    supply_effects: list[SupplyInvokeResult],
) -> TerminalExecResult | None:
    """Resolve ``SupplyInvokeResult`` at the invoke *supply* boundary (``InvokeSend``).

    The invoke twin of ``resolve_supply_results``: no work has run, so the suppliers' carried
    terminal results fold **among themselves** via ``_fold_carried_results`` (``original=None``),
    the same precedence/merge the REPL supply boundary uses — restricted to terminal results,
    since ``SupplyInvokeResult`` cannot carry a ``Continue``. Distinct carried ``Return`` values
    still can't merge (``ReturnValueConflictError``): a supplied return names THE value.

    Returns the folded terminal result, or ``None`` when nothing was supplied (run the invoke).
    """
    folded = _fold_carried_results(
        [eff.result for eff in supply_effects], original=None
    )
    # Terminal-in ⇒ terminal-out, as in ``resolve_invoke_modify_results`` (here there is no
    # ``original`` either, so the ``Continue`` branch is doubly unreachable). Narrow the generic
    # ``ExecResult | None`` accordingly.
    return cast("TerminalExecResult | None", folded)


# The fields ``ModifyLLMResponse`` may override; everything else on an ``LLMResponse``
# (``raw_response``, ``extra``, ``cached_tokens``, ``tokens``) is always preserved from the
# original. Kept as a tuple so the resolver and any test iterate the same source of truth.
_LLM_MODIFY_FIELDS = (
    "content",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cost_usd",
    "extra",
    "raw_response",
    "tokens",
)


def resolve_llm_modify_results(
    *,
    modify_effects: list[ModifyLLMResponse],
    original: LLMResponse,
) -> LLMResponse | None:
    """Resolve ``ModifyLLMResponse`` at the response-transform boundary (``LLMQueryComplete``).

    The LLM-query twin of ``resolve_modify_results``, but the merge is **field-wise** rather than
    the ``ExecResult`` precedence fold (an ``LLMResponse`` has no kind precedence): for each of the
    response's fields (see ``_LLM_MODIFY_FIELDS``) it takes the single non-``UNSET`` value the
    effects carry, and two effects setting the *same* field to *different* values raise
    ``ReturnValueConflictError``\\ 's LLM sibling ``LLMResponseConflictError`` (identical values
    coalesce). Order-independent, so a content-redactor and a re-pricer compose. ``UNSET`` (not
    ``None``) is the "keep" marker, so an override *to* ``None`` is honored; a field left ``UNSET``
    by every effect is preserved from ``original`` by ``dataclasses.replace``.

    Returns a new ``LLMResponse`` when any field was overridden, or ``None`` when nothing was
    (keep ``original`` — the identity the agent's post-close ``is``-check relies on to leave the
    non-transform path untouched).
    """
    from jaz.exceptions import LLMResponseConflictError

    overrides: dict[str, object] = {}
    for field_name in _LLM_MODIFY_FIELDS:
        # The distinct values any effect actually sets for this field (UNSET = untouched).
        values = [
            v for eff in modify_effects if (v := getattr(eff, field_name)) is not UNSET
        ]
        if not values:
            continue
        first = values[0]
        if any(v != first for v in values[1:]):
            raise LLMResponseConflictError(
                f"Multiple ModifyLLMResponse effects set '{field_name}' to distinct values "
                f"({values!r}) at one LLMQueryComplete boundary; field-wise merge is "
                "order-independent, so a field's value cannot be resolved without relying on hook "
                "registration / with-nesting order. Have at most one hook set a given field."
            )
        overrides[field_name] = first

    if not overrides:
        return None
    return replace(original, **overrides)
