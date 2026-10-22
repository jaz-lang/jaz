"""Event definitions for the hook system.

Every event has a matching handler on :class:`jaz.hooks.Hook`, named after it
(:class:`LLMQueryEnter` → ``on_llm_query_enter``). A handler observes the event and returns a
list of effects; returning effects is the only way a hook influences the run. Which effects
are accepted depends on the event — each event class documents its own allowed set, and an
effect returned at an event outside that set raises
:class:`~jaz.exceptions.InvalidEffectError` (out-of-stage effects are hook bugs and fail
loudly, never silently no-op). A hook that treats every event alike overrides ``on_any``
instead::

    from jaz.hooks import Hook
    from jaz.hooks.effects import Abort, Effect
    from jaz.hooks.events import LLMQueryEnter

    class StopAfterTenTurns(Hook):
        def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
            if event.iteration >= 10:  # iterations are 0-based
                return [Abort(error=RuntimeError("too many turns"))]
            return []

Each span's events follow one pipeline of value states::

    proposal ──[edits]──▸ committed input ──[work or supply]──▸ raw result ──[modify]──▸ outcome

    *Enter     observes the proposal          controls edits + abort     fires always
    *Send      observes the committed input   controls supply + abort    fires iff the input committed
    *Complete  observes the raw result        controls modify + abort    fires iff a raw result exists
    *Exit      observes the outcome union     controls abort (per-turn)  fires whenever the span opened

Events mark **value states becoming fixed**, not phases of the machinery: an event's
payload is the commit of the previous stage's effects, so no event can observe its own
composition's output — only the next event can carry it. ``*Send`` therefore fires on the
supplied/replayed path too (the input commit is real even when no provider is called), and
``*Exit`` carries a tagged outcome union (``Completed[...] | Aborted | Failed`` — the variant
IS the span status, its payload the **post-transform** result or the terminating exception),
so an interval that opened always closes, with the true outcome. A
:class:`~jaz.hooks.effects.BlackboardWrite` is valid at every event, these included; beyond it
the per-turn ``LLMQueryExit`` / ``REPLExecExit`` accept an :class:`~jaz.hooks.effects.Abort` on
every arm, and ``InvokeExit`` accepts nothing.

The events, by span:

- :class:`InvokeEnter` / :class:`InvokeSend` / :class:`InvokeComplete` / :class:`InvokeExit`
  — one quadruple per invoke.
- :class:`LLMQueryEnter` / :class:`LLMQuerySend` / :class:`LLMQueryComplete` /
  :class:`LLMQueryExit` — the per-turn LLM call.
- :class:`LLMQueryRetry` — fired per retry attempt of that call.
- :class:`REPLExecEnter` / :class:`REPLExecSend` / :class:`REPLExecComplete` /
  :class:`REPLExecExit` — the turn's REPL execution. This is the one conditional span: it opens
  only on a turn that reaches execution, so it is skipped when the query span ended without
  getting there — the call errored, or a hook aborted at any of the ``LLMQuery`` events,
  ``LLMQueryComplete`` and ``LLMQueryExit`` included, which abort *after* a response exists.
  What does **not** skip it is the response's content: :meth:`BaseProtocol.parse` is contracted
  to return the code and never raise, and an empty parse is empty *code*, so an empty, blank or
  prose-only response opens the span like any other. (A protocol that raises anyway is terminal
  — nothing catches it — and ends the invoke rather than skipping the span.)

The shape of every event follows from four principles:

1. **Ambient invoke context rides every event.** What belongs to the invoke rather than to any
   one span — ``config``, ``scope``, ``hooks``, ``blackboard``, plus the identity
   ``invoke_id``/``depth`` — is a base :class:`~jaz.hooks.Event` field, stamped on every event
   by the dispatcher. None of it is a proposal: no ``*Enter`` effect edits it and no ``*Send``
   carries an edit record for it. (``blackboard`` still advances between events, by
   :class:`BlackboardWrite`, and ``hooks`` is the live active set.)
2. **An ``*Enter`` carries what its effects can edit, as the proposal.** Under the field's own
   name: ``inputs``, ``messages``, ``code`` and ``namespace``.
3. **A ``*Send`` carries the same fields committed, plus one ``<thing>_edits`` per field.**
   ``input_edits``, ``message_edits``, ``code_edits``, ``namespace_edits`` — each a
   ``Resolved<Thing>Edits`` of self-contained records of what every enter-time effect did, one
   sequence per effect named by its verb (``adds``/``drops``, ``inserts``/``deletes``),
   ordered by the coordinate a record names (name, position, offset) and, where a coordinate
   can repeat, by the order the fold applied them within it — a drop record carrying the value
   it removed, read off the Enter snapshot. Only same-slot message adds and same-offset code
   inserts can share a coordinate; each sequence's own docstring says what its order resolves
   to there.
4. **An effect edits the field it is named for, and only that.** ``DropInputs`` edits
   ``inputs`` (a ``jaz.scope`` value is not an input); ``AddToNamespace`` edits ``namespace``.

Which gives, per span::

    ┌─────────────────────┬──────────────────────┬──────────────────────┬──────────────────────────────────┐
    │                     │ Invoke               │ LLMQuery             │ REPLExec                         │
    ├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────────────────┤
    │ Enter: the proposal │ inputs               │ messages             │ code, namespace                  │
    │ Send: committed     │ inputs               │ messages             │ code, namespace                  │
    │ Send: the edits     │ input_edits          │ message_edits        │ code_edits, namespace_edits      │
    │   ← effects         │ AddInputs/DropInputs │ AddMessages/         │ InsertCode/DeleteCode,           │
    │                     │                      │ DropMessages         │ AddToNamespace/DropFromNamespace │
    │   record sequences  │ adds, drops          │ adds, drops          │ inserts, deletes; adds, drops    │
    │ Enter → Send shape  │ same                 │ same (stamped)       │ same                             │
    │ Loop → span handoff │ send(inputs)         │ send(shown_messages) │ send(namespace)                  │
    └─────────────────────┴──────────────────────┴──────────────────────┴──────────────────────────────────┘

Outside that shape, and each deliberate:

- :class:`DisableRecursion` and :class:`WithholdREPLHistory` are ``InvokeEnter`` effects that edit no
  field (they withhold ``invoke`` and ``__history__``, which are not inputs), so they are
  recorded as :attr:`InvokeSend.recursion_disabled` and :attr:`InvokeSend.repl_history_withheld`
  rather than inside any ``<thing>_edits``; their result shows one span later, as the absence of
  ``invoke`` or ``__history__`` from ``namespace["__builtins__"]`` in
  :attr:`REPLExecEnter.namespace`.
- :class:`LLMQueryRetry` is not a value-state event, so the stage table above does not describe
  it: it fires per *retry* rather than at a commit, only on this span, and accepts no effects.
  It rides the retry policy's before-sleep, so it counts the attempts that are followed by
  another one: a turn whose first attempt succeeds never fires it, and the final failure — the
  one that exhausts the policy and propagates — fires nothing either. It has no ``*Enter``/``*Send``
  counterpart because a retry commits nothing, and no counterpart on the other spans because
  only the provider call is retried.
- ``*Exit`` is terminal on every span, but the two per-turn spans accept :class:`Abort` there,
  on every arm, while :class:`InvokeExit` stays observation-only. The per-turn opening buys
  aborting on the *post-transform* result — a re-priced cost, say. An invoke aborting at its
  own terminal ``Exit`` would instead make that event's payload diverge from what the caller
  receives, and :class:`InvokeComplete` already governs the final result.

Events also carry coordinates that place them, which the table does not show because they are
neither proposal nor edit and no effect touches them: which turn (``iteration``, on every
per-turn event), which invoke and how deep (``invoke_id``, ``depth``), when (``timestamp``,
a base field like the ambient context but varying per event rather than per invoke), and on
:class:`InvokeEnter` which invoke and which of its turns spawned this one
(``parent_invoke_id``, ``parent_repl_iteration``). The payload a ``*Complete`` or ``*Exit``
carries is not in the table either; those stages are the pipeline's, described above it.

Two deliberate non-uniformities in the records: only the message records carry ``persistent``
(only the message buffer has a per-query vs carried-forward distinction), and
``REPLExecEnter.namespace``/``REPLExecSend.namespace`` are exact for :class:`~jaz.repl.PythonREPL`
and best-effort for a REPL whose namespace it cannot read (see :meth:`BaseREPL.get_namespace`).
An observer written against one Send reads the others the same way.

**Experimental.** The hook system is an experimental feature; its interfaces may change
in a future release.
"""

# The pipeline framing is design/design_features/span_event_lifecycle.md, fully landed:
# the four-stage lattice is live (suppliers compose at *Send, transformers at *Complete, *Exit
# always fires and transforms nothing — the per-turn LLMQueryExit / REPLExecExit accept an Abort
# on every arm, InvokeExit no effect at all), and the abort model propagates an Abort's
# carried exception from the span CMs (the aborting invoke's spans close Aborted).

from .._base import Event
from ._base import Aborted, Completed, Failed
from ._invoke import (
    InputAddRecord,
    InputDropRecord,
    InvokeComplete,
    InvokeEnter,
    InvokeExit,
    InvokeOutcome,
    InvokeSend,
    ResolvedInputEdits,
)
from ._llm_query import (
    LLMQueryComplete,
    LLMQueryEnter,
    LLMQueryExit,
    LLMQueryOutcome,
    LLMQueryRetry,
    LLMQuerySend,
    MessageAddRecord,
    MessageDropRecord,
    ResolvedMessageEdits,
)
from ._repl_execution import (
    CodeDeleteRecord,
    CodeInsertRecord,
    NamespaceAddRecord,
    NamespaceDropRecord,
    REPLExecComplete,
    REPLExecEnter,
    REPLExecExit,
    REPLExecOutcome,
    REPLExecSend,
    ResolvedCodeEdits,
    ResolvedNamespaceEdits,
)

# Public event surface (`jaz.hooks.events`) — the typed `event` a hook handler receives.
# `Event` is the base. The `*Context`/`*Span` variants are the dispatcher's internal
# effect-composition contracts (zero consumer usage) — imported above for reachability but
# deliberately kept OUT of __all__.
__all__ = [
    "Event",
    # Span-outcome variants (every *Exit's ``outcome`` is Completed[...] | Aborted | Failed)
    "Completed",
    "Aborted",
    "Failed",
    # Per-span outcome-union aliases
    "LLMQueryOutcome",
    "REPLExecOutcome",
    "InvokeOutcome",
    # REPL execution
    "REPLExecEnter",
    "REPLExecSend",
    "ResolvedCodeEdits",
    "ResolvedNamespaceEdits",
    "CodeInsertRecord",
    "CodeDeleteRecord",
    "NamespaceAddRecord",
    "NamespaceDropRecord",
    "REPLExecComplete",
    "REPLExecExit",
    # LLM query
    "LLMQueryEnter",
    "LLMQuerySend",
    "LLMQueryComplete",
    "LLMQueryExit",
    # Resolved-edit records carried by LLMQuerySend
    "ResolvedMessageEdits",
    "MessageDropRecord",
    "MessageAddRecord",
    # Invoke
    "InvokeEnter",
    "InvokeSend",
    "ResolvedInputEdits",
    "InputAddRecord",
    "InputDropRecord",
    "InvokeComplete",
    "InvokeExit",
    # LLM retry
    "LLMQueryRetry",
]
