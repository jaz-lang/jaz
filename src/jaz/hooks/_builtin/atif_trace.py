"""Hook that emits ATIF (Agent Trajectory Interchange Format) traces.

Produces Harbor-compatible trajectory JSON directly from JAZ execution events,
without an intermediate conversion step.

Usage::

    from jaz import invoke
    from jaz.hooks import TrajectoryRecorder

    with TrajectoryRecorder(output_path="./trace.atif.json"):
        result = invoke(task="Do something")

    # Creates: ./trace.atif.json validated by harbor.utils.trajectory_validator

ATIF is an external standard (Harbor), currently ``ATIF-v1.7``; we target that version and
do not invent our own. Its ``extra`` field — the spec's slot for custom information — carries
message provenance and compaction edits, letting it be the sole conversation-history
format:

- ``extra.provenance`` on system/user steps: the message's :class:`MessageProvenance`
  (``provenance_of(msg).to_dict()``) — its true origin plus a stable per-message id — and
  ``extra.position``, the message's slot in the buffer at the enter it was first seen (an older
  trace spells it ``extra.index``).
- ``extra.writer_records`` on the trajectory: what this writer records beyond core ATIF, so a
  reader never has to date a document by probing for a field. See ``WRITER_RECORDS``.
- ``extra.raw_message`` on an agent step whose content a Complete-stage ``ModifyLLMResponse``
  changed: the PRE-transform content. Absent when nothing changed it, which on a document that
  declares the capability means "not transformed" rather than "unknown".
- ``extra.message_edits`` on the **last message emitted the turn the edits were composed**: the
  compaction drops/adds for that query, named after :attr:`LLMQuerySend.message_edits` and shaped
  like it — ``{"iteration": N, "drops": [...], "adds": [...]}``. ``iteration`` is
  the query the edits shaped (block-level, once) — distinct from the anchor step's own
  ``extra.iteration``, since the edits are composed at enter N but hang on the last message of turn
  N-1's output. A drop is ``{id, position, persistent}``, standing in for
  :class:`MessageDropRecord` with an id where the record has the message (its content already lives
  in that message's own step); an add is ``{position, persistent, messages}`` as
  :class:`MessageAddRecord` has them, each message as the buffer holds it — ``role`` and
  ``content`` and whatever else the backend needs, such as a tool call's ``tool_call_id`` — plus
  ``provenance`` once the message has an id. A persistent add is given its id after the query it
  was composed for, so the id is written when the *next* query runs: an add on the invoke's last
  query, including the single-query case, records no ``provenance`` at all. ``position`` is the
  record's own field, the edit's input coordinate (the resolved slot). See ``on_llm_query_send``.
- ``extra`` on the trajectory itself: ``depth``, ``parent_repl_iteration``, ``inputs`` /
  ``scope`` (name → ``{type, repr_prefix_10000}``), ``input_edits`` (``{"adds": [name], "drops":
  [name]}``), ``recursion_disabled`` when a hook withheld ``subinvoke`` and
  ``repl_history_withheld`` when one withheld ``__history__`` (all from :class:`InvokeSend`),
  ``hooks``, and the run's failure records, each keyed the way the event is: ``retries``
  (``{iteration, attempt_number, exception, wait_seconds}`` from :class:`LLMQueryRetry`),
  ``query_failures`` (``{iteration, outcome}`` for every non-completed :class:`LLMQueryExit`) and
  ``outcome`` (:class:`InvokeExit`'s). An outcome is ``{"type": "aborted" | "failed", "exception":
  ...}`` or ``{"type": "completed", "result": ...}`` whose result is ``{"type": "return",
  "return_value": ...}`` or ``{"type": "raise", "exception": ...}``; a value is ``{type,
  repr_prefix_10000}`` and an exception is text, ``<class>: <message>`` (one such line per member of
  an exception group).

See:
    https://www.harborframework.com/docs/agents/trajectory-format
    https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

# The provenance channel is #599 (stable per-message ids); riding ``extra`` to make ATIF
# the sole conversation-history format is #565's decision.

# The document-shape block in TrajectoryRecorder's docstring exists because the standard-vs-ours
# split is not recoverable from the file: `subagent_trajectories` and `final_metrics` are both
# ATIF-v1.7 fields, but "excludes subagents" and "total_steps counts REPL iterations" are our
# readings of them, and `message`-is-pre-parse is invisible until a fenced response shows up.
# Written after a consumer measured a rate over the root `steps` alone and reported it as the
# whole run: the recursion is the standard's, but nothing in our docs said to take it.
#
# Its per-trajectory `extra` list is in the order `_build_atif` writes the keys, which is the order
# they appear in the JSON — so the list can be diffed against the emitter rather than re-derived.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaz._messages import MessageDict
from jaz._provenance import PROVENANCE_KEY, MessageKind, provenance_of
from jaz._string_utils import safe_repr, summarize_exception
from jaz.hooks._dispatcher import Hook
from jaz.hooks.events import Completed, InvokeOutcome
from jaz.hooks.events._invoke import InvokeEnter, InvokeExit, InvokeSend
from jaz.hooks.events._llm_query import (
    LLMQueryComplete,
    LLMQueryEnter,
    LLMQueryExit,
    LLMQueryRetry,
    LLMQuerySend,
)
from jaz.repl._types import Raise, Return

SCHEMA_VERSION = "ATIF-v1.7"

#: The ``extra`` extensions a reader has to GATE on, declared verbatim on every document this
#: writer writes (``extra.writer_records``), so a reader asks "does this trace record X?" instead
#: of inferring it from a version.
#:
#: Not an inventory of ``extra``. A field this writer emits unconditionally — ``provenance``,
#: ``inputs``, ``scope``, ``depth`` — needs no declaration, because its absence from a step is
#: an answer about that step rather than about the writer. Only a field whose absence is
#: ambiguous between "nothing to record here" and "this writer never records it" belongs, which
#: is why both entries below are conditional ones. ``_writer_records`` answers ``False`` for
#: anything not listed, so listing a field that is always written would say nothing, and asking
#: about one is a question that does not arise.
#:
#: WHY A CAPABILITY LIST AND NOT A VERSION. ``SCHEMA_VERSION`` tracks the EXTERNAL ATIF standard
#: (Harbor) and says nothing about our ``extra`` extensions: it has read "ATIF-v1.7" since the
#: format landed (#381) and was deliberately not bumped when ``extra.edits`` was added (#1089),
#: because the document still conforms to v1.7 — the extension rides ``extra``, which is the
#: spec's slot for exactly this. Bumping it would claim conformance to a Harbor version that
#: does not exist. So our extensions need their own signal, and ``_trace_records_edits`` had to
#: date documents by probing for an unrelated field (``extra.provenance``) that happened to land
#: in the same commit — a heuristic that works only because nobody has since added a third field.
#: This list replaces that: it is written UNCONDITIONALLY, so its absence dates a document as
#: pre-declaration and its contents answer every later question directly.
#:
#: Append-only in practice. Removing an entry tells past readers a field they may rely on is
#: gone, which is a compatibility break even though the list itself is only metadata.
#:
#: The key is unprefixed (``extra.writer_records``, not ``extra.jaz_writer_records``) to match
#: every other extension we put in ``extra`` — ``provenance``, ``message_edits``, ``inputs``,
#: ``scope``.
#: ``agent.name`` already identifies the writer as jaz, so the namespace is the document's, and a
#: reader that has reached ``extra`` has already accepted this writer's conventions.
WRITER_RECORDS: tuple[str, ...] = (
    # ``extra.message_edits`` on the last message of the turn the edits were composed (#1089).
    # Declared under the key it writes: a trace written before that block was named after
    # ``LLMQuerySend.message_edits`` declares the same capability as ``edits``, and a reader is
    # expected to accept either -- ``_trace_records_edits`` in replay.py does.
    "message_edits",
    # ``extra.raw_message`` on an agent step whose response a Complete-stage ``ModifyLLMResponse``
    # changed — the PRE-transform content, so replay can supply what the producer actually
    # returned rather than the transformed turn (#1309).
    "raw_message",
)

#: ATIF v1.7 ``MetricsSchema`` fields that ``LLMResponse`` carries inside ``extra`` rather
#: than as fields of its own, and that the emitter therefore lifts back to the top level
#: of ``metrics`` on the way out.
ATIF_EXTRA_HOISTED_METRICS = ("prompt_token_ids", "completion_token_ids", "logprobs")
# These three are optional, provider-rare and per-token — large enough that promoting them
# to `LLMResponse` fields would write a retention cost into the contract every backend
# nominally implements (see the comment on `LLMResponse`). Backends spell them with these
# ATIF names in `extra` already, so the emitter is where the format's conformance is owed and
# cheapest to pay.


def hoist_atif_metrics(metrics: dict[str, Any], extra: dict[str, Any]) -> None:
    """Move ATIF's top-level token-id/logprob metrics out of ``extra`` into ``metrics``.

    Mutates both: ``extra`` is left holding only genuinely provider-specific keys, which is
    what ATIF's ``extra`` is specified to be.
    """
    for key in ATIF_EXTRA_HOISTED_METRICS:
        if key in extra:
            metrics[key] = extra.pop(key)


def _serialize_value(value: object) -> dict[str, Any]:
    """An arbitrary Python value in the ATIF extra form: its class name and a bounded repr."""
    # Never raises (see ``safe_repr``): inputs and scope are serialized in ``teardown``, outside
    # the dispatcher's per-event guard, so a raising ``__repr__`` there would escape the caller's
    # ``with`` block in place of the run's own error, with no file written.
    return {"type": type(value).__name__, "repr_prefix_10000": safe_repr(value, 10000)}


def _serialize_namespace(namespace: dict[str, Any]) -> dict[str, Any]:
    """A name→value namespace (inputs or scope), each value via :func:`_serialize_value`."""
    return {k: _serialize_value(v) for k, v in namespace.items()}


def _serialize_exception(exc: BaseException) -> str:
    """An exception in the ATIF extra form: ``<class>: <message>`` as text, one such line
    per member for an exception group."""
    # The same text OTelTracing records under its exception paths, so the two hooks
    # agree; a ``{type, message}`` pair would only repeat the class name that
    # ``summarize_exception`` already leads with. summarize_exception, not str():
    # multiple aborts at one boundary group into an ExceptionGroup by law, and str() on
    # a group renders only a sub-exception count, dropping every child's text — on
    # exactly the failure records this form exists for.
    return summarize_exception(exc)


def _serialize_outcome(outcome: InvokeOutcome) -> dict[str, Any]:
    """An outcome union value, shaped like the event field it came from.

    Every node carries its variant under ``type`` (``completed`` / ``aborted`` / ``failed``,
    and ``return`` / ``raise`` under a completed invoke's ``result``) and then the
    variant's own fields: ``exception`` on Aborted, Failed and Raise, ``return_value`` on
    Return.
    """
    # The block mirrors ``InvokeExit.outcome`` field-for-field (as OTelTracing's attribute
    # paths do), so a reader can go from the event to the trace and back without a
    # renaming table. ``type`` is the lowercased class name because the variant *is* the
    # status: there is no parallel enum to name. Only the recorder-specific
    # serialization differs: a bounded repr for values, text for exceptions. A ``None``
    # return value is not omitted: it is serialized like any other value
    # (``{"type": "NoneType", ...}``), the same rule ``_serialize_namespace`` applies to a
    # ``None`` input, so the document has one rule and a reader can tell "returned None"
    # from "no result".
    if not isinstance(outcome, Completed):
        return {
            "type": type(outcome).__name__.lower(),
            "exception": _serialize_exception(outcome.exception),
        }
    match outcome.result:
        case Return(return_value=rv):
            result: dict[str, Any] = {
                "type": "return",
                "return_value": _serialize_value(rv),
            }
        case Raise(exception=exc):
            result = {"type": "raise", "exception": _serialize_exception(exc)}
        case other:  # pragma: no cover - the outcome union admits nothing else here
            raise TypeError(f"unexpected invoke result {type(other).__name__}")
    return {"type": "completed", "result": result}


def _serialize_message(message: MessageDict) -> dict[str, Any]:
    """Serialize a message to JSON-safe form for the ``extra.message_edits`` record.

    Minimal local serializer: strips the internal ``PROVENANCE_KEY`` and re-emits it
    as a clean ``"provenance"`` sub-object, and falls back to ``repr`` for any non-JSON value.
    """
    result: dict[str, Any] = {}
    for key, value in message.items():
        if key == PROVENANCE_KEY:
            # Emitted as a clean "provenance" sub-object below, not the raw magic key.
            continue
        try:
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            result[key] = safe_repr(value)
    prov = provenance_of(message)
    if prov is not None:
        result["provenance"] = prov.to_dict()
    return result


@dataclass
class _InvokeState:
    """Tracks per-invoke state for ATIF step building."""

    invoke_id: str
    parent_invoke_id: str | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    step_counter: int = 0
    model_name: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost: float = 0.0
    total_repl_iterations: int = 0
    current_iteration: int | None = None
    # The raw (pre-transform) content of the query now in flight, captured at ``LLMQueryComplete``
    # and consumed by the matching ``LLMQueryExit``.
    #
    # TWO FIELDS, NOT ONE, because ``LLMResponse.content`` is ``str | None`` and a Complete-stage
    # transform may legitimately FILL IN a ``None`` response. Folding "no Complete seen" into the
    # same ``None`` would record that turn as untransformed, so no raw would be written, and the
    # replay would then supply the final and let the rerun's transform run over it a second time —
    # a spurious divergence on a faithful rerun, the exact failure this field exists to prevent.
    #
    # Single-slot is safe under concurrency, unlike ``TrajectoryReplay``'s stack (#1310): this state is
    # per-invoke (``_invokes`` is keyed by ``invoke_id``), and one invoke drives at most one query
    # at a time, so concurrent ``ainvoke``s cannot interleave into the same slot.
    pending_raw_content: str | None = None
    pending_raw_seen: bool = False
    # Ids of messages already emitted as steps (SEED/OBSERVATION), so each is emitted once even
    # though it reappears in every later enter's buffer. Keyed by the stable per-message id (#599).
    emitted_ids: set[str] = field(default_factory=set)
    # The step the most recent turn's ``extra.message_edits`` hangs on: that turn's observation,
    # or a seed on the first turn. Chosen in ``on_llm_query_enter`` — see the anchor rule there.
    edits_anchor_step: dict[str, Any] | None = None
    # Agent steps by iteration. An agent step is emitted at exit *before* its assistant message is
    # stamped, so its provenance/id is filled in at the next enter (when the stamped assistant
    # appears) — letting a drop that names the assistant's id resolve to its step.
    agent_step_by_iteration: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Persistent added messages whose serialized record is written but whose provenance does not
    # exist yet, keyed by the turn that added them: ``Agent`` stamps ADDED provenance after the
    # query returns, so at ``LLMQuerySend`` there is no id to record. Each entry pairs the buffer
    # message with the dict already emitted into the block, so the next enter -- where the message
    # arrives stamped -- fills the id in by identity. Transient adds are not tracked: the agent
    # never stamps one, because it never enters the buffer.
    #
    # Entries are REMOVED once stamped, which is what keeps the map bounded: an add that is never
    # followed by another query (the invoke's last turn) leaves its entry behind, so without the
    # removal the map would also pin every dropped message's dict for the invoke's lifetime.
    # Named for its key like the sibling ``agent_step_by_iteration``; it holds pairs, not messages.
    pending_adds_by_iteration: dict[int, list[tuple[MessageDict, dict[str, Any]]]] = (
        field(default_factory=dict)
    )
    children: list[_InvokeState] = field(default_factory=list)
    parent_repl_iteration: int | None = None
    depth: int | None = None
    # Metadata captured from the invoke events for ATIF extra fields. There is no separate
    # `prompt` field anymore (#538): `task` is an ordinary input, captured in `inputs` below.
    # Explicit `**inputs` kwargs and resolved ambient `jaz.scope`, kept as SEPARATE provenance
    # channels (#727) — emitted as distinct ``extra.inputs`` / ``extra.scope`` blocks.
    # ``inputs`` is snapshotted at InvokeSend (the committed set, post AddInputs/DropInputs),
    # NOT at InvokeEnter: a trace built from Enter mis-records what the child actually
    # received whenever a hook injects an input (the input-side #906 pattern — e.g.
    # ultrahorizon's per-child ``env`` library). ``scope`` stays an Enter snapshot: it is not
    # editable at InvokeEnter, so Enter's copy is already the committed one.
    inputs: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    # Which of the committed input names hooks added / which caller inputs hooks dropped
    # (``InvokeSend.input_edits``, under the same names) — the provenance split that separates
    # hook-injected inputs from caller-passed ones inside ``extra.inputs``. Names only, so an
    # empty ``adds`` or ``drops`` is left out entirely rather than written as a bare list.
    input_edits: dict[str, list[str]] = field(default_factory=dict)
    # ``InvokeSend.recursion_disabled``: whether a hook withheld this invoke's ``subinvoke``.
    recursion_disabled: bool = False
    # ``InvokeSend.repl_history_withheld``: whether a hook withheld this invoke's ``__history__``.
    repl_history_withheld: bool = False
    # Per-query non-completed closes (LLMQueryExit with an Aborted/Failed outcome): the
    # turns that have no agent step, with the reason why.
    query_failures: list[dict[str, Any]] = field(default_factory=list)
    # Per-attempt LLM retry records (LLMQueryRetry) — a call that failed and retried
    # leaves its failed attempts, errors, and backoff in the trace rather than showing
    # one clean call however many attempts it took.
    retries: list[dict[str, Any]] = field(default_factory=list)
    # ``InvokeExit.outcome``, serialized (see ``_serialize_outcome``); None until exit.
    outcome: dict[str, Any] | None = None
    # ``repr()`` of the hooks active for this invoke (propagating ‖ local), captured at receipt
    # from the live ``event.hooks`` (#727). Strings, not structured dicts: hooks have no
    # serialization contract, so what a hook chooses to show in its ``__repr__`` is the whole
    # record. Rendered here rather than at write time because ``event.hooks`` holds live instances
    # whose state moves on after the event.
    hooks: tuple[str, ...] = ()

    def next_step_id(self) -> int:
        self.step_counter += 1
        return self.step_counter


class TrajectoryRecorder(Hook):
    """Write the run as an ATIF trajectory JSON file.

    ATIF (Agent Trajectory Interchange Format) is the standard trace format JAZ reads and
    writes; a trajectory records an invoke's turns, messages, and per-call metrics.

    A pure observer: it emits no effects, so it never influences the run. The file is
    written once, when the hook's scope ends — a single trajectory object when the scope
    held one top-level ``invoke``, an array when it held several (nested invokes appear
    under their parent's ``subagent_trajectories``, not as extra roots). The resulting
    trace is also the input :class:`TrajectoryReplay` resumes from.

    Because the write happens at scope end, nothing is on disk while the run is in progress:
    ``tail -f`` on ``output_path`` shows nothing until it finishes, and a process killed outright
    leaves no file at all. An ordinary exception does still write, and :meth:`get_trajectories`
    returns the same documents mid-run. For a trace that reaches disk *as it happens*, see
    :class:`TrajectoryDirectoryRecorder`, which writes a browsable directory rather than this JSON.

    Under ``with`` the trace covers invokes nested inside. Passed positionally, only that
    invoke is traced and its sub-invokes are absent from the output.

    **Document shape.** ``ATIF-v1.7`` fixes the field names and where custom data may live; what the
    values *mean* for a JAZ run is set here, and a generic ATIF reader cannot assume it.

    Standard fields written:

    - ``schema_version`` — ``"ATIF-v1.7"``. ``trajectory_id`` — the invoke id.
    - ``agent`` — ``{"name", "version"}``, plus ``model_name`` when the backend reports one
      (omitted when ``model_info.name`` is ``None``, as for :class:`~jaz.llm.MockLLM`). The model
      is recorded once per trajectory, so the per-step ``model_name`` the standard allows is
      never written.
    - ``steps`` — one per message, in buffer order. ``source`` is ``system`` / ``user`` / ``agent``.
      On an ``agent`` step, ``message`` is the LLM's response **verbatim, as it arrived and before
      the protocol parses it**: a response wrapped in markdown fences appears here with its fences,
      and is not what the REPL went on to execute. ``llm_call_count`` is 1 there and 0 elsewhere;
      ``metrics`` rides the same step, carrying tokens, ``cost_usd``, and ``cached_tokens``
      whenever the backend reports a count at all — 0 included, so the key's presence says the
      backend tracks caching, not that this turn hit a cache.
    - ``final_metrics`` — totals **for this invoke alone, excluding its subagents**; sum the tree
      yourself for a whole-run figure. ``total_steps`` counts REPL iterations, not ``len(steps)``.
      The document's top-level ``notes`` says so inside the file, for a reader holding only the
      JSON.
    - ``subagent_trajectories`` — one complete, independently-valid trajectory per nested invoke,
      recursively. **A reader that walks only the top-level ``steps`` sees the root invoke and
      nothing else**, which on a delegating run is a small minority of the turns.

    Standard fields deliberately absent: ``session_id``, ``continued_trajectory_ref``, and the
    step-level ``reasoning_content``, ``tool_calls``, ``observation``, ``is_copied_context`` and
    ``reasoning_effort``. A REPL agent acts by writing code in ``message`` rather than by calling
    tools, and its observations arrive as ordinary ``user`` steps.

    Everything JAZ-specific rides ``extra``, the standard's slot for it, and no other reader should
    expect it. Per trajectory: ``writer_records`` (always, so a reader need not date the document by
    probing for a field), then ``depth``, ``parent_repl_iteration``, ``inputs``, ``input_edits``,
    ``recursion_disabled``, ``repl_history_withheld``, ``scope``, ``hooks``, ``query_failures``,
    ``retries`` and ``outcome``, in that order — each omitted when empty, so a plain completed run
    carries few of them. The invoke's result is inside ``outcome``, shaped like
    :attr:`InvokeExit.outcome` field for field, rather than in flat ``result_*`` keys. Per step:
    ``provenance``, ``position``, ``iteration``, ``message_edits`` and ``raw_message`` on a
    transformed agent step — the conversation-history channel this module's own docstring specifies
    in full.

    Example::

        from jaz import invoke
        from jaz.hooks import TrajectoryRecorder

        with TrajectoryRecorder(output_path="./trace.atif.json"):
            result = invoke(task="Do something")

    Args:
        output_path: Where to write the trajectory JSON when the scope ends. ``None``
            (the default) writes no file — read the documents in-memory with
            :meth:`get_trajectories` instead.
        indent: Indentation for the written JSON; ``None`` writes compact single-line
            output.
    """

    # Builds ATIF steps directly as events fire, rather than converting an intermediate
    # per-iteration JSON in a second pass.

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        indent: int | None = 2,
    ):
        self.output_path = Path(output_path) if output_path else None
        self.indent = indent
        self._invokes: dict[str, _InvokeState] = {}
        self._root_invokes: list[_InvokeState] = []

    def on_invoke_enter(self, event: InvokeEnter) -> list:
        state = _InvokeState(
            invoke_id=event.invoke_id,
            parent_invoke_id=event.parent_invoke_id,
            parent_repl_iteration=event.parent_repl_iteration,
            depth=event.depth,
            # ``inputs`` is NOT captured here — it is snapshotted at on_invoke_send, from the
            # committed post-add/drop set (see the field comment on _InvokeState.inputs).
            # ``scope`` is not editable at enter, so this copy is already the committed one.
            scope=dict(event.scope),
            # Render the live active-hook set here, at receipt, so the trace pins the governance
            # active AT InvokeEnter (not whatever a live peer mutates to later) (#727).
            # ``safe_repr``: a hook subclass's ``__repr__`` is user code, and raising here would
            # abort the enter before this state is registered, losing the whole trajectory.
            hooks=tuple(safe_repr(h) for h in event.hooks),
        )
        self._invokes[event.invoke_id] = state
        if event.parent_invoke_id and event.parent_invoke_id in self._invokes:
            self._invokes[event.parent_invoke_id].children.append(state)
        else:
            self._root_invokes.append(state)
        return []

    def on_invoke_send(self, event: InvokeSend) -> list:
        state = self._invokes.get(event.invoke_id)
        if not state:
            return []
        # The COMMITTED explicit inputs (post AddInputs/DropInputs) — what the prompt renders
        # and the REPL binds, so the trace records what the invoke actually received.
        # SHALLOW snapshot: dict() pins the key set / top-level bindings at receipt, but nested
        # MUTABLE values are shared with the live namespace, so a value the agent mutates during
        # execution is reflected in the trace serialized later (at teardown). This is the same
        # deep/shallow ceiling as event.inputs itself — pinning nested values would need eager
        # deep serialization here, which is costly and can fail on non-copyable inputs, so it
        # is a documented limitation (#831), not a bug.
        state.inputs = dict(event.inputs)
        # The hook-vs-caller provenance split within the committed set (#727's channel
        # discipline, extended to hook-injected inputs): names only. An added name's value is
        # already in ``inputs`` above, so repeating it would double the payload; a dropped
        # name's value is in neither, since ``inputs`` is the post-drop set and the proposal
        # at ``InvokeEnter`` is never snapshotted — the trace says what was un-passed, not
        # what it held.
        # Taken in the event's own order rather than re-sorted: the dispatcher builds both
        # sequences sorted by name, which is the ordering rule ``ResolvedInputEdits``
        # documents, so a hand-built event that ignores it is recorded as it is.
        edits = {
            "adds": [record.name for record in event.input_edits.adds],
            "drops": [record.name for record in event.input_edits.drops],
        }
        state.input_edits = {verb: names for verb, names in edits.items() if names}
        # The two InvokeEnter effects with no ``<thing>_edits`` home: they edit no field, they
        # withhold the ``subinvoke`` binding or ``__history__``. Recorded because nothing else in
        # the document shows them — without them a subagent denied recursion reads exactly like
        # one that never recursed.
        state.recursion_disabled = event.recursion_disabled
        state.repl_history_withheld = event.repl_history_withheld
        return []

    def on_llm_query_send(self, event: LLMQuerySend) -> list:
        state = self._invokes.get(event.invoke_id)
        if not state:
            return []
        # Record this turn's compaction edits on the anchor step (the last message emitted this turn;
        # see on_llm_query_enter), not the agent step. The records arrive pre-resolved on the event
        # (LLMQuerySend.message_edits): each drop is {id, position, persistent} and each add is
        # {position, persistent, messages} as MessageAddRecord has them, its messages carrying
        # provenance when stamped, under a block-level `iteration` = the
        # query these edits shaped. Per-record iteration is not repeated. A dropped message's content
        # already lives in its own step (#599), so the id is a non-redundant
        # reference. This replaced the former exit-time resolution against a per-hook snapshot
        # of LLMQueryEnter.messages (this hook kept its own enter-buffer copy and reused the
        # effects layer's private index resolver): the dispatcher now resolves once against
        # the same snapshot the buffer fold used, so the trace agrees with the buffer by
        # construction and this hook keeps no index math of its own.
        # ``edits_anchor_step is None`` would drop this turn's edits, but no in-tree path
        # reaches it, so no fallback anchor is warranted — #565 review. Every buffer message
        # is stamped (_agent stamps SEED/ASSISTANT/OBSERVATION and persistent ADDED; transient
        # adds never enter the buffer, and LLMQueryEnter.messages is an enter-time
        # snapshot of the buffer — same membership, shared message dicts), so the
        # ``prov is None`` skip in on_llm_query_enter can't fire; iteration 1 always emits the
        # seed, and every continuing turn appends a fresh OBSERVATION that the next enter
        # emits. That observation cannot be pre-empted by a drop: edits composed at turn i are
        # applied to the buffer BEFORE turn i's assistant and observation are appended, so a
        # drop can never name a message that does not exist yet. Only a custom protocol
        # returning an empty render_observation could get here — which BaseProtocol's "one or
        # more ready messages" contract already forbids.
        edits = event.message_edits
        if (edits.drops or edits.adds) and state.edits_anchor_step is not None:
            drops: list[dict[str, Any]] = []
            for record in edits.drops:
                drop_prov = provenance_of(record.message)
                # ``MessageDropRecord`` minus its ``message``, which is an id here: the content
                # already lives on that message's own step. No per-record iteration: the turn the
                # drop happens is the block's ``iteration`` (below), and the dropped message's own
                # origin turn lives on its own step (via provenance).
                drops.append(
                    {
                        "id": drop_prov.id if drop_prov is not None else None,
                        "position": record.position,
                        "persistent": record.persistent,
                    }
                )
            adds: list[dict[str, Any]] = []
            for record in edits.adds:
                # ``MessageAddRecord`` field for field. ``position`` is the record's, not each
                # message's: it is the add's input coordinate (the resolved insert-before slot),
                # not a computed final slot, since several adds landing at earlier positions shift
                # the later ones and ``position + offset`` would not be the true post-fold index.
                # A persistent add's ``provenance`` (with id) is backfilled at the next enter --
                # see ``pending_adds_by_iteration``. The *agent* never stamps a transient add, so
                # one carries provenance only when whoever composed the add handed in a dict that
                # was already stamped (a hook re-adding a message the buffer holds); the id is then
                # that earlier message's, which is why the backfill overwrites rather than defers to
                # it. Per-record iteration is not needed -- the whole block carries one (below).
                messages = [_serialize_message(m) for m in record.messages]
                if record.persistent:
                    pending = state.pending_adds_by_iteration.setdefault(
                        event.iteration, []
                    )
                    pending.extend(zip(record.messages, messages, strict=True))
                adds.append(
                    {
                        "position": record.position,
                        "persistent": record.persistent,
                        "messages": messages,
                    }
                )
            # ``iteration`` is the query these edits were composed for (``event.iteration``), carried
            # once for the whole block. It is NOT the anchor step's ``extra.iteration``: the edits are
            # composed at enter N but hang on the last message of turn N-1's output, so the step reads
            # N-1 while the edits belong to N -- recording it here is what makes that unambiguous.
            # Named and shaped after ``LLMQuerySend.message_edits``, like every other key this
            # hook writes. A trace written before the rename spells it ``extra.edits`` with
            # ``dropped`` / ``added`` and a per-message ``index``; the reader that has to care is
            # ``TrajectoryReplay``, whose ``_recorded_edit_signature`` reduces either spelling to
            # the same signature, so an old trace still detects divergence rather than silently
            # reading as "no edits this turn".
            state.edits_anchor_step.setdefault("extra", {})["message_edits"] = {
                "iteration": event.iteration,
                "drops": drops,
                "adds": adds,
            }
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list:
        state = self._invokes.get(event.invoke_id)
        if not state:
            return []

        # #481 removed the REPLIteration* events, and each iteration issues exactly one LLMQuery,
        # so LLMQueryEnter is the per-iteration boundary.
        state.current_iteration = event.iteration
        state.total_repl_iterations += 1

        # Emit the messages new this turn, identified by provenance -- not by a buffer diff. Every
        # buffer message carries a stable id (#599), so emit each SEED and OBSERVATION exactly once
        # (keyed by id) and skip ASSISTANT (emitted at LLMQueryExit as the agent step) and ADDED (a
        # compaction summary / hook injection -- in extra.message_edits, not a conversation step).
        # The step list is thus append-only in spirit -- seed + one observation per turn + agent
        # responses -- so a persistent compaction that shrinks/reorders the buffer never perturbs
        # it; the compaction is recorded separately at LLMQuerySend. (Replaces a prefix/suffix buffer diff,
        # which was an unprincipled heuristic -- #565 review.)
        # Message steps are stamped with the ENTER event's emission time rather than a
        # per-hook clock read: one authoritative clock per event, so every observer of
        # this turn records identical times. Event stamps are aware UTC already.
        now = event.timestamp.isoformat()
        state.edits_anchor_step = None
        # The turn's edits hang on the LAST message emitted this enter -- the simplest rule and the
        # least surprising mental model: the edits are applied to the buffer as it stands right after
        # that last message, so they belong immediately after it. It is also chronologically correct
        # when a turn emits more than one observation (no in-tree protocol does today, but
        # ``render_observation`` returns a list and an out-of-tree one may): the edits act on the
        # buffer after ALL of them, so anchoring on the last renders them after all, whereas the
        # first-observation rule would wrongly slot them in between. On iteration 1 the last emitted
        # step is a seed, which works the same way.
        last_emitted_step: dict[str, Any] | None = None
        for position, msg in enumerate(event.messages):
            if not isinstance(msg, dict):
                continue
            prov = provenance_of(msg)
            if prov is None:
                continue  # an un-stamped, hand-built message has no identity -- skip it
            if prov.kind is MessageKind.ASSISTANT:
                # The agent step was emitted at the previous exit, before this assistant message
                # was stamped, so fill its provenance/id -- and its position -- in now, so a drop that
                # names this message's id can resolve to its step. (An assistant always carries an
                # int iteration; the guard is for the type-checker.) NOTE the resulting cross-frame
                # pairing: extra.iteration is the assistant's turn N (stamped at exit), but the
                # backfilled position is its slot at THIS enter (N+1) -- the first snapshot it appears
                # in, since the turn-N assistant is appended after turn N's query. `setdefault` pins
                # that first-seen slot against the assistant's re-appearance in later enters.
                if prov.iteration is not None:
                    agent_step = state.agent_step_by_iteration.get(prov.iteration)
                    if agent_step is not None:
                        agent_extra = agent_step.setdefault("extra", {})
                        agent_extra.setdefault("provenance", prov.to_dict())
                        agent_extra.setdefault("position", position)
                continue
            if prov.kind is MessageKind.ADDED:
                # Not a conversation step of its own -- it is in ``extra.message_edits`` at send,
                # written before it had an id. This is the first snapshot it appears in stamped,
                # so fill that id into the record now, the way the assistant's is filled in above.
                # Matched by IDENTITY: the fold inserts the add's dicts into the buffer by
                # reference, so the object here is the one the record serialized.
                #
                # Assign rather than ``setdefault``: the record may ALREADY carry a provenance,
                # because ``_serialize_message`` copies whatever stamp the handed-in dict had at
                # send time, and a hook that persistently re-adds a message the buffer already
                # holds hands in a stamped one. That stamp is the earlier message's, so deferring
                # to it files this add under another message's id and leaves its own id nowhere in
                # the trace -- two distinct buffer entries claiming one id, which is exactly what
                # the id is supposed to rule out.
                #
                # Removing the entries once stamped is what replaces ``setdefault``'s guard against
                # the message re-appearing at later enters: there is nothing left to re-stamp. It
                # also handles one dict appearing in several records (or twice in one), which a
                # stamp-the-first-match-and-break loop silently left half-filled.
                if prov.iteration is not None:
                    pending = state.pending_adds_by_iteration.get(prov.iteration)
                    if pending is not None:
                        unstamped = [
                            (source, serialized)
                            for source, serialized in pending
                            if source is not msg
                        ]
                        for source, serialized in pending:
                            if source is msg:
                                serialized["provenance"] = prov.to_dict()
                        if unstamped:
                            state.pending_adds_by_iteration[prov.iteration] = unstamped
                        else:
                            del state.pending_adds_by_iteration[prov.iteration]
                continue
            if prov.id in state.emitted_ids:
                continue
            # ``position`` is the message's slot in this enter snapshot, captured the first time it is
            # seen (right after it entered the buffer). A live buffer position, so it drifts as later
            # turns add/drop around it; provenance's id + iteration are the stable identity.
            #
            # ``iteration`` is the message's own provenance iteration -- the turn it was PRODUCED, not
            # the enter it is first shown. These differ for an OBSERVATION (produced at turn N-1, first
            # emitted at enter N), so stamping ``current_iteration`` here would tag it N, off by one;
            # ``prov.iteration`` keeps it N-1, matching how the assistant step (stamped at its own
            # exit) already agrees with its provenance. It is ``None`` for a seed, which belongs to no
            # REPL iteration (it exists before the loop) -- an honest absence, not a turn number.
            extra: dict[str, Any] = {
                "provenance": prov.to_dict(),
                "position": position,
                "iteration": prov.iteration,
            }
            step_data: dict[str, Any] = {
                "step_id": state.next_step_id(),
                "source": msg.get("role", "user"),
                "message": msg.get("content", ""),
                "timestamp": now,
                "llm_call_count": 0,
                "extra": extra,
            }
            state.steps.append(step_data)
            state.emitted_ids.add(prov.id)
            last_emitted_step = step_data
        state.edits_anchor_step = last_emitted_step
        return []

    def on_llm_query_retry(self, event: LLMQueryRetry) -> list:
        state = self._invokes.get(event.invoke_id)
        if state is not None:
            state.retries.append(
                {
                    "iteration": event.iteration,
                    "attempt_number": event.attempt_number,
                    "exception": _serialize_exception(event.exception),
                    "wait_seconds": event.wait_seconds,
                }
            )
        return []

    def on_llm_query_complete(self, event: LLMQueryComplete) -> list:
        """Capture the raw response, before any Complete-stage transform runs."""
        # The ONLY point where the pre-transform content is observable: ``LLMQueryComplete``
        # fires with the raw response and ``ModifyLLMResponse`` effects compose after it, so by
        # ``LLMQueryExit`` the original is gone. Recording it is what lets a replay supply the
        # producer's own output instead of a turn that already carries the transform — without
        # it, a rerun under the same hook applies the transform twice (#1307 shape 1, made loud
        # by #1308 and fixed at the source here).
        state = self._invokes.get(event.invoke_id)
        if state is not None:
            state.pending_raw_content = event.response.content
            state.pending_raw_seen = True
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list:
        outcome = event.outcome
        # Non-completed arms (#892 outcome union): there is no response, no billed usage, and
        # no agent step to write — record the outcome and its exception instead, so a trace
        # whose turn vanished shows *why* (a failed provider call, an aborted turn) rather
        # than silently ending.
        if not isinstance(outcome, Completed):
            failed_state = self._invokes.get(event.invoke_id)
            if failed_state is not None:
                failed_state.query_failures.append(
                    {
                        "iteration": event.iteration,
                        "outcome": _serialize_outcome(outcome),
                    }
                )
                failed_state.pending_raw_content = None
                failed_state.pending_raw_seen = False
            return []
        response = outcome.result
        response_content = response.content
        state = self._invokes.get(event.invoke_id)
        if not state:
            return []
        # Empty/None content is NOT a reason to skip the turn. This point is only reached on
        # the Completed arm (a failed call took the arm above), so empty content means a
        # real, billed query that returned nothing usable -- a content filter, a reasoning-only
        # response, max_tokens spent on reasoning. Skipping it used to lose the agent step, this
        # turn's edits, AND the call's tokens/cost from the trajectory totals, while
        # ConversationHistory records the response unconditionally -- the one gap that falsified
        # "ATIF records everything ConversationHistory does" on the axis (cost) the rest of this
        # stack means to drive off ATIF (#565 review).

        # Re-read on every query, though it cannot change within one invoke: `Config` is frozen
        # and `Agent` never reassigns it, and `BaseLLM.model_info` is a `cached_property` whose
        # contract is that a backend's model facts are fixed for the object's lifetime. Every
        # write therefore stores the same value, so the unconditional overwrite needs no
        # first-write guard and loses nothing.
        #
        # That constancy is also why no per-step `model_name` is written, though ATIF allows one
        # and lets it override this field: with the model fixed across an invoke, a per-step copy
        # could only repeat this one. It is per *trajectory* rather than per document, though,
        # because a nested invoke resolves its own config -- sibling trajectories in one file can
        # legitimately name different models.
        #
        # Fixed is the *configured* model, which is not always the one that answered: a route can
        # pick its own model server-side (`openrouter/auto`), and nothing here records which.
        # LiteLLM features that would answer a call as another model (`fallbacks`, `model_list`,
        # `context_window_fallback_dict`, and the global `litellm.model_fallbacks`) are refused by
        # `LiteLLM` (see `_REFUSED_QUERY_PARAMS` and `_build_params` in `jaz.llm._litellm`).
        state.model_name = event.config.llm.model_info.name

        prompt_tokens = response.prompt_tokens or 0
        completion_tokens = response.completion_tokens or 0
        # Keep the raw value (None = provider reported no caching) distinct from
        # the summing value (None coerced to 0).
        raw_cached_tokens = response.cached_tokens
        cached_tokens = raw_cached_tokens or 0
        cost = response.cost_usd or 0.0

        state.total_prompt_tokens += prompt_tokens
        state.total_completion_tokens += completion_tokens
        state.total_cached_tokens += cached_tokens
        state.total_cost += cost

        # The agent step's timestamp is the exit's emission time (Event.timestamp —
        # every event carries one; no interval fields exist), aware UTC already.
        ts = event.timestamp.isoformat()

        metrics: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
        }
        # Write an explicit 0 when the model reports caching but had no hit (raw
        # == 0); omit only when caching is unreported (raw is None).
        if raw_cached_tokens is not None:
            metrics["cached_tokens"] = raw_cached_tokens
        remaining_extra = dict(response.extra)
        hoist_atif_metrics(metrics, remaining_extra)
        if remaining_extra:
            metrics["extra"] = remaining_extra
        step: dict[str, Any] = {
            "step_id": state.next_step_id(),
            "source": "agent",
            # Coerced: content is ``str | None``, and an ATIF step's message is a string.
            "message": response_content or "",
            "timestamp": ts,
            "llm_call_count": 1,
            "metrics": metrics,
        }
        # Written ONLY when the transform actually changed the content, which is zero bytes on
        # the overwhelmingly common no-transform path: an absent ``raw_message`` on a document
        # that declares the capability means "this turn was not transformed", and replay can
        # then supply the final safely because the final IS the raw. That reading is sound only
        # because the declaration dates the writer — on a document without it, absent is
        # "unknown" instead, which is the ambiguity ``WRITER_RECORDS`` exists to remove.
        raw_content = state.pending_raw_content
        raw_seen = state.pending_raw_seen
        state.pending_raw_content = None
        state.pending_raw_seen = False
        transformed = raw_seen and raw_content != response_content
        if state.current_iteration is not None or transformed:
            step["extra"] = {}
        if state.current_iteration is not None:
            step["extra"]["iteration"] = state.current_iteration
        if transformed:
            # May be ``None``: a transform that filled in an empty response recorded a raw of
            # ``None``, and replay must supply that rather than the filled-in final.
            step["extra"]["raw_message"] = raw_content
        if state.current_iteration is not None:
            # The assistant message this step reports is stamped only after this exit, so its
            # provenance/id is filled in at the next enter (see on_llm_query_enter). Track the
            # step by iteration so enter can find it.
            # Consequence: the FINAL iteration's agent step keeps no provenance, since no enter
            # follows it. Accepted, not overlooked -- the final assistant is never dropped, so no
            # drop ever has to resolve its id, and ConversationHistory records no assistant ids
            # either. Filling it at on_invoke_exit is not an option: InvokeExit carries only the
            # result, never the message buffer, so the stamped assistant is unreachable there.
            state.agent_step_by_iteration[state.current_iteration] = step
        state.steps.append(step)
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list:
        # Every arm is recorded the same way, non-completed ones included: the per-invoke
        # state is kept, so teardown still writes everything traced up to a failure, now
        # labelled with how it ended.
        state = self._invokes.get(event.invoke_id)
        if state is not None:
            state.outcome = _serialize_outcome(event.outcome)
        return []

    def _build_atif(self, state: _InvokeState) -> dict[str, Any]:
        """Build an ATIF document from an invoke state."""
        atif: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "agent": {
                "name": "jaz",
                "version": __import__("jaz").__version__,
            },
            "steps": state.steps,
            "final_metrics": {
                "total_prompt_tokens": state.total_prompt_tokens,
                "total_completion_tokens": state.total_completion_tokens,
                "total_cached_tokens": state.total_cached_tokens,
                "total_cost_usd": round(state.total_cost, 6),
                "total_steps": state.total_repl_iterations,
            },
            "notes": (
                "total_steps is the number of REPL iterations (think-act-observe cycles), "
                "not len(steps). Each REPL iteration corresponds to one LLM call."
            ),
        }

        if state.model_name:
            atif["agent"]["model_name"] = state.model_name
        if state.invoke_id:
            atif["trajectory_id"] = state.invoke_id

        # Extra metadata not in core ATIF schema but needed for full coverage. The former
        # `extra["prompt"]` is gone (#538); the prompt lives in `extra["inputs"]` now.
        # Unconditional, unlike every other key here: this is the signal that dates the writer,
        # so a reader must be able to tell "absent because old" from "absent because empty".
        extra: dict[str, Any] = {"writer_records": list(WRITER_RECORDS)}
        if state.depth is not None:
            extra["depth"] = state.depth
        if state.parent_repl_iteration is not None:
            # Which REPL iteration of the parent invoke spawned this subagent.
            extra["parent_repl_iteration"] = state.parent_repl_iteration
        if state.inputs:
            # Serialize inputs: try JSON-serializable value, fall back to repr
            # TODO: revisit — should we use the serialization shown to the agent?
            extra["inputs"] = _serialize_namespace(state.inputs)
        if state.input_edits:
            # ``adds`` names ``inputs`` entries injected by hooks (the values live in
            # ``inputs``); ``drops`` names caller inputs hooks un-passed, absent from
            # ``inputs`` by design rather than by omission.
            extra["input_edits"] = dict(state.input_edits)
        if state.recursion_disabled:
            # Written only when true: a run where nothing withheld recursion carries no key,
            # the same rule the edit lists follow.
            extra["recursion_disabled"] = True
        if state.repl_history_withheld:
            extra["repl_history_withheld"] = True
        if state.scope:
            # Resolved ambient jaz.scope, logged as its own provenance channel (#727).
            extra["scope"] = _serialize_namespace(state.scope)
        if state.hooks:
            # Governance active for this invoke: the resolved active hooks' reprs (#727).
            extra["hooks"] = list(state.hooks)
        if state.query_failures:
            # Per-query non-completed closes, keyed by iteration — the turns whose agent
            # step is absent above, with the reason why.
            extra["query_failures"] = list(state.query_failures)
        if state.retries:
            # Per-attempt LLM retry records, keyed by iteration — absent entirely on a
            # retry-free run, so the common document shape is unchanged.
            extra["retries"] = list(state.retries)
        if state.outcome is not None:
            # How the invoke ended, shaped like ``InvokeExit.outcome``; absent only when
            # the trace was written before any InvokeExit arrived.
            extra["outcome"] = state.outcome
        # Always non-empty now (``writer_records``), but kept as a guard rather than an
        # unconditional assignment so removing that key cannot silently emit ``"extra": {}``.
        if extra:
            atif["extra"] = extra

        # Nested invokes: store in top-level subagent_trajectories[]
        if state.children:
            atif["subagent_trajectories"] = [
                self._build_atif(child) for child in state.children
            ]

        return atif

    def get_trajectories(self) -> list[dict[str, Any]]:
        """Return the ATIF trajectory documents traced so far, one per top-level invoke.

        Built fresh from the recorded state on each call, so it can be read mid-run or
        after the scope ends; nested invokes appear inside their parent's document.
        """
        return [self._build_atif(state) for state in self._root_invokes]

    def _write_output(self) -> None:
        """Write ATIF JSON to the output file."""
        if not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        trajectories = self.get_trajectories()
        # Single root → write as object; multiple → write as array
        data = trajectories[0] if len(trajectories) == 1 else trajectories
        self.output_path.write_text(json.dumps(data, indent=self.indent))

    def setup(self) -> None:
        """Prepare for a run: with an ``output_path``, refuse reuse; without one, start clean.

        Raises :class:`~jaz.exceptions.HookActivationError` when this instance has already been
        activated and ``output_path`` is set, because the second run's file would replace the
        first's.
        """
        # The policy splits on `output_path` because the two configurations fail differently, and
        # only this class knows which one it is in.
        #
        # With a path, `teardown` writes the whole document to it. A second activation writes the
        # same path again, so run 1's trace is gone from disk — measured, not assumed. Clearing
        # in-memory state does not help: the loss is on the filesystem, and by the time the second
        # run starts there is nothing left to protect. Refusing is the only thing that keeps it.
        #
        # Without a path, the in-memory read IS the output, and refusing reuse would buy nothing
        # while breaking a documented affordance — see the clearing rationale below.
        if self.output_path is not None:
            self.forbid_reactivation()
        # Clearing at SETUP, not teardown. Sequential re-activation of one instance is legal —
        # the dispatcher rejects only CONCURRENT activation — and without a clear somewhere the
        # first run's root stayed in `_root_invokes` and was written into the second run's
        # document. Doing it at teardown would close that leak and break a documented read:
        # `get_trajectories` promises the documents can be read "after the scope ends", and with
        # `output_path=None` (the default) that in-memory read is the only way to get them at
        # all, so a post-scope call would return []. setup runs once per activation on BOTH
        # paths (`with` and positional/local_hooks), so the second run still starts clean while
        # the first run's documents survive its own scope. The clearing used to live in a public
        # `reset()` with no caller anywhere.
        self._invokes.clear()
        self._root_invokes.clear()

    def teardown(self, exc: BaseException | None = None) -> None:
        """Write the output file when the hook's scope ends."""
        # teardown, NOT __exit__: it runs on BOTH activation paths — `with` (via the base
        # __exit__) and positional/local_hooks (called directly by _activate_local_hooks,
        # which never touches __exit__). The write used to live in a __exit__ override,
        # so a positionally-passed TrajectoryRecorder(output_path=...) silently never wrote its
        # file even though the docstring documents positional use.
        self._write_output()
