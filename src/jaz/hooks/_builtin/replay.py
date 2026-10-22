"""TrajectoryReplay: resume agent runs from a saved ATIF trace.

Reads LLM responses from a saved ATIF trace and replays them as
``SupplyLLMResponse`` effects. Once all saved responses are exhausted, the hook
becomes a no-op and live LLM calls resume seamlessly.

An ATIF trace is the canonical replay/cost format. It carries real per-call
``prompt_tokens`` / ``completion_tokens`` / ``cost_usd`` (no reconstruction),
replays each invoke from its OWN per-invoke queue (nested ``invoke`` calls do
not shift a sibling/parent cursor), AND supports per-invoke divergence detection.

Usage::

    from jaz import invoke
    from jaz.hooks import TrajectoryReplay

    # Real tokens+cost, per-invoke replay
    with TrajectoryReplay("trace.atif.json"):
        result = invoke(task="...")

    # Same, but stop replaying an invoke as soon as its live prompt, assistant turn or
    # REPL output diverges from the saved trace; earlier calls and other invokes still
    # replay free.
    with TrajectoryReplay("trace.atif.json", detect_divergence=True):
        result = invoke(task="...")
"""

# ATIF as the sole replay source is #813's canonical-record decision; the per-invoke
# queue/divergence design below is its implementation. The trace path IS the
# constructor argument (no from_atif factory): with a single source format there is
# exactly one way to build the hook, and the former factory's only job was hiding a
# private tree type that has no business in a public signature (#1167 review).

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaz._provenance import MessageKind, provenance_of
from jaz.exceptions import HookActivationError
from jaz.hooks._dispatcher import Hook
from jaz.hooks.effects import SupplyLLMResponse
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
    LLMQuerySend,
    REPLExecExit,
)
from jaz.hooks.events._base import Completed
from jaz.hooks.events._llm_query import ResolvedMessageEdits
from jaz.repl._types import Continue

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Message-edit signatures: the comparable form of one query's edits
# ------------------------------------------------------------------

#: One dropped message: ``(index, persistent)``. ``index`` is ``None`` for a record whose
#: trace does not carry a usable one (see ``_edit_index``).
type _DropSig = tuple[int | None, bool]
#: One ``AddMessages`` edit: ``(index, persistent, ((role, content), ...))``.
type _AddSig = tuple[int | None, bool, tuple[tuple[str, str], ...]]
#: One query's message edits reduced to the fields divergence compares, in the order
#: ``_canonical`` puts them in.
type _EditSignature = tuple[tuple[_DropSig, ...], tuple[_AddSig, ...]]

#: The signature of a query that composed no edits. Distinct from ``None`` (see
#: ``_InvokeNode.expected_edits``): this one is compared, ``None`` is skipped.
_NO_EDITS: _EditSignature = ((), ())

# What participates, and why each field earns its place:
#
#   * an ADD's ``role``/``content`` — the message the model is shown, the whole point;
#   * an ADD's ``index`` — WHERE it lands changes the prompt as much as whether it lands;
#   * ``persistent`` (both kinds) — a transient add is gone next turn, a persistent one is
#     not, so the same content at the same slot still means two different runs;
#   * a DROP's ``index`` — the only thing that identifies the dropped message across runs.
#
# What deliberately does NOT participate:
#
#   * a DROP's ``id``. ATIF names a dropped message by its provenance id, which is a fresh
#     uuid4 per message per RUN (``MessageProvenance.id``) — "not a stable key for diffing
#     trace JSON across runs" in its own words. Comparing ids would diverge on every turn of
#     every replay. The dropped message's CONTENT is not in the edits block either (it lives
#     in that message's own step), so ``index`` is all the record affords across runs;
#     consequence, stated plainly: dropping a DIFFERENT message from the same slot does not
#     diverge here.
#   * an added message's ``provenance`` sub-object (same per-run id), and every key beyond
#     ``role``/``content``. ``_serialize_message`` records those (``name``, ``tool_calls``,
#     ``cache_control``, …) and ``to_wire_messages`` forwards whatever the backend consumes, so
#     a hook that injects a message differing ONLY in such a key changes the prompt and not the
#     signature. A scope call: role and content are what every protocol puts in front of the
#     model, and widening the tuple would grow the false-positive surface for keys most
#     backends strip.
#   * an add contributing zero messages — it changes no prompt, and the recorded form stamps
#     the index on the messages, so an empty one carries no index to compare. Dropped from
#     both sides so the asymmetry cannot read as a difference.


def _edit_value(value: Any) -> str:
    """Canonicalize one message field for comparison against the trace's JSON form.

    Mirrors ``TrajectoryRecorder._serialize_message``: a ``str`` unchanged, otherwise JSON where the
    value is JSON-able and its ``repr`` where it is not — so the live object and the string the
    trace holds for it compare equal. ``sort_keys`` normalizes dict ordering, which a JSON round
    trip does not.
    """
    # The ``str`` fast path is not just an optimization, it is the whole common case: every
    # shipped protocol's message content is a ``str``, and ``_serialize_message`` stores a str
    # as a str, so both sides skip the encode symmetrically. Without it this JSON-encodes the
    # full content of every added message on every query — O(bytes a hook injected), which a
    # compaction summary makes non-trivial.
    if isinstance(value, str):
        return value
    # A container holding a non-serializable item is the one case the two sides can still
    # disagree (the trace reprs the whole container, this reprs it too — matching — but a
    # future serializer using ``default=repr`` would repr only the inner item).
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps(repr(value))


def _edit_index(value: Any) -> int | None:
    """An edit record's resolved slot, or ``None`` where the trace does not carry a usable one.

    ``None`` participates in the comparison as its own value: an unusable index is a fact about
    the record, and two traces that agree on it still compare equal.
    """
    # ``bool`` is an ``int`` subclass, hence the explicit exclusion. Returning ``None`` rather
    # than coercing (``int("3")``) or raising keeps the reducer total over any JSON document —
    # the file is on disk and may have been written by another tool.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _edit_records(
    block: dict[str, Any], key: str, legacy_key: str
) -> list[dict[str, Any]]:
    """One edit sequence of a recorded block under either spelling, malformed entries dropped.

    The current key wins by being PRESENT, not by being non-empty: a block that spells the
    sequence the current way and records nothing under it says "no edits of that kind", which an
    emptiness test would mistake for "not written this way" and answer from the legacy key.
    """
    records = block[key] if key in block else block.get(legacy_key)
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _canonical(drops: Sequence[_DropSig], adds: Sequence[_AddSig]) -> _EditSignature:
    """Put both halves of a signature in the one order the comparison is defined over."""
    # The comparison must be exactly as strict as the prompt it stands for. Drops are a set
    # (union), so their order carries nothing: fully sorted. Adds are stable-sorted by slot
    # only: adds at distinct slots commute, so their relative order carries nothing either,
    # while the order WITHIN a slot is the order the fold applied them (hook order) and does
    # reach the model — the same two hooks composed in the other ``with`` nesting produce a
    # different prompt, and a comparison that sorted within a slot would replay a recorded
    # response against a prompt the model never saw. Sorting here rather than trusting the
    # order the records arrive in keeps the reducer's contract independent of the writer's:
    # a trace written before the dispatcher slot-sorted its records lists adds in composition
    # order, and must still compare equal to an identical live rerun. Cross-slot, it does;
    # within a slot, such a trace recorded composition order while its fold applied content
    # order, so two same-slot adds whose content order differs from hook order compare equal
    # to a live rerun whose prompt differs. Dev-only: no release shipped the ``added`` block
    # with that fold.
    return tuple(sorted(drops, key=_sort_key)), tuple(sorted(adds, key=_slot_key))


def _slot_key(entry: tuple[Any, ...]) -> tuple[bool, int]:
    """Slot-only order over add signatures (stable, so within-slot order is preserved)."""
    index = entry[0]
    return (index is None, 0 if index is None else index)


def _sort_key(entry: tuple[Any, ...]) -> tuple[Any, ...]:
    """Total order over signature entries whose leading index may be ``None``."""
    # A bare ``sorted()`` would raise on a malformed trace, where one record's index is ``None``
    # and another's is an int — the one input this module promises not to die on.
    return (*_slot_key(entry), *entry[1:])


def _live_edit_signature(edits: ResolvedMessageEdits) -> _EditSignature:
    """Reduce a live query's ``LLMQuerySend.message_edits`` to its comparable signature."""
    drops = [(d.position, d.persistent) for d in edits.drops]
    adds: list[_AddSig] = []
    for add in edits.adds:
        messages = tuple(
            (_edit_value(m.get("role")), _edit_value(m.get("content")))
            for m in add.messages
        )
        if not messages:
            continue
        adds.append((add.position, add.persistent, messages))
    return _canonical(drops, adds)


def _recorded_edit_signature(block: Any) -> _EditSignature:
    """Reduce a trace's ``extra.message_edits`` block to the same signature
    ``_live_edit_signature`` produces. ``None`` (no block on this turn) reduces to ``_NO_EDITS``.

    Reads either spelling of the block, since a trace is a document that outlives the writer
    that produced it: ``drops`` / ``adds`` with a record ``position`` as written now, and
    ``dropped`` / ``added`` with a per-message ``index`` as written before they were named
    after ``LLMQuerySend.message_edits``.

    Defensive about shape throughout: a trace is an on-disk document that may have been
    written by another tool, and a malformed block must not raise mid-replay.
    """
    if not isinstance(block, dict):
        return _NO_EDITS
    drops = [
        # ``position`` is the record's field; ``index`` was the pre-rename spelling of it.
        (_edit_index(d.get("position", d.get("index"))), bool(d.get("persistent")))
        for d in _edit_records(block, "drops", "dropped")
    ]
    adds: list[_AddSig] = []
    for record in _edit_records(block, "adds", "added"):
        raw = [m for m in (record.get("messages") or []) if isinstance(m, dict)]
        if not raw:
            continue
        messages = tuple(
            (_edit_value(m.get("role")), _edit_value(m.get("content"))) for m in raw
        )
        # The slot is the record's ``position``. A pre-rename trace stamped it on each message
        # of the record instead, all with the same value, so the first message carries it.
        position = record.get("position", raw[0].get("index"))
        adds.append((_edit_index(position), bool(record.get("persistent")), messages))
    return _canonical(drops, adds)


@dataclass
class _InvokeNode:
    """A node in the per-invoke response tree.

    Each node holds the ordered list of LLM responses that belong to THIS
    invoke's own iterations (i.e. LLM calls made directly by this agent, not
    by nested sub-invokes).  ``children`` is a FIFO queue of child nodes
    representing nested ``jaz.invoke`` calls spawned during this invoke's
    execution; each child is popped when the corresponding ``InvokeEnter``
    event fires during replay.
    """

    responses: list[_SavedResponse]
    children: list[_InvokeNode]  # ordered by source-execution order
    cursor: int = 0
    child_cursor: int = 0
    # The observation that FOLLOWED each response's exec, aligned 1:1 with ``responses`` — the
    # LIST of that observation's message contents, or ``None`` where the response had no following
    # observation (e.g. a terminal response). A list rather than a string because
    # ``render_observation`` returns one (``CodeOnlyProtocol`` always returns exactly one message,
    # but the seam admits several — one ``tool`` message per parallel call, say). Compared
    # message-by-message against the live render (#1055), so a difference in the NUMBER of messages
    # is caught structurally rather than hidden by concatenation. Populated by
    # ``_load_invoke_roots``; default via default_factory so an ``_InvokeNode(...)`` built without
    # it (e.g. the progress tests) keeps working.
    expected_outputs: list[list[str] | None] = field(default_factory=list)
    # The message edits the trace recorded for each response's OWN query, aligned 1:1 with
    # ``responses`` — the recorded message-edits block reduced to its comparable fields (see
    # ``_recorded_edit_signature``), or ``None`` where the trace's writer did not record edits at
    # all (the back-compat rule; see ``_trace_records_edits``). ``_NO_EDITS`` — the empty
    # signature — is the *positive* statement "this turn had no edits", which is what a v1.7+
    # trace's missing block means, and is therefore compared rather than skipped. Populated by
    # ``_load_invoke_roots``; default via default_factory so an ``_InvokeNode(...)`` built without
    # it (e.g. the progress tests) keeps working, exactly as ``expected_outputs`` does.
    expected_edits: list[_EditSignature | None] = field(default_factory=list)
    # The seed prompt the trace recorded for this invoke, as ``(role, content)`` per message, or
    # ``None`` where the trace cannot say (a hand-assembled node, or a writer that did not stamp
    # the seed's provenance) and it is not compared. Trace data, like the two lists above, so
    # ``rewind`` leaves it alone.
    expected_seed: list[tuple[Any, Any]] | None = None
    # Per-invoke divergence latch: once True, this invoke's remaining LLM calls go live.
    diverged: bool = False
    # Which response this node SUPPLIED for the query now in flight, or ``None`` when that query
    # went live (this node exhausted, diverged, or absent). Written on every ``LLMQuerySend``
    # that reaches the node and read once at the ``LLMQueryExit`` that closes the same query —
    # the next event on the node whenever this invoke's queries are sequential, which is the
    # regime the whole stack design assumes (#1310).
    # It exists so the assistant-turn detector compares against the response the query was
    # actually ANSWERED with: ``cursor - 1`` would name the previous turn's response on a query
    # that went live, and report that as a divergence.
    supplied_index: int | None = None

    def __post_init__(self) -> None:
        # Pad both expectation lists up to ``responses`` so every served response HAS a slot,
        # and the detectors can index by cursor without a bounds branch. A hand-assembled node
        # (``_from_invoke_tree``, the progress tests) sets neither list, and a bounds branch
        # there would collapse "this node carries no expectations" into the same silent skip as
        # a real out-of-range index — the three states have to stay distinguishable:
        # absent-by-construction (padded to ``None`` here), absent-because-the-writer-did-not-
        # record-it (``None``, skipped, the back-compat rule), and recorded-as-empty
        # (``_NO_EDITS``, compared). Trace-built nodes never pad: ``_build_invoke_tree_from_atif``
        # raises on a length mismatch before constructing, so padding cannot mask a builder bug.
        for expectations in (self.expected_outputs, self.expected_edits):
            expectations.extend([None] * (len(self.responses) - len(expectations)))

    @property
    def is_exhausted(self) -> bool:
        return self.cursor >= len(self.responses)

    def next(self) -> _SavedResponse | None:
        if self.is_exhausted:
            return None
        r = self.responses[self.cursor]
        self.cursor += 1
        return r

    def next_child(self) -> _InvokeNode | None:
        if self.child_cursor >= len(self.children):
            return None
        child = self.children[self.child_cursor]
        self.child_cursor += 1
        return child

    def rewind(self) -> None:
        """Reset this subtree's run state, so a replay starts from its first response again."""
        # Every field a run writes is reset here, and nothing else: the response lists and the
        # expectations are the trace, loaded once. In place rather than by deep-copying a pristine
        # tree, which would buy a second tree to keep consistent with the first. test_replay's
        # rewind test compares a played-then-rewound tree against a never-played one, which catches
        # a missing line only for a field its scripted run writes: a field written on another path
        # (assistant-turn divergence, say) needs that path added to the test.
        self.cursor = 0
        self.child_cursor = 0
        self.diverged = False
        self.supplied_index = None
        for child in self.children:
            child.rewind()


@dataclass
class _SavedResponse:
    """A single saved LLM response for replay."""

    content: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    # The PRE-transform content, when the recording's writer saw a Complete-stage
    # ``ModifyLLMResponse`` change this turn.
    #
    # Guarded by its own flag rather than by ``raw_content is None``, mirroring the writer's
    # two-field capture: a transform that FILLS IN an empty response records a raw of ``None``,
    # and that turn must still supply the raw. Collapsing the two would supply the filled-in
    # final and let the rerun's transform run over it again — a spurious divergence on a
    # faithful rerun. ``has_raw=False`` covers both the untransformed turn and the trace whose
    # writer never recorded the field; those two are separated by the writer declaration at load
    # time, not here (see ``_writer_records``).
    raw_content: str | None = None
    has_raw: bool = False

    @property
    def supplied_content(self) -> str | None:
        """What replay serves in the producer's position for this turn."""
        # The raw when we have it, so the rerun's own transform re-derives ``content`` rather
        # than compounding on top of it. ``_response_diverges`` still compares the committed turn
        # against ``content`` — the recorded FINAL — which is what makes a faithful rerun under
        # the same transform compare equal instead of diverging on turn 1.
        return self.raw_content if self.has_raw else self.content

    # A CHIMERA WHEN THE RAW IS SERVED, and deliberately so: the content is PRE-transform while
    # ``prompt_tokens``/``completion_tokens``/``cost`` are the POST-transform values the trace
    # recorded, since those are the only ones it has. It is the right trade — the metrics are what
    # the run was actually billed, which is what a replay's cost accounting must reproduce, and a
    # transform that rewrites content rarely re-prices. It does mean a hook reading the served
    # response sees a combination no live call ever produced. ``BudgetPool``'s uncosted-turn
    # backstop is unaffected: it reads the ``LLMResponse`` on the exit event rather than this
    # stand-in, and looks only at ``cost_usd``, which a served turn carries as recorded.


class TrajectoryReplay(Hook):
    """Replay saved LLM responses from an ATIF trace instead of calling the model.

    ATIF (Agent Trajectory Interchange Format) is the standard trace format JAZ reads and
    writes — here, the saved run that responses are replayed from.

    Each turn consumes the next saved response for the active invoke in order. Once an
    invoke's responses run out, its live LLM calls resume.

    The trace (as written by :class:`TrajectoryRecorder`) is the canonical replay/cost format.
    Each invoke replays from its own per-invoke queue — one ``TrajectoryReplay`` covers the
    whole invoke tree, and a nested ``invoke`` never shifts a sibling's or parent's
    cursor. The per-call tokens and cost are the *real* metrics recorded in the trace
    (``prompt_tokens`` / ``completion_tokens`` / ``cost_usd``) — no reconstruction — so
    replayed calls report exact token totals as well as exact dollar cost.

    With ``detect_divergence=True``, replay stops for an invoke as soon as it stops
    matching, on any of four comparisons: before an invoke's first query, its starting prompt
    is compared to the saved one; before each query, the message edits hooks
    composed for it are compared to the saved ones; as each query closes, the assistant turn
    the run is about to commit is compared to the saved one; after each REPL exec, the output
    is compared to the saved observation. On the first difference that invoke *diverges* —
    every LLM call from then on is live, including ones that would still have matched, and
    a differing prompt takes the *current* query live rather than answering it from the
    trace. Divergence is per-invoke, so a diverged invoke never desynchronizes a
    still-replaying sibling or parent. Use this to re-run an old trace under a change:
    everything before the first difference replays for free, and everything after it is
    paid for.

    Output comparison is not byte-for-byte. The live output is abbreviated the way the
    original run abbreviated it, and REPL session IDs and traceback line numbers are
    normalized away, so edits elsewhere in the source do not read as divergence. It is
    **not** stable across a change to the truncation format: after such a change, every
    saved iteration whose output was truncated reports a divergence even on an otherwise
    identical rerun. Edit and assistant-turn comparison have no such exposure — the trace
    stores an added message's role and content, and the assistant turn, verbatim, and all are
    compared as values.

    The starting-prompt comparison covers everything the prompt shows, including each input's
    rendering, so it diverges on an input whose text differs between runs, such as a fresh temp
    path, even when nothing else changed. It cannot see past what the prompt shows: an input too
    long for the prompt is shown abbreviated, and a difference only in the part left out is
    caught only if it changes a REPL output.

    Two limits worth knowing. A trace written before message edits were recorded is detected
    and its edit comparison skipped entirely (the other checks are unaffected), and one written
    before the starting prompt was stamped skips that comparison the same way.

    An invoke launched by a turn of its parent that did not replay calls the model live as well,
    since the code launching it is not the code that launched the recorded one. That covers a
    parent turn that went live, in either mode, and with ``detect_divergence`` one that was
    served but diverged as its query closed.

    A content-changing ``ModifyLLMResponse`` is handled, but only for a trace whose writer
    recorded the *pre*-transform content — traces written by this version and later, which
    declare ``raw_message`` in ``extra.writer_records``. On those, replay supplies the raw turn,
    the rerun's own transform re-derives the recorded one, and a rerun under the same hook
    replays free with no divergence; a transform *added* or *dropped* since the recording does
    differ, and is reported. On an OLDER trace the saved response is the post-transform turn and
    there is no raw to supply, so a content-changing transform on the rerun re-applies itself to
    a turn that already carries it, the committed turn differs, and the invoke diverges on its
    FIRST turn — one free response per invoke and live calls after it, even when nothing about
    the run has changed. That one free turn is itself the doubly-transformed one and divergence
    does not undo it (detection happens as the query closes, by which point the turn is
    committed), so such a rerun keeps a first turn matching neither the trace nor a live run.
    Neither comparison sees a transform that changes only tokens or cost, on a trace of any
    vintage, so a re-pricing hook silently drifts the run's cost totals away from the trace's.

    Under ``with`` the replay covers invokes nested inside. Passed positionally, only that
    invoke replays; this hook supplies nothing to the invokes it nests, so they call the
    model live unless something else replays them.

    A trace of several top-level invokes, which :class:`TrajectoryRecorder` writes when its scope
    ran more than one, replays them in the order they were recorded: each top-level ``invoke`` in
    the scope takes the next one. A top-level invoke beyond the recorded ones calls the model live
    and logs a warning. Recordings are matched to invokes by order, which assumes the scope re-runs
    the recorded invokes one after another in the order they ran -- what a script resuming its own
    interrupted run does.

    One instance can be activated again once its scope has ended, and each activation replays the
    trace from its start; :attr:`is_exhausted` and :attr:`replayed_response_count` describe the
    most recent run.

    Args:
        path: Path to an ATIF ``.atif.json`` trace (as written by :class:`TrajectoryRecorder`).
        detect_divergence: When ``True``, each invoke's starting prompt is compared to the saved
            one, and each saved response is paired with the message edits composed for its
            query, with the assistant turn the query committed, and with the REPL observation
            that followed its exec; a differing starting prompt, the first query whose edits or
            committed turn differ, or the first exec whose output differs marks THAT invoke
            diverged, switching it (only it) to live LLM for the rest of the run.

    Raises:
        ValueError: If the trace holds no trajectory, or if a trajectory's recorded observations
            or message edits are not 1:1 with its responses (a misaligned trace cannot be
            replayed, since the divergence checks index them by response).
    """

    # Intended use (user call): resuming an interrupted run by re-running the same script on the
    # same task, as an eval harness does. That is why recordings are matched to top-level invokes
    # by order alone, with no check of each invoke's inputs: in that use the order cannot differ,
    # and the one real deviation, the resumed run going further than the recording, is handled by
    # going live with a warning. A check would only guard other uses, at the price of false alarms
    # (the recorded inputs are a capped repr). Concurrent top-level invokes fall outside the same
    # assumption (#1310).
    #
    # Sole authority for the model's turn: two active Replays would both try to author every
    # response, so the seed-time check rejects that pairing (see Hook.reserves_llm_response) —
    # one TrajectoryReplay owns the per-invoke queues.
    reserves_llm_response = True

    # Mechanics: overrides at ``LLMQuerySend`` via ``SupplyLLMResponse`` — the supply
    # boundary, where the committed post-edit message list is in view — serving from the
    # active invoke's per-invoke queue (top of ``_invoke_stack``); divergence detection runs at
    # three points across the turn — the committed input's edits at ``LLMQuerySend`` (before
    # serving, so a differing prompt takes THAT query live), the settled assistant turn at
    # ``LLMQueryExit``, and the exec's output at ``REPLExecExit`` — and each latches
    # ``_InvokeNode.diverged`` per invoke, so a diverged invoke never desynchronizes a
    # still-replaying sibling or parent. What the three cover, and what they do not, is laid
    # out below ``_response_diverges``.
    #
    # SEQUENTIAL INVOKES ASSUMED. ``_invoke_stack`` is one stack whose TOP is "the active
    # invoke", which holds only while invokes nest strictly in time. ``ainvoke`` advertises
    # concurrent execution via ``asyncio.gather``, and under it the top at any event need not be
    # the invoke that raised it — so a query can be served from a sibling's queue and a
    # difference in one invoke can latch another's ``diverged``. Pre-existing (mis-serving was
    # already possible), tracked in #1310; the assistant-turn detector inherits it rather than
    # introducing it.
    #
    # Cursor semantics are unchanged by the Enter→Send migration: LLMQuerySend fires exactly
    # once per query, on the dispatched AND the supplied path alike (the input commit is real
    # on both), so ``node.cursor`` still advances once per query and all three detectors' indices
    # into the node stay aligned — ``cursor`` for edits (read BEFORE the advance), ``cursor - 1``
    # for observations (read after it), and ``supplied_index`` for the assistant turn, which is
    # latched at the serve rather than derived from the cursor at all: on a query that went live
    # the cursor points at a response that query did not answer (see ``_InvokeNode``'s field
    # comment). Pinned by test_replay's cursor-alignment test.

    def __init__(
        self,
        path: str | Path,
        *,
        detect_divergence: bool = False,
    ) -> None:
        # Args are documented in the class docstring (the shipped reference page), not
        # duplicated here — the TrajectoryRecorder pattern.
        self._init_from_roots(_load_invoke_roots(path), detect_divergence)

    @classmethod
    def _from_invoke_tree(
        cls,
        invoke_tree: _InvokeNode,
        *,
        detect_divergence: bool = False,
    ) -> TrajectoryReplay:
        """Test seam: build from a hand-assembled response tree, no trace file involved.

        Private on purpose — the public constructor takes the trace path, so the
        private tree type never appears in a public signature.
        """
        self = cls.__new__(cls)
        self._init_from_roots([invoke_tree], detect_divergence)
        return self

    def _init_from_roots(
        self, roots: list[_InvokeNode], detect_divergence: bool
    ) -> None:
        # Latch so we warn at most once per hook lifetime when a replayed response has
        # None completion/total tokens. ATIF carries real per-call metrics, so this
        # should not fire on a real ATIF trace; kept as a defensive guard.
        self._warned_unknown_completion_tokens = False

        # One root per top-level invoke the recorded scope ran, consumed in order.
        self._roots = roots
        self._detect_divergence = detect_divergence
        # The scope above the roots: a node with no responses of its own whose children are the
        # roots, so a top-level invoke takes the next root exactly as a nested invoke takes its
        # parent's next child. It sits at the bottom of every stack, and holding it here rather
        # than rebuilding it per run is what lets one ``rewind`` reset which root comes next.
        # Without it, a top-level InvokeEnter would treat the first root as its parent and be
        # served the root's first CHILD's queue instead of the root's own.
        self._scope = _InvokeNode(responses=[], children=list(roots))

        # Stack of (_InvokeNode, invoke_id) pairs; the top is the currently-active invoke.
        self._invoke_stack = self._fresh_stack()
        total = self._count_tree_responses(self._scope)
        logger.info(
            f"TrajectoryReplay: {total} saved responses loaded for replay "
            f"(per-invoke queues, no cursor drift)"
        )

    def _fresh_stack(self) -> list[tuple[_InvokeNode | None, str]]:
        """The invoke stack a run starts from: just the scope node."""
        # ``_InvokeNode | None`` because ``on_invoke_enter`` pushes ``None`` for an invoke the
        # trace has no node for — see there. Spelling that in the annotation rather than carrying
        # a ``type: ignore`` on the push is what makes the ``node is None`` guards below read as
        # reachable code instead of defensive noise.
        return [(self._scope, "__sentinel__")]

    def setup(self) -> None:
        """Rewind to the start of the trace, so each activation replays it from its first response.

        Raises:
            HookActivationError: If an invoke on this instance is still open: a run is in
                progress elsewhere, or an earlier run was interrupted before this hook saw its
                end. Construct a new ``TrajectoryReplay`` instead.
        """
        # Per-run state cleared at SETUP, the hooks README's row for state callers read after the
        # scope ends (`is_exhausted`, `replayed_response_count` describe the run just finished).
        # Without it a reused instance keeps the first run's spent cursors, so its second
        # activation finds no node for the top-level invoke and every query goes to the live
        # model, with no warning. Rewinding rather than refusing reuse with
        # `forbid_reactivation`: replay writes nothing, so a second run destroys nothing of the
        # first's, and replaying one trace twice is an ordinary thing to want.
        #
        # Refused while an invoke is open on this replay (user call). The hooks README lets a
        # worker thread activate an instance that is live elsewhere, but this hook exists to
        # resume an interrupted run, one invoke after another, and no use of it activates one
        # instance from two places at once. Rewinding then would reset the running replay
        # mid-flight and send the rest of it to the live model; skipping the rewind would leave
        # both sharing one set of cursors, itself unsound under concurrent invokes (#1310). A
        # stack left open by a lost event -- another hook raising a BaseException while the
        # top-level InvokeEnter or InvokeExit was dispatched, so this hook never saw it -- is
        # refused too. A rewind would recover that one, but ``setup`` cannot tell it apart from
        # a run in progress, so the cost is that the instance has to be replaced. Read off the
        # stack rather than a count of open activations: the stack unwinds through InvokeExit
        # even when the run raises, while a count paired with teardown drifts whenever teardown
        # does not run exactly once per setup -- a doubled ``__exit__`` runs it twice.
        if len(self._invoke_stack) != 1:
            raise HookActivationError(
                f"{type(self).__name__}: an invoke on this instance is still open -- either a "
                "run is in progress elsewhere (one instance cannot replay two runs at once) or "
                "an earlier run was interrupted before this hook saw its end. Construct a new "
                f"{type(self).__name__} for this run."
            )
        self._scope.rewind()
        self._invoke_stack = self._fresh_stack()

    @staticmethod
    def _count_tree_responses(node: _InvokeNode) -> int:
        return len(node.responses) + sum(
            TrajectoryReplay._count_tree_responses(c) for c in node.children
        )

    @staticmethod
    def _count_tree_replayed(node: _InvokeNode) -> int:
        """Responses consumed so far in this subtree — the sum of each node's
        own ``cursor`` (its position in its response list)."""
        return node.cursor + sum(
            TrajectoryReplay._count_tree_replayed(c) for c in node.children
        )

    @property
    def is_exhausted(self) -> bool:
        """Whether all saved responses have been replayed."""
        # Exhausted once every node has consumed its responses (computed from the tree cursors).
        return self._count_tree_replayed(self._scope) >= self._count_tree_responses(
            self._scope
        )

    @property
    def replayed_response_count(self) -> int:
        """How many saved responses have been handed back so far.

        Counts individual responses, not passes over the trajectory: a recorded run is a tree of
        invokes, each node holding its own response list, and this sums how far each node's cursor
        has advanced. It never exceeds :attr:`~TrajectoryReplay.saved_response_count`, but it does not have to reach
        it: once the live run diverges from the recording, the diverged node stops advancing and
        the count stalls below the total for the rest of the run.
        """
        return self._count_tree_replayed(self._scope)

    @property
    def saved_response_count(self) -> int:
        """Total number of saved responses this hook will serve.

        The public count used by resume banners; the total lives in the invoke tree.
        """
        return self._count_tree_responses(self._scope)

    # Event handling is split across the typed per-event handlers below rather
    # than a single ``on_any`` catch-all, so the framework router (#691)
    # dispatches each event to the right handler with no match/isinstance
    # boilerplate.

    def on_invoke_enter(self, event: InvokeEnter) -> list:
        # Pop the next child from the parent node and push it as the new active frame. If
        # the parent has no more children (e.g., the replay had fewer invokes than the live
        # run), push None so LLM calls fall through to live.
        # ``child`` is ``None`` in two situations, and both are ordinary rather than defensive:
        # the live run opened MORE nested invokes at this level than the trace recorded (the
        # parent's children are used up), or this invoke is already inside such an unrecorded
        # one, whose node is itself ``None``. The ``None`` propagates down the subtree, so every
        # invoke under an unrecorded one goes live too, which is the only sound thing to do --
        # there is no queue to serve it from.
        if (
            len(self._invoke_stack) > 1
            and event.parent_invoke_id != self._invoke_stack[-1][1]
        ):
            # The innermost open frame is not this invoke's parent, so the stack is out of step
            # with the run: an earlier enter or exit never reached this hook (another hook raised
            # a BaseException while it was dispatched), or invokes are running concurrently
            # (#1310). Either way responses may now be served from the wrong queue, or none.
            logger.warning(
                f"TrajectoryReplay: invoke {event.invoke_id[:8]}… entered, but the replay's "
                f"innermost open invoke is not its parent; its position in the trace is stale "
                f"and responses may come from the wrong place or the live model"
            )
        parent_node = self._invoke_stack[-1][0] if self._invoke_stack else None
        if (
            parent_node is not None
            and parent_node is not self._scope
            and (parent_node.supplied_index is None or parent_node.diverged)
        ):
            # The parent turn running this invoke's call was not served from the trace -- it went
            # live, or its committed turn already differs from the recorded one -- so the code
            # making the call is not the recorded code, and its recorded child answered some other
            # call. The child goes live with its parent rather than replaying those answers.
            logger.info(
                f"TrajectoryReplay: invoke {event.invoke_id[:8]}… was launched by a live turn "
                f"of its parent; it calls the live model"
            )
            child = None
        else:
            child = parent_node.next_child() if parent_node is not None else None
        if child is None and len(self._invoke_stack) == 1:
            # A top-level invoke with nothing to replay -- the scope has run more of them than the
            # trace recorded -- calls the model live for its whole run, which is the outcome this
            # hook exists to avoid, so it is said out loud rather than at the info level nested
            # misses get.
            logger.warning(
                f"TrajectoryReplay: no recorded trajectory left for top-level invoke "
                f"{event.invoke_id[:8]}…; it calls the live model"
            )
        self._invoke_stack.append((child, event.invoke_id))
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list:
        # Pop down to this invoke's frame, not only when it is on top: frames above it belong to
        # nested invokes whose exits never arrived, and leaving them would hold the stack open
        # after the run, so every later activation would be refused.
        if any(invoke_id == event.invoke_id for _, invoke_id in self._invoke_stack[1:]):
            while self._invoke_stack[-1][1] != event.invoke_id:
                self._invoke_stack.pop()
            node, _ = self._invoke_stack.pop()
            if node is not None and not node.is_exhausted:
                logger.info(
                    f"TrajectoryReplay: invoke {event.invoke_id[:8]}… exited with "
                    f"{len(node.responses) - node.cursor} response(s) "
                    f"unused in its queue (live run ran fewer iterations "
                    f"than source)"
                )
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list:
        # The seed is compared at the invoke's first query, before anything is served: a seed
        # the recording never saw makes every recorded response an answer to a different prompt.
        # At Enter rather than Send because Enter's list is the buffer the recorder captured the
        # seed from, before any hook's edits, which ``_edits_diverge`` already covers at Send.
        if not self._detect_divergence or not self._invoke_stack:
            return []
        node, _ = self._invoke_stack[-1]
        if (
            node is None
            or node.diverged
            or node.cursor != 0
            or node.expected_seed is None
        ):
            return []
        live_seed = [
            (message.get("role"), message.get("content", ""))
            for message in event.messages
            if isinstance(message, dict)
            and (provenance := provenance_of(message)) is not None
            and provenance.kind is MessageKind.SEED
        ]
        if live_seed != node.expected_seed:
            node.diverged = True
            logger.info(
                f"TrajectoryReplay: seed divergence detected for invoke {event.invoke_id} — its "
                f"starting prompt differs from the trace's; all its LLM calls will go live"
            )
        return []

    def on_llm_query_send(self, event: LLMQuerySend) -> list:
        # Serve from the top-of-stack node's own per-invoke queue.
        if not self._invoke_stack:
            return []
        node, _ = self._invoke_stack[-1]
        # An invoke the trace has no node for (see ``on_invoke_enter``): nothing to serve, so the
        # query goes live.
        if node is None:
            return []
        # Cleared here and set only on the serving path at the end, so a query this node does
        # NOT answer leaves behind no stale index for ``_response_diverges`` to compare the
        # live turn against. Every ``LLMQuerySend`` that reaches a node passes through this
        # line, and its ``LLMQueryExit`` is the very next event on the node, so the field is
        # always the current query's.
        node.supplied_index = None
        if node.is_exhausted:
            # This invoke has no more saved responses — go live.
            return []
        # Per-invoke divergence latch: once this invoke's live REPL output diverged from
        # the trace, all its remaining calls go live. Checked BEFORE serving so the
        # diverging response itself is not re-served.
        if node.diverged:
            return []
        # PROMPT divergence, checked here rather than at the next REPLExecExit: the committed
        # message list is in view at Send, so the very query whose prompt first differs is the
        # one that goes live — which is what the feature promises ("everything before the first
        # difference replays for free"). Serving a saved response to a prompt the original run
        # never saw is the failure this closes: the run then LOOKS replayed while carrying a
        # message the trajectory never answered.
        if self._detect_divergence and self._edits_diverge(node, event):
            node.diverged = True
            logger.info(
                f"TrajectoryReplay: prompt divergence detected at response #{node.cursor} "
                f"for invoke {event.invoke_id} — this query's message edits differ from "
                f"the trace; this and the remaining LLM calls for this invoke will go live"
            )
            return []
        r = node.next()
        if r is None:
            # is_exhausted was checked above, so this is unreachable;
            # guard anyway to satisfy the type checker and be safe.
            return []
        # ``next()`` advanced the cursor past the response it returned, so the slot it came from
        # is the one behind it.
        node.supplied_index = node.cursor - 1
        # ATIF carries real completion tokens, so the substitution below should be
        # a no-op; kept as a defensive guard against a trace with missing metrics.
        completion_tokens = r.completion_tokens
        if completion_tokens is None:
            if not self._warned_unknown_completion_tokens:
                logger.warning(
                    "TrajectoryReplay: completion_tokens is None for at least one "
                    "replayed call. Substituting 0 so the cost-tracker "
                    "contract holds; total_llm_completion_tokens will be "
                    "understated for the replayed portion. `cost` and "
                    "`prompt_tokens` are exact."
                )
                self._warned_unknown_completion_tokens = True
            completion_tokens = 0
        return [
            SupplyLLMResponse(
                content=r.supplied_content,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=r.cost,
            )
        ]

    def _edits_diverge(self, node: _InvokeNode, event: LLMQuerySend) -> bool:
        """Whether this query's live message edits differ from the ones the trace recorded
        for the same query."""
        # Indexed by ``node.cursor`` — the slot the response ABOUT to be served occupies, since
        # the cursor advances in ``node.next()`` below this check. (The observation detector
        # indexes ``cursor - 1``, running after that advance; the two are consistent, not a
        # discrepancy.)
        #
        # Unguarded on purpose. The caller returns early on ``node.is_exhausted``
        # (``cursor >= len(responses)``) and ``__post_init__`` pads ``expected_edits`` to
        # ``len(responses)``, so ``0 <= cursor < len(expected_edits)`` holds by construction. A
        # bounds branch here would be worse than absent: an alignment bug would return ``False``
        # for this query and every later one in the node, switching edit checking off while the
        # run still reports a clean, complete, zero-cost replay — the exact silent-wrong-replay
        # this detector exists to prevent. An IndexError is the right failure for a broken
        # invariant; a loaded trace cannot reach one, because the builder raises first.
        expected = node.expected_edits[node.cursor]
        # Back-compat: ``None`` means the trace's writer did not record edits at all, so
        # "absent" carries no information and comparing against it would flag every edit the
        # live run makes. Skipped rather than treated as empty — see ``_trace_records_edits``.
        if expected is None:
            return False
        # Compared BY VALUE, never re-rendered. The observation detector has to re-render with
        # today's protocol and so carries a documented format-drift limitation (see
        # ``_detect_divergence_on_exec_exit``); edits do not, because the recorded block stores an
        # added message's role and content verbatim. Routing them through ``render_observation``
        # or ``_normalize_output`` would import that limitation for nothing.
        return _live_edit_signature(event.message_edits) != expected

    def on_llm_query_exit(self, event: LLMQueryExit) -> list:
        # The turn is settled HERE and nowhere earlier. ``LLMQueryComplete`` fires with the RAW
        # response and its ``ModifyLLMResponse`` effects compose after every Complete hook has
        # run, so a hook watching Complete sees the pre-transform content — which on the replayed
        # path is the supplied content itself, and would compare equal by construction. Exit
        # carries the post-transform object, the same one the agent re-reads once the span closes
        # and commits to the buffer, so it is the first point at which the turn this run will KEEP
        # can be compared with the turn the trace kept.
        #
        # Exit is observation-only for everything but ``Abort``, so this returns no effects: the
        # divergence is latched on the node and paid for by the invoke's NEXT query going live.
        # The already-committed turn is not undone — it cannot be, the query is over.
        if not self._detect_divergence or not self._invoke_stack:
            return []
        node, _ = self._invoke_stack[-1]
        if node is None or node.diverged:
            return []
        if self._response_diverges(node, event):
            node.diverged = True
            logger.info(
                f"TrajectoryReplay: response divergence detected at response "
                f"#{node.supplied_index} for invoke {event.invoke_id} — the assistant turn "
                f"this run commits differs from the one the trace recorded; the remaining LLM "
                f"calls for this invoke will go live"
            )
        return []

    def _response_diverges(self, node: _InvokeNode, event: LLMQueryExit) -> bool:
        """Whether the assistant turn this query is about to commit differs from the one the
        trace recorded for the same query."""
        # The single seam for the assistant-turn comparison, the way ``_edits_diverge`` is the
        # single seam for the edit one: widening what counts as "the same turn" — ``.content`` is
        # all it compares today, so a tokens/cost-only transform is invisible (see the covers
        # block below for why that is the right default and what it costs) — is an edit to this
        # method and nothing else.
        index = node.supplied_index
        # ``None`` = this query was NOT answered from the queue. A live query has no recorded
        # turn it is the replay of, and comparing one against the trace is the observation
        # detector's job downstream, not this one's.
        if index is None:
            return False
        # Non-completed arms (#892 outcome union) carry no response: an aborted or failed query
        # commits no turn, so there is nothing to compare.
        if not isinstance(event.outcome, Completed):
            return False
        # Compared against the RECORDED FINAL (``.content``), which is the right side in both
        # regimes, though for different reasons.
        #
        # ``TrajectoryRecorder`` writes the agent step from ``LLMQueryExit``, i.e. POST-transform, while
        # replay supplies at ``LLMQuerySend``, i.e. in the PRE-transform producer's position. On a
        # trace recording ``raw_message`` the supplied string is the recorded RAW, so the rerun's
        # own transform re-derives this exact value and a faithful rerun compares equal. On an
        # older trace there is no raw to supply and ``.content`` does both jobs — supplied value
        # and recorded turn — which is the conflation this detector exists to expose: a
        # Complete-stage ``ModifyLLMResponse`` then re-runs over content that already carries it,
        # and the difference is reported here instead of entering the buffer unnoticed.
        #
        # With no response transform in play, on a trace of either vintage, the two coincide and
        # this compares equal by construction — the false-positive floor that matters, since a
        # spurious divergence takes the rest of the invoke live and costs real money.
        #
        # BACK-COMPAT: no version gate, unlike ``_edits_diverge``. Both sides of this comparison
        # predate every schema question — the recorded side is the agent step's ``message``, which
        # every ATIF version has always written, and the live side is this run's own response. So
        # there is no "the writer did not record it" state to distinguish from "it was empty", and
        # a gate would only switch the detector off on old traces for nothing.
        # ``or ""`` because the recorded side went through the same coercion: the writer stores
        # a step's ``message`` as ``content or ""``, so a transform that nulls the content is
        # recorded as the empty string. Comparing the live ``None`` against that empty string
        # reported a divergence for a rerun that reproduced the recording exactly.
        return (event.outcome.result.content or "") != node.responses[index].content

    # ------------------------------------------------------------------
    # What these detectors cover, and what they do not
    # ------------------------------------------------------------------
    #
    # The committed prompt is ``_apply_message_edits(buffer, edits)``
    # (``Agent._compose_shown_messages``; the provider gets its ``to_wire_messages``
    # projection at the call), over a buffer of seed + assistant turns +
    # observations + persistent edits. TrajectoryReplay watches all four:
    #
    #   * the seed        — compared at the invoke's first ``LLMQueryEnter`` against
    #                       ``expected_seed``, the leading seed steps the recorder wrote verbatim;
    #   * observations    — compared at ``REPLExecExit`` against ``expected_outputs``;
    #   * edits           — compared at ``LLMQuerySend`` against ``expected_edits``;
    #   * the ASSISTANT turn's CONTENT — compared at ``LLMQueryExit`` against the response the
    #     query was served (``_response_diverges``, above). This is the ``ModifyLLMResponse``
    #     route: a Complete-stage transform is authoritative (the agent re-reads the response once
    #     the span closes, ``_agent.py``). The trace records the turn POST-transform; it now ALSO
    #     records the pre-transform content when the two differ (``extra.raw_message``, #1309), and
    #     replay supplies that raw in the producer's position, so the rerun's transform lands on
    #     the same input the recording's did and re-derives the recorded turn. The comparison is
    #     unchanged by that — it was always final-vs-recorded-final — but its meaning sharpens: a
    #     faithful rerun now compares equal instead of compounding the transform, and a transform
    #     DROPPED since the recording is reported for the first time.
    #
    #     On a trace with no recorded raw the old regime applies: the rerun's transform compounds
    #     on a turn that already carries it, and the invoke diverges on its first turn. THERE the
    #     observation detector is only a partial backstop — it fires only when the rewrite changes
    #     the EXECUTED code. Under ``CodeOnlyProtocol`` the response IS the code, so most rewrites
    #     do change it; one that touches only a comment changes the code bytes while leaving
    #     behaviour identical, and the observation the detector compares is what stays the same.
    #
    #     CONTENT and nothing else, deliberately. A ``ModifyLLMResponse`` that only RE-PRICES a
    #     response (``BudgetPool``'s repricing at Complete) leaves the committed turn identical and
    #     correctly reports nothing — comparing the whole object would take those runs live for a
    #     turn the model would not notice. The price is the mirror of that: a transform touching
    #     only tokens or cost is invisible here, so replay serves the recorded post-transform
    #     cost, the hook re-prices it again, and the run's cost totals drift from the trace's with
    #     no divergence reported.
    #
    #     THE MIRROR DIRECTION — a transform the RECORDING had and the rerun does not — depends
    #     on the trace's vintage, and this is the one place the two regimes genuinely differ.
    #     On a trace recording ``extra.raw_message`` (#1309) it IS reported: replay supplies the
    #     raw, the untransformed rerun commits it unchanged, and it differs from the recorded
    #     final. On an older trace it is not, and that is the right answer there rather than a
    #     gap — the recorded turn is already the transformed one, so the untransformed rerun
    #     commits precisely it and the criterion ("the committed turn differs from the recorded
    #     one") is genuinely not met. Nothing in the older schema distinguishes a recording that
    #     transformed from one that did not, which is why the field had to be added rather than
    #     inferred.
    #
    # NOT covered, and each of these replays with no divergence reported at all:
    #
    #   * a buffer message mutated IN PLACE. ``LLMQueryEnter.messages`` and
    #     ``LLMQuerySend.messages`` are tuple-coerced for membership but the dicts inside are
    #     the buffer's own, so a hook that rewrites ``event.messages[i]["content"]`` at either
    #     event changes the prompt for this and every later turn while composing no edit record.
    #   * the wire projection. ``to_wire_messages(keep=llm_client.extra_message_keys)``
    #     means swapping backends can change what the PROVIDER receives from an identical
    #     buffer and identical edits (the committed list on ``LLMQuerySend`` is stamped and
    #     backend-independent; the projection happens after it).
    #
    # Widening any of these is not an edit to the reducers. Each channel has exactly one
    # comparison site: ``_edits_diverge`` for prompt edits (with ``_live_edit_signature`` /
    # ``_recorded_edit_signature`` as the only definition of "the same prompt edit"),
    # ``_response_diverges`` for the assistant turn, ``on_llm_query_enter`` for the seed.
    #
    # The seed is compared as rendered, values and all (user call). A difference there is a real
    # divergence, not a false alarm: the recorded responses answered a different prompt. An
    # object passed as an input itself is rendered from what it exposes (its methods, a
    # function's signature), not its address. What can differ between two runs of the same code
    # is text the model genuinely reads: a fresh temp path, a set's per-process order, a custom
    # repr that prints an id, or an object held inside a list or dict input, which the
    # container's repr shows by its default ``<… at 0x…>`` address.
    #
    # WHY NOT COMPARE THE COMMITTED LIST ITSELF, the alternative weighed against this one: ATIF
    # records the conversation as steps plus edits, not the wire list, so the recorded side
    # would have to be RECONSTRUCTED — a second implementation of ``_apply_message_edits``\' fold
    # (persistent vs transient views, same-slot add ordering, drops named by a per-run uuid,
    # adds that never become steps of their own), maintained here and re-run on every replayed
    # query. Any drift between the two implementations reads as a false divergence on an
    # identical rerun, which takes the whole run live and so costs real money — the harder
    # failure to notice and the more expensive one. Comparing the edit records is exact, and on
    # the hot path is a tuple build over what the hooks edited plus one equality.

    def on_repl_exec_exit(self, event: REPLExecExit) -> list:
        # Per-invoke divergence, keyed by the ACTIVE node (top of stack): each node
        # carries its own expected_outputs + diverged latch, so a diverged invoke never
        # desynchronizes a sibling/parent.
        return self._detect_divergence_on_exec_exit(event)

    def _detect_divergence_on_exec_exit(self, event: REPLExecExit) -> list:
        """Per-invoke divergence detection against the loaded trace.

        Keyed per-invoke via the ACTIVE node (top of ``_invoke_stack``): the exec that
        just finished followed the response this node served on its most recent
        ``on_llm_query_send`` (``node.cursor - 1``). Compare the live output against that
        response's ``expected_outputs`` slot, abbreviating + normalizing so identical execs
        compare equal, and latch ``node.diverged`` on the first mismatch.
        """
        if not self._detect_divergence:
            return []
        # Non-completed arms (#892 outcome union) carry no result to compare — return
        # without marking a divergence: a machinery failure is not the live output
        # differing from the trace.
        if not isinstance(event.outcome, Completed):
            return []
        # The active invoke is the top of the stack. A nested invoke's own
        # REPLExecExit fires while that child is on top, so its output compares
        # against the child node — the per-invoke keying is the stack itself.
        if not self._invoke_stack:
            return []
        node, _ = self._invoke_stack[-1]
        if node is None or node.diverged:
            return []
        idx = node.cursor - 1
        if not (0 <= idx < len(node.expected_outputs)):
            return []
        expected = node.expected_outputs[idx]
        if expected is None:
            return []
        # Reproduce the observation by *calling* the codec's own render seam (#1055): the raw,
        # untruncated exec output observed here is not what the agent saw — render_observation is,
        # applying whatever transform (abbreviation included) the protocol renders it with. Driving
        # that seam, rather than re-deriving the truncation from CodeOnlyProtocol attributes the
        # BaseProtocol ABC never declared, means any protocol is reproduced correctly and none can
        # AttributeError here (the config boundary supports protocols that implement only some
        # settings). Compare the FULL observation the agent saw MESSAGE-BY-MESSAGE — every message
        # render_observation returns, against the recorded observation's messages — so a protocol
        # that changes how many messages a turn emits diverges too, not just one that changes the
        # body text.
        #
        # Drive `event.config.protocol` — the RESOLVED codec for the invoke that produced this
        # event (stamped on every Event since #463), NOT ambient `get_config().protocol`. They
        # differ whenever the invoke ran under a per-call `ConfigOverride` (folded onto a throwaway
        # copy of the stack, so it never reaches an ambient read) and on a worker thread, where the
        # ContextVar did not propagate and the ambient read sees a bare stack. A different codec
        # would render a different observation and flag a false divergence,
        # silently defeating replay determinism (#826 §1). `event.config.protocol` is exactly the
        # codec the agent rendered with, so the comparison is codec-consistent by construction.
        #
        # KNOWN LIMITATION — codec-consistent but not *format*-stable. It re-renders with whatever
        # the protocol renders today and diffs that against text a past run wrote, so changing the
        # observation/truncation format invalidates traces recorded before the change: each
        # truncated iteration reports a divergence even on a byte-identical rerun.
        # `_normalize_output` does not paper over this — it strips REPL session IDs and traceback
        # line numbers, not the truncation marker. Accepted deliberately: normalizing the marker
        # away on both sides would restore cross-format replay only by making the omitted count
        # invisible to divergence detection (a run dropping a different amount of output would start
        # comparing equal), a worse loss than replay of traces recorded before a one-off change.
        result = event.outcome.result
        if isinstance(result, Continue):
            rendered = event.config.protocol.render_observation(result)
            actual: list[str] | None = [
                _normalize_output(str(m.get("content") or "")) for m in rendered
            ]
        else:
            # The recording had an observation here (expected is not None), but this run finished
            # or raised at this turn instead of continuing — a divergence, with nothing to render.
            actual = None
        expected_norm = [_normalize_output(s) for s in expected]
        if actual != expected_norm:
            node.diverged = True
            logger.info(
                f"TrajectoryReplay: divergence detected at response #{idx} "
                f"for invoke {event.invoke_id} — remaining LLM calls for "
                f"this invoke will go live"
            )
        return []


# ------------------------------------------------------------------
# ATIF per-invoke tree builder
# ------------------------------------------------------------------


def _load_invoke_roots(path: str | Path) -> list[_InvokeNode]:
    """Load an ATIF trace file into one per-invoke response tree per top-level invoke."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    # TrajectoryRecorder writes a single root trajectory as an object, or an array when the
    # traced scope had several top-level invokes -- one root each, in the order they ran.
    documents = data if isinstance(data, list) else [data]
    if not documents:
        # The recorder writes ``[]`` for a scope that ran no invoke: there is no recorded invoke to
        # hand any top-level invoke, so the file is refused rather than accepted as a replay of
        # nothing. A root with no responses -- an invoke recorded before its first query
        # completed -- is a recording like any other, and loads.
        raise ValueError(f"TrajectoryReplay: no trajectory to replay in {path}")
    roots = [
        _build_invoke_tree_from_atif(
            document,
            edits_recorded=_trace_records_edits(document),
            # No fallback: a writer that did not declare ``raw_message`` never wrote one, so
            # there is nothing to read and replay keeps supplying the recorded final — today's
            # behaviour, and the reason old traces still replay exactly as they did.
            raw_recorded=_writer_records(document, "raw_message") is True,
        )
        for document in documents
    ]
    logger.info(f"TrajectoryReplay: loaded ATIF trace from {path}")
    return roots


#: Floor for the version half of the gate. NOT the version ``extra.edits`` arrived in — the
#: literal was never bumped for it, which is why the gate needs the provenance probe as well.
_EDITS_SINCE_SCHEMA = (1, 7)


def _writer_records(traj: dict[str, Any], capability: str) -> bool | None:
    """Whether this trace's writer declares it records ``capability``, or ``None`` if it declares
    nothing at all.

    ``None`` is the undeclared answer and is NOT ``False``: the document says nothing about what
    its writer records, so the caller must fall back to whatever it did before rather than
    conclude a field is absent on purpose. A declared empty list is ``False``, not ``None`` — it
    says "records nothing", which is an answer.

    Undeclared usually means "written before the declaration existed", but it also covers a
    document from another writer and one a tool has stripped, and this cannot tell those apart.
    Every caller must therefore treat ``None`` as the conservative branch: for ``raw_message``
    that is "no raw to read", which degrades to serving the recorded finals rather than erroring.
    """
    # The declaration exists because a version cannot answer this: ``schema_version`` tracks the
    # external ATIF standard and stayed at "ATIF-v1.7" across every extension we added (see
    # ``WRITER_RECORDS`` in atif_trace.py). Dating a document by probing for some other field
    # that happened to land in the same commit — what ``_trace_records_edits`` below does — works
    # once and stops working the moment a third field is added, which is precisely what adding
    # ``raw_message`` would have done.
    #
    # NO POSITIVE-EVIDENCE FALLBACK, unlike ``_trace_records_edits``'s probe. A document whose
    # ``extra`` keys a tool has stripped, or that predates the declaration, reads as undeclared
    # even if its steps still carry ``raw_message`` — replay then serves the recorded finals, i.e.
    # exactly what it did before #1309. That is a silent revert to the old regime rather than an
    # error, which is the right failure direction (worst case is the divergence-on-turn-1 cost
    # this feature removes, never a wrong turn served) but is worth knowing when a trace that
    # should replay free does not.
    extra = traj.get("extra")
    records = extra.get("writer_records") if isinstance(extra, dict) else None
    if not isinstance(records, list):
        return None
    return capability in records


def _trace_records_edits(traj: dict[str, Any]) -> bool:
    """Whether this trace's writer emits the per-query message-edits block.

    A document that declares its capabilities is taken at its word, either way, and under either
    spelling of this one (``message_edits`` now, ``edits`` before the rename). One that does not
    is dated by probing: ``True`` when it declares ``ATIF-v1.7`` or later AND carries step-level
    ``extra.provenance``, and anything else — an older or unparseable version, or a v1.7 document
    from before the provenance/edits channel existed — reads ``False``.
    """
    # THE BACK-COMPAT RULE, and why it takes two signals rather than one.
    # The block is ``extra.message_edits`` as written now and ``extra.edits`` before the rename;
    # this gate predates neither and probes for neither, which is why it keeps working across it.
    #
    # TrajectoryRecorder omits the block entirely on a turn that composed no edits, so within a
    # trace "absent" and "empty" are the same bytes. That is fine for a trace whose writer
    # records the field — absent then genuinely means "no edits this turn", and comparing it
    # against a live run that DOES edit is exactly the divergence to report. It is not fine for
    # a trace written before the field existed, where absent means "unknown": every turn would
    # report a divergence the moment any hook edits messages, on an otherwise identical rerun.
    #
    # The schema version ALONE does not separate the two, and believing it did was a bug. The
    # version literal has read "ATIF-v1.7" since the format landed (#381) and was never bumped
    # when ``extra.edits`` was added a month later (#1089), so every trace written in between
    # declares v1.7 and carries no edits — the exact spurious-divergence case above, on ordinary
    # in-tree traces rather than the hypothetical third-party one.
    #
    # SUPERSEDED FOR NEW TRACES by ``extra.writer_records`` (checked first, below): a document
    # that declares its capabilities answers this directly, and the two-signal probe survives only
    # for documents written before the declaration existed.
    #
    # ``extra.provenance`` on a step is what dates the document: it landed in the same commit as
    # ``extra.edits`` and is written UNCONDITIONALLY on every seed/observation step, so its
    # presence means the writer is post-#1089 without depending on whether any hook happened to
    # edit anything. That last part is why probing for the edits block itself was rejected: a run
    # recorded with no editing hooks writes no blocks anywhere, so that probe would disable
    # detection on precisely the traces this feature is for — a trace recorded without a hook,
    # replayed with one. Probing provenance has none of that false negative.
    #
    # Residual exposure, now genuinely hypothetical: a post-#1089-shaped document from a
    # third-party writer that emits provenance but not edits would report spurious divergence.
    declared = _writer_records(traj, "message_edits")
    if declared is not None:
        # The list exists, so it answers — but under EITHER spelling: a trace written before the
        # block was renamed declares this capability as ``edits``, and its list not naming
        # ``message_edits`` is a definite False rather than the silence that would fall through
        # to the probe. Asking only the current name would read every such trace as recording no
        # edits and quietly stop comparing them.
        return declared or _writer_records(traj, "edits") is True
    version = traj.get("schema_version")
    if not isinstance(version, str):
        return False
    digits = version.lstrip("ATIFatif-vV_ ")
    try:
        parts = tuple(int(part) for part in digits.split(".")[:2])
    except ValueError:
        return False
    # ``len(parts) >= 1`` so a future single-component ``"ATIF-v2"`` reads ``(2,) >= (1, 7)`` —
    # requiring a minor would silently SKIP comparison on a newer trace, the wrong direction.
    if not (len(parts) >= 1 and parts >= _EDITS_SINCE_SCHEMA):
        return False
    return _has_step_provenance(traj)


def _has_step_provenance(traj: dict[str, Any]) -> bool:
    """Whether any step in this trajectory or its descendants carries ``extra.provenance``."""
    # Whole-document, not per-trajectory: the flag describes the WRITER, and one trajectory of a
    # post-#1089 document could in principle hold no provenance-carrying step.
    for step in traj.get("steps", []):
        if isinstance(step, dict) and (step.get("extra") or {}).get("provenance"):
            return True
    return any(
        _has_step_provenance(child)
        for child in traj.get("subagent_trajectories", [])
        if isinstance(child, dict)
    )


def _build_invoke_tree_from_atif(
    traj: dict[str, Any], *, edits_recorded: bool = False, raw_recorded: bool = False
) -> _InvokeNode:
    """Build a per-invoke ``_InvokeNode`` tree from one ATIF trajectory dict.

    One ``_SavedResponse`` per ``agent`` step (in order), carrying the step's real
    ``metrics`` (``prompt_tokens`` / ``completion_tokens`` / ``cost_usd`` → ``cost``, all
    real, so no ``None`` substitution). Children are built recursively from
    ``subagent_trajectories`` in stored (execution) order.

    Expected-output alignment (per-trajectory, positional assistant→following-observation
    pairing): each ``agent`` step is paired with the message of the FIRST ``user`` step that
    appears AFTER it and BEFORE the next ``agent`` step — that user step is the REPL
    observation of this response's exec (the observation message content *is* the
    REPL output). A terminal ``agent`` step, or one immediately followed by another ``agent``
    step with no intervening ``user`` step, has expected output ``None``. The leading
    ``system``/``user`` steps before the first ``agent`` step are the seed prompt, not
    observations, so they pair with nothing.

    Expected-EDIT alignment (same positional scoping): each ``agent`` step is paired with the
    message-edits block found on any non-``agent`` step between it and the PREVIOUS ``agent``
    step (the leading seed steps, for the first one). That window is exactly the set of steps
    ``TrajectoryRecorder`` emits at the enter of this response's own query, and the block hangs on the
    last of them, so the window holds this query's edits and no other's. Pass
    ``edits_recorded=False`` (the default) for a trajectory whose writer does not record the
    field at all: every slot is then ``None`` — unknown, not empty — and the comparison is
    skipped rather than reporting every live edit as a divergence.

    Because a nested invoke's steps live in its OWN child trajectory (ATIF nests them under
    ``subagent_trajectories``, not inline), the parent's observations never include a child's
    — the pairing is naturally scoped to one trajectory.

    The observation is identified by **provenance** (``extra.provenance.kind == "observation"``,
    stamped since the provenance channel landed), not merely by position. This is why divergence
    detection is safe here: a hook-ADDED user message between the agent step and the observation
    (e.g. the persistent ``<return_type>`` reminder emitted when a return type is set) is NOT the
    observation and is correctly skipped, where a positional "first user step" rule would mispair
    it and read a false divergence on that turn. Only when no user step in the window carries
    observation provenance (a trace from before that channel, or an unstamped message) does it
    fall back to the positional first-user rule.
    """
    # The observation message content is the rendered observation itself since #928, so the tree
    # can be rebuilt from the trace without re-rendering.
    steps = traj.get("steps", [])
    responses: list[_SavedResponse] = []
    expected_outputs: list[list[str] | None] = []
    expected_edits: list[_EditSignature | None] = []
    n = len(steps)
    # The edits block seen since the previous agent step — see the alignment note above. Reset
    # after each agent step consumes it, so a block can never be paired with two responses.
    pending_edits: Any = None
    for i, step in enumerate(steps):
        if step.get("source") != "agent":
            extra = step.get("extra") or {}
            # Either spelling, current key first; see ``_recorded_edit_signature``.
            block = extra.get("message_edits", extra.get("edits"))
            pending_edits = pending_edits if block is None else block
            continue
        metrics = step.get("metrics") or {}
        pt = metrics.get("prompt_tokens")
        ct = metrics.get("completion_tokens")
        cost = metrics.get("cost_usd")
        # Read only when the writer declared it. On a declaring document an absent
        # ``raw_message`` means "this turn was not transformed", so the final is the raw and
        # ``supplied_content`` falls through to ``content``. Presence is tested with ``in``, not
        # by the value: a recorded ``null`` is a real raw (a transform that filled in an empty
        # response) and must be served as one.
        step_extra = step.get("extra") or {}
        has_raw = raw_recorded and "raw_message" in step_extra
        responses.append(
            _SavedResponse(
                content=step.get("message"),
                prompt_tokens=pt,
                completion_tokens=ct,
                cost=cost,
                raw_content=step_extra.get("raw_message") if has_raw else None,
                has_raw=has_raw,
            )
        )
        # The FULL observation the agent saw between this agent step and the next: the LIST of
        # EVERY user step tagged observation provenance — the driver stamps OBSERVATION on each
        # message ``render_observation`` returned, however many — compared message-by-message
        # against the live render (#1055), so every one of them participates in divergence
        # detection. A positionally-first user step that is a hook reminder (e.g.
        # `<return_type>`) carries no observation provenance and is skipped; fall back to the first
        # user step (as a one-element list) only when NONE in the window is provenance-tagged (a
        # trace predating the provenance channel); ``None`` when there is no observation at all.
        obs_parts: list[str] = []
        first_user: str | None = None
        for j in range(i + 1, n):
            src = steps[j].get("source")
            if src == "agent":
                break
            if src != "user":
                continue
            if first_user is None:
                first_user = steps[j].get("message")
            prov = (steps[j].get("extra") or {}).get("provenance") or {}
            if prov.get("kind") == "observation":
                obs_parts.append(str(steps[j].get("message") or ""))
        if obs_parts:
            expected_outputs.append(obs_parts)
        elif first_user is not None:
            expected_outputs.append([str(first_user)])
        else:
            expected_outputs.append(None)
        expected_edits.append(
            _recorded_edit_signature(pending_edits) if edits_recorded else None
        )
        pending_edits = None

    _require_aligned(traj, responses, expected_outputs, expected_edits)
    children = [
        # The ROOT's declaration governs every child. Subagent trajectories are nested inside the
        # one document, written by the one ``TrajectoryRecorder`` in the one process, so a child cannot
        # have been produced by a writer of a different vintage; reading each child's own key
        # would add a way for a hand-edited document to disagree with itself and no way for a
        # real one to.
        _build_invoke_tree_from_atif(
            child, edits_recorded=edits_recorded, raw_recorded=raw_recorded
        )
        for child in traj.get("subagent_trajectories", [])
    ]
    return _InvokeNode(
        responses=responses,
        children=children,
        expected_outputs=expected_outputs,
        expected_edits=expected_edits,
        expected_seed=_recorded_seed(steps),
    )


def _recorded_seed(steps: list[dict[str, Any]]) -> list[tuple[Any, Any]] | None:
    """The seed prompt one trajectory recorded: the steps before its first ``agent`` step."""
    # Read only when every one of those steps is stamped as a seed. That is what the recorder
    # writes at an invoke's first enter; anything else (an unstamped older trace, an odd
    # document) is unknown rather than empty, and is skipped like an unrecorded edit block.
    seed: list[tuple[Any, Any]] = []
    for step in steps:
        if step.get("source") == "agent":
            break
        provenance = (step.get("extra") or {}).get("provenance") or {}
        if provenance.get("kind") != "seed":
            return None
        seed.append((step.get("source"), step.get("message")))
    return seed


def _require_aligned(
    traj: dict[str, Any],
    responses: list[_SavedResponse],
    expected_outputs: list[list[str] | None],
    expected_edits: list[_EditSignature | None],
) -> None:
    """Raise unless both expectation lists are 1:1 with ``responses``.

    Raises:
        ValueError: If either list has a different length — the loaded trajectory cannot be
            replayed, since every divergence check indexes them by the response cursor.
    """
    # A ValueError at load, matching ``_load_invoke_roots``'s refusal of an empty trace, rather than an
    # ``assert`` — ``-O`` strips asserts, and this is the one condition under which replay would
    # otherwise degrade SILENTLY: the detectors index these lists by cursor, so a short list
    # reads as "nothing recorded for this turn" and switches checking off for the rest of the
    # node while the run still reports a clean, complete replay. Loud at load is the only
    # acceptable failure. Nothing in the builder can violate this today — both lists are
    # appended exactly once per ``agent`` step, in the same iteration with no ``continue``
    # between them — which is precisely why the assumption needs a guard rather than a comment:
    # a later edit that adds a filter or an early ``continue`` breaks it invisibly.
    n = len(responses)
    if len(expected_outputs) != n or len(expected_edits) != n:
        raise ValueError(
            f"TrajectoryReplay: trajectory {traj.get('trajectory_id') or '<unnamed>'} loaded "
            f"{n} response(s) but {len(expected_outputs)} expected output(s) and "
            f"{len(expected_edits)} expected edit(s); the divergence checks index these "
            f"lists by response, so a misaligned trace cannot be replayed."
        )


_REPL_SESSION_ID_RE = re.compile(r"<repl_[0-9a-f]+_\d+>")
_TRACEBACK_LINE_RE = re.compile(r'(File "[^"]+", line) \d+')


def _normalize_output(s: str) -> str:
    """Normalize a REPL output string for divergence comparison.

    The caller is expected to have already rendered the live output through the
    protocol's ``render_observation`` so it matches the observation the
    original run wrote. On top of that, this strips two known sources of
    run-to-run non-semantic noise that appear in REPL exceptions:

    1. **Random REPL session IDs** in agent-code tracebacks
       (``<repl_e0657f5a_5>`` → ``<repl_X>``). The hex prefix is randomized
       per session and the trailing ``_N`` is the recursion-depth counter;
       neither carries semantic meaning.
    2. **Source-file line numbers** in tracebacks (``File "...py", line 123``
       → ``File "...py", line N``). Editing any source file between the
       saved run and the replay shifts these numbers without changing
       behavior — the file path is enough to identify the frame.

    Without these two normalizations, *every* task whose original run raised
    a Python exception would be a false-positive divergence (observed: 100%
    of diverged tasks in the AppWorld no_meta 0607 replay before this fix).
    Trailing whitespace is stripped last to absorb any trailing newline the
    observation body carries.
    """
    # Routed through the protocol's render_observation (#1055) rather than reproducing its
    # formatting, so replay matches whatever the protocol currently emits.
    s = _REPL_SESSION_ID_RE.sub("<repl_X>", s)
    s = _TRACEBACK_LINE_RE.sub(r"\1 N", s)
    return s.rstrip()
