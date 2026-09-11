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

from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import SupplyLLMResponse
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryExit,
    LLMQuerySend,
    REPLExecExit,
)
from jaz.hooks.events.base import Completed
from jaz.hooks.events.llm_query import ResolvedMessageEdits
from jaz.repl.types import Continue

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Message-edit signatures: the comparable form of one query's edits
# ------------------------------------------------------------------

#: One dropped message: ``(index, persistent)``. ``index`` is ``None`` for a record whose
#: trace does not carry a usable one (see ``_edit_index``).
type _DropSig = tuple[int | None, bool]
#: One ``AddMessages`` edit: ``(index, persistent, ((role, content), ...))``.
type _AddSig = tuple[int | None, bool, tuple[tuple[str, str], ...]]
#: One query's message edits reduced to the fields divergence compares, each half in the
#: canonical order ``_canonical`` puts it in.
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


def _canonical(drops: Sequence[_DropSig], adds: Sequence[_AddSig]) -> _EditSignature:
    """Put both halves of a signature in the one order the comparison is defined over."""
    # Sorted, NOT in arrival order, because the PROMPT is order-insensitive here and the
    # comparison must not be stricter than the thing it stands for. ``apply_message_edits``
    # groups adds by resolved slot and sorts each group by ``_add_sort_key``, so the order
    # hooks emitted them in never reaches the model: the same two hooks composed in the other
    # ``with`` nesting produce a byte-identical prompt, and comparing arrival order would
    # report that as a divergence and take the whole run live. Sorting is not the fold's own
    # key, and does not need to be — it only has to be the SAME canonical order on both sides.
    # Drops are already position-sorted by the dispatcher's resolver; sorting them here too
    # keeps this reducer's contract independent of that.
    #
    # What this cannot see: ``AddMessages.sort_key``, which outranks content in the fold's
    # ordering, is not in ``extra.edits``. Two same-slot adds that differ only by ``sort_key``
    # fold in a different order and compare equal here — the same class of blind spot as a
    # drop's identity in the exclusion list above, and not fixable without an ATIF change.
    return tuple(sorted(drops, key=_sort_key)), tuple(sorted(adds, key=_sort_key))


def _sort_key(entry: tuple[Any, ...]) -> tuple[Any, ...]:
    """Total order over signature entries whose leading index may be ``None``."""
    # A bare ``sorted()`` would raise on a malformed trace, where one record's index is ``None``
    # and another's is an int — the one input this module promises not to die on.
    index = entry[0]
    return (index is None, 0 if index is None else index, *entry[1:])


def _live_edit_signature(edits: ResolvedMessageEdits) -> _EditSignature:
    """Reduce a live query's ``LLMQuerySend.edits`` to its comparable signature."""
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
    """Reduce a trace's ``extra.edits`` block to the same signature ``_live_edit_signature``
    produces. ``None`` (no block on this turn) reduces to ``_NO_EDITS``.

    Defensive about shape throughout: a trace is an on-disk document that may have been
    written by another tool, and a malformed block must not raise mid-replay.
    """
    if not isinstance(block, dict):
        return _NO_EDITS
    drops = [
        (_edit_index(d.get("index")), bool(d.get("persistent")))
        for d in (block.get("dropped") or [])
        if isinstance(d, dict)
    ]
    adds: list[_AddSig] = []
    for record in block.get("added") or []:
        if not isinstance(record, dict):
            continue
        raw = [m for m in (record.get("messages") or []) if isinstance(m, dict)]
        if not raw:
            continue
        messages = tuple(
            (_edit_value(m.get("role")), _edit_value(m.get("content"))) for m in raw
        )
        # Every message of one add record carries the same ``index`` (the record's resolved
        # insert-before slot, stamped per message by TrajectoryRecorder), so the first one is it.
        adds.append(
            (_edit_index(raw[0].get("index")), bool(record.get("persistent")), messages)
        )
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
    # ``_load_invoke_tree``; default via default_factory so an ``_InvokeNode(...)`` built without
    # it (e.g. the progress tests) keeps working.
    expected_outputs: list[list[str] | None] = field(default_factory=list)
    # The message edits the trace recorded for each response's OWN query, aligned 1:1 with
    # ``responses`` — the recorded ``extra.edits`` block reduced to its comparable fields (see
    # ``_recorded_edit_signature``), or ``None`` where the trace's writer did not record edits at
    # all (the back-compat rule; see ``_trace_records_edits``). ``_NO_EDITS`` — the empty
    # signature — is the *positive* statement "this turn had no edits", which is what a v1.7+
    # trace's missing block means, and is therefore compared rather than skipped. Populated by
    # ``_load_invoke_tree``; default via default_factory so an ``_InvokeNode(...)`` built without
    # it (e.g. the progress tests) keeps working, exactly as ``expected_outputs`` does.
    expected_edits: list[_EditSignature | None] = field(default_factory=list)
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


@dataclass
class _SavedResponse:
    """A single saved LLM response for replay."""

    content: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None


class TrajectoryReplay(Hook):
    """Replay saved LLM responses from an ATIF trace instead of calling the model.

    Each turn consumes the next saved response for the active invoke in order. Once an
    invoke's responses run out, its live LLM calls resume.

    The trace (as written by :class:`TrajectoryRecorder`) is the canonical replay/cost format.
    Each invoke replays from its own per-invoke queue — one ``TrajectoryReplay`` covers the
    whole invoke tree, and a nested ``invoke`` never shifts a sibling's or parent's
    cursor. The per-call tokens and cost are the *real* metrics recorded in the trace
    (``prompt_tokens`` / ``completion_tokens`` / ``cost_usd``) — no reconstruction — so
    replayed calls report exact token totals as well as exact dollar cost.

    With ``detect_divergence=True``, replay stops for an invoke as soon as it stops
    matching, on any of three comparisons: before each query, the message edits hooks
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

    Three limits worth knowing. A trace written before message edits were recorded is detected
    and its edit comparison skipped entirely (the other two are unaffected). No comparison looks
    at the *seed* prompt, so re-running a trace under a changed system prompt, task template, or
    inputs rendering reports no divergence.

    And a ``ModifyLLMResponse`` costs this feature most of its savings, in both directions. A
    saved response is already the *post*-transform turn, so any content-changing response
    transform active on the rerun re-applies itself to a turn that already carries it: the
    committed turn differs, and the invoke diverges on its FIRST turn. Replaying a trace under
    such a hook therefore yields exactly one free response per invoke and pays live for every
    call after it — even when nothing about the run has changed, which is the one place the
    "everything before the first difference replays for free" promise above does not hold.
    That one free turn is itself the doubly-transformed one, and divergence does not undo it:
    detection happens as the query closes, by which point the turn is committed. So the rerun
    keeps a first turn that matches neither the trace nor a live run, and continues from it.
    A transform the recording had and the rerun does *not* is the opposite case and reports
    nothing, the rerun committing precisely the recorded turn. Neither comparison sees a
    transform that changes only tokens or cost, so a re-pricing hook silently drifts the run's
    cost totals away from the trace's.

    Under ``with`` the replay covers invokes nested inside. Passed positionally, only that
    invoke replays; this hook supplies nothing to the invokes it nests, so they call the
    model live unless something else replays them.

    Args:
        path: Path to an ATIF ``.atif.json`` trace (as written by :class:`TrajectoryRecorder`).
        detect_divergence: When ``True``, each saved response is paired with the message
            edits composed for its query, with the assistant turn the query committed, and
            with the REPL observation that followed its exec; the first query whose edits or
            committed turn differ, or first exec whose output differs, marks THAT invoke
            diverged, switching it (only it) to live LLM for the rest of the run.

    Raises:
        ValueError: If the trace holds more than one root trajectory (a multi-root trace has
            no single invoke tree to drive), or if a trajectory's recorded observations or
            message edits are not 1:1 with its responses (a misaligned trace cannot be
            replayed, since the divergence checks index them by response).
    """

    # Sole authority for the model's turn: two active Replays would both try to author every
    # response, so the seed-time check rejects that pairing (see Hook.supplies_llm_response) —
    # one Replay owns the per-invoke queues.
    supplies_llm_response = True

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
        self._init_from_tree(_load_invoke_tree(path), detect_divergence)

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
        self._init_from_tree(invoke_tree, detect_divergence)
        return self

    def _init_from_tree(
        self, invoke_tree: _InvokeNode, detect_divergence: bool
    ) -> None:
        # Latch so we warn at most once per hook lifetime when a replayed response has
        # None completion/total tokens. ATIF carries real per-call metrics, so this
        # should not fire on a real ATIF trace; kept as a defensive guard.
        self._warned_unknown_completion_tokens = False

        # Stack of (_InvokeNode, invoke_id) pairs; the top is the currently-active invoke.
        self._invoke_tree = invoke_tree
        self._detect_divergence = detect_divergence

        # Wrap the root invoke node in a sentinel so that the first InvokeEnter event
        # (for the top-level invoke) correctly pops invoke_tree from the sentinel's
        # children and pushes it as the active frame.  Without the sentinel wrapper,
        # InvokeEnter for the top-level invoke would mistakenly treat invoke_tree itself
        # as the parent and pop invoke_tree.children[0] (the first nested invoke) as the
        # active frame — causing the top-level invoke's LLM calls to be served from the
        # first child's queue instead of its own.
        sentinel_root = _InvokeNode(responses=[], children=[invoke_tree])
        # ``_InvokeNode | None`` because ``on_invoke_enter`` pushes ``None`` for an invoke the
        # trace has no node for — see there. Spelling that in the annotation rather than carrying
        # a ``type: ignore`` on the push is what makes the ``node is None`` guards below read as
        # reachable code instead of defensive noise.
        self._invoke_stack: list[tuple[_InvokeNode | None, str]] = [
            (sentinel_root, "__sentinel__")
        ]
        total = self._count_tree_responses(invoke_tree)
        logger.info(
            f"TrajectoryReplay: {total} saved responses loaded for replay "
            f"(per-invoke queues, no cursor drift)"
        )

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
        return self._count_tree_replayed(
            self._invoke_tree
        ) >= self._count_tree_responses(self._invoke_tree)

    @property
    def replayed_count(self) -> int:
        """Number of responses replayed so far (summed over the tree cursors)."""
        return self._count_tree_replayed(self._invoke_tree)

    @property
    def saved_response_count(self) -> int:
        """Total number of saved responses this hook will serve.

        The public count used by resume banners; the total lives in the invoke tree.
        """
        return self._count_tree_responses(self._invoke_tree)

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
        parent_node = self._invoke_stack[-1][0] if self._invoke_stack else None
        child = parent_node.next_child() if parent_node is not None else None
        self._invoke_stack.append((child, event.invoke_id))
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list:
        if self._invoke_stack and self._invoke_stack[-1][1] == event.invoke_id:
            node, _ = self._invoke_stack.pop()
            if node is not None and not node.is_exhausted:
                logger.info(
                    f"TrajectoryReplay: invoke {event.invoke_id[:8]}… exited with "
                    f"{len(node.responses) - node.cursor} response(s) "
                    f"unused in its queue (live run ran fewer iterations "
                    f"than source)"
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
                content=r.content,
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
        # ``_detect_divergence_on_exec_exit``); edits do not, because ``extra.edits`` stores an
        # added message's role and content verbatim. Routing them through ``render_observation``
        # or ``_normalize_output`` would import that limitation for nothing.
        return _live_edit_signature(event.edits) != expected

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
        # Compared against the SUPPLIED response's content, which is also what the trace recorded
        # for this turn — one string doing both jobs, which is exactly the conflation this
        # detector exists to expose. ``TrajectoryRecorder`` writes the agent step from ``LLMQueryExit``,
        # i.e. POST-transform, while replay supplies at ``LLMQuerySend``, i.e. as the PRE-transform
        # producer output. With no response transform in play the two coincide and this compares
        # equal by construction — the false-positive floor that matters, since a spurious
        # divergence takes the rest of the invoke live and costs real money. A Complete-stage
        # ``ModifyLLMResponse`` breaks the identity: the transform re-runs over content that
        # already carries it, and the difference is reported here instead of entering the buffer
        # unnoticed.
        #
        # BACK-COMPAT: no version gate, unlike ``_edits_diverge``. Both sides of this comparison
        # predate every schema question — the recorded side is the agent step's ``message``, which
        # every ATIF version has always written, and the live side is this run's own response. So
        # there is no "the writer did not record it" state to distinguish from "it was empty", and
        # a gate would only switch the detector off on old traces for nothing.
        return event.outcome.result.content != node.responses[index].content

    # ------------------------------------------------------------------
    # What these detectors cover, and what they do not
    # ------------------------------------------------------------------
    #
    # The committed prompt is ``to_wire_messages(apply_message_edits(buffer, edits))``
    # (``Agent._compose_shown_messages``), over a buffer of seed + assistant turns +
    # observations + persistent edits. Replay watches three of those four:
    #
    #   * observations    — compared at ``REPLExecExit`` against ``expected_outputs``;
    #   * edits           — compared at ``LLMQuerySend`` against ``expected_edits``;
    #   * the ASSISTANT turn's CONTENT — compared at ``LLMQueryExit`` against the response the
    #     query was served (``_response_diverges``, above). This is the ``ModifyLLMResponse``
    #     route: a Complete-stage transform is authoritative (the agent re-reads the response once
    #     the span closes, ``_agent.py``), and the trace records the turn POST-transform while
    #     replay supplies it PRE-transform, so a rerun under the SAME transform applies it a second
    #     time to content that already carries it. The observation detector is only a partial
    #     backstop there — it fires only when the rewrite changes the EXECUTED code. Under
    #     ``CodeOnlyProtocol`` the response IS the code, so most rewrites do change it; one that
    #     touches only a comment changes the code bytes while leaving behaviour identical, and
    #     the observation the detector compares is what stays the same.
    #
    #     CONTENT and nothing else, deliberately. A ``ModifyLLMResponse`` that only RE-PRICES a
    #     response (``BudgetPool``'s repricing at Complete) leaves the committed turn identical and
    #     correctly reports nothing — comparing the whole object would take those runs live for a
    #     turn the model would not notice. The price is the mirror of that: a transform touching
    #     only tokens or cost is invisible here, so replay serves the recorded post-transform
    #     cost, the hook re-prices it again, and the run's cost totals drift from the trace's with
    #     no divergence reported.
    #
    #     THE MIRROR DIRECTION IS NOT REPORTED: a transform the RECORDING had and the rerun does
    #     not. Under this design that is the right answer rather than a gap — the recorded turn is
    #     already the transformed one, so the untransformed rerun commits precisely it, and the
    #     criterion ("the committed turn differs from the recorded one") is genuinely not met. It
    #     is in any case not reachable from the ATIF schema as written: ``TrajectoryRecorder`` has no
    #     ``on_llm_query_complete``, so it never observes a raw response, and nothing it writes
    #     tells a recording that transformed apart from one that did not. Under a schema that
    #     recorded the PRE-transform content it becomes both a real gap and free to close — and
    #     that same change would let a faithful rerun under a transform replay for FREE instead of
    #     diverging on turn 1, which is this detector's standing cost. See #1309; do not read the
    #     paragraph above as a finding that the schema should not change.
    #
    # NOT covered, and each of these replays with no divergence reported at all:
    #
    #   * the SEED. A changed system prompt, task template, or ``<inputs>`` rendering. Left out
    #     deliberately — the seed embeds caller inputs rendered with ``repr``, so object
    #     addresses and temp paths differ between two runs of the SAME code, and comparing it
    #     would diverge on turn 1 of most replays.
    #   * a buffer message mutated IN PLACE. ``LLMQueryEnter.messages`` is tuple-coerced for
    #     membership but the dicts inside are the buffer's own, so a hook that rewrites
    #     ``event.messages[i]["content"]`` changes the prompt while composing no edit record.
    #   * the wire projection. ``to_wire_messages(keep=llm_client.consumes_internal_keys)``
    #     means swapping backends can change the committed list from an identical buffer and
    #     identical edits.
    #
    # Widening any of these is not an edit to the reducers. Each channel has exactly one
    # comparison site: ``_edits_diverge`` for prompt edits (with ``_live_edit_signature`` /
    # ``_recorded_edit_signature`` as the only definition of "the same prompt edit"),
    # ``_response_diverges`` for the assistant turn. For the seed it is neither — it needs a new
    # recorded artefact as well as a new comparison point, since it is not in the trace in a form
    # this could read.
    #
    # WHY NOT COMPARE THE COMMITTED LIST ITSELF, the alternative weighed against this one: ATIF
    # records the conversation as steps plus edits, not the wire list, so the recorded side
    # would have to be RECONSTRUCTED — a second implementation of ``apply_message_edits``\' fold
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
        # Drive `event.config.protocol` — the DEPTH-RESOLVED codec for the invoke that produced
        # this event (stamped on every Event since #463), NOT ambient `get_config().protocol`. They
        # differ whenever the invoke ran under a per-depth (`configure_by_depth` /
        # `ConfigOverrideByDepth`) or per-call `ConfigOverride`: `get_config()` is depth-less
        # (post-#727 it folds `_default_config` + layers but never applies a DepthLayer), so a
        # different codec would render a different observation and flag a false divergence,
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
            rendered = event.config.protocol.render_observation(result, event.iteration)
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


def _load_invoke_tree(path: str | Path) -> _InvokeNode:
    """Load an ATIF trace file into the per-invoke response tree."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    # TrajectoryRecorder writes a single root trajectory as an object, or an array when the
    # traced scope had multiple top-level invokes. TrajectoryReplay drives one root tree, so a
    # multi-root trace is out of scope here (each would need its own top-level frame).
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(
                f"TrajectoryReplay: expected a single root trajectory, got "
                f"{len(data)} in {path}. Multi-root ATIF traces are not supported."
            )
        data = data[0]
    root = _build_invoke_tree_from_atif(data, edits_recorded=_trace_records_edits(data))
    logger.info(f"TrajectoryReplay: loaded ATIF trace from {path}")
    return root


#: Floor for the version half of the gate. NOT the version ``extra.edits`` arrived in — the
#: literal was never bumped for it, which is why the gate needs the provenance probe as well.
_EDITS_SINCE_SCHEMA = (1, 7)


def _trace_records_edits(traj: dict[str, Any]) -> bool:
    """Whether this trace's writer emits ``extra.edits``.

    ``True`` for a document that declares ``ATIF-v1.7`` or later AND carries step-level
    ``extra.provenance``; anything else — an older or unparseable version, or a v1.7 document
    from before the provenance/edits channel existed — reads ``False``.
    """
    # THE BACK-COMPAT RULE, and why it takes two signals rather than one.
    # TrajectoryRecorder omits ``extra.edits`` entirely on a turn that composed no edits, so within a
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
    # ``extra.provenance`` on a step is what dates the document: it landed in the same commit as
    # ``extra.edits`` and is written UNCONDITIONALLY on every seed/observation step, so its
    # presence means the writer is post-#1089 without depending on whether any hook happened to
    # edit anything. That last part is why probing for ``extra.edits`` itself was rejected: a run
    # recorded with no editing hooks writes no blocks anywhere, so that probe would disable
    # detection on precisely the traces this feature is for — a trace recorded without a hook,
    # replayed with one. Probing provenance has none of that false negative.
    #
    # Residual exposure, now genuinely hypothetical: a post-#1089-shaped document from a
    # third-party writer that emits provenance but not edits would report spurious divergence.
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
    traj: dict[str, Any], *, edits_recorded: bool = False
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
    ``extra.edits`` block found on any non-``agent`` step between it and the PREVIOUS ``agent``
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
    stamped since the provenance channel landed), not merely by position — a capability logs
    never had, since they
    carry no provenance. This is why divergence detection is safe here: a hook-ADDED user
    message between the agent step and the observation (e.g. the persistent ``<return_type>``
    reminder emitted when a return type is set) is NOT the observation and is correctly skipped,
    where a positional "first user step" rule would mispair it and read a false divergence on that
    turn. Only when no user step in the window carries observation provenance (a trace from
    before that channel, or an unstamped message) does it fall back to the positional
    first-user rule.
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
            pending_edits = (step.get("extra") or {}).get("edits", pending_edits)
            continue
        metrics = step.get("metrics") or {}
        pt = metrics.get("prompt_tokens")
        ct = metrics.get("completion_tokens")
        cost = metrics.get("cost_usd")
        responses.append(
            _SavedResponse(
                content=step.get("message"),
                prompt_tokens=pt,
                completion_tokens=ct,
                cost=cost,
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
        _build_invoke_tree_from_atif(child, edits_recorded=edits_recorded)
        for child in traj.get("subagent_trajectories", [])
    ]
    return _InvokeNode(
        responses=responses,
        children=children,
        expected_outputs=expected_outputs,
        expected_edits=expected_edits,
    )


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
    # A ValueError at load, matching ``_load_invoke_tree``'s multi-root refusal, rather than an
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
