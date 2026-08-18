"""ATIFReplay: resume agent runs from a saved ATIF trace.

Reads LLM responses from a saved ATIF trace and replays them as
``SupplyLLMResponse`` effects. Once all saved responses are exhausted, the hook
becomes a no-op and live LLM calls resume seamlessly.

An ATIF trace is the canonical replay/cost format. It carries real per-call
``prompt_tokens`` / ``completion_tokens`` / ``cost_usd`` (no reconstruction),
replays each invoke from its OWN per-invoke queue (nested ``invoke`` calls do
not shift a sibling/parent cursor), AND supports per-invoke divergence detection.

Usage::

    from jaz import invoke
    from jaz.hooks import ATIFReplay

    # Real tokens+cost, per-invoke replay
    with ATIFReplay("trace.atif.json"):
        result = invoke(task="...")

    # Same, but stop replaying an invoke as soon as its live REPL output diverges
    # from the saved trace; earlier calls and other invokes still replay for free.
    with ATIFReplay("trace.atif.json", detect_divergence=True):
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import SupplyLLMResponse
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQuerySend,
    REPLExecExit,
)
from jaz.hooks.events.base import Completed
from jaz.repl.types import Continue

logger = logging.getLogger(__name__)


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
    # LIST of that observation's message contents (a truncated turn is body + a separate advice
    # message), or ``None`` where the response had no following observation (e.g. a terminal
    # response). Compared message-by-message against the live render (#1055), so a body/advice
    # split difference is caught structurally rather than hidden by concatenation. Populated by
    # ``_load_invoke_tree``; default via default_factory so an ``_InvokeNode(...)`` built without
    # it (e.g. the progress tests) keeps working.
    expected_outputs: list[list[str] | None] = field(default_factory=list)
    # Per-invoke divergence latch: once True, this invoke's remaining LLM calls go live.
    diverged: bool = False

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


class ATIFReplay(Hook):
    """Replay saved LLM responses from an ATIF trace instead of calling the model.

    Each turn consumes the next saved response for the active invoke in order. Once an
    invoke's responses run out, its live LLM calls resume.

    The trace (as written by :class:`ATIFTrace`) is the canonical replay/cost format.
    Each invoke replays from its own per-invoke queue — one ``ATIFReplay`` covers the
    whole invoke tree, and a nested ``invoke`` never shifts a sibling's or parent's
    cursor. The per-call tokens and cost are the *real* metrics recorded in the trace
    (``prompt_tokens`` / ``completion_tokens`` / ``cost_usd``) — no reconstruction — so
    replayed calls report exact token totals as well as exact dollar cost.

    With ``detect_divergence=True``, replay stops for an invoke as soon as it stops
    matching: after each REPL exec the output is compared to the saved one, and on the
    first difference that invoke *diverges* — every LLM call from then on is live,
    including ones that would still have matched. Divergence is per-invoke, so a
    diverged invoke never desynchronizes a still-replaying sibling or parent. Use this
    to re-run an old trace under a change: everything before the first difference
    replays for free, and everything after it is paid for.

    Comparison is not byte-for-byte. The live output is abbreviated the way the original run
    abbreviated it, and REPL session IDs and traceback line numbers are normalized away, so
    edits elsewhere in the source do not read as divergence. It is **not** stable across a
    change to the truncation format: after such a change, every saved iteration whose output
    was truncated reports a divergence even on an otherwise identical rerun.

    Under ``with`` the replay covers invokes nested inside. Passed positionally, only that
    invoke replays; this hook supplies nothing to the invokes it nests, so they call the
    model live unless something else replays them.

    Args:
        path: Path to an ATIF ``.atif.json`` trace (as written by :class:`ATIFTrace`).
        detect_divergence: When ``True``, each saved response is paired with the REPL
            observation that followed its exec, and the first live exec whose output
            differs from the saved observation marks THAT invoke diverged, switching
            it (only it) to live LLM for the rest of the run.

    Raises:
        ValueError: If the trace holds more than one root trajectory (a multi-root
            trace has no single invoke tree to drive).
    """

    # Sole authority for the model's turn: two active Replays would both try to author every
    # response, so the seed-time check rejects that pairing (see Hook.supplies_llm_response) —
    # one Replay owns the per-invoke queues.
    supplies_llm_response = True

    # Mechanics: overrides at ``LLMQuerySend`` via ``SupplyLLMResponse`` — the supply
    # boundary, where the committed post-edit message list is in view — serving from the
    # active invoke's per-invoke queue (top of ``_invoke_stack``); divergence detection
    # observes ``REPLExecExit`` and latches ``_InvokeNode.diverged`` per invoke, so a diverged
    # invoke never desynchronizes a still-replaying sibling or parent.
    #
    # Cursor semantics are unchanged by the Enter→Send migration: LLMQuerySend fires exactly
    # once per query, on the dispatched AND the supplied path alike (the input commit is real
    # on both), so ``node.cursor`` still advances once per query and the divergence
    # detector's ``node.cursor - 1`` indexing stays aligned (pinned by
    # test_replay's cursor-alignment test).

    def __init__(
        self,
        path: str | Path,
        *,
        detect_divergence: bool = False,
    ) -> None:
        # Args are documented in the class docstring (the shipped reference page), not
        # duplicated here — the ATIFTrace pattern.
        self._init_from_tree(_load_invoke_tree(path), detect_divergence)

    @classmethod
    def _from_invoke_tree(
        cls,
        invoke_tree: _InvokeNode,
        *,
        detect_divergence: bool = False,
    ) -> ATIFReplay:
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
        self._invoke_stack: list[tuple[_InvokeNode, str]] = [
            (sentinel_root, "__sentinel__")
        ]
        total = self._count_tree_responses(invoke_tree)
        logger.info(
            f"ATIFReplay: {total} saved responses loaded for replay "
            f"(per-invoke queues, no cursor drift)"
        )

    @staticmethod
    def _count_tree_responses(node: _InvokeNode) -> int:
        return len(node.responses) + sum(
            ATIFReplay._count_tree_responses(c) for c in node.children
        )

    @staticmethod
    def _count_tree_replayed(node: _InvokeNode) -> int:
        """Responses consumed so far in this subtree — the sum of each node's
        own ``cursor`` (its position in its response list)."""
        return node.cursor + sum(
            ATIFReplay._count_tree_replayed(c) for c in node.children
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
        parent_node = self._invoke_stack[-1][0] if self._invoke_stack else None
        child = parent_node.next_child() if parent_node is not None else None
        self._invoke_stack.append((child, event.invoke_id))  # type: ignore[arg-type]
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list:
        if self._invoke_stack and self._invoke_stack[-1][1] == event.invoke_id:
            node, _ = self._invoke_stack.pop()
            if node is not None and not node.is_exhausted:
                logger.info(
                    f"ATIFReplay: invoke {event.invoke_id[:8]}… exited with "
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
        if node is None or node.is_exhausted:
            # This invoke has no more saved responses — go live.
            return []
        # Per-invoke divergence latch: once this invoke's live REPL output diverged from
        # the trace, all its remaining calls go live. Checked BEFORE serving so the
        # diverging response itself is not re-served.
        if node.diverged:
            return []
        r = node.next()
        if r is None:
            # is_exhausted was checked above, so this is unreachable;
            # guard anyway to satisfy the type checker and be safe.
            return []
        # ATIF carries real completion tokens, so the substitution below should be
        # a no-op; kept as a defensive guard against a trace with missing metrics.
        completion_tokens = r.completion_tokens
        if completion_tokens is None:
            if not self._warned_unknown_completion_tokens:
                logger.warning(
                    "ATIFReplay: completion_tokens is None for at least one "
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
        # AttributeError here (create_protocol supports protocols that implement only some
        # settings). Compare the FULL observation the agent saw MESSAGE-BY-MESSAGE — every message
        # render_observation returns (a truncated turn is the body PLUS a separate advice message)
        # against the recorded observation's messages — so a change in the advice, or in the number
        # of messages (body-only vs body+advice), diverges too, not just a change in the body.
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
                f"ATIFReplay: divergence detected at response #{idx} "
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
    # ATIFTrace writes a single root trajectory as an object, or an array when the
    # traced scope had multiple top-level invokes. ATIFReplay drives one root tree, so a
    # multi-root trace is out of scope here (each would need its own top-level frame).
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(
                f"ATIFReplay: expected a single root trajectory, got "
                f"{len(data)} in {path}. Multi-root ATIF traces are not supported."
            )
        data = data[0]
    root = _build_invoke_tree_from_atif(data)
    logger.info(f"ATIFReplay: loaded ATIF trace from {path}")
    return root


def _build_invoke_tree_from_atif(traj: dict[str, Any]) -> _InvokeNode:
    """Build a per-invoke ``_InvokeNode`` tree from one ATIF trajectory dict.

    One ``_SavedResponse`` per ``agent`` step (in order), carrying the step's real
    ``metrics`` (``prompt_tokens`` / ``completion_tokens`` / ``cost_usd`` → ``cost``, all
    real, so no ``None`` substitution). Children are built recursively from
    ``subagent_trajectories`` in stored (execution) order.

    Expected-output alignment (per-trajectory, positional assistant→following-observation
    pairing): each ``agent`` step is paired with the message of the FIRST ``user`` step that
    appears AFTER it and BEFORE the next ``agent`` step — that user step is the REPL
    observation of this response's exec (since #928 the observation message content *is* the
    REPL output). A terminal ``agent`` step, or one immediately followed by another ``agent``
    step with no intervening ``user`` step, has expected output ``None``. The leading
    ``system``/``user`` steps before the first ``agent`` step are the seed prompt, not
    observations, so they pair with nothing.

    Because a nested invoke's steps live in its OWN child trajectory (ATIF nests them under
    ``subagent_trajectories``, not inline), the parent's observations never include a child's
    — the pairing is naturally scoped to one trajectory.

    The observation is identified by **provenance** (``extra.provenance.kind == "observation"``,
    stamped since ATIF v1.7), not merely by position — a capability logs never had, since they
    carry no provenance. This is why divergence detection is safe here: a hook-ADDED user
    message between the agent step and the observation (e.g. the persistent ``<return_type>``
    reminder emitted when a return type is set) is NOT the observation and is correctly skipped,
    where a positional "first user step" rule would mispair it and read a false divergence on that
    turn. Only when no user step in the window carries observation provenance (a pre-v1.7 trace, or
    an unstamped message) does it fall back to the positional first-user rule.
    """
    steps = traj.get("steps", [])
    responses: list[_SavedResponse] = []
    expected_outputs: list[list[str] | None] = []
    n = len(steps)
    for i, step in enumerate(steps):
        if step.get("source") != "agent":
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
        # EVERY user step tagged observation provenance — a truncated turn is the body PLUS a
        # separate advice message, both stamped OBSERVATION by the driver — compared
        # message-by-message against the live render (#1055), so the advice participates in
        # divergence detection too. A positionally-first user step that is a hook reminder (e.g.
        # `<return_type>`) carries no observation provenance and is skipped; fall back to the first
        # user step (as a one-element list) only when NONE in the window is provenance-tagged (a
        # pre-v1.7 trace); ``None`` when there is no following observation at all.
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

    children = [
        _build_invoke_tree_from_atif(child)
        for child in traj.get("subagent_trajectories", [])
    ]
    return _InvokeNode(
        responses=responses,
        children=children,
        expected_outputs=expected_outputs,
    )


_REPL_SESSION_ID_RE = re.compile(r"<repl_[0-9a-f]+_\d+>")
_TRACEBACK_LINE_RE = re.compile(r'(File "[^"]+", line) \d+')


def _normalize_output(s: str) -> str:
    """Normalize a REPL output string for divergence comparison.

    The caller is expected to have already rendered the live output through the
    protocol's ``render_observation`` (#1055) so it matches the observation the
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
    s = _REPL_SESSION_ID_RE.sub("<repl_X>", s)
    s = _TRACEBACK_LINE_RE.sub(r"\1 N", s)
    return s.rstrip()


#: Deprecated aliases for the pre-rename spellings — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries these at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_legacy_hook_names_still_importable``). Two generations of renames:
#: ``ReplayHook`` → ``Replay`` (the family-wide ``Hook``-suffix drop) → ``ATIFReplay``
#: (format-qualified alongside its promotion, so a future token-native replay does not
#: collide with the bare name).
Replay = ATIFReplay
ReplayHook = ATIFReplay
