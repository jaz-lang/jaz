"""Hook that writes a browsable trajectory directory *as the run happens*.

The live counterpart to :func:`jaz._utils.trace_to_directory`, which expands a finished
``agent.atif.json`` in one pass at the end. :class:`TrajectoryDirectoryRecorder` writes each
message the moment it is seen, so a long run can be watched (``tail -f .../trace.md``) while it
is still going, and a run that dies leaves everything up to the failure on disk.

Usage::

    from jaz import invoke
    from jaz.hooks import TrajectoryDirectoryRecorder

    with TrajectoryDirectoryRecorder(output_dir="./trace"):
        result = invoke(task="Do something")

It is a pure observer: every handler returns ``[]``, so attaching it can never change the run it
records.

The *directory* layout is the same flat, sibling-based one the batch converter produces, so a trace
from either is navigable the same way -- with one exception, multi-root traces, where a converter that
sees the whole trace at once can name the roots in a way a streaming writer cannot (see ``_open_node``).
The per-message content format differs deliberately: each message is a fenced block preceded by a
provenance header, not the batch converter's ``## Role`` sections.

    <node>/trace.md         -- THIS agent's conversation only (sub-agents excluded), one fenced block
                               per message, appended live. Each block is preceded by the message's
                               provenance (role, ATIF ``kind``, stable id, iteration, buffer position,
                               persistence).
    <node>/overview.md      -- metadata + per-iteration REPL snippets + links to the sub-agents each
                               iteration launched; rewritten (it is small) whenever the node advances.
    <node>/iter<N>_sub<M>/  -- the sub-agent launched in REPL iteration N, index M -- a SIBLING
                               dir, itself the same structure recursively.

Buffer messages are rendered once, as each first appears -- except hook-*added* ones
(``kind=ADDED``), which the enter walk skips. Hook *edits* -- messages a hook ADDED this turn (a
compaction summary, a ``ContextWindowWarning`` delegation prompt) or DROPPED -- are rendered
separately, per turn, from ``LLMQuerySend``. That separate path is necessary because a *transient*
add (shown for one query, never kept in the buffer) never appears in a buffer snapshot; drops leave
no buffer trace; and a *persistent* add, though it does sit in the buffer, is skipped there so it is
not rendered twice. So this view shows every edit the batch converter's ``_render_hook_edits`` shows
-- persistent and transient adds, and drops -- each exactly once.

Message timing follows the same model :class:`TrajectoryRecorder` uses: a turn's assistant message is
stamped only at the *next* enter, so it -- and its stable provenance id -- become available one turn
late. It is deliberately emitted then (fully stamped) rather than at exit (id-less), because every
message should show its provenance. The single exception is the final assistant, which no enter
follows: it is flushed from invoke-exit with the content captured at its query-exit, carrying no id
(an honest ``?``).
"""

# Written for jaz-evals as a two-class pair -- a jaz-free `StreamingTraceWriter` driven by a
# `StreamingTraceDir(Hook)` -- because that repo's CI installs no jaz, so the formatting and I/O had
# to sit outside anything importing `jaz.hooks` to be testable there. In core that constraint does
# not exist, so the two are one class: the event handlers below translate each event into values, and
# the private methods under `# -- writing` do all the formatting and I/O.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from jaz._provenance import MessageKind, provenance_of
from jaz._string_utils import safe_repr, summarize_exception
from jaz.hooks._dispatcher import Hook
from jaz.hooks.effects import Effect
from jaz.hooks.events import Completed
from jaz.hooks.events._invoke import InvokeEnter, InvokeExit, InvokeSend
from jaz.hooks.events._llm_query import LLMQueryEnter, LLMQueryExit, LLMQuerySend
from jaz.repl._types import Raise, Return


class AddedEdit(TypedDict):
    """One message a hook added this turn (persistent, or transient like a ContextWindowWarning)."""

    role: str
    persistent: bool
    id: str | None
    position: int | None
    content: str


class DroppedEdit(TypedDict):
    """One message a hook dropped this turn (a compaction, a warning it retracted)."""

    role: str
    persistent: bool
    id: str | None
    position: int | None
    content: str


# ``position`` is :attr:`MessageAddRecord.position` / :attr:`MessageDropRecord.position`, the field
# these are read from, and the word every message coordinate here uses: a coordinate into an
# enter-time snapshot of the message list — a slot in it, or for an append the insertion point one
# past its end. Positions drift, since each snapshot renumbers around the edits that landed in
# between, and a message's own can equal an edit's for the same message, so a position is a
# coordinate rather than an identity. Which snapshot one indexes is not on the line that renders
# it: a message's is the enter where it was first seen, which can be later than the ``iter=``
# beside it (that names the message's own turn), while an edit's is the enter of the turn its
# section names.


# "First couple of lines" of REPL code/output shown per iteration in overview.md.
_SNIPPET_LINES = 3
# Heavy single-line cap on each `inputs`/`scope` value's repr in overview.md.
_OVERVIEW_VALUE_LEN = 200
# Single-line cap on a dropped message's content snippet in a hook-edits block: enough head to tell which
# message left (its own block above carries the whole thing), not the whole message.
_DROPPED_SNIPPET_LEN = 200
# Fence width for a message block. Three backticks is the default (the standard Markdown fence); a message
# whose content itself contains a run of >=3 backticks escalates to one longer than that run, so the fence a
# reader relies on to delimit the block can never be closed early by the content.
_MIN_FENCE = 3

# Message role -> Markdown heading label. `agent`/`assistant` both mean the model's turn.
_ROLE_LABEL = {
    "system": "System",
    "user": "User",
    "agent": "Assistant",
    "assistant": "Assistant",
}


def _role_label(role: str) -> str:
    return _ROLE_LABEL.get(role, role.capitalize() or "?")


def _is_agent(role: str) -> bool:
    """Whether `role` is the model's own turn -- the only role that gets a `py`-tagged fence."""
    return role in ("agent", "assistant")


# TODO(jaz-evals-internal#93): these helpers, the snippet/length constants, the role-label map, the
# overview.md layout in `_write_overview`, and the subtree-cost roll-up (`_subtree_cost` here /
# `_subtree_cost_usd` there) are duplicated from `trace_to_directory.py`; consolidate the shared
# format in one place.
def _one_line(text: str, max_len: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= max_len else collapsed[:max_len] + "…"


def _first_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    head = "\n".join(lines[:n])
    return head + ("\n…" if len(lines) > n else "")


def _short_id(msg_id: str) -> str:
    """First 8 hex chars of a provenance id -- enough to Ctrl-F within one trace. Callers omit the id
    field entirely when it is absent, so this is only reached with a real id."""
    return msg_id[:8]


def _fence(content: str) -> str:
    """The backtick fence for a message: `_MIN_FENCE` normally, widened past any backtick run in the
    content so the block cannot be closed early."""
    longest_run = max((len(m) for m in re.findall(r"`+", content)), default=0)
    return "`" * max(_MIN_FENCE, longest_run + 1)


def _provenance_line(
    role: str,
    kind: str | None,
    msg_id: str | None,
    iteration: int | None,
    position: int | None,
    persistent: bool | None,
) -> str:
    """The `### Role — kind=… · id=… · iter=… · position=… · persistent=…` header shown before a message,
    dropping whichever coordinates the trace does not carry."""
    meta: list[str] = []
    if kind is not None:
        meta.append(f"kind={kind}")
    if msg_id:
        # Omitted (not shown as `id=?`) when absent -- only the final assistant, which no enter follows
        # and so is never stamped; its content is complete, just un-ided.
        meta.append(f"id={_short_id(msg_id)}")
    if iteration is not None:
        meta.append(f"iter={iteration}")
    if position is not None:
        meta.append(f"position={position}")
    if persistent is not None:
        meta.append(f"persistent={str(persistent).lower()}")
    return f"### {_role_label(role)} — " + " · ".join(meta)


def _message_block(
    role: str,
    kind: str | None,
    msg_id: str | None,
    iteration: int | None,
    position: int | None,
    persistent: bool | None,
    content: str,
) -> str:
    """One message rendered for trace.md: provenance header, then the content in a fenced block
    (```py`` for the model's turn, a bare fence otherwise)."""
    fence = _fence(content)
    open_fence = f"{fence}py" if _is_agent(role) else fence
    return (
        "\n".join(
            [
                _provenance_line(role, kind, msg_id, iteration, position, persistent),
                open_fence,
                content,
                fence,
                "",
            ]
        )
        + "\n"
    )


def _edit_coords(msg_id: str | None, position: int | None) -> str:
    """`id=… · position=…` for a hook edit, dropping whichever it does not carry (an add has no id
    yet -- see ``on_llm_query_send``)."""
    parts: list[str] = []
    if msg_id:
        parts.append(f"id={_short_id(msg_id)}")
    if position is not None:
        parts.append(f"position={position}")
    return " · ".join(parts)


def _edits_block(
    iteration: int | None, added: list[AddedEdit], dropped: list[DroppedEdit]
) -> str:
    """A turn's hook edits rendered for trace.md: each ADDED message in a fenced block (labelled
    persistent/transient), then a compact list of DROPPED messages. Transient adds never enter the buffer,
    so this is the only place they -- and drops -- appear, matching the batch converter's hook-edits view."""
    it = "" if iteration is None else f" — iter {iteration}"
    lines: list[str] = [f"### Hook edits{it}", ""]
    for a in added:
        persistence = "persistent" if a["persistent"] else "transient"
        header = f"#### Added ({persistence}, {_role_label(a['role'])})"
        coords = _edit_coords(a["id"], a["position"])
        if coords:
            header += f" — {coords}"
        fence = _fence(a["content"])
        open_fence = f"{fence}py" if _is_agent(a["role"]) else fence
        lines.extend([header, open_fence, a["content"], fence, ""])
    if dropped:
        lines.extend(["#### Dropped", ""])
        for d in dropped:
            persistence = "persistent" if d["persistent"] else "transient"
            coords = _edit_coords(d["id"], d["position"])
            tags = f"{coords} · {persistence}" if coords else persistence
            snippet = _one_line(d["content"], _DROPPED_SNIPPET_LEN)
            lines.append(f"- {_role_label(d['role'])} ({tags}): {snippet}")
        lines.append("")
    return "\n".join(lines) + "\n"


@dataclass
class _Turn:
    """One REPL iteration's data, accumulated for overview.md."""

    # `code` and `output` hold only the first `_SNIPPET_LINES` lines, truncated at capture rather
    # than at render: that is all `_write_overview` ever shows, and holding the full text would keep
    # a second copy of the whole conversation (including every REPL output) in memory for the life of
    # the run, on top of the trace.md already written. The cost is that `_SNIPPET_LINES` is fixed
    # when the turn is recorded; nothing varies it per render.
    code: str
    output: str | None = None
    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class _Node:
    """Per-invoke writer state: where its files live, what it has already emitted, and the running
    material overview.md is rebuilt from."""

    node_dir: Path
    depth: int
    parent_invoke_id: str | None
    parent_repl_iteration: int | None
    label: str = "main"
    # `{name: (type_name, repr)}` for the invoke's committed inputs / resolved scope, shown in overview.
    inputs: dict[str, tuple[str, str]] = field(
        default_factory=dict[str, tuple[str, str]]
    )
    scope: dict[str, tuple[str, str]] = field(
        default_factory=dict[str, tuple[str, str]]
    )
    emitted_ids: set[str] = field(default_factory=set[str])
    turns: dict[int, _Turn] = field(default_factory=dict[int, "_Turn"])
    turn_order: list[int] = field(default_factory=list[int])
    # iteration -> [(sibling dir name, sub-agent task label)] for the overview's per-iteration sub links.
    subs_by_iter: dict[int, list[tuple[str, str]]] = field(
        default_factory=dict[int, list[tuple[str, str]]]
    )
    sub_counts: dict[int, int] = field(default_factory=dict[int, int])
    result_str: str = ""
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_iterations: int = 0
    # Child invoke ids, for rolling this node's `total_cost` up over the whole delegation subtree.
    children: list[str] = field(default_factory=list[str])

    @property
    def trace_path(self) -> Path:
        return self.node_dir / "trace.md"

    @property
    def overview_path(self) -> Path:
        return self.node_dir / "overview.md"


# Cap on a captured repr before the writer re-truncates it for display -- bounds memory/IO on a huge input.
_REPR_CAP = 2000


def _safe_repr(value: Any) -> str:
    """`repr(value)`, capped, and never raising -- an object's `__repr__` is user code."""
    # Any `repr` of a user object this hook renders goes through here: an unguarded one in the
    # exit handler would abort it and leave `overview.md` at "(in progress)" for good.
    return safe_repr(value, _REPR_CAP)


def _serialize_namespace(namespace: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """`{name: (type_name, repr)}` for an inputs/scope namespace, each repr capped."""
    return {k: (type(v).__name__, _safe_repr(v)) for k, v in namespace.items()}


def _task_label(inputs: dict[str, Any]) -> str:
    """A short heading label for an invoke, from its `task` input (else `"main"`)."""
    task = inputs.get("task")
    if isinstance(task, str) and task:
        return " ".join(task.split())[:70]
    return "main" if task is None else _safe_repr(task)[:70]


def _result_str(outcome: Any) -> str:
    """The invoke's outcome as `type: value` / `type: exception`."""
    # A `None` return renders like any other value (`NoneType: None`), the rule the ATIF recorder's
    # `outcome` follows. The batch converter shows a bare type there, but only because its
    # `value is None` tests for the flat `result_value` key being absent, not for a `None` return.
    if not isinstance(outcome, Completed):
        # A non-completed close (aborted/failed) carries the reason on the outcome itself.
        return f"{type(outcome).__name__}: {summarize_exception(outcome.exception)}"
    result = outcome.result
    match result:
        case Return(return_value=rv):
            return f"{type(rv).__name__}: {_safe_repr(rv)}"
        case Raise(exception=exc):
            # `summarize_exception` already leads with the class name.
            return summarize_exception(exc)
        case _:
            return type(result).__name__


@dataclass
class _InvokeAcc:
    """Per-invoke accumulation the events don't carry directly (running totals, the deferred assistant)."""

    scope: dict[str, tuple[str, str]] = field(default_factory=dict)
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_iterations: int = 0
    # The most recent turn's (iteration, assistant content), captured at query-exit and cleared once its
    # stamped copy reappears at the next enter. Whatever remains at invoke-exit is the final assistant.
    pending_assistant: tuple[int, str] | None = None


class TrajectoryDirectoryRecorder(Hook):
    """Write a browsable trajectory directory incrementally as a run executes.

    Each message is written the moment it is seen, so a long run can be watched
    (``tail -f <output_dir>/trace.md``) while it is still going, and a run that dies keeps
    everything up to the failure. A pure observer: it emits no effects, so it never influences the
    run it records.

    The two neighbours it is easily confused with: :class:`TrajectoryRecorder` writes machine-
    readable ATIF (Agent Trajectory Interchange Format) JSON, once, at scope end, and is what
    :class:`TrajectoryReplay` reads; the ``trace_to_directory`` utility writes this same kind of
    tree, but from a finished ATIF file after the run. Reach for this hook when the point is
    watching a run in progress.

    Args:
        output_dir: Directory to stream the trajectory into, created if absent. The first top-level
            invoke expands directly into it; sub-agents and any further roots get their own
            sub-directories. It is written into **as-is**: a trace already at that path is
            overwritten file by file as the new run reaches each one, and any sub-agent directories
            the old run left behind survive and interleave with the new ones. Give each run its own
            directory.

    Raises:
        OSError: If ``output_dir`` cannot be created or written to.
    """

    # State is keyed on invoke id throughout. An id this hook never saw an enter for is ignored (a
    # defensive no-op) rather than raising, so an event for an invoke that opened before the hook was
    # attached cannot crash the run it is only observing.
    #
    # NOT THREAD-SAFE. `_nodes`, `_acc`, `_root_count` and each node's `sub_counts` are read-modify-
    # written with no lock, so two invokes opening concurrently under one parent can take the same
    # `iter<N>_sub<M>` directory and clobber each other's `trace.md`, and two `_write_overview` calls
    # on one node can interleave. jaz does dispatch REPL execution off the main thread
    # (`repl/_base.py`) and documents the `ThreadPoolExecutor` case (`_scope.py`), so this is reachable,
    # not hypothetical -- it is simply not what the hook is built for today. Attach it to a run whose
    # delegation is sequential; a locked version is the fix if that stops being enough.

    def __init__(self, output_dir: str | Path):
        self._root = Path(output_dir)
        # Create the root eagerly so a bad path fails HERE, where the caller wrote it, rather than out
        # of `on_invoke_enter` -- the dispatcher catches a handler exception and logs it, so the run
        # would otherwise carry on and silently produce no trace at all.
        self._root.mkdir(parents=True, exist_ok=True)
        self._nodes: dict[str, _Node] = {}
        self._root_count = 0
        self._acc: dict[str, _InvokeAcc] = {}

    def setup(self) -> None:
        """Refuse reuse: a second run would overwrite the directory this one wrote.

        Raises :class:`~jaz.exceptions.HookActivationError` when this instance has already been
        activated. Construct a new recorder, with its own ``output_dir``, for each run.
        """
        # Refusing, not clearing. This used to drop its per-run maps in `teardown` so a second
        # `with` block would not find `_root_count == 1` and write its root into `task00/` — which
        # fixed the bookkeeping and left the real loss untouched: `output_dir` is not optional
        # here, every run streams into the same `trace.md`, and run 2 replaces run 1's file.
        # Measured: after a second activation the trace names only the second task. Clearing an
        # in-memory map does not un-write a file, so this is the "refuse" row of the policy on
        # `Hook` — unlike `TrajectoryRecorder`, which holds its output in memory when
        # `output_path` is None and must clear so a post-scope read still works.
        self.forbid_reactivation()

    # -- execution events ----------------------------------------------------------------------

    def on_invoke_enter(self, event: InvokeEnter) -> list[Effect]:
        # Open the node BEFORE reading the scope: the values are arbitrary user objects, and
        # `_serialize_namespace` calls `repr()` on each. It swallows a raising `__repr__` per value,
        # but ordering the directory creation first means even an unforeseen failure in the capture
        # costs this invoke's metadata rather than the whole subtree's trace.
        self._open_node(
            event.invoke_id,
            event.parent_invoke_id,
            event.parent_repl_iteration,
            event.depth,
        )
        self._acc[event.invoke_id] = _InvokeAcc(
            scope=_serialize_namespace(dict(event.scope))
        )
        return []

    def on_invoke_send(self, event: InvokeSend) -> list[Effect]:
        acc = self._acc.get(event.invoke_id)
        if acc is None:
            return []
        inputs = dict(event.inputs)
        # `label` is the conventional blackboard key for a name attached to an invoke, so an
        # explicit `MetaData(label=...)` wins here as it does in every other consumer. The
        # fallback differs from theirs on purpose: they need an identifier and settle for
        # `"main"`, while this is a heading, so an unlabelled invoke reads better as its task
        # text than as "main".
        label = str(event.blackboard.get("label") or _task_label(inputs))
        self._set_meta(
            event.invoke_id,
            label,
            _serialize_namespace(inputs),
            acc.scope,
        )
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        acc = self._acc.get(event.invoke_id)
        if acc is None:
            return []
        acc.total_iterations = max(acc.total_iterations, event.iteration + 1)
        for position, msg in enumerate(event.messages):
            if not isinstance(msg, dict):
                continue
            prov = provenance_of(msg)
            if prov is None:
                continue  # an un-stamped, hand-built message has no identity -- skip it
            if prov.kind is MessageKind.ADDED:
                # A hook-added message (a persistent add stays in the buffer, so it appears here too). It is
                # rendered from `extra.message_edits` in `on_llm_query_send`'s block instead, so skip it
                # here to avoid double-rendering -- exactly as TrajectoryRecorder does. (Transient adds never reach
                # the buffer at all, so this only affects persistent ones.)
                continue
            content = msg.get("content", "") or ""
            self._append_message(
                event.invoke_id,
                str(msg.get("role", "user")),
                prov.kind.value,
                prov.id,
                prov.iteration,
                position,
                prov.persistent,
                str(content),
            )
            if prov.kind is MessageKind.ASSISTANT and acc.pending_assistant is not None:
                if prov.iteration == acc.pending_assistant[0]:
                    acc.pending_assistant = (
                        None  # its stamped copy is now written; nothing to flush
                    )
            elif prov.kind is MessageKind.OBSERVATION and prov.iteration is not None:
                self._set_turn_output(event.invoke_id, prov.iteration, str(content))
        return []

    def on_llm_query_send(self, event: LLMQuerySend) -> list[Effect]:
        # This turn's hook edits, pre-resolved on the event (same source TrajectoryRecorder reads). Rendered here
        # because a transient add -- a ContextWindowWarning delegation prompt, say -- is shown for this one
        # query and never kept in the buffer, so it never reaches on_llm_query_enter's message walk. Emitted
        # after that turn's buffer messages and before its assistant (which appears at the next enter), so it
        # sits chronologically where the model saw it.
        if event.invoke_id not in self._acc:
            return []
        edits = event.message_edits
        if not (edits.adds or edits.drops):
            return []
        added: list[AddedEdit] = []
        for record in edits.adds:
            for msg in record.messages:
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                # TODO(#1553): usually ``None``, and when it is not, it is the wrong id.
                # ``Agent`` stamps ADDED provenance in ``_apply_persistent_message_edits``, which
                # runs after the query, so the agent never stamps an add at ``LLMQuerySend`` -- a
                # persistent one is stamped by the next enter, where it appears as a message of its
                # own. So an added message is normally rendered without an id, and two adds landing
                # in one slot render alike. The exception is a dict that arrived already stamped
                # (a hook re-adding a message the buffer holds), whose id is the earlier message's.
                # Unlike ``TrajectoryRecorder`` this has no backfill, so the two recorders disagree
                # about an add's id for the same run.
                prov = provenance_of(msg)
                added.append(
                    {
                        "role": str(msg.get("role", "user")),
                        "persistent": record.persistent,
                        "id": prov.id if prov is not None else None,
                        "position": record.position,
                        "content": content,
                    }
                )
        dropped: list[DroppedEdit] = []
        for record in edits.drops:
            prov = provenance_of(record.message)
            dropped.append(
                {
                    "role": str(record.message.get("role", "?")),
                    "persistent": record.persistent,
                    "id": prov.id if prov is not None else None,
                    "position": record.position,
                    "content": str(record.message.get("content", "")),
                }
            )
        self._append_edits(event.invoke_id, event.iteration, added, dropped)
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        acc = self._acc.get(event.invoke_id)
        if acc is None or not isinstance(event.outcome, Completed):
            return []  # a failed/aborted turn produced no assistant message and no billed step
        response = event.outcome.result
        cost = response.cost_usd or 0.0
        prompt_tokens = response.prompt_tokens or 0
        acc.total_cost += cost
        acc.total_prompt_tokens += prompt_tokens
        content = response.content or ""
        acc.pending_assistant = (event.iteration, content)
        self._record_turn(
            event.invoke_id,
            event.iteration,
            content,
            cost,
            prompt_tokens,
            response.completion_tokens or 0,
        )
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        acc = self._acc.get(event.invoke_id)
        if acc is None:
            return []
        # Flush the final assistant: no enter follows it, so it was never stamped/emitted above.
        if acc.pending_assistant is not None:
            iteration, content = acc.pending_assistant
            self._append_message(
                event.invoke_id,
                "assistant",
                MessageKind.ASSISTANT.value,
                None,
                iteration,
                None,
                None,
                content,
            )
            acc.pending_assistant = None
        self._close_node(
            event.invoke_id,
            _result_str(event.outcome),
            acc.total_cost,
            acc.total_prompt_tokens,
            acc.total_iterations,
        )
        # The accumulator is dead once its totals are on the node, and a long-lived driver can make
        # thousands of invokes under one `with`. Dropping it is safe because every handler already
        # treats a missing acc as a no-op -- the same path an event for an invoke this hook never saw
        # takes. `_nodes` has to stay: the subtree-cost roll-up walks closed children.
        del self._acc[event.invoke_id]
        return []

    # -- writing -----------------------------------------------------------------------------

    def _open_node(
        self,
        invoke_id: str,
        parent_invoke_id: str | None,
        parent_repl_iteration: int | None,
        depth: int,
    ) -> None:
        """Create the directory for a newly-entered invoke and start its (empty) trace.md."""
        parent = (
            self._nodes.get(parent_invoke_id) if parent_invoke_id is not None else None
        )
        if parent is not None and parent_repl_iteration is not None:
            # A sub-agent: a sibling dir under the parent, named by the launching iteration and its index
            # among that iteration's sub-agents (an iteration can launch more than one).
            m = parent.sub_counts.get(parent_repl_iteration, 0)
            parent.sub_counts[parent_repl_iteration] = m + 1
            dirname = f"iter{parent_repl_iteration}_sub{m}"
            node_dir = parent.node_dir / dirname
            parent.subs_by_iter.setdefault(parent_repl_iteration, []).append(
                (dirname, "main")
            )
        elif self._root_count == 0:
            # The first top-level invoke expands directly into the output dir -- the layout
            # `trace_to_directory` writes for a SINGLE root, and the shape of every trace with one
            # `jaz.invoke` in scope.
            node_dir = self._root
            self._root_count = 1
        else:
            # Subsequent roots get `task<n>`. This is THIS HOOK'S OWN naming, and it deliberately does
            # NOT match `trace_to_directory`, which writes `{i}_{label}` for every root once it
            # knows there is more than one. The divergence is forced rather than chosen: the batch
            # converter has the whole trace in hand and can name the first root retroactively, while a
            # streaming writer has to commit the first root's location before knowing whether a second
            # will ever arrive -- and defaulting it to a subdirectory would push every single-root run
            # (the common case) down a level. A task name is unavailable here too: it arrives at
            # `InvokeSend`, one event after the directory has to exist.
            #
            # So for multi-root traces the two layouts differ in both the first root's location and the
            # sibling names. `task<n>`, zero-padded to two digits, keeps the siblings sorting in start
            # order in a file browser, which is the property that matters for reading one.
            self._root_count += 1
            node_dir = self._root / f"task{self._root_count - 1:02d}"

        node_dir.mkdir(parents=True, exist_ok=True)
        node = _Node(
            node_dir=node_dir,
            depth=depth,
            parent_invoke_id=parent_invoke_id,
            parent_repl_iteration=parent_repl_iteration,
        )
        self._nodes[invoke_id] = node
        if parent is not None:
            parent.children.append(
                invoke_id
            )  # record the edge for subtree-cost roll-up
        node.trace_path.write_text("", encoding="utf-8")
        self._write_overview(node)
        if parent is not None:
            self._write_overview(
                parent
            )  # surface the new (as-yet-unnamed) sub link right away

    def _set_meta(
        self,
        invoke_id: str,
        label: str,
        inputs: dict[str, tuple[str, str]],
        scope: dict[str, tuple[str, str]],
    ) -> None:
        """Record the invoke's task label and committed inputs/scope (known at send, not enter)."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        node.label = label
        node.inputs = inputs
        node.scope = scope
        self._write_overview(node)
        # Update the parent's sub link so it shows this sub-agent's task, not the "main" placeholder.
        self._relabel_in_parent(node)

    def _close_node(
        self,
        invoke_id: str,
        result_str: str,
        total_cost: float,
        total_prompt_tokens: int,
        total_iterations: int,
    ) -> None:
        """Finalize the invoke: record its result and totals and write the last overview."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        node.result_str = result_str
        node.total_cost = total_cost
        node.total_prompt_tokens = total_prompt_tokens
        node.total_iterations = total_iterations
        self._write_overview(node)

    # -- content -----------------------------------------------------------------------------------

    def _append_message(
        self,
        invoke_id: str,
        role: str,
        kind: str | None,
        msg_id: str | None,
        iteration: int | None,
        position: int | None,
        persistent: bool | None,
        content: str,
    ) -> None:
        """Append one message to the node's trace.md, deduped by stable id (a buffer message reappears
        every later turn). A message with no id -- only the final assistant, unstamped because no enter
        follows it -- is always appended, since it is emitted exactly once from invoke-exit."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        if msg_id is not None:
            if msg_id in node.emitted_ids:
                return
            node.emitted_ids.add(msg_id)
        with node.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(
                _message_block(
                    role, kind, msg_id, iteration, position, persistent, content
                )
            )

    def _append_edits(
        self,
        invoke_id: str,
        iteration: int | None,
        added: list[AddedEdit],
        dropped: list[DroppedEdit],
    ) -> None:
        """Append a turn's hook edits to the node's trace.md: the messages a hook ADDED (persistent or
        transient) and the ones it DROPPED. Transient adds -- e.g. a `ContextWindowWarning` delegation
        prompt -- never enter the message buffer, so they are invisible to `_append_message` (which renders
        the buffer) and surface only here. This is the streaming analogue of the batch converter's
        `_render_hook_edits`, so both show every edit."""
        if not (added or dropped):
            return
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        with node.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(_edits_block(iteration, added, dropped))

    def _record_turn(
        self,
        invoke_id: str,
        iteration: int,
        code: str,
        cost_usd: float | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        """Record an iteration's REPL code + per-call metrics for overview.md (its output arrives later,
        via `_set_turn_output`)."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        if iteration not in node.turns:
            node.turn_order.append(iteration)
        node.turns[iteration] = _Turn(
            code=_first_lines(code, _SNIPPET_LINES),
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._write_overview(node)

    def _set_turn_output(self, invoke_id: str, iteration: int, output: str) -> None:
        """Fill in an iteration's REPL output (its observation, seen at the following enter)."""
        node = self._nodes.get(invoke_id)
        if node is None:
            return
        turn = node.turns.get(iteration)
        if turn is not None:
            turn.output = _first_lines(output, _SNIPPET_LINES)
            self._write_overview(node)

    # -- overview rendering ------------------------------------------------------------------------

    def _relabel_in_parent(self, node: _Node) -> None:
        parent = (
            self._nodes.get(node.parent_invoke_id)
            if node.parent_invoke_id is not None
            else None
        )
        if parent is None or node.parent_repl_iteration is None:
            return
        subs = parent.subs_by_iter.get(node.parent_repl_iteration)
        if not subs:
            return
        expected = node.node_dir.name
        parent.subs_by_iter[node.parent_repl_iteration] = [
            (d, node.label if d == expected else sub_label) for d, sub_label in subs
        ]
        self._write_overview(parent)

    def _subtree_cost(self, node: _Node) -> float:
        """This node's own cost plus every descendant's -- the whole delegation subtree's spend.

        Correct to read at any node's `_close_node`: jaz delegation is synchronous, so every descendant
        invoke has already closed (its `total_cost` set) before an ancestor closes. Read mid-flight (an
        intermediate overview write) it undercounts still-open children, but the file is rewritten at
        close, so the final overview is exact.

        One residual undercount survives to the final write: a descendant whose `_close_node` never fires
        -- an interrupted run (budget/context/crash), or a sub-agent that dies without an invoke-exit --
        keeps `total_cost=0.0` and contributes nothing. That is the same downward direction as the
        mid-flight case (never an overcount), and early stopping is an ordinary outcome here.
        """
        return node.total_cost + sum(
            self._subtree_cost(child)
            for cid in node.children
            if (child := self._nodes.get(cid)) is not None
        )

    def _write_overview(self, node: _Node) -> None:
        iters = node.total_iterations or len(node.turn_order)
        lines: list[str] = [
            f"# {node.label}",
            "",
            f"**Depth:** {node.depth}  |  **Iterations:** {iters}  |  "
            f"**Total cost:** ${node.total_cost:.4f}  |  **Prompt tokens:** {node.total_prompt_tokens}",
        ]
        # `total_cost` above mirrors ATIF's per-trajectory `total_cost_usd` (self-cost only). When this
        # node delegated, also show the subtree roll-up so the figure agrees with the run's authoritative
        # cost; omitted for a leaf, where it would just repeat the self-cost.
        # Gated on delegation, not on subtree != self, by decision: a node that delegated to ~zero-cost
        # children would show a line equal to its self-cost, but that case is negligible and the presence
        # check is simpler than comparing rounded costs.
        if node.children:
            lines.append(
                f"**Subtree cost (incl. subagents):** ${self._subtree_cost(node):.4f}"
            )
        lines.extend([f"**Result:** {node.result_str or '(in progress)'}", ""])
        for section, namespace in (("Inputs", node.inputs), ("Scope", node.scope)):
            if not namespace:
                continue
            lines.extend([f"## {section}", ""])
            lines.extend(
                f"- `{name}` ({typ}): {_one_line(rep, _OVERVIEW_VALUE_LEN)}"
                for name, (typ, rep) in namespace.items()
            )
            lines.append("")

        lines.extend(["## Iterations", ""])
        for iteration in node.turn_order:
            turn = node.turns[iteration]
            lines.extend([f"### Iteration {iteration}", ""])
            if turn.cost_usd is not None:
                toks = f"{turn.prompt_tokens or 0}->{turn.completion_tokens or 0}"
                lines.extend([f"*cost ${turn.cost_usd:.4f} | tokens {toks}*", ""])
            lines.extend(["REPL code:", "```", turn.code, "```"])
            if turn.output:
                lines.extend(["REPL output:", "```", turn.output, "```"])
            for dirname, label in node.subs_by_iter.get(iteration, []):
                lines.append(
                    f"Sub-agent: [`{dirname}/`]({dirname}/overview.md) — {label}"
                )
            lines.append("")

        node.overview_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
