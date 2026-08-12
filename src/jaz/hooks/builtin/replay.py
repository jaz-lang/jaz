"""Replay hook for resuming agent runs from saved conversation logs.

Reads LLM responses from a saved conversation history (ConversationHistory
JSON) or from a .log + costs.json pair, and replays them as
``OverrideResponse`` effects. Once all saved responses are exhausted, the
hook becomes a no-op and live LLM calls resume seamlessly.

Usage::

    # From conversation history JSON (preferred)
    with Replay.from_conversation_history("conversation.json"):
        result = jaz.invoke(...)

    # From .log + costs.json (fallback for runs without ConversationHistory)
    with Replay.from_log_and_costs("0_0.log", "0_0_costs.json"):
        result = jaz.invoke(...)

    # Same but with divergence detection — on the first REPL exec whose output
    # differs from the saved log (e.g., because a new repl_input_validator now
    # rejects an input the original run executed), the affected invoke is
    # marked diverged and all of its remaining LLM calls go live::

        with Replay.from_log_and_costs(
            "0_0.log", "0_0_costs.json", detect_divergence=True
        ):
            result = jaz.invoke(...)
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import OverrideResponse
from jaz.hooks.events import InvokeEnter, InvokeExit, LLMQueryEnter, REPLExecExit
from jaz.repl.types import Continue, Raise, Return
from jaz.string_utils import abbreviate_string

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
    total_tokens: int | None = None
    cost: float | None = None


class Replay(Hook):
    """Replay saved LLM responses instead of calling the model.

    Each turn consumes the next saved response in order. Once they run out, live LLM calls
    resume.

    With ``expected_repl_outputs``, replay stops as soon as the run stops matching: after each
    REPL exec the output is compared to the saved one, and on the first difference that invoke
    *diverges* — every LLM call from then on is live, including ones that would still have
    matched. Use this to re-run an old trace under a change: everything before the first
    difference replays for free, and everything after it is paid for.

    Comparison is not byte-for-byte. The live output is abbreviated the way the original run
    abbreviated it, and REPL session IDs and traceback line numbers are normalized away, so
    edits elsewhere in the source do not read as divergence. It is **not** stable across a
    change to the truncation format: after such a change, every saved iteration whose output
    was truncated reports a divergence even on an otherwise identical rerun.

    **Single-threaded only.** Responses given here are consumed as one flat sequence, so
    concurrent threads or nested invokes consume them out of order and replay incorrectly. Use
    :meth:`from_log_as_tree` for a run with nested invokes: it gives each invoke its own queue,
    at the cost of divergence detection, which it does not support.

    Replayed calls report the saved ``cost`` and ``prompt_tokens`` exactly. A saved response
    with no completion-token count — anything from a reconstructed ``costs.json`` — replays as
    zero completion tokens, so token totals are understated for that portion of the run while
    the dollar cost stays right.

    Under ``with`` the replay covers invokes nested inside (subject to the above). Passed
    positionally, only that invoke replays; this hook supplies nothing to the invokes it
    nests, so they call the model live unless something else replays them.

    Args:
        responses: Ordered list of saved LLM responses to replay.
        expected_repl_outputs: Saved REPL output text, one entry per response. ``None``
            entries never trigger divergence. Omit to disable divergence detection and
            replay straight through.
    """

    # Mechanics: overrides at ``LLMQueryEnter`` via ``OverrideResponse``, from a positional
    # cursor into ``responses``; divergence detection observes ``REPLExecExit``.
    #
    # The single-threaded limitation is the cursor's: per-invoke diverged state is tracked
    # correctly, but the cursor advances *globally*, so a diverged invoke sharing the queue
    # with a still-replaying one desynchronizes the latter. Correct nested-invoke replay would
    # need responses keyed by ``(invoke_id, iteration)`` and looked up structurally rather than
    # by position.

    def __init__(
        self,
        responses: list[_SavedResponse],
        expected_repl_outputs: list[str | None] | None = None,
        *,
        invoke_tree: _InvokeNode | None = None,
    ) -> None:
        """Create a Replay.

        Args:
            responses: Flat ordered list of saved responses (flat-cursor mode).
                Mutually exclusive with ``invoke_tree``.
            expected_repl_outputs: Per-response expected REPL outputs for
                divergence detection.  Only valid in flat-cursor mode.
            invoke_tree: Root node of a per-invoke response tree (tree-cursor
                mode).  When provided, ``responses`` must be empty and
                ``expected_repl_outputs`` must be None.  In this mode, each
                nested ``jaz.invoke`` call gets its own response queue so that
                extra iterations inside one invoke do not shift the cursor of
                sibling or parent invokes.
        """
        if invoke_tree is not None:
            if responses:
                raise ValueError(
                    "Replay: invoke_tree and responses are mutually exclusive"
                )
            # Divergence detection matches against a single flat
            # expected_repl_outputs sequence, which has no coherent meaning once
            # replay is driven by the per-invoke tree cursor; unifying the two is
            # left as future work.
            if expected_repl_outputs is not None:
                raise ValueError(
                    "Replay: divergence detection is not supported in "
                    "tree-cursor mode (invoke_tree)"
                )
        if expected_repl_outputs is not None and len(expected_repl_outputs) != len(
            responses
        ):
            raise ValueError(
                f"Replay: expected_repl_outputs length "
                f"({len(expected_repl_outputs)}) must equal responses length "
                f"({len(responses)})"
            )
        self._responses = responses
        self._expected_repl_outputs = expected_repl_outputs
        self._cursor = 0
        self._total = len(responses)
        # Per-invoke divergence state. Once an invoke_id is added here, all
        # subsequent LLM calls for that invoke fall through to live LLM.
        self._diverged_invokes: set[str] = set()
        # Latch so we warn at most once per hook lifetime when replaying a
        # reconstructed costs.json that has None completion/total tokens.
        self._warned_unknown_completion_tokens = False

        # Tree-cursor mode: stack of (_InvokeNode, invoke_id) pairs.
        # The top of the stack is the currently-active invoke.
        self._invoke_tree = invoke_tree
        self._invoke_stack: list[tuple[_InvokeNode, str]] = []

        if invoke_tree is not None:
            # Wrap the root invoke node in a sentinel so that the first
            # InvokeEnter event (for the top-level invoke) correctly pops
            # invoke_tree from the sentinel's children and pushes it as the
            # active frame.  Without the sentinel wrapper, InvokeEnter for the
            # top-level invoke would mistakenly treat invoke_tree itself as the
            # parent and pop invoke_tree.children[0] (the first nested invoke)
            # as the active frame — causing the top-level invoke's LLM calls
            # to be served from the first child's queue instead of its own.
            sentinel_root = _InvokeNode(responses=[], children=[invoke_tree])
            self._invoke_stack = [(sentinel_root, "__sentinel__")]
            total = self._count_tree_responses(invoke_tree)
            logger.info(
                f"Replay: {total} saved responses loaded for replay "
                f"(tree-cursor mode — per-invoke queues, no cursor drift)"
            )
        elif self._total > 0:
            mode = (
                "with divergence detection"
                if expected_repl_outputs is not None
                else "pass-through"
            )
            logger.info(
                f"Replay: {self._total} saved responses loaded for replay ({mode})"
            )

    @staticmethod
    def _count_tree_responses(node: _InvokeNode) -> int:
        return len(node.responses) + sum(
            Replay._count_tree_responses(c) for c in node.children
        )

    @staticmethod
    def _count_tree_replayed(node: _InvokeNode) -> int:
        """Responses consumed so far in this subtree — the sum of each node's
        own ``cursor`` (its position in its response list)."""
        return node.cursor + sum(Replay._count_tree_replayed(c) for c in node.children)

    @property
    def is_exhausted(self) -> bool:
        """Whether all saved responses have been replayed."""
        if self._invoke_tree is not None:
            # Tree mode: exhausted once every node has consumed its responses.
            # Computed from the tree cursors, not the (unused) flat cursor.
            return self._count_tree_replayed(
                self._invoke_tree
            ) >= self._count_tree_responses(self._invoke_tree)
        return self._cursor >= self._total

    @property
    def replayed_count(self) -> int:
        """Number of responses replayed so far (summed over the tree cursors in
        tree mode, or the flat cursor otherwise)."""
        if self._invoke_tree is not None:
            return self._count_tree_replayed(self._invoke_tree)
        return self._cursor

    # Event handling is split across the typed per-event handlers below rather
    # than a single ``on_any`` catch-all, so the framework router (#691)
    # dispatches each event to the right handler with no match/isinstance
    # boilerplate. Each handler branches on cursor mode (tree vs flat) internally.

    def on_invoke_enter(self, event: InvokeEnter) -> list:
        # Tree-cursor mode only: pop the next child from the parent node and push
        # it as the new active frame. If the parent has no more children (e.g.,
        # the replay had fewer invokes than the live run), push None so LLM calls
        # fall through to live. Flat mode ignores invoke structure.
        if self._invoke_tree is None:
            return []
        parent_node = self._invoke_stack[-1][0] if self._invoke_stack else None
        child = parent_node.next_child() if parent_node is not None else None
        self._invoke_stack.append((child, event.invoke_id))  # type: ignore[arg-type]
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list:
        if self._invoke_tree is None:
            return []
        if self._invoke_stack and self._invoke_stack[-1][1] == event.invoke_id:
            node, _ = self._invoke_stack.pop()
            if node is not None and not node.is_exhausted:
                logger.info(
                    f"Replay: invoke {event.invoke_id[:8]}… exited with "
                    f"{len(node.responses) - node.cursor} response(s) "
                    f"unused in its queue (live run ran fewer iterations "
                    f"than source)"
                )
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list:
        invoke_id = event.invoke_id
        if self._invoke_tree is not None:
            # Tree-cursor mode: serve from the top-of-stack node's own queue.
            if not self._invoke_stack:
                return []
            node, _ = self._invoke_stack[-1]
            if node is None or node.is_exhausted:
                # This invoke has no more saved responses — go live.
                return []
            r = node.next()
            if r is None:
                # is_exhausted was checked above, so this is unreachable;
                # guard anyway to satisfy the type checker and be safe.
                return []
            # completions/total_tokens substitution same as flat mode
            completion_tokens = r.completion_tokens
            total_tokens = r.total_tokens
            if completion_tokens is None or total_tokens is None:
                if not self._warned_unknown_completion_tokens:
                    logger.warning(
                        "Replay: completion_tokens / total_tokens "
                        "are None for at least one replayed call "
                        "(reconstructed costs.json). Substituting 0."
                    )
                    self._warned_unknown_completion_tokens = True
                completion_tokens = completion_tokens or 0
                total_tokens = total_tokens or (
                    (r.prompt_tokens or 0) + (completion_tokens or 0)
                )
            return [
                OverrideResponse(
                    content=r.content,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=r.cost,
                )
            ]

        # Flat-cursor mode (original path).
        if self.is_exhausted:
            return []
        if invoke_id in self._diverged_invokes:
            return []
        r = self._responses[self._cursor]
        self._cursor += 1
        if self._cursor == self._total:
            logger.info(
                f"Replay: replay exhausted after {self._total} "
                f"responses — switching to live LLM calls"
            )
        # When replaying from a costs.json reconstructed by
        # `scripts/reconstruct_costs_from_log.py`, completion_tokens
        # and total_tokens are None: reasoning-model APIs bill hidden
        # chain-of-thought as part of completion_tokens, and those
        # tokens never appear in the log content, so tiktoken-counting
        # the visible response gives a known-wrong number (~35% of
        # truth on gpt-5). The framework's cost_tracker contract
        # (agent.py:247) requires non-None ints though, so substitute
        # 0 here and warn once. The dollar `cost` and `prompt_tokens`
        # remain exact.
        completion_tokens = r.completion_tokens
        total_tokens = r.total_tokens
        if completion_tokens is None or total_tokens is None:
            if not self._warned_unknown_completion_tokens:
                logger.warning(
                    "Replay: completion_tokens / total_tokens "
                    "are None for at least one replayed call "
                    "(reconstructed costs.json). Substituting 0 so "
                    "the cost-tracker contract holds; "
                    "total_llm_completion_tokens will be understated "
                    "for the replayed portion. `cost` and "
                    "`prompt_tokens` are exact."
                )
                self._warned_unknown_completion_tokens = True
            completion_tokens = completion_tokens or 0
            total_tokens = total_tokens or ((r.prompt_tokens or 0) + completion_tokens)
        return [
            OverrideResponse(
                content=r.content,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=r.cost,
            )
        ]

    def on_repl_exec_exit(self, event: REPLExecExit) -> list:
        # Flat-cursor divergence detection only: it matches against a single flat
        # expected_repl_outputs sequence, which has no coherent meaning under the
        # tree cursor (see the constructor guard).
        if self._invoke_tree is not None:
            return []
        if self._expected_repl_outputs is None:
            return []
        invoke_id = event.invoke_id
        if invoke_id in self._diverged_invokes:
            return []
        # The exec_result here is the REPL exec that followed the LLM
        # response at index cursor - 1 (the most recently replayed
        # response). cursor == 0 means no response has been replayed
        # yet, so there's nothing to compare against.
        idx = self._cursor - 1
        if idx < 0:
            return []
        expected = self._expected_repl_outputs[idx]
        if expected is None:
            return []
        # The saved observation body holds the output the agent
        # *abbreviated* before forming the user message (see
        # agent.py: render_observation(..., self.config) ->
        # protocol/default.py abbreviate_string(output, max_repl_output_length,
        # truncation_prefix_ratio)). The exec_result.output we observe
        # here is the raw, untruncated output. Apply the same
        # abbreviation now so identical execs compare equal regardless
        # of length.
        #
        # Use `event.config` — the DEPTH-RESOLVED effective config for the invoke
        # that produced this event (stamped on every Event since #463) — NOT ambient
        # `get_config()`. They differ whenever the invoke ran under a per-depth
        # (`configure_by_depth` / `ConfigOverrideByDepth`) or per-call `ConfigOverride`
        # for `max_repl_output_length` / `truncation_prefix_ratio`: `get_config()` is
        # depth-less (post-#727 it folds `_default_config` + layers but never applies a
        # DepthLayer), so reading it here would re-abbreviate with a different threshold
        # than the live agent used and flag a false divergence, silently defeating replay
        # determinism (#826 §1). `event.config` is exactly the `self.config` the agent
        # abbreviated with, so the comparison is threshold-consistent by construction.
        #
        # KNOWN LIMITATION — this comparison is threshold-consistent but not *format*-stable.
        # It re-abbreviates with whatever `abbreviate_string` renders today and diffs that
        # against text a past run wrote, so changing the truncation format invalidates every
        # log recorded before the change: each iteration whose output was truncated reports a
        # divergence even on a byte-identical rerun. `_normalize_output` does not paper over
        # this — it strips REPL session IDs and traceback line numbers, not the marker.
        #
        # Accepted deliberately when the format moved to a single in-place
        # `[N characters omitted]` splice. The alternative — normalizing the marker away on
        # both sides — would restore cross-format replay only by making the omitted count
        # invisible to divergence detection, i.e. a run that drops a different amount of
        # output would start comparing equal. Losing that signal is worse than losing replay
        # of logs recorded before a format change, which is a one-off cost paid at the change.
        cfg = event.config
        actual_full = _exec_result_output(event.exec_result)
        # Ask the *protocol*, not the config, because this reproduces the abbreviation
        # `DefaultProtocol.render_observation` applied — the two must use the same numbers or
        # every turn reads as diverged. Reading `cfg.protocol.*` was only equivalent while the
        # config carried a value for every setting; now that it records just what was authored,
        # an unset one has no value there at all and only the constructed codec knows the
        # effective number.
        #
        # No construction: `cfg.protocol` IS the codec this run uses, so the numbers are the
        # ones it applied by definition rather than by reconstruction. This used to build a
        # protocol per `REPLExecExit` from a `{tag, params}` description.
        #
        # KNOWN LIMITATION (TODO(#1055)): `max_output_length` / `truncation_prefix_ratio` are
        # `DefaultProtocol` attributes, not part of the `InteractionProtocol` ABC (which declares
        # only `parse`, `render_observation`, `render_initial_message_list`,
        # `build_history_entry`). A registered protocol that does not happen to define them
        # raises `AttributeError` here, once per exec — and `create_protocol` explicitly supports
        # protocols implementing only some settings, so partial implementations are expected
        # rather than exotic.
        #
        # A `getattr(..., 10000)` fallback is NOT the fix: it reinstates exactly the
        # silently-wrong number this read replaced, and a wrong number makes every turn read as
        # diverged. The shape that closes it is putting the abbreviation behind the codec —
        # `protocol.abbreviate_observation(text)` — so replay reproduces rendering by *calling*
        # it rather than re-deriving it from two attributes.
        #
        # Accepted rather than mitigated (user call, 2026-08-09), over degrading to
        # skip-divergence-detection when the attributes are absent. That would have left no
        # known-broken path, but it trades a loud failure for a silently reduced signal in a
        # subsystem that is already marked experimental (`jaz/hooks/__init__.py`) and is expected
        # to be reworked substantially — at which point #1055 lands with it. A crash that names
        # the missing attribute is the more useful failure until then.

        protocol = cfg.protocol
        actual_abbrev, _ = abbreviate_string(
            actual_full,
            # type: ignore[attr-defined] — see the KNOWN LIMITATION above and TODO(#1055).
            # These are `DefaultProtocol` attributes, not part of the ABC. Storing the codec on
            # the config turned that from a runtime `AttributeError` into a *static* error,
            # which is strictly better: the gap is now visible without running the hook.
            max_length=protocol.max_output_length,  # type: ignore[attr-defined]
            prefix_ratio=protocol.truncation_prefix_ratio,  # type: ignore[attr-defined]
        )
        actual = _normalize_output(actual_abbrev)
        expected_norm = _normalize_output(expected)
        if actual != expected_norm:
            self._diverged_invokes.add(invoke_id)
            logger.info(
                f"Replay: divergence detected at response #{idx} "
                f"for invoke {invoke_id} — remaining LLM calls for "
                f"this invoke will go live"
            )
        return []

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_conversation_history(cls, path: str | Path) -> Replay:
        """Create a Replay from a ConversationHistory JSON file.

        The JSON must follow the structure produced by
        ``ConversationHistory``: a list of root invoke nodes, each
        containing ``repl_iterations`` with ``llm_response`` dicts.
        """
        # TODO: Deprecate and replace with ATIF format when conversation history
        # gets deprecated
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        responses: list[_SavedResponse] = []
        _extract_responses_from_tree(data, responses)

        logger.info(f"Replay: loaded {len(responses)} responses from {path}")
        return cls(responses)

    @classmethod
    def from_log_and_costs(
        cls,
        log_path: str | Path,
        costs_path: str | Path,
        detect_divergence: bool = False,
    ) -> Replay:
        """Create a Replay by parsing a .log file and costs.json.

        Extracts assistant message content from the log file and joins
        with per-call token counts from costs.json.

        Args:
            log_path: Path to the FileLogger ``.log`` file.
            costs_path: Path to the matching ``*_costs.json`` file.
            detect_divergence: When ``True``, also take each assistant
                message's *following* user message as the observation body
                (an observation carries no framing since #928), and
                pass the resulting list to the constructor as
                ``expected_repl_outputs``. The hook will then mark an invoke
                as diverged on the first REPL exec whose output differs from
                the saved block, and switch to live LLM for the rest of that
                invoke. Default ``False`` preserves pure pass-through replay.

        TODO: unify with from_log_as_tree
        """
        log_path = Path(log_path)
        costs_path = Path(costs_path)

        # Single pass over the log → (assistant_content, repl_output) per iteration,
        # positionally aligned by construction (see _parse_iterations_from_log).
        iterations = _parse_iterations_from_log(log_path)
        assistant_contents = [content for content, _ in iterations]

        # Parse cost data
        with open(costs_path) as f:
            costs_data = json.load(f)
        llm_calls = costs_data.get("llm_calls", [])

        if len(assistant_contents) != len(llm_calls):
            raise ValueError(
                f"Replay: assistant message count ({len(assistant_contents)}) "
                f"!= llm_calls count ({len(llm_calls)}) in costs.json — "
                f"token counts may be misaligned"
            )

        # Join by index — both are ordered by iteration
        responses: list[_SavedResponse] = []
        for content, call in zip(assistant_contents, llm_calls, strict=True):
            responses.append(
                _SavedResponse(
                    content=content,
                    prompt_tokens=call.get("prompt_tokens"),
                    completion_tokens=call.get("completion_tokens"),
                    total_tokens=call.get("total_tokens"),
                    cost=call.get("cost"),
                )
            )

        expected_repl_outputs: list[str | None] | None = None
        if detect_divergence:
            # Projected from the same single-pass iterations as assistant_contents, so
            # outputs stay aligned with responses by construction — no runtime length
            # check needed (#587).
            expected_repl_outputs = [output for _, output in iterations]

        logger.info(
            f"Replay: loaded {len(responses)} responses from {log_path} + {costs_path}"
        )
        return cls(responses, expected_repl_outputs=expected_repl_outputs)

    @classmethod
    def from_log_as_tree(cls, log_path: str | Path) -> Replay:
        """Create a Replay with a per-invoke cursor tree from a .log file.

        Unlike ``from_log_and_costs`` (which uses a single flat cursor), this
        factory builds a tree of ``_InvokeNode`` objects that mirrors the
        nested invoke structure in the log.  Each invoke gets its own response
        queue, so extra iterations inside one invoke (e.g., a return_validator
        firing and causing a retry) consume only that invoke's slots and do NOT
        shift the cursor of sibling or parent invokes.

        This is the correct mode for replaying TTSI runs where the meta-agent
        makes dozens of nested ``jaz.invoke`` calls and per-invoke isolation is
        critical for faithful reproduction.

        Token counts and costs are derived the same way as in
        ``from_log_and_costs`` (tiktoken prompt_tokens + exact dollar cost from
        the log line; completion_tokens left None for reasoning models).

        TODO: unify with from_log_and_costs
        """
        log_path = Path(log_path)
        root = _build_invoke_tree_from_log(log_path)
        return cls([], invoke_tree=root)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_responses_from_tree(
    nodes: list[dict[str, Any]], out: list[_SavedResponse]
) -> None:
    """Recursively extract LLM responses from a conversation history tree.

    Traverses in execution order: for each invoke node, iterates through
    repl_iterations in order. For each iteration, appends the LLM response,
    then recurses into any children (nested invokes).
    """
    if isinstance(nodes, dict):
        # ConversationHistory always writes a list, but handle a single
        # root dict defensively in case the JSON was hand-edited or produced
        # by external tooling.
        nodes = [nodes]

    for node in nodes:
        iterations = node.get("repl_iterations", [])
        for iteration in iterations:
            llm_resp = iteration.get("llm_response")
            if llm_resp is not None:
                pt = llm_resp.get("prompt_tokens")
                ct = llm_resp.get("completion_tokens")
                # Prefer the provider-reported total (it may include reasoning /
                # cached tokens counted separately from prompt+completion), and
                # only derive pt+ct as a fallback. This keeps this factory's
                # token accounting consistent with from_log_and_costs, which
                # reads total_tokens directly from costs.json.
                total = llm_resp.get("total_tokens")
                if total is None and pt is not None and ct is not None:
                    total = pt + ct
                out.append(
                    _SavedResponse(
                        content=llm_resp.get("content"),
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        total_tokens=total,
                        cost=llm_resp.get("cost"),
                    )
                )
            # Recurse into children (nested invokes spawned during this iteration)
            children = iteration.get("children", [])
            if children:
                _extract_responses_from_tree(children, out)


# Regex to match log lines like:
#   2026-05-21 04:46:51 [Agent] message: {'role': 'assistant', 'content': '...'}
_AGENT_MESSAGE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[Agent\] message: (.+)$"
)

# Regex to match the start of an invoke. When a kill+restart happens with the
# same ``run_id``, ``FileLogger`` opens the per-task .log in append mode (and
# ``CostTracker.save_to_json`` overwrites the per-task costs.json), so the
# .log holds *every* attempt concatenated while costs.json holds only the
# latest. Resetting at each ``invoke enter`` marker discards earlier-attempt
# data and keeps the log parser aligned with costs.json. Single-attempt logs
# (the common case) hit one reset on line 1 with empty state — same result as
# never resetting.
_INVOKE_ENTER_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[Agent\] invoke enter:"
)


# NOTE: there is no observation *pattern* to match anymore. #928 stripped the ``<repl_output>``
# wrapper, the ``<error>`` block, and finally the ``**REPL output (iteration #N):**`` header, so a
# ``DefaultProtocol`` observation message's content is exactly the abbreviated REPL output with no
# framing at all. Extraction is therefore **positional**, not syntactic: the observation is the
# first ``user`` message following an ``assistant`` message (the truncation advice, when present,
# is a later message and the per-turn footer was removed in #634).
#
# The cost of that simplicity: an observation is no longer self-identifying, so this parser cannot
# validate what it read — anything sitting in that slot is taken as REPL output. Emitting an extra
# user message before the observation would silently corrupt divergence detection rather than fail.
#
# Pre-#928 logs are deliberately not supported via a fallback tag match: such a log predates the
# whole observation reshaping, so its saved text could not compare equal to what a live run
# produces today — it would be a *false* divergence signal, not a usable one. Replaying an old log
# without ``detect_divergence`` is unaffected.


def _parse_iterations_from_log(
    log_path: Path,
) -> list[tuple[str | None, str | None]]:
    """Parse ``(assistant_content, repl_output)`` per agent iteration from a FileLogger
    .log file, in order and positionally aligned by construction.

    A single pass over the log. For each ``assistant`` message, ``repl_output`` is the full content
    of the *following* ``user`` message (the REPL exec that ran in response) — since #928 an
    observation carries no framing, so the message *is* the output — or ``None`` when there is no
    following user message. The final assistant with no
    following user yields ``(content, None)``. Project the pairs to get the aligned
    lists: ``[a for a, _ in ...]`` for replay responses, ``[o for _, o in ...]`` for
    ``expected_repl_outputs``.

    Returning the pair from one walk is deliberate (#587): assistant content and REPL
    output must stay positionally aligned, and deriving them together makes that hold
    *by construction* — the previous design ran two independent parsers over the same
    log and leaned on a runtime length check that a future edit to one could silently
    violate.

    When the log holds multiple ``[Agent] invoke enter`` lines (the kill+restart
    concatenation case described above ``_INVOKE_ENTER_RE``), only the segment after the
    **last** invoke enter is kept — matching the per-task costs.json, which is
    overwritten on each restart and thus contains only the latest attempt's calls.

    NOTE: This is the documented fallback path, tightly coupled to the exact line format
    emitted by ``FileLogger`` (see ``jaz/hooks/builtin/loggers.py``): one
    ``[Agent] message: <dict-repr>`` line per message, with no embedded newlines (a
    message whose repr spans multiple lines would have its tail — and any observation body —
    silently dropped by this line-by-line scan). FileLogger formats
    dicts via ``repr``, which encodes newlines as ``\\n`` literals on a single line, so
    this works for messages logged through that path. There is no version guard, so any
    change to the logger's line format must be mirrored in ``_AGENT_MESSAGE_RE`` /
    ``_INVOKE_ENTER_RE`` above. The observation itself is read *positionally* (see the note above),
    so a change to ``DefaultProtocol.render_observation`` that alters how many messages an
    observation occupies must be mirrored here. Prefer the
    structured ConversationHistory JSON / costs.json factories when available.
    """
    # TODO: Deprecate and replace with ATIF format, and make all our eval scripts have
    # the ATIF hook on by default to make runs resumable.
    iterations: list[tuple[str | None, str | None]] = []
    # Content of an assistant message awaiting its following user message (to pair its
    # observation body). ``have_pending`` distinguishes "no pending assistant" from a
    # "pending assistant whose content is itself None".
    pending_content: str | None = None
    have_pending = False
    with open(log_path) as f:
        for line in f:
            stripped = line.rstrip("\n")
            if _INVOKE_ENTER_RE.match(stripped):
                # New attempt — discard everything from the previous (killed) attempt so
                # we stay aligned with the (overwritten) costs.json.
                iterations = []
                pending_content = None
                have_pending = False
                continue
            m = _AGENT_MESSAGE_RE.match(stripped)
            if m is None:
                continue
            # The matched group is a Python dict literal
            msg = ast.literal_eval(m.group(1))
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "assistant":
                if have_pending:
                    # Previous assistant had no following user message (e.g. two
                    # assistants in a row) — flush it with no output.
                    iterations.append((pending_content, None))
                pending_content = msg.get("content")
                have_pending = True
            elif role == "user" and have_pending:
                content = msg.get("content")
                # The whole message IS the observation — see the note above ``_INVOKE_ENTER_RE``'s
                # successor comment: #928 left the output unframed, so there is nothing to strip.
                repl_output = content if isinstance(content, str) else None
                iterations.append((pending_content, repl_output))
                pending_content = None
                have_pending = False
        if have_pending:
            # Final assistant has no following user message — no comparison available
            # (e.g. the run ended after this response).
            iterations.append((pending_content, None))
    return iterations


def _exec_result_output(
    result: Continue | Raise | Return,
) -> str:  # noqa: UP007
    """Return the captured stdout/output text of an ExecResult, or ``""`` for a terminal result.

    Only a ``Continue`` carries ``output`` now (#903 dropped it from the terminal ``Return`` /
    ``Raise``, whose return value / exception is the payload, not agent-facing text). This helper
    centralizes that so divergence comparison handles the variants in one place.
    """
    return result.output if isinstance(result, Continue) else ""


# ------------------------------------------------------------------
# Per-invoke tree builder
# ------------------------------------------------------------------

_LOG_INVOKE_ENTER_D_RE = re.compile(r"\[Agent\] invoke enter:.*depth=(\d+)")
_LOG_INVOKE_EXIT_D_RE = re.compile(r"\[Agent\] invoke exit:.*depth=(\d+)")
_LOG_LLM_ENTER_RE = re.compile(
    r"\[LLM\] query enter: model=(\S+?), messages=(\d+) messages"
)
_LOG_LLM_EXIT_RE = re.compile(r"\[LLM\] query exit: model=(\S+?), cost=\$([0-9.]+)")
_LOG_MSG_RE = re.compile(r"\[Agent\] message: (\{.*\})$")


def _build_invoke_tree_from_log(log_path: Path) -> _InvokeNode:
    """Walk a FileLogger ``.log`` and build a per-invoke ``_InvokeNode`` tree.

    Each ``_InvokeNode`` holds the ordered list of ``_SavedResponse`` objects
    for LLM calls made directly by that invoke, plus a queue of child nodes
    for nested invokes.  The root node represents the top-level invoke (the
    first ``[Agent] invoke enter`` in the log).

    Token counts are tiktoken-approximated from accumulated message content
    (prompt_tokens ~99.4% of truth); dollar cost is exact from the log line;
    completion_tokens is left None (unknowable for reasoning models — see
    module docstring).
    """
    try:
        import tiktoken as _tiktoken  # pyright: ignore[reportMissingImports]  # optional dep (eval-only); only used here

        _enc_cache: dict[str, Any] = {}

        def _encoding_for(model: str):
            if model not in _enc_cache:
                name = model.split("/", 1)[-1].lower()
                try:
                    enc = _tiktoken.encoding_for_model(name)
                except Exception:
                    enc = _tiktoken.get_encoding(
                        "o200k_base"
                        if (name.startswith("gpt-5") or name.startswith("o"))
                        else "cl100k_base"
                    )
                _enc_cache[model] = enc
            return _enc_cache[model]

        def _count_tokens(text: str, model: str) -> int:
            if not text:
                return 0
            return len(_encoding_for(model).encode(text, disallowed_special=()))

    except ImportError:

        def _count_tokens(text: str, model: str) -> int:  # type: ignore[misc]
            return 0

    # Sentinel root wraps the actual top-level node so pop logic is uniform.
    sentinel = _InvokeNode(responses=[], children=[])
    # Stack entries: (node, depth, message_buffer)
    stack: list[tuple[_InvokeNode, int, list[str]]] = [(sentinel, -1, [])]
    pending_llm_enter: dict[int, dict[str, Any]] = {}
    iter_counter: dict[int, int] = {}

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            m = _LOG_INVOKE_ENTER_D_RE.search(line)
            if m:
                depth = int(m.group(1))
                node = _InvokeNode(responses=[], children=[])
                stack.append((node, depth, []))
                iter_counter[depth] = 0
                pending_llm_enter.pop(depth, None)
                continue

            m = _LOG_INVOKE_EXIT_D_RE.search(line)
            if m:
                if len(stack) > 1:
                    node, depth, _ = stack.pop()
                    iter_counter.pop(depth, None)
                    pending_llm_enter.pop(depth, None)
                    parent_node = stack[-1][0]
                    parent_node.children.append(node)
                continue

            m = _LOG_LLM_ENTER_RE.search(line)
            if m:
                depth = len(stack) - 1
                pending_llm_enter[depth] = {
                    "model": m.group(1),
                    "n_msgs": int(m.group(2)),
                }
                continue

            m = _LOG_LLM_EXIT_RE.search(line)
            if m:
                depth = len(stack) - 1
                pending = pending_llm_enter.pop(depth, None)
                if pending is None:
                    continue
                cost = float(m.group(2))
                buf = stack[-1][2]
                window = buf[-pending["n_msgs"] :] if pending["n_msgs"] > 0 else []
                prompt_tokens = _count_tokens("\n".join(window), pending["model"])
                idx = iter_counter.get(depth, 0)
                iter_counter[depth] = idx + 1
                node = stack[-1][0]
                node.responses.append(
                    _SavedResponse(
                        content=None,  # filled in by next assistant message
                        prompt_tokens=prompt_tokens,
                        completion_tokens=None,
                        total_tokens=None,
                        cost=cost,
                    )
                )
                continue

            m = _LOG_MSG_RE.search(line)
            if m:
                try:
                    msg = ast.literal_eval(m.group(1))
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "") or ""
                depth = len(stack) - 1
                stack[-1][2].append(content)
                if role == "assistant":
                    node = stack[-1][0]
                    if node.responses and node.responses[-1].content is None:
                        r = node.responses[-1]
                        node.responses[-1] = _SavedResponse(
                            content=content,
                            prompt_tokens=r.prompt_tokens,
                            completion_tokens=r.completion_tokens,
                            total_tokens=r.total_tokens,
                            cost=r.cost,
                        )
                continue

    # Close any open invokes (truncated/killed log)
    while len(stack) > 1:
        node, _, _ = stack.pop()
        stack[-1][0].children.append(node)

    # Unwrap sentinel
    if len(sentinel.children) == 1:
        return sentinel.children[0]
    # Pathological: return sentinel as root if no invokes were found
    return sentinel


_REPL_SESSION_ID_RE = re.compile(r"<repl_[0-9a-f]+_\d+>")
_TRACEBACK_LINE_RE = re.compile(r'(File "[^"]+", line) \d+')


def _normalize_output(s: str) -> str:
    """Normalize a REPL output string for divergence comparison.

    The caller is expected to have already applied ``abbreviate_string`` to
    the live output so it matches the abbreviation the original run applied
    before writing the saved block. On top of that, this strips two known
    sources of run-to-run non-semantic noise that appear in REPL exceptions:

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


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``).
ReplayHook = Replay
