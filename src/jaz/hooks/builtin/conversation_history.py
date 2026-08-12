"""Conversation history hook that stores the entire conversation as JSON.

This hook captures all messages from LLM interactions across all invoke calls,
preserving the natural tree structure of nested jaz.invoke calls.

The output JSON structure organizes by REPL iterations:
{
    "invoke_id": "...",
    "task_name": "...",
    "inputs": {...},
    "repl_iterations": [
        {
            "iteration": 1,
            "messages": [...],      # Messages at start of this iteration
            "llm_response": {...},  # LLM response for this iteration
            "children": [...]       # Nested invoke calls spawned during this iteration
        },
        ...
    ],
    "result": {...}
}
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Effect
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
)
from jaz.provenance import PROVENANCE_KEY, provenance_of
from jaz.repl.types import Continue, ExecResult, Raise, Return


@dataclass
class REPLIteration:
    """Represents a single REPL iteration within an invoke."""

    iteration: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    llm_response: dict[str, Any] | None = None
    children: list["InvokeNode"] = field(default_factory=list)
    # Provenance of message edits (drops/adds) composed for this iteration's query — the
    # only record of what was *removed*, since a dropped message leaves the buffer and so
    # can't be recovered from the surviving messages' per-message provenance. None unless a
    # hook emitted DropMessages / AddMessages this iteration.
    edits: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert iteration to a JSON-serializable dictionary."""
        return {
            "iteration": self.iteration,
            "messages": self.messages,
            "llm_response": self.llm_response,
            "edits": self.edits,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class InvokeNode:
    """Represents a single invoke call in the conversation tree."""

    invoke_id: str
    parent_invoke_id: str | None
    parent_repl_iteration: int | None
    task_name: str
    # Explicit `**inputs` kwargs and resolved ambient `jaz.scope` — distinct provenance channels
    # (#727), serialized as separate ``inputs`` / ``scope`` keys.
    inputs: dict[str, Any]
    scope: dict[str, Any]
    depth: int

    # Serialized ``to_dict()`` dicts of the hooks active for this invoke (baseline ‖ propagating ‖
    # local), captured at receipt from the live ``event.hooks`` (#727).
    # Defaulted so pre-existing node constructions need not supply it.
    hooks: tuple[dict, ...] = field(default_factory=tuple)

    # Accumulated during execution - organized by REPL iteration
    repl_iterations: dict[int, REPLIteration] = field(default_factory=dict)
    current_iteration: int | None = None
    result: ExecResult | None = None
    # Snapshot of the message buffer at the previous iteration's query, so each iteration
    # records only what changed. A count would assume the buffer only ever grows; a
    # persistent message edit (compaction) can shrink/insert/reorder it, so we diff against
    # the actual snapshot by common prefix + suffix instead (see on_llm_query_enter).
    last_messages: list[dict[str, Any]] = field(default_factory=list)

    def get_or_create_iteration(self, iteration: int) -> REPLIteration:
        """Get or create a REPLIteration for the given iteration number."""
        if iteration not in self.repl_iterations:
            self.repl_iterations[iteration] = REPLIteration(iteration=iteration)
        return self.repl_iterations[iteration]

    def to_dict(self) -> dict[str, Any]:
        """Convert node to a JSON-serializable dictionary."""
        # Sort iterations by number
        sorted_iterations = sorted(
            self.repl_iterations.values(), key=lambda x: x.iteration
        )
        return {
            "invoke_id": self.invoke_id,
            "task_name": self.task_name,
            "inputs": _serialize_inputs(self.inputs),
            "scope": _serialize_inputs(self.scope),
            "hooks": list(self.hooks),
            "depth": self.depth,
            "repl_iterations": [it.to_dict() for it in sorted_iterations],
            "result": _serialize_result(self.result),
        }


def _serialize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Serialize inputs to JSON-compatible format."""
    result = {}
    for key, value in inputs.items():
        try:
            # Test if it's JSON serializable
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            # Fall back to repr for non-serializable values
            result[key] = repr(value)
    return result


def _serialize_result(result: ExecResult | None) -> dict[str, Any] | None:
    """Serialize an execution result to JSON-compatible format."""
    if result is None:
        return None

    result_dict: dict[str, Any] = {"type": type(result).__name__}

    # Add result-specific fields
    if isinstance(result, Return):
        value = result.return_value
        if value is not None:
            try:
                json.dumps(value)
                result_dict["value"] = value
            except (TypeError, ValueError):
                result_dict["value"] = repr(value)

    if isinstance(result, Continue | Raise):
        if result.exception is not None:
            result_dict["exception"] = str(result.exception)
            result_dict["exception_type"] = type(result.exception).__name__

    return result_dict


def _serialize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Serialize a message to JSON-compatible format."""
    result = {}
    for key, value in message.items():
        if key == PROVENANCE_KEY:
            # Emitted as a clean "provenance" sub-object below, not as the raw magic key.
            continue
        if key == "content":
            # Handle content which can be string or list of content parts
            if isinstance(value, str):
                result[key] = value
            elif isinstance(value, list):
                # Content parts (e.g., text + images)
                serialized_parts = []
                for part in value:
                    if isinstance(part, dict):
                        serialized_parts.append(part)
                    else:
                        serialized_parts.append(repr(part))
                result[key] = serialized_parts
            else:
                result[key] = repr(value)
        else:
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                result[key] = repr(value)
    prov = provenance_of(message)
    if prov is not None:
        result["provenance"] = prov.to_dict()
    return result


class ConversationHistory(Hook):
    """Capture the entire conversation history as JSON.

    Records every LLM interaction in the hook's scope, preserving the tree structure of the
    execution. Under ``with ConversationHistory(...):`` that includes invokes nested inside;
    passed positionally to one ``jaz.invoke``, only that invoke's own interactions are
    recorded and its sub-invokes are absent from the output.

    A pure observer: it emits no effects, so it never influences the run.

    Examples:
        with ConversationHistory(output_path="./conversation.json"):
            result = invoke(ReturnType(str), task="Do something")
        # Creates ./conversation.json with the full conversation tree.

    Args:
        output_path: Path to write the JSON output. If not provided,
            the conversation history is only kept in memory and can be
            accessed via the `root_nodes` attribute after execution.
        indent: JSON indentation level (default: 2). Set to None for compact output.
    """

    # Handles ``InvokeEnter``/``InvokeExit`` (tree structure) and ``LLMQueryEnter``/
    # ``LLMQueryExit`` (the messages and responses). Out of the docstring: which events a hook
    # subscribes to is not something a caller of the hook acts on — "emits no effects" is.

    # TODO: eventually remove this hook in favor of ``ATIFTrace``, which emits the same
    # information in the standardized ATIF v1.7 format. Migrate existing consumers off the
    # ``ConversationHistory`` JSON first.

    # Captures the per-invoke ``task_name`` label off the blackboard (set by a
    # ``MetaData`` hook) into each InvokeNode; declaring it lets that seed validate.
    # Absent → ``"main"``.
    blackboard_consumes = {"task_name": "Human label recorded on each invoke node."}

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        indent: int | None = 2,
    ):
        """Initialize the conversation history hook.

        Args:
            output_path: Path to write JSON output (optional)
            indent: JSON indentation level (default: 2)
        """
        self.output_path = Path(output_path) if output_path else None
        self.indent = indent

        # Map invoke_id to InvokeNode
        self.nodes: dict[str, InvokeNode] = {}

        # Track root-level invokes (those without parents)
        self.root_nodes: list[InvokeNode] = []

    # This hook is a passive observer: every handler builds the conversation tree
    # and returns an empty effect list.

    def on_invoke_enter(self, event: InvokeEnter) -> list[Effect]:
        # Create node for this invoke
        node = InvokeNode(
            invoke_id=event.invoke_id,
            parent_invoke_id=event.parent_invoke_id,
            parent_repl_iteration=event.parent_repl_iteration,
            # Per-invoke label carried on the blackboard (seeded by a MetaData hook),
            # not a core event field. Absent → "main".
            task_name=str(event.blackboard.get("task_name", "main")),
            # Explicit kwargs and resolved scope kept separate (#727) — see the node fields above.
            # Defensive copies (like ATIFTrace) so the node's long-lived snapshot never aliases
            # the event's dicts. This is a SHALLOW snapshot: it pins the key set / top-level bindings
            # at receipt, but nested MUTABLE values are shared with the live namespace, and
            # `_serialize_inputs` keeps JSON-safe values by reference and runs lazily (at to_dict),
            # so a value the agent mutates during execution is reflected in the serialized node.
            # Pinning nested values would need eager deep serialization here — costly and can fail on
            # non-copyable inputs — so it is a documented limitation (#831), not a bug.
            inputs=dict(event.inputs),
            scope=dict(event.scope),
            # Serialize the live active-hook set to dicts here, at receipt, so the recorded node
            # pins the governance active AT InvokeEnter and stays JSON-safe (#727).
            hooks=tuple(h.to_dict() for h in event.hooks),
            depth=event.depth,
        )

        # Store in lookup dict
        self.nodes[event.invoke_id] = node

        # Link to parent's specific iteration or add as root
        if event.parent_invoke_id and event.parent_invoke_id in self.nodes:
            parent_node = self.nodes[event.parent_invoke_id]
            if event.parent_repl_iteration is not None:
                # Add to the specific iteration that spawned this invoke
                iteration = parent_node.get_or_create_iteration(
                    event.parent_repl_iteration
                )
                iteration.children.append(node)
            else:
                # Fallback: add to current iteration if known
                if parent_node.current_iteration is not None:
                    iteration = parent_node.get_or_create_iteration(
                        parent_node.current_iteration
                    )
                    iteration.children.append(node)
        else:
            self.root_nodes.append(node)
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # LLMQueryEnter is the always-present per-turn boundary (it replaced the removed
        # REPLIterationEnter as the per-turn anchor — #481): mark which iteration this turn
        # belongs to and ensure its record exists, before recording the turn's messages
        # below. event.iteration is None only on the non-loop query() path — skip there.
        node = self.nodes.get(event.invoke_id)
        if node and event.iteration is not None:
            node.current_iteration = event.iteration
            node.get_or_create_iteration(event.iteration)

        # Record only what changed this iteration, not the accumulated buffer. Diff
        # against the previous query's snapshot by longest common prefix + suffix and log
        # the changed *middle*: in the common append-only case that is exactly the new
        # tail; when a persistent edit (compaction) rewrites the middle it is just the
        # inserted/replaced messages — never the whole buffer. (A bare message count would
        # assume the buffer only grows; the prefix/suffix comparison survives a
        # shrink/insert/reorder. Non-contiguous edits log a slightly wider span — still
        # bounded by the buffer, never larger.)
        #
        # The match is by *value* (dict equality), not identity, so two value-equal
        # messages (e.g. duplicate short acknowledgements, or a compaction that re-inserts
        # a byte-identical message) can make the prefix/suffix scan attribute the boundary
        # to the wrong side. Never loses data — worst case is the same "slightly wider
        # span" over-logging above — but it's a second reason, beyond reordering, that this
        # is a stopgap rather than the fix: #599's message-identity primitive is. Per-message
        # provenance (#605) narrows but does not close this gap once it lands: its stamped
        # `iteration` field rides inside the same dict, so two same-content messages from
        # *different* iterations stop comparing equal — but duplicates stamped within the
        # same iteration are still indistinguishable by value.
        if node and node.current_iteration is not None:
            cur = [dict(m) for m in event.messages]
            prev = node.last_messages
            p = 0
            while p < len(prev) and p < len(cur) and prev[p] == cur[p]:
                p += 1
            s = 0
            while (
                s < len(prev) - p and s < len(cur) - p and prev[-1 - s] == cur[-1 - s]
            ):
                s += 1
            changed = cur[p : len(cur) - s]
            iteration = node.get_or_create_iteration(node.current_iteration)
            iteration.messages = [_serialize_message(m) for m in changed]
            node.last_messages = cur
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        # Store the LLM response in the current iteration
        node = self.nodes.get(event.invoke_id)
        if node and node.current_iteration is not None:
            response_data: dict[str, Any] = {
                "model": event.model,
                "content": event.response.content,
                "prompt_tokens": event.response.prompt_tokens,
                "completion_tokens": event.response.completion_tokens,
                "cost": event.response.cost,
            }
            # ATIF v1.7 MetricsSchema usage fields (added when present)
            if event.response.cached_tokens is not None:
                response_data["cached_tokens"] = event.response.cached_tokens
            if event.response.extra:
                response_data["extra"] = dict(event.response.extra)
            iteration = node.get_or_create_iteration(node.current_iteration)
            iteration.llm_response = response_data

            # Record what this query's composed edits removed/inserted. Drop indices point
            # into the enter snapshot, which on_llm_query_enter stored as last_messages this
            # same iteration, so resolve them there to log the actual dropped messages +
            # reasons. Persistent adds are stamped ADDED only *after* this exit fires, so
            # their serialized form here simply carries no provenance sub-object — the
            # reason/persistent fields below are the record.
            edits = event.message_edits
            if edits is not None and (edits.all_drops or edits.all_adds):
                snapshot = node.last_messages
                dropped = [
                    {
                        "index": i,
                        "persistent": i in edits.persistent_drops,
                        "message": _serialize_message(snapshot[i]),
                    }
                    for i in sorted(edits.all_drops)
                    if 0 <= i < len(snapshot)
                ]
                added = [
                    {
                        "index": add.index,
                        "persistent": add.persistent,
                        "messages": [_serialize_message(m) for m in add.messages],
                    }
                    for add in edits.all_adds
                ]
                iteration.edits = {"dropped": dropped, "added": added}
        # Per-turn flush to disk for crash recovery. LLMQueryExit is the always-present
        # per-turn exit (it replaced the removed REPLIterationExit as the flush point —
        # #481); it fires once per turn after the response is recorded above.
        self._write_output()
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        # Store the result
        node = self.nodes.get(event.invoke_id)
        if node:
            node.result = event.result
        # Flush to disk after invoke completes
        self._write_output()
        return []

    def get_conversation_tree(self) -> list[dict[str, Any]]:
        """Get the conversation tree as a list of dictionaries.

        Returns:
            List of root invoke nodes as dictionaries.
        """
        return [node.to_dict() for node in self.root_nodes]

    def get_json(self) -> str:
        """Get the conversation history as a JSON string.

        Returns:
            JSON string of the conversation tree.
        """
        return json.dumps(self.get_conversation_tree(), indent=self.indent)

    def _write_output(self) -> None:
        """Write the conversation history to the output file.

        Uses write-to-temp-then-rename for atomic writes, so a crash
        mid-write won't leave a truncated/unparsable JSON file. The temp file is
        created via ``tempfile.mkstemp`` (unique even across concurrent threads
        and processes), so two writers targeting the same output_path can't
        clobber each other's temp file mid-rename; the final rename is atomic so
        a reader always sees either the old file or a complete new one.
        """
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=self.output_path.parent,
                prefix=self.output_path.name + ".",
                suffix=".tmp",
            )
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(self.get_json())
                tmp.rename(self.output_path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise

    def teardown(self, exc: BaseException | None = None) -> None:
        """Write the output file on exit.

        Note: We intentionally do NOT clear state here so that the conversation
        tree can be accessed after exiting the context manager (as documented).
        Users who want to reuse the hook should call reset() explicitly.
        """
        self._write_output()

    def reset(self) -> None:
        """Clear the conversation history state.

        Call this if you want to reuse the hook for a new conversation.
        """
        self.nodes.clear()
        self.root_nodes.clear()


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``).
ConversationHistoryHook = ConversationHistory
