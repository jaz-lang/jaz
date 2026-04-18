"""Conversation history hook that stores the entire conversation as JSON.

This hook captures all messages from LLM interactions across all invoke calls,
preserving the natural tree structure of nested jaz.invoke calls.

The output JSON structure organizes by REPL iterations:
{
    "invoke_id": "...",
    "task_name": "...",
    "prompt": "...",
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
    ReplIterationEnter,
    ReplIterationExit,
)
from jaz.repl.types import ErrorResult, ExecResult, RaiseResult, ReturnResult


@dataclass
class ReplIteration:
    """Represents a single REPL iteration within an invoke."""

    iteration: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    llm_response: dict[str, Any] | None = None
    children: list["InvokeNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert iteration to a JSON-serializable dictionary."""
        return {
            "iteration": self.iteration,
            "messages": self.messages,
            "llm_response": self.llm_response,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class InvokeNode:
    """Represents a single invoke call in the conversation tree."""

    invoke_id: str
    parent_invoke_id: str | None
    parent_repl_iteration: int | None
    task_name: str
    prompt: str
    return_type: str | None
    inputs: dict[str, Any]
    recursion_depth: int

    # Accumulated during execution - organized by REPL iteration
    repl_iterations: dict[int, ReplIteration] = field(default_factory=dict)
    current_iteration: int | None = None
    result: ExecResult | None = None
    # Track message count to only store new messages per iteration
    last_message_count: int = 0

    def get_or_create_iteration(self, iteration: int) -> ReplIteration:
        """Get or create a ReplIteration for the given iteration number."""
        if iteration not in self.repl_iterations:
            self.repl_iterations[iteration] = ReplIteration(iteration=iteration)
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
            "prompt": self.prompt,
            "return_type": self.return_type,
            "inputs": _serialize_inputs(self.inputs),
            "recursion_depth": self.recursion_depth,
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
    if isinstance(result, ReturnResult):
        value = result.return_value
        if value is not None:
            try:
                json.dumps(value)
                result_dict["value"] = value
            except (TypeError, ValueError):
                result_dict["value"] = repr(value)

    if isinstance(result, ErrorResult | RaiseResult):
        if result.exception is not None:
            result_dict["exception"] = str(result.exception)
            result_dict["exception_type"] = type(result.exception).__name__

    return result_dict


def _serialize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Serialize a message to JSON-compatible format."""
    result = {}
    for key, value in message.items():
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
    return result


class ConversationHistoryHook(Hook):
    """Hook that captures and stores the entire conversation history as JSON.

    This hook tracks all LLM interactions across all invoke calls (including
    nested invokes), preserving the natural tree structure of the execution.

    Usage:
        with ConversationHistoryHook(output_path="./conversation.json"):
            result = invoke("Do something", return_type=str)

        # Creates: ./conversation.json with full conversation tree

    Args:
        output_path: Path to write the JSON output. If not provided,
            the conversation history is only kept in memory and can be
            accessed via the `root_nodes` attribute after execution.
        indent: JSON indentation level (default: 2). Set to None for compact output.
    """

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

    def on_event(self, event: Event) -> list:
        """Process events and build conversation tree.

        Args:
            event: The hook event

        Returns:
            Empty list (this is a passive observer hook)
        """
        match event:
            case InvokeEnter(
                invoke_id=invoke_id,
                parent_invoke_id=parent_invoke_id,
                parent_repl_iteration=parent_repl_iteration,
                prompt=prompt,
                task_name=task_name,
                return_type=return_type,
                inputs=inputs,
                cur_recursion_depth=depth,
            ):
                # Get return type as string
                return_type_str = None
                if return_type is not None:
                    return_type_str = getattr(return_type, "__name__", None) or str(
                        return_type
                    )

                # Create node for this invoke
                node = InvokeNode(
                    invoke_id=invoke_id,
                    parent_invoke_id=parent_invoke_id,
                    parent_repl_iteration=parent_repl_iteration,
                    task_name=task_name,
                    prompt=prompt,
                    return_type=return_type_str,
                    inputs=inputs,
                    recursion_depth=depth,
                )

                # Store in lookup dict
                self.nodes[invoke_id] = node

                # Link to parent's specific iteration or add as root
                if parent_invoke_id and parent_invoke_id in self.nodes:
                    parent_node = self.nodes[parent_invoke_id]
                    if parent_repl_iteration is not None:
                        # Add to the specific iteration that spawned this invoke
                        iteration = parent_node.get_or_create_iteration(
                            parent_repl_iteration
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

            case ReplIterationEnter(invoke_id=invoke_id, iteration=iteration):
                # Track current iteration for this invoke
                node = self.nodes.get(invoke_id)
                if node:
                    node.current_iteration = iteration
                    # Ensure the iteration exists
                    node.get_or_create_iteration(iteration)

            case ReplIterationExit(invoke_id=invoke_id, iteration=iteration):
                # Iteration complete - current_iteration stays set for child tracking
                pass

            case LLMQueryEnter(invoke_id=invoke_id, messages=messages):
                # Store only new messages for this iteration (not accumulated from previous)
                node = self.nodes.get(invoke_id)
                if node and node.current_iteration is not None:
                    # Only serialize messages added since last iteration
                    new_messages = messages[node.last_message_count :]
                    serialized_messages = [
                        _serialize_message(dict(msg)) for msg in new_messages
                    ]
                    iteration = node.get_or_create_iteration(node.current_iteration)
                    iteration.messages = serialized_messages
                    # Update the count for next iteration
                    node.last_message_count = len(messages)

            case LLMQueryExit(
                invoke_id=invoke_id,
                response_content=response_content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            ):
                # Store the LLM response in the current iteration
                node = self.nodes.get(invoke_id)
                if node and node.current_iteration is not None:
                    response_data = {
                        "model": model,
                        "content": response_content,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cost": cost,
                    }
                    iteration = node.get_or_create_iteration(node.current_iteration)
                    iteration.llm_response = response_data

            case InvokeExit(invoke_id=invoke_id, result=result):
                # Store the result
                node = self.nodes.get(invoke_id)
                if node:
                    node.result = result

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
        """Write the conversation history to the output file."""
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(self.get_json())

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context and write output file."""
        # Write output file if configured
        self._write_output()

        # Note: We intentionally do NOT clear state here so that the conversation
        # tree can be accessed after exiting the context manager (as documented).
        # Users who want to reuse the hook should call reset() explicitly.

        # Call parent to reset hook context (only if __enter__ was called)
        if hasattr(self, "_token"):
            return super().__exit__(exc_type, exc_value, traceback)

    def reset(self) -> None:
        """Clear the conversation history state.

        Call this if you want to reuse the hook for a new conversation.
        """
        self.nodes.clear()
        self.root_nodes.clear()
