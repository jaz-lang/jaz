"""Convert ConversationHistory JSON traces to ATIF v1.7 format.

ATIF (Agent Trajectory Interchange Format) is Harbor's standardized format
for logging agent interaction history.  See:
  https://www.harborframework.com/docs/agents/trajectory-format
  https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md

Usage:
    from jaz.utils.trace_to_atif import atif_from_trace
    atif_from_trace("trace.json", "trace.atif.json")

CLI:
    python -m jaz.utils.trace_to_atif trace.json trace.atif.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ATIF-v1.7"


def _convert_invoke_node(node: dict[str, Any]) -> dict[str, Any]:
    """Convert a single ConversationHistory invoke node to ATIF."""
    steps: list[dict[str, Any]] = []
    step_id = 0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_cost = 0.0
    model_name = None

    iterations = node.get("repl_iterations", [])

    for _it_idx, iteration in enumerate(iterations):
        messages = iteration.get("messages", [])
        llm_response = iteration.get("llm_response") or {}
        it_num = iteration.get("iteration")

        if llm_response.get("model"):
            model_name = llm_response["model"]

        # Messages: emit system/user from the conversation, skipping duplicates.
        # - Assistant messages duplicate the previous iteration's llm_response.
        # Emit system/user messages as steps. Skip assistant (duplicate of
        # previous iteration's llm_response). Following mini-swe-agent pattern:
        # no structured tool_calls/observation — code and output are inline.
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "assistant":
                continue

            source = "system" if role == "system" else "user"
            step_id += 1
            step_data: dict[str, Any] = {
                "step_id": step_id,
                "source": source,
                "message": content,
                "llm_call_count": 0,
            }
            if it_num is not None:
                step_data["extra"] = {"iteration": it_num}
            steps.append(step_data)

        # LLM response step (the actual generation)
        if llm_response.get("content"):
            step_id += 1
            llm_content = llm_response["content"]
            prompt_tokens = llm_response.get("prompt_tokens", 0) or 0
            completion_tokens = llm_response.get("completion_tokens", 0) or 0
            # Keep the raw value (None means the provider reported no caching info
            # at all) distinct from the summing value (None coerced to 0).
            raw_cached_tokens = llm_response.get("cached_tokens")
            cached_tokens = raw_cached_tokens or 0
            cost = llm_response.get("cost", 0) or 0
            response_extra = llm_response.get("extra") or {}

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_cached_tokens += cached_tokens
            total_cost += cost

            metrics: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
            }
            # Write an explicit 0 when the model *does* report caching but had no
            # hit (raw == 0); only omit when caching is unreported (raw is None).
            # This mirrors conversation_history's `is not None` gate so all three
            # ATIF consumers agree on the convention.
            if raw_cached_tokens is not None:
                metrics["cached_tokens"] = raw_cached_tokens
            if response_extra:
                metrics["extra"] = dict(response_extra)
            llm_step: dict[str, Any] = {
                "step_id": step_id,
                "source": "agent",
                "message": llm_content,
                "llm_call_count": 1,
                "metrics": metrics,
            }
            if it_num is not None:
                llm_step["extra"] = {"iteration": it_num}

            steps.append(llm_step)

        # Nested invoke children are handled below via subagent_trajectories

    # Build the ATIF document
    atif: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "agent": {
            "name": "jaz",
            "version": __import__("jaz").__version__,
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_cached_tokens": total_cached_tokens,
            "total_cost_usd": round(total_cost, 6),
            "total_steps": len(iterations),
        },
        "notes": (
            "total_steps is the number of REPL iterations (think-act-observe cycles), "
            "not len(steps). Each REPL iteration corresponds to one LLM call."
        ),
    }

    if model_name:
        atif["agent"]["model_name"] = model_name

    if node.get("invoke_id"):
        atif["trajectory_id"] = node["invoke_id"]

    # Extra metadata for full coverage of ConversationHistory fields
    extra: dict[str, Any] = {}
    if node.get("task_name"):
        extra["task_name"] = node["task_name"]
    if node.get("return_type"):
        extra["return_type"] = node["return_type"]
    if node.get("recursion_depth") is not None:
        extra["recursion_depth"] = node["recursion_depth"]
    if node.get("result"):
        result = node["result"]
        if isinstance(result, dict):
            extra["result_type"] = result.get("type")
            if "value" in result:
                # Note: ConversationHistory serializes result values as
                # JSON when possible (repr fallback), so this preserves
                # structured data.  The ATIFTrace instead always stores
                # {"type": ..., "repr_prefix_10000": repr(...)[:10000]}.
                extra["result_value"] = result["value"]
        else:
            extra["result_type"] = str(result)
    if extra:
        atif["extra"] = extra

    # Nested invokes: store in top-level subagent_trajectories[]; stamp each
    # child with the parent REPL iteration that spawned it.
    subagent_trajs: list[dict[str, Any]] = []
    for iteration in iterations:
        parent_it_num = iteration.get("iteration")
        for child in iteration.get("children", []):
            child_atif = _convert_invoke_node(child)
            if parent_it_num is not None:
                child_extra = child_atif.setdefault("extra", {})
                child_extra["parent_repl_iteration"] = parent_it_num
            subagent_trajs.append(child_atif)
    if subagent_trajs:
        atif["subagent_trajectories"] = subagent_trajs

    return atif


def atif_from_trace(
    trace_path: str | Path,
    output_path: str | Path,
    *,
    all_nodes: bool = False,
    indent: int = 2,
) -> None:
    """Convert a ConversationHistory JSON file to ATIF format.

    Args:
        trace_path: Path to the ConversationHistory JSON file.
        output_path: Path to write the ATIF JSON output.
        all_nodes: If True, convert all top-level invoke nodes.
                   If False (default), convert only the first one.
        indent: JSON indentation level.
    """
    trace_path = Path(trace_path)
    output_path = Path(output_path)

    data = json.loads(trace_path.read_text())
    if not isinstance(data, list) or not data:
        print(f"Error: expected a non-empty list in {trace_path}")
        return

    nodes = data if all_nodes else data[:1]

    if len(nodes) == 1:
        atif = _convert_invoke_node(nodes[0])
    else:
        # Multiple top-level invokes: wrap in a list
        atif = [_convert_invoke_node(node) for node in nodes]

    output_path.write_text(json.dumps(atif, indent=indent))
    n_steps = (
        atif.get("final_metrics", {}).get("total_steps", 0)
        if isinstance(atif, dict)
        else sum(a.get("final_metrics", {}).get("total_steps", 0) for a in atif)
    )
    print(f"Wrote ATIF ({n_steps} steps) to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a ConversationHistory JSON trace to ATIF v1.7 format."
    )
    parser.add_argument("trace_json", type=Path, help="Path to trace JSON file")
    parser.add_argument("output_path", type=Path, help="Output ATIF JSON file")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all top-level invoke nodes (default: first only)",
    )
    args = parser.parse_args()

    if not args.trace_json.exists():
        print(f"Error: file not found: {args.trace_json}")
        sys.exit(1)

    atif_from_trace(args.trace_json, args.output_path, all_nodes=args.all)


if __name__ == "__main__":
    main()
