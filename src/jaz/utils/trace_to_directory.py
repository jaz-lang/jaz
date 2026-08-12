#!/usr/bin/env python3
"""Convert a ConversationHistory JSON trace into a browsable directory.

Each invoke node becomes a directory with:
    overview.md          — metadata, prompt, iteration index
    iter_1/step.md       — repl_input + repl_output
    iter_1/repl_input.txt
    iter_1/repl_output.txt
    iter_1/subagent_0/   — recursively nested child invokes
    iter_2/...

Usage::

    python -m jaz.utils.trace_to_directory trace.json output_dir/
    python -m jaz.utils.trace_to_directory trace.json output_dir/ --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _get_repl_output(iterations: list[dict[str, Any]], idx: int) -> str | None:
    """Get the REPL output for iteration at `idx`.

    The output is the *first* user message of the next iteration, in full. #928 stripped every bit
    of framing from an observation (``<repl_output>`` wrapper, ``<error>`` block, and finally the
    ``**REPL output (iteration #N):**`` header), so there is no tag to extract — the message is the
    output. That makes this positional rather than syntactic: whatever occupies that slot is
    reported as REPL output, and a later message (e.g. truncation advice) is correctly skipped only
    because it is not the first.
    """
    next_idx = idx + 1
    if next_idx >= len(iterations):
        return None
    for msg in iterations[next_idx].get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
    return None


def _format_cost(cost: float | None) -> str:
    if cost is None:
        return "N/A"
    return f"${cost:.4f}"


def _write_iteration(
    iter_dir: Path,
    iteration: dict[str, Any],
    repl_output: str | None,
    iterations: list[dict[str, Any]],
    idx: int,
) -> str:
    """Write files for a single iteration. Returns a short summary line."""
    iter_dir.mkdir(parents=True, exist_ok=True)

    llm_resp = iteration.get("llm_response") or {}
    content = llm_resp.get("content", "")
    cost = llm_resp.get("cost")
    prompt_tokens = llm_resp.get("prompt_tokens")
    completion_tokens = llm_resp.get("completion_tokens")

    # The response IS the code — there is no wrapper tag to extract (the XML protocol was
    # removed). Kept as a separate name so the rest of this function reads unchanged.
    repl_input = content

    # Write repl_input.txt
    if repl_input:
        (iter_dir / "repl_input.txt").write_text(repl_input + "\n")

    # Write repl_output.txt
    if repl_output:
        (iter_dir / "repl_output.txt").write_text(repl_output + "\n")

    # Write step.md
    iter_num = iteration.get("iteration", idx + 1)
    lines = [f"# Iteration {iter_num}", ""]

    cost_parts = []
    if cost is not None:
        cost_parts.append(f"Cost: {_format_cost(cost)}")
    if prompt_tokens is not None:
        cost_parts.append(f"Prompt tokens: {prompt_tokens}")
    if completion_tokens is not None:
        cost_parts.append(f"Completion tokens: {completion_tokens}")
    if cost_parts:
        lines.append(f"*{' | '.join(cost_parts)}*")
        lines.append("")

    if repl_input:
        lines.extend(["## REPL Input", "", "```python", repl_input, "```", ""])

    if repl_output:
        lines.extend(["## REPL Output", "", "```", repl_output, "```", ""])

    # Write child invokes (subagents)
    children = iteration.get("children", [])
    if children:
        lines.extend(["## Sub-agents", ""])
        for child_idx, child in enumerate(children):
            child_dir = iter_dir / f"subagent_{child_idx}"
            _write_invoke_node(child_dir, child)
            child_name = child.get("task_name", f"subagent_{child_idx}")
            lines.append(f"- `subagent_{child_idx}/` — {child_name}")
        lines.append("")

    (iter_dir / "step.md").write_text("\n".join(lines))

    # Return summary for the overview
    summary = (repl_input or content or "")[:70].replace("\n", " ").strip()
    return summary


def _write_invoke_node(output_dir: Path, node: dict[str, Any]) -> None:
    """Recursively write an invoke node as a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    task_name = node.get("task_name", "main")
    # `recursion_depth` fallback reads cost/trace files written before the depth rename (#743);
    # the current writer emits `depth`.
    depth = node.get("depth", node.get("recursion_depth", 1))
    return_type = node.get("return_type")
    prompt = node.get("prompt", "")
    result = node.get("result") or {}
    iterations = node.get("repl_iterations", [])

    # Compute totals
    total_cost = 0.0
    total_prompt_tokens = 0
    for it in iterations:
        resp = it.get("llm_response") or {}
        total_cost += resp.get("cost", 0) or 0
        total_prompt_tokens += resp.get("prompt_tokens", 0) or 0

    # Format result
    result_type = result.get("type", "unknown")
    result_value = result.get("value")
    if result_value is not None:
        result_str = f"{result_type}: {result_value}"
    else:
        result_str = result_type

    # Write iterations and collect summaries
    summaries = []
    for idx, it in enumerate(iterations):
        iter_num = it.get("iteration", idx + 1)
        iter_dir = output_dir / f"iter_{iter_num}"
        repl_output = _get_repl_output(iterations, idx)
        summary = _write_iteration(iter_dir, it, repl_output, iterations, idx)
        summaries.append((iter_num, summary))

    # Write overview.md
    lines = [
        f"# {task_name}",
        "",
        f"**Depth:** {depth}",
        f"**Return type:** `{return_type}`",
        f"**Iterations:** {len(iterations)}",
        f"**Total cost:** {_format_cost(total_cost)}",
        f"**Total prompt tokens:** {total_prompt_tokens}",
        f"**Result:** {result_str}",
        "",
    ]

    # Truncate prompt for overview
    if prompt:
        prompt_display = (
            prompt if len(prompt) <= 2000 else prompt[:2000] + "\n\n*... (truncated)*"
        )
        lines.extend(["## Prompt", "", prompt_display, ""])

    if summaries:
        lines.extend(["## Iterations", ""])
        for iter_num, summary in summaries:
            lines.append(f"- `iter_{iter_num}/` — `{summary}`")
        lines.append("")

    (output_dir / "overview.md").write_text("\n".join(lines))


def trace_to_directory(
    trace_path: str | Path,
    output_dir: str | Path,
    *,
    all_nodes: bool = False,
) -> None:
    """Convert a trace JSON file to a browsable directory.

    Args:
        trace_path: Path to the ConversationHistory JSON file.
        output_dir: Directory to write the output to.
        all_nodes: If True, write all top-level invoke nodes.
                   If False (default), write only the first one.
    """
    trace_path = Path(trace_path)
    output_dir = Path(output_dir)

    data = json.loads(trace_path.read_text())

    if not isinstance(data, list) or not data:
        print(f"Error: expected a non-empty list in {trace_path}")
        sys.exit(1)

    nodes = data if all_nodes else data[:1]

    if len(nodes) == 1:
        _write_invoke_node(output_dir, nodes[0])
    else:
        for i, node in enumerate(nodes):
            task_name = node.get("task_name", f"invoke_{i}")
            _write_invoke_node(output_dir / f"{i}_{task_name}", node)

    print(f"Wrote {len(nodes)} invoke node(s) to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a ConversationHistory JSON trace into a browsable directory."
    )
    parser.add_argument("trace_json", type=Path, help="Path to trace JSON file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write all top-level invoke nodes (default: first only)",
    )
    args = parser.parse_args()

    if not args.trace_json.exists():
        print(f"Error: file not found: {args.trace_json}")
        sys.exit(1)

    trace_to_directory(args.trace_json, args.output_dir, all_nodes=args.all)


if __name__ == "__main__":
    main()
