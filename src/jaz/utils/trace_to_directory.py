#!/usr/bin/env python3
"""Convert an ATIF JSON trace into a browsable directory.

Reads an ATIF trace as written by ``ATIFTrace`` (schema v1.7) and expands it into
a directory tree. Each invoke trajectory becomes a directory with:

    overview.md          — metadata, prompt, iteration index
    iter_1/step.md       — repl_input + repl_output
    iter_1/repl_input.txt
    iter_1/repl_output.txt
    iter_1/subagent_0/   — recursively nested child invokes
    iter_2/...

Usage::

    python -m jaz.utils.trace_to_directory trace.atif.json output_dir/
    python -m jaz.utils.trace_to_directory trace.atif.json output_dir/ --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _seed_prompt(steps: list[dict[str, Any]]) -> str:
    """The seed prompt: the leading system/user steps before the first agent step.

    ATIF stores the initial system prompt and task as ``system``/``user`` steps that
    precede the first ``agent`` step; those are the seed, not observations (which follow an
    agent step). Concatenate their messages, mirroring how the old per-iteration reader
    used the node's ``prompt`` for overview.md.
    """
    parts: list[str] = []
    for step in steps:
        if step.get("source") == "agent":
            break
        msg = step.get("message")
        if isinstance(msg, str) and msg:
            parts.append(msg)
    return "\n\n".join(parts)


def _agent_steps_with_outputs(
    steps: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str | None]]:
    """Pair each ``agent`` step with its REPL observation output.

    The observation is the message of the FIRST ``user`` step that appears AFTER the agent
    step and BEFORE the next agent step — PREFERRING the user step whose
    ``extra.provenance.kind == "observation"`` (stamped since ATIF v1.7), falling back to the
    first user step in that window when none is provenance-tagged. This mirrors replay's
    ``_build_invoke_tree_from_atif``: a hook-added user message (e.g. the ``<return_type>``
    reminder) between the agent step and the observation is correctly skipped when provenance
    is present. A terminal agent step (or one immediately followed by another agent step) has
    no observation → ``None``.
    """
    n = len(steps)
    pairs: list[tuple[dict[str, Any], str | None]] = []
    for i, step in enumerate(steps):
        if step.get("source") != "agent":
            continue
        observation: str | None = None
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
                observation = steps[j].get("message")
                break
        pairs.append((step, observation if observation is not None else first_user))
    return pairs


def _format_cost(cost: float | None) -> str:
    if cost is None:
        return "N/A"
    return f"${cost:.4f}"


def _write_iteration(
    iter_dir: Path,
    agent_step: dict[str, Any],
    repl_output: str | None,
    iter_num: int,
    children: list[dict[str, Any]],
) -> str:
    """Write files for a single iteration. Returns a short summary line."""
    iter_dir.mkdir(parents=True, exist_ok=True)

    # The agent step's message IS the code — there is no wrapper tag to extract (the XML
    # protocol was removed). Metrics carry the per-call cost/tokens.
    repl_input = agent_step.get("message", "") or ""
    metrics = agent_step.get("metrics") or {}
    cost = metrics.get("cost_usd")
    prompt_tokens = metrics.get("prompt_tokens")
    completion_tokens = metrics.get("completion_tokens")

    # Write repl_input.txt
    if repl_input:
        (iter_dir / "repl_input.txt").write_text(repl_input + "\n")

    # Write repl_output.txt
    if repl_output:
        (iter_dir / "repl_output.txt").write_text(repl_output + "\n")

    # Write step.md
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
    if children:
        lines.extend(["## Sub-agents", ""])
        for child_idx, child in enumerate(children):
            child_dir = iter_dir / f"subagent_{child_idx}"
            _write_invoke_node(child_dir, child)
            child_name = _task_name(child)
            lines.append(f"- `subagent_{child_idx}/` — {child_name}")
        lines.append("")

    (iter_dir / "step.md").write_text("\n".join(lines))

    # Return summary for the overview
    summary = repl_input[:70].replace("\n", " ").strip()
    return summary


def _task_name(node: dict[str, Any]) -> str:
    """Human label for a trajectory: ATIF has no ``task_name`` field, so read it from
    ``extra.inputs.task`` when present, else fall back to ``"main"``."""
    inputs = (node.get("extra") or {}).get("inputs") or {}
    task = inputs.get("task")
    if isinstance(task, dict):
        # ATIFTrace serializes inputs as {"type": ..., "repr_prefix_10000": ...}
        rep = task.get("repr_prefix_10000")
        if isinstance(rep, str) and rep:
            return rep[:70].replace("\n", " ").strip()
    return "main"


def _write_invoke_node(output_dir: Path, node: dict[str, Any]) -> None:
    """Recursively write an ATIF trajectory as a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = node.get("steps", [])
    extra = node.get("extra") or {}
    final_metrics = node.get("final_metrics") or {}

    task_name = _task_name(node)
    depth = extra.get("depth", 1)
    prompt = _seed_prompt(steps)

    # Format result from the ATIF extra block.
    result_type = extra.get("result_type", "unknown")
    result_value = extra.get("result_value")
    result_exception = extra.get("result_exception")
    if result_value is not None:
        result_str = f"{result_type}: {result_value}"
    elif result_exception is not None:
        result_str = f"{result_type}: {result_exception}"
    else:
        result_str = str(result_type)

    total_cost = final_metrics.get("total_cost_usd", 0.0) or 0.0
    total_prompt_tokens = final_metrics.get("total_prompt_tokens", 0) or 0

    # Subagent trajectories are stored top-level (not inline per step); group them by the
    # parent REPL iteration that spawned them (stamped in extra.parent_repl_iteration) so each
    # iteration's directory gets its own subagent_K children. A subagent lacking
    # parent_repl_iteration lands under key None and is matched to no iteration below, so it is
    # omitted from the browsable tree — real nested invokes always stamp it, so this is a
    # documented gap for hand-authored/partial traces, not an expected path.
    subagents = node.get("subagent_trajectories", [])
    children_by_iter: dict[int | None, list[dict[str, Any]]] = {}
    for child in subagents:
        parent_it = (child.get("extra") or {}).get("parent_repl_iteration")
        children_by_iter.setdefault(parent_it, []).append(child)

    # Write iterations and collect summaries. One agent step = one iteration; the iteration
    # index comes from extra.iteration, falling back to positional enumeration.
    pairs = _agent_steps_with_outputs(steps)
    summaries = []
    for pos, (agent_step, repl_output) in enumerate(pairs):
        # `pos` fallback (not `pos + 1`): iterations are 0-based (#719), so positional order
        # already matches the iteration number when the trace omits it.
        iter_num = (agent_step.get("extra") or {}).get("iteration", pos)
        iter_dir = output_dir / f"iter_{iter_num}"
        children = children_by_iter.get(iter_num, [])
        summary = _write_iteration(
            iter_dir, agent_step, repl_output, iter_num, children
        )
        summaries.append((iter_num, summary))

    # Write overview.md
    lines = [
        f"# {task_name}",
        "",
        f"**Depth:** {depth}",
        f"**Iterations:** {len(pairs)}",
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
    """Convert an ATIF trace JSON file to a browsable directory.

    Args:
        trace_path: Path to the ATIF JSON file (as written by ``ATIFTrace``).
        output_dir: Directory to write the output to.
        all_nodes: If True, write all top-level invoke trajectories.
                   If False (default), write only the first one.

    Raises:
        ValueError: If the trace is not an ATIF object or non-empty list.
    """
    trace_path = Path(trace_path)
    output_dir = Path(output_dir)

    data = json.loads(trace_path.read_text())

    # ATIFTrace writes a single root trajectory as an object, or an array when the traced
    # scope had multiple top-level invokes. Normalize both to a list of trajectory dicts.
    if isinstance(data, dict):
        nodes: list[dict[str, Any]] = [data]
    elif isinstance(data, list):
        nodes = data
    else:
        raise ValueError(f"expected an ATIF object or non-empty list in {trace_path}")

    if not nodes:
        raise ValueError(f"expected a non-empty ATIF trace in {trace_path}")

    nodes = nodes if all_nodes else nodes[:1]

    if len(nodes) == 1:
        _write_invoke_node(output_dir, nodes[0])
    else:
        for i, node in enumerate(nodes):
            task_name = _task_name(node)
            _write_invoke_node(output_dir / f"{i}_{task_name}", node)

    print(f"Wrote {len(nodes)} invoke node(s) to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an ATIF JSON trace into a browsable directory."
    )
    parser.add_argument("trace_json", type=Path, help="Path to ATIF trace JSON file")
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

    try:
        trace_to_directory(args.trace_json, args.output_dir, all_nodes=args.all)
    except ValueError as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
