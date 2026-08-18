#!/usr/bin/env python3
"""Convert an RLM log (.jsonl) into a readable Markdown trace.

Usage::

    python rlm_log_viewer.py <path_to_rlm_log.jsonl> [--output FILE]

Reads all entries: metadata + each iteration. Each iteration contributes its
unique new messages (compared to the previous iteration) plus its code blocks
and response.
"""

import argparse
import json
import sys


def _format_message(msg: dict | str) -> str:
    """Format a single message as Markdown."""
    if isinstance(msg, dict):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
    else:
        role = "unknown"
        content = str(msg)

    role_label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
        role, role.title()
    )
    return f"### {role_label}\n\n{content}\n"


def _format_code_blocks(code_blocks: list[dict]) -> str:
    """Format code blocks with their results."""
    if not code_blocks:
        return ""
    parts = ["### Code Execution\n"]
    for block in code_blocks:
        code = block.get("code", "")
        result = block.get("result", {})
        stdout = result.get("stdout", "") if isinstance(result, dict) else str(result)
        stderr = result.get("stderr", "") if isinstance(result, dict) else ""

        parts.append(f"```python\n{code}\n```\n")
        if stdout:
            parts.append(f"**stdout:**\n```\n{stdout}\n```\n")
        if stderr:
            parts.append(f"**stderr:**\n```\n{stderr}\n```\n")
    return "\n".join(parts)


def convert_log(input_path: str) -> str:
    entries = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        return "# Empty RLM log\n"

    parts = []

    # Metadata (first entry)
    meta = entries[0]
    if meta.get("type") == "metadata":
        lines = ["# RLM Log Trace\n"]
        lines.append(f"- **Model**: {meta.get('root_model', '?')}")
        lines.append(f"- **Backend**: {meta.get('backend', '?')}")
        lines.append(f"- **Max depth**: {meta.get('max_depth', '?')}")
        lines.append(f"- **Max iterations**: {meta.get('max_iterations', '?')}")
        lines.append(f"- **Timestamp**: {meta.get('timestamp', '?')}")
        bk = meta.get("backend_kwargs", {})
        if bk:
            lines.append(f"- **Backend kwargs**: {json.dumps(bk, default=str)}")
        # Derive the total from the iteration counter rather than counting
        # entries present, so the count stays correct even if the log was
        # trimmed (e.g. to metadata + final iteration).
        # +1 because `iteration` is a 0-based index and this reports a count (#719): the
        # highest index seen is one less than the number of iterations. `default=-1` keeps an
        # empty log at 0 rather than reporting a phantom first iteration. Entries missing the
        # key are skipped rather than counted as index 0, which would pull against that default
        # by implying a first iteration that was never recorded.
        #
        # Logs written before #719 carried 1-based numbers, so this over-reports them by one.
        # Accepted: pre-#719 logs are out of scope, and the alternative (sniffing whether the
        # minimum index is 0 or 1) guesses at the format from its contents. Nothing in the file
        # records which base wrote it — if that ever matters, the fix is a version field in the
        # metadata entry, not a heuristic here.
        num_iters = (
            max(
                (
                    e["iteration"]
                    for e in entries
                    if e.get("type") == "iteration" and "iteration" in e
                ),
                default=-1,
            )
            + 1
        )
        lines.append(f"- **Total iterations**: {num_iters}")
        lines.append("")
        parts.append("\n".join(lines))

    # Process each iteration, showing only new messages compared to previous
    prev_prompt_len = 0
    for entry in entries:
        if entry.get("type") != "iteration":
            continue

        iteration = entry.get("iteration", "?")
        timestamp = entry.get("timestamp", "")
        iteration_time = entry.get("iteration_time")
        prompt = entry.get("prompt", [])
        response = entry.get("response", "")
        code_blocks = entry.get("code_blocks", [])
        final_answer = entry.get("final_answer")

        iter_parts = [f"---\n\n## Iteration {iteration}"]
        if timestamp:
            iter_parts.append(f"*{timestamp}*")
        if iteration_time is not None:
            iter_parts.append(f"*Duration: {iteration_time:.1f}s*")
        iter_parts.append("")

        # Show messages that are new since the previous iteration
        new_msgs = prompt[prev_prompt_len:]
        for i, msg in enumerate(new_msgs):
            msg_idx = prev_prompt_len + i
            iter_parts.append(f"**[{msg_idx}]** {_format_message(msg)}")

        prev_prompt_len = len(prompt)

        # Code blocks
        if code_blocks:
            iter_parts.append(_format_code_blocks(code_blocks))

        # Response text
        if response:
            iter_parts.append(f"### Response\n\n{response}\n")

        # Final answer
        if final_answer:
            iter_parts.append(f"### FINAL ANSWER\n\n```\n{final_answer}\n```\n")

        parts.append("\n".join(iter_parts))

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RLM JSONL log to readable Markdown"
    )
    parser.add_argument("input", help="Path to RLM .jsonl log file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    md = convert_log(args.input)

    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
