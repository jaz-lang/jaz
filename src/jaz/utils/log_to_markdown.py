#!/usr/bin/env python3
"""
Extract conversation history from a JAZ log file and format it as Markdown.
"""

import argparse
import ast
import re
import sys
from pathlib import Path


def parse_message_line(line: str) -> tuple[str, str] | None:
    """
    Parse a line containing [Agent] message: {'role': '...', 'content': '...'}

    Returns:
        Tuple of (role, content) if successful, None otherwise
    """
    # Match lines that start with [Agent] message:, optionally preceded by a timestamp
    match = re.match(
        r"(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} )?\[Agent\] message:\s*(.+)", line
    )
    if not match:
        return None

    # Try to parse the dictionary
    dict_str = match.group(1)
    try:
        # Use ast.literal_eval to safely parse the dictionary
        message_dict = ast.literal_eval(dict_str)
        role = message_dict.get("role", "").lower()
        content = message_dict.get("content", "")
        return (role, content)
    except (ValueError, SyntaxError):
        return None


def _parse_invoke_depth(line: str, event: str) -> int | None:
    """Parse depth from an invoke enter/exit line."""
    pattern = (
        r"(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} )?"
        rf"\[Agent\] invoke {event}:.*depth=(\d+)"
    )
    match = re.match(pattern, line)
    if match:
        return int(match.group(1))
    return None


class AgentNode:
    """Represents one invoke call in the agent call tree."""

    def __init__(self, depth: int, child_index: int):
        self.depth = depth
        # 1-indexed position among siblings (0 for root)
        self.child_index = child_index
        # Ordered list of content items:
        #   ("messages", [(role, content), ...])
        #   ("subagent",  AgentNode)
        self.items: list = []
        self._pending: list[tuple[str, str]] = []
        self._child_count = 0

    def add_message(self, role: str, content: str) -> None:
        self._pending.append((role, content))

    def _flush(self) -> None:
        if self._pending:
            self.items.append(("messages", self._pending[:]))
            self._pending.clear()

    def add_child(self, child: "AgentNode") -> None:
        self._flush()
        self.items.append(("subagent", child))

    def finalize(self) -> None:
        self._flush()

    def has_subagents(self) -> bool:
        return any(kind == "subagent" for kind, _ in self.items)


def parse_agent_tree(log_file_path: Path) -> AgentNode | None:
    """
    Parse a JAZ log file into a tree of AgentNode objects.

    Uses depth directly to determine parent-child relationships.
    A new ``invoke enter depth=D`` always attaches to the active node at
    ``depth=D-1``, even when a previous node at depth=D never received its
    ``invoke exit`` (e.g. aborted due to a failed return validator).

    Returns the root AgentNode, or None if no invoke enter line is found.
    """
    with open(log_file_path, encoding="utf-8") as f:
        lines = f.readlines()

    root: AgentNode | None = None
    # Maps depth → the currently active AgentNode at that depth
    active: dict[int, AgentNode] = {}

    def _finalize_depth_and_deeper(min_depth: int) -> None:
        for d in sorted(k for k in active if k >= min_depth):
            active.pop(d).finalize()

    for line in lines:
        line = line.strip()

        depth = _parse_invoke_depth(line, "enter")
        if depth is not None:
            # Implicitly close any existing node at this depth (missing exit)
            # and any deeper orphaned nodes.
            _finalize_depth_and_deeper(depth)

            if depth == 1:
                node = AgentNode(depth=depth, child_index=0)
                root = node
            else:
                parent = active.get(depth - 1)
                if parent is not None:
                    parent._child_count += 1
                    node = AgentNode(depth=depth, child_index=parent._child_count)
                    parent.add_child(node)
                else:
                    # No active parent — treat as an additional root
                    node = AgentNode(depth=depth, child_index=0)
                    if root is None:
                        root = node

            active[depth] = node
            continue

        depth = _parse_invoke_depth(line, "exit")
        if depth is not None:
            _finalize_depth_and_deeper(depth)
            continue

        parsed = parse_message_line(line)
        if parsed and active:
            role, content = parsed
            # Assign to the deepest currently active node
            active[max(active)].add_message(role, content)

    # Finalize any nodes still active (truncated logs)
    for node in list(active.values()):
        node.finalize()

    return root


def _messages_to_markdown(messages: list[tuple[str, str]]) -> list[str]:
    lines = []
    for role, content in messages:
        if role == "system":
            header = "## System"
        elif role == "user":
            header = "## User"
        elif role == "assistant":
            header = "## Assistant"
        else:
            header = f"## {role.capitalize()}"
        lines.append(header)
        lines.append("")
        lines.append(content)
        lines.append("")
    return lines


def _write_node(node: AgentNode, node_dir: Path) -> None:
    """Write the markdown for this node and recursively for all children."""
    node_dir.mkdir(parents=True, exist_ok=True)

    if node.depth == 1:
        md_path = node_dir / "root.md"
    else:
        md_path = node_dir / "subagent.md"

    md_lines: list[str] = []
    subagent_num = 0

    for kind, data in node.items:
        if kind == "messages":
            md_lines.extend(_messages_to_markdown(data))
        elif kind == "subagent":
            child: AgentNode = data
            subagent_num += 1
            child_dir_name = f"subagent_{child.child_index}"
            rel_path = f"{child_dir_name}/subagent.md"
            md_lines.append("## Subagent")
            md_lines.append("")
            md_lines.append(f"[Subagent {child.child_index}]({rel_path})")
            md_lines.append("")
            # Recurse into child
            _write_node(child, node_dir / child_dir_name)

    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def format_conversation_to_markdown_tree(
    log_file_path: Path,
    output_dir: Path | None = None,
    *,
    quiet: bool = False,
) -> None:
    """
    Parse a JAZ log file and write a directory tree of Markdown files
    mirroring the agent call tree.

    The root agent's conversation goes in root.md; each subagent gets its
    own subdirectory subagent_N/ containing subagent.md (and further
    subdirectories for its own children).

    Args:
        log_file_path: Path to the JAZ log file.
        output_dir: Directory to write into.  Defaults to a directory named
            after the log file's stem, placed next to the log file.
        quiet: Suppress progress messages.
    """
    if output_dir is None:
        output_dir = log_file_path.parent / log_file_path.stem

    root = parse_agent_tree(log_file_path)

    if root is None or not root.items:
        # No structured agent data found — fall back to a flat root.md
        if not quiet:
            print(f"No agent structure found in {log_file_path}; writing flat root.md")
        output_dir.mkdir(parents=True, exist_ok=True)
        format_conversation_to_markdown(
            log_file_path, output_dir / "root.md", quiet=quiet
        )
        return

    _write_node(root, output_dir)

    if not quiet:
        print(f"Agent tree written to: {output_dir}/")


def format_conversation_to_markdown(
    log_file_path: Path, output_file_path: Path | None = None, *, quiet: bool = False
):
    """
    Extract conversation history from JAZ log and format as Markdown.

    Args:
        log_file_path: Path to the JAZ log file
        output_file_path: Optional path to output file. If None, prints to stdout.
    """
    # Read the log file
    with open(log_file_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Extract messages
    messages = []
    for line in lines:
        parsed = parse_message_line(line.strip())
        if parsed:
            role, content = parsed
            messages.append((role, content))

    # Format messages as Markdown
    markdown_lines = _messages_to_markdown(messages)

    # Join all lines
    markdown_content = "\n".join(markdown_lines)

    # Output to file or stdout
    if output_file_path:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        if not quiet:
            print(f"Conversation history written to: {output_file_path}")
    else:
        print(markdown_content)


def find_all_jaz_logs(
    search_path: Path | None = None, search_pattern: str = "*.log"
) -> list[Path]:
    """
    Find all {search_pattern} files recursively from the search path.

    Args:
        search_path: Directory to search from (default: current directory)

    Returns:
        List of Path objects for all found {search_pattern} files
    """
    if search_path is None:
        search_path = Path.cwd()
    return sorted(search_path.rglob(search_pattern))


def process_all_logs(
    search_path: Path | None = None,
    search_pattern: str = "*.log",
    *,
    tree: bool = False,
):
    """
    Process all {search_pattern} files and save conversation.md in each subdirectory.

    Args:
        search_path: Directory to search from (default: current directory)
        tree: If True, write directory trees instead of flat markdown files.
    """
    log_files = find_all_jaz_logs(search_path, search_pattern)

    if not log_files:
        print(f"No {search_pattern} files found in {search_path}")
        return

    print(f"Found {len(log_files)} {search_pattern} file(s)")

    for log_file in log_files:
        print(f"Processing: {log_file}")
        if tree:
            output_dir = log_file.parent / log_file.stem
            format_conversation_to_markdown_tree(log_file, output_dir)
        else:
            output_file = log_file.parent / f"{log_file.stem}.md"
            format_conversation_to_markdown(log_file, output_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract conversation history from JAZ log files and format as Markdown"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all {search_pattern} files recursively from current or specified directory",
    )
    parser.add_argument(
        "--search-pattern",
        type=str,
        default="*.log",
        help="Pattern to search for log files when using --all (default: *.log)",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help=(
            "Write a directory tree of Markdown files mirroring the agent call tree. "
            "The root agent's conversation goes in root.md; each subagent gets its own "
            "subdirectory subagent_N/ with subagent.md (recursively). "
            "Without --all, the output positional argument is treated as a directory path."
        ),
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        type=Path,
        help="Path to a single JAZ log file (not used with --all)",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        type=Path,
        help=(
            "Path to output Markdown file (default: stdout, not used with --all). "
            "With --tree, this is treated as the output directory path."
        ),
    )
    parser.add_argument(
        "--search-path",
        type=Path,
        default=Path.cwd(),
        help="Directory to search for log files when using --all (default: current directory)",
    )

    args = parser.parse_args()

    if args.all:
        process_all_logs(args.search_path, args.search_pattern, tree=args.tree)
    else:
        # Process single file
        if not args.log_file:
            parser.error("log_file is required when not using --all")

        if not args.log_file.exists():
            print(f"Error: File not found: {args.log_file}")
            sys.exit(1)

        if args.tree:
            output_dir = args.output_file  # may be None; function handles that
            format_conversation_to_markdown_tree(args.log_file, output_dir)
        else:
            format_conversation_to_markdown(args.log_file, args.output_file)


if __name__ == "__main__":
    main()
