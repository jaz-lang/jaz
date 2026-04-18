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
    markdown_lines = []

    for role, content in messages:
        # Map role to header
        if role == "system":
            header = "## System"
        elif role == "user":
            header = "## User"
        elif role == "assistant":
            header = "## Assistant"
        else:
            # Unknown role, use as-is
            header = f"## {role.capitalize()}"

        # Add header and content
        markdown_lines.append(header)
        markdown_lines.append("")
        markdown_lines.append(content)
        markdown_lines.append("")

    # Join all lines
    markdown_content = "\n".join(markdown_lines)

    # Output to file or stdout
    if output_file_path:
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


def process_all_logs(search_path: Path | None = None, search_pattern: str = "*.log"):
    """
    Process all {search_pattern} files and save conversation.md in each subdirectory.

    Args:
        search_path: Directory to search from (default: current directory)
    """
    log_files = find_all_jaz_logs(search_path, search_pattern)

    if not log_files:
        print(f"No {search_pattern} files found in {search_path}")
        return

    print(f"Found {len(log_files)} {search_pattern} file(s)")

    for log_file in log_files:
        # Save conversation.md in the same directory as the log file
        output_file = log_file.parent / f"{log_file.stem}.md"
        print(f"Processing: {log_file}")
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
        "log_file",
        nargs="?",
        type=Path,
        help="Path to a single JAZ log file (not used with --all)",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        type=Path,
        help="Path to output Markdown file (default: stdout, not used with --all)",
    )
    parser.add_argument(
        "--search-path",
        type=Path,
        default=Path.cwd(),
        help="Directory to search for log files when using --all (default: current directory)",
    )

    args = parser.parse_args()

    if args.all:
        # Process all *.jaz.log files
        process_all_logs(args.search_path, args.search_pattern)
    else:
        # Process single file
        if not args.log_file:
            parser.error("log_file is required when not using --all")

        if not args.log_file.exists():
            print(f"Error: File not found: {args.log_file}")
            sys.exit(1)

        format_conversation_to_markdown(args.log_file, args.output_file)


if __name__ == "__main__":
    main()
