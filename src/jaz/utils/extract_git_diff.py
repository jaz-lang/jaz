#!/usr/bin/env python3
"""Extract the git_diff field from a config.json and write it to a file."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract git_diff from config.json to a .diff file."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to config.json (or a log directory containing one)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: git_diff.diff next to config.json)",
    )
    args = parser.parse_args()

    path: Path = args.config_path
    if path.is_dir():
        path = path / "config.json"

    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    config = json.loads(path.read_text())
    git_diff = config.get("git_diff")
    if not git_diff:
        print("No git_diff field found in config", file=sys.stderr)
        sys.exit(1)

    output = args.output or path.parent / "git_diff.diff"
    output.write_text(git_diff)
    print(f"Wrote {len(git_diff)} bytes to {output}")


if __name__ == "__main__":
    main()
