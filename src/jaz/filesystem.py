"""Scoped local files and directories, passed as ordinary JAZ inputs and outputs.

Register a directory in host code with :func:`from_local`, then pass only the handles
an agent needs. Handles share live files; they do not copy, version, or own them.
Permissions are checked by these methods, independently of the REPL's ``open`` policy.
This in-process interface is not an OS sandbox. Other programs, processes, and direct
filesystem access do not participate in its operation locks.
"""

from __future__ import annotations

import os as _os
from contextlib import closing as _closing
from itertools import islice as _islice
from pathlib import Path as _Path
from typing import Literal

from ._filesystem import (
    _child_state,
    _entries,
    _Handle,
    _make_handle,
    _open_file,
    _State,
    _validate_access,
    _write_text,
)

__all__ = ["File", "Directory", "from_local"]

_MAX_READ_BYTES = 64 * 1024
_PREVIEW_ENTRIES = 20


class File(_Handle):
    """A live local file with immutable read or write access.

    Obtain one from ``directory.file(path)``. Write access includes reading. Handles
    remain usable after an agent returns; deleting the backing file makes reads fail
    normally. Each operation is serialized with other operations through JAZ handles
    to the same file, including aliases. A read followed by a write is not a transaction.
    """

    __slots__ = ()

    def read_text(self, *, offset: int = 1, limit: int = 200) -> str:
        """Read up to ``limit`` UTF-8 lines, starting at the 1-based ``offset``.

        Newlines are preserved (lines end at LF, including CRLF). Past EOF returns
        an empty string. A result exceeding 64 KiB raises ValueError instead of
        silently truncating; request fewer lines. Both arguments must be positive ints.
        """
        for name, value in (("offset", offset), ("limit", limit)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        with _open_file(self) as stream:
            for _ in range(offset - 1):
                # Skip long lines in bounded chunks rather than materializing them.
                while True:
                    chunk = stream.readline(_MAX_READ_BYTES + 1)
                    if not chunk:
                        return ""
                    if chunk.endswith("\n"):
                        break
            lines = []
            remaining = _MAX_READ_BYTES
            for _ in range(limit):
                line = stream.readline(remaining + 1)
                if not line:
                    break
                remaining -= len(line.encode("utf-8"))
                if remaining < 0:
                    raise ValueError(
                        "Read exceeds 64 KiB; request fewer lines with limit."
                    )
                lines.append(line)
            return "".join(lines)

    def write_text(self, text: str) -> None:
        """Create or replace this file with UTF-8 text; its parent must already exist.

        Requires write access. Does not create directories or provide rollback on I/O
        failure. Other operations through JAZ handles cannot observe a partial write.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        text.encode("utf-8")  # Reject invalid Unicode before changing the file.
        with _open_file(self, write=True, create=True) as stream:
            _write_text(stream, text)

    def replace_text(self, old: str, new: str) -> None:
        """Replace exactly one occurrence of ``old`` in the current file with ``new``.

        Requires write access and an existing UTF-8 file. Empty, missing, or ambiguous
        matches raise ValueError without changing the file. The read, match check, and
        write hold one operation lock; no lock persists after the method returns.
        """
        if not isinstance(old, str) or not isinstance(new, str):
            raise TypeError("old and new must be strings.")
        if not old:
            raise ValueError("old must not be empty.")
        new.encode("utf-8")
        with _open_file(self, write=True) as stream:
            text = stream.read()
            start = text.find(old)
            if start < 0 or text.find(old, start + 1) >= 0:
                raise ValueError(
                    "Expected exactly one occurrence of old in the current file."
                )
            _write_text(stream, text[:start] + new + text[start + len(old) :])

    def __str__(self) -> str:
        methods = [
            "read_text(*, offset=1, limit=200): UTF-8 text, 1-based lines, at most 64 KiB",
            "handle(*, access='read'|'write'): share this file; access can only narrow",
        ]
        if self.access == "write":
            methods += [
                "write_text(text): create or replace the file",
                "replace_text(old, new): replace exactly one match in the current file",
            ]
        return f"{self!r}\n" + "\n".join(f"  {method}" for method in methods)


class Directory(_Handle):
    """A local directory subtree with immutable read or write access.

    Obtain a root with ``from_local`` in host code. Navigation accepts only relative
    paths within this subtree; derived directories cannot navigate back to their parent.
    In-scope symlinks are resolved when a handle is created, pinning the target path.
    """

    __slots__ = ()

    def file(self, path: str | _os.PathLike[str]) -> File:
        """Get a file handle with this directory's access, without reading its contents.

        The file may not exist yet; writing creates it if its parent exists. Paths
        outside this subtree, including escaping symlinks, are rejected.
        """
        state = _child_state(self, path)
        if state.path.exists() and not state.path.is_file():
            if state.path.is_dir():
                raise IsADirectoryError(str(path))
            raise ValueError("File handles only support regular files.")
        return _make_handle(File, state)

    def directory(self, path: str | _os.PathLike[str]) -> Directory:
        """Get an existing subdirectory with this access and a narrower subtree scope."""
        state = _child_state(self, path)
        if not state.path.exists():
            raise FileNotFoundError(str(path))
        if not state.path.is_dir():
            raise NotADirectoryError(str(path))
        return _make_handle(Directory, state)

    def list(self) -> list[File | Directory]:
        """List immediate file/directory handles, sorted by name, inheriting this access.

        No recursion. Broken or escaping symlinks and non-regular files are omitted.
        """
        return sorted(
            (
                _make_handle(Directory if is_dir else File, state)
                for state, is_dir in _entries(self)
            ),
            key=lambda handle: handle.name,
        )

    def __str__(self) -> str:
        description = [
            repr(self),
            "  file(path): get a file inside this directory (may not exist yet)",
            "  directory(path): get an existing subdirectory",
            "  list(): list immediate file/directory handles, sorted by name",
            "  handle(*, access='read'|'write'): share this subtree; access can only narrow",
        ]
        try:
            # Automatic prompt rendering visits at most 21 accessible entries, never
            # recursively. Close scandir even when the preview stops before exhaustion.
            with _closing(_entries(self)) as entries:
                preview = list(_islice(entries, _PREVIEW_ENTRIES + 1))
        except OSError as exc:
            description.append(
                f"  Entries unavailable ({type(exc).__name__}); operations may fail."
            )
        else:
            description.append("  Entries:" if preview else "  Entries: (empty)")
            for state, is_dir in sorted(
                preview[:_PREVIEW_ENTRIES], key=lambda item: item[0].name
            ):
                description.append(f"    {state.name!r}{'/' if is_dir else ''}")
            if len(preview) > _PREVIEW_ENTRIES:
                description.append("    ... more entries; call list()")
        return "\n".join(description)


def from_local(
    path: str | _os.PathLike[str], *, access: Literal["read", "write"] = "read"
) -> Directory:
    """Register an existing local directory in host code; defaults to read access.

    Pass selected handles to agents, not this factory. ``access='write'`` grants both
    reading and writing; ``handle(access='read')`` narrows that grant for a recipient.
    The host owns the directory's lifetime. No files are copied or removed on return.
    """
    _validate_access(access)
    root = _Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(str(path))
    return _make_handle(Directory, _State(root, access, root.name or str(root)))
