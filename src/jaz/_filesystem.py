"""Local filesystem backing for the public, immutable filesystem handles."""

from __future__ import annotations

import errno
import os
import stat
from _thread import RLock
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Literal, Self, TextIO
from weakref import WeakKeyDictionary, WeakValueDictionary

type _Access = Literal["read", "write"]


@dataclass(frozen=True, slots=True)
class _State:
    path: Path
    access: _Access
    name: str


# Keep authority off the agent-visible object: even a single-underscore attribute is
# accessible under the default REPL policy. Weak keys let returned handles outlive an
# invoke without retaining handles that callers have discarded.
_guard = Lock()
_states: WeakKeyDictionary[_Handle, _State] = WeakKeyDictionary()
_locks: WeakValueDictionary[tuple, RLock] = WeakValueDictionary()


class _Handle:
    __slots__ = ("__weakref__",)

    def __new__(cls) -> Self:
        raise TypeError(
            "Create handles with jaz.filesystem.from_local() and directory methods."
        )

    @property
    def name(self) -> str:
        """The file or directory's name, without its host filesystem path."""
        return _state(self).name

    @property
    def access(self) -> Literal["read", "write"]:
        """The granted access: 'read', or 'write' (which also permits reading)."""
        return _state(self).access

    def handle(self, *, access: Literal["read", "write"]) -> Self:
        """Share this resource with the same or narrower access; never copy its contents.

        Both handles remain usable. A read handle cannot grant write access.
        """
        _validate_access(access)
        state = _state(self)
        if state.access == "read" and access == "write":
            raise PermissionError("A read handle cannot grant write access.")
        return _make_handle(type(self), replace(state, access=access))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, access={self.access!r})"

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict) -> Self:
        # The grant is immutable; copying Python containers must not copy disk contents.
        return self


def _validate_access(access: str) -> None:
    # Store only literal strings, never objects with mutable/custom equality.
    if type(access) is not str or access not in ("read", "write"):
        raise ValueError("access must be 'read' or 'write'.")


def _state(handle: _Handle) -> _State:
    with _guard:
        return _states[handle]


def _make_handle[T: _Handle](cls: type[T], state: _State) -> T:
    handle = object.__new__(cls)
    with _guard:
        _states[handle] = state
    return handle


def _checked_path(handle: _Handle) -> Path:
    path = _state(handle).path
    # Handles pin a resolved path. Replacing it (or an ancestor) with a symlink must
    # not redirect a previously issued grant somewhere else.
    if path.resolve() != path:
        raise PermissionError(
            "The handle's path has been redirected by a symbolic link."
        )
    return path


def _child_state(directory: _Handle, path: str | os.PathLike[str]) -> _State:
    relative = Path(path)
    if relative.is_absolute():
        raise PermissionError("Use a relative path inside this directory.")
    root = _checked_path(directory)
    child = (root / relative).resolve()
    if not child.is_relative_to(root):
        raise PermissionError("The path is outside this directory's scope.")
    # Public properties can be overridden on a class; authority comes from the
    # registered state, not from an overridable attribute on an input object.
    return _State(
        child, _state(directory).access, relative.name or root.name or str(root)
    )


def _entries(directory: _Handle) -> Generator[tuple[_State, bool], None, None]:
    """Yield accessible regular files and directories, without descending into them."""
    with os.scandir(_checked_path(directory)) as entries:
        for entry in entries:
            try:
                state = _child_state(directory, entry.name)
                mode = state.path.stat().st_mode
            except (
                PermissionError,
                FileNotFoundError,
                NotADirectoryError,
                RuntimeError,
            ):
                # Outside/broken/looping symlinks and concurrently removed entries do
                # not become grants merely because they occur in a directory listing.
                continue
            except OSError as exc:
                if exc.errno != errno.ELOOP:
                    raise
                continue
            if stat.S_ISDIR(mode):
                yield state, True
            elif stat.S_ISREG(mode):
                yield state, False


def _lock_for(key: tuple) -> RLock:
    with _guard:
        lock = _locks.get(key)
        if lock is None:
            lock = RLock()
            _locks[key] = lock
        return lock


@contextmanager
def _open_file(
    handle: _Handle, *, write: bool = False, create: bool = False
) -> Iterator[TextIO]:
    state = _state(handle)
    if write and state.access != "write":
        raise PermissionError("This file handle is read-only.")
    # The path lock also serializes creation; the inode lock joins independently
    # registered paths and hard links. Neither lock is held across agent turns.
    with _lock_for(("path", state.path)):
        path = _checked_path(handle)
        flags = os.O_RDWR if write else os.O_RDONLY
        if create:
            flags |= os.O_CREAT
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags, 0o666)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("File handles only support regular files.")
            with _lock_for(("inode", info.st_dev, info.st_ino)):
                stream = os.fdopen(
                    fd, "r+" if write else "r", encoding="utf-8", newline="\n"
                )
                fd = None  # The stream now owns the descriptor.
                with stream:
                    yield stream
        finally:
            if fd is not None:
                os.close(fd)


def _write_text(stream: TextIO, text: str) -> None:
    stream.seek(0)
    stream.write(text)
    stream.truncate()
