from __future__ import annotations

import builtins
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from jaz.exceptions import JazPermissionError

logger = logging.getLogger(__name__)


class _OpenPolicy(Protocol):
    @property
    def allow_file_writes(self) -> bool: ...

    @property
    def allowed_file_roots(self) -> list[str] | None: ...

    @property
    def log_file_access(self) -> bool: ...


def _get_open_mode(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "mode" in kwargs:
        return kwargs["mode"]
    if len(args) > 1:
        return args[1]
    return "r"


def _is_write_mode(mode: str) -> bool:
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _normalize_file_path(file_arg: Any) -> Path | None:
    """Return normalized path for path-like args, or None for file descriptors."""
    if isinstance(file_arg, int):
        return None
    fs_path = os.fspath(file_arg)
    if isinstance(fs_path, bytes):
        fs_path = os.fsdecode(fs_path)
    return Path(fs_path).expanduser().resolve(strict=False)


def _normalize_allowed_roots(roots: list[str] | None) -> tuple[Path, ...] | None:
    if roots is None:
        return None
    return tuple(Path(root).expanduser().resolve(strict=False) for root in roots)


def _is_within_allowed_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def make_secure_open(policy: _OpenPolicy) -> Callable[..., Any]:
    """Create an open wrapper bound to a permission policy."""
    allowed_roots = _normalize_allowed_roots(policy.allowed_file_roots)

    def _secure_open(*args: Any, **kwargs: Any) -> Any:
        mode = _get_open_mode(args, kwargs)
        file_arg = kwargs.get("file", args[0] if args else None)

        if (
            isinstance(mode, str)
            and _is_write_mode(mode)
            and not policy.allow_file_writes
        ):
            if policy.log_file_access:
                logger.info(
                    "Denied file write access path=%r mode=%r: writes are disabled",
                    file_arg,
                    mode,
                )
            raise JazPermissionError(
                "Opening files in write/append/update mode is not allowed in this REPL environment."
            )

        if allowed_roots is not None:
            if file_arg is None:
                raise TypeError("open() missing required argument 'file'")
            normalized_path = _normalize_file_path(file_arg)
            if normalized_path is None:
                if policy.log_file_access:
                    logger.info(
                        "Denied file descriptor access fd=%r mode=%r: root restrictions are active",
                        file_arg,
                        mode,
                    )
                raise JazPermissionError(
                    "File descriptor access is not allowed when allowed_file_roots is configured."
                )
            if not _is_within_allowed_roots(normalized_path, allowed_roots):
                if policy.log_file_access:
                    logger.info(
                        "Denied file access path=%s mode=%r: outside allowed roots",
                        normalized_path,
                        mode,
                    )
                roots_display = (
                    ", ".join(str(root) for root in allowed_roots) or "<none>"
                )
                raise JazPermissionError(
                    f"Access to '{normalized_path}' is not allowed. "
                    f"Allowed file roots: {roots_display}."
                )

        if policy.log_file_access:
            logger.debug("Allowed file access path=%r mode=%r", file_arg, mode)
        return builtins.open(*args, **kwargs)

    return _secure_open


class _StrictPolicy:
    def __init__(
        self,
        *,
        allow_file_writes: bool,
        allowed_file_roots: list[str] | None,
        log_file_access: bool,
    ) -> None:
        self.allow_file_writes = allow_file_writes
        self.allowed_file_roots = allowed_file_roots
        self.log_file_access = log_file_access


_DEFAULT_SECURE_OPEN = make_secure_open(
    _StrictPolicy(
        allow_file_writes=False,
        allowed_file_roots=None,
        log_file_access=False,
    )
)


def secure_open(*args: Any, **kwargs: Any) -> Any:
    """Default secure open wrapper with strict-deny write policy."""
    return _DEFAULT_SECURE_OPEN(*args, **kwargs)
