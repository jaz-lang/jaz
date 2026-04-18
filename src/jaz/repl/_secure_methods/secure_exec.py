from __future__ import annotations

import builtins
import inspect
from collections.abc import Callable
from typing import Any, Protocol

from jaz.exceptions import JazPermissionError


class _ExecPolicy(Protocol):
    @property
    def allow_eval_exec(self) -> bool: ...


def make_secure_exec(policy: _ExecPolicy) -> Callable[..., Any]:
    """Create an exec wrapper bound to a permission policy."""

    def _secure_exec(*args: Any, **kwargs: Any) -> Any:
        if not policy.allow_eval_exec:
            raise JazPermissionError("exec() is not allowed in this REPL environment.")
        if len(args) == 1 and "globals" not in kwargs and "locals" not in kwargs:
            frame = inspect.currentframe()
            try:
                if frame is not None and frame.f_back is not None:
                    caller = frame.f_back
                    return builtins.exec(args[0], caller.f_globals, caller.f_locals)
            finally:
                del frame
        return builtins.exec(*args, **kwargs)

    return _secure_exec


def secure_exec(*_args: Any, **_kwargs: Any) -> Any:
    """Default secure exec wrapper with strict-deny policy."""
    raise JazPermissionError("exec() is not allowed in this REPL environment.")
