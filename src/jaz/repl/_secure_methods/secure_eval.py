from __future__ import annotations

import builtins
import inspect
from collections.abc import Callable
from typing import Any, Protocol

from jaz.exceptions import JazPermissionError


class _EvalPolicy(Protocol):
    @property
    def allow_eval_exec(self) -> bool: ...


def make_secure_eval(policy: _EvalPolicy) -> Callable[..., Any]:
    """Create an eval wrapper bound to a permission policy."""

    def _secure_eval(*args: Any, **kwargs: Any) -> Any:
        if not policy.allow_eval_exec:
            raise JazPermissionError("eval() is not allowed in this REPL environment.")
        if len(args) == 1 and "globals" not in kwargs and "locals" not in kwargs:
            frame = inspect.currentframe()
            try:
                if frame is not None and frame.f_back is not None:
                    caller = frame.f_back
                    return builtins.eval(args[0], caller.f_globals, caller.f_locals)
            finally:
                del frame
        return builtins.eval(*args, **kwargs)

    return _secure_eval


def secure_eval(*_args: Any, **_kwargs: Any) -> Any:
    """Default secure eval wrapper with strict-deny policy."""
    raise JazPermissionError("eval() is not allowed in this REPL environment.")
