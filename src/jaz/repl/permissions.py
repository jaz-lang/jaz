from __future__ import annotations

import builtins
from dataclasses import dataclass

from ._secure_methods.secure_eval import make_secure_eval
from ._secure_methods.secure_exec import make_secure_exec
from ._secure_methods.secure_open import make_secure_open


@dataclass(frozen=True)
class ReplPermissionPolicy:
    """Permission policy for Python REPL builtins."""

    allow_eval_exec: bool = False
    allow_file_writes: bool = False
    allowed_file_roots: list[str] | None = None
    log_file_access: bool = False


def build_allowed_builtins(
    *,
    base_builtins: dict[str, object] | None = None,
    policy: ReplPermissionPolicy | None = None,
) -> dict[str, object]:
    """Build a REPL builtins map with policy-bound secure wrappers."""
    if policy is None:
        policy = ReplPermissionPolicy()

    allowed_builtins = (
        builtins.__dict__.copy() if base_builtins is None else base_builtins.copy()
    )
    allowed_builtins["open"] = make_secure_open(policy)
    allowed_builtins["exec"] = make_secure_exec(policy)
    allowed_builtins["eval"] = make_secure_eval(policy)
    return allowed_builtins


DEFAULT_ALLOWED_BUILTINS = build_allowed_builtins()
