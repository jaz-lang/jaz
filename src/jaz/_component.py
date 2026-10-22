"""How a configured component describes itself: its ``__repr__`` is its constructor call.

A component (an LLM backend, a REPL, a protocol) is configured by constructing it and keeps no
getters for its settings — they go in and are used. What a client that needs to *describe* a
live component gets is ``repr(component)``, rendered by the class from its own state as the
call that built it, secrets as ``'***'``. It is the same convention a hook has through its
dataclass ``repr``, and what the console's settings view and a trace show.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# Words that mark a secret-bearing setting, matched as whole words of a key (split on `_`, `-`
# and case changes; case-insensitive) at any depth of a mapping value: `api_key`, `github_token`,
# `client_secret`, `aws_credentials`, `apiKey`, and the header case
# `extra_headers={"Authorization": ...}`. Whole words rather than substrings because this is a
# public `repr`: `max_tokens` and `max_completion_tokens` are not secrets and must read as what
# they are. `prompt_cache_key` is a routing key, not a credential, and is named as such.
SECRET_WORDS = frozenset(
    {
        "key",
        "secret",
        "token",
        "password",
        "credential",
        "credentials",
        "authorization",
        "auth",
    }
)
NOT_SECRET = frozenset({"prompt_cache_key"})


def is_secret_name(name: str) -> bool:
    """Whether a setting name denotes a secret, by whole-word match."""
    if name in NOT_SECRET:
        return False
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return any(w in SECRET_WORDS for w in re.split(r"[^a-z0-9]+", words) if w)


def repr_setting(name: str, value: Any) -> str:
    """``repr`` of one setting for a component's ``__repr__``: ``'***'`` for a secret."""
    if is_secret_name(name):
        return "'***'" if value is not None else "None"
    if isinstance(value, Mapping):
        items = ", ".join(f"{k!r}: {repr_setting(str(k), v)}" for k, v in value.items())
        return "{" + items + "}"
    if callable(value) and not isinstance(value, type):
        return f"<{getattr(value, '__name__', type(value).__name__)}>"
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def repr_call(cls_name: str, *settings: tuple[str, Any]) -> str:
    """``ClassName(a=..., b=...)`` from ``(name, value)`` pairs — the constructor call, rendered."""
    return (
        f"{cls_name}("
        + ", ".join(f"{n}={repr_setting(n, v)}" for n, v in settings)
        + ")"
    )
