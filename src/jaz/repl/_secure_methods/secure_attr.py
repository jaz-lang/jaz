"""Runtime attribute-access backstops for ``getattr``/``hasattr``/``setattr``/``delattr``/``vars``.

The compiler's :class:`AllowedAttributesChecker` is *static* — it only sees literal
``ast.Attribute`` nodes (``obj.__globals__``). It cannot see ``getattr(obj, "__globals__")``
or ``vars(obj)``, whose attribute name is a runtime value, so those builtins are a hole
straight through the ``allowed_attributes`` allow-list. These wrappers re-check the attribute
*name* argument against that same allow-list at call time — exactly parallel to the runtime
``__import__`` backstop (:mod:`secure_import`), which exists because the static import checker
likewise misses computed names.

``allowed_attributes`` is a gitignore-style glob allow-list (see :mod:`.._glob_allowlist`): a name
is permitted iff it matches. The secure default ``["*", "!__*"]`` allows every attribute except
dunder-prefixed ones — the allow-only inverse of the old ``forbidden_attributes=["__*"]`` denylist.

Scope and limits — these backstops are **defense-in-depth, not a security boundary**; genuinely
untrusted code needs OS/process isolation. They close the *builtin* ``getattr``-family routes but do
**not** make the dunder policy airtight:
1. The default allow-list is *dunder-scoped*, so the entire **non-dunder introspection surface is
   un-policed** by both the static checker and these backstops — ``gi_frame``/``cr_frame``/``ag_frame``
   (generators/coroutines), ``tb_frame``/``tb_next`` (tracebacks), ``f_globals``/``f_builtins``/
   ``f_back``/``f_code`` (frames) are ordinary (non-dunder) names, so ``["*", "!__*"]`` *allows* them.
   A real-module frame's ``f_globals["__builtins__"]`` is the real builtins table → reaches and calls
   real ``eval``. Closing it would mean denylisting those names — the whack-a-mole the allow-list
   design avoids — so it is deliberately left open.
2. ``"{0.__globals__}".format(fn)`` reaches attributes at the C level via ``PyObject_GetAttr``
   without ever calling the wrapped ``getattr`` (a known, un-closeable CPython vector; read-only).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from .._glob_allowlist import allows_everything, compile_allowlist, is_allowed

# Runtime attribute access that trips the policy raises AttributeError (not the compiler's
# SyntaxError): a SyntaxError from a *running* getattr call would be nonsensical, and
# AttributeError is the natural failure for attribute access. Like every sandbox violation it
# is Exception-rooted, so the REPL's exec boundary renders it as recoverable agent feedback.
_MSG = "Access to attribute {name!r} is not allowed in the REPL."


def make_secure_getattr(patterns: Sequence[str]):
    """A ``getattr`` that rejects a name NOT in the ``patterns`` allow-list before delegating.

    Returns the real builtin unchanged when the allow-list is trivially allow-all (``["*"]`` /
    ``["**"]``) — zero overhead, identical semantics — so an un-sandboxed REPL is unaffected."""
    if allows_everything(patterns):
        return getattr
    spec = compile_allowlist(patterns)

    def secure_getattr(obj, name, *default):
        if isinstance(name, str) and not is_allowed(spec, name):
            raise AttributeError(_MSG.format(name=name))
        return getattr(obj, name, *default)

    return secure_getattr


def make_secure_hasattr(patterns: Sequence[str]):
    """A ``hasattr`` that rejects a name NOT in the ``patterns`` allow-list before delegating.

    ``hasattr`` does the same C-level attribute lookup ``getattr`` does, so an unwrapped
    ``hasattr`` is an existence *oracle* for disallowed names (``hasattr(x, "__globals__")``). We
    raise rather than return ``False`` so the behavior matches the ``getattr`` backstop exactly
    (and reveals nothing about whether the attribute exists)."""
    if allows_everything(patterns):
        return hasattr
    spec = compile_allowlist(patterns)

    def secure_hasattr(obj, name):
        if isinstance(name, str) and not is_allowed(spec, name):
            raise AttributeError(_MSG.format(name=name))
        return hasattr(obj, name)

    return secure_hasattr


def make_secure_setattr(patterns: Sequence[str]):
    """A ``setattr`` that rejects a name NOT allowed (e.g. ``setattr(o, "__class__", X)``)."""
    if allows_everything(patterns):
        return setattr
    spec = compile_allowlist(patterns)

    def secure_setattr(obj, name, value):
        if isinstance(name, str) and not is_allowed(spec, name):
            raise AttributeError(_MSG.format(name=name))
        return setattr(obj, name, value)

    return secure_setattr


def make_secure_delattr(patterns: Sequence[str]):
    """A ``delattr`` that rejects a name NOT allowed."""
    if allows_everything(patterns):
        return delattr
    spec = compile_allowlist(patterns)

    def secure_delattr(obj, name):
        if isinstance(name, str) and not is_allowed(spec, name):
            raise AttributeError(_MSG.format(name=name))
        return delattr(obj, name)

    return secure_delattr


def make_secure_vars(patterns: Sequence[str]):
    """A ``vars`` guarded on the object form (``vars(obj)`` ≡ ``obj.__dict__``).

    ``vars(obj)`` reaches ``__dict__``, so it is rejected whenever ``"__dict__"`` is not in the
    allow-list (it isn't under the default ``["*", "!__*"]``). ``vars()`` with no argument returns
    the caller's own namespace and exposes nothing new; it is preserved. We read the caller's locals
    via ``sys._getframe(1)`` because delegating to the real ``vars()`` from inside this wrapper would
    return the *wrapper's* frame, not the agent's."""
    if allows_everything(patterns):
        return vars
    spec = compile_allowlist(patterns)
    dict_allowed = is_allowed(spec, "__dict__")

    def secure_vars(*args):
        if not args:
            # vars() ≡ locals() of the *caller* (the agent's exec frame is one level up).
            return sys._getframe(1).f_locals
        if not dict_allowed:
            raise AttributeError(_MSG.format(name="__dict__"))
        return vars(args[0])

    return secure_vars
