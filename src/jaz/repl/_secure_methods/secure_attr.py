"""Runtime attribute-access backstops for ``getattr``/``hasattr``/``setattr``/``delattr``/``vars``.

The compiler's :class:`AllowedAttributesChecker` is *static* — it only sees literal
``ast.Attribute`` nodes (``obj.__globals__``). It cannot see ``getattr(obj, "__globals__")``
or ``vars(obj)``, whose attribute name is a runtime value, so those builtins are a hole
straight through the ``allowed_attributes`` allow-list. These wrappers re-check the attribute
*name* argument against that same allow-list at call time — exactly parallel to the runtime
``__import__`` backstop (:mod:`secure_import`), which exists because the static import checker
likewise misses computed names.

``allowed_attributes`` is a gitignore-style glob allow-list (see :mod:`.._glob_allowlist`): a name
is permitted iff it matches. The secure default (``DEFAULT_ALLOWED_ATTRIBUTES``) allows every
attribute except dunder-prefixed ones and the frame-bearing interpreter surface.

Scope and limits — these backstops are **defense-in-depth, not a security boundary**; genuinely
untrusted code needs OS/process isolation. They close the *builtin* ``getattr``-family routes but do
**not** make the attribute policy airtight:
1. ``"{0.__globals__}".format(fn)`` reaches attributes at the C level via ``PyObject_GetAttr``
   without ever calling the wrapped ``getattr`` (a known, un-closeable CPython vector; read-only).
   This applies to the frame surface too: ``"{0.gi_frame}".format(gen)`` still *stringifies* a frame
   the allow-list denies. It cannot hand back a callable, so the escape-to-execution stays closed,
   but do not read the frame denials as making frames unobservable.
2. **A name allow-list cannot police what an allowed name returns.** Any module object reachable
   through an ordinary (non-dunder, non-frame) attribute of an object handed into the sandbox is a
   full bypass of every axis — if an input ``h`` holds ``h.os``, then ``h.os.system(...)`` runs, no
   import and no denied attribute involved. What a host passes in ``inputs`` is part of the sandbox
   boundary, and nothing here can check it.
3. A host that widens the allow-list (``["*"]``, or one that re-admits ``f_globals``/``gi_frame``)
   reopens the escapes the default closes; these wrappers enforce the policy they are given, they do
   not add one of their own.
"""
# Limitation 1 used to read "the entire non-dunder introspection surface is un-policed ... so it is
# deliberately left open": `["*", "!__*"]` allowed `gi_frame`/`f_back`/`f_globals`, and a frame walk
# reached the real builtins table and the real `eval`. That is no longer true — the default now
# denies the walkable frame surface by name (`_DENIED_FRAME_ATTRIBUTES` in `python_repl`, closing
# #827), which is why the whack-a-mole objection was overruled: the names are enumerable and a
# drift test re-derives them from the running interpreter. The inert scalars on those same objects
# (`f_lineno`, `tb_lineno`, ...) stay allowed — they hand back nothing to walk.

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
