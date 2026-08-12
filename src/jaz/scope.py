"""Ambient inputs via ``jaz.scope``.

A *scoped* input is a value bound with ``with jaz.scope(name=value): ...`` that
is automatically available to every ``jaz.invoke`` call lexically enclosed by the
block — and to every nested invoke below them — without threading it by hand as a
kwarg at each layer. It is the general form of the auto-propagation that
``libraries=`` already provides for tool catalogs, lifted to arbitrary values
(a session id, an API client, a shared dataframe, ...).

Why a context manager rather than threading-by-kwarg: a value that should live
across many levels of nested invokes otherwise has to be re-passed at every
intermediate ``jaz.invoke(...)`` call, and a single forgotten layer silently
drops it from the rest of the subtree. ``jaz.scope`` makes the value ambient and
makes re-passing it an *error* (see the conflict check in ``invoke._invoke``), so
the binding can't silently disagree with an explicit kwarg.

Storage — **a single dict-valued ``ContextVar``**:

- One mapping (rather than one ContextVar per name) keeps propagation uniform:
  each ``Task`` / ``copy_context()`` boundary copies exactly one dict, and
  "what's currently in scope" is a single read.
- The ContextVar is deliberately only a *same-context bridge*: it carries scope
  from a ``with jaz.scope(...)`` block to the ``jaz.invoke()`` call lexically
  enclosed by it (always the same context, so it always works). It is **not**
  relied on to cross thread/task boundaries. The cross-boundary hop into a nested
  invoke is done by snapshotting the scope at the invoke boundary and binding it
  as plain data on the agent-facing ``jaz`` object — exactly how ``config``
  already propagates (see ``invoke._invoke`` / ``library.get_jaz_library``). A
  bare ContextVar read inside a ``ThreadPoolExecutor`` worker would see an empty
  context and silently drop scope; binding-as-data does not.

Concurrency / mutation semantics:

- The snapshot copies the *mapping*; the values themselves are shared **by
  reference**. Under sequential or asyncio execution that is benign (no two
  invokes mutate concurrently). Under a ``ThreadPoolExecutor`` the scope is
  visible *and* truly parallel, so two workers mutating the same scoped
  ``list``/``CostTracker`` is a data race the caller must guard — bind immutable
  values, or rebind a fresh instance inside each worker via a nested
  ``jaz.scope(...)`` block (which is properly isolated per Task/context).
- Process boundaries are forbidden for now; ``invoke._invoke`` raises if it runs
  outside the main process (the bound ``jaz`` and most scoped values don't
  survive pickling).

Composition with descriptions: a scoped value carrying a ``jaz.describe``
description renders using that description in every invoke's prompt header, with
no extra wiring (the rendering path is shared with ordinary inputs).
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

# The single dict-valued scope ContextVar. See the module docstring for why this
# is only a same-context bridge and not the cross-boundary propagation channel.
# The default is `None` (not `{}`) because a mutable ContextVar default is shared
# across contexts; readers normalize `None` to an empty dict.
_scope_var: contextvars.ContextVar[dict[str, object] | None] = contextvars.ContextVar(
    "jaz_scope", default=None
)


@contextmanager
def scope(**values: object) -> Iterator[None]:
    """Bind ``values`` as ambient inputs for all enclosed ``jaz.invoke`` calls.

    Equivalent to passing each value as a kwarg to every invoke (and every nested
    invoke) within the ``with`` block — but without manual threading::

        with jaz.scope(fs=fs_tools, web=web_client):
            jaz.invoke(jaz.ReturnType(Report), task="analyze logs", target_date="2026-05-30")
            # fs and web are bound in the REPL of this invoke and all recursive
            # sub-invokes. On the other hand, task and target_date are local to this
            # top-level invoke and do not propagate to sub-invokes.

    The construct is **symmetric**: the agent uses it inside its REPL exactly as
    user code does, to make an intermediate value ambient to the sub-invokes it
    spawns.

    Semantics:

    - **Kwargs-only.** Names must be valid Python identifiers (they land as REPL
      globals); splat a dynamic mapping with ``jaz.scope(**d)``.
    - **Inner shadows outer on the same name.** Nesting
      ``with jaz.scope(fs=A): with jaz.scope(fs=B): ...`` rebinds ``fs`` to ``B``
      inside the inner block and restores ``A`` on exit (standard ``with``
      semantics).
    - **An explicit kwarg colliding with a scoped name raises.** Passing a scoped
      name again as an invoke kwarg raises a conflict ``ValueError`` at invoke
      time. To give one invoke a different value for a scoped name, nest another
      ``jaz.scope`` around it rather than re-passing the kwarg.
    - **No partial unbinding.** There is no way to remove a name for a block; an
      inner scope can only shadow it with a new value.
    """
    # Validate at the binding site, not at a distant invoke: keys land as REPL
    # globals and prompt variable names, so a non-identifier key (reachable via
    # `jaz.scope(**some_dict)`) would otherwise fail cryptically in the REPL or
    # produce a malformed prompt block far from where the bad name was introduced.
    for k in values:
        if not k.isidentifier():
            raise ValueError(f"jaz.scope key {k!r} is not a valid Python identifier")

    prev = _scope_var.get() or {}
    merged = {**prev, **values}  # inner shadows outer for same key
    token = _scope_var.set(merged)
    try:
        yield
    finally:
        _scope_var.reset(token)


def current_scope() -> dict[str, object]:
    """Return a copy of the values currently in scope (a runtime escape hatch).

    Exposed both to user code and in the agent's REPL (``jaz.current_scope``) for
    introspection and testing. Returns a defensive copy so callers cannot mutate
    the live binding; mutating a *value* inside the returned dict still mutates
    the shared object (values are shared by reference — see the module docstring).
    """
    return dict(_scope_var.get() or {})
