"""Ambient inputs via ``jaz.scope``.

A *scoped* input is a value bound with ``with jaz.scope(name=value): ...`` that
is automatically available to every ``jaz.invoke`` call lexically enclosed by the
block — and to every nested invoke below them — without threading it by hand as a
kwarg at each layer. It lifts ambient propagation to arbitrary values
(a session id, an API client, a shared dataframe, ...).

Why a context manager rather than threading-by-kwarg: a value that should live
across many levels of nested invokes otherwise has to be re-passed at every
intermediate ``jaz.invoke(...)`` call, and a single forgotten layer silently
drops it from the rest of the subtree. ``jaz.scope`` makes the value ambient and
makes re-passing it an *error* (passing a scoped name again as an invoke kwarg
raises), so the binding can't silently disagree with an explicit kwarg.

Storage — **a single dict-valued ``ContextVar``**:

- One mapping (rather than one ContextVar per name) keeps propagation uniform:
  each ``Task`` / ``copy_context()`` boundary copies exactly one dict, and
  "what's currently in scope" is a single read.
- The ContextVar is deliberately only a *same-context bridge*: it carries scope
  from a ``with jaz.scope(...)`` block to the ``jaz.invoke()`` call lexically
  enclosed by it (always the same context, so it always works). It is **not**
  relied on to cross thread/task boundaries. The cross-boundary hop into a nested
  invoke is done by snapshotting the scope at the invoke boundary and closing the
  agent-facing ``invoke`` primitive over it as plain data — exactly how ``config``
  propagates (see ``_invoke._build_invoke_setup`` / ``_subinvoke.get_subinvoke``). A
  bare ContextVar read inside a ``ThreadPoolExecutor`` worker would see an empty
  context and silently drop scope; binding-as-data does not.

Concurrency / mutation semantics:

- The snapshot copies the *mapping*; the values themselves are shared **by
  reference**. Under sequential or asyncio execution that is benign (no two
  invokes mutate concurrently). Under a ``ThreadPoolExecutor`` the scope is
  visible *and* truly parallel, so two workers mutating the same scoped
  ``list``/``dict`` is a data race the caller must guard — bind immutable
  values, or rebind a fresh instance inside each worker via a nested
  ``jaz.scope(...)`` block (which is properly isolated per Task/context).
- **A scope does not propagate across a process boundary, and what a child sees
  depends on the start method.** Under ``spawn`` the child is a fresh
  interpreter, so an invoke there runs with no scope at all; bind one inside the
  worker, which works and is the supported way to give a worker ambient values.
  Under ``fork`` the child inherits the binding as copied memory and invokes
  there do run under it — but that is not a guarantee to rely on, because an
  object inherited across a fork may be unusable in the child: a lock held at
  fork time can never be acquired there, and a buffered file handle replays the
  parent's unflushed bytes. Nothing serializes a scope and nothing checks for
  this, so rebind inside the worker rather than inheriting.

Composition with rendering: a scoped value renders in every invoke's prompt header exactly
as an ordinary input does, with no extra wiring — the rendering path is shared.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

# Process boundaries: there is deliberately NO runtime check that an invoke's scope was bound in
# this process. Nothing pickles a scope, so a "MainProcess only" guard in
# `_invoke._build_invoke_setup` would have nothing to protect: under spawn the child's
# `_scope_var` is unset (and a guard inside `if scoped:` could never see that case), under fork
# the binding is inherited as copied memory and works, and what such a guard reliably rejects is
# the supported pattern — a worker that sets its own scope and invokes.
#
# The one cost, documented in the module docstring: a forked child invokes under an inherited
# scope. Measured, the inherited *objects* are the hazard, not the propagation — a
# `threading.Lock` held at fork time fails a 2s-timeout `acquire()` in the child (an untimed one
# deadlocks), and an inherited buffered file handle rewrote the parent's unflushed bytes into
# the file.
#
# Considered and rejected: stamp `os.getpid()` at bind time and refuse (or warn) at invoke when
# it differs. It separates the two cases exactly — a worker's own binding passes, an inherited
# one does not — but it repeats the shape of that guard's mistake. Fork-unsafe inherited state is a
# property of fork, not of scope: the same lock in a module global is just as broken and nothing
# guards it, so the check would fire on the common benign case (a forked child under a scope of
# plain data) while implying an assurance it cannot give for any other channel. Documented
# instead; revisit if a scope-specific failure is ever observed in the wild.

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

        from jaz import invoke, scope

        with scope(fs=fs_tools, web=web_client):
            invoke(task="analyze logs", target_date="2026-05-30")
            # fs and web are bound in the REPL of this invoke and all recursive
            # sub-invokes. On the other hand, task and target_date are local to this
            # top-level invoke and do not propagate to sub-invokes.

    ``jaz.scope`` is user/host-facing. It is not bound in the agent's REPL (which sees only
    the ``invoke`` tool); a host that wants an agent to open scopes from its own REPL passes
    ``scope`` in as an ordinary input.

    Semantics:

    - **Kwargs-only.** Names must be valid Python identifiers (they land as REPL
      globals); splat a dynamic mapping with ``scope(**d)``.
    - **Inner shadows outer on the same name.** Nesting
      ``with scope(fs=A): with scope(fs=B): ...`` rebinds ``fs`` to ``B``
      inside the inner block and restores ``A`` on exit (standard ``with``
      semantics).
    - **An explicit kwarg colliding with a scoped name raises.** Passing a scoped
      name again as an invoke kwarg raises a conflict ``ValueError`` at invoke
      time. To give one invoke a different value for a scoped name, nest another
      ``scope`` block around it rather than re-passing the kwarg.
    - **No partial unbinding.** There is no way to remove a name for a block; an
      inner scope can only shadow it with a new value.
    - **Process boundaries.** A scope is a ContextVar binding, so whether a child process
      sees one follows the start method (https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods):
      ``fork`` inherits it — the names survive, the objects in them may not — while
      ``spawn`` and ``forkserver`` start clean, so an invoke there runs unscoped. Nothing
      serializes a scope across; bind one inside the worker.
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


def get_scope() -> dict[str, object]:
    """Return a copy of the values currently in scope (a runtime escape hatch).

    Available to user code for introspection and testing. (Not bound in the agent's REPL,
    which sees only the ``invoke`` tool.)
    Returns a defensive copy so callers cannot mutate
    the live binding; mutating a *value* inside the returned dict still mutates
    the shared object (values are shared by reference — see the module docstring).
    """
    return dict(_scope_var.get() or {})
