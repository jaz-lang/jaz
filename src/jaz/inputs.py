"""Input payload resolution — the ``__jaz_get__`` protocol.

An input value may be a *wrapper* that carries metadata while the agent should bind its
*payload*. The canonical example is :class:`jaz.library.Library`: ``jaz.invoke(env=lib)`` binds
the library's root *module* as ``env`` (so the agent calls ``env.bash(...)``), while the
``Library`` itself is bookkeeping and also renders a tool catalog via
``__jaz_description__``. The ``__jaz_get__`` protocol — modeled on Python's descriptor
``__get__`` (attribute access yields ``descriptor.__get__()``, never the descriptor) —
performs that payload substitution at bind time: an input whose value defines
``__jaz_get__`` binds ``value.__jaz_get__()`` (never the wrapper); a plain value binds
itself. It is the *only* binding primitive — every input binds exactly one name (its
kwarg), so there is no multi-name binding and no collision rule beyond plain Python's.

This resolution is a **core, REPL-agnostic input concern**: the protocol is defined on
framework objects (e.g. ``Library``), not by any one REPL, and every REPL that binds
inputs needs the same substitution. So the agent loop resolves inputs *before* handing
them to ``REPL.initialize`` — each REPL then receives already-bound payloads, and the
resolved set is the single canonical input set used for namespace binding,
``__inputs__``, and ``__history__``. The prompt builder keeps the *raw* wrappers so
value-attached descriptions (``__jaz_description__`` / ``jaz.Display``) still render;
resolution feeds binding, not display.

Hook-injected inputs added *after* initialization via ``REPL.add_inputs`` route through
this same ``resolve_inputs`` primitive, so every binding path applies ``__jaz_get__``
identically. They augment the namespace but do not alter the canonical ``__inputs__``
snapshot, which stays fixed at initialization.
"""

from collections.abc import Mapping


def resolve_inputs(inputs: Mapping[str, object]) -> dict[str, object]:
    """Resolve each input to the payload to bind, per the ``__jaz_get__`` protocol.

    Returns a new dict mapping each input name to ``value.__jaz_get__()`` when the value
    defines ``__jaz_get__``, else to the value itself. Pure and side-effect-free beyond
    invoking each getter once.
    """
    resolved: dict[str, object] = {}
    for name, value in inputs.items():
        # A *class* is never its own wrapper: skip `__jaz_get__` when `value` is a type.
        # `getattr(SomeClass, "__jaz_get__")` returns the plain (unbound) function, not None,
        # so calling it would invoke it with zero args and raise `TypeError: missing 'self'`.
        # This matters because framework helper *classes* are now delivered as inputs
        # (`jaz.invoke(Library=jaz.Library, ...)`, #946 model) — and `Library` is exactly the
        # symbol that defines `__jaz_get__`, so binding it as an input used to crash here.
        getter = (
            None if isinstance(value, type) else getattr(value, "__jaz_get__", None)
        )
        resolved[name] = getter() if getter is not None else value
    return resolved
