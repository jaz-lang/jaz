"""The ``jaz.Catalog`` render mode: a tool catalog for any module/namespace.

A *catalog* is the recursively-bulleted listing of every public callable (with its
signature and docstring summary) reachable under a namespace value — the rendering
that ``Library`` used to own exclusively for the ``<tool_libraries>`` prompt block.
This module lifts it into a **general render mode** usable on any module,
``SimpleNamespace``, or plain object via ``jaz.describe``/``jaz.Display``, so a tool
namespace no longer has to be wrapped in a ``Library`` to get a catalog (see
``design/design_features/library_as_input.md``).

Why a *sentinel* (``Catalog``) and not the magic string ``"catalog"``: in the
``describe``/``Display`` slots a ``str`` already means "literal text to render in the
prompt." A render *mode* selects *how* to render (walk the namespace, emit the
catalog) rather than supplying literal text — exactly the distinction between an
f-string conversion (``!r``) and a literal. A unique sentinel object can never
collide with a user's literal description string; the string ``"catalog"`` would.

**Bound name is the source of truth.** ``render_catalog`` roots the dotted paths at
``bound_name`` — the kwarg name the value is passed under at the ``jaz.invoke`` call
site — *not* at any internal module name the value happens to carry. So
``jaz.invoke(tools=lib)`` renders ``tools.bash(...)`` even if ``lib``'s root module is
internally named ``env``. This decouples the binding name (free, chosen at the call
site) from a value's internal name, which is what makes a ``Library`` a genuinely
normal input. When ``bound_name`` is ``None`` (e.g. a bare
``jaz.get_description(value)`` call at runtime, which has no kwarg context) we fall
back to the value's own ``__name__``.

Known limitation: re-rooting covers *structure* only — the auto-generated dotted
paths. Free-text prose or docstrings that hardcode an internal root name (e.g. "use
``env.view_file()``") are not rewritten. Authors should avoid hardcoding the root
name in prose and let the catalog render it.
"""

from __future__ import annotations

import inspect
from types import ModuleType, SimpleNamespace

# TODO: Have this be configurable (mirrors the old Library cap).
_MAX_PROMPT_OBJECT_ENTRIES = 100


class _CatalogMode:
    """Singleton render-mode sentinel exported as ``jaz.Catalog``.

    A distinct type (not a bare ``object()``) so it has a readable ``repr`` in
    tracebacks and is unambiguous in ``is`` checks against stored descriptions /
    display directives.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "jaz.Catalog"


# The render mode. Compared by identity (`is Catalog`) everywhere it is recognized.
Catalog = _CatalogMode()


def _is_namespace(obj: object) -> bool:
    """Whether ``obj`` should be *recursed into* as a sub-namespace.

    Modules and ``SimpleNamespace`` are walked recursively (a ``Library`` builds its
    tree out of nested ``ModuleType`` objects; user code commonly nests
    ``SimpleNamespace``). Arbitrary objects with a ``__dict__`` are *not* recursed —
    only listed as a single entry — to avoid walking deeply into unrelated objects.
    """
    return isinstance(obj, (ModuleType, SimpleNamespace))


def _summarize_doc(obj: object, *, full: bool) -> str | None:
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    doc = doc.strip()
    return doc if full else doc.splitlines()[0]


def _format_signature(obj: object) -> str:
    if not callable(obj):
        return ""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "()"


def _format_entry(dotted_path: str, obj: object, *, full_docstrings: bool) -> str:
    if _is_namespace(obj):
        doc = _summarize_doc(obj, full=full_docstrings) or "(no description available)"
        return f"- `{dotted_path}` [{type(obj).__name__}]: {doc}"
    if callable(obj):
        sig = _format_signature(obj)
        doc = _summarize_doc(obj, full=full_docstrings) or "(no description available)"
        return f"- `{dotted_path}{sig}`: {doc}"
    return f"- `{dotted_path}` [{type(obj).__name__}]"


def _walk(
    namespace: object,
    dotted_path: str,
    *,
    seen: set[int],
    remaining: list[int],
    full_docstrings: bool,
) -> list[str]:
    """Recursively render ``namespace`` and its public members as catalog lines.

    ``remaining`` is a one-element list used as a mutable counter so the global
    ``_MAX_PROMPT_OBJECT_ENTRIES`` cap is shared across the whole recursion.
    ``seen`` guards against reference cycles (a namespace re-exporting an ancestor).
    """
    if remaining[0] <= 0:
        return []

    ns_id = id(namespace)
    if ns_id in seen:
        return [
            f"- `{dotted_path}` [{type(namespace).__name__}]: (already documented above)"
        ]
    seen.add(ns_id)

    entries = [_format_entry(dotted_path, namespace, full_docstrings=full_docstrings)]
    remaining[0] -= 1

    members = vars(namespace) if hasattr(namespace, "__dict__") else {}
    for name, value in sorted(members.items()):
        if remaining[0] <= 0:
            break
        if name.startswith("_"):
            continue
        child_path = f"{dotted_path}.{name}"
        if _is_namespace(value):
            entries.extend(
                _walk(
                    value,
                    child_path,
                    seen=seen,
                    remaining=remaining,
                    full_docstrings=full_docstrings,
                )
            )
        else:
            entries.append(
                _format_entry(child_path, value, full_docstrings=full_docstrings)
            )
            remaining[0] -= 1
    return entries


def render_catalog(
    value: object, bound_name: str | None, *, full_docstrings: bool = True
) -> str:
    """Render ``value``'s tool catalog rooted at ``bound_name``.

    Produces the recursively-bulleted listing of public callables/sub-namespaces
    reachable under ``value``, with dotted paths rooted at ``bound_name`` (the kwarg
    name the agent will use). Falls back to ``value.__name__`` when ``bound_name`` is
    ``None`` (no call-site context — e.g. a bare ``get_description`` call), and to ``"value"`` if the
    object has no ``__name__`` either. Capped at ``_MAX_PROMPT_OBJECT_ENTRIES``
    objects with a truncation marker, matching the legacy ``Library`` renderer.

    ``full_docstrings`` defaults to ``True``: the catalog should show things
    **comprehensively** (each object's whole docstring) by default. Compact
    rendering (``full_docstrings=False`` — signature + docstring first line only)
    is an opt-in *optimization* for cases where the full catalog would be too large
    (many tools, or large libraries passed as inputs). Executive call: the default
    is comprehensiveness; shrinking the output is the deliberate, caller-chosen
    exception, not the reverse.

    Known gap: the auto-render *input* paths (``describe`` / ``get_description``,
    ``Display``, ``default_description``) call this with no flag, so a large library
    passed as an input always renders full — compact is not reachable there today
    (those APIs don't thread ``full_docstrings``, and the entry cap bounds object
    count, not per-docstring size). A reachable escape hatch is tracked in #726.
    """
    root = bound_name or getattr(value, "__name__", None) or "value"
    seen: set[int] = set()
    remaining = [_MAX_PROMPT_OBJECT_ENTRIES]
    entries = _walk(
        value, root, seen=seen, remaining=remaining, full_docstrings=full_docstrings
    )
    if remaining[0] <= 0:
        entries.append(f"- ... truncated after {_MAX_PROMPT_OBJECT_ENTRIES} objects")
    return "\n".join(entries)


def as_catalog(value: object) -> object:
    """Attach the ``Catalog`` render mode permanently to ``value``; returns ``value``.

    Sugar for ``describe(value, jaz.Catalog)``: from now on this value renders as
    its tool catalog in any ``jaz.invoke`` prompt header, rooted at whatever kwarg
    name it is passed under. ``jaz.Display(value, jaz.Catalog)`` does the same for a
    single call without attaching anything, but it is experimental and warns on use.

    ``describe`` is imported lazily because ``descriptions`` imports this module (for
    the ``Catalog`` sentinel and ``render_catalog``), so a top-level import here would
    cycle.
    """
    from .descriptions import describe

    return describe(value, Catalog)
