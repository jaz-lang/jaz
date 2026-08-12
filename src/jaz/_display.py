"""Per-invoke display directives (``jaz.Display``).

This is the *display* half of the input-presentation feature, deliberately kept
separate from value-attached descriptions (see :mod:`jaz.descriptions`):

- A **description** (``jaz.describe``) is a *permanent property of the object* —
  it answers "what is this thing?" and travels with the value into every nested
  ``jaz.invoke``.
- A **display directive** (``jaz.Display``) is an *ephemeral, call-site* concern —
  it answers "how should *this* invoke render this input?" It attaches nothing to
  the object, mutates nothing, works on any value (lists, dicts, ints included),
  and applies only to the one ``jaz.invoke`` call it wraps.

``jaz.Display(value, text)`` wraps a value for a single call: ``text`` may be a
``str`` (render that), ``None`` (hide the input from the prompt header), or a
``Callable[[object], str]`` (compute the text lazily). The directive plugs into the
same ``__jaz_get__`` / ``__jaz_description__`` protocol as ``jaz.Library``: the REPL
binds ``__jaz_get__()`` (the real ``value``), and the prompt builder reads its
description via the normal :func:`jaz.get_description` lookup — so there is a single
substitution path for every metadata wrapper, not a parallel one for :class:`Display`.

"""

# WHY ONE CLASS, and not the former `show_as()` factory over a private wrapper.
#
# These used to be two names: a public lowercase `show_as(value, text)` factory in front
# of a private `DisplayDirective` dataclass, on the reasoning that callers only ever
# *construct* a directive and never hold one (the agent loop resolves `__jaz_get__`
# before hooks or the REPL see it) — the stdlib `dataclasses.field()` / `pytest.param()`
# idiom. They were collapsed because that idiom buys nothing here: the factory was
# unconditional — one `return DisplayDirective(value, text)`, no branching, no
# validation, no alternate return — so the two names described a single object with no
# behavior between them.
#
# The verb also mis-sold what happens: `show_as` reads as an action, but the call
# displays nothing; it builds an inert value that `jaz.invoke` interprets later. A
# CapWords noun says "you are constructing a wrapper", which is what a caller is
# actually doing, and it gives `describe` a concrete class to name when its retaining
# fallback is the wrong trade.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ._catalog import _CatalogMode, render_catalog
from .descriptions import HIDDEN, _HiddenInput, _max_positional_args

# What a directive renders to: a string, `None` (hide), the `jaz.Catalog` render mode
# (render the value as a tool catalog), or a lazily-evaluated callable. Resolved
# against the wrapped value at prompt-build time.
type DisplayText = str | None | _CatalogMode | Callable[..., str]


@dataclass(frozen=True)
class Display:
    """Control how ``value`` renders in *this* ``jaz.invoke``'s prompt header.

    **Experimental — not public API.** ``Display`` is outside ``jaz.__all__``: reachable,
    but unsupported, and it may change or be removed once :func:`jaz.describe` is shown to
    cover its use cases. Every route to the name — ``jaz.Display``, ``jaz.display``, or
    ``from jaz.display import Display`` — emits
    :class:`~jaz.exceptions.NonPublicAPIWarning`. Where a supported alternative exists,
    prefer it: describing a small object that *contains* the value covers relabeling and
    (by nesting) hiding, without the demoted surface.

    Construct it inline as an input::

        jaz.invoke(task=prompt, context=jaz.Display(context, None))   # hide it
        jaz.invoke(task=prompt, table=jaz.Display(table, "Q1 sales")) # relabel it

    ``text`` may be:

    - a ``str`` — render this text instead of the auto-stringified value;
    - ``None`` — omit this input from the prompt header (it is still bound as a
      Python variable in the sub-agent's REPL);
    - a ``Callable[[object], str]`` — evaluated with ``value`` at render time;
    - ``jaz.Catalog`` — render the value as a tool catalog.

    Unlike :func:`jaz.describe`, this attaches nothing to ``value`` and does not
    persist: it applies only to the single invoke it is passed to. Two consequences
    follow, both worth knowing:

    - It **retains nothing**, so it is safe in a loop or a per-request path where
      ``describe`` on a bare built-in would keep the value alive for the lifetime of
      the process.
    - It is the only one of the two that can **hide** an input (``text=None``). Hiding
      is inherently per-call — "don't show this *here*" is not a property of the object
      — so ``describe`` has no equivalent.

    The wrapper itself never reaches the agent: the REPL binds ``value``, and the
    wrapper's own type does not appear in the prompt.
    """

    # Implements the framework metadata protocol, so no part of the pipeline
    # special-cases it: `__jaz_get__` returns the real value (the same payload
    # substitution `Library` uses) and `__jaz_description__` resolves `text` for the
    # prompt header, reporting the `HIDDEN` sentinel for `text is None` so the input's
    # block is dropped while the value stays bound. The agent loop resolves
    # `__jaz_get__` up front, which is why hooks, events and the REPL only ever see the
    # unwrapped payload.

    value: object
    text: DisplayText

    def __jaz_get__(self) -> object:
        """Return the real value to bind into the REPL (never the wrapper)."""
        return self.value

    def __jaz_description__(self, bound_name: str | None = None) -> str | _HiddenInput:
        """Resolve the per-call render text, or ``HIDDEN`` to drop the input block.

        Any ``None`` outcome maps to ``HIDDEN`` (hide) rather than ``None`` so the
        prompt builder can tell "suppress this input" apart from "no description,
        render the default" — the latter never applies to a directive, which always
        carries an explicit instruction. ``None`` arises both from ``text is None``
        and from a callable ``text`` that returns ``None`` for this value; both mean
        hide.
        """
        resolved = resolve_display_text(self.text, self.value, bound_name)
        return HIDDEN if resolved is None else resolved


def resolve_display_text(
    text: DisplayText, value: object, bound_name: str | None = None
) -> str | None:
    """Resolve a display ``text`` against its ``value`` to ``str`` or ``None``.

    ``None`` (hide) passes through; the ``jaz.Catalog`` render mode renders ``value``
    as a tool catalog rooted at ``bound_name``; a callable is invoked with ``value``
    (and ``bound_name`` appended iff it accepts it, mirroring :func:`describe` callables);
    a string is returned as-is. ``bound_name`` is the input's kwarg name, threaded in
    by the prompt builder.
    """
    if text is None:
        return None
    if isinstance(text, _CatalogMode):  # the `jaz.Catalog` render mode
        return render_catalog(value, bound_name)
    if callable(text):
        if _max_positional_args(text) >= 2:
            return text(value, bound_name)
        return text(value)
    return text
