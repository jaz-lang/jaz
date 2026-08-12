"""t-string (PEP 750) support for ``jaz.invoke``.

Lets a caller write::

    jaz.invoke(task=t"Analyze the {data}")

as shorthand for the equivalent explicit form (the t-string is passed as the
``task`` input value)::

    jaz.invoke(task="Analyze the `data`", data=data)

A literal *f-string* cannot do this: it is evaluated eagerly, so by the time
``invoke`` runs only ``str(data)`` survives — both the variable name and the live
object are gone. A *t-string* instead produces a :class:`string.templatelib.Template`
that preserves, per interpolation, the source ``expression``, the live ``value``, and
any ``!conversion`` / ``:format_spec``. :func:`normalize_task` walks that template and
splits it back into the ``(prompt, inputs)`` pair the rest of ``invoke`` already expects,
so t-strings are pure sugar over the existing input-binding machinery.

**Version requirement.** The ``t"..."`` literal syntax is PEP 750, new in Python 3.14, so
the t-string *call shape* is a 3.14+ feature. The package floor stays at >=3.12: this
module imports and ``normalize_inputs`` runs on every invoke regardless (delegating to
``normalize_task`` only for ``Template``-valued inputs), but on <3.14 the
``string.templatelib`` import is unavailable, so :data:`Template` falls back to a dummy and
``normalize_inputs`` simply passes plain input values through untouched. The documented
explicit equivalent (``jaz.invoke(task="Analyze the `data`", data=data)``) works on every
supported interpreter; only the literal sugar requires 3.14. We chose this gate over
bumping the floor so 3.12/3.13 users keep full functionality minus the syntax.

Executive design decisions (settled with the user; see also ``Display`` in
:mod:`jaz._display` and ``jaz.describe`` in :mod:`jaz.descriptions`):

- **Every interpolation binds its object** as a REPL input; the ``{expr}`` position in
  the sentence becomes a `` `name` `` backtick reference so the prose reads naturally and
  points the agent at the bound variable. This mirrors the explicit input form.
- **Default prompt rendering of the bound object is its ``__jaz_description__``** — i.e.
  the existing default input rendering in ``jaz.prompts``. We deliberately do *not*
  inline ``str(value)`` at the ``{expr}`` site.
- **A ``!conversion`` / ``:format_spec`` overrides that rendering**, not the binding. The
  object is still bound; only its prompt-header rendering becomes the f-string-style
  ``format(convert(value), spec)`` text. This is wired through ``Display``, which returns
  a directive exposing the real value via ``__jaz_get__`` and the override text via
  ``__jaz_description__`` — the same metadata protocol every input rendering already uses,
  so no ``agent``/``prompts`` changes are needed.
  Rationale: in a t-string a ``!``/``:`` carries no eager effect of its own, so the only
  coherent intent is "render the bound value this way in the prompt".

  **This override path is EXPERIMENTAL and outside** ``jaz.__all__``. ``Display`` was
  demoted to non-public, and a render override is its last user-facing entry point, so
  using one emits :class:`~jaz.exceptions.NonPublicAPIWarning` at the ``invoke`` call.
  Plain ``{value}`` interpolation is unaffected — it binds the value and lets the normal
  description chain render it — and is the supported form.
- **Binding name** = the interpolation's ``expression`` when it is a bare identifier
  (e.g. ``data``); otherwise a synthesized, collision-free name (e.g. ``_jaz_arg0`` for
  ``{df.head()}``), since a complex expression is not a usable variable name.

Limitations (intentional, v1): a repeated bare ``{data}`` de-dupes to one binding with
multiple references, but the *same* name appearing with conflicting renderings (e.g.
``t"{data} vs {data!r}"``) raises rather than silently picking one; and a bare-identifier
interpolation whose name collides with an explicitly-passed kwarg raises, since that is
almost always a mistake.

- **``_jaz_arg*`` is a reserved prefix** for synthesized bindings (see ``_binding_name``).
  Interpolating a variable literally named ``_jaz_arg0`` alongside a non-identifier
  interpolation in the same template can collide with a minted name and raise the
  conflicting-values error even though the caller wrote each name only once. Vanishingly
  rare; treat ``_jaz_arg*`` as off-limits for hand-written interpolation names.
- **Nested format specs are untested and unsupported in v1.** ``_render_interpolation``
  hands ``interp.format_spec`` straight to ``format()``, so a spec that itself contains an
  interpolation (e.g. ``t"rate is {rate:.{prec}f}"``) has undefined behavior here — it has
  not been verified on 3.14, and may raise or misrender if the runtime leaves an
  unresolved ``{prec}`` in the spec string. Use a pre-formatted value or a plain
  ``:format_spec`` without nesting.

For downstream conflict checks, a t-string interpolation counts as an *explicit*
binding: its name is merged into ``inputs`` here, so it participates in the same
collision/scope checks an explicit kwarg would (e.g. ``jaz.scope``'s scoped-vs-input
conflict). ``jaz.invoke(task=t"summarize {data}")`` while ``data`` is ambiently scoped
therefore raises the scope-conflict error — correct in substance (the name is bound
twice), even though the message speaks of kwargs rather than the t-string the caller
actually wrote.
"""

from __future__ import annotations

import keyword
import os.path
import warnings
from typing import TYPE_CHECKING

# string.templatelib (PEP 750) only exists on Python 3.14+, but the package floor stays at
# >=3.12 and pyright is pinned at pythonVersion 3.12 (so it still flags real 3.12
# incompatibilities elsewhere). We deliberately keep the type-check view and the runtime
# binding on *separate* paths:
#   - For type-checkers, define local placeholder types mirroring only the members this
#     module touches — rather than importing the real ``string.templatelib`` names. Under
#     the 3.12 pin that import is *inconsistent across CI runners*: on a 3.12/3.13 runner the
#     module is physically absent so pyright degrades the names to Unknown (harmless), but on
#     the 3.14 runner the real module is present on the path and pyright — not using 3.14
#     typeshed stubs, since it targets 3.12 — infers its C-backed ``Template`` /
#     ``Interpolation`` as *variables*, which then trip ``reportInvalidTypeForm`` at every
#     ``str | Template`` annotation (here and in ``invoke.py`` / ``library/jaz.py``). Local
#     placeholders give pyright one identical, valid-as-a-type view on every interpreter
#     without forcing a 3.14-only pythonVersion for the whole project.
#   - At runtime, try the real import and fall back to dummies on <3.14 so the module still
#     imports and ``isinstance(task, Template)`` works (always False). The t-string path is
#     then unreachable — no ``t"..."`` literal can be constructed on <3.14 — so the dummies
#     are never actually used. This is what keeps the literal sugar a 3.14-only feature
#     without raising the floor (see the "Version requirement" note in the module docstring).
if TYPE_CHECKING:
    from collections.abc import Iterator

    class Interpolation:
        """Type-check placeholder for ``string.templatelib.Interpolation`` (see above)."""

        value: object
        expression: str
        conversion: str | None
        format_spec: str

    class Template:
        """Type-check placeholder for ``string.templatelib.Template`` (see above)."""

        def __iter__(self) -> Iterator[str | Interpolation]: ...
else:
    try:
        from string.templatelib import Interpolation, Template
    except ImportError:
        Interpolation = Template = type("_TStringUnavailable", (), {})

from ._display import Display
from .exceptions import NonPublicAPIWarning

# f-string ``!conversion`` letters -> the stringify they apply *before* formatting.
_CONVERSIONS = {"a": ascii, "r": repr, "s": str}


def _render_interpolation(interp: Interpolation) -> str:
    """Reproduce f-string rendering for an interpolation carrying a conversion/spec.

    Applies the ``!conversion`` (if any) then the ``:format_spec`` — the same two steps a
    real f-string performs eagerly. Used only when the interpolation actually carries one
    of them; a bare ``{value}`` keeps the default description-based rendering instead.
    """
    value = interp.value
    if interp.conversion:
        value = _CONVERSIONS[interp.conversion](value)
    # Passes format_spec through verbatim: a *nested* spec (t"{rate:.{prec}f}") is untested
    # and unsupported in v1 — if the runtime leaves an unresolved {prec} here, format() will
    # raise or misrender. See the module docstring's limitations.
    return format(value, interp.format_spec)


def _binding_name(expression: str, used: set[str]) -> str:
    """Pick the REPL variable name to bind an interpolation's value under.

    A bare identifier expression (``{data}``) is used verbatim. Anything else
    (``{df.head()}``, ``{a + b}``) is not a usable name, so we synthesize a stable
    ``_jaz_arg{n}`` that doesn't collide with ``used`` (existing inputs + names already
    minted for this template).

    TODO(#659): let callers name a complex interpolation explicitly via the format-spec
    slot (``t"{list(range(100)):numbers}"`` binds under ``numbers``), instead of only
    auto-synthesizing ``_jaz_arg{n}``. Deferred to a separate PR — it repurposes the
    ``:format_spec`` rendering-override channel, so it needs its own design.
    """
    candidate = expression.strip()
    if candidate.isidentifier() and not keyword.iskeyword(candidate):
        return candidate
    i = 0
    while True:
        name = f"_jaz_arg{i}"
        if name not in used:
            return name
        i += 1


def normalize_task(
    task: str | Template, inputs: dict[str, object]
) -> tuple[str, dict[str, object]]:
    """Split a t-string ``task`` into ``(prompt_string, merged_inputs)``.

    A plain ``str`` task is returned unchanged (with ``inputs`` untouched). A
    :class:`~string.templatelib.Template` is walked in order: literal chunks are appended
    verbatim, and each interpolation is turned into a `` `name` `` reference plus an input
    binding (see the module docstring for the binding/rendering rules).

    Raises ``ValueError`` on a name collision (template-derived name vs. an explicit
    kwarg, or the same name bound to conflicting values/renderings within the template).
    """
    if not isinstance(task, Template):
        return task, inputs

    explicit_names = set(inputs)
    merged: dict[str, object] = dict(inputs)
    # name -> (value identity, rendered-override-or-None) for intra-template de-dupe.
    derived: dict[str, tuple[int, str | None]] = {}
    parts: list[str] = []

    for item in task:
        if isinstance(item, str):
            parts.append(item)
            continue

        interp: Interpolation = item
        has_override = bool(interp.conversion) or bool(interp.format_spec)
        if has_override:
            # The render-override half of t-strings is EXPERIMENTAL and outside `jaz.__all__`
            # — it is the one remaining user-facing entry point to `jaz.Display`, which was
            # itself demoted. Warned here rather than at import because a format spec is not
            # an importable name: the syntax *is* the API, so the use site is the only place
            # a caller can be told. Plain `t"...{value}"` interpolation (no `!r`/`:spec`) is
            # unaffected and does not warn — only the override path is provisional.
            # Reproduce the source syntax exactly — `!r`, `:.0%`, or `!r:>10` — so the
            # message names what the caller actually typed. Building it from both parts
            # (rather than always prefixing `!`) is why a spec-only override reads `:.0%`
            # and not `!:.0%`.
            override_syntax = (f"!{interp.conversion}" if interp.conversion else "") + (
                f":{interp.format_spec}" if interp.format_spec else ""
            )
            warnings.warn(
                f"t-string render overrides (`{override_syntax}` on "
                f"{{{interp.expression}}}) are experimental and not part of the official "
                "public API. They are reachable but may change or be removed in the near "
                "future. Interpolate without a conversion/format-spec to bind the value "
                "and let its description render.",
                NonPublicAPIWarning,
                # Attribute the warning to the caller's `t"..."` line by skipping every
                # frame inside the package, rather than counting frames with `stacklevel`.
                # No fixed count works: the depth from here varies by entry path (invoke /
                # ainvoke / a direct `normalize_task` call), so any single value is right
                # for one and lands inside `jaz` for the others — which matters more than
                # usual because a t-string has no importable name, leaving the source line
                # as the only thing that locates the override in a codebase using several.
                # `skip_file_prefixes` is 3.12+, and the package floor is already 3.12.
                skip_file_prefixes=(os.path.dirname(__file__),),
            )
        override_text = _render_interpolation(interp) if has_override else None
        name = _binding_name(interp.expression, set(merged) | set(derived))

        if name in derived:
            # Repeated reference to the same template name. Allow only if it is the
            # same value rendered the same way; otherwise the two would fight over one
            # binding/header entry.
            prev_id, prev_override = derived[name]
            if prev_id != id(interp.value) or prev_override != override_text:
                raise ValueError(
                    f"t-string binds name {name!r} more than once with conflicting "
                    f"values or renderings; assign to distinct names instead."
                )
        else:
            if name in explicit_names:
                raise ValueError(
                    f"t-string interpolation {{{interp.expression}}} would bind input "
                    f"{name!r}, which is also passed explicitly to invoke(); rename one."
                )
            merged[name] = (
                Display(interp.value, override_text)
                if override_text is not None
                else interp.value
            )
            derived[name] = (id(interp.value), override_text)

        parts.append(f"`{name}`")

    return "".join(parts), merged


def normalize_inputs(inputs: dict[str, object]) -> dict[str, object]:
    """Lower any t-string **input values** into text + sibling bindings (#538).

    After ``task`` was demoted to an ordinary input, a t-string is no longer a positional
    prompt — it is passed as an *input value* (e.g. ``invoke(task=t"Summarize {data}")``).
    For each input whose value is a :class:`~string.templatelib.Template`, this lowers it
    exactly as the former positional ``task`` was lowered by :func:`normalize_task`: literal
    chunks are kept verbatim, each interpolation becomes a `` `name` `` reference, and the
    interpolated object is bound as a *sibling* input (its own REPL variable + prompt block).
    The t-string input itself then binds to the lowered *text*. A plain (non-Template) value
    passes through untouched, so the common case (all plain kwargs) is a cheap identity walk.

    Semantics deliberately match the old positional behavior: interpolations *mint* sibling
    inputs, and a template-derived name that collides with another explicit input raises (via
    :func:`normalize_task`). So the ergonomic form is to let the t-string supply its own
    interpolated inputs — ``invoke(task=t"Summarize {data}")`` binds ``data`` from the
    interpolation — rather than also passing ``data=`` separately. Referencing a *separately*
    passed input by name inside a t-string is therefore not supported in v1 (it would collide);
    that richer "reference a sibling input" mode is left for a follow-up.

    Inputs are walked in insertion order, threading the accumulating namespace so each
    t-string's minted siblings are visible to the collision checks of later t-strings.

    The collision check is *order-independent*: whichever direction the duplicate appears —
    a later t-string minting a name already bound (raised by :func:`normalize_task`), or a
    plain (non-Template) input whose name was already minted as a sibling by an *earlier*
    t-string (raised by the ``else`` branch below) — is rejected. This mirrors Python's
    ``SyntaxError`` on duplicate keyword arguments: ``invoke(key=t"... {kw} ...", kw=...)``
    binds ``kw`` twice and is an error regardless of argument order. The reliable form is to
    let a t-string supply its own interpolated siblings rather than also passing a colliding
    plain input.

    This lowering is purely *structural* and runs at the ``invoke`` boundary
    (:func:`jaz.invoke`), strictly *before* ``__jaz_get__`` payload resolution
    (:func:`~jaz.inputs.resolve_inputs`, applied at REPL-bind time). A minted sibling binds
    the raw interpolated object (or a ``Display`` directive when a ``!r``/format-spec override
    is present), so if that object carries ``__jaz_get__`` its payload is substituted
    downstream exactly like a top-level input — ``normalize_inputs`` never resolves
    ``__jaz_get__`` itself.

    Two known v1 limitations at that boundary (tracked in #857), both from ``__jaz_get__``
    resolution being single-level:

    - **Override + a ``__jaz_get__`` value drops the payload.** ``t"{lib!r}"`` (where ``lib``
      carries ``__jaz_get__``, e.g. a :class:`~jaz.Library`) minus the override binds the
      payload, but *with* the override the sibling becomes a ``Display`` directive wrapping
      ``lib``; :func:`resolve_inputs` unwinds only that outer layer, so the REPL binds the
      ``lib`` *wrapper*, not ``lib.__jaz_get__()`` (and the header renders ``repr(lib)``).
    - **A ``__jaz_get__`` carrier holding a Template is not lowered.** If an input value is not
      itself a :class:`~string.templatelib.Template` but yields one via ``__jaz_get__`` (e.g.
      ``Display(t"...", ...)``), the ``isinstance`` check below skips lowering and the REPL
      binds a raw, un-lowered ``Template``. (A ``Template`` subclass that *directly* defines
      ``__jaz_get__`` is impossible — the C type is not subclassable.)
    """
    result: dict[str, object] = {}
    for name, value in inputs.items():
        if isinstance(value, Template):
            # `normalize_task` folds the interpolation bindings into a copy of `result`
            # and returns the lowered `text`; the t-string input then binds to that text.
            text, result = normalize_task(value, result)
            result[name] = text
        else:
            if name in result:
                # `name` can only already be in `result` if an earlier t-string minted a
                # sibling of the same name (outer input keys are unique). Treat this like a
                # duplicate kwarg and raise, so the collision check is symmetric with the
                # normalize_task direction rather than silently overwriting the sibling.
                raise ValueError(
                    f"input {name!r} is passed explicitly to invoke(), but a t-string "
                    f"interpolation earlier in the call already minted a sibling input of "
                    f"that name; rename one."
                )
            result[name] = value
    return result


__all__ = ["normalize_task", "normalize_inputs"]
