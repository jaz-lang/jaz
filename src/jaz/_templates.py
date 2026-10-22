"""t-string (PEP 750) support for ``jaz.invoke``.

Lets a caller write::

    invoke(task=t"Analyze the {data}")

as shorthand for the equivalent explicit form (the t-string is passed as the
``task`` input value)::

    invoke(task="Analyze the `data`", data=data)

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

Executive design decisions (settled with the user):

- **Every interpolation binds its object** as an invoke input; the ``{expr}`` position in
  the sentence becomes a `` `name` `` backtick reference so the prose reads naturally and
  points the agent at the bound variable. This mirrors the explicit input form.
- **The bound object renders as any other input does**, through
  ``jaz._value_rendering``. We deliberately do *not* inline ``str(value)`` at the ``{expr}``
  site.
- **The ``:name`` slot names the binding**::

      invoke(t"add the numbers {list(range(100)):numbers}")   # binds under `numbers`

  A complex interpolation is not a usable variable name, so it is REQUIRED there: without
  it there is no name to bind under, and the prose would have to refer to the agent's
  variable by one nobody chose. The name must be an identifier and must not be a keyword.
- **Binding name**, in full: the ``:name`` slot when present; else the interpolation's
  ``expression`` when it is a bare identifier (e.g. ``data``). There is no third case —
  anything else raises ``ValueError`` asking for a name.
- **``!conversion`` does not render, and neither does a ``:spec``.** Both raise
  ``ValueError`` naming the interpolation, rather than being ignored, so a caller who wrote
  ``{rate:.0%}`` is told the slot changed meaning instead of silently getting a variable
  named ``.0%``.

Limitations (intentional, v1): a repeated bare ``{data}`` de-dupes to one binding with
multiple references, but the same name bound to two *different* values (e.g.
``t"{a:rows} vs {b:rows}"``) raises rather than silently picking one; and an
interpolation whose name collides with an explicitly-passed kwarg raises, since that is
almost always a mistake. Binding one value under two names (``t"{data} vs {data:rows}"``)
is allowed — they are two references to one object, which is what the caller asked for.

- **A nested spec names the binding from a runtime value.** ``t"{value:{chosen}}"`` arrives
  here already resolved — the runtime substitutes the inner interpolation, so this module
  receives a plain string and cannot tell it from a literal one. ``chosen = "picked"`` binds
  under ``picked``. It is validated like any other name, so a nested spec that resolves to
  something unusable raises; but the error quotes the *resolved* text, which the caller never
  typed (``t"{rate:.{prec}f}"`` with ``prec = 5`` reports ``.5f``). Prefer writing the name
  literally.

For downstream conflict checks, a t-string interpolation counts as an *explicit*
binding: its name is merged into ``inputs`` here, so it participates in the same
collision/scope checks an explicit kwarg would (e.g. ``jaz.scope``'s scoped-vs-input
conflict). ``jaz.invoke(task=t"summarize {data}")`` while ``data`` is ambiently scoped
therefore raises the scope-conflict error — correct in substance (the name is bound
twice), even though the message speaks of kwargs rather than the t-string the caller
actually wrote.
"""

# Why the `:` slot names a binding rather than overriding the rendering (#659):
#
# `!conversion`/`:format_spec` used to re-render the bound value as
# `format(convert(value), spec)` in place of its rendering, wired through a `Display`
# directive. That design was never substantiated. In a t-string a `!`/`:` carries no eager
# effect of its own, so "render it this way" was INFERRED as the only coherent intent rather
# than being a use case anyone asked for, and nothing in the tree used it outside the tests
# written for it. #659 recorded the competing reading — the input renderer already covers
# rendering, so the slot is better spent on names — and that is the call taken here.
#
# Raising on both stale forms rather than ignoring them: `{rate:.0%}` would otherwise bind a
# variable named `.0%` that no agent can reference, and a dropped `!r` would change what the
# agent sees with no signal. The identifier check reports the offending text so the error
# doubles as the migration message.

from __future__ import annotations

import keyword
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


def _explicit_name(interp: Interpolation) -> str:
    """Validate and return the binding name a caller wrote in the ``:name`` slot.

    Raises ``ValueError`` with the offending text when it is not usable as a REPL
    variable — which is also what a stale f-string-style render override now hits, since
    ``:.0%`` is not an identifier.
    """
    name = interp.format_spec
    what = f"t-string interpolation {{{interp.expression}:{name}}}"
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f"{what}: the `:` slot names the REPL variable the value binds under, so "
            f"{name!r} must be a valid Python identifier. If you meant an f-string "
            f"format spec, `:` no longer renders — format the value before "
            f"interpolating it, or pass it as an explicit input."
        )
    return name


def _binding_name(expression: str) -> str:
    """The REPL variable name for an interpolation the caller did not name.

    A bare identifier expression (``{data}``) is used verbatim. Anything else
    (``{df.head()}``, ``{a + b}``) is not a usable name and raises, naming the
    interpolation and asking for a ``:name``.

    Only reached when the caller did not name the binding themselves — ``:name`` is handled
    by :func:`_explicit_name` before this is consulted.
    """
    # This used to MINT a name instead — a collision-free `_jaz_arg{n}`. That name is not
    # internal: it goes into the prompt as `` `_jaz_arg0` `` and labels the input block, so
    # the agent reasons about a variable nobody chose. Which is the very thing the `:` slot
    # was repurposed to fix, left reachable for exactly the case that needed it most: a
    # caller who wrote no name got a worse prompt and no signal. Asking is one word of work
    # and the error says which interpolation and what to write.
    candidate = expression.strip()
    if candidate.isidentifier() and not keyword.iskeyword(candidate):
        return candidate
    raise ValueError(
        f"t-string interpolation {{{expression}}} has no usable binding name: "
        f"{candidate!r} is not a Python identifier, so name it with the `:` slot — "
        f"e.g. {{{expression}:rows}} binds the value under `rows`."
    )


def normalize_task(
    task: str | Template, inputs: dict[str, object]
) -> tuple[str, dict[str, object]]:
    """Split a t-string ``task`` into ``(prompt_string, merged_inputs)``.

    A plain ``str`` task is returned unchanged (with ``inputs`` untouched). A
    :class:`~string.templatelib.Template` is walked in order: literal chunks are appended
    verbatim, and each interpolation is turned into a `` `name` `` reference plus an input
    binding (see the module docstring for the naming rules).

    Raises ``ValueError`` on a name collision (template-derived name vs. an explicit
    kwarg, or one name bound to two different values within the template), on an unusable
    ``:name``, and on a ``!conversion`` (which no longer has a meaning).
    """
    if not isinstance(task, Template):
        return task, inputs

    explicit_names = set(inputs)
    merged: dict[str, object] = dict(inputs)
    # name -> value identity, for intra-template de-dupe.
    derived: dict[str, int] = {}
    parts: list[str] = []

    for item in task:
        if isinstance(item, str):
            parts.append(item)
            continue

        interp: Interpolation = item
        if interp.conversion:
            # `!r`/`!s`/`!a` used to mean "render the bound value this way", the same
            # rejected premise as `:spec`. Raising rather than ignoring: a caller who typed
            # a conversion expects it to do something, and silently dropping it would change
            # what the agent sees with no signal.
            # Reproduce BOTH parts the caller typed. Reporting only `{x!r}` for `{x!r:who}`
            # names an interpolation that is not in their source, and the advice to "use
            # `:name`" then reads as a correction for something they already did.
            spec = f":{interp.format_spec}" if interp.format_spec else ""
            advice = (
                "Drop it to keep the binding name you chose."
                if interp.format_spec
                else "Interpolate the value plainly to bind it, or use `:name` to choose "
                "its binding name."
            )
            raise ValueError(
                f"t-string interpolation "
                f"{{{interp.expression}!{interp.conversion}{spec}}}: "
                f"`!{interp.conversion}` no longer overrides how the value renders. "
                f"{advice} To format the value, do it before interpolating."
            )
        # `merged` already contains every name `derived` holds — `derived` is a subset kept
        # for the value-identity check — so the union that used to be passed here was
        # redundant.
        name = (
            _explicit_name(interp)
            if interp.format_spec
            else _binding_name(interp.expression)
        )

        if name in derived:
            # Repeated reference to the same template name. Allow only if it is the
            # same value rendered the same way; otherwise the two would fight over one
            # binding/header entry.
            if derived[name] != id(interp.value):
                raise ValueError(
                    f"t-string binds name {name!r} more than once with conflicting "
                    f"values; assign to distinct names instead."
                )
        else:
            if name in explicit_names:
                raise ValueError(
                    f"t-string interpolation {{{interp.expression}}} would bind input "
                    f"{name!r}, which is also passed explicitly to invoke(); rename one."
                )
            merged[name] = interp.value
            derived[name] = id(interp.value)

        parts.append(f"`{name}`")

    return "".join(parts), merged


def normalize_inputs(inputs: dict[str, object]) -> dict[str, object]:
    """Lower any t-string **input values** into text + sibling bindings.

    A t-string is passed as an *input value* (e.g. ``invoke(task=t"Summarize {data}")``).
    For each input whose value is a :class:`~string.templatelib.Template`, this lowers it
    exactly as the former positional ``task`` was lowered by :func:`normalize_task`: literal
    chunks are kept verbatim, each interpolation becomes a `` `name` `` reference, and the
    interpolated object is bound as a *sibling* input (its own REPL variable + prompt block).
    The t-string input itself then binds to the lowered *text*. A plain (non-Template) value
    passes through untouched, so the common case (all plain kwargs) is a cheap identity walk.

    Interpolations *mint* sibling inputs, and a template-derived name that collides with
    another explicit input raises (via
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
    (:func:`jaz.invoke`). A minted sibling binds the raw interpolated object, exactly like a
    top-level input.
    """
    # Exists because #538 demoted ``task`` to an ordinary input: a t-string stopped being a
    # positional prompt and became an input value, so the lowering the positional ``task`` used
    # to get had to apply to any input.
    #
    # #857 tracked two limitations here, both from `__jaz_get__` resolution being single-level:
    # a wrapper holding a Template was not lowered, and an override wrapping a carrier bound the
    # wrapper instead of its payload. Both are gone rather than fixed — the first with the
    # `__jaz_get__` protocol, the second with the render overrides that minted the wrapper.
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


__all__ = ["Template", "normalize_inputs"]

# The list is this module's surface *to the rest of the package*: the names another module under
# `src/jaz` imports. `_invoke` imports both — `normalize_inputs` to lower an invoke's inputs, and
# `Template` because this module is where the <3.14 fallback for it lives, so importing it from
# `typing`/`string.templatelib` directly would break on the older interpreter.
#
# `normalize_task` is deliberately absent. It lowers a t-string into text plus bindings for
# `normalize_inputs` on this module's own path, and nothing else under `src/jaz` calls it.
# `tests/test_templates.py` does import it directly, which is the normal way a test reaches an
# internal; a test importer is not what makes a name module surface, or every helper would be
# listed.
