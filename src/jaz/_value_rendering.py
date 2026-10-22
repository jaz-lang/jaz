"""Value rendering: the text an agent's prompt shows for each input.

:func:`render_value` turns an arbitrary object into that text. Most values render as their
``str()``; a class, a module, any other callable, and an object with no ``str()`` of its own
render as a description of how to use them. The prompt builder calls it for every input, and
the console's ``?expr`` calls it to show what an agent would see.
"""

from __future__ import annotations

import enum
import functools
import inspect
import sys
import types
import typing
from collections.abc import Callable

# PEP 649 (3.14+). Before that, annotations are a plain dict in the class namespace and
# this module does not exist. A `version_info` guard rather than try/except, so a type
# checker pinned to an older version sees the branch as unreachable instead of a missing
# import.
if sys.version_info >= (3, 14):
    import annotationlib
else:
    annotationlib = None  # type: ignore[assignment]


# `render_value` is private; there is no "default" left for it to be the default OF. It used
# to be `default_desc`, the half of the experimental description system that nothing in-tree
# overrode: a type could supply prompt text separate from its `str()` through `__desc__`, and
# this rendered every value that did not. v1 does not include the description system, so the
# override went and this rule stayed, because the prompt builder cannot work without it.
#
# What went with it, accepted: a value can no longer read differently in a prompt than in a
# REPL, since `__str__` now serves both, and `__str__` takes no arguments, so rendered text
# cannot be phrased against the kwarg the value is bound under. `_console.ConversationHistory`
# was the one in-tree object that used both.
#
# The rule is public behaviour even though the name is not: the agent-facing `invoke`
# docstring states it to every sub-agent author (`_subinvoke.py` has the exact, deliberately
# understated wording).


def render_value(value: object, bound_name: str | None = None) -> str:
    """Render ``value`` as the text an agent's prompt shows for it.

    A ``str`` for most values, with three exceptions: a class or a module renders as its
    docstring and public members (for a module, ``__all__`` when it declares one); any other
    callable (a function, a method, or an object with ``__call__``) as its signature and
    docstring, which for an object with ``__call__`` is followed by ``__call__``'s own
    docstring when it has a different one, and then by its public members; and an object
    that renders itself *neither* through ``__str__`` nor through ``__repr__`` as its class
    docstring and public attributes.

    A member that belongs to the value itself -- defined or annotated on its own class, set on the
    instance, or for a module a function defined in the module's own package -- renders in
    full. Members inherited from a base class are listed by name, grouped under the base that
    defines them, and functions a module imports from another package by name, grouped under
    the module they come from; ``help()`` on one gives the rest. A class listed as a member
    renders as its signature and the first paragraph of its docstring. A member marked
    deprecated with PEP 702's ``@deprecated`` is not listed.

    A class or callable is described as above even when it defines ``__str__``. A value meant
    to be read through its ``__str__`` should therefore not define ``__call__``.

    "Renders itself" is the load-bearing test, and it asks about both dunders because
    ``str()`` falls back to ``__repr__``. Asking only about ``__str__`` sent every library
    type that defines ``__repr__`` — a ``DataFrame``, a ``Series`` — into the member walk;
    see :func:`_has_own_rendering`.

    Nothing here is truncated. Bounding what reaches the prompt is the caller's job and the
    caller already has the knob — ``max_invoke_input_length`` on the protocol — so a second
    ceiling inside the renderer would be one this function picked, unconfigurable, silently
    overriding a caller who deliberately raised theirs.

    ``bound_name`` is the name the value renders under, normally the kwarg it is passed to
    :func:`jaz.invoke` as. Without one, a class or function renders under its own
    ``__name__`` and anything else renders unnamed — no placeholder is invented.

    Members of a nested namespace are **not** listed: a namespace renders its own top-level
    names, and a namespace among them renders as a single entry rather than being walked into.
    """
    # Not recursive, by executive call (user, 2026-09-09): walking nested namespaces is
    # unbounded in the one direction that costs the most — an input two levels deep can
    # dominate the prompt, and the agent pays for it on every turn. Top-level names only,
    # so the size of this rendering is a function of the value the author actually passed.
    #
    # Shape decides before `__str__` is consulted, by executive call (user, 2026-10-03): what
    # you can do with a value decides which channel describes it. A thing you call or
    # instantiate is described by how to use it (signature and docstring); a thing you read
    # by what it holds (`str()`). A callable's `__str__` is almost always a NAME — written to
    # sit inside other output (`ZZ` in `Poly(x, domain=ZZ)`, an op name in a graph dump,
    # `<function f at 0x…>` through a wrapper) — not a description for a reader starting cold.
    #
    # Measured before deciding, over module-level objects in 145 installed packages (#1585
    # has the scan): of 418 callables and classes with a Python-written `__str__`, deferring
    # to it rendered 238 worse than their docstring, 161 no better and 19 better; deferring
    # to a C-level one broke 1,281 of 1,289 (`np.sqrt` as `<ufunc 'sqrt'>`; the other 8
    # raise either way). Rejected with it: deferring only to a Python-written `__str__` (the
    # 238 above), only when `str() != repr()` (`np.all` passes, its two differ only in the
    # address), and only to a multi-line `str()` (fits the scan, but fitted to three
    # positives). The cost is data that happens to be callable — `mpmath.pi`,
    # `matplotlib.colormaps` — renders as a callable; its docstring still says what it holds,
    # or how to get it, in 15 of the 19 cases. A value that wants its `__str__` rendered
    # should not be callable: AppWorld's `apis` root prompted this, is never called itself,
    # and drops `__call__` in jaz-lang/jaz-evals-internal#332.
    if isinstance(value, type):
        return _format_class(bound_name, value)
    if callable(value):
        return _format_callable(bound_name, value)
    if isinstance(value, types.ModuleType):
        return _format_module(bound_name, value)
    # A bare namespace is walked even though it defines `__repr__`, for the same reason the
    # module branch above exists: `SimpleNamespace` is the stdlib's ANONYMOUS NAMESPACE type,
    # with no domain meaning of its own beyond the names it carries, so the names ARE the
    # description. A `dataclass`, an `argparse.Namespace` or an attrs class is not exempted
    # and does not want to be — each is a domain type whose generated repr states its own
    # values, which is a better description than a walk of its fields.
    #
    # (Unbounded recursion is a consequence, not the reason. `SimpleNamespace.__repr__`
    # nests without limit and spells leaves as `<function f at 0x104e85940>` — but so does
    # every generated repr, and those are left alone.)
    # `type(...) is`, not isinstance: a SUBCLASS may define its own `__repr__`, and then it
    # does print itself and the argument below does not apply to it.
    if type(value) is types.SimpleNamespace:
        return _format_object(bound_name, value)
    if not _has_own_rendering(value):
        return _format_object(bound_name, value)
    return str(value)


def _own_name(value: object) -> str | None:
    """The value's own ``__name__``, when it has a usable one.

    Only functions, classes and modules carry one. `getattr` rather than an `isinstance`
    chain because anything defining it means it as its name — and a non-`str` one (a `Mock`
    fabricates attributes) is not usable as a label.
    """
    own = getattr(value, "__name__", None)
    return own if isinstance(own, str) and own else None


def _has_own_rendering(value: object) -> bool:
    """Whether ``value``'s type renders itself, via its own ``__str__`` OR ``__repr__``.

    `str()` falls back to `__repr__`, so asking only about `__str__` — as this dispatch once
    did — misreads every library that follows the convention of defining `__repr__` and
    leaving `__str__` alone. pandas is the case that made it urgent: a `DataFrame` answered
    "no rendering of its own" and took the member walk, producing ~715,000 characters of
    pandas API docs that never once mentioned the frame's columns or its data, where
    `str(df)` is a few dozen characters that show both.

    Read out of each class's `vars()` rather than by comparing `type(value).__str__` to
    `object.__str__`: when the metaclass defines `__str__` as a property, the comparison form
    evaluates it, and this predicate runs on every described input. (A metaclass
    `__getattr__` is *not* a hazard here — `__str__` always resolves on `object`, so the
    fallback hook never fires for it.)
    """
    return any(
        attr in vars(base)
        for base in type(value).__mro__
        if base is not object
        for attr in ("__str__", "__repr__")
    )


def _first_paragraph(text: str) -> str:
    """The first paragraph of ``text`` — what a nested or bulk listing shows.

    A paragraph rather than a first line: numpydoc-style summaries wrap, so cutting at the
    newline truncates mid-sentence for a minority of members and does so silently.
    """
    return text.split("\n\n", 1)[0].strip()


def _format_module(name: str | None, mod: types.ModuleType) -> str:
    """Render a module as its docstring and a line per public member, as a class renders.

    The public members are ``__all__`` when the module declares one, since that is the
    author's own statement of its public surface, and otherwise every name it assigns or
    annotates without a leading underscore. A nested module renders as one line, not walked
    into. On Python 3.14, reading the module's annotations evaluates them.
    """
    # Rendered like a class, by executive call (user, 2026-10-04): a module is a namespace of
    # things to call, which is what the class walk describes. It used to render only its
    # docstring's first paragraph and the bare names it exports, because a walk of `numpy`
    # with full docstrings reached ~1,056,000 characters; that bounded big libraries by
    # starving every module of signatures, a five-function `tools.py` included. Size is #1586's
    # question, for classes and modules alike, and is not answered here.
    label = name or _own_name(mod)
    # Unnamed, it says so in prose: the bare `[module]:` form is the one `_format_object`
    # argues against, and the two should not contradict each other.
    head = f"`{label}` [module]" if label else "A module"
    # vars(), not getattr: a lazy-import module defines a PEP 562 module-level `__getattr__`,
    # and those are exactly the modules someone hands an agent.
    own = vars(mod)
    # On 3.14 this evaluates the module's annotation expressions (`_own_annotations` says
    # why no format avoids it) -- user code the names-only rendering never ran. Read even
    # with `__all__`, since a declared name may be annotated and never assigned.
    annotations = _own_annotations(mod)
    exported = own.get("__all__")
    if isinstance(exported, (list, tuple)) and all(
        isinstance(n, str) for n in exported
    ):
        # `dict.fromkeys` because a hand-written `__all__` can repeat a name.
        names = sorted(dict.fromkeys(exported))
    else:
        # Annotated but unassigned names too (`count: int`), as the class walk lists them.
        names = sorted(n for n in {*own, *annotations} if not n.startswith("_"))
    lines = _head_block(head, _get_doc(mod), "  ", bool(names))
    # `vars()`, as for everything else here; a module may have deleted its `__name__` or set it
    # to `None`, and then nothing can be told apart from it, so every member is its own.
    mod_name = own.get("__name__")
    package = mod_name.split(".")[0] if isinstance(mod_name, str) else None
    imported: dict[str, list[str]] = {}
    for n in names:
        if n in own and _is_deprecated(own[n]):
            continue
        if n in own or n in annotations:
            # In full when defined in this module's package; NAMED, grouped by where it comes
            # from, when imported from another, as an inherited member is (`_walk_members` has
            # why named rather than dropped or described). The package, not the module: a
            # package re-exports its submodules' API from `__init__` (`requests.get` lives in
            # `requests.api`), and that is its own. Only a function says where it was defined; a
            # plain value is the module's own, whatever its type's origin. A function from
            # `__main__` -- a script or a notebook -- is the user's own code wherever it is
            # collected. Known limitation: "package" is the first dotted component, so separately
            # distributed namespace packages sharing one (`google.protobuf`, `google.cloud`) count
            # as one package, and a function one imports from another renders in full.
            origin = _defined_in(own[n]) if n in own else None
            if (
                package is None
                or origin is None
                or origin == "__main__"
                or origin.split(".")[0] == package
            ):
                lines.append(_render_member(mod, n, annotations))
            else:
                imported.setdefault(origin, []).append(n)
        elif "__getattr__" in own:
            # Declared in `__all__`, supplied by the module's PEP 562 `__getattr__` on first
            # access. Not fired: that is typically an import. (A module built with
            # `importlib.util.LazyLoader` is different: `vars()` above loads it, as it did
            # before.)
            lines.append(f"  - `{n}` (loaded on first access)")
        else:
            # A stale `__all__` entry: nothing defines it. Not `(unreadable)`, which means
            # reading a value that exists raised.
            lines.append(f"  - `{n}` (not defined)")
    lines.extend(
        _named_group(f"imported from `{origin}`", group)
        for origin, group in imported.items()
    )
    return "\n".join(lines)


def _defined_in(value: object) -> str | None:
    """The module a function says it was defined in; ``None`` for anything else."""
    # Functions only. A class in a walk is summarised to its first paragraph whatever its
    # origin, so it stays on its own line. A `partial` comes from where the function it
    # wraps does (`attrs.frozen` is a `partial` of `attr._next_gen.define`), not from
    # `functools` -- which from 3.14, where `isroutine` is true for one, it would otherwise
    # read as. By plain attribute: a routine has no hook to fire. Not `getattr_static`, which
    # returns `function.__module__`'s descriptor rather than its value.
    while isinstance(value, functools.partial):
        value = value.func
    if not inspect.isroutine(value):
        return None
    origin = getattr(value, "__module__", None)
    return origin if isinstance(origin, str) else None


def _format_callable(name: str | None, value: Callable[..., object]) -> str:
    """Render a callable, falling back to its own ``__name__`` when unnamed.

    A callable with neither — a `functools.partial`, an instance with `__call__` — renders
    as ``A callable with signature `(...)`: ...``, with nothing standing in for the name it
    does not have.
    """
    # Through the same ladder and the same doc reader the member walk uses. This used to
    # call `inspect.signature`/`_get_doc` directly, which meant a proxy — a class defining
    # both `__getattr__` and `__call__` — was probed 8 times as a top-level INPUT and 0 times
    # as someone's attribute, and a `functools.partial` showed `functools.partial`'s own
    # class docstring instead of the wrapped function's. Same asymmetry, same direction, as
    # the constructor-in-the-header fix.
    label = name or _own_name(value)
    sig = _static_signature(value)
    doc = _indent_continuation(_callable_doc(value), "  ")
    # Unnamed only off the prompt path (the console's `?expr`, a direct call without a name).
    # In prose, as `_format_module` and `_format_object` say it for theirs: a bare `` `(y)` ``
    # reads as a tuple, and `A callable `(y)`` as a callable named `y`.
    head = (
        f"`{label}{sig}`: {doc}"
        if label
        else f"A callable with signature `{sig}`: {doc}"
    )
    # An object that is callable is still an object: its other public members are listed as
    # an object's are, by executive call (user, 2026-10-04). Without them a tool object's
    # `.reset()` or `.fetch()` was invisible -- of 613 public methods the scanned callable
    # objects' own classes define, the header named 24%. A function, a method and a `partial`
    # are not walked: their attributes are machinery, not interface. (A class never reaches
    # here; `render_value` renders it first.) (From 3.14
    # `isroutine` already covers a `partial`; before it, only the explicit check does.)
    if inspect.isroutine(value) or isinstance(value, functools.partial):
        return head
    members = _walk_members(value, type(value))
    if not members:
        return head
    return "\n".join([head, *([""] if "\n" in head else []), *members])


def _safe_getattr(obj: object, name: str) -> tuple[object, bool]:
    """``(value, ok)`` for ``obj.name``, never raising.

    Rendering a value walks its public attributes, and reading one runs arbitrary user
    code: a `@property` that raises, a descriptor that hits the network, a lazy attribute
    whose backing store is gone.

    This swallows where `render_value`'s final `str()` raises, and the asymmetry is the point: a member
    whose VALUE cannot be read costs one line, annotated `(unreadable)`, while the other
    members still render — whereas a failed `str()` leaves no description at all. Local
    degradation is worth hiding; total failure is not.

    The guard covers reading the value and nothing else. Other user code the walk touches —
    a raising `__doc__` property, a metaclass `__doc__` that kills the head line before any
    member renders, a `__signature__` raising something `_get_signature` does not catch —
    still propagates raw. That is pre-existing, not a property of this asymmetry, and is why
    the sentence above says "cannot be read" rather than "fails".
    """
    try:
        return getattr(obj, name), True
    except Exception:  # noqa: BLE001 — arbitrary user code; nothing here is recoverable
        return None, False


# HOW A MEMBER IS CLASSIFIED, and why not `callable(value)`.
#
# The walk used to ask `callable(getattr(obj, name))`. Two things are wrong with that. It is
# not a test for "method" — `self.handler = lambda a: 5` is an attribute whose value happens
# to be callable, and it rendered in method shape. And `getattr` RUNS the user's code to find
# out: a `@property` getter, a descriptor that opens a socket, pandas' `.sparse` accessor
# (which raises on an ordinary frame and took the whole invoke down with it in 0.2.0a4).
#
# So the member is classified from `inspect.getattr_static`, which resolves a name the way
# Python does — data descriptor, then instance dict, then non-data descriptor, then class
# variable — WITHOUT invoking any of them. Only kinds whose `__get__` is implemented in C or
# in the stdlib are then bound; everything else renders from the descriptor object itself.
#
# `_safe_getattr` survives, and two paths still reach it: reading the value behind a C
# slot/getset on an instance, and `str()` on the self-describing branch.
#
# Those are not quite the only places user code runs. Rendering a signature calls `repr()` on
# every default — unavoidable, since `a=0` cannot be shown without it — and formats every
# annotation. What the walk guarantees is narrower than "nothing of yours runs", and worth
# stating exactly: **nothing runs merely to decide what KIND a member is**, and nothing that
# does run can take the description down. Classification is entirely static; `_render_signature`
# contains the repr hazard by eliding the defaults it cannot read.

# Kinds whose `__get__` is C or stdlib, so binding one cannot execute anything of the user's.
# Spelled as exact types on purpose: `inspect.ismethoddescriptor` is duck-typed ("has
# `__get__`, no `__set__`, is not a function or class") and matches pandas' `Accessor`, so
# using the predicate here would re-open the crash this rewrite exists to close.
# NOT `BuiltinFunctionType`: a builtin function is not a descriptor at all, so `x = len` in
# a class body stays exactly `len` on lookup. It falls through to the value branch, which
# renders it as the callable it is.
_BINDABLE_C_KINDS = (
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.ClassMethodDescriptorType,
)


def _is_type_alias(value: object) -> bool:
    """``list[int]``, ``Optional[str]``, a ``TypeVar`` — a type expression, not a value."""
    if isinstance(value, typing.TypeAliasType):
        # A 3.12 `type Row = list[int]`. Its `str()` is just "Row", so show what it aliases.
        return True
    return (
        isinstance(value, (types.GenericAlias, types.UnionType, typing.TypeVar))
        or type(value).__module__ == "typing"
    )


def _defines_call(cls: type) -> bool:
    """Whether instances of ``cls`` are callable in their own right.

    Scans each class's `vars()` rather than asking `getattr_static(cls, "__call__")`, which
    falls back to the METATYPE: `type.__call__` exists for every class, so the lookup form
    answers True for `5`, `"s"`, `[]` and everything else, and the branches below it would
    never be reached.
    """
    return any("__call__" in vars(base) for base in cls.__mro__ if base is not object)


class _Elided:
    """Stands in for a default whose own ``repr()`` cannot be trusted."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "..."


_ELIDED = _Elided()


def _render_signature(sig: inspect.Signature) -> str:
    """``str(sig)``, with any default that cannot be shown elided to ``...``.

    `Signature.__str__` calls `repr()` on every default, which runs user code: a default whose
    `__repr__` raises would take the whole description down. Each default is rendered on its
    own, so one that cannot be shown costs that parameter only; the call's shape (which
    parameters are optional) is kept either way.
    """
    try:
        params = [
            p
            if p.default is inspect.Parameter.empty
            else p.replace(default=_default_text(p.default))
            for p in sig.parameters.values()
        ]
        return str(sig.replace(parameters=params))
    except Exception:  # noqa: BLE001 — an annotation's own repr, or an exotic Signature
        return "(...)"


class _DefaultText:
    """A default's display text, given as its ``repr`` so ``Signature`` prints it verbatim."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _default_text(default: object) -> object:
    """What a signature shows for ``default``: its ``repr()``, or ``...`` when that raises."""
    # The repr is shown as written, memory addresses included (`<object object at 0x...>`).
    # Executive call (user, 2026-10-08): no address scrubbing in the renderer. The reasons that
    # were given for it do not hold up -- replay divergence also sees addresses in REPL output,
    # which the renderer cannot reach (tracked for the replay comparator in #1627); prompt caching
    # is unaffected within a run and the opener is the only prefix runs could share; and a diff
    # that differs only in an address is easy to read past. An identity-based scrubber with
    # per-prompt labels was built and dropped for that reason.
    try:
        return _DefaultText(repr(default))
    except Exception:  # noqa: BLE001 — an arbitrary `__repr__`
        return _ELIDED


def _bound_signature(sig: inspect.Signature) -> inspect.Signature | None:
    """``sig`` as seen through a bound receiver, or ``None`` if it cannot take one."""
    # The stdlib's rule, from `inspect._signature_bound_method` (private, so restated rather
    # than called): a leading positional parameter is the receiver and goes; a leading
    # `*args` absorbs the receiver and stays, so `__call__(*args)` is called with arguments,
    # not `()`; with no parameter at all, or a keyword-only or `**kwargs` one first, there is
    # nowhere for the receiver to go and calling it as a method raises.
    params = list(sig.parameters.values())
    if not params or params[0].kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.VAR_KEYWORD,
    ):
        return None
    if params[0].kind is inspect.Parameter.VAR_POSITIONAL:
        return sig
    return sig.replace(parameters=params[1:])


def _drop_first_param(sig: inspect.Signature) -> str:
    bound = _bound_signature(sig)
    return _UNKNOWN_SIGNATURE if bound is None else _render_signature(bound)


def _class_tag(cls: type) -> str:
    """``[class]``, or ``[abstract class]`` when calling it would raise.

    The constructor is still shown for an abstract class: it is the signature every concrete
    subclass takes, and a reader handed the base usually wants exactly that. What the tag adds
    is the one thing the signature cannot say — that this particular call raises
    ``TypeError``. `inspect.isabstract` reads `__flags__`, so it runs no user code.
    """
    return "[abstract class]" if inspect.isabstract(cls) else "[class]"


def _metaclass_call(cls: type) -> types.FunctionType | None:
    """The metaclass's ``__call__``, when one is written in Python.

    Every class is callable through its metatype, but for a plain class that is ``type``'s
    C slot wrapper (and ``EnumMeta``'s is C as well), so this returns ``None`` for them and a
    real function only for a metaclass someone wrote.
    """
    # `enum` is carved out, and it is the reason this needs a carve-out at all: on 3.12
    # `EnumType.__call__` IS a Python function, but it is the FUNCTIONAL API — the thing that
    # builds a whole new enum class (`Color("Color", "RED GREEN")`) overloaded with value
    # lookup. Reporting it would advertise
    # `Color(value, names=..., *values, module=None, ...)` for a type whose members you reach
    # as `Color.RED`. No static signal separates a lookup protocol from a constructor, so the
    # exception is named rather than inferred.
    if isinstance(cls, enum.EnumMeta):
        return None
    call = inspect.getattr_static(type(cls), "__call__", None)
    return call if inspect.isfunction(call) else None


def _ctor_dunders(cls: type) -> tuple[types.FunctionType | None, ...]:
    """``__init__`` then ``__new__``, unwrapped, keeping only Python functions."""
    out: list[types.FunctionType | None] = []
    for dunder in ("__init__", "__new__"):
        fn = inspect.getattr_static(cls, dunder, None)
        fn = fn.__func__ if isinstance(fn, (staticmethod, classmethod)) else fn
        out.append(fn if inspect.isfunction(fn) else None)
    return tuple(out)


def _static_signature(value: object) -> str:
    """``value``'s call signature, obtained without running the user's code.

    `inspect.signature` on anything that is not a routine reads a handful of dunders off the
    object (`__signature__`, `__wrapped__`, `__call__` and friends), so on a proxy or a mock
    — a class defining both `__getattr__` and `__call__` — merely describing it fires
    `__getattr__` once per probe. Hence the ladder: a declared
    `__signature__` first (what `functools.wraps`, click and typer set, and the one thing a
    naive `__call__` read destroys), then the stdlib wrappers, then the `__call__` slot.
    """
    if inspect.isroutine(value):
        # Functions, bound methods, builtins and C descriptors have no `__getattr__` to fire.
        return _get_signature(value)
    if isinstance(value, type):
        # A class's constructor, NOT `inspect.signature(cls)` — that one fires the metaclass
        # `__getattr__` for `__signature__` and `_partialmethod`.
        #
        # A declared `__signature__` first, same as the non-class ladder below: a class that
        # states its own call signature is stating the constructor, and the `__init__` read
        # would silently overrule it.
        own_sig = inspect.getattr_static(value, "__signature__", None)
        if isinstance(own_sig, inspect.Signature):
            return _render_signature(own_sig)
        # `Cls(...)` runs `type(Cls).__call__`, so a metaclass that defines one IS the
        # constructor and `__init__` is only whatever it chooses to delegate to. Read it
        # first — otherwise the header advertises `Client()` for a class whose real call is
        # `Client(name, *, region='us')`, which is wrong rather than merely incomplete.
        #
        # `isfunction` is the whole discriminator: every class has a `__call__` through its
        # metatype, but on a plain class that is `type`'s C slot wrapper, and on `EnumMeta`
        # it is C too. Only a metaclass written in Python reaches this.
        for source in (_metaclass_call(value), *_ctor_dunders(value)):
            if source is None:
                continue
            # `-> None` is `__init__`'s annotation, not the constructor's: calling the class
            # returns an instance. Showing it would tell the agent the call has no useful
            # result.
            sig = inspect.signature(source).replace(
                return_annotation=inspect.Signature.empty
            )
            # Through `self`, or the metaclass's `cls`, as a method binds it. A source with
            # nowhere to put the receiver makes construction raise, so the class has no
            # constructor to show. Unknown, not the next source: that would advertise a call
            # this source breaks (`__init__()` beside `__new__(cls, x)` renders `C(x)`, and
            # `C(1)` raises).
            bound = _bound_signature(sig)
            if bound is None:
                return _UNKNOWN_SIGNATURE
            params = list(bound.parameters.values())
            # A constructor that reduces to `*args, **kwargs` describes nothing, and on a
            # class that never declared one it is actively misleading: an `enum.Enum`
            # inherits `Enum.__new__` and would advertise `Color(*args, **kwds)` for a type
            # you reach as `Color.RED`. Fall through to the next source rather than claiming
            # it — which is also what makes a FORWARDING metaclass (`__call__(cls, *args,
            # **kwargs)` delegating to `super()`) yield to the real `__init__` beneath it,
            # instead of erasing it. If nothing informative is left, it is unknown.
            if params and all(
                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params
            ):
                continue
            return _render_signature(bound)
        # Neither dunder is a Python function. `()` would be a CLAIM — and a false one for
        # every class whose constructor is C-level: `class Config(dict)` accepts `Config(a=1)`
        # while inheriting `dict.__init__`, and so does every `Exception`, `list`, `str` and
        # `datetime` subclass. `inspect.signature` reports those as unknown by raising; saying
        # `()` converts "I cannot tell" into "takes no arguments". Only when both dunders are
        # literally `object`'s is the no-argument constructor real.
        inherits_object_ctor = all(
            inspect.getattr_static(value, d, None) is getattr(object, d)
            for d in ("__init__", "__new__")
        )
        return "()" if inherits_object_ctor else _UNKNOWN_SIGNATURE
    declared = inspect.getattr_static(value, "__signature__", None)
    if isinstance(declared, inspect.Signature):
        return _render_signature(declared)
    if isinstance(value, (functools.partial, functools.partialmethod)):
        # Stdlib types with no `__getattr__`, and `signature` is what computes the residual
        # signature from `.func`/`.args`/`.keywords`. The `__call__` slot would say only
        # `(*args, **kwargs)`.
        return _get_signature(value)  # type: ignore[arg-type]
    call = inspect.getattr_static(type(value), "__call__", None)
    # Read raw from the class, a `staticmethod` or `classmethod` is its wrapper, not a
    # function; unwrapped here as `_call_doc` unwraps it, or the header claimed `()` beside a
    # docstring describing the real parameters. A staticmethod receives no receiver, so it
    # keeps every parameter; a classmethod's `cls` is bound, so it drops one like `self`.
    if isinstance(call, staticmethod) and inspect.isfunction(call.__func__):
        return _render_signature(inspect.signature(call.__func__))
    if isinstance(call, classmethod) and inspect.isfunction(call.__func__):
        return _drop_first_param(inspect.signature(call.__func__))
    if isinstance(call, (staticmethod, classmethod)):
        # Around something other than a Python function (`staticmethod(len)`, a `partial`):
        # unknown. `()` would claim it takes no arguments -- the class branch's reasoning.
        return _UNKNOWN_SIGNATURE
    if inspect.isfunction(call):
        return _drop_first_param(inspect.signature(call))
    # A C-level `__call__` (`np.sqrt`, `operator.itemgetter(1)`): unknown, as for a class
    # above, not `()`, which would tell the agent it takes no arguments.
    return _UNKNOWN_SIGNATURE


def _static_doc(value: object) -> str:
    """``value``'s docstring, read without triggering attribute lookup on the object.

    `getdoc` semantics — the object's own `__doc__` if it has one, else its class's — which
    is what `help()` shows and what a `functools.update_wrapper` wrapper needs, since that is
    where `wraps` puts the wrapped function's docstring.

    A `partial` is the exception: it is callable and its class has a docstring, but that
    docstring describes `functools.partial` itself, so rendering it would tell the agent how
    partial application works instead of what this particular callable does.
    """
    if isinstance(value, (functools.partial, functools.partialmethod)):
        return _static_doc(value.func)
    if not isinstance(value, type) and not inspect.isroutine(value):
        own = inspect.getattr_static(value, "__doc__", None)
        if isinstance(own, str) and own.strip():
            # `cleandoc`, as `getdoc` does for the class path below: before 3.13 the compiler
            # leaves a docstring's source margin in `__doc__`, and `strip()` keeps it on every
            # line after the first.
            return inspect.cleandoc(own)
    return _get_doc(value)  # type: ignore[arg-type]


def _callable_doc(value: object) -> str:
    """A callable's docstring, followed by its ``__call__``'s when it differs.

    The signature a callable instance renders with is ``__call__``'s, but the docstring
    beside it is the object's — what the instance IS, not what calling it does.
    """
    # Measured over 6,299 module-level callable instances in 97 packages before adding this:
    # 594 had a documented `__call__`, and the rendering dropped 587 of them. About 400 said
    # what the call does that nothing else did — a matplotlib `Colormap` renders
    # `(X, alpha=None, bytes=False)` beside a docstring about lookup tables, while "X: the
    # data value(s) to convert to RGBA" sat on `__call__`; a scipy distribution's call
    # freezes it. About 180 add one generic line ("Alias for :meth:`main`." on every click
    # command, "Execute the call behavior." on numpy.ma's operations), accepted as cheaper
    # than a rule for telling them apart.
    #
    # A class-based decorator that copies the wrapped function's metadata onto itself
    # (`functools.update_wrapper(self, f)`) gets its `__call__` docstring too, after the wrapped
    # function's: "Call the wrapped function, retrying up to 3 times" is worth having, "Invoke
    # the decorated function." is not. Kept, by executive call (user, 2026-10-03), for
    # simplicity where the difference is one line either way. Rejected: skipping the
    # `__call__` docstring when the instance carries `__wrapped__`; none of the 587 above did.
    own = _static_doc(value)
    # Through a `partial` to what it wraps, as `_static_doc` already does for the docstring.
    target = value.func if isinstance(value, functools.partial) else value
    call = _call_doc(target)
    # Both sides are already dedented -- `own` by `_static_doc`, `call` by `_call_doc` -- so a
    # multi-line docstring repeated on `__call__` compares equal on every version, where
    # before 3.13 a source margin left in `__doc__` would make it render twice.
    if call is None or call == own:
        return own
    if own == _NO_DESCRIPTION:
        return call
    return f"{own}\n\n`__call__`: {call}"


def _call_doc(value: object) -> str | None:
    """The docstring of a callable instance's ``__call__``, when one is written in Python."""
    # Instances only. A class's `__call__` is its metaclass's — construction, and for an `Enum`
    # some 25 lines of `EnumType.__call__` — so a class is refused here: the class branch
    # renders a bare one, but `_callable_doc` follows a `partial` to its `.func`, which can be
    # a class. Python-written only: a function's, method's and `partial`'s `__call__` are C
    # slots whose docstring is "Call self as a function.".
    #
    # Looked up, and unwrapped from a `staticmethod` or `classmethod`, as `_static_signature`
    # and `_is_async_callable` do it, and each of the three follows a `partial` to what it
    # wraps, so the signature, the `async` prefix and this docstring describe the same
    # function. It runs no attribute lookup on the instance and no
    # descriptor `__get__`; a metaclass `__getattribute__` still sees `__mro__` and
    # `__dict__`, as it does for every static read.
    if isinstance(value, type):
        return None
    call = inspect.getattr_static(type(value), "__call__", None)
    # A `staticmethod` or `classmethod` around a Python function is still Python-written.
    # Any other object assigned as `__call__` (a callable instance, a non-`wraps` decorator's
    # result) is skipped: its `__doc__` describes that object, not the call.
    if isinstance(call, (staticmethod, classmethod)):
        call = call.__func__
    if not inspect.isfunction(call):
        return None
    doc = inspect.cleandoc(call.__doc__ or "")
    return doc or None


def _is_async_callable(value: object) -> bool:
    """Whether calling ``value`` returns an awaitable, decided without attribute lookup.

    `inspect.iscoroutinefunction` does the same `getattr` sweep over dunders that
    `_static_signature` climbs a ladder to avoid, so it is safe only once the value is known
    to be a routine. For anything else the `__call__` slot answers it statically.
    """
    # Through a `partial` to what it wraps, as the signature and the docstring already go.
    # Before the routine check: from 3.14 `isroutine` is true for a `partial`, and
    # `iscoroutinefunction` then misses one wrapping an instance whose `__call__` is async.
    if isinstance(value, functools.partial):
        return _is_async_callable(value.func)
    if inspect.isroutine(value):
        return inspect.iscoroutinefunction(value)
    call = inspect.getattr_static(type(value), "__call__", None)
    if isinstance(call, (staticmethod, classmethod)):
        call = call.__func__
    return inspect.isfunction(call) and inspect.iscoroutinefunction(call)


def _callable_member_line(attr_name: str, value: object) -> str:
    prefix = "async " if _is_async_callable(value) else ""
    doc = _indent_continuation(_callable_doc(value), "    ")
    return f"  - `{prefix}{attr_name}{_static_signature(value)}`: {doc}"


def _value_member_line(attr_name: str, value: object) -> str:
    """Render a member from its VALUE, by the same branches `render_value` uses on an input.

    The same branches `render_value` applies to a top-level input, one level shallower: a
    nested class or object contributes its summary rather than a walk of its own, which is
    what keeps a two-deep input from dominating the prompt (and is the non-recursion rule
    recorded in `render_value`).
    """
    if _is_type_alias(value):
        # A `TypeAliasType` prints as its own name, which would render `Row = Row`.
        shown = value.__value__ if isinstance(value, typing.TypeAliasType) else value
        return f"  - `{attr_name}` = {shown}"
    if isinstance(value, type):
        # Tagged `[class]`, matching BOTH conventions already in this rendering: the header
        # of a class input is `` `Env()` [class] ``, and a non-callable member already carries
        # its type in brackets (`` `retries` [int] ``). Untagged, a class member was the one
        # thing with a type that did not say so — and on an instance input it was
        # indistinguishable from a method, since both render as name + signature + docstring.
        # Calling either is the same keystroke; what differs is that one hands back an
        # instance, which is what the agent needs to know before deciding to call it.
        doc = _first_paragraph(_get_doc(value))
        return f"  - `{attr_name}{_static_signature(value)}` {_class_tag(value)}: {doc}"
    if inspect.isroutine(value) or _defines_call(type(value)):
        return _callable_member_line(attr_name, value)
    if isinstance(value, types.ModuleType):
        return f"  - `{attr_name}` [module]: {_first_paragraph(_get_doc(value))}"
    if not _has_own_rendering(value) or isinstance(value, types.SimpleNamespace):
        doc = _first_paragraph(_get_doc(type(value)))
        return f"  - `{attr_name}` [{type(value).__name__}]: {doc}"
    return f"  - `{attr_name}` [{type(value).__name__}]"


def _annotated_type_name(annotation: object) -> str:
    """Render an annotation the way its author wrote it.

    NOT `__name__` first: on a parameterised generic that is the bare origin, so
    `Optional[str]` rendered as `Optional` — naming a type that does not exist — and
    `list[int]` lost the `int`, while `int | None` kept everything because `UnionType` has
    no `__name__`. `inspect.formatannotation` keeps the parameters for every spelling and
    still collapses a plain class to its short name.
    """
    if isinstance(annotation, str):
        return annotation  # an unresolved forward reference; quote it as written
    try:
        return inspect.formatannotation(annotation)
    except Exception:  # noqa: BLE001 — a hand-rolled annotation object may do anything
        return str(annotation)


def _render_member(
    obj: object, attr_name: str, annotations: dict[str, object] | None = None
) -> str:
    """One member line, classified without invoking anything the user wrote."""
    annotations = annotations or {}
    is_class = isinstance(obj, type)
    own = () if is_class else getattr(obj, "__dict__", ())

    try:
        raw = inspect.getattr_static(obj, attr_name)
    except AttributeError:
        # Declared (`x: int`, or a `__slots__` name) but never assigned. A dataclass passed
        # as a CLASS is the common case: its undefaulted fields live only in `__annotations__`
        # and were invisible to a `vars()`-only walk, so the header advertised a constructor
        # parameter the member list denied.
        if attr_name in annotations:
            return f"  - `{attr_name}` [{_annotated_type_name(annotations[attr_name])}]"
        return f"  - `{attr_name}` (unreadable)"

    if isinstance(raw, (property, functools.cached_property)):
        getter = raw.fget if isinstance(raw, property) else raw.func
        if getter is None:
            return f"  - `{attr_name}` (write-only)"
        try:
            ret = inspect.signature(getter).return_annotation
        except (TypeError, ValueError):
            ret = inspect.Signature.empty
        suffix = (
            "" if ret is inspect.Signature.empty else f" [{_annotated_type_name(ret)}]"
        )
        own_doc = inspect.getattr_static(raw, "__doc__", None)
        doc_text = (
            inspect.cleandoc(own_doc)
            if isinstance(own_doc, str) and own_doc.strip()
            else _static_doc(getter)
        )
        doc = _indent_continuation(doc_text, "    ")
        return f"  - `{attr_name}`{suffix}: {doc}"

    if inspect.isdatadescriptor(raw):
        c_storage = inspect.ismemberdescriptor(raw) or inspect.isgetsetdescriptor(raw)
        # A C member or getset descriptor keeps its docstring in a slot that `getattr_static`
        # does not see -- it returns the descriptor TYPE's `__doc__` slot -- so read it plainly:
        # it is a C object, and reading it runs no Python. (`timedelta.days`: "Number of days.")
        descriptor_doc = (
            getattr(raw, "__doc__", None)
            if c_storage
            else inspect.getattr_static(raw, "__doc__", None)
        )
        # Dedented and indented under the member line like every other member's: raw, a
        # multi-line one kept its source margin before 3.13 and fell to column 0 from 3.13.
        descriptor_doc = (
            _indent_continuation(inspect.cleandoc(descriptor_doc), "    ")
            if isinstance(descriptor_doc, str) and descriptor_doc.strip()
            else None
        )
        if c_storage:
            # C-level storage: the value itself is the only thing that says what it holds,
            # and reading it runs no Python. One of the two paths that still uses getattr.
            if not is_class:
                value, ok = _safe_getattr(obj, attr_name)
                if ok:
                    doc = f": {descriptor_doc}" if descriptor_doc else ""
                    return f"  - `{attr_name}` [{type(value).__name__}]{doc}"
                return f"  - `{attr_name}` (unreadable)"
        doc = f": {descriptor_doc}" if descriptor_doc else ""
        if attr_name in annotations:
            return f"  - `{attr_name}` [{_annotated_type_name(annotations[attr_name])}]{doc}"
        return f"  - `{attr_name}`{doc}"

    if attr_name in own:
        # Checked AFTER the descriptor branches, because that is Python's own order: a data
        # descriptor beats the instance dict, and `getattr_static` has already applied that
        # — so reaching here means the instance dict really does win. It is a stored VALUE,
        # not a class-body definition, so it is not bound: `self.handler = some_function`
        # would otherwise render `handler(q)`, implying `obj.handler(5)` works when it
        # raises for want of `self`.
        return _value_member_line(attr_name, raw)

    bindable = (
        inspect.isfunction(raw)
        or isinstance(raw, _BINDABLE_C_KINDS)
        or (
            isinstance(raw, (staticmethod, classmethod))
            and inspect.isfunction(raw.__func__)
        )
        or isinstance(raw, functools.partialmethod)
    )
    if bindable:
        # `__get__(None, cls)` on a class input, not `__get__(cls, type(cls))`: the latter
        # strips `self` from every instance method — advertising `W.method(x)`, which raises
        # — and binds a classmethod to the METAclass rather than to the class.
        # Called explicitly rather than through `getattr`, which would route via
        # `type(obj).__getattribute__` and so run a user-defined one.
        descriptor_get = typing.cast(
            "Callable[[object, object | None, type | None], object]", type(raw).__get__
        )
        bound = (
            descriptor_get(raw, None, obj)
            if is_class
            else descriptor_get(raw, obj, type(obj))
        )
        return _callable_member_line(attr_name, bound)

    if isinstance(raw, functools.singledispatchmethod):
        # Bound like a method, but `__get__` returns an `update_wrapper`-ed closure whose
        # `__wrapped__` points back at the UNDECORATED function, so `inspect.signature`
        # follows it and the receiver survives — the render would advertise
        # `obj.handle(self, arg)`, which raises, while `obj.handle(arg)` is the real call.
        # Read the underlying function directly and drop the receiver by hand instead.
        func = raw.func
        doc = _indent_continuation(_static_doc(func), "    ")
        sig = inspect.signature(func)
        rendered = _drop_first_param(sig) if not is_class else _render_signature(sig)
        return f"  - `{attr_name}{rendered}`: {doc}"

    if isinstance(raw, (staticmethod, classmethod)):
        # `__func__` is not a Python function — `staticmethod(len)`, a `partial`, a callable
        # object. `__get__` is still C, so binding is safe, and the result carries whatever
        # signature and docstring the wrapped callable has; the alternative is a bare
        # `[staticmethod]` with both thrown away.
        return _callable_member_line(
            attr_name, raw.__get__(None, obj if is_class else type(obj))
        )

    if isinstance(raw, functools.partial):
        # 3.13 gave `functools.partial` a `__get__` and 3.14 made it actually bind, so from
        # 3.13 a partial stored in a class body reaches the non-data-descriptor branch below
        # and rendered as a bare `[partial]`. Render it as the callable it is instead: the
        # residual signature and the wrapped function's docstring, which is what `Cls.name(...)`
        # takes on every version. (Binding it is deliberately NOT done — on 3.14 `obj.name(x)`
        # passes `obj` through to the wrapped function and usually raises, which is the
        # behaviour change that `FutureWarning` on 3.13 is warning about; a caller who wants
        # the old meaning wraps it in `staticmethod`.)
        return _value_member_line(attr_name, raw)

    if hasattr(type(raw), "__get__"):
        # A non-data descriptor of the user's own making — pandas' `CachedAccessor` is the
        # one that mattered. Its `__get__` is user code, so it renders unbound.
        return f"  - `{attr_name}` [{type(raw).__name__}]"

    return _value_member_line(attr_name, raw)


def _member_names(obj: object, cls: type, annotations: dict[str, object]) -> list[str]:
    """Every public name the walk can see, from all four places one can hide."""
    names: set[str] = set(getattr(obj, "__dict__", ())) if obj is not cls else set()
    for base in cls.__mro__:
        if base is object:
            continue
        # `vars(base)` alone: CPython puts a `member_descriptor` in the class dict for every
        # `__slots__` entry, so reading `__slots__` would add nothing — while firing a
        # metaclass `__getattr__`, and iterating `__slots__ = "name"` character-wise into
        # four phantom members.
        names.update(vars(base))
    names.update(annotations)
    return sorted(n for n in names if not n.startswith("_"))


def _own_annotations(base: type | types.ModuleType) -> dict[str, object]:
    """``base``'s OWN declared annotations — never a base class's.

    Before 3.14 these are a plain dict in the class namespace. PEP 649 replaced it with a
    lazily-evaluated `__annotate__` function, so from 3.14 `vars(base)` has no
    `__annotations__` at all and every annotated field silently vanished from the walk.

    `FORWARDREF` rather than `VALUE`: it tolerates a name the annotation cannot resolve
    (returning a `ForwardRef` instead of raising) and yields the same objects the pre-3.14
    dict held, so the rendering is identical across versions. It is NOT cheaper in terms of
    user code — measured on 3.14, every format including `STRING` evaluates the annotation
    expressions at read time — which is the one place this module cannot keep its "classifying
    a member runs nothing of yours" property on 3.14, for a class and for a module alike. There
    is no format that avoids it.
    """
    if annotationlib is not None:
        try:
            return dict(
                annotationlib.get_annotations(
                    base, format=annotationlib.Format.FORWARDREF
                )
            )
        except Exception:  # noqa: BLE001 — an annotation expression is user code
            return {}
    # vars(), not getattr: `cls.__annotations__` fires a metaclass `__getattr__`, and on a
    # class with no annotations of its own it silently returns a BASE's dict, which would
    # attribute inherited fields to the wrong class.
    #
    # A dict only, as `annotationlib.get_annotations` accepts on 3.14 (it rejects anything
    # else, which the except above turns into `{}`), so both paths agree. The case that
    # needs it: on `type` itself, `vars(type)["__annotations__"]` is the getset descriptor
    # that serves every class's annotations, and every metaclass has `type` in its MRO --
    # rendering `abc.ABCMeta` or any user metaclass would crash iterating it.
    own = vars(base).get("__annotations__")
    return own if isinstance(own, dict) else {}


def _walk_members(obj: object, cls: type) -> list[str]:
    """One line per member the value itself defines, in full; then, per base class, the names of
    the members inherited from it."""
    # By executive call (user, 2026-10-04 and -05): what the value's own class adds is the part the
    # agent cannot know, so it renders in full. An inherited member is NAMED, not dropped and not
    # described: a base from a framework absent from the model's training data is as unknown as
    # the subclass, whether a base is well known cannot be decided here, and a name is what the
    # agent needs to look the member up with `help()` in its REPL.
    #
    # Names, over a summary per member, was measured three ways before deciding, on 13 surveyed
    # objects and FLE's six pydantic models: summaries (signature and first sentence) totalled
    # 86,629 and 35,477 characters, names 60,027 and 5,510, omitting inherited members 55,692 and
    # 2,762. Names cost little more than omitting and keep every inherited member findable --
    # pyparsing's `parse_string`, the method the library is used through, is inherited -- while
    # summaries cost ~5,000 more characters per framework subclass. Bounding size further is #1601.
    #
    # A class's own annotations are its own members even when absent from `vars(cls)`: pydantic
    # removes its fields from the class namespace, so without them a model's fields would be
    # filed as inherited, under no base.
    #
    # Each base's annotations are read ONCE per walk and shared: on 3.14 reading them evaluates
    # their expressions (`_own_annotations` says why), and one that cannot resolve is not cached,
    # so reading per inherited name re-ran user code once per name.
    #
    # A value a base stores in `__slots__` is instance state like one in `__dict__`, so it is
    # the instance's own: a `@dataclass(slots=True)` base must not hide what a plain one shows.
    per_base = {
        base: _own_annotations(base) for base in cls.__mro__ if base is not object
    }
    annotations: dict[str, object] = {}
    for base in reversed(cls.__mro__):  # base-first, so a subclass's narrowing wins
        annotations.update(per_base.get(base, {}))
    instance = getattr(obj, "__dict__", ()) if obj is not cls else ()
    lines: list[str] = []
    inherited: dict[type, list[str]] = {}
    for n in _member_names(obj, cls, annotations):
        if _is_deprecated_member(obj, n):
            continue
        base = _defining_base(cls, n, per_base)
        if (
            n in instance
            or base is cls
            or (obj is not cls and inspect.ismemberdescriptor(vars(base).get(n)))
        ):
            lines.append(_render_member(obj, n, annotations))
        else:
            inherited.setdefault(base, []).append(n)
    for base in cls.__mro__:
        if base in inherited:
            lines.append(
                _named_group(f"inherited from `{_qualified(base)}`", inherited[base])
            )
    return lines


def _is_deprecated(raw: object) -> bool:
    """Whether ``raw`` carries PEP 702's deprecation marker, read without running user code."""
    # The marker sits on a function or a class; anything else is followed to the function it
    # wraps -- a `staticmethod`/`classmethod`'s `__func__`, a property's getter, a
    # `cached_property`'s, `partialmethod`'s, `singledispatchmethod`'s or `partial`'s `func` --
    # since `@deprecated` is applied to the function and the wrapper goes on outside it.
    #
    # EXACT types, not `isinstance`: an object can claim any `__class__` to `isinstance`, and a
    # wrapper subclass could answer `.func` through its own `__getattr__`; either would run its
    # code. On the exact stdlib wrappers a plain attribute read runs nothing of the user's (and
    # `getattr_static` would return the C slot's descriptor, not the function in it). A
    # function's marker is read from its own dict.
    wrappers: dict[type, str] = {
        staticmethod: "__func__",
        classmethod: "__func__",
        property: "fget",
        functools.cached_property: "func",
        functools.partialmethod: "func",
        functools.singledispatchmethod: "func",
        functools.partial: "func",
    }
    # Bounded: a property of a cached_property of ... must not loop forever.
    for _ in range(8):
        attr = wrappers.get(type(raw))
        if attr is None:
            break
        raw = getattr(raw, attr, None)
    if type(raw) is types.FunctionType:
        marker = vars(raw).get("__deprecated__")
    elif issubclass(type(raw), type):  # not `isinstance`: it falls back to `__class__`
        marker = vars(raw).get("__deprecated__")
    else:
        return False
    return isinstance(marker, str)


def _is_deprecated_member(obj: object, name: str) -> bool:
    """Whether the member ``name`` of ``obj``, read statically, is marked deprecated."""
    try:
        raw = inspect.getattr_static(obj, name)
    except AttributeError:
        return False
    return _is_deprecated(raw)


def _defining_base(
    cls: type, name: str, per_base: dict[type, dict[str, object]]
) -> type:
    """The first class in ``cls``'s MRO that defines or annotates ``name``, else ``cls``.

    ``cls`` too for a name only the instance has, which is the instance's own.
    """
    for base in cls.__mro__:
        if name in vars(base) or name in per_base.get(base, {}):
            return base
    return cls


def _qualified(cls: type) -> str:
    """``module.QualName``, without the module for a builtin."""
    # Through `type`'s own descriptors, which no metaclass `__getattribute__` intercepts --
    # `cls.__qualname__` goes through it -- and which also give a C class its module, where
    # `vars(cls)` has none (`collections.OrderedDict`, `numpy.ndarray`).
    qualname = type.__dict__["__qualname__"].__get__(cls)
    module = type.__dict__["__module__"].__get__(cls)
    if not isinstance(module, str) or module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def _named_group(label: str, names: list[str]) -> str:
    """One member line naming several members: ``  - <label>: a, b, c``."""
    return f"  - {label}: " + ", ".join(names)


def _format_class(name: str | None, cls: type) -> str:
    # A class always has `__name__`, so the fallback is total here — unlike `_format_object`,
    # where an unnamed value has nothing of its own to fall back to.
    label = name or _own_name(cls)
    # The constructor belongs in the header. A class handed to an agent is almost always
    # something it has to BUILD, and `__init__` is filtered out of the member walk by the
    # leading-underscore rule — so without this the top-level rendering was the one place the
    # constructor never appeared, while the same class nested as someone's attribute showed
    # it in full (`_value_member_line`). That asymmetry was backwards: the input case is the
    # one that needs it most.
    tag = _class_tag(cls)
    head = f"`{label}{_static_signature(cls)}` {tag}" if label else tag
    members = _walk_members(cls, cls)
    lines: list[str] = _head_block(head, _get_doc(cls), "  ", bool(members))
    lines.extend(members)
    return "\n".join(lines)


def _format_object(name: str | None, obj: object) -> str:
    # An instance has no name of its own — `__name__` belongs to its CLASS, and borrowing it
    # would label the value with its type twice. A placeholder like `value` is worse still: it
    # reads as a real binding the agent could reference and is not.
    #
    # Unnamed, it says so in prose rather than dropping to a bare `[Widget]:`. The bracket is
    # a type tag ATTACHED TO A NAME — every other bracket in this rendering sits after a
    # backticked name — so on its own it reads as a fragment, and it needs a convention the
    # reader may not have. (Square brackets are not a type notation in the wider world either;
    # they conventionally mean *optional*. Python writes `x: T`, Google docstrings `x (T):`.)
    # "Instance of Widget" also distinguishes the value from its CLASS, which renders as
    # `` `Widget` [class] `` — a confusion this module already guards against elsewhere.
    #
    # The NAMED form is left alone: it is agent-facing prompt text, and changing it would
    # break reproducibility against recorded eval runs. This branch is not — it is reachable
    # only from the console's `?expr` and a direct `render_value(x)` call.
    cls = type(obj)
    head = f"`{name}` [{cls.__name__}]" if name else f"Instance of {cls.__name__}"
    members = _walk_members(obj, cls)
    lines: list[str] = _head_block(head, _get_doc(cls), "  ", bool(members))
    lines.extend(members)
    return "\n".join(lines)


def _get_signature(obj: Callable[..., object]) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        # `inspect` cannot tell (`getattr`, `dir`, `anext` among builtins): unknown, as
        # `_static_signature` says for a class it cannot read. `()` would claim no arguments.
        return _UNKNOWN_SIGNATURE
    return _render_signature(sig)


_NO_DESCRIPTION = "(no description available)"

# A signature that cannot be read renders as `(...)`, by executive call (user, 2026-10-04): it
# keeps "this is called, with arguments unknown" visible, as pydoc prints `getattr(...)`. `()`
# would claim no arguments, and an empty string (the earlier choice) left a callable member
# looking like a plain value -- `` `ga`: doc `` with no parentheses and no type tag.
_UNKNOWN_SIGNATURE = "(...)"


def _get_doc(obj: type | types.ModuleType | Callable[..., object]) -> str:
    doc = inspect.getdoc(obj)
    if not doc:
        return _NO_DESCRIPTION
    return doc.strip()


def _indent_continuation(text: str, indent: str) -> str:
    """Indent all lines of `text` after the first by `indent`.

    A blank line is left blank rather than indented, so a docstring's paragraph separator
    does not reach the prompt as a line of pure whitespace. Tested with `strip()`, not
    truthiness: `inspect.cleandoc` removes the COMMON margin, so a separator indented past
    it keeps its residual spaces and is not empty.
    """
    lines = text.splitlines()
    if len(lines) <= 1:
        return text
    return (
        lines[0]
        + "\n"
        + "\n".join(indent + line if line.strip() else "" for line in lines[1:])
    )


def _head_block(head: str, doc: str, indent: str, has_body: bool) -> list[str]:
    """The header line, plus the blank that separates it from whatever is listed beneath.

    Continuation lines of the docstring carry the same indent as the entries under it, so on
    a docstring of more than one line the prose and the body meet at the same column. The
    walks prefix each entry with ``- ``, which tells them apart once you are looking for it;
    a blank line is easier to read.

    It is a separator, NOT a unique signal — a docstring that blank-line-separates its own
    paragraphs already contains blanks, and a member's own docstring may contain more. What
    this guarantees is only that the head block and the body never meet without one.

    `has_body` is passed rather than inferred: whether anything follows is the caller's
    knowledge, and a trailing blank on a walk that found nothing would be noise.
    """
    rendered = _indent_continuation(doc, indent)
    lines = [f"{head}: {rendered}"]
    if has_body and "\n" in rendered:
        lines.append("")
    return lines
