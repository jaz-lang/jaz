"""Private helper implementing the public/non-public API boundary.

A package advertises its supported surface via ``__all__``. Several names outside
``__all__`` remain *reachable* (``jaz.Agent``, ``jaz.hooks.HookDispatcher``, the flat hook
vocabulary, ...) for back-compat, but they are unsupported and may
change or be removed. Accessing one of those emits
:class:`jaz.exceptions.NonPublicAPIWarning` so callers learn they're off the supported
path — while the name still resolves.

The warning *category* lives in :mod:`jaz.exceptions`, not here: it is in ``jaz.__all__``,
and the docs generator pages every public symbol by its **defining module**, so a class
defined in this private module would publish a page titled ``jaz._warnings`` — a
leading-underscore module presented as public API. Only the machinery stays private.

Three mechanisms live here, because a demoted name is reachable by three routes and no one
hook sees all of them:

- :func:`make_lazy_getattr` — names the package does **not** bind (``jaz.Agent``).
- :func:`warn_on_nonpublic_submodules` — submodules the import system binds behind the
  package's back (``jaz.agent``, and so ``jaz.agent.Agent``).
- :func:`make_private_module_shim` — the ``from jaz.agent import Agent`` form, whose final
  ``getattr`` lands on the *submodule* object and never touches the package.

The residue is deliberate: ``jaz._agent.Agent`` does not warn. It cannot — that is the
module the shim forwards *to*, so warning there would fire on every internal import. The
leading underscore is the contract; the warning is the courtesy for paths that still look
supported.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Sequence
from types import ModuleType

from .exceptions import NonPublicAPIWarning


def _warn_nonpublic(module_name: str, name: str, *, stacklevel: int = 3) -> None:
    """Emit the one canonical off-the-supported-path warning.

    Wording is deliberately "experimental / not yet part of the official public API" rather
    than a flat "unsupported": these names are reachable on purpose and most are expected to
    be promoted or replaced, not deleted, so the message says what a caller actually needs to
    plan around — that it may move — without implying the name is abandoned.
    """
    warnings.warn(
        f"{module_name}.{name} is currently experimental and not yet part of the official "
        f"public API. It is reachable but may change or be removed in the near future.",
        NonPublicAPIWarning,
        stacklevel=stacklevel,
    )


def make_lazy_getattr(
    module_name: str,
    public: list[str],
    demoted: dict[str, tuple[str, str]],
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Build ``(__getattr__, __dir__)`` for a package with a demoted-name boundary.

    ``public`` is the module's ``__all__`` (eagerly bound as real attributes — never
    routed here). ``demoted`` maps each reachable-but-unsupported name to
    ``(defining_module, attribute)``; accessing one warns and then resolves it lazily.
    Everything else raises ``AttributeError`` (normal missing-attribute behavior).

    Lazy (rather than eager ``# noqa: F401`` imports) is what lets the warning fire at
    all: an eagerly-bound attribute is found by normal lookup and never reaches
    ``__getattr__``.

    **Warns on every access, and deliberately does not cache.** The obvious optimization —
    ``setattr(module, name, resolved)`` after the first hit, the usual PEP 562 lazy-import
    idiom — would make the attribute a real module member, so subsequent lookups bypass
    ``__getattr__`` entirely and the *second* call site never learns it is off the
    supported path. Since the point of this machinery is to tell each caller, not to tell
    the process once, we re-warn. Python's default warning filter still collapses the
    output to one line per call site, so the practical noise is the same; only
    ``-W always`` / ``simplefilter("always")`` shows the repeats.

    The cost of not caching is an ``importlib.import_module`` (a ``sys.modules`` dict hit
    after the first import) plus a ``getattr`` per access — negligible next to an LLM
    call, and paid only by code that is already on the unsupported path.
    """

    def __getattr__(name: str) -> object:
        target = demoted.get(name)
        if target is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        _warn_nonpublic(module_name, name)
        mod, attr = target
        return getattr(importlib.import_module(mod), attr)

    def __dir__() -> list[str]:
        # Include demoted names so autocomplete / dir() still surface them.
        return sorted(set(public) | set(demoted))

    return __getattr__, __dir__


class _NonPublicSubmoduleWarner(ModuleType):
    """Module type that warns when a non-public **submodule** is read off the package.

    ``__getattr__`` (PEP 562) cannot cover submodules: importing ``jaz.agent`` anywhere —
    including from inside the package, as ``invoke.py`` does — makes the import system bind
    ``agent`` as a real attribute of the ``jaz`` module object, so the lookup succeeds by
    normal means and never reaches ``__getattr__``. That is why ``jaz.agent`` was silent
    while ``jaz.Agent`` warned, even though both are equally off the supported path.

    Intercepting ``__getattribute__`` is the only hook that sees an *already-bound*
    attribute. It is scoped as tightly as possible: it warns only when the resolved value is
    a module belonging to this package and its name is not in ``__all__``. Everything else —
    every public name, every demoted non-module name (still ``__getattr__``'s job), every
    dunder — takes one dict membership test and falls straight through.

    Cost is confined to attribute access *on the package object* (``jaz.x``). Code inside the
    package reads its own globals directly and never goes through here, so the hot paths are
    untouched.
    """

    _public: frozenset[str]

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if name.startswith("_") or name in super().__getattribute__("_public"):
            return value
        if isinstance(value, ModuleType):
            package = super().__getattribute__("__name__")
            if value.__name__.startswith(f"{package}."):
                _warn_nonpublic(package, name)
        return value


def make_private_module_shim(
    module_name: str,
    private_module: str,
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Build ``(__getattr__, __dir__)`` for a public-*named* module that is entirely internal.

    The definitions live in ``private_module`` (a ``_``-prefixed sibling) and the
    public-named module is left as a pure forwarding shim, so every name read off it warns
    and then resolves.

    This closes the one path the package-level machinery structurally cannot reach:
    ``from jaz.agent import Agent``. :class:`_NonPublicSubmoduleWarner` covers attribute
    reads on the *package* (``jaz.agent``, and therefore ``jaz.agent.Agent``), but a
    ``from`` import does its final ``getattr`` on the **submodule** object, never touching
    ``jaz``. And PEP 562's ``__getattr__`` fires only for names *absent* from a module's
    namespace, so while ``Agent`` was defined in ``jaz/agent.py`` no hook installed there
    could observe it either. Relocating the definition is what makes the name absent, and
    therefore observable.

    Forwards by lookup rather than an explicit name list: the shim then cannot drift from
    what the module actually defines, and it reproduces the old module's surface exactly —
    including re-exported imports, which ``from jaz.agent import Continue`` relied on.

    Note the compound path ``jaz.agent.Agent`` now warns **twice** — once for reaching the
    non-public submodule, once for the non-public name on it. They are genuinely two
    events, and the default warning filter collapses each to one line per call site, so the
    duplication costs a line rather than a flood.
    """

    # WHICH modules get a shim: the criterion is "the module holds a `_DEMOTED` target", not
    # the broader "the module holds only non-public names". The four shimmed modules
    # (`agent`, `catalog`, `llm_client`, `parent_output`) each define something reachable as
    # `jaz.<name>`, so the `from jaz.<module> import <name>` form is a second route to a name
    # the package already warns about — closing one while leaving the other open is the gap
    # this exists for.
    #
    # `jaz.llm_client_rlm` reads like a fifth candidate under the broader rule (one name,
    # `RLMClient`, not in `__all__`) and is deliberately excluded: `RLMClient` is not in
    # `_DEMOTED`, so `jaz.RLMClient` was never a route and there is no already-warning path to
    # keep consistent. Shimming it would also force `jaz/__init__.py` to import a module gated
    # behind the optional `rlm` extra just to bind the submodule attribute — paying an import
    # cost, on every caller, for a name no one was told they could use. The consequence is
    # that `from jaz.llm_client_rlm import RLMClient` stays silent; it is a private module by
    # convention only.
    #
    # Reviewed and left as-is deliberately (#994): `RLMClient` is slated for removal, so
    # wiring a warning onto a route nobody was told about buys one release of signal on a
    # module that is going away. Revisit only if it outlives that plan.

    def __getattr__(name: str) -> object:
        # Leading underscore: the shim forwards the module's public surface only. Dunders in
        # particular must NOT be forwarded — letting `__path__` or `__all__` resolve to the
        # private module's would confuse the import system about which module this is.
        if name.startswith("_"):
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        try:
            value = getattr(importlib.import_module(private_module), name)
        except AttributeError:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from None
        _warn_nonpublic(module_name, name)
        return value

    def __dir__() -> list[str]:
        return sorted(
            n
            for n in dir(importlib.import_module(private_module))
            if not n.startswith("_")
        )

    return __getattr__, __dir__


def warn_on_nonpublic_submodules(module: ModuleType, public: Sequence[str]) -> None:
    """Make ``module``'s non-public submodules warn on access, like its demoted names do.

    Call from a package ``__init__`` that already installed :func:`make_lazy_getattr`; the
    two are complements — that one covers names the package *doesn't* bind, this one covers
    submodules the import system binds behind its back.
    """
    module._public = frozenset(public)  # type: ignore[attr-defined]
    module.__class__ = _NonPublicSubmoduleWarner
