"""JAZ — build and optimize LLM agents that execute in a REPL loop.

An agent runs as a conversation with a Python REPL: the LLM writes code, sees what it
evaluated to, and iterates until it finishes with a native ``return`` (or ``raise``)::

    import jaz
    from jaz.hooks import ReturnType

    total = jaz.invoke(instruction="Return the sum of the numbers", numbers=[1, 2, 3])
    num_people: int = jaz.invoke(ReturnType(int), question="How many people are there on Earth?")

The v1 public API is exactly ``__all__`` below, in two parts.

**Core symbols** — reached flat as ``jaz.<name>``:

- :func:`invoke` / :func:`ainvoke` — run an agent on the given inputs and return its result.
- :func:`configure` / :class:`ConfigOverride` / :func:`get_config` — set the global
  defaults, override them for a scope or a single call, and read the effective config.
- :class:`Library` — define a namespace of typed tools for agents to call.
- :func:`describe` — attach a description to a value; it renders in the prompt header
  instead of the auto-stringified value and travels into nested invokes.
- :func:`scope` — bind variables that propagate into sub-invokes.
- :class:`NonPublicAPIWarning` — warned on access to a demoted name (see below).

**Public submodules** — reached as ``jaz.<mod>.<name>``, not re-exported flat:

- ``jaz.hooks`` — the :class:`Hook` primitive and the built-in hooks
  (:class:`ReturnType`, :class:`PrintLogger`, :class:`BudgetPool`,
  :class:`IterationLimit`, ...), plus the ``jaz.hooks.events`` and ``jaz.hooks.effects``
  vocabularies a handler receives and returns.
- ``jaz.repl`` — the REPL execution-result taxonomy (:class:`ExecResult` and its members) and
  the per-language REPL config types.
- ``jaz.exceptions`` — the exception taxonomy every JAZ failure belongs to.

Names outside ``__all__`` (``Agent``, the ``LLM`` seams, the per-depth config
machinery, ...) are not part of the official public API. They are reachable, but they are
experimental and may change or be removed.
"""

# Reachability mechanics (#994, "Close the from-import gap"). Kept as comments rather than
# in the docstring above: the docstring describes the *API*, and how the demotion is
# enforced is plumbing. Recorded here because the enforcement is non-obvious and the gaps
# in it are deliberate.
#
# - EVERY route to a demoted name warns (``NonPublicAPIWarning``): ``jaz.Agent``, the
#   submodule ``jaz.agent``, and ``from jaz.agent import Agent`` alike.
# - The modules holding a ``_DEMOTED`` target (``jaz.agent``, ``jaz.catalog``,
#   ``jaz.display``, ``jaz.llm_client``, ``jaz.parent_output``) are forwarding shims; the
#   definitions live in ``_``-prefixed siblings. Holding a ``_DEMOTED`` target — not merely
#   "holds no public name" — is the criterion: a shim exists to close the second route to a
#   name ``jaz.<name>`` already warns about (see ``make_private_module_shim`` for why
#   ``jaz.llm_client_rlm`` is excluded despite fitting the broader reading).
# - Three silent paths remain, each signposted by something other than a warning: importing
#   straight from a ``_``-prefixed module (``jaz._agent`` — the underscore is the
#   contract); a ``from`` import of a demoted name out of a module that *also* holds
#   public ones (``from jaz.config import configure_by_depth``; ``jaz.config`` cannot
#   become a shim while ``configure`` is defined there); and ``from jaz.llm_client_rlm
#   import RLMClient``, whose module is private by convention only.
# - The supported move in every case is a name in ``__all__``.

# TODO: revise docstring

# TODO: mark `Library` as experimental in v0.2.
from typing import TYPE_CHECKING

# `agent` / `catalog` / `llm_client` / `parent_output` are the legacy-path SHIMS (see each
# module's docstring) and are imported purely for their binding side effect: a submodule is
# an attribute of its package only because something imported it, and internal code now
# imports the `_`-prefixed definitions instead. Without these, `import jaz; jaz.agent` would
# raise AttributeError rather than warn — breaking a path that is meant to keep working,
# just loudly. Cheap: a shim is a docstring plus a `__getattr__`, and it does not import its
# private sibling until a name is actually read off it.
from . import (  # noqa: F401
    agent,  # legacy-path shim over jaz._agent — NOT in __all__
    catalog,  # legacy-path shim over jaz._catalog — NOT in __all__
    display,  # legacy-path shim over jaz._display — NOT in __all__
    exceptions,  # public exception taxonomy under jaz.exceptions — in __all__
    hooks,  # public hook API (ReturnType, Validate*, loggers, budgets, ...) — in __all__
    llm_client,  # legacy-path shim over jaz._llm_client — NOT in __all__
    parent_output,  # legacy-path shim over jaz._parent_output — NOT in __all__
    protocol,  # imported for interaction-protocol registration side effect
    providers,  # imported for provider registration side effect
    repl,  # public REPL result/config types under jaz.repl — in __all__ (also registers built-in REPLs)
)
from ._warnings import make_lazy_getattr, warn_on_nonpublic_submodules
from .config import ConfigOverride, configure, get_config
from .descriptions import describe
from .exceptions import NonPublicAPIWarning
from .invoke import ainvoke, invoke
from .library import Library
from .scope import scope

if TYPE_CHECKING:
    # Static types for the reachable-but-demoted names (resolved lazily at runtime via
    # __getattr__, which warns). Not executed at runtime — no eager binding — so the
    # warning still fires; type checkers read this block so existing callers type-check.
    from ._agent import Agent as Agent
    from ._catalog import Catalog as Catalog
    from ._catalog import as_catalog as as_catalog
    from ._display import Display as Display
    from ._llm_client import LLM as LLM
    from ._llm_client import LLMResponse as LLMResponse
    from ._llm_client import MockLLMClient as MockLLMClient
    from ._parent_output import parent_print as parent_print
    from .config import ConfigOverrideByDepth as ConfigOverrideByDepth
    from .config import configure_by_depth as configure_by_depth
    from .descriptions import DescriptionOverride as DescriptionOverride
    from .descriptions import get_description as get_description
    from .hooks import ReturnType as ReturnType
    from .hooks import ValidateREPLInput as ValidateREPLInput
    from .hooks import ValidateReturn as ValidateReturn
    from .protocol import InteractionProtocol as InteractionProtocol
    from .scope import current_scope as current_scope

__version__ = "0.2.0a1"

# The v1 public API — the documented, supported surface (API-surface review,
# 2026-07-26). `__all__` is the single source of truth for several things at once:
# what `from jaz import *` binds, what the docs generator documents
# (scripts/gen_api_docs.py reads exactly this list), and — by convention — what the
# first release commits to keeping stable. Keep it TIGHT: every name here is a
# compatibility promise.
#
# It has two kinds of entry:
#   1. Core symbols — functions/classes imported flat onto `jaz` (use `jaz.invoke`).
#   2. Public submodules — `hooks`, `repl`, `exceptions`. Listing a *submodule* in a
#      package's `__all__` is the standard way to advertise it as public (see the Python
#      tutorial on packages): `__all__`-aware tooling then treats it and its own `__all__`
#      as the public API of that namespace, reached as `jaz.hooks.X` / `jaz.repl.X` /
#      `jaz.exceptions.X` (NOT re-exported flat onto `jaz`). This is why the docs
#      generator documents them automatically — no separate allowlist; to publish another
#      submodule, add its name here. `providers` and `protocol` are imported above for
#      registration side effects but deliberately kept OUT of `__all__`.
#
# Names NOT in this list (Agent, the per-depth config machinery, the LLM seams,
# DescriptionOverride, Display, Catalog/as_catalog, current_scope, parent_print, and the
# ReturnType/Validate* hooks — whose blessed path is now `jaz.hooks.ReturnType`) are
# deliberately demoted. They are NOT eagerly bound: they resolve lazily via `__getattr__`
# (see `_DEMOTED`), which emits `NonPublicAPIWarning` on access — reachable for existing
# `jaz.<name>` callers, but unsupported, undocumented, and excluded from `import *`.
# Promote a name into `__all__` only when it earns a place in the API.
__all__ = [
    # Core symbols — import directly from `jaz`.
    "invoke",
    "ainvoke",
    "configure",
    "ConfigOverride",
    # Public and staying that way: #826 ("remove ambient reads") targets the *internal*
    # ambient config reads, not this public call shape — retiring `get_config()` itself
    # is not on the table, so its place in the v1 contract isn't provisional.
    "get_config",
    "Library",
    "describe",
    "scope",
    "NonPublicAPIWarning",
    "__version__",
    # Public submodules — reached as `jaz.hooks.X` / `jaz.repl.X` / `jaz.exceptions.X`.
    "hooks",
    "repl",
    "exceptions",
]

# Reachable-but-unsupported names: {name: (defining_module, attribute)}. Accessing any of
# these via `jaz.<name>` / `from jaz import <name>` warns (NonPublicAPIWarning) and then
# resolves lazily. Their blessed homes are noted per entry.
_DEMOTED = {
    "Agent": ("jaz._agent", "Agent"),
    "Catalog": ("jaz._catalog", "Catalog"),
    "as_catalog": ("jaz._catalog", "as_catalog"),
    "ConfigOverrideByDepth": ("jaz.config", "ConfigOverrideByDepth"),
    "configure_by_depth": ("jaz.config", "configure_by_depth"),
    "DescriptionOverride": ("jaz.descriptions", "DescriptionOverride"),
    "Display": ("jaz._display", "Display"),
    "get_description": ("jaz.descriptions", "get_description"),
    # blessed path: jaz.hooks.<name>
    "ReturnType": ("jaz.hooks", "ReturnType"),
    "ValidateReturn": ("jaz.hooks", "ValidateReturn"),
    "ValidateREPLInput": ("jaz.hooks", "ValidateREPLInput"),
    "LLM": ("jaz._llm_client", "LLM"),
    "LLMResponse": ("jaz._llm_client", "LLMResponse"),
    "MockLLMClient": ("jaz._llm_client", "MockLLMClient"),
    "parent_print": ("jaz._parent_output", "parent_print"),
    "InteractionProtocol": ("jaz.protocol", "InteractionProtocol"),
    "current_scope": ("jaz.scope", "current_scope"),
}

__getattr__, __dir__ = make_lazy_getattr(__name__, __all__, _DEMOTED)

# Submodules need the second half of the machinery: `jaz.agent`, `jaz.config`,
# `jaz.protocol`, ... are bound as real attributes by the import system the moment anything
# imports them (and `invoke.py` imports `.agent` on the first call), so they never reach the
# `__getattr__` above. `hooks` / `repl` / `exceptions` are in `__all__` and stay silent.
import sys as _sys  # noqa: E402

warn_on_nonpublic_submodules(_sys.modules[__name__], __all__)
