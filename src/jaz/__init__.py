"""JAZ — build and optimize LLM agents that execute in a REPL loop.

An agent runs as a conversation with a Python REPL: the LLM writes code, sees what it
evaluated to, and iterates until it finishes with a native ``return`` (or ``raise``)::

    from jaz import invoke
    from jaz.hooks import ReturnType

    total = invoke(instruction="Return the sum of the numbers", numbers=[1, 2, 3])
    num_people: int = invoke(ReturnType(int), question="How many people are there on Earth?")

The v1 public API is exactly ``__all__`` below, in two parts.

**Core symbols** — reached flat as ``jaz.<name>``:

- :func:`invoke` / :func:`ainvoke` — run an agent on the given inputs and return its result.
- :func:`configure` / :class:`ConfigOverride` / :func:`get_config` — set the global
  defaults, override them for a scope or a single call, and read the effective config.
- :func:`describe` — attach a description to a value; it renders in the prompt header
  instead of the auto-stringified value and travels into nested invokes.
- :func:`scope` — bind variables that propagate into sub-invokes.
- :class:`NonPublicAPIWarning` — warned on access to a demoted name (see below).

**Public submodules** — reached as ``jaz.<mod>.<name>``, not re-exported flat:

- ``jaz.hooks`` — the :class:`Hook` primitive and the built-in hooks
  (:class:`ReturnType`, :class:`PrintLogger`, :class:`BudgetPool`,
  :class:`IterationLimit`, ...), plus the ``jaz.hooks.events`` and ``jaz.hooks.effects``
  vocabularies a handler receives and returns.
- ``jaz.repl`` — the REPL component: the base :class:`BaseREPL` and default
  :class:`PythonREPL`, plus the execution-result taxonomy (:class:`ExecResult` and its members).
- ``jaz.llm`` — the LLM-backend component: the base :class:`BaseLLM` and the default
  :class:`LiteLLM` backend.
- ``jaz.protocol`` — the interaction-protocol component: the base :class:`BaseProtocol`
  and default :class:`CodeOnlyProtocol`.
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
#   ``jaz.display``, ``jaz.library``, ``jaz.llm_client``, ``jaz.parent_output``) are
#   forwarding shims; the definitions live in ``_``-prefixed siblings. Holding a
#   ``_DEMOTED`` target — not merely "holds no public name" — is the criterion: a shim
#   exists to close the second route to a name ``jaz.<name>`` already warns about.
# - Two silent paths remain, each signposted by something other than a warning: importing
#   straight from a ``_``-prefixed module (``jaz._agent`` / ``jaz._library`` — the
#   underscore is the contract); and a ``from`` import of a demoted name out of a module that
#   *also* holds public ones (``from jaz.config import configure_by_depth``; ``jaz.config``
#   cannot become a shim while ``configure`` is defined there).
# - The supported move in every case is a name in ``__all__``.

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
    credentials,  # public credentials store under jaz.credentials — in __all__
    display,  # legacy-path shim over jaz._display — NOT in __all__
    exceptions,  # public exception taxonomy under jaz.exceptions — in __all__
    hooks,  # public hook API (ReturnType, Validate*, loggers, budgets, ...) — in __all__
    library,  # legacy-path shim over jaz._library (experimental Library) — NOT in __all__
    llm,  # public LLM component under jaz.llm — in __all__ (also registers built-in backends)
    llm_client,  # legacy-path shim over jaz._llm_client — NOT in __all__
    parent_output,  # legacy-path shim over jaz._parent_output — NOT in __all__
    protocol,  # public protocol component under jaz.protocol — in __all__ (also registers built-in protocols)
    repl,  # public REPL component under jaz.repl — in __all__ (also registers built-in REPLs)
)
from ._warnings import make_lazy_getattr, warn_on_nonpublic_submodules
from .config import ConfigOverride, configure, get_config
from .descriptions import describe
from .exceptions import NonPublicAPIWarning
from .invoke import ainvoke, invoke
from .scope import scope

if TYPE_CHECKING:
    # Static types for the reachable-but-demoted names (resolved lazily at runtime via
    # __getattr__, which warns). Not executed at runtime — no eager binding — so the
    # warning still fires; type checkers read this block so existing callers type-check.
    from ._agent import Agent as Agent
    from ._catalog import Catalog as Catalog
    from ._catalog import as_catalog as as_catalog
    from ._display import Display as Display
    from ._library import Library as Library
    from ._llm_client import LLMResponse as LLMResponse
    from ._llm_client import MockLLMClient as MockLLMClient
    from ._parent_output import parent_print as parent_print
    from .config import ConfigOverrideByDepth as ConfigOverrideByDepth
    from .config import configure_by_depth as configure_by_depth
    from .descriptions import DescriptionOverride as DescriptionOverride
    from .descriptions import get_description as get_description
    from .hooks import ReturnType as ReturnType
    from .hooks import ValidateREPLCode as ValidateREPLCode
    from .hooks import ValidateREPLInput as ValidateREPLInput  # deprecated alias
    from .hooks import ValidateReturn as ValidateReturn
    from .scope import current_scope as current_scope
    from .scope import get_scope as get_scope

__version__ = "0.2.0a3"

# The v1 public API — the documented, supported surface (API-surface review,
# 2026-07-26). `__all__` is the single source of truth for several things at once:
# what `from jaz import *` binds, what the docs generator documents
# (scripts/gen_api_docs.py reads exactly this list), and — by convention — what the
# first release commits to keeping stable. Keep it TIGHT: every name here is a
# compatibility promise.
#
# It has two kinds of entry:
#   1. Core symbols — functions/classes imported flat onto `jaz` (use `jaz.invoke`).
#   2. Public submodules — `hooks`, `repl`, `llm`, `protocol`, `exceptions`, `credentials`.
#      Listing a *submodule* in a package's `__all__` is the standard way to advertise it as
#      public (see the Python tutorial on packages): `__all__`-aware tooling then treats it
#      and its own `__all__` as the public API of that namespace, reached as `jaz.hooks.X` /
#      `jaz.repl.X` / `jaz.exceptions.X` / `jaz.credentials.X` (NOT re-exported flat onto
#      `jaz`). This is why the docs generator documents them automatically — no separate
#      allowlist; to publish another submodule, add its name here. `llm` and `protocol` are
#      *also* imported above for their backend/protocol registration side effects, which fire
#      on import regardless.
#
# Names NOT in this list (Agent, the per-depth config machinery, the LLM seams,
# DescriptionOverride, Display, Catalog/as_catalog, get_scope (+ its current_scope alias),
# parent_print, Library — experimental in v1, shimmed via `jaz.library` over `jaz._library` — and the
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
    "describe",
    "scope",
    "NonPublicAPIWarning",
    "__version__",
    # Public submodules — reached as `jaz.hooks.X` / `jaz.repl.X` / `jaz.llm.X` /
    # `jaz.protocol.X` / `jaz.exceptions.X` / `jaz.credentials.X`. The three
    # pluggable-component packages (repl, llm, protocol) are uniformly public: each exports
    # its base class + default concrete (BaseREPL/PythonREPL, BaseLLM/LiteLLM,
    # BaseProtocol/CodeOnlyProtocol).
    "hooks",
    "repl",
    "llm",
    "protocol",
    "exceptions",
    # EXECUTIVE CALL (user, 2026-08-15) — `credentials` is public because of what it
    # READS, not what it writes. The governing rule, settled after weighing #1076's
    # "unified vs split SDK/CLI credential model": **library code reads ambient
    # credentials; only the CLI writes them.** Reading is legitimate SDK surface — a
    # `@register_llm` backend author needs `resolve_credential` to opt its tag into the
    # store — so the submodule is advertised here and its own `__all__` publishes the
    # three readers. Writing is an administrative act that permanently changes the
    # machine, so the writer is `_set_credential`: reachable for the console, which is
    # JAZ's CLI, and off the supported path for a script.
    #
    # Why the asymmetry is the right one rather than "publish both". Credentials are
    # ambient *authority*, not ambient *behavior* — which is what licenses a library read
    # at all, and why `jaz.user_settings` is console-scoped while this file is read on the
    # LLM call path (the same code with a different key does the same thing; the same code
    # with different settings does not). But that argument only ever
    # covered reads. The convention for writes is near-universal in the other direction:
    # boto3, google-auth and azure-identity read their ambient credential files and never
    # write them (`aws configure` / `gcloud auth` do), and Claude Code writes only via
    # `/login`, with `claude setup-token` deliberately PRINTING a token for scripts rather
    # than persisting it. The tools that do ship an SDK writer — huggingface_hub.login,
    # wandb.login, comet_ml.login — are notebook-first, where the setup step happens
    # inside the session because there is nowhere else for it to happen. JAZ is not that:
    # the README's documented path is `import jaz` in a script with the key in the
    # environment. If JAZ ever targets notebooks, this decision flips (that is exactly why
    # huggingface_hub ships `notebook_login()`), and #1076 is where it gets revisited.
    #
    # The rejected alternative was promoting `set_credential` too, on the grounds that the
    # console binds it anyway so the privacy is a fiction. It is not a fiction: it is the
    # same category `jprint` and the `>`/`<-`/`?`/`%` sigils already occupy — supported,
    # documented and stable as *console* affordances, never importable API.
    #
    # What the underscore buys is CONVENTION, not enforcement, and it is worth being exact
    # about that: `jaz.credentials._set_credential` emits no warning. It cannot — `credentials`
    # is in this list, so the submodule access is clean, and PEP 562 `__getattr__` only fires
    # for names ABSENT from a module, which a defined `_set_credential` is not. The demotion
    # machinery in `_warnings.py` works at submodule granularity, so it has no way to warn on
    # one name inside a public module; leaving `credentials` out of `__all__` to get a warning
    # would demote `resolve_credential` with it, which backends need. Adding a `__getattr__`
    # shim purely to warn on an underscore name was rejected as fighting the existing design —
    # `_warnings.py` is explicit that the underscore IS the contract and the warning is only a
    # courtesy for paths that still look supported. So what this buys is the ordinary Python
    # contract: excluded from `import *` and the generated docs, flagged by linters as private
    # access, and unambiguous to a reader. That is weaker than a runtime warning but it is the
    # same guarantee every other `_`-prefixed name in this package carries.
    "credentials",
]

# Reachable-but-unsupported names: {name: (defining_module, attribute)}. Accessing any of
# these via `jaz.<name>` / `from jaz import <name>` warns (NonPublicAPIWarning) and then
# resolves lazily. Their blessed homes are noted per entry.
_DEMOTED = {
    "Agent": ("jaz._agent", "Agent"),
    # Experimental in v1, and shimmed like `Agent`: the definition lives in `jaz._library`
    # and `jaz.library` is a forwarding shim, so EVERY public-named route warns — `jaz.Library`,
    # `from jaz.library import Library`, `jaz.library`, `jaz.library.Library`. The only silent
    # path is `from jaz._library import Library` (the underscore is the contract).
    "Library": ("jaz._library", "Library"),
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
    "ValidateREPLCode": ("jaz.hooks", "ValidateREPLCode"),
    # deprecated alias of ValidateREPLCode
    "ValidateREPLInput": ("jaz.hooks", "ValidateREPLCode"),
    # Base classes are reached via their (now uniformly public) package — jaz.llm.BaseLLM,
    # jaz.protocol.BaseProtocol, jaz.repl.BaseREPL — so they are NOT demoted top-level aliases
    # of `jaz` (BaseREPL never was one; this makes LLM/Protocol match it).
    "LLMResponse": ("jaz._llm_client", "LLMResponse"),
    "MockLLMClient": ("jaz._llm_client", "MockLLMClient"),
    "parent_print": ("jaz._parent_output", "parent_print"),
    "get_scope": ("jaz.scope", "get_scope"),
    "current_scope": ("jaz.scope", "current_scope"),  # deprecated alias of get_scope
}

__getattr__, __dir__ = make_lazy_getattr(__name__, __all__, _DEMOTED)

# Submodules need the second half of the machinery: `jaz.agent`, `jaz.config`,
# `jaz.protocol`, ... are bound as real attributes by the import system the moment anything
# imports them (and `invoke.py` imports `.agent` on the first call), so they never reach the
# `__getattr__` above. `hooks` / `repl` / `llm` / `protocol` / `exceptions` are in `__all__`
# and stay silent.
import sys as _sys  # noqa: E402

warn_on_nonpublic_submodules(_sys.modules[__name__], __all__)
