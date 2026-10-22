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
- :func:`scope` — bind variables that propagate into sub-invokes.

**Public submodules** — reached as ``jaz.<mod>.<name>``, not re-exported flat:

- ``jaz.hooks`` — the :class:`Hook` primitive and the built-in hooks
  (:class:`ReturnType`, :class:`PrintLogger`, :class:`BudgetPool`,
  :class:`IterationLimit`, ...), plus the ``jaz.hooks.events`` and ``jaz.hooks.effects``
  vocabularies a handler receives and returns.
- ``jaz.repl`` — the REPL component: the base :class:`BaseREPL` and default
  :class:`PythonREPL`, plus the execution-result taxonomy (:class:`ExecResult` and its members).
- ``jaz.llm`` — the LLM-backend component: the base :class:`BaseLLM`, the :class:`LLMResponse`
  it returns, the three shipped backends (:class:`LiteLLM` by default, plus
  :class:`GitHubCopilotLLM` and :class:`SGLangLLM`), and the chat-format seam.
- ``jaz.protocol`` — the interaction-protocol component: the base :class:`BaseProtocol`
  and default :class:`CodeOnlyProtocol`.
- ``jaz.exceptions`` — the exception taxonomy every JAZ failure belongs to.

Everything outside ``__all__`` lives in a ``_``-prefixed module and is not part of the
official public API. It is reachable — Python has no enforcement — but it is internal, and
may change or be removed without notice.
"""

# Every module that is not in `__all__` carries a leading underscore, so the package layout
# states the boundary by itself: `jaz._config`, `jaz._invoke`, `jaz.repl._python_repl`. There
# is deliberately no runtime warning on private access: a `__getattr__` guard on the package
# attribute can only ever be a courtesy, and it has a gap an underscore does not — `from
# jaz._config import ConfigStack` never touches the attribute a guard hooks, so it would stay
# silent while `jaz._config.ConfigStack` warned. An underscore needs no code.

# Imported for the binding side effect: a submodule is an attribute of its package only
# because something imported it, and these are the public ones `jaz.<mod>` must resolve.
from . import (  # noqa: F401
    credentials,
    exceptions,
    hooks,
    llm,
    protocol,
    repl,
)
from ._config import Config, ConfigOverride, configure, get_config
from ._invoke import ainvoke, invoke
from ._scope import scope
from ._sentinel import UNSET, UnsetType
from ._template_loader import add_template_loader

__version__ = "0.2.0b1"

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
#      allowlist; to publish another submodule, add its name here.
#
# Names NOT in this list — `Agent`, `get_scope`, and the `ReturnType`/`Validate*` hooks (whose
# blessed path is `jaz.hooks.<name>`) — are not bound here at all. `Agent` and `get_scope` live
# in `_`-prefixed modules, which is the whole statement: no top-level alias, no lazy
# `__getattr__`, no warning. Promote a name into `__all__` only when it earns a place in the API.
__all__ = [
    # Core symbols — import directly from `jaz`.
    "invoke",
    "ainvoke",
    "configure",
    "ConfigOverride",
    # Public and staying that way: retiring the *internal* ambient config reads (#826) never
    # included this public call shape — retiring `get_config()` itself is not on the table, so
    # its place in the v1 contract isn't provisional.
    "get_config",
    # The object `get_config()` returns and every hook is handed as `event.config`. Public by
    # the same rule as `LLMResponse` and the execution-result taxonomy: a caller is handed one,
    # so a hook author annotating `event.config` needs a name for it — and the only one that
    # existed was `jaz._config.Config`. Its surface is the three component fields, each an
    # already-public ABC, and no public mutators (EXECUTIVE CALL, user, 2026-09-13): in-place
    # config mutation from a hook is explicitly unsupported and `ConfigOverride` is the
    # sanctioned path.
    "Config",
    # The framework-wide "no value supplied" sentinel and its type, at the package root —
    # the convention every library that ships one follows: `openai.NOT_GIVEN`/`NotGiven`,
    # `attrs.NOTHING`, `dataclasses.MISSING`, `httpx.USE_CLIENT_DEFAULT`. Each is exported
    # from the root even when defined in a private module (`attr._make`), and pointedly NOT
    # from the leaf that consumes it — `openai.resources`, whose methods take `NOT_GIVEN`,
    # does not re-export it.
    #
    # It belongs above its consumers because it has more than one: `ModifyLLMResponse` uses it
    # for "leave this field untouched", and `configure()`/`ConfigOverride` for "this group was
    # not passed". Exported only from `jaz.hooks.effects`, a caller doing *config* had to
    # import from a *hooks* module to name the type in `configure`'s own signature.
    #
    # This is the ONLY export. `jaz.hooks.effects` imports it for its own annotations but does
    # not re-export it (EXECUTIVE CALL, user, 2026-09-15): a second path is surface, and before
    # v1 an external import breaking costs nothing, so there is no reason to carry one.
    "UNSET",
    "UnsetType",
    "scope",
    # The supported way to supply your own prompt templates. Public because the three
    # `*_template` settings on `CodeOnlyProtocol` / `PythonREPL` exist for customization and
    # were otherwise unreachable: they name templates, and every route to registering one was
    # non-public. Takes a loader rather than a directory so templates can live in a package,
    # in memory, or anywhere else a `jinja2.BaseLoader` reaches.
    "add_template_loader",
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
    # backend author needs `resolve_credential` to opt its tag into the
    # store — so the submodule is advertised here and its own `__all__` publishes the
    # three readers. Writing is an administrative act that permanently changes the
    # machine, so the writer is `_set_credential`: reachable for the console, which is
    # JAZ's CLI, and off the supported path for a script.
    #
    # Why the asymmetry is the right one rather than "publish both". Credentials are
    # ambient *authority*, not ambient *behavior* — which is what licenses a library read
    # at all, and why `jaz._user_settings` is console-scoped while this file is read on the
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
    # for names ABSENT from a module, which a defined `_set_credential` is not. The underscore
    # boundary works at module granularity, so it has no way to mark one name inside a public
    # module; leaving `credentials` out of `__all__` to make the whole module non-public would
    # demote `resolve_credential` with it, which backends need. Adding a `__getattr__` shim
    # purely to warn on an underscore name was rejected as fighting the existing design — the
    # underscore IS the contract. So what this buys is the ordinary Python
    # contract: excluded from `import *` and the generated docs, flagged by linters as private
    # access, and unambiguous to a reader. That is weaker than a runtime warning but it is the
    # same guarantee every other `_`-prefixed name in this package carries.
    "credentials",
]
