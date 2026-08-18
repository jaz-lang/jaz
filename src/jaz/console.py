"""The interactive ``jaz`` console — a real Python REPL with conversational sugar.

After ``pip install jaz`` the ``jaz`` command (registered in ``[project.scripts]``)
drops the user into a Python console that has ``jaz``/``invoke``/``jprint`` pre-wired,
plus a **conversational shorthand** for talking to an agent that is much shorter than a
full ``invoke(...)`` call.

Why this module exists (the problem it replaces)
-------------------------------------------------
The previous interactive surface was the one-off script ``examples/interactive_repl.py``.
It demonstrated the idea but was not a product:

- **Not installed.** No console-script entry point, so ``jaz`` was not on ``PATH`` — a
  user had to know the repo was checked out and run ``python examples/interactive_repl.py``.
- **Hard-coded external deps.** It unconditionally entered ``JaegerTracingHook()``, which
  assumes a Jaeger collector on ``localhost:4318``; a fresh user with no Jaeger got a
  degraded/erroring session out of the box.
- **``invoke(...)`` is verbose for conversational use.** Every turn was
  ``invoke("Examine the `data`", data=data)`` — boilerplate that buries the request and
  forces each input to be repeated as both a backtick reference *and* a kwarg.

Design principles (settled with the user — see design/design_features/interactive_console.md)
---------------------------------------------------------------------------------------------
- **It stays a real Python REPL.** The conversational layer is *additive sugar* over a
  normal ``code.InteractiveConsole`` — arbitrary Python runs, variables persist, ``_``
  holds the last result, history and tab-completion work. Any line that is not a
  recognized sigil is passed through **unchanged** as Python (so the console is never
  ambiguous; there is deliberately no "chat mode" where bare text becomes a prompt).
- **Sugar = rewrite to ``invoke(task=t"...")`` source, reusing the t-string input path.**
  The interpolation -> text + sibling-input normalization already lives in
  :func:`jaz.templates.normalize_inputs`: a t-string passed as the ``task`` input is lowered
  there (``task`` is an ordinary input since #538). Rather than reimplement it, the console
  rewrites a typed line into literal ``invoke(task=t"...")`` *source* and hands it to the
  normal console compiler, so ``{var}`` binding, ``!conversion``/``:format_spec`` rendering,
  name selection and collision handling are inherited with **zero** duplicated logic. This is
  why the console requires the t-string Python floor (PEP 750 t-strings -> **3.14+**).
- **No hooks by default.** Bare ``jaz`` enables nothing that requires an external service
  (fixing the Jaeger footgun); tracing/logging/budgets are opt-in CLI flags.
- **Config via CLI flags + in-session reconfigure.** Startup flags feed ``jaz.configure``;
  because it is a real console, ``jaz.configure(...)`` also works mid-session. Both are
  per-session; defaults that should outlive the session go in ``~/.jaz/settings.json``
  (:mod:`jaz.user_settings`), and API keys in ``~/.jaz/credentials.json`` — written from
  here by :func:`set_credential`, which prompts so the key never reaches the history file.

The sigils
----------
Each recognized form begins with a token that is illegal as the start of a Python
statement, so claiming it as sugar can never shadow valid Python:

===================  ==================================================================
Typed line           Rewrites to
===================  ==================================================================
``> prompt``         ``invoke(task=t"prompt")`` — a *bare expression*, so the stdlib
                     ``single``-mode displayhook both prints the agent's result and binds
                     ``_``. (Un-annotated omits ``ReturnType`` entirely -> no return-type
                     contract, any value comes back (#568); NOT ``ReturnType(None)``, which
                     would force a ``None`` result.)
``target <- prompt`` ``target = invoke(task=t"prompt")``
``t: T <- prompt``   ``t = invoke(ReturnType(T), task=t"prompt")`` (``T`` eval'd in the ns)
``?expr``            ``jprint(expr)`` — print ``expr``'s ``jaz.describe`` description.
``% request``        ``__jaz_settings__("request", globals())`` — a *sandboxed* helper
                     agent for anything jaz: it answers questions (settings, usage, how jaz
                     works — it can read the current config and the jaz source) as a printed
                     ``answer``, and meets action requests (``jaz.configure(...)`` etc.) with
                     a ``snippet`` run only after the user confirms ``Run this? [y/N]``.
                     See :func:`jaz_settings`.
===================  ==================================================================

Known limitations (intentional, v1)
------------------------------------
- **Requires Python 3.14 (PEP 750 t-strings).** Lines are rewritten to literal ``t"..."``
  source, which only compiles on 3.14+, and binding relies on ``normalize_inputs``. Chosen
  deliberately so the console reuses the t-string input path instead of duplicating it;
  installs on 3.12/3.13 cannot ship the console.
- **``<-`` is collision-*minimized*, not collision-*free*.** ``name <- prompt`` is
  technically valid Python (``name < -prompt``), but never a meaningful bare statement, so
  the console claims it. ``>`` and ``?`` *are* fully collision-free.
- **The sigil forms are console-only.** They are not valid Python and do not work in a
  ``.py`` file; the portable form is plain ``invoke(task=t"...")`` / ``jprint(...)``.
- **Literal braces in prose must be doubled.** The rewritten body is a t-string, so ``{``
  and ``}`` mean interpolation. Brace-bearing prose therefore lowers to broken source:
  ``> explain {} in Python`` becomes an empty interpolation, and ``> what does {x} do``
  quietly binds ``x`` as an input the user never meant. Escape literal braces by doubling
  them (``{{``/``}}``). This is inherent to reusing the t-string path — but a lone/empty
  brace that *would* raise a ``SyntaxError`` is now caught and re-raised as a clean
  ``SyntaxError`` (naming the unpaired or empty brace, plus one-line recovery guidance)
  instead of a raw traceback from the synthesized t-string (:func:`_tstring_brace_error`,
  #547); the
  ``{x}``-binds-an-unintended-input case is valid syntax and so stays a documented surprise.
- **Double quotes inside ``{...}`` interpolations are not supported by the sugar.** The
  rewrite escapes ``"`` so the synthesized ``t"..."`` literal stays well-formed, which also
  escapes quotes *inside* an interpolation expression. Use single quotes inside ``{...}``
  (``{d['k']}``) or drop to explicit ``invoke(task=t"...")`` for the rare double-quote case.
- **One-shot mode has no live variables.** ``jaz -c "...{x}"`` has no session namespace,
  so ``{x}`` resolves to an undefined name (interpolation is effectively disabled there).
- **Piped stdin is a *script*, not a prompt.** ``cat script.jaz | jaz`` runs line-by-line
  exactly like ``jaz script.jaz`` (sigils per line; non-sigil lines run as Python). So a
  bare ``echo "capital of France" | jaz`` is a Python ``SyntaxError``, not an agent turn —
  use ``jaz -c "capital of France"`` to send one ad-hoc line to the agent.
- **``%`` is single-line and verbatim.** The settings request has no ``\\``-continuation or
  block form, and no ``{var}`` interpolation — the text after ``%`` is passed to the helper
  as-is (a plain ``str``, not a t-string; that is also why the ``%`` path, unlike the prompt
  sigils, compiles on any interpreter). With stdin not a tty there is nobody to answer the
  confirmation, so a ``%`` request that comes back as a *snippet* prints it but never runs
  it; a request answered in plain language needs no confirmation and prints normally.
  Consequence: whether a given ``%`` line in a script raises (snippet) or exits 0 (answer)
  is the helper model's per-run choice, so the same ``%`` line's exit code is not guaranteed
  reproducible across runs — see :func:`jaz_settings` for why that trade is accepted.
"""

from __future__ import annotations

import argparse
import atexit
import code
import difflib
import getpass
import inspect
import random
import re
import sys
import threading
import time
import warnings
from collections.abc import Mapping
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TextIO

import jaz

# ReturnType comes from jaz.hooks (its public home, listed in jaz.hooks.__all__), NOT
# via the `jaz.ReturnType` attribute: that spelling is absent from jaz.__all__, so every
# access trips the #962 NonPublicAPIWarning — which surfaced as warning noise in the
# user's console session.
from jaz.hooks import Hook, ReturnType
from jaz.paths import ensure_config_dir

from .instantiate import build_config
from .protocol.code_only import CodeOnlyProtocol
from .repl.python_repl import DEFAULT_ALLOWED_ATTRIBUTES, PythonREPL
from .user_settings import (
    SETTINGS_RECOVERY_HINT,
    load_user_settings,
    settings_path,
)

if TYPE_CHECKING:
    from jaz.hooks.effects import Effect
    from jaz.hooks.events import (
        InvokeExit,
        LLMQueryEnter,
        LLMQueryExit,
        LLMQueryRetry,
        REPLExecEnter,
        REPLExecExit,
    )

# A ``target [: T] <- prompt`` capture line. ``target`` must be a bare identifier; an
# optional ``: T`` annotation becomes the return type (a leading ``ReturnType(T)`` hook). We
# intentionally require the ``<-`` arrow (not ``=``): the user asked for a token "clearly not
# Python so there's no
# collision/overloading". ``=`` would silently shadow real assignment and ``<=`` is a valid
# comparison, whereas ``name <- prompt`` only parses in Python as ``name < -prompt`` (a
# discarded boolean), which is never written as a bare statement.
_CAPTURE_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?::\s*(.+?))?\s*<-\s*(.+)$")


# ---------------------------------------------------------------------------
# `?expr` inspect primitive
# ---------------------------------------------------------------------------
def jprint(value: object) -> None:
    """Print ``value``'s attached JAZ description, or a compact ``repr`` if it has none.

    Backs the ``?expr`` inspect sugar (and is pre-wired into the console namespace by
    :func:`build_namespace`). This is a *local* reimplementation of the former
    ``jaz.prompts.jprint`` REPL global, which was removed in #706 when the agent-facing
    ``jprint`` export was dropped for a leaner public API. The console still wants ``?expr``
    to reveal what has been attached to a value via ``jaz.describe`` / ``jaz.Display``, so we
    rebuild the thin wrapper here on the *public* ``jaz.get_description`` lookup rather than
    reviving the removed export or importing ``jaz.prompts`` internals.

    ``jaz.get_description`` returns the rendered description ``str`` when one is attached, else
    ``None`` (or the private ``HIDDEN`` sentinel, whose own contract says non-prompt callers
    should treat it as "no description"). The ``isinstance(..., str)`` check therefore folds
    the ``None`` and ``HIDDEN`` cases together and falls back to ``repr`` without importing the
    sentinel.

    Fidelity trade-off (deliberate): the removed ``jaz.prompts.jprint`` rendered *undescribed*
    values through the prompt builder's private ``_format_input_value_default`` (signature+doc
    for callables/classes, a ``__dict__`` dump for plain objects), so ``?x`` mirrored byte-for-
    byte what the agent sees. Depending on that private helper is exactly the coupling that
    broke this module when the ``jprint`` export moved, so we intentionally give up that
    fidelity on the *undescribed* path for a public-API-only compact ``repr``. The *described*
    path — the actual point of ``?expr`` — stays exact.
    """
    description = jaz.get_description(value)
    print(description if isinstance(description, str) else repr(value))


# ---------------------------------------------------------------------------
# `% request` jaz helper: answer in words, or propose a snippet to confirm and run
# ---------------------------------------------------------------------------
# Née the *settings* helper — successive PRs gave it a config view, the jaz source, and
# finally a general "anything jaz" mission. The internal names (`jaz_settings`,
# `_propose_settings_response`, `_settings_reference`, the "settings" sigil kind) keep the
# original spelling on purpose: they are private, renaming them would churn every test and
# rewrite-path reference for zero behaviour, and settings remain the one *action* surface.
def _strip_code_fences(text: str) -> str:
    """Strip a single markdown code-fence wrapper (```` ```lang ... ``` ````), if present.

    The helper's task says "no markdown fences", but models wrap code in fences often
    enough that silently unwrapping is friendlier than exec'ing a ``SyntaxError``. Only a
    whole-string fence is unwrapped; anything else is returned stripped, verbatim.
    """
    m = re.match(r"^\s*```[\w-]*[ \t]*\n(.*?)\n?[ \t]*```\s*$", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# Curated addendum for the helper's reference document: everything a settings snippet
# might need that the `jaz.configure` / `jaz.ConfigOverride` docstrings do not cover
# (hook-based knobs) or do not make operational enough for reliable generation.
# The non-settings block deliberately pins no how-to phrasing to ('code', ...): "show me
# how to …" is ambiguous between wanting an explanation and wanting to run something, so
# the general question/action split does the routing rather than a per-example steer.
_SETTINGS_ADDENDUM = """\
Additional jaz settings knowledge:

- Config holds CONFIGURED COMPONENTS, one per group: llm=, repl=, protocol=. You pass the
  object itself, and its constructor is the list of that group's settings. Dicts of leaves
  and flat option names (model_config, repl_configs, ...) are REJECTED. Import what you use:
      from jaz.llm import LiteLLM
      from jaz.repl.python_repl import PythonREPL
      from jaz.protocol.code_only import CodeOnlyProtocol
- Setting a group REPLACES it — a component states itself completely, so there is no partial
  update. To change one setting, restate the whole component INCLUDING the values you want to
  keep, reading them off the current config first (see READING below).
- Model: jaz.configure(llm=LiteLLM(model="openai/gpt-5-mini")). LiteLLM's
  constructor takes model (a LiteLLM route like openai/gpt-5-mini), the retry_* settings,
  allowed_openai_params/drop_params, and any per-request default (temperature, max_tokens,
  reasoning_effort, ...) forwarded to LiteLLM. The "provider/" prefix is LiteLLM's routing key:
  it selects the provider within the litellm backend, not the jaz backend — the class does that.
- The agent's Python sandbox is configured by allow-lists on the REPL (gitignore-glob
  patterns): allowed_imports (default [] = deny all; ["*"] = allow all; ["numpy", "pandas"]
  = just those root modules), allowed_attributes, allowed_read_paths, allowed_write_paths,
  and allow_raise, plus exec_timeout / exec_memory_limit:
      jaz.configure(repl=PythonREPL(allowed_imports=["numpy"], exec_timeout=60))
  Because the REPL is replaced, every axis you do not pass returns to its default — restate
  the ones that must survive.
- Scope of a change:
  * session-global: jaz.configure(llm=...) / jaz.configure(repl=...)
  * scoped block (propagates to nested invokes): with jaz.ConfigOverride(repl=...): ...
  * one invoke only: jaz.invoke(jaz.ConfigOverride(repl=...), task=...)
- Budget / iteration / recursion caps are HOOKS, not config fields:
      with jaz.hooks.BudgetPool(cost_budget=1.0, calls_budget=50): ...
      with jaz.hooks.IterationLimit(max_iterations=10): ...
      with jaz.hooks.RecursionLimit(max_depth=3): ...
  (RecursionLimit is `with`-only — it rejects positional/per-invoke activation.)
- ANSWERING a question about current settings ("what model am I using?"): a read-only
  snapshot of the CURRENT config is handed to you as `config`. Read it and answer in words —
  e.g. config.llm.type, config.llm.model, config.repl.exec_timeout, config.repl.allowed_imports,
  config.protocol.max_invoke_input_length, config.protocol.max_repl_output_length. It is plain data (no
  methods; secrets like api_key are redacted). Each group IS the component that will be used,
  so what you read is what is in force — there is no separate "but what is the default?"
  question. Return ('answer', ...) with the value, not code.
- Any CODE you propose that must itself read current settings — e.g. to restate a component
  while preserving values you are not changing — reads them at runtime with jaz.get_config(),
  e.g. cfg = jaz.get_config(); jaz.configure(llm=OpenAILLM(model=cfg.llm.model, base_url=...)).
  That code runs later in the user's console (which can import jaz); you cannot, which is why
  you ANSWER from `config` but generated CODE reads from jaz.get_config().
- In the interactive console a session-global jaz.configure(...) line is usually what the
  user wants; produce `with`-scoped forms only when the request asks for a temporary or
  one-off change.
- READING THE JAZ SOURCE: the installed jaz package is readable with the builtin open().
  `source_root` is the package directory and `source_files` lists every Python file under
  it, relative to that root — e.g. open(f"{source_root}/config.py").read(). Nothing outside
  the package tree is readable, and there is nothing to import. Prefer `config` and this
  reference for settings questions; read the source when a question turns on how jaz
  actually behaves (defaults, error text, mechanism). Big files: print a slice at a time.
  When your answer leans on source you read, cite the file path so the user can look too.
- NON-SETTINGS REQUESTS are in scope: this reference only covers settings, so answer
  usage/behaviour/debugging questions from the source tree. When the user asks you to run
  something, hand back a ('code', ...) snippet they confirm before it runs — jaz, invoke and
  jprint are already bound in their console. Route by the question/action split above: a
  "how do I …" seeking understanding is a QUESTION (answer it, code block inline if useful);
  reserve ('code', ...) for an action they want executed now.
"""


# The helper invoke pins protocol.max_invoke_input_length to this so the reference document (handed
# in as an ordinary invoke input, and thus subject to per-input truncation) can never be
# silently truncated as the config docstrings grow. The pin is a fixed, doc-appropriate ceiling
# held independent of the ambient default: the live reference is ~10.2k, so 40k sits well clear
# of it — and if the doc ever outgrew the ceiling, truncation_prefix_ratio=0.5 would take the
# *middle*, i.e. exactly the per-component catalogues a snippet most needs. Generating those from
# the registry is what makes the doc grow with every backend registered, so the pin has to stay
# well clear of it. It now sits *below* the 50k default rather than above the old 10k one, which
# is fine: 40k still clears the ~10.2k reference, and the point is a stable ceiling a host's
# default can't move, not one relative to it. Shared with the drift-guard test so the ceiling is
# asserted against the same number that is enforced.
_HELPER_MAX_INPUT_LENGTH = 40000

# Ceiling for the helper's rendered REPL output, pinned alongside max_invoke_input_length. Even
# the raised 50k default is too small for this helper's main use: printing a source file to study
# it. 80k is chosen to fit the settings-relevant core modules
# — `config.py` (~56k) and `_agent.py` (~77k) — in a single read: `config.py` in particular is
# the file the helper is likeliest to open whole for a settings question, and truncating it
# would force a slice-and-re-read that costs turns against the tight first-grant budget. The
# three genuine giants — `dispatcher.py` (~106k), `python_repl.py` (~112k), `console.py` (~137k)
# — still exceed it and are read in slices; whole-file reads of those are rarely what a
# *settings* answer needs, and an unbounded ceiling would spend the turn/token budget on one
# dump. Truncation past the ceiling is not silent — abbreviate_string splices a visible
# "[...N characters omitted...]" marker at the cut — so the helper sees a read was cut and where,
# which is what makes the task's "print a slice at a time" instruction actionable on the giants.
_HELPER_MAX_OUTPUT_LENGTH = 80000

# The one directory the helper may READ: the installed jaz package itself. Resolved so the
# pattern and secure_open's resolved targets share one canonical space (a symlink that
# points outside the tree resolves outside it and is denied — fail-closed). The f"/{...}"
# spelling yields a "//<abs-path>/**" pattern, i.e. absolute-anchored in secure_open's
# anchor vocabulary. Computed, not a literal, unlike the other sandbox pins: the install
# location is only known at import time, and pinning a stale path would deny everything
# (fail-closed) rather than widen anything.
_JAZ_SOURCE_ROOT = Path(jaz.__file__).resolve().parent


# Iteration budget for the settings-helper invoke, and how it grows when the helper can't
# finish. The first attempt runs on a deliberately small budget: most settings requests are
# one or two turns, and a tight cap keeps a model that dithers in its REPL cheap. If that
# attempt exhausts without a response, the user is asked whether to grant more turns, and the
# offered grant DOUBLES each round — 10, then 20, then 40, ... — so a genuinely hard request
# can be pushed further on demand while a hopeless one is abandoned after one or two declines.
#
# Each grant is a *fresh* invoke with the larger cap, not a resumption: an invoke does not
# expose a "continue with N more turns" entry point, and the helper is stateless across calls
# anyway (its only input is the request text). So "give it 10 more turns" is implemented as
# "re-run it from scratch allowed 10 turns" — the user-facing promise (more room to finish)
# holds, and 10 > the initial 3, 20 > 10, 40 > 20, so every round really is *more* than the
# last attempt got.
_HELPER_INITIAL_ITERATIONS = 3
_HELPER_FIRST_GRANT = 10
# Cap the number of escalation rounds so a genuinely hopeless request is abandoned after a few
# declines rather than re-prompting with a doubled offer indefinitely: 3 rounds = grants of 10,
# 20, 40 before falling through to the rephrase hint. Only the turn-cap
# (IterationLimitExhaustedError) path escalates; a spent BudgetPool is caught separately and
# reported without an offer (#1079 gave the two causes distinct exception types), so this cap is no
# longer doing double duty as the BudgetPool backstop.
_HELPER_MAX_GRANTS = 3


# The settings helper's sandbox pin, hoisted to a module constant so the security property can
# be asserted against the real value rather than a copy of it (see
# ``test_settings_helper_sandbox_pins_every_containment_axis``). See the block comment in
# ``_propose_settings_response`` for why every axis is listed and why the values are literals.
#
# Configured components rather than dicts of leaves, because that is what config holds — and
# naming a ``PythonREPL`` *is* pinning the language, so the deny-all still applies in a session
# that had switched REPLs. Built once at import and shared: safe because ``REPL.initialize()``
# returns a copy with its mutable allow-lists rebound, so a run cannot write back through this
# template and widen the sandbox for the next one.
#
# allowed_read_paths grants exactly the jaz package tree (and nothing else): the helper's
# job is to explain jaz, and the installed source is public, secret-free material for that.
# The grant rides the existing secure_open enforcement — reads happen through the builtin
# open(), the only file API the sandbox exposes. allowed_imports staying [] is load-bearing
# for this axis too, not just for reaching `configure`: pathlib/os/io read files through
# their own OS calls, not the wrapped open(), so any import that can touch the filesystem
# would bypass the read allow-list entirely.
_HELPER_SANDBOX_OVERRIDE: dict[str, Any] = {
    "repl": PythonREPL(
        allowed_imports=[],
        # Restates the shipped default rather than a tighter list of its own. The frame-bearing
        # deny-list this pin used to carry lives in `DEFAULT_ALLOWED_ATTRIBUTES` now (see the block
        # comment there): the escape it closes is not helper-specific, so fixing it per-sandbox
        # meant every future sandbox re-deriving the same list and the one that forgot being
        # silently escapable. Consequence accepted deliberately: the helper no longer holds an
        # attribute policy *stricter* than the default, so a widening of the default widens the
        # helper too — this change already is one, since the default re-admits `__init__`/`__name__`/
        # `__qualname__` and the helper had none of them before. The pin's job is to be independent
        # of SESSION config — which it still is, since a group is replaced wholesale — not of the
        # module default.
        #
        # Passed explicitly even though it equals the default (as `allowed_imports=[]` already is),
        # so `test_settings_helper_sandbox_pins_every_containment_axis` asserts every axis against a
        # real value and a future edit dropping an axis is a test failure rather than a silent
        # inherit.
        allowed_attributes=list(DEFAULT_ALLOWED_ATTRIBUTES),
        # Reads only, jaz package tree only (absolute-anchored `//<root>/**`); writes stay []
        # even inside the tree. This is the grant that lets the helper consult the source.
        allowed_read_paths=[f"/{_JAZ_SOURCE_ROOT}/**"],
        allowed_write_paths=[],
    ),
    "protocol": CodeOnlyProtocol(
        max_invoke_input_length=_HELPER_MAX_INPUT_LENGTH,
        max_repl_output_length=_HELPER_MAX_OUTPUT_LENGTH,
    ),
}


def _component_catalogue(group: str, entries: Mapping[str, type], default: type) -> str:
    """One reference section per *registered* component of a group.

    Generated from the registry rather than naming one class, so a backend/REPL/protocol added
    with ``@register_llm`` & co. is documented to the helper without touching this module —
    previously only the built-in default appeared, so a snippet could never name a custom one.

    Each entry carries the import path and the ``__init__`` signature because, with config
    holding components, the constructor *is* that group's settings: the helper needs both to
    write ``jaz.configure(llm=MyBackend(...))`` at all.
    """
    from jaz.llm.base import declared_init_keys

    lines = []
    for tag, cls in sorted(entries.items()):
        try:
            params = list(inspect.signature(cls.__init__).parameters.values())[1:]
        except (TypeError, ValueError):  # pragma: no cover - builtins / C types
            params = []
        shown = {p.name for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD}
        # The `**kwargs` tail is dropped from the rendered signature rather than printed. Its
        # local name is the forwarding parameter's (`**retry` on a backend), which reads as
        # "only retry knobs go here" — the opposite of what an open tail means. What is behind
        # it is spelled out below instead.
        sig = "(" + ", ".join(str(p) for p in params if p.name in shown) + ")"

        default_marker = "  [default]" if cls is default else ""
        # Not written as `group=tag` — an assignment form would be the most prominent line of
        # every entry, and `configure(llm='openai')` raises: config takes the component, and a
        # tag names a component only in authored data, which this helper never writes.
        lines.append(f"### {cls.__name__} — tag {tag!r}{default_marker}")
        lines.append(f"from {cls.__module__} import {cls.__name__}")
        lines.append(f"{cls.__name__}{sig}")
        # A `**kwargs` tail hides everything the base declares — for `OpenAILLM` that is
        # `model` and every `retry_*`, i.e. the settings a snippet most often needs. The
        # signature alone would document the backend as taking `api_key`/`base_url`/`timeout`
        # and nothing else, so the inherited keys are listed from the same MRO walk the
        # boundary uses to decide what a component accepts.
        inherited = sorted(declared_init_keys(cls) - shown)
        if inherited:
            lines.append("also accepts: " + ", ".join(inherited))
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            # Said explicitly because the list above is *not* exhaustive for such a component:
            # an LLM forwards anything it does not name as a per-request default. Enumerating
            # those is impossible, so the note stops the enumeration reading as closed.
            lines.append(
                "plus any other keyword, kept as a per-request default and sent to the "
                "provider API (temperature, max_tokens, reasoning_effort, ...)"
            )
        doc = inspect.getdoc(cls)
        if doc:
            lines.append(doc)
        lines.append("")
    return "\n".join(lines)


def _registered_components() -> dict[str, Mapping[str, type]]:
    """Every registered component per group, resolving deferred entries.

    ``LLM_REGISTRY`` alone would miss a backend whose module is imported on first use (``rlm``),
    so the tags come from :func:`~jaz.llm.registry.registered_llm_tags` and each is
    resolved through :func:`~jaz.llm.registry.resolve_llm`, which materialises those.
    """
    # `ImportError` is skipped rather than fatal: this builds a *document*, and one entry
    # failing to import must not take the settings helper down with it.
    #
    # NOT what the built-in `rlm` does — it imports its optional dependency inside
    # `RLMClient.__init__`/`complete`, so it resolves and is listed even with `rlm` absent.
    # The guard is for an out-of-tree lazy backend that imports its dependency at module
    # scope, which is the ordinary way to write one.
    # Imported from the registry modules, not the `jaz.protocol` / `jaz.repl` package
    # re-exports: those are flagged experimental and emit `NonPublicAPIWarning` on attribute
    # access, which a console path must not do (pinned by
    # `test_console_paths_emit_no_non_public_api_warnings`).
    from jaz.llm.registry import registered_llm_tags, resolve_llm
    from jaz.protocol.registry import INTERACTION_PROTOCOL_MAP
    from jaz.repl.registry import REPL_LANGUAGE_MAP

    llms: dict[str, type] = {}
    for tag in registered_llm_tags():
        try:
            cls = resolve_llm(tag)
        except ImportError:
            # Narrow on purpose: `resolve_llm` raises `RuntimeError` when a lazy target fails
            # to self-register and `AttributeError` for a malformed `module:attr` target, and
            # swallowing those would turn a registration bug into an entry that silently
            # vanishes from the reference — exactly what those errors exist to surface.
            continue
        if cls is not None:
            llms[tag] = cls
    return {
        "llm": llms,
        "repl": dict(REPL_LANGUAGE_MAP),
        "protocol": dict(INTERACTION_PROTOCOL_MAP),
    }


def _settings_reference() -> str:
    """The reference document handed to the settings-helper agent.

    Built from ``inspect.getdoc`` on ``Config`` (the nested set-surface contract) and on
    ``jaz.ConfigOverride``, plus a generated catalogue per group — every *registered*
    component with its import path, constructor and docstring — plus the curated
    ``_SETTINGS_ADDENDUM`` for hook-based knobs and console conventions.

    Why dynamic docstrings rather than the alternatives (deliberate):

    - ``inspect.getsource(Config)`` was rejected: hundreds of lines dominated by
      validation code and internal design rationale, which would mislead a
      snippet-writing model and burn tokens.
    - A fully static curated text was rejected: it drifts silently as config fields
      evolve. The docstrings move with the code; only the addendum can drift, and a test
      pins the key names it must keep mentioning.
    - ``inspect.getdoc(jaz.configure)`` (the original source here) was dropped when #961
      landed: the ``configure`` docstring still documents the pre-nesting flat option
      names, which now *raise* — feeding it to the helper would actively teach it broken
      code. The class docstrings are the ones #961 kept current. If the ``configure``
      docstring is refreshed upstream, it would be the better single source again.
    """
    from jaz.config import Config, _default_components

    parts = [
        f"## jaz Config (set via jaz.configure(**groups))\n{inspect.getdoc(Config) or ''}"
    ]

    # Every registered component, not just the built-in default — a group *is* its configured
    # component now, so the reference has to name each one the helper is allowed to build.
    registered = _registered_components()
    # Read off `_default_components`, the same call a fresh `Config` makes, rather than a
    # second list here. A `[default]` marker is the one hint telling the helper which component
    # it gets for free, so a copy that silently drifts is worse than no marker at all.
    defaults = {group: type(c) for group, c in _default_components().items()}
    for group, title in (
        ("llm", "llm — backends (jaz.configure(llm=...))"),
        ("repl", "repl — REPL execution layers (jaz.configure(repl=...))"),
        ("protocol", "protocol — LLM<->REPL codecs (jaz.configure(protocol=...))"),
    ):
        parts.append(
            f"## {title}\n"
            + _component_catalogue(group, registered[group], defaults[group])
        )

    parts.append(f"## jaz.ConfigOverride\n{inspect.getdoc(jaz.ConfigOverride) or ''}")
    parts.append("## " + _SETTINGS_ADDENDUM)
    return "\n\n".join(parts)


# Redact any setting whose NAME looks secret, so a secret value never enters the helper's
# prompt/sandbox. Substring match (case-insensitive) on these markers rather than an exact name:
# the view is built generically from any registered backend's `declared_init_keys`, and a
# third-party backend's secret may be spelled `auth_token`, `client_secret`,
# `aws_secret_access_key`, `password`, `credentials`, ... — the built-ins all use `api_key`.
# Chosen (user's call on the PR that added this view) over an allow-list of safe names, which
# would lose the "a newly added non-secret setting shows up on its own" property, and over the
# prior single-name deny-list, which failed *open* for any backend not named here. It still
# guesses by name, so it errs two ways, both deliberately toward safety: a secret under a name
# matching none of these markers slips through, and a benign field whose name happens to contain
# one (e.g. `max_tokens` → "token") is over-redacted and simply won't appear. Hiding a non-secret
# is a cosmetic loss; leaking a secret is not.
_SETTINGS_SECRET_MARKERS = ("key", "secret", "token", "password", "credential")


def _is_secret_setting(name: str) -> bool:
    """True if ``name`` looks like a secret-bearing setting that must be kept out of the view."""
    lowered = name.lower()
    return any(marker in lowered for marker in _SETTINGS_SECRET_MARKERS)


def _current_settings_view() -> str:
    """A read-only, secret-redacted snapshot of the CURRENT config as plain text — one line per
    group (``config.llm`` / ``config.repl`` / ``config.protocol``).

    Handed to the settings helper as ``config`` so a read-only question ("what model am I
    using?") is answered directly from live state, instead of coming back as code the user must
    confirm and run. The helper's sandbox denies imports, so it cannot call
    :func:`jaz.get_config` itself; this text gives it the values — and only the values — it needs.

    Each line names the component class as ``type`` (for "what backend am I on?") followed by
    that component's declared settings, rendered with ``repr``. Built from ``declared_init_keys``
    (the same MRO walk the config boundary uses), so a component's settings appear without being
    named here — a newly added, non-secret setting shows up on its own. Secret-looking fields are
    dropped (see :func:`_is_secret_setting`). Per-request defaults riding the ``**request_defaults``
    tail (``temperature`` &c.) are not enumerated here — a follow-up can surface them if needed.
    """
    # Rendered as TEXT, not an object, by explicit user decision (weighing a review note that a
    # live object is the first non-string ever put in this sandbox). The helper only reads values
    # to write prose — it never navigates the snapshot programmatically — so text loses nothing.
    # A string keeps the sandbox's inputs strings-only: there is no instance whose `__class__` /
    # `__globals__` a reader could walk, so read-only holds BY CONSTRUCTION rather than via a
    # guard, and the one un-closeable CPython route a live object would expose —
    # `"{0.__globals__}".format(obj)`, which reaches attributes at the C level below the sandbox's
    # `allowed_attributes` deny-list — has nothing to reach. The alternative (a frozen
    # `_ReadOnlyView` object plus a test that its class is rejected in the sandbox) was rejected as
    # strictly more surface for zero functional gain. `repr` on each value is likewise safe by
    # construction: text can't leak a live object, and repr never raises on the stdlib types here.
    from jaz.llm.base import declared_init_keys

    cfg = jaz.get_config()
    lines: list[str] = []
    for group in ("llm", "repl", "protocol"):
        component = getattr(cfg, group)
        # `type` is the class name (the "what backend am I on?" answer). Seed it first and skip a
        # declared key literally named `type`, so a component that declares one can't overwrite the
        # class name — the model would otherwise answer confidently and wrongly.
        pairs = [f"type={type(component).__name__!r}"]
        for key in sorted(declared_init_keys(type(component))):
            if key == "type" or _is_secret_setting(key) or not hasattr(component, key):
                continue
            pairs.append(f"{key}={getattr(component, key)!r}")
        lines.append(f"config.{group}: " + ", ".join(pairs))
    return "\n".join(lines)


def _jaz_source_listing() -> list[str]:
    """Relative paths of every Python file in the installed jaz package, sorted.

    Handed to the settings helper as ``source_files`` — its map of what it may read. The
    sandbox denies all imports, so the helper has ``open()`` but no way to *list* a
    directory (``os``/``pathlib`` are unreachable); without this input the read grant on
    ``_JAZ_SOURCE_ROOT`` would only cover files it could guess the names of.
    """
    # ~99 files ≈ 3k characters — comfortably inside the 40k per-input ceiling, so the
    # listing cannot be the thing truncation eats. __pycache__ is skipped as noise (the
    # read grant still covers it; hiding it from the map is curation, not containment).
    return sorted(
        p.relative_to(_JAZ_SOURCE_ROOT).as_posix()
        for p in _JAZ_SOURCE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    )


# The helper answers in one of two shapes, tagged by the first tuple element so the console
# can tell them apart without guessing from the text:
#   ("code",   snippet) — Python to CONFIRM then run: a settings change (jaz.configure(...))
#                         or any other jaz-related action the user asked to perform.
#   ("answer", text)    — a plain-language reply to PRINT verbatim: a question answered from the
#                         redacted `config` text snapshot, `reference`, or the jaz source, or why a
#                         request can't be met.
# A tagged tuple of built-ins (not a custom dataclass) is deliberate: the helper's sandbox denies
# all imports, so it can only `return` values it can spell with literals — a `(str, str)` tuple it
# can, a `SettingsResponse(...)` it could not (the class is unreachable without an import). The
# `Literal` tag is the "how does it say which kind" mechanism; beartype checks it at the return
# boundary, so a mistagged reply is rejected and retried rather than mis-dispatched here.
_HELPER_RESPONSE_TYPE = tuple[Literal["code", "answer"], str]


def _propose_settings_response(
    request: str, max_iterations: int = _HELPER_INITIAL_ITERATIONS
) -> _HELPER_RESPONSE_TYPE:
    """Ask a sandboxed helper agent how to handle ``request``; return ``(kind, content)``.

    ``kind`` is ``"code"`` (``content`` is a Python snippet the caller confirms and runs) or
    ``"answer"`` (``content`` is plain text the caller prints verbatim). See
    ``_HELPER_RESPONSE_TYPE`` for why the two modes exist and why they ride a tagged tuple.

    **This function never executes anything** — proposing and applying are separated on
    purpose (the user's core requirement: the helper must not be able to change settings
    itself). The sandbox makes that a guarantee, not a convention:

    1. The positional ``ConfigOverride`` forces both ``language="python"`` and
       ``allowed_imports=[]`` (deny-all) regardless of how the session is configured — the
       allow-lists are protected, agent-un-strippable settings (#690), and naming a
       ``PythonREPL`` pins the language, so the deny-all applies to the REPL the helper
       actually gets even in a bash session — so the helper agent cannot ``import jaz`` (nor
       ``sys``/``importlib``) to reach ``configure``.
    2. ``RecursionLimit(max_depth=1)`` makes the helper a leaf: ``DisableRecursion`` means
       no agent-facing ``jaz`` library is bound in its REPL at all — no ``configure``, no
       nested ``invoke``. (Entered via ``with`` — RecursionLimit rejects positional use.)
    3. Its REPL namespace holds only the inputs — the ``request``/``reference`` strings, the
       ``config`` TEXT snapshot (a redacted view of current settings, see
       :func:`_current_settings_view`), and the ``source_root``/``source_files`` map of the jaz
       package; the console namespace is unreachable from inside the invoke. Because ``config``
       is text and not an object, it widens what the helper can *read* without adding any
       instance whose class or globals a reader could walk. File reads go through the sandboxed
       ``open()``, whose allow-list grants exactly the jaz package tree (see
       ``_HELPER_SANDBOX_OVERRIDE``) — the same read-not-reach property, extended from config
       values to jaz's own source.
    4. Its sole effect channel is the returned ``(kind, content)`` tuple. A ``"code"``
       ``content`` crosses into the console only through :func:`jaz_settings`'
       printed-snippet + ``[y/N]`` confirmation — the same trust boundary as the user typing
       the code themselves. An ``"answer"`` ``content`` is printed verbatim and never
       executed, so it carries no privilege regardless of what the model wrote.

    Cost/model: uses the ambient model config (respects ``--model`` and in-session
    ``jaz.configure``) rather than a hardcoded cheap model — there is no guarantee which
    provider keys exist. Cost is bounded instead: leaf-only + ``IterationLimit(max_iterations)``
    + a small prompt. ``max_iterations`` defaults to a tight cap; :func:`jaz_settings` re-calls
    with a larger one when the user grants the helper more turns (see ``_HELPER_FIRST_GRANT``).

    The task is a plain ``str`` (NOT a t-string): ``task`` is an ordinary input, and this
    module must stay importable on Python < 3.14, so no literal ``t"..."`` may appear in
    this file's source (only the sigil *rewrites* may emit them).
    """
    from jaz.hooks import IterationLimit, RecursionLimit

    # Task-prompt subtleties, learned from live failures:
    # - "deliver by RETURNING the value" must be explicit. The first wording ("return
    #   only the code") read as *formatting* advice; on a read-only request the agent
    #   wrote `#`-comment REPL turns without ever finishing, burned its 3 iterations, and
    #   the user got an IterationLimitExhaustedError traceback instead of a proposal. (The
    #   wording used to add "finish IMMEDIATELY — do NOT run anything"; the source-read grant
    #   retired that half, since consulting a file IS running something. The pinned part is
    #   the delivery contract — the response arrives by `return`, never as comment/print.)
    # - Questions are first-class: a *helper* that rejects *questions* is exactly the
    #   discoverability failure `%` exists to fix. Every question is an ("answer", text):
    #   live-state questions are read straight off the `config` text snapshot handed in below,
    #   how-to/conceptual ones off the `reference`, how-does-jaz-work ones off the source
    #   tree, and an impossible request explains itself. ("code", ...) is reserved for
    #   ACTIONS — a settings change or other jaz code the user asked to run — so the [y/N]
    #   gate only ever appears when something would actually execute. This split is what the
    #   `config` text snapshot buys — before it, a live-state question had to come back as code
    #   that reads jaz.get_config() at run time (the sandbox denies imports, so the helper
    #   itself could not read the running config), forcing a needless [y/N] on a pure
    #   question.
    # Naming a `PythonREPL` pins the language as well as the deny-all import list, and that is
    # load-bearing: the allow-lists are Python-REPL settings, so in a bash session the helper
    # would otherwise inherit a shell and they would be inert — quietly turning "a sandbox that
    # can't touch settings" into "an LLM with a shell". Passing the component makes the two
    # inseparable; they could previously drift apart because `language` and `params` were
    # independent leaves.
    #
    # EVERY containment axis is pinned explicitly, not just `allowed_imports`. An earlier version
    # pinned imports alone and claimed the rest reset to their fail-closed defaults because the
    # params bag was "rebound wholesale". It is not: `Config.update` deep-merges dict leaves via
    # `_merge_leaf`, so only the keys actually named are replaced. A session that had widened
    # `allowed_attributes` to ["*"] therefore leaked dunder access — and with it
    # `__class__`/`__subclasses__` reachability — straight into the helper's sandbox, defeating
    # the deny-all import list it was paired with.
    #
    # The values are literals rather than references to `DEFAULT_ALLOWED_ATTRIBUTES` /
    # `_DEFAULT_ALLOWED_*` on purpose: a security pin must not track a constant that a future
    # default change — or tests/conftest.py's `_relaxed_sandbox_defaults` fixture, which
    # monkeypatches exactly those constants — can move underneath it.
    #
    # `allow_raise` is deliberately NOT pinned: it decides whether a raise finishes the turn,
    # its default is True, and it is not a containment axis.
    #
    # protocol.max_invoke_input_length is pinned so the reference doc can't be truncated regardless of
    # docstring growth (see _HELPER_MAX_INPUT_LENGTH).
    with RecursionLimit(max_depth=1):
        raw = jaz.invoke(
            ReturnType(_HELPER_RESPONSE_TYPE),
            jaz.ConfigOverride(**_HELPER_SANDBOX_OVERRIDE),
            # Supply the last-turn nudge explicitly: core hooks no longer emit any default
            # framing, so a bare IterationLimit would give the helper only the hard abort. With
            # warn_remaining=1 the agent gets this warning on its final allowed turn (mirrors the
            # old built-in "last allowed REPL interaction" NOTE, now caller-owned text).
            IterationLimit(
                max_iterations=max_iterations,
                warning_text=(
                    "This is your last allowed REPL iteration -- you must finish up "
                    "your work and terminate the REPL session in this turn."
                ),
                warn_remaining=1,
            ),
            task=(
                "The user typed a request about jaz (the LLM-agent framework this console "
                "runs) — its settings, API, usage, behaviour, or a problem they hit; it is in "
                "`request`. You are given `config`, a read-only TEXT snapshot of the CURRENT "
                "settings (one line per group — `config.llm`, `config.repl`, `config.protocol` — "
                "each listing `type` and its settings, e.g. the `config.llm` line shows `type` "
                "and `model`), and `reference`, the settings API docs. You can also READ the jaz "
                "source code: `source_files` lists every Python file in the installed jaz package "
                "as a path relative to `source_root`, and open(f'{source_root}/<path>').read() "
                "returns one — consult it whenever `config` and `reference` do not settle the "
                "question (print slices of big files). Only these are available; decide how "
                "to respond and finish by RETURNING a (kind, content) tuple — do NOT leave "
                "your response as a REPL comment or print it.\n"
                "- Return ('answer', text) for any QUESTION: read the answer off `config` for "
                "current settings ('what model am I using?'), settle how-to/conceptual "
                "questions from `reference`, read the source for anything about how jaz "
                "works or why it behaved some way, or explain why a request cannot be done. "
                "`text` is the reply, e.g. return ('answer', 'You are using ...').\n"
                "- Return ('code', snippet) for any ACTION the user asked to perform: a "
                "settings change (usually jaz.configure(...)) or other jaz-related Python "
                "they want to run (e.g. an example jaz.invoke(...) call). The user confirms "
                "before it runs in their console. `snippet` is one plain string of Python, no "
                "prose and no markdown fences, e.g. return ('code', 'jaz.configure(...)')."
            ),
            request=request,
            reference=_settings_reference(),
            config=_current_settings_view(),
            source_root=str(_JAZ_SOURCE_ROOT),
            source_files=_jaz_source_listing(),
        )
    kind, content = raw
    # Fences only make sense to strip on code; an answer is printed verbatim.
    return kind, (_strip_code_fences(content) if kind == "code" else content)


def _grant_more_helper_turns(grant: int) -> bool:
    """Ask the user whether to retry the settings helper with a budget of ``grant``; True on yes.

    Called only after the helper exhausted its budget, and only with a tty present (the caller
    checks), so there is always a human to answer. Defaults to No — like the ``Run this?``
    prompt, silence does not spend more of the user's budget.
    """
    # "Retry with a budget of {grant}", not "give it {grant} more": each round is a fresh invoke
    # capped at ``grant`` turns *total* (the prior attempt's turns are discarded, not added to),
    # so additive phrasing would overstate what the grant buys. "Retry" also signals the restart.
    answer = input(
        f"%: the helper ran out of turns. Retry with a budget of {grant}? [y/N] "
    )
    return answer.strip().lower() in {"y", "yes"}


def jaz_settings(request: str, namespace: dict[str, Any]) -> None:
    """Back the ``% request`` sigil: get the helper's response, then answer or confirm+run.

    The helper (sandboxed — see :func:`_propose_settings_response` for why it cannot apply
    anything itself) replies in one of two shapes:

    - ``"answer"`` — a plain-language reply (a how-to, an explanation of jaz behaviour, or
      why a request can't be done). It is **printed verbatim and never executed**, so there
      is nothing to confirm: it prints in every mode, including a ``.jaz`` script or piped
      stdin, because reading back text is safe with no human present.
    - ``"code"`` — a snippet to run. It is printed, then ``Run this? [y/N]`` is asked; only an
      explicit ``y``/``yes`` executes it **in the console namespace**, exactly as if the user
      had typed it at the prompt (``namespace`` arrives via the ``globals()`` argument in the
      rewritten call, so this works identically for the interactive, one-shot, and script
      paths). After "y" the snippet runs unsandboxed *by design*; the printed-snippet +
      confirmation (defaulting to No) is the trust boundary.

    Exec errors propagate: the call site is a rewritten console line, so a raising
    snippet is reported exactly like any erroring line (traceback via ``showtraceback``,
    ``errored`` set for the script runner's fail-fast, exit 1 in one-shot mode).

    One deliberate exception to "errors propagate": a helper invoke that runs out of
    *iterations* without returning a response (``IterationLimitExhaustedError``, e.g. the model
    dithered in its REPL) is a *helper quality* failure, not a user error — surfacing it
    as a 30-line traceback through ``jaz.invoke`` internals (the observed UX) buries the
    one actionable fact. Before giving up, the user is offered progressively larger turn
    budgets (see ``_HELPER_FIRST_GRANT``): a request that just needed more room to finish can
    be pushed further on demand, and only after a decline (or the offer cap, or with no tty
    to ask) is it reported as a one-line hint to rephrase. A spent session cost/calls pool
    (``BudgetPoolExhaustedError`` from ``jaz --max-cost/--max-calls``) is reported the same
    one-line way but *without* a turn offer — more turns cannot refill it. Other exceptions
    (config errors, provider/network failures) still propagate with full tracebacks — those
    need their detail.

    Applying ``"code"`` is interactive-only. When stdin is not a tty (a ``.jaz`` script or
    piped input) there is nobody to confirm, so the snippet is printed (for copy-paste) and
    the call then **raises**: an un-appliable settings line is a script error, not a silent
    no-op, so the run fails fast with a nonzero exit like any other erroring line — rather
    than continuing against the unconfigured session and exiting 0. (An ``"answer"`` needs no
    confirmation, so it does not hit this path.)
    """
    from jaz.exceptions import BudgetExhaustedError, IterationLimitExhaustedError

    # Start on the tight default budget. On the helper's own turn-cap exhaustion, offer the
    # user a doubling grant (10 -> 20 -> 40 -> ...) and re-run with it, up to _HELPER_MAX_GRANTS
    # rounds. Interactive-only: a run with no tty has nobody to grant more turns, so it falls
    # straight through to the rephrase hint. A session-wide BudgetPool (jaz --max-cost/--max-calls)
    # propagates into this nested invoke and exhausts as a BudgetPoolExhaustedError; more *turns*
    # cannot refill a spent cost/calls pool, so that case is reported without an offer. The distinct
    # exception types (#1079) are what let the two be told apart here — previously both arrived as a
    # bare BudgetExhaustedError and only the round cap kept the pool case from looping forever.
    budget = _HELPER_INITIAL_ITERATIONS
    grant = _HELPER_FIRST_GRANT
    grants_offered = 0
    while True:
        try:
            kind, content = _propose_settings_response(request, max_iterations=budget)
            break
        except IterationLimitExhaustedError as exc:
            if (
                grants_offered >= _HELPER_MAX_GRANTS
                or not sys.stdin.isatty()
                or not _grant_more_helper_turns(grant)
            ):
                print(
                    f"%: the helper could not produce a response ({exc}). "
                    "Try rephrasing the request.",
                    file=sys.stderr,
                )
                return
            budget, grant = grant, grant * 2
            grants_offered += 1
        except BudgetExhaustedError as exc:
            # A spent cost/calls pool (BudgetPoolExhaustedError, or any other non-iteration
            # BudgetExhaustedError): more turns cannot help, so report and stop with no offer.
            # Listed after IterationLimitExhaustedError so the subclass branch wins.
            print(
                f"%: the helper could not run — the session budget is exhausted ({exc}). "
                "Raise the --max-cost/--max-calls ceiling or start a new session.",
                file=sys.stderr,
            )
            return

    # An answer is just text: print it and stop. No confirmation, and no interactive-only
    # gate — nothing runs, so it is safe with no human at the prompt (a script/piped `%
    # what does allowed_imports do?` still gets its answer).
    if kind == "answer":
        print(f"\n{content}")
        return

    snippet = content
    tty = sys.stdout.isatty()
    dim, reset = ("\033[2m", "\033[0m") if tty else ("", "")
    print("\nThe helper proposes:\n")
    for line in snippet.splitlines():
        print(f"    {dim}{line}{reset}")
    print()
    if not sys.stdin.isatty():
        # `%` is interactive-only (settled with the user): applying a snippet requires a human
        # at the prompt to confirm it, and there is no tty here (a `.jaz` script or piped
        # stdin). Raise rather than print-and-return: a settings line that silently applied
        # nothing yet exited 0 would let a script that "% configure then work" run against the
        # unconfigured session and *look* like it succeeded. Raising routes through the same
        # showtraceback -> `errored` path as any other failing line, so the run fails fast
        # (nonzero exit) like `python file.py`. The snippet is already printed above, so the
        # copy-paste-into-an-interactive-session affordance is preserved.
        #
        # Accepted trade (the answer path added a second way to reach exit 0 in a script): the
        # early `answer` return above prints-and-continues, so the same `% ...` line can now
        # raise or exit 0 depending on the model's tag, and a script's exit code for it isn't
        # reproducible run-to-run. This is fine precisely because it splits on whether anything
        # was *meant* to apply: an `answer` applies nothing by construction (no silent drop to
        # guard against), whereas this raise guards the `code` path — the only one where a
        # settings change was intended and would otherwise be silently skipped.
        raise RuntimeError(
            "%: applying the helper's code is interactive-only — it needs a tty to confirm "
            "before applying, so it cannot run in a script or piped input. The proposed "
            "snippet is shown above; apply it from an interactive session or paste it into "
            "your script directly."
        )
    answer = input("Run this? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        print("%: discarded — nothing was changed.")
        return
    exec(compile(snippet, "<jaz-settings>", "exec"), namespace)
    print("✓ applied")


def _settings_source(text: str) -> str:
    """Build the rewritten source for a ``% request`` line.

    ``{text!r}`` (repr) escapes quotes/backslashes exactly, and the request is passed
    verbatim — no t-string, so no ``{var}`` interpolation and no brace/quote limitations
    (and the ``%`` path compiles on any interpreter, unlike the prompt sigils).
    ``globals()``, evaluated *inside* the rewritten line, is the console namespace itself —
    which is how :func:`jaz_settings` gets the namespace to exec a confirmed snippet into
    with zero extra plumbing, uniformly across the interactive/one-shot/script paths.
    """
    return f"__jaz_settings__({text!r}, globals())"


# ---------------------------------------------------------------------------
# ASCII-art banner (moved verbatim from examples/interactive_repl.py so this
# module is the single source of truth for the banner; the example imports it back).
# ---------------------------------------------------------------------------
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    # Bright colors
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    WHITE = "\033[97m"
    # Rainbow sequence for animation
    RAINBOW = ["\033[91m", "\033[93m", "\033[92m", "\033[96m", "\033[94m", "\033[95m"]


FLAVOR_TEXTS = [
    "As seen on TV!",
    "The new coding experience",
    "Now with 50% more recursion",
    "Agents all the way down",
    "We put an LLM in your REPL",
    "Traces included, batteries not included",
    "It's not a bug, it's a feature request",
    "Making Python do the thinking",
    "Because copy-paste from ChatGPT was too slow",
    "It's actually Just Alex Zhang (JAZ)!",
]


# Each letter as separate art for individual coloring
LETTER_J = [
    r"     ██╗",
    r"     ██║",
    r"     ██║",
    r"██   ██║",
    r"╚█████╔╝",
    r" ╚════╝ ",
]

LETTER_A = [
    r" █████╗ ",
    r"██╔══██╗",
    r"███████║",
    r"██╔══██║",
    r"██║  ██║",
    r"╚═╝  ╚═╝",
]

LETTER_Z = [
    r"███████╗",
    r"╚══███╔╝",
    r"  ███╔╝ ",
    r" ███╔╝  ",
    r"███████╗",
    r"╚══════╝",
]


def print_colored_ascii() -> None:
    """Print the ASCII art with vertical gradient colors per letter."""
    print()

    # Vertical gradients for each letter (top to bottom)
    j_gradient = [
        "\033[38;5;51m",  # bright cyan
        "\033[38;5;50m",
        "\033[38;5;49m",
        "\033[38;5;48m",
        "\033[38;5;47m",
        "\033[38;5;46m",  # bright green
    ]

    a_gradient = [
        "\033[38;5;201m",  # bright magenta
        "\033[38;5;200m",
        "\033[38;5;199m",
        "\033[38;5;198m",
        "\033[38;5;197m",
        "\033[38;5;196m",  # red
    ]

    z_gradient = [
        "\033[38;5;226m",  # bright yellow
        "\033[38;5;220m",
        "\033[38;5;214m",
        "\033[38;5;208m",
        "\033[38;5;202m",
        "\033[38;5;196m",  # red/orange
    ]

    # Print line by line, combining all letters horizontally
    for row_idx in range(6):
        line = ""
        # J with vertical gradient
        line += f"{Colors.BOLD}{j_gradient[row_idx]}{LETTER_J[row_idx]}{Colors.RESET}"
        line += " "
        # A with vertical gradient
        line += f"{Colors.BOLD}{a_gradient[row_idx]}{LETTER_A[row_idx]}{Colors.RESET}"
        line += " "
        # Z with vertical gradient
        line += f"{Colors.BOLD}{z_gradient[row_idx]}{LETTER_Z[row_idx]}{Colors.RESET}"
        print(f"    {line}")

    # Decorative underline with gradient
    underline = "·bg~═══════════════════════════~dy·"
    print(f"    {Colors.BOLD}\033[38;5;93m{underline}{Colors.RESET}")
    print()


def animate_flavor_text(text: str) -> None:
    """Print flavor text with a rainbow typing animation."""
    padding = "  "
    sys.stdout.write(padding)
    sys.stdout.flush()

    for i, char in enumerate(text):
        color = Colors.RAINBOW[i % len(Colors.RAINBOW)]
        sys.stdout.write(f"{Colors.BOLD}{color}{char}{Colors.RESET}")
        sys.stdout.flush()
        time.sleep(0.03)

    # Final shimmer effect
    time.sleep(0.2)
    for _ in range(2):
        # Shift colors
        sys.stdout.write("\r" + padding)
        for i, char in enumerate(text):
            color = Colors.RAINBOW[(i + 1) % len(Colors.RAINBOW)]
            sys.stdout.write(f"{Colors.BOLD}{color}{char}{Colors.RESET}")
        sys.stdout.flush()
        time.sleep(0.1)

        sys.stdout.write("\r" + padding)
        for i, char in enumerate(text):
            color = Colors.RAINBOW[(i + 2) % len(Colors.RAINBOW)]
            sys.stdout.write(f"{Colors.BOLD}{color}{char}{Colors.RESET}")
        sys.stdout.flush()
        time.sleep(0.1)

    print("\n")


# ---------------------------------------------------------------------------
# Live progress display: spinner while the LLM thinks, typewriter code reveal
# ---------------------------------------------------------------------------
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Spinner labels, chosen so the animated line names what the process is *actually* doing
# rather than always claiming "thinking…" (which was misleading while an LLM call was being
# retried). The label is swapped by the relevant handler between the two LLM-in-flight
# phases: waiting on the response, or backing off before a retry. The spinner runs only
# during the LLM call — not during code exec (see on_repl_exec_enter for why). Kept free of
# a trailing "…" — `_spin` appends it.
_LABEL_WAITING = "waiting for response"
_LABEL_RETRYING = "retrying"

# ConsoleProgress instances whose activation scope (`with hook:` in main's ExitStack) is
# currently live — maintained by setup()/teardown(). Exists for exactly one caller:
# `_finish_progress_turns()`, the console's end-of-line cleanup. The hook's scope is the
# whole *session*, but a spinner belongs to one *turn* — and while the span CMs now fire
# LLMQueryExit/InvokeExit on every path including exceptional unwinds (#892), the exit
# *emission itself* is not unconditional in the ways that matter here: a Ctrl-C can land
# during the exit-emit, and a hook exception on the guarded close is swallowed by the
# dispatcher rather than retried. Before the console grew this end-of-turn signal of its
# own, a mid-LLM-call abort orphaned a live spinner thread, which kept overdrawing
# "⠹ thinking…" on the idle prompt — making an already interrupted invoke look
# un-killable (each further Ctrl-C just printed another KeyboardInterrupt while the
# orphan kept animating).
_ACTIVE_PROGRESS: list[ConsoleProgress] = []


def _finish_progress_turns() -> None:
    """Stop every active :class:`ConsoleProgress` spinner — the end-of-line backstop.

    Called (in a ``finally``) by :meth:`JazConsole.runcode` and :func:`_run_source`, the
    two places a console turn ends — normally *or* by exception. In almost every case the
    spinner is already stopped (the ``*Exit`` events fire on every path since #892, the
    abnormal arms included), so this is a cheap no-op; it remains the one stop that does
    not depend on the exit-emit itself surviving (a Ctrl-C landing during the emit, a
    swallowed hook error — see ``_ACTIVE_PROGRESS``).
    """
    for progress in list(_ACTIVE_PROGRESS):
        progress.finish_turn()


class ConsoleProgress(Hook):
    """Console UI hook: live progress while an ``invoke`` runs.

    Fixes the console's "silent while working" problem (settled with the user, who chose
    this UX over static status lines): before this hook, ``> some prompt`` printed nothing
    until the final result, so a multi-iteration invoke looked hung.

    What the user sees, per agent turn:

    - a braille **spinner** shown while an LLM query is in flight, labelled ``waiting for
      response…`` — or ``retrying (attempt N)…`` (N is the attempt now being made) if that
      call is being retried;
    - each code action the agent produces, revealed with a fast **typewriter** animation,
      prefixed by its iteration number and indented by ``depth - 1`` (nested invokes nest
      visually);
    - a compact ``✓``/``✗`` + elapsed-seconds line after each execution.

    The final result still arrives via the console's normal ``single``-mode displayhook —
    this hook is a pure observer (every handler returns ``[]``) and needs no coordination
    with it: the last turn's spinner is already stopped and its line cleared before the
    displayhook prints.

    Output-safety design (two independent reasons this never corrupts an agent's
    observation buffer, even though REPL exec captures stdout):

    1. **Bound-at-construction stream.** ``self._out`` is captured here in ``__init__``,
       exactly like ``PrintLogger`` (see the long rationale in
       ``jaz/hooks/builtin/loggers.py``): it is a saved reference to the host sink, distinct
       from the per-exec capture proxy that ``repl.exec`` routes ``sys.stdout`` through.
    2. **The spinner thread never inherits the capture ContextVar.** The stdout proxy
       routes by a ContextVar (``jaz/repl/stdout_proxy.py``); a raw ``threading.Thread``
       started by this hook does not carry the exec's context, so even a write through the
       proxy from that thread would fall through to the host sink.

    Spinner lifecycle (the one subtle bit): ``LLMQueryExit`` fires on every path once the
    query span opened — an ``Abort`` at ``LLMQueryEnter`` closes it ``Aborted``, and an
    exception from the LLM call (Ctrl-C included) closes it with a ``Failed`` outcome
    (the #892 abnormal arm) — but the exit handlers are hook
    code the dispatcher may be unwinding past (their errors are swallowed on the guarded
    close), so stopping the spinner keeps *backstops*:
    ``REPLExecEnter``, ``InvokeExit``, and ``teardown()`` all stop it — plus the decisive
    one, :func:`_finish_progress_turns`, which the console runs in a ``finally`` after
    every line, and which remains the guarantee the others only approximate.
    ``_stop_spinner`` joins the thread *before* clearing the
    line so a late frame can never interleave with subsequent output. Hook exceptions are
    swallowed by the dispatcher, so a UI bug here degrades to a logged error rather than
    killing the invoke.

    The hook is inert (no writes, no thread) when the bound stream is not a tty, so
    entering it unconditionally in ``main`` is safe for piped/redirected runs.

    Args:
        stream: Output stream; defaults to ``sys.stdout`` bound at construction (see above).
        char_delay: Per-character typewriter delay in seconds. ``0`` disables the animation
            (each code block is written in one pass) — used by tests.
        max_typing_time: Cap on the total reveal time of one code block, so a long block
            speeds up instead of dragging (per-char delay is
            ``min(char_delay, max_typing_time / len(code))``).
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        char_delay: float = 0.0015,
        max_typing_time: float = 0.25,
    ) -> None:
        # Bound at construction for capture immunity — see the class docstring, point 1.
        self._out: TextIO = stream if stream is not None else sys.stdout
        isatty = getattr(self._out, "isatty", None)
        self._enabled = bool(isatty()) if callable(isatty) else False
        self._char_delay = char_delay
        self._max_typing_time = max_typing_time
        # Stop signal for the *current* spinner thread, replaced with a fresh Event on
        # every _start_spinner. Per-thread rather than one shared Event: a timed-out join
        # in _stop_spinner_locked can return with the thread still alive, and a shared flag
        # would then be revived by the next _start_spinner (un-signalling the orphan, which
        # never sees the stop it was told about and keeps overdrawing the prompt — the exact
        # double-spinner the "defensive: never two spinner threads" guard exists to prevent).
        # Each thread only ever watches the Event it was born with, so it can't be revived.
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # The spinner's current label, read live by `_spin` each frame. Stored on the
        # instance (not passed to the thread) so a phase change — e.g. a retry landing
        # mid-LLM-call — can relabel the *running* spinner without tearing down its thread.
        # Plain str assignment is atomic enough for a display string; the lock still guards
        # the thread's start/stop, which is where the real races are.
        self._label = _LABEL_WAITING
        # Guards _thread/_stop transitions: handlers run on the invoke's thread, but a
        # nested invoke's events and the backstops can interleave start/stop calls.
        self._lock = threading.Lock()
        # (invoke_id, iteration) -> monotonic start time, for the ✓/✗ elapsed display.
        # Keyed per invoke so nested invokes' iterations can't collide.
        self._exec_started: dict[tuple[str, int], float] = {}

    # -- spinner -----------------------------------------------------------
    def _spin(self, indent: str, stop: threading.Event) -> None:
        # Watches the Event passed at construction, not self._stop, so a later restart that
        # rebinds self._stop can never reach into a still-running (orphaned) spinner.
        i = 0
        while True:
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            # Erase the line before each redraw: labels vary in length across phases, so a
            # shorter one (e.g. "running") must not leave the tail of a longer one behind.
            self._out.write(f"\r\x1b[2K{indent}{frame} {self._label}…")
            self._out.flush()
            i += 1
            if stop.wait(0.08):
                return

    def _start_spinner(self, depth: int, label: str = _LABEL_WAITING) -> None:
        with self._lock:
            self._stop_spinner_locked()  # defensive: never two spinner threads
            self._label = label
            # Fresh per-thread stop signal (see __init__): a never-cleared Event, so an
            # orphan that survived a timed-out join stays stopped and dies on its own.
            stop = threading.Event()
            self._stop = stop
            self._thread = threading.Thread(
                target=self._spin,
                args=("  " * (depth - 1), stop),
                daemon=True,
                name="jaz-console-progress",
            )
            self._thread.start()

    def _stop_spinner(self) -> None:
        with self._lock:
            self._stop_spinner_locked()

    def _stop_spinner_locked(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()  # this thread's own signal; a later restart won't un-set it
        # Join BEFORE clearing the line, so no late spinner frame lands after the clear.
        # A timed-out join leaves the thread alive, but it still holds the set() Event above
        # and exits at its next wait() — it cannot be revived, so at worst one stray frame
        # lands before the clear rather than an immortal orphan.
        thread.join(timeout=0.5)
        self._thread = None
        self._out.write("\r\x1b[2K")  # carriage return + ANSI erase-line (tty-gated)
        self._out.flush()

    # -- typewriter --------------------------------------------------------
    def _typewriter(self, text: str) -> None:
        delay = min(self._char_delay, self._max_typing_time / max(len(text), 1))
        if delay <= 0:
            self._out.write(text)
        else:
            for char in text:
                self._out.write(char)
                self._out.flush()
                time.sleep(delay)
        self._out.flush()

    # -- handlers (pure observer: every handler returns no effects) ---------
    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        if self._enabled:
            self._start_spinner(event.depth, _LABEL_WAITING)
        return []

    def on_llm_query_retry(self, event: LLMQueryRetry) -> list[Effect]:
        # Relabel the *live* spinner rather than restarting it: the retry fires inside a
        # single in-flight LLMQuery span (tenacity retries beneath one Enter/Exit pair), so
        # the "waiting for response" spinner is already running when the call fails. Show
        # the attempt count so a stuck-in-a-retry-loop turn is visibly distinct from a slow
        # first response. If the spinner isn't running (unusual — a retry with no live
        # spinner), this is a harmless no-op label write.
        #
        # +1 because tenacity's before_sleep fires *after* an attempt fails, so
        # event.attempt_number is the attempt that just failed; the retry we're announcing
        # is the next one about to be made ("retrying (attempt 2)" after attempt 1 failed).
        if self._enabled:
            self._label = f"{_LABEL_RETRYING} (attempt {event.attempt_number + 1})"
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        if self._enabled:
            self._stop_spinner()
        return []

    def on_repl_exec_enter(self, event: REPLExecEnter) -> list[Effect]:
        if not self._enabled:
            return []
        # Defensive stop: LLMQueryExit fires on every path now (#892), but its handler can
        # be skipped if the exit-emit itself failed (guarded-close errors are swallowed).
        self._stop_spinner()
        self._exec_started[(event.invoke_id, event.iteration)] = time.monotonic()
        indent = "  " * (event.depth - 1)
        prefix = f"[{event.iteration}] "
        self._out.write(f"{indent}\033[2m{prefix}")
        # Continuation lines align under the first code line, not under the `[n] ` prefix.
        hang = indent + " " * len(prefix)
        code_text = ("\n" + hang).join(event.code.splitlines() or [""])
        self._typewriter(code_text)
        self._out.write("\033[0m\n")
        self._out.flush()
        # No spinner while the code runs. The code is already on screen, so a slow command
        # (a long shell build, a nested invoke) reads as "the last thing printed" without
        # one. A spinner here would also redraw over uncaptured subprocess/fd-level output —
        # writes that bypass stdout_proxy and reach the tty directly (see repl/stdout_proxy
        # scope notes) — mangling exactly the long-build output it was meant to reassure on.
        return []

    def on_repl_exec_exit(self, event: REPLExecExit) -> list[Effect]:
        from jaz.hooks.events.base import Completed

        if not self._enabled:
            return []
        self._stop_spinner()  # defensive: exec runs no spinner of its own; clears a leak
        start = self._exec_started.pop((event.invoke_id, event.iteration), None)
        elapsed = "" if start is None else f" ({time.monotonic() - start:.1f}s)"
        # `exception` via getattr: ExecResult is a union and the `Return` member has no
        # `exception` field (a return IS a clean success). Continue-with-exception is a
        # recoverable error the agent will iterate on — still shown as ✗ so the user sees
        # that the action failed. A non-completed exit (#892: the span unwound before a
        # result existed) is likewise a ✗ — the timing pop and spinner stop above are
        # cleanup and run on every outcome.
        ok = isinstance(event.outcome, Completed) and (
            getattr(event.outcome.result, "exception", None) is None
        )
        mark = "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"
        indent = "  " * (event.depth - 1)
        self._out.write(f"{indent}{mark}\033[2m{elapsed}\033[0m\n")
        self._out.flush()
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        if self._enabled:
            # Defensive stop: normally a no-op (LLMQueryExit fires on every path, #892);
            # covers only a failed exit-emit, whose guarded-close errors are swallowed.
            self._stop_spinner()
        return []

    # -- turn/scope lifecycle ------------------------------------------------
    def finish_turn(self) -> None:
        """End-of-console-line cleanup: stop the spinner, drop per-turn timing state.

        Called via :func:`_finish_progress_turns` after every console line. This is the
        stop path that is *guaranteed* to run when an invoke dies exceptionally
        (Ctrl-C, LLM error) — the abnormal-arm exit events (#892) usually also fire and
        stop it, but this backstop does not depend on them; see the class docstring.
        """
        self._stop_spinner()
        self._exec_started.clear()

    def setup(self) -> None:
        _ACTIVE_PROGRESS.append(self)

    def teardown(self, exc: BaseException | None = None) -> None:
        # Last-resort backstop: whatever happened, never leave a live spinner thread (or a
        # half-drawn spinner line) behind when the hook's scope exits.
        self.finish_turn()
        try:
            _ACTIVE_PROGRESS.remove(self)
        except ValueError:
            pass  # never registered (constructed but not entered) — nothing to remove


# ---------------------------------------------------------------------------
# Line rewriting: typed sugar -> ordinary Python source
# ---------------------------------------------------------------------------
def _as_tstring_literal(text: str) -> str:
    """Splice ``text`` into a well-formed single-quoted ``t"..."`` literal.

    The typed prompt text becomes the *body* of a t-string, so ``{var}`` interpolations
    and ``!conversion``/``:format_spec`` are preserved and handled downstream by
    :func:`jaz.templates.normalize_inputs` (the t-string is passed as the ``task`` input).
    We escape only what would break the literal:

    - ``\\`` -> ``\\\\`` so backslashes in prose stay literal (no accidental ``\\t`` tabs);
    - ``"`` -> ``\\"`` so an embedded quote can't terminate the literal;
    - real newlines -> the two-char ``\\n`` escape, so a multi-line prompt fits in a
      single-quoted literal.

    Escaping ``"`` also escapes quotes *inside* an interpolation expression, so the sugar
    does not support double quotes within ``{...}`` (use single quotes, or explicit
    ``invoke(t"...")``); see the module docstring's known limitations.
    """
    body = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f't"{body}"'


# Recovery guidance appended after the SyntaxError. Deliberately free of the t-string
# mechanism: the ``invoke(t"...")`` rewrite is an implementation detail an iJAZ user never
# sees, so the message names only what they typed (a brace) and how to fix it (#713 review).
_BRACE_RECOVERY = (
    "Double a literal brace as '{{' or '}}', or write '{expr}' to interpolate a value."
)


def _tstring_brace_error(text: str) -> tuple[str, int] | None:
    """Return ``(message, index)`` if ``text`` won't form a valid ``t"..."`` body, else ``None``.

    ``message`` is a concise ``SyntaxError`` description naming the fault (``"unpaired '{'"``,
    ``"unpaired '}'"``, or ``"empty '{}'"`` — whichever applies); ``index`` is the 0-based
    offset of the offending brace, for the caret.

    The console lowers a prompt into ``invoke(t"<text>")`` (see :func:`_as_tstring_literal`),
    so braces are t-string interpolation syntax. A lone or empty brace — natural when talking
    *about* code, which is the console's whole purpose — makes the synthesized literal a
    ``SyntaxError``. Detect that here (empty ``{}``; an unmatched ``{`` or ``}``) so ``push``
    can raise a clean, mechanism-free ``SyntaxError`` instead of the raw traceback the
    synthesized t-string would throw (#547).

    Conservative by design: it flags only the ``SyntaxError`` shapes and never a well-formed
    ``{expr}`` interpolation (that is a *semantic* surprise — binding an unintended input —
    not a syntax error; see the module docstring). ``{{``/``}}`` escapes and nested braces in
    an interpolation (``{x:>{w}}``, ``{d['k']}``) are handled, so a valid prompt is very
    unlikely to trip it.

    Two known gaps, both acceptable for a pre-compile *diagnostic* (the raw t-string error is
    the fallback, never a crash): the depth scan does not skip string literals inside an
    interpolation, so a literal brace inside a quoted string (``{d['}']}``, valid under PEP
    701 in 3.12+) can false-positive; and an empty expression *with* a format spec (``{:>10}``)
    is a real ``SyntaxError`` we let through, since the empty-brace check only fires on a
    genuinely empty ``{}``. Both are rare enough not to warrant a full interpolation tokenizer.
    """
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "}":
            # A `}` reached at the top level (interpolations are consumed by the `{` branch
            # below) is a lone closer unless it's the ``}}`` literal escape — SyntaxError.
            if i + 1 < n and text[i + 1] == "}":
                i += 2
                continue
            return "unpaired '}'", i
        if ch == "{":
            if i + 1 < n and text[i + 1] == "{":
                i += 2  # ``{{`` literal escape
                continue
            # A single ``{`` opens an interpolation; scan to its matching ``}``, depth-counted
            # so nested braces inside the expression / format spec don't confuse the match.
            depth, j = 1, i + 1
            while j < n and depth:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth:
                return "unpaired '{'", i  # unmatched ``{``
            if text[i + 1 : j - 1].strip() == "":
                return "empty '{}'", i  # empty ``{}``
            i = j
            continue
        i += 1
    return None


def _prompt_source(text: str, target: str | None, return_type: str | None) -> str:
    """Build the ``invoke(...)`` source for a prompt, with optional capture/typing.

    The prompt text is passed as the ``task`` input (a t-string), so its ``{var}``
    interpolations mint sibling inputs via :func:`jaz.templates.normalize_inputs` — ``task``
    is an ordinary input now, not a positional prompt (#538).

    - No ``target``: emit a *bare expression* ``invoke(task=t"...")`` so the console's
      ``single``-mode displayhook prints the result and binds ``_`` (the doc's "printed and
      bound to ``_``"), exactly mirroring the stdlib console's own ``_``.
    - With ``target``: emit ``target = invoke(...)``; the left-side annotation (if any) *is*
      the return type — assignment and typing in one operator.

    Return type is expressed as a positional ``ReturnType(T)`` hook (#528), NOT the old
    ``return_type=`` keyword (removed — passing it now raises ``TypeError``). When the user
    gives **no annotation** we *omit* ``ReturnType`` entirely: per #568 that means "no
    return-type contract — the agent may return any value, returned unchecked", which is the
    conversational default the console wants (the displayhook then prints whatever came back).
    We deliberately do NOT emit ``ReturnType(None)``: that would *enforce* a ``None`` return,
    so ``> what is 2+2`` would print nothing / loop. An explicit ``x: T <- ...`` emits a
    leading ``ReturnType(T)`` — leading so ``jaz.invoke``'s typed overload infers ``T``.
    """
    literal = _as_tstring_literal(text)
    call = (
        f"invoke(task={literal})"
        if return_type is None
        else f"invoke(ReturnType({return_type}), task={literal})"
    )
    if target is None:
        return call
    return f"{target} = {call}"


# A parsed single-line sigil: (kind, target, return_type, text).
#   kind == "inspect"  -> `?expr`; text is the expression, target/return_type are None.
#   kind == "settings" -> `% request`; text is the natural-language settings request,
#                         target/return_type are None.
#   kind == "prompt"   -> `> ...` or `target [: T] <- ...`; text is the prompt body,
#                         target/return_type carry the optional capture name/annotation.
# `text` is stripped but keeps a trailing `\` intact so callers can detect continuation.
_ParsedSigil = tuple[str, str | None, str | None, str]


def _parse_sigil(line: str) -> _ParsedSigil | None:
    """Classify one line as a sigil form, or ``None`` if it is plain Python.

    Single source of truth for the ``?``/``>``/``<-`` rules. It deliberately does **not**
    decide continuation/block behavior — that (a trailing ``\\`` or a bare ``>`` block) is
    stateful and lives only in :meth:`JazConsole.push`. Everything that needs to recognize a
    sigil (:func:`_translate_line` for the one-shot/file path, :func:`_is_sugar`, and
    ``push`` itself) routes through here so the classification can never drift between the
    interactive and non-interactive paths — the previous duplication was a standing
    lockstep-edit hazard.
    """
    stripped = line.lstrip()
    if stripped.startswith("?"):
        # Inspect: print the value's jaz.describe description (or compact repr). Leading
        # `?` (not IPython's trailing `expr?`) keeps the whole-line rule uniform with `>`/`<-`.
        return ("inspect", None, None, stripped[1:].strip())
    if stripped.startswith("%"):
        # Settings helper: `% <natural language>` proposes (and, after confirmation, runs)
        # a settings snippet — see `jaz_settings`. `%` is collision-free as a line start
        # (it is a binary operator, so no Python statement can begin with it), the same
        # guarantee class as `>`/`?`. The sigil itself was an executive call by the user
        # (chosen over `>?` and a plain `settings()` function) for its IPython-magic
        # familiarity.
        return ("settings", None, None, stripped[1:].strip())
    if stripped.startswith(">"):
        return ("prompt", None, None, stripped[1:].strip())
    m = _CAPTURE_RE.match(line)
    if m:
        target, rtype, text = m.group(1), m.group(2), m.group(3)
        return ("prompt", target, rtype.strip() if rtype else None, text.strip())
    return None


def _translate_line(line: str) -> str | None:
    """Translate a single typed line into Python source, or ``None`` if it is plain Python.

    Handles the non-continuation forms used by both the one-shot path and the console's
    simple (single-line) path. Multi-line prompt continuation (a trailing ``\\`` or a bare
    ``>`` block) is handled by :class:`JazConsole.push`, which assembles the full text and
    then calls :func:`_prompt_source` directly.
    """
    parsed = _parse_sigil(line)
    if parsed is None:
        return None
    kind, target, rtype, text = parsed
    if kind == "inspect":
        return f"jprint({text})"
    if kind == "settings":
        return _settings_source(text)
    return _prompt_source(text, target, rtype)


def _is_sugar(line: str) -> bool:
    """Whether ``line`` is a recognized console sigil line (vs. plain Python / bare prompt)."""
    return _parse_sigil(line) is not None


def _empty_body_hint(kind: str) -> str:
    """A short, friendly message for a lone ``?`` / ``>`` (empty inspect/prompt body).

    A bare sigil with no body is almost always a fat-finger, and lowering it verbatim is
    user-hostile: ``?`` alone becomes ``jprint()`` (``TypeError: missing 1 required
    positional argument``, printed as a full traceback) and ``>`` alone on the one-shot/file
    path becomes ``invoke(t"", ...)`` (an empty prompt actually sent to the agent). Callers
    detect the empty body and print this hint as a no-op instead. Kept out of ``_parse_sigil``
    on purpose: classification stays a pure predicate, and the empty-body policy differs per
    caller (in interactive :meth:`JazConsole.push` a lone ``>`` is *not* empty — it opens a
    multi-line prompt block, an intended feature — so only ``push``'s inspect branch and the
    one-shot path treat an empty body as an error).
    """
    if kind == "inspect":
        return "?: nothing to inspect — use `?<expr>`, e.g. `?doc`"
    if kind == "settings":
        return (
            "%: say what you need from jaz — e.g. `% allow the agent to import numpy` "
            "or `% why did my invoke time out?`"
        )
    return ">: empty prompt — put your request after `>`, e.g. `> summarize {doc}`"


class JazConsole(code.InteractiveConsole):
    """A Python console that rewrites the conversational sigils into ``invoke(t"...")``.

    Subclasses :class:`code.InteractiveConsole` and overrides :meth:`push` to preprocess
    each input line. Non-sigil lines are forwarded untouched, so this is a strict superset
    of the stdlib console. Two pieces of console-only state implement multi-line prompts
    (intent 5 in the design doc):

    - ``_cont``: a ``>``/``<-`` line ending in ``\\`` continues the prompt onto the next
      line;
    - ``_block``: a bare ``>`` on its own line opens a prompt block terminated by a blank
      line.

    Both mirror how the stdlib console already handles continuations, and both buffer the
    typed text *before* rewriting, so a multi-line prompt becomes one ``invoke(t"...")``.

    ``errored`` tracks whether any line raised (set via the ``showtraceback`` /
    ``showsyntaxerror`` overrides). It exists so the *script* runner can fail fast like
    ``python file.py``; the interactive loop ignores it and stays shell-forgiving.
    """

    def __init__(
        self, locals: dict[str, Any] | None = None, filename: str = "<stdin>"
    ) -> None:
        # filename defaults to "<stdin>", overriding InteractiveConsole's own "<console>", so
        # every error this console prints — genuine SyntaxError, runtime traceback, and the
        # synthesized brace diagnostic below — names the source exactly as the real `python`
        # REPL does (which uses "<stdin>" for both interactive and piped input). The banner
        # sells this as "a real Python REPL", so anything a plain REPL shows should show up
        # identically; "<console>" was a `code` module divergence, not the REPL convention.
        # (#713 review.) A `.jaz` *file* passes its path instead, so its tracebacks name the
        # file the way `python file.py` does rather than "<stdin>" (see _run_file).
        #
        # Known limitation: line numbers stay *statement*-relative (each pushed statement
        # compiles as line 1), so a file traceback names the right file but not the right line
        # — the line-by-line sugar preprocessor collapses multi-line prompts/blocks into one
        # synthesized statement, so a file-absolute offset isn't recoverable here. Pre-existing
        # (the "<stdin>" path already showed line 1); the filename is the recoverable half.
        super().__init__(locals=locals, filename=filename)
        # (target, return_type, accumulated_text) while continuing a `\`-terminated prompt.
        self._cont: tuple[str | None, str | None, str] | None = None
        # Accumulated lines while inside a bare-`>` prompt block (None = not in a block).
        self._block: list[str] | None = None
        # Set True by showtraceback/showsyntaxerror when a line errors. Read ONLY by the
        # script runner (:func:`_run_script_source`) to fail fast + exit nonzero like
        # `python file.py`. The interactive loop (`.interact()`) never reads it, so a bad
        # line there just prints and the session continues — matching the stdlib `>>>` shell.
        self.errored = False

    # ``code.InteractiveInterpreter`` funnels *both* error kinds through these two hooks:
    # syntax errors via ``showsyntaxerror``, runtime exceptions via ``showtraceback`` (a
    # rewritten `> ...` whose ``invoke`` raises lands here too). Recording the error in one
    # place — rather than trying to read a status out of ``push`` (which returns "need more
    # input", not "errored") — is what lets the script runner detect a failed line.
    def showtraceback(self, *args: Any, **kwargs: Any) -> None:
        self.errored = True
        super().showtraceback(*args, **kwargs)

    def showsyntaxerror(self, *args: Any, **kwargs: Any) -> None:
        self.errored = True
        super().showsyntaxerror(*args, **kwargs)

    def runcode(self, code: Any) -> None:
        # Reimplements the (tiny) stdlib runcode — exec + SystemExit passthrough +
        # showtraceback — instead of delegating, for two console-UX reasons:
        #
        # 1. The `finally` is the guaranteed end-of-turn signal for the live progress
        #    display: a Ctrl-C (or LLM error) aborts a running invoke with NO hook exit
        #    event, which would otherwise orphan a spinner thread that keeps overdrawing
        #    the prompt — the "I can't quit this" symptom. See _finish_progress_turns.
        # 2. KeyboardInterrupt gets its own branch: it is a *user action*, not a bug, but
        #    the stdlib path runs it through showtraceback — and an interrupted invoke's
        #    stack is ~50 frames of jaz/tenacity/httpx internals, so Ctrl-C printed a
        #    wall of traceback (observed live). We stop the spinner FIRST (so no frame
        #    interleaves with the message, and the ^C echo on the spinner line is wiped
        #    by its line-clear), then print one line. `errored` is still set so a `.jaz`
        #    script Ctrl-C'd mid-turn fails fast like any other error.
        try:
            exec(code, self.locals)  # type: ignore[arg-type]  # mirrors stdlib runcode
        except SystemExit:
            raise
        except KeyboardInterrupt:
            _finish_progress_turns()
            self.errored = True
            print("\nKeyboardInterrupt — turn aborted.", file=sys.stderr)
        except BaseException:  # noqa: BLE001 - mirrors the stdlib's bare `except:`
            self.showtraceback()
        finally:
            _finish_progress_turns()

    def _push_prompt(
        self,
        text: str,
        target: str | None,
        rtype: str | None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Lower one prompt to ``invoke(t"...")`` and push it — but if its braces would make
        the synthesized t-string a ``SyntaxError``, report that as a clean ``SyntaxError``
        instead of the raw traceback (#547). Centralized so every prompt path (bare-``>``
        block, ``\\`` continuation, single line, ``flush``) gets the same diagnostic."""
        fault = _tstring_brace_error(text)
        if fault is not None:
            self._report_brace_syntaxerror(*fault, text)
            return False
        return super().push(_prompt_source(text, target, rtype), *args, **kwargs)

    def _report_brace_syntaxerror(self, message: str, index: int, text: str) -> None:
        """Render a lone/empty brace as a normal REPL ``SyntaxError``, then the recovery
        guidance — the friendly diagnostic of #547, reshaped per #713 review to *be* a
        SyntaxError rather than a bespoke ``jaz console:`` line.

        Routed through :meth:`showsyntaxerror` (not a hand-built string) so it renders exactly
        like any other REPL syntax error — a ``File "…"`` header naming ``self.filename``
        (``<stdin>`` for stdin/interactive, the script path for a ``.jaz`` file), source line,
        caret — under the same filename a genuine error would use, and so the override sets
        ``errored``, letting a ``.jaz`` script fail fast like ``python file.py`` does on a real
        ``SyntaxError``. ``index`` is an offset into the whole (possibly multi-line) prompt, so
        resolve it to a 1-based line/column for the caret."""
        line_start = text.rfind("\n", 0, index) + 1
        line_end = text.find("\n", index)
        line_end = len(text) if line_end == -1 else line_end
        lineno = text.count("\n", 0, index) + 1
        col = index - line_start + 1
        # showsyntaxerror reads sys.exc_info(), so raise-then-catch to give it a live
        # exception; pass self.filename so it matches every other error this console prints.
        try:
            raise SyntaxError(
                message, (self.filename, lineno, col, text[line_start:line_end])
            )
        except SyntaxError:
            self.showsyntaxerror(self.filename)
        self.write(_BRACE_RECOVERY + "\n")

    def push(self, line: str, *args: Any, **kwargs: Any) -> bool:  # type: ignore[override]
        # If the stdlib console is mid multi-line *Python* statement (e.g. inside a `def`),
        # never re-interpret its continuation lines as sugar — forward them untouched.
        if self.buffer:
            return super().push(line, *args, **kwargs)

        # Inside a bare-`>` block: a blank line closes it and runs the assembled prompt.
        if self._block is not None:
            if line.strip() == "":
                text = "\n".join(self._block)
                self._block = None
                return self._push_prompt(text, None, None, *args, **kwargs)
            self._block.append(line)
            return True

        # Continuing a `\`-terminated prompt line.
        if self._cont is not None:
            target, rtype, text = self._cont
            if line.endswith("\\"):
                self._cont = (target, rtype, text + "\n" + line[:-1])
                return True
            self._cont = None
            full = text + "\n" + line
            return self._push_prompt(full, target, rtype, *args, **kwargs)

        # Classify the line with the shared single-line rules; `push` only adds the
        # continuation/block state on top (the classification itself lives in `_parse_sigil`).
        parsed = _parse_sigil(line)
        if parsed is None:
            return super().push(line, *args, **kwargs)  # plain Python

        kind, target, rtype, text = parsed

        # `?expr` — no continuation, no t-string.
        if kind == "inspect":
            # A lone `?` (empty body) would lower to `jprint()` and raise a `TypeError`
            # traceback; print a hint and treat it as a no-op. (Unlike `>`, `?` has no
            # multi-line-block form, so an empty inspect is unambiguously a typo.)
            if not text:
                print(_empty_body_hint(kind), file=sys.stderr)
                return False
            return super().push(f"jprint({text})", *args, **kwargs)

        # `% request` — jaz helper. Like `?`, no continuation/block form: the request
        # is single-line prose passed verbatim (a trailing `\` stays part of the text
        # rather than opening a continuation), and a lone `%` is unambiguously a typo.
        if kind == "settings":
            if not text:
                print(_empty_body_hint(kind), file=sys.stderr)
                return False
            return super().push(_settings_source(text), *args, **kwargs)

        # Bare `>` alone (no capture target, empty body) opens a multi-line prompt block.
        if target is None and text == "":
            self._block = []
            return True

        # A trailing `\` starts a continuation (applies to both `>` and `<-` capture forms).
        if text.endswith("\\"):
            self._cont = (target, rtype, text[:-1])
            return True

        return self._push_prompt(text, target, rtype, *args, **kwargs)

    def flush(self) -> bool:
        """Finalize a prompt left buffered at end-of-input; return ``False`` if none.

        Called for finite *script* input — a ``.jaz`` file and piped stdin, both via
        :func:`_run_script_source` — *not* the interactive loop (which gets its closing
        newline from the user) nor ``-c``/one-shot (:func:`_run_oneshot` never builds a
        :class:`JazConsole`, so it has nothing to flush). A file can end mid
        ``\\``-continuation or inside an unclosed bare-``>``
        block with no trailing blank line to close it — interactively the user supplies that
        closing newline, but a file does not, so without an explicit flush the last prompt
        would be silently dropped (no output, no error).

        Limitation: ``flush()`` only finalizes the *sigil* buffers (``_block``/``_cont``). A
        script that ends mid plain-*Python* statement (an open ``def``/paren) still drops
        that trailing statement — that buffer belongs to the stdlib console, and leaving it
        mirrors :meth:`code.InteractiveConsole.interact`, which discards it the same way.
        """
        if self._block is not None:
            text = "\n".join(self._block)
            self._block = None
            return self._push_prompt(text, None, None)
        if self._cont is not None:
            target, rtype, text = self._cont
            self._cont = None
            return self._push_prompt(text, target, rtype)
        return False


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------
#: Console history lives beside the settings and credentials files, in ``~/.jaz/``.
_HISTORY_FILENAME = "history"

# The pre-``~/.jaz/`` location ``~/.jaz_repl_history`` is intentionally NOT migrated:
# on first run after upgrading, history simply starts empty and the old dotfile is left
# untouched next to the new directory. We deliberately don't rename-on-first-use (a silent
# write to $HOME, and it would need a rule for when both files exist) or read the old path
# as a permanent fallback (dead code that lives forever). The one-time loss of shell
# history is cheap enough to accept; this comment stands in for a banner/release note so
# the loss is documented rather than silent.


def _history_file() -> str:
    """Path of the readline history file, creating ``~/.jaz`` if needed."""
    # Resolved per call, not as a module constant, because `config_dir()` reads
    # JAZ_CONFIG_DIR at call time — a constant computed at import would freeze whatever the
    # environment looked like when `jaz.console` was first imported, which in the test suite
    # is before any fixture has redirected it.
    return str(ensure_config_dir() / _HISTORY_FILENAME)


#: Where to send a user whose provider tag matched nothing close enough to suggest. Naming the
#: canonical list beats printing an arbitrary slice of ~150 tags into a terminal.
_PROVIDER_LIST_URL = "https://docs.litellm.ai/docs/providers"


def set_credential(provider: str, api_key: str | None = None) -> None:
    """Store an API key for ``provider`` so later jaz sessions can use it.

    Writes ``~/.jaz/credentials.json`` (mode ``0o600``); see :mod:`jaz.credentials`. The
    key takes effect from the next agent run — no restart needed.

    Called with one argument it prompts for the key without echoing it::

        >>> set_credential("openai")
        API key for openai: <not shown>
        Stored openai api_key in /home/you/.jaz/credentials.json

    Passing ``api_key`` directly is supported for scripts, but avoid it at an interactive
    prompt: the line is saved to the console's history file in plain text.
    """
    # Prompting is the DEFAULT (rather than requiring the key as an argument) specifically
    # because of that history file. The console persists every typed line to
    # ~/.jaz/history via readline's atexit hook, so `set_credential("openai", "sk-...")`
    # writes the secret to a plaintext file the user will never think to clean — the exact
    # accident this feature exists to prevent. `getpass` keeps the key off the screen AND
    # out of the history, since it reads from the tty rather than through readline.
    #
    # A non-tty stdin is rejected instead of falling through to getpass's stdin fallback:
    # in a piped script (`cat setup.jaz | jaz`) getpass would silently swallow the NEXT
    # LINE OF THE SCRIPT as the key and store it. Failing loudly is the only safe reading.
    #
    # Reaching for the underscore-private `_set_credential` is deliberate, not a layering slip:
    # the store's published surface is read-only, and this console binding is the ONE sanctioned
    # writer (see the rule beside `"credentials"` in `jaz/__init__.py:__all__` — library code
    # reads ambient credentials, only the CLI writes them). That privacy is convention, not
    # enforcement: no warning fires on `jaz.credentials._set_credential`, because `credentials`
    # is public and PEP 562 `__getattr__` cannot intercept a name the module actually defines.
    # The underscore carries the same weight here as everywhere else in the package.
    from jaz.credentials import _set_credential as _store_credential

    # Reject an unknown provider tag BEFORE prompting/storing. Without this a typo
    # (`set_credential("opeani")`) prompts, stores, and prints "Stored opeani api_key in …"
    # — reporting success — and the user then hits "OpenAI API key not found" on every run
    # afterwards with nothing connecting the two. For a one-per-machine setup step a
    # silent-success-that-does-nothing is the worst failure mode, so it fails here instead.
    # Checked before the prompt so a typo never costs a getpass round-trip whose result is
    # thrown away.
    #
    # EXECUTIVE CALL (user, 2026-08-15) — the valid set is EXACTLY the upstream API providers,
    # taken from LiteLLM, and deliberately NOT unioned with `registered_llm_tags()`. A credential
    # is keyed by the vendor a request is authenticated against, which is never a JAZ backend tag:
    # `resolve_credential` is only ever called with `"openai"`, `"anthropic"`, or the provider
    # LiteLLM routed to. `litellm`/`rlm`/`sglang` are backend tags, so accepting them let
    # `set_credential("litellm", …)` report success and then do nothing forever — the exact
    # silent-success-that-does-nothing this guard exists to prevent, and a very plausible thing
    # for a default-backend user to type.
    #
    # The cost, accepted knowingly: a custom `@register_llm` backend that resolves a credential
    # under its OWN tag can no longer have that key stored from the console. It is still readable
    # (`resolve_credential` is public) — the file just has to be edited by hand. Weighed against
    # every user of the default backend being able to store an inert key, that is the better trade.
    #
    # Importing litellm costs ~2 s; acceptable for a once-per-machine setup call, and it is why
    # this import is inside the function rather than at module scope.
    from jaz.llm._litellm import _litellm

    # No `getattr` defaults on either lookup. A default here would not be defensive, it would be
    # a silent re-run of the bug this guard was just fixed for: if LiteLLM renames `provider_list`
    # or stops using an enum, falling back to `()` would reject every real provider with nothing
    # to say why. Better to fail loudly at the one call site.
    #
    # `models_by_provider` is unioned in because it is not a subset: it carries five legacy tags
    # (`aleph_alpha`, `anyscale`, `azure_anthropic`, `llamagate`, `palm`) that `provider_list`
    # dropped. The set is meant to be everything LiteLLM will route, so a superset is the safe
    # direction — a tag we accept but LiteLLM never asks for is an unread file entry, whereas one
    # we reject is a user blocked from storing a key that would have worked.
    #
    # Elements are unwrapped with an explicit `isinstance` rather than `getattr(p, "value", p)`
    # because LiteLLM's own type for `provider_list` is `list[str] | list[LlmProviders]` — both
    # shapes are declared, so this handles a real union instead of papering over a surprise. A
    # `getattr` default would additionally turn any *third* shape into its repr, seeding the set
    # with garbage tags that then reject the real ones.
    litellm = _litellm()
    known = {
        p.value if isinstance(p, Enum) else p for p in litellm.provider_list
    } | set(litellm.models_by_provider)
    if provider not in known:
        # A near-miss is the whole point of the message: an alphabetical slice of ~150 providers
        # can never contain `"openai"` for a `"opeani"` typo, so listing one is useless for the
        # case this message exists for.
        #
        # Matched on the case-folded tag because `get_close_matches` is case-sensitive and every
        # known tag is lowercase, so `"OpenAI"` would otherwise score no match at all — the one
        # near-miss most likely to come from a human typing a vendor's brand capitalisation.
        # Suggested rather than silently accepted: the tag is the storage key, so normalising it
        # behind the user's back would store under a name they did not choose.
        suggestions = difflib.get_close_matches(provider.lower(), sorted(known), n=3)
        hint = (
            f" Did you mean {' or '.join(repr(s) for s in suggestions)}?"
            if suggestions
            else f" See {_PROVIDER_LIST_URL} for the {len(known)} known providers."
        )
        raise ValueError(f"Unknown provider {provider!r}.{hint}")

    if api_key is None:
        if not sys.stdin.isatty():
            raise ValueError(
                "Cannot prompt for a key with stdin redirected. Pass it explicitly: "
                f"set_credential({provider!r}, api_key=...)"
            )
        api_key = getpass.getpass(f"API key for {provider}: ")

    location = _store_credential(provider, api_key)
    # Confirm with the location, never the value — echoing a key someone just took care not
    # to display would defeat the prompt, and the console's output is what gets screen-shared.
    # `location` is a printable string rather than a Path so an OS-keyring backend (#1076)
    # can name a keychain here without changing this call site.
    print(f"Stored {provider} api_key in {location}")


def build_namespace() -> dict[str, Any]:
    """The console namespace: a real REPL with ``jaz``/``invoke``/``jprint`` pre-wired.

    **Public API — compose, don't rebuild.** The sugar rewrites to source that references
    *five* names: ``jaz``, ``invoke``, ``jprint``, ``ReturnType`` (a typed ``x: T <- ...``
    lowers to ``invoke(ReturnType(T), task=...)`` — see :func:`_prompt_source`), and
    ``__jaz_settings__`` (a ``% request`` lowers to ``__jaz_settings__(..., globals())`` —
    see :func:`_settings_source`). A consumer that wants the console with extra bindings
    (e.g. layering tracing hooks) must keep all five or the sugar breaks at runtime —
    ``?x`` raises ``NameError`` without ``jprint``, a typed capture raises ``NameError``
    without ``ReturnType``, and ``% ...`` without ``__jaz_settings__``. This function is
    public precisely so such consumers write ``build_namespace() | {my overrides}`` instead
    of hand-rebuilding the dict and silently dropping a key (which is exactly the bug the
    tracing example hit). Exposing it keeps the contract in one place rather than duplicated
    across callers that will drift.

    ``set_credential`` is also bound, so a key can be stored from inside the session.

    No hooks are placed in the namespace by default (req 5 — no external-service deps out
    of the box). The tracing example layers hooks on top via ``examples/interactive_repl.py``.
    """
    return {
        "jaz": jaz,
        "invoke": jaz.invoke,
        "jprint": jprint,
        "ReturnType": ReturnType,
        # Dunder-ish name on purpose: it is rewrite plumbing (like `_`/`__builtins__`),
        # not a name users are meant to call; the user-facing spelling is the `%` sigil.
        "__jaz_settings__": jaz_settings,
        # Bound under a plain name (no sigil) deliberately. Storing a key is a rare,
        # one-per-machine setup step, not a per-turn action, so it does not earn one of the
        # few legal-Python-shadowing characters the sugar can claim — and unlike the sigils
        # it needs no rewrite, being an ordinary call the user can also make from a script.
        # Unlike the four names above it is a convenience rather than a sugar dependency:
        # dropping it breaks nothing the console rewrites.
        "set_credential": set_credential,
    }


def _apply_user_settings() -> str | None:
    """Apply ``~/.jaz/settings.json`` to the session config; return an error message if it is
    malformed, else ``None``.

    The persistent settings file is **console-scoped**: it is an ambient baseline for the ``jaz``
    console, NOT for library ``import jaz`` (which stays fully determined by its call site, so code
    written on top of jaz is portable and cannot be broken by a file in someone's home directory).
    So the file is read here, at console startup, rather than in ``Config.__init__``.

    Applied before :func:`_apply_config` so a ``--model`` flag overrides the file, and before any
    invoke so the whole session sees it. On a malformed file the console fails fast: `main` prints
    the returned message (which names the file and the fix) and exits non-zero — you repair the
    file in a text editor. (A future console helper-agent that can repair it in-session is #1070.)
    """
    # `load_user_settings` raises (path + recovery hint) on unreadable/invalid-JSON/non-object and
    # on a blocked key (llm.base_url/api_key); `build_config`/`configure` raise on a value the
    # schema rejects. Both become a returned message rather than a traceback — a broken settings
    # file is a user error, not a jaz bug.
    #
    # A sandbox-loosening key is NOT fatal: `load_user_settings` emits it as a UserWarning (allowed
    # but flagged, per #1047). Capture it here and print to stderr as a non-fatal notice rather than
    # letting Python's default warning filter decide whether the user ever sees it — the whole point
    # is that loosening the sandbox is never silent, and this is the interactive session where the
    # agent's output lands.
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            partials = load_user_settings()
    except ValueError as exc:
        return str(exc)
    for entry in caught:
        print(f"jaz: warning: {entry.message}", file=sys.stderr)
    if not partials:
        return None
    try:
        jaz.configure(**build_config(partials))
    except (TypeError, ValueError) as exc:
        return f"{settings_path()}: {exc}\n{SETTINGS_RECOVERY_HINT}"
    return None


def _apply_config(args: argparse.Namespace) -> None:
    """Translate startup flags into ``jaz.configure(...)`` calls.

    NOTE (doc-vs-reality): the design doc / old example wrote
    ``jaz.configure(model=..., max_cost_budget=...)``, but ``Config`` has no top-level
    ``model`` — it is a constructor parameter of the backend. ``--model`` is a string the user
    typed, so it is compiled through :func:`~jaz.instantiate.build_config` and ``configure``
    receives the built backend. A combined ``"provider/model"`` spelling is carried into the
    request as written; it does NOT select the backend. The other three knobs are not
    ``configure`` fields at all
    and are applied as scoped hooks in :func:`_build_hooks` instead: a USD cost budget
    is ``jaz.hooks.BudgetPool`` and a nesting ceiling is ``jaz.hooks.RecursionLimit``. In
    particular there is no ``max_repl_recursion`` config field — current jaz imposes no
    recursion ceiling except via that opt-in hook (an earlier draft of this console targeted a
    since-removed ``max_repl_recursion`` field; passing it to ``configure`` now raises).
    """
    if args.model is not None:
        # Through the boundary: `--model` is authored data (a string a user typed), and config
        # takes only built components.
        #
        # `--model` names only the model; it rides the config's default backend — litellm, v1's
        # sole backend — rather than naming one, so `--model openai/gpt-5-mini`. A future
        # `--backend` flag would feed the backend slot.
        llm = build_config({"llm": {"model": args.model}})["llm"]
        # Validated at startup rather than at the first request: `get_model` rejects an empty id,
        # and the backend's `validate_model` rejects an id it would not send verbatim. For the
        # litellm backend the `provider/` prefix IS the routing key, so a prefixed id is expected
        # here — the opposite of the built-in backends, which rejected it.
        llm.validate_model(llm.get_model())
        jaz.configure(llm=llm)


def _build_hooks(args: argparse.Namespace) -> list[object]:
    """Build the opt-in hooks from flags. Bare ``jaz`` returns ``[]`` (no hooks).

    ``--max-cost``/``--max-calls`` and ``--max-recursion`` surface as scoped hooks (not
    ``configure`` fields) because on current jaz the underlying knobs *are* hooks: a budget
    is ``jaz.hooks.BudgetPool`` and a nesting ceiling is ``jaz.hooks.RecursionLimit``. Both are
    entered for the whole session / one-shot run by :func:`main` via its ``ExitStack``, so a
    ``--max-recursion 1`` run makes the top agent a leaf (no nested ``jaz.invoke``).
    """
    hooks: list[object] = []
    # One BudgetPool carries both budgets — they are two ceilings on the same pool, not two
    # hooks. --max-calls is the pricing-free ceiling: BudgetPool fails a cost budget closed on
    # a model the price table can't price (ModelPricingUnavailableError), so a calls budget is
    # the way to bound a model too new for the table without disabling enforcement.
    if args.max_cost is not None or args.max_calls is not None:
        from jaz.hooks import BudgetPool

        hooks.append(BudgetPool(cost_budget=args.max_cost, calls_budget=args.max_calls))
    if args.max_recursion is not None:
        from jaz.hooks import RecursionLimit

        hooks.append(RecursionLimit(max_depth=args.max_recursion))
    if args.log is not None:
        from jaz.hooks import FileLogger

        hooks.append(FileLogger(args.log))
    if args.trace:
        # Imported lazily so the `tracing` extra is only needed when --trace is used.
        try:
            from jaz.hooks import JaegerTracingHook
        except ImportError as exc:  # pragma: no cover - exercised manually
            raise SystemExit(
                "--trace requires the tracing extra. Install it with "
                "`pip install jaz[tracing]` and start a Jaeger collector "
                "(see examples/interactive_repl.py)."
            ) from exc
        hooks.append(JaegerTracingHook())
    return hooks


def _build_progress(args: argparse.Namespace) -> ConsoleProgress | None:
    """The live progress hook for this run, or ``None`` when it should be off.

    Off when ``--no-progress`` is passed or stdout is not a tty (piped/redirected runs get
    clean output with no ANSI/thread overhead). A separate factory (rather than inline in
    ``main``) so the flag→hook mapping is unit-testable like :func:`_build_hooks`; kept
    *out* of ``_build_hooks`` because it is default-ON UI, not an opt-in governance hook.
    """
    # Guard the isatty probe exactly as ConsoleProgress.__init__ does: a replaced/wrapped
    # sys.stdout without an isatty attribute should disable progress, not crash main().
    isatty = getattr(sys.stdout, "isatty", None)
    stdout_is_tty = bool(isatty()) if callable(isatty) else False
    if args.no_progress or not stdout_is_tty:
        return None
    return ConsoleProgress()


def _run_source(source: str, namespace: dict[str, Any]) -> int:
    """Compile+exec a single rewritten line in ``single`` mode and return an exit status.

    ``single`` mode routes a bare ``invoke(t"...")`` expression through ``sys.displayhook``,
    so the result is printed and ``_`` is bound exactly as in the interactive console.
    Returns 0 on success, 1 if execution raised.
    """
    # filename="<string>" to match `python -c` (which only ``-c``/one-shot reaches — see
    # _run_oneshot), so a traceback from a one-shot turn names its source the way a plain
    # `python -c` would, not with a jaz-internal "<jaz>" (#713 review).
    try:
        exec(compile(source, "<string>", "single"), namespace)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        # Mirror JazConsole.runcode: a Ctrl-C is a user action — stop the spinner first
        # (no frame interleaves; its line-clear wipes the ^C echo), then one line
        # instead of the ~50-frame jaz/httpx traceback. Nonzero exit: the turn did not
        # complete.
        _finish_progress_turns()
        print("\nKeyboardInterrupt — turn aborted.", file=sys.stderr)
        return 1
    except BaseException:  # noqa: BLE001 - mirror the console: report, don't crash the CLI
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Same end-of-turn cleanup as JazConsole.runcode: an exceptionally-ended invoke
        # fires no hook exit event, so this is what stops an orphaned progress spinner
        # on the one-shot (`-c`) path.
        _finish_progress_turns()
    return 0


def _run_oneshot(text: str, namespace: dict[str, Any]) -> int:
    """Run a single turn (``-c PROMPT``) and exit.

    A sigil line is translated like an interactive line; any other text is treated as a
    bare prompt (implicit ``>``), since the whole point of one-shot mode is to ask one
    thing. ``{...}`` has no live session to bind from, so it resolves against the fresh
    namespace -> undefined-name error (interpolation is effectively disabled here).

    Only ``-c`` uses this path. Piped stdin is *script* input and goes through
    :func:`_run_script_source` (line-by-line, non-sigil lines run as Python), so a bare
    piped line is not treated as a prompt — see that function and :func:`main`.
    """
    if _is_sugar(text):
        parsed = _parse_sigil(text)
        assert parsed is not None  # _is_sugar guarantees a classification
        # A lone `?`/`>` has an empty body: `?` would lower to `jprint()` (TypeError) and `>`
        # to `invoke(t"", ...)` (an empty prompt sent to the agent). There is no block form in
        # one-shot mode, so print a hint and no-op instead. Nonzero exit signals "no turn ran".
        if not parsed[3]:
            print(_empty_body_hint(parsed[0]), file=sys.stderr)
            return 1
        source = _translate_line(text)
        assert source is not None  # _is_sugar guarantees a translation
    else:
        source = _prompt_source(text.strip(), None, None)
    return _run_source(source, namespace)


def _run_script_source(
    contents: str, namespace: dict[str, Any], filename: str = "<stdin>"
) -> int:
    """Feed multi-line *script* text through the console preprocessor line-by-line, then flush.

    ``filename`` names the source in tracebacks: the default ``"<stdin>"`` suits piped stdin
    (matching ``cat x | python``), while :func:`_run_file` passes the script's path so a
    ``.jaz`` file reports like ``python file.py`` (line numbers stay statement-relative — see
    :class:`JazConsole`).

    Shared by :func:`_run_file` (a ``.jaz`` path) and the piped-stdin branch of :func:`main`
    so the two are identical: ``cat script.jaz | jaz`` behaves exactly like ``jaz script.jaz``
    — each line is classified for sigils, plain-Python lines run as usual, and a dangling
    trailing prompt is flushed at EOF. Previously piped stdin went through :func:`_run_oneshot`
    and collapsed the whole input into one bare ``invoke(t"...")``, so the same bytes ran with
    very different semantics depending on file-vs-pipe; routing both here removes that surprise
    (the reviewer's "someone piping a ``.jaz`` script would be surprised").

    This is deliberately **not** one-shot semantics: a non-sigil line runs as plain Python
    (it is a *script*, not a single prompt), so ``echo "capital of France" | jaz`` is a Python
    ``SyntaxError``, not an agent turn. Sending one ad-hoc line to the agent is what ``-c`` is
    for (``jaz -c "capital of France"``). Keeping bare-text-as-Python also honors the module's
    "no chat mode where bare text becomes a prompt" principle uniformly across file and pipe.

    Errors are **fatal (settled with the user):** a ``.jaz`` script should mirror
    ``python file.py`` — the first line that raises stops the run and returns a nonzero exit,
    and later lines do **not** execute. This is a deliberate departure from the forgiving
    interactive ``>>>`` loop (which prints the error and keeps going): a script/CI consumer
    must be able to tell a run failed, and it also matches the ``-c``/one-shot path, which
    already returns 1 on error. The alternative — run every line and only exit nonzero at the
    end — was considered and rejected: running lines *after* a failed one is exactly the
    surprise (acting on a half-broken state) that ``python file.py`` avoids by stopping.
    """
    console = JazConsole(locals=namespace, filename=filename)
    for line in contents.splitlines():
        console.push(line)
        if console.errored:
            return (
                1  # fail fast, like `python file.py`: stop here, don't run later lines
            )
    # A script with no trailing blank line can leave a `\`-continuation or bare-`>` block
    # buffered; flush it so the final turn runs instead of being silently dropped at EOF.
    # flush() runs one last turn, which can itself error — hence the final `errored` check.
    console.flush()
    return 1 if console.errored else 0


def _run_file(path: str, namespace: dict[str, Any]) -> int:
    """Execute a script file line-by-line through the console preprocessor, then exit.

    Lets a ``.jaz`` script use the sigil sugar; plain-Python lines run as usual. Multi-line
    Python constructs are supported because :class:`JazConsole` defers to the stdlib
    console's own buffering for non-sigil lines.
    """
    try:
        with open(path) as f:
            contents = f.read()
    except OSError as exc:
        print(f"jaz: cannot open {path!r}: {exc}", file=sys.stderr)
        return 1
    # Pass `path` as the traceback filename (as typed on the command line, like `python
    # file.py`) so a file error names the file, not the piped-stdin default "<stdin>".
    return _run_script_source(contents, namespace, filename=path)


def _interactive(namespace: dict[str, Any]) -> int:
    """Print the banner, wire up readline history/completion, and enter the console."""
    # The colored ASCII art + typing animation write raw ANSI and force ~1s of `time.sleep`.
    # Skip both when stdout is not a terminal (redirected/piped, or a dumb terminal): there
    # the escape codes are visible garbage and the sleeps only delay a non-interactive
    # consumer. The plain-text `banner` below still prints via `interact`.
    if sys.stdout.isatty():
        print_colored_ascii()
        animate_flavor_text(random.choice(FLAVOR_TEXTS))

    banner = (
        "Interactive jaz console — a real Python REPL with conversational sugar.\n"
        "\n"
        "  > summarize {doc}            ask the agent (binds `doc` as input)\n"
        "  n: int <- how many words?    capture a typed result\n"
        "  ?doc                         show what the agent sees for `doc`\n"
        "  % allow numpy imports        ask the jaz helper — questions are answered,\n"
        "  % why did that time out?     changes are confirmed before running\n"
        "  <any Python>                 runs as normal Python\n"
        "\n"
        "  set_credential('openai')     save an API key for future sessions (prompts)\n"
        "\n"
        "jaz, invoke and jprint are pre-imported. Type exit() or Ctrl-D to quit."
    )

    # readline may be unavailable (e.g. on some Windows setups); the console still works.
    # OSError joins it now that the history file lives in a directory we have to create:
    # a read-only or full HOME must cost the user their history, not their console. The old
    # path wrote straight into an existing HOME and so had nothing to fail at this point.
    try:
        import readline

        history_file = _history_file()
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass
        readline.set_history_length(1000)
        atexit.register(readline.write_history_file, history_file)
        readline.parse_and_bind("tab: complete")
    except ImportError:  # pragma: no cover - platform dependent
        pass
    # OSError is exercised by test_an_unwritable_home_costs_history_not_the_console,
    # so it stays un-pragma'd; only the platform-dependent ImportError is uncovered.
    except OSError:
        pass

    JazConsole(locals=namespace).interact(banner=banner, exitmsg="Goodbye!")
    return 0


def _positive_float(value: str) -> float:
    """Parse ``value`` as a strictly-positive float, for argparse ``type=``.

    Raises ``argparse.ArgumentTypeError`` for zero, negatives, or non-numbers.
    """
    # Reject <= 0 at parse time: a budget of 0 would abort before the first LLM call
    # (BudgetPool tests `>= budget`, and `0 >= 0`), which reads as "unlimited" to some
    # users but means "abort immediately" — a footgun, not a valid ceiling.
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {value!r}")
    return parsed


def _positive_int(value: str) -> int:
    """Parse ``value`` as a strictly-positive int, for argparse ``type=``.

    Raises ``argparse.ArgumentTypeError`` for zero, negatives, or non-integers.
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jaz", description="Interactive jaz console.")
    parser.add_argument(
        "--model",
        help=(
            "Model id as a LiteLLM route, provider-prefixed (e.g. openai/gpt-5-mini, "
            "anthropic/claude-sonnet-5). Sets llm.model on the default litellm backend."
        ),
    )
    parser.add_argument(
        "--max-cost",
        type=_positive_float,
        help="LLM cost budget in USD for the session (jaz.hooks.BudgetPool).",
    )
    parser.add_argument(
        "--max-calls",
        type=_positive_int,
        help="LLM call-count budget for the session (jaz.hooks.BudgetPool). Needs no "
        "pricing data, so use it to bound a model the price table can't price.",
    )
    parser.add_argument(
        "--max-recursion",
        type=int,
        help="Max nested jaz.invoke recursion depth (jaz.hooks.RecursionLimit).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable Jaeger tracing (requires the `tracing` extra + a Jaeger collector).",
    )
    parser.add_argument("--log", metavar="FILE", help="Write a FileLogger log to FILE.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live progress display (spinner + code reveal) during invokes.",
    )
    parser.add_argument(
        "-c",
        dest="command",
        metavar="PROMPT",
        help="Run a single prompt and exit (one-shot mode).",
    )
    parser.add_argument(
        "file", nargs="?", help="Run a script file through the console, then exit."
    )
    return parser


_MIN_PYTHON = (3, 14)


def _python_version_error() -> str | None:
    """Return an error message if the interpreter is too old for the console, else ``None``."""
    # Every *sugar* line — the `> prompt`/`<-`/`?` forms, plus `-c`, which is always a
    # prompt — is rewritten into `invoke(task=t"...")` *source* and compiled at runtime (see
    # the module docstring). (Bare lines in a script file or piped stdin are passed through
    # *unchanged* as Python and never lower to a t-string; but a console whose defining sugar
    # can't compile isn't the product, so the whole console is gated regardless.) t-strings
    # (PEP 750) are a Python 3.14 language feature, so on 3.13 and below that generated source
    # raises SyntaxError deep in the REPL, pointing at code the user never typed. Gating up
    # front turns that opaque failure into one clear message.
    # requires-python stays >=3.12 on purpose: the jaz *library* still supports 3.12; only
    # this interactive console needs 3.14, so the floor lives here rather than in the metadata.
    if sys.version_info >= _MIN_PYTHON:
        return None
    need = ".".join(str(v) for v in _MIN_PYTHON)
    have = ".".join(str(v) for v in sys.version_info[:3])
    return (
        f"The interactive jaz console requires Python {need}+, but is running under "
        f"Python {have}.\n"
        f"Its conversational shorthand rewrites input into t-strings (PEP 750), a Python "
        f"{need} language feature, so on older interpreters every line fails with a cryptic "
        f"SyntaxError.\n"
        f"Relaunch under Python {need}+, e.g. `python{need} -m jaz`."
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``jaz`` console script and ``python -m jaz``.

    Returns an integer exit status (``console_scripts`` calls this with no arguments and
    uses the return value as the process exit code).
    """
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    # Gate *after* parse_args: argparse handles `--help`/`--version`/usage errors during
    # parse_args and exits before returning here, so `--help` — exactly what someone reaches
    # for when a command won't start — keeps working on 3.13. Nothing in the parser needs
    # 3.14, and the guard still sits ahead of every path that actually runs input below.
    version_error = _python_version_error()
    if version_error is not None:
        print(version_error, file=sys.stderr)
        return 1

    # Ambient baseline before flags: settings.json first, then `--model` etc. override it. A
    # malformed file fails the console fast (mirrors the version guard) rather than starting on a
    # config the user did not write. Library `import jaz` never reaches here — the file is
    # console-scoped, so a broken ~/.jaz cannot break an `import jaz`.
    settings_error = _apply_user_settings()
    if settings_error is not None:
        print(settings_error, file=sys.stderr)
        return 1

    _apply_config(args)
    namespace = build_namespace()

    # Enter opt-in hooks (none by default) for the whole session/one-shot run.
    with ExitStack() as stack:
        for hook in _build_hooks(args):
            stack.enter_context(hook)  # type: ignore[arg-type]
        # Live progress display — default-on UI, entered for every path (interactive, -c,
        # file, piped): the tty gate inside _build_progress is what actually decides.
        if (progress := _build_progress(args)) is not None:
            stack.enter_context(progress)

        if args.command is not None:
            return _run_oneshot(args.command, namespace)
        if args.file is not None:
            return _run_file(args.file, namespace)
        # Piped/redirected stdin (non-TTY) with no -c: run it as a script, mirroring
        # `jaz <file>` — so `cat x.jaz | jaz` == `jaz x.jaz` (per-line sigils + EOF flush)
        # rather than collapsing to a single prompt. An ad-hoc one-liner prompt is `-c`.
        if not sys.stdin.isatty():
            return _run_script_source(sys.stdin.read(), namespace)
        return _interactive(namespace)


if __name__ == "__main__":
    raise SystemExit(main())
