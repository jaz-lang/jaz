import ast
import math
import numbers
import re
import sys
import threading
import traceback
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from enum import StrEnum
from io import StringIO
from types import CodeType, TracebackType
from typing import Any, override

from typing_extensions import TypedDict

from jaz._invoke_tool import InvokeTool
from jaz.exceptions import (
    FatalError,
    MissingDropTargetError,
    REPLInputConflictError,
    REPLTimeoutPragmaError,
    _JazInternalError,
    is_fatal,
)
from jaz.inputs import resolve_inputs
from jaz.string_utils import summarize_exception
from jaz.template_loader import _jinja_env

from ._ast_utils import parse_allowing_toplevel_return
from ._cookies import CookieJar
from ._exec_guards import (
    REPLMemoryError,
    REPLTimeoutError,
    guarded_aexec,
    guarded_exec,
    is_our_memory_error,
    is_our_timeout,
    new_owner,
)
from ._glob_allowlist import allows_everything
from .base import BaseREPL, REPLState
from .compiler import secure_compile
from .permissions import REPLPermissionPolicy, build_allowed_builtins
from .registry import register_repl
from .types import Continue, ExecResult, Raise, Return

# Default Jinja template for REPL help in system prompt (jaz/templates/).
_DEFAULT_REPL_DESCRIPTION_TEMPLATE = "python_repl_description.jinja2"


class TracebackVerbosity(StrEnum):
    """How much of an error traceback the agent is shown.

    The REPL's own engine frames — the ``.py`` plumbing that ``exec``'s the
    agent's code, sitting *above* the agent's ``<repl_...>`` frame in the stack —
    are ALWAYS stripped: they are an artifact of the REPL implementation, never
    part of the agent's error. These levels say how much of the remaining,
    agent-relevant traceback to keep (most → least):
    """

    ALL = "all"
    """All agent-relevant frames: the agent's own ``<repl_...>`` frame(s) plus
    any library frames its code called into (e.g. a tool/proxy that raised).
    This is the historical default."""

    REPL_ONLY = "repl_only"
    """Only the agent's own ``<repl_...>`` frames — every one in the chain, even
    a frame reached via a library callback (e.g. the agent passes ``f`` to an
    external ``g`` and ``g`` calls ``f``). All library frames are dropped. Hides
    library internals — and their file paths, which can leak the benchmark/
    harness identity — while keeping the agent's failing line(s) and the
    exception message."""

    MESSAGE_ONLY = "message_only"
    """No frames at all — just ``ExceptionType: message``."""


def _is_foreign_timeout(e: BaseException, owner_id: object) -> bool:
    """Whether ``e`` is an *outer* exec's expired deadline (not our own).

    A foreign timeout fires inside a nested invoke when the enclosing exec's
    deadline expires. It must not be converted to a ``Continue`` for this
    (inner) agent -- the inner agent cannot recover from a deadline it does
    not own, and would burn LLM calls reacting to it. Instead the catch sites
    surface it as a ``Raise``, the graceful "terminate this invoke with
    an exception" channel: the agent loop completes the invoke span and
    re-raises, propagating the timeout out of ``jaz.invoke`` to the parent
    exec, whose catch site recognizes it as its own.
    """
    return isinstance(e, REPLTimeoutError) and not is_our_timeout(e, owner_id)


def _is_foreign_memory_error(e: BaseException, owner_id: object) -> bool:
    """A ``REPLMemoryError`` for an *enclosing* exec's memory cap (not this one).

    Same rationale as ``_is_foreign_timeout``: an inner agent can't recover from a
    memory ceiling it doesn't own, so a foreign cap breach surfaces as a ``Raise``
    that propagates out to the owning parent exec rather than becoming recoverable
    feedback here. This exec's *own* cap breach is not foreign and is rendered as a
    ``Continue`` (the agent sees the limit message and can adjust).
    """
    return isinstance(e, REPLMemoryError) and not is_our_memory_error(e, owner_id)


def _reraise_if_fatal(e: BaseException, timeout_owner: object) -> None:
    """Re-raise an exception that must escape the invoke; return for recoverable ones.

    Two exceptions don't become recoverable agent feedback at a REPL exec boundary:
    a *foreign timeout / memory cap* (an enclosing exec's deadline or ceiling, see
    ``_is_foreign_timeout`` / ``_is_foreign_memory_error``) and any *fatal* exception
    (``is_fatal`` — ``FatalError`` / internal errors / ``KeyboardInterrupt`` …). Both
    **re-raise** the original exception out of ``repl.exec``: the REPL-exec span closes
    ``Failed`` with it and the exception keeps propagating out of the invoke, so a fatal
    error stays fatal through every enclosing invoke, and a foreign timeout reaches the
    exec that owns the deadline — which recognizes it as its own (``is_our_timeout``) and
    renders it there as recoverable feedback. Anything else returns and the caller renders
    it as a recoverable ``Continue``.
    """
    # This used to RETURN a `Raise(exception=e)` result instead of re-raising — the last
    # laundering relay of the pre-propagation abort model: every REPL boundary re-wrapped
    # a fatal into a terminal *result*, so a child's FatalError re-entered the parent's
    # transform pipeline (record as Completed-with-Raise, transformers get a shot) instead
    # of unwinding past it. Re-raising is what makes FatalError containment structural
    # (span_event_lifecycle.md stage 3); the is_fatal classification itself — including
    # the non-negotiable abort-in-group containment rule — transfers unchanged (see
    # ``jaz.exceptions.is_fatal``).
    if (
        _is_foreign_timeout(e, timeout_owner)
        or _is_foreign_memory_error(e, timeout_owner)
        or is_fatal(e)
    ):
        raise e


class PythonREPLConfig(TypedDict, total=False):
    """Configuration dictionary for PythonREPL."""

    allow_raise: bool
    repl_description_template: str
    # Present for the same reason as ``allow_raise``: ``get_description`` renders the effective
    # bound into the prompt (alongside the ``# timeout:`` pragma that overrides it), so a
    # description built from a config bag must be able to state the same number the REPL enforces.
    exec_timeout: float | None
    allow_timeout_pragma: bool
    # Fail-closed file-access allowlists (gitignore globs, #804 follow-up). Anchors //=fs, ~/=home,
    # /=cwd, ./=cwd, bare=cwd; a slash anchors, a slash-less name recurses. Absent/[] denies all;
    # ["**"] = cwd subtree, ["//**"] = whole filesystem; reads/writes gated separately. No None.
    allowed_read_paths: list[str]
    allowed_write_paths: list[str]
    # Compiler sandbox (#688). Enforced at secure_compile on every exec. The import and file
    # allow-lists are also stated to the agent (get_description renders them into the system
    # prompt); the attribute allow-list is enforced silently — see the sandbox-policy comment in
    # python_repl_description.jinja2. The
    # policy is NOT un-strippable by an agent: `repl_configs` is an ordinary config key, so a host
    # that passes `jaz.ConfigOverride` in as an input (or otherwise hands the agent a
    # config-override surface) opts into letting the agent reconfigure this sandbox for its
    # sub-invokes. Keep the sandbox off the agent's reach if that matters.
    # Uniform gitignore-glob allow-lists (imports/attributes/files), no "unrestricted"/None state:
    # key absent → the secure default; []: deny all; [...]: allow-list; ["*"]: allow all.
    allowed_imports: list[str]  # default [] (deny all); matched on the root module name
    # default DEFAULT_ALLOWED_ATTRIBUTES: allow all attrs except double-underscore-PREFIXED names
    # and the frame-bearing interpreter surface (f_back/gi_frame/...); [] denies all.
    allowed_attributes: list[str]
    # for -R, suppress recursion (install RecursionLimit(max_depth=1) so the agent has no
    # recursive jaz.invoke)
    reject_finish_on_printed_output: (
        # True (default): a finishing `return`/`raise` is rejected if that same code also
        # printed output (so the agent reviews it before finishing); False: finish anyway.
        bool
    )
    traceback_verbosity: (
        TracebackVerbosity | str
    )  # how much traceback the agent sees: "all" (default) > "repl_only" > "message_only"


# Secure-by-default sandbox: an unconfigured REPL is fail-closed on every axis, all now uniform
# gitignore-glob allow-lists (imports/attributes/files) with NO "unrestricted"/None escape — an
# empty allow-list denies everything, "allow all" is the explicit ``["*"]`` (see _glob_allowlist).
#
# Imports default to ``[]`` (deny ALL). Rationale: a security floor should fail closed; code that
# silently runs *with* imports it wasn't meant to have is a worse failure than being loudly denied.
# A host grants imports with an allow-list (``["json", "re", ...]``, matched on the root module
# name) or ``["*"]`` for all. EXECUTIVE CALL — this *reverses* the earlier "``allowed_imports``
# stays opt-in" stance; a curated safe-default allow-list (json/re/math/…) was rejected because it
# still fails *open* for anything unenumerated. The accepted cost is that an unconfigured agent can
# import nothing until a host opts in.
#
# Attributes default to ``["*", "!__*"]`` refined by the two lists below — allow every attribute
# EXCEPT double-underscore-prefixed ones, minus the frame-bearing interpreter surface, plus three
# inert dunders put back. NOTE ``__*`` is *prefix*, so it denies more than just true ``__x__``
# dunders — it also catches name-mangled ``__private`` attributes (leading ``__``, no trailing
# ``__``). That is strictly safer (denies a superset), just broader than the word "dunder" implies.
# The dunder attribute surface (``__globals__``, ``__closure__``, ``__subclasses__``,
# ``__mro__``, ``__dict__``, …) is the standard sandbox-escape vector; denying the whole class with
# one gitignore-negation rule beats enumerating it (a curated denylist always misses one). This is
# the allow-only inverse of the former ``forbidden_attributes=["__*"]`` denylist. A host widens the
# allow-list (e.g. ``["*"]`` to allow everything incl. dunders) or replaces it outright.


# Interpreter-owned attributes that hand back a *frame*, and through it the real module globals and
# the real builtins table. They are the hole in the ``!__*`` half of the default: the dunder deny is
# what stops ``x.__class__`` / ``__globals__`` / ``__subclasses__``, but every name here is
# ordinary-looking and non-dunder, so a bare ``["*", "!__*"]`` allows all of them.
#
# That is not theoretical — on the bare default this code escapes, using no dunder attribute and
# no import:
#
#     def gen():
#         yield g.gi_frame.f_back          # the frame that called next(g)
#     g = gen()
#     fr = next(g)
#     while "eval" not in fr.f_builtins:   # walk out of the REPL into a real jaz module frame
#         fr = fr.f_back
#     fr.f_builtins["eval"]("().__class__.__mro__[-1].__subclasses__()")
#
# The ``eval`` recovered that way is the *real* one, so it runs a string the compiler's
# ``AllowedAttributesChecker`` never sees — measured reaching ``(1).__class__`` and the classic
# ``__subclasses__()`` walk. (``open``/``__import__`` in that table are still jaz's secured
# wrappers, so the file and import allow-lists hold; what is bypassed is specifically the attribute
# axis.)
#
# EXECUTIVE CALL — this closes TODO(#827) in favour of hardening the SHIPPED DEFAULT, and reverses
# this PR's own first cut, which denied these names only in the console settings helper's sandbox
# pin (`console._HELPER_SANDBOX_OVERRIDE`) and left the default alone as "much larger blast radius,
# and not this pin's call to make". That scoping was wrong: the helper is the *narrow* sandbox, and
# the general REPL — which runs arbitrary model-generated code — is the one that actually needs the
# hole closed. Fixing it per-caller means every future sandbox re-derives the same list, and the
# one that forgets is silently escapable. The accepted cost is repo-wide: an agent that legitimately
# walks a frame or traceback *object* (``tb_frame``, ``f_back``) is now denied by default and needs
# a host to widen the list.
#
# Scoped by REACHABILITY, and spelled out one name at a time — both narrow the deny set against the
# obvious ``!f_*``/``!gi_*``/``!cr_*``/``!ag_*``/``!tb_*`` globs, and both are the user's call. The
# invariant is specifically that **no non-dunder name may hand back a frame**, directly or in one
# more hop (a generator/coroutine/traceback carries one; a code object and ``f_globals``/
# ``f_builtins`` are the payload the walk exists to reach). Read it that narrowly: "hands back
# something walkable" over-generalizes — ``property.fget`` hands back a function and is not denied,
# because a function's route onward is ``__globals__``, which ``!__*`` already closes.
# The inert names in ``_INERT_FRAME_ATTRIBUTES`` below are deliberately left
# ALLOWED, because reading ``tb_lineno`` off a traceback is how an agent reports where its own code
# failed and denying it buys no containment. And an explicit name denies exactly the interpreter
# surface, where a prefix glob would also catch any *ordinary* attribute starting with those two
# letters (a component setting, a caller's data) — blast radius that grows with code nobody has
# written yet. Denying the whole prefix surface uniformly was the first cut, free in the settings
# helper's sandbox (its namespace holds three strings, so no legitimate turn reads any of these) and
# not free as a shipped default.
#
# The cost of enumerating is that the list does not self-extend: a future Python that adds a frame
# attribute would be in neither list, which is why ``test_default_classifies_every_live_frame_
# attribute`` re-derives the surface from the running interpreter and fails on anything unclassified
# rather than letting the gap open silently. ``f_generator`` is 3.13+; listing it is harmless on
# 3.12, where it simply never matches.
_DENIED_FRAME_ATTRIBUTES: tuple[str, ...] = (
    # frame objects → frame / dict / code / function
    "!f_back",
    "!f_builtins",
    "!f_code",
    "!f_generator",
    "!f_globals",
    "!f_locals",
    # `f_trace` is a function; the two `f_trace_*` flags are bools, but they exist only to configure
    # `f_trace` and this allow-list governs `setattr` too, so they stay denied as write levers.
    "!f_trace",
    "!f_trace_lines",
    "!f_trace_opcodes",
    # generators → code / frame / iterator
    "!gi_code",
    "!gi_frame",
    "!gi_yieldfrom",
    # coroutines → awaitable / code / frame
    "!cr_await",
    "!cr_code",
    "!cr_frame",
    # async generators → awaitable / code / frame
    "!ag_await",
    "!ag_code",
    "!ag_frame",
    # tracebacks → frame / traceback
    "!tb_frame",
    "!tb_next",
)


# Audited counterpart to ``_DENIED_FRAME_ATTRIBUTES``: interpreter attributes on the same objects
# that hand back only ints, bools, or strings — nothing to walk — and are therefore left allowed by
# ``*``. NOT part of the policy: every name here is already permitted, and listing it changes no
# behavior. It exists so the audit is recorded and so the drift test can partition the live
# ``f_``/``gi_``/``cr_``/``ag_``/``tb_`` surface into "denied" and "deliberately allowed" and fail on
# anything in neither, which is what turns a new Python version's frame attribute into a red build
# rather than a silent reopening.
#
# ``cr_origin`` is the one that is not a bare scalar: a tuple of ``(filename, lineno, funcname)``
# triples (``None`` unless ``sys.set_coroutine_origin_tracking_depth`` is on). It stays allowed
# because inertness is what this list is scoped by and strings are inert. An earlier draft denied it
# as source-path disclosure — wrong reasoning, since ``allowed_read_paths``/``allowed_write_paths``
# gate *opening* files and never claimed to hide path names, so there was no property to protect.
#
# WRITE AXIS audited too, because this allow-list governs ``setattr`` as well as reads (the same
# asymmetry that keeps ``__class__`` out of ``_REALLOWED_DUNDERS``). Every name here is read-only in
# CPython, so permitting it grants no write capability — ``test_inert_frame_attributes_are_readonly``
# re-checks that against the running interpreter. The one with a caveat is ``f_lineno``: it *is*
# settable, but only from inside a trace function (the debugger "jump" feature; otherwise
# ``ValueError``), and installing one needs ``f_trace`` (denied here) or ``sys.settrace`` (needs an
# import the default denies). Left allowed because denying it would cost the read case this whole
# scoping exists to preserve, and a host that allows ``sys`` has opened far more than this.
_INERT_FRAME_ATTRIBUTES: tuple[str, ...] = (
    "f_lasti",
    "f_lineno",
    "gi_running",
    "gi_suspended",
    "cr_origin",
    "cr_running",
    "cr_suspended",
    "ag_running",
    "ag_suspended",
    "tb_lasti",
    "tb_lineno",
)


# Dunders re-admitted after ``!__*``, because blanket-denying every ``__*`` name costs ordinary
# Python without buying containment. ``super().__init__(...)`` is an ``ast.Attribute`` like any
# other, so a blanket deny makes plain subclassing a SyntaxError; ``type(x).__name__`` is likewise
# the normal way to name a class. Both were rejected by the default before this list existed —
# a usability bug, not a security property.
#
# Verified free rather than assumed free: with these three allowed, every dunder that actually
# *reaches* something stays denied — ``__globals__``, ``__mro__``, ``__bases__``, ``__base__``,
# ``__subclasses__``, ``__dict__``, ``__code__``, ``__self__`` — statically and through a computed
# ``getattr(cls, "__mro__")``, so no classic escape chain reopens. What ``__init__`` hands back is a
# slot wrapper / bound method that is inert without ``__globals__``/``__code__``/``__self__``;
# ``__name__``/``__qualname__`` are strings whose value ``repr(type(x))`` already leaks.
#
# ``__class__`` is deliberately NOT here, and the reason is the read/write asymmetry rather than
# reachability. Reading it is free to the point of redundancy — ``type()`` is already a permitted
# builtin, so ``x.__class__`` is a second spelling of ``type(x)``. But this allow-list also governs
# ``setattr``, so admitting the name re-enables ``obj.__class__ = Fake`` type confusion on any
# object handed into the sandbox that lacks its own ``__setattr__`` guard. Nothing readable is
# gained and a write capability is lost, so it stays denied.
_REALLOWED_DUNDERS: tuple[str, ...] = (
    "__init__",
    "__name__",
    "__qualname__",
)


# Order matters: gitignore semantics are last-match-wins, so the exclusions must follow the ``*``
# that would otherwise re-admit them, and the re-admitted dunders must follow ``!__*`` in turn.
# Anything placed *before* the pattern that overrides it is inert.
#
# KNOWN RESIDUAL (inherits #811's Known Limitations): ``str.format`` reaches attributes at the C
# level via ``PyObject_GetAttr`` without ever calling the wrapped ``getattr``, so
# ``"{0.__globals__}".format(fn)`` bypasses the runtime backstop entirely (read-only, and
# un-closeable in CPython). That applies to the frame names above too — ``"{0.gi_frame}".format(g)``
# still stringifies a frame — so the denials close the escape-to-*execution* (``format`` cannot hand
# back a callable) without making frames unobservable. The other standing hole is not an attribute
# name at all: a module reachable through an ordinary attribute of a host-supplied input
# (``h.os.system(...)``) bypasses every axis, because a name allow-list cannot police what an
# allowed name returns. This default is defense-in-depth, NOT a tight attribute boundary —
# genuinely untrusted code needs OS/process isolation; don't over-trust this as a substitute.
DEFAULT_ALLOWED_ATTRIBUTES: list[str] = [
    "*",
    "!__*",
    *_DENIED_FRAME_ATTRIBUTES,
    *_REALLOWED_DUNDERS,
]


# The default's WRITE axis: the same list without the re-admissions, so `!__*` denies those three
# again for `setattr`/`delattr` and `obj.attr = ...`/`del obj.attr`.
#
# EXECUTIVE CALL — the read/write split exists because re-admitting `_REALLOWED_DUNDERS` on both
# axes would hand agent code a write lever it never needed. `type(host).__init__ = agent_fn` is
# accepted when the name is writable, and classes are process-global, so the patched constructor
# outlives the exec and is visible to later invokes under different policies; `delattr` of it and
# renaming via `__name__` are the same shape. Ordinary method patching (`type(host).greet = ...`)
# was already reachable under a bare `["*", "!__*"]`, so the new exposure is specifically the
# constructor, its deletion, and the class's reported name — small, but the re-admission was for
# `super().__init__(...)` and `type(x).__name__`, both of which are READS. Splitting buys the read
# fix with no write cost, which is what the widening was actually for.
#
# A host-supplied `allowed_attributes` governs BOTH axes: this split is a property of the default's
# re-admission, not a general policy, and a host that writes its own list has said what it wants —
# silently subtracting from it would be the surprise. So `["*"]` still allows dunder writes.
DEFAULT_ALLOWED_WRITE_ATTRIBUTES: list[str] = [
    "*",
    "!__*",
    *_DENIED_FRAME_ATTRIBUTES,
]


def _effective_allowed_attributes(config: PythonREPLConfig) -> list[str]:
    """Attribute allow-list to enforce: the secure default (``DEFAULT_ALLOWED_ATTRIBUTES``) unless
    the host set the key explicitly. Returns a fresh list on the default path so a caller mutating
    the result can't corrupt the shared module-level default for later invokes."""
    if "allowed_attributes" not in config:
        return list(DEFAULT_ALLOWED_ATTRIBUTES)
    return config["allowed_attributes"]


# The allow-list applied when ``repl_configs["python"]`` omits ``allowed_imports``. The production
# value is ``[]`` — deny ALL imports, fail-closed. Exposed as a module constant (rather than an
# inline literal) so the test suite can monkeypatch it to ``["*"]`` (allow all) for tests that
# aren't about the sandbox — incidental ``import`` scaffolding in feature tests then keeps working,
# mirroring how the eval harness sets its own allow-list. See tests/conftest.py; sandbox tests opt
# into this real default via a pytest marker.
_DEFAULT_ALLOWED_IMPORTS: list[str] = []

# The per-exec wall-clock bound, owned here rather than in the config layer.
#
# It previously lived in BOTH places and they disagreed: `REPLConfig.params` materialized 30.0
# while this constructor defaulted to `None` (= unbounded). Every REPL built through `configure`
# got 30 s; every REPL built directly — a test helper, an embedding caller — silently got no
# timeout at all. The config value was the one users saw documented, so it wins, and the
# duplicate is removed rather than pinned by a test: a default has one home, and it is the
# component that consumes it.
_DEFAULT_EXEC_TIMEOUT: float | None = 30.0

# The per-exec timeout pragma: `# timeout: <seconds>` among the comment lines that OPEN the code.
#
# Scanned over the leading run of comment/blank lines and no further — the first line of real code
# ends the search. That run needs no tokenizer to identify: if every preceding line is blank or a
# comment then no string literal is open, so a leading `#` can only be a comment. Which matters,
# because the code may not parse at all (a syntax error still gets the timeout its author asked
# for), and because a `# timeout:` further down — including inside a string literal the agent is
# authoring — stays ordinary content.
#
# The block, not just the first line. MEASURED (2026-08-10, tests/test_timeout_pragma_integration
# .py, 5 trials x 4 small models, exec_timeout=2s and a task needing 10s): under the first-line
# rule gpt-5-mini scored 5/5 but 5.4-nano 0/5, 5.4-mini 2/5 and 5.6-luna 0/5. The failures were
# NOT models ignoring the feature — they wrote the pragma on line 2 or 3, under the plan comment
# that `system_prompt.jinja2` asks for ("a brief plan ... in comments on the first few lines").
# The two instructions were in direct conflict and the prompt-side one won; two of those models
# then abandoned the pragma and chunk-slept across 8-9 turns. Matching what they already write is
# cheaper than teaching every model to order its own comments. Scanning the block took 5.4-nano to
# 3/5 and 5.4-mini to 5/5; the residual misses write no pragma until the timeout fires, which is
# an anticipation problem in the prompt rather than a placement one (full table in that module).
#
# First match wins if the block somehow carries two — no conflict detection, since a model that
# writes two different bounds has a bigger problem than which one we pick.
#
# Case-SENSITIVE, and the colon must be adjacent to the keyword. Both narrow what counts as a
# pragma at all, which is what keeps the loud rejection off ordinary prose: since a malformed
# value is rejected rather than ignored, every line the pattern matches is a line the agent can
# lose a turn on. That narrowness matters more now than under the first-line rule, because every
# leading comment is a candidate. The case that motivated it is the plan comment an agent writes
# on the turn right after it gets cut off — `# Timeout: the last run timed out` — which reads as
# prose. `# timeout: <words>` in lower case still raises; that residual is accepted.
_TIMEOUT_PRAGMA_REGEX = re.compile(r"#\s*timeout:(?P<value>.*)")


def _parse_timeout_pragma(src: str) -> float | None:
    """Return the timeout in seconds the code requests via ``# timeout: <seconds>``, else ``None``.

    The pragma may sit on any of the comment lines that open the code, so it can appear above or
    below the agent's plan comment; the first line of real code ends the search. A trailing ``#``
    comment after the number is allowed (``# timeout: 120  # long build``).

    Raises:
        REPLTimeoutPragmaError: A leading comment is a ``# timeout:`` line whose value is not a
            positive, finite number of seconds.
    """
    for line in src.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            return None  # first real code line: the leading comment block is over
        match = _TIMEOUT_PRAGMA_REGEX.fullmatch(stripped)
        if match is None:
            continue  # an ordinary comment (the plan the system prompt asks for)
        # Everything from a second `#` on is a trailing comment, so the agent can say why it wants
        # the time on the same line it asks for it.
        raw = match.group("value").split("#", 1)[0].strip()
        try:
            seconds = float(raw)
        except ValueError:
            seconds = math.nan
        # `math.isfinite` also rejects `inf`, which `float` accepts and `> 0` would wave through.
        # An unbounded exec is not expressible as a pragma: `None` is the *configured* way to say
        # "no timeout", and an agent granting itself one would be unstoppable by wall-clock.
        if not math.isfinite(seconds) or seconds <= 0:
            raise REPLTimeoutPragmaError(
                f"Invalid `# timeout:` value {raw!r} — expected a positive number without units, "
                "e.g. `# timeout: 120`."
            )
        return seconds
    return None


def _effective_allowed_imports(config: PythonREPLConfig) -> list[str]:
    """Import allow-list to enforce, fail-closed by default. Absent key → the module default
    (``[]`` in production: deny ALL imports); an explicit value passes through. There is no
    ``None`` opt-out — ``["*"]`` is the explicit "allow all". Returns a fresh list on the default
    path so a caller can't mutate the shared module constant."""
    if "allowed_imports" not in config:
        return list(_DEFAULT_ALLOWED_IMPORTS)
    return config["allowed_imports"]


# Default file-access allowlists — used when the ``repl_configs["python"]`` key is absent (not in
# the config dict at all; a present-but-set value passes through unchanged, see
# ``_effective_allowed_read_paths``). The production value is ``[]`` — deny ALL file reads/writes.
# Like ``_DEFAULT_ALLOWED_IMPORTS``, exposed as module constants so tests/conftest.py can
# monkeypatch them to ``["//**"]`` (allow the whole filesystem) for non-sandbox feature tests. There
# is deliberately no "unrestricted"/``None`` state — file access is a plain glob list (see
# secure_open.py). Sandbox tests opt into the real deny-all default via the ``real_sandbox_defaults``
# marker.
_DEFAULT_ALLOWED_READ_PATHS: list[str] = []
_DEFAULT_ALLOWED_WRITE_PATHS: list[str] = []

# WIDENING THESE REACHES ~/.jaz. The lists REPLACE the default rather than merging with it, and the
# patterns are gitwildmatch, whose `*`/`**` match dotted segments (unlike shell globs) — so a grant
# as ordinary as `["~/**"]` includes `~/.jaz/credentials.json` and `~/.jaz/settings.json` without
# ever naming them. Exclude them with a trailing negation if that is not what you meant:
#
#     allowed_read_paths=["~/**", "!~/.jaz/**"]
#
# EXECUTIVE CALL (user, 2026-08-15) — this is ADVICE, not enforcement. A deny of `~/.jaz` checked
# ahead of these lists, which no allow-list could grant, was implemented and then reverted (#1047).
# Two reasons, and NEITHER is that the deny was leaky — it would be reverted just the same if it were
# enforced below the `open()` wrapper and airtight, because the objection is to the behaviour rather
# than to its reach:
#
#  1. THE ALLOW-LIST IS AUTHORITATIVE. Its entire contract is that what it grants is granted. A
#     hidden denial silently subtracting from an explicit grant is the failure we refuse to ship:
#     someone who allows a path and finds it still denied cannot reason about the policy they wrote,
#     and has no way to discover the subtraction from the config surface. Being wrong in the
#     permissive direction is the author's to fix, and the negation above is how; being wrong in the
#     restrictive direction is ours, and it is undebuggable from where the user is standing.
#  2. IT MATCHES THE CLOSEST PRECEDENT. Claude Code protects nothing on the READ side of its own
#     `~/.claude` — its hardcoded protected-path list covers writes only, degrades to a prompt
#     rather than a veto, and is lifted entirely in `bypassPermissions`. So on Linux, where its
#     OAuth token sits in `~/.claude/.credentials.json`, an explicit broad grant does include the
#     credentials. That is the accepted behaviour of a mature agent CLI, not an oversight.
#
# If credentials should survive a broad grant, the answer is to keep them out of a readable file
# (OS keyring, #1076) — which is exactly what Claude Code relies on for its own — not to make the
# allow-list mean something other than what it says.


def _effective_allowed_read_paths(config: PythonREPLConfig) -> list[str]:
    """Read-path gitignore allowlist, fail-closed by default (absent key → deny all reads).

    Unlike ``_effective_allowed_imports`` there is no ``None`` opt-out: file access is a plain
    glob list where ``[]`` denies everything and ``["**"]``/``["//**"]`` grant it. Absent key →
    the module default (``[]`` in production); an explicit value passes through. Returns a fresh
    list on the default path so a caller can't mutate the shared module constant."""
    if "allowed_read_paths" not in config:
        return list(_DEFAULT_ALLOWED_READ_PATHS)
    return config["allowed_read_paths"]


def _effective_allowed_write_paths(config: PythonREPLConfig) -> list[str]:
    """Write-path gitignore allowlist, fail-closed by default (absent key → deny all writes).
    Same absent→module-default / explicit-passthrough rule as ``_effective_allowed_read_paths``.
    """
    if "allowed_write_paths" not in config:
        return list(_DEFAULT_ALLOWED_WRITE_PATHS)
    return config["allowed_write_paths"]


def _make_print(output: StringIO):
    """Create a print function that writes to the given StringIO.

    Writes to ``output`` are serialized by a lock: an agent can hand this closure to
    a raw ``ThreadPoolExecutor`` (lexical ``print`` resolves to it in every worker),
    so several threads may write the one ``StringIO`` concurrently — and ``StringIO``
    is not thread-safe, so unsynchronized writes can tear/interleave. The lock guards
    only the default (captured) sink; an explicit ``file=`` target is left alone.
    """
    lock = threading.Lock()

    def custom_print(
        *args: object,
        sep: str = " ",
        end: str = "\n",
        file: Any = None,
        flush: bool = False,
    ) -> None:
        if file is not None:
            print(*args, sep=sep, end=end, file=file, flush=flush)
            return
        with lock:
            print(*args, sep=sep, end=end, file=output, flush=flush)

    return custom_print


# ---------------------------------------------------------------------------
# Native ``return`` / ``raise`` machinery
# ---------------------------------------------------------------------------
# The REPL reinterprets top-level ``return`` / ``raise`` statements (those NOT inside a nested
# function or class body) as REPL exit points, so the agent finishes with ordinary Python syntax
# rather than old-style ``RETURN`` / ``RAISE`` commands. They work inside ``if`` / ``for`` /
# ``while`` / ``try`` blocks, just like in a real function body.
#
# ``return`` and ``raise`` are handled asymmetrically:
#   * A top-level ``return <value>`` is rewritten to ``raise _JazReturnSignal(<value>)`` — a
#     sentinel the exec catch site turns back into a ``Return``. A ``return`` always finishes.
#   * A top-level ``raise <exc>`` (when ``allow_raise``) is NOT wrapped. It is rewritten to
#     ``raise _jaz_tag_raise(<exc>)``, which records the *real* exception's identity and
#     re-raises it, so the agent's own ``try`` / ``except`` runs under normal Python semantics.
#     The exec boundary finishes with a ``Raise`` only if the *escaping* exception was tagged.
#     So a ``raise`` the agent catches itself, a bare ``raise`` (never tagged), and a ``raise``
#     inside a ``def`` the agent wrote (never rewritten) all stay ordinary exceptions rather than
#     finishing the turn — matching the reference engine (agentica-internal), and removing the
#     need to statically reason about ``try`` nesting.


class _JazReturnSignal(BaseException):
    """Sentinel raised when user code executes a top-level ``return``.

    Derives from ``BaseException`` (not ``Exception``) so an ``except Exception:`` in the
    agent's own code does NOT swallow the finishing signal.
    """

    def __init__(self, value: object) -> None:
        self.value = value


class _NativeReturnRaiseTransformer(ast.NodeTransformer):
    """Rewrite top-level ``return`` / ``raise`` so the REPL can finish on them.

    * ``return <value>`` -> ``raise _JazReturnSignal(<value>)`` (a sentinel; always finishes).
    * ``raise <exc>``    -> ``raise _jaz_tag_raise(<exc>)`` (if *allow_raise*): the real
      exception is tagged and re-raised, NOT wrapped. Whether it finishes the turn is decided at
      the exec boundary by whether it *escapes* — see the module comment above.

    Neither is rewritten inside a nested ``def`` / ``async def`` / ``class`` body — there it
    belongs to the user's own function/class, not the REPL turn. A bare ``raise`` (re-raise) is
    also never rewritten, so it is never a deliberate finish. Crucially, ``raise`` no longer
    needs any ``try``-nesting analysis: because the real exception is raised (not a sentinel),
    the agent's own ``try`` / ``except`` catches it under normal Python semantics, and only an
    *uncaught* tagged ``raise`` reaches the boundary as a finish.
    """

    def __init__(self, allow_raise: bool, jar: CookieJar) -> None:
        self.allow_raise = allow_raise
        # Cookie jar supplies the ``_JazReturnSignal`` / ``_jaz_tag_raise`` references as co_consts
        # placeholders (roles ``return_signal`` / ``tag_raise``) rather than nameable ``ast.Name``s,
        # so the agent cannot name, catch, or override them. Substituted post-compile (see jar).
        self.jar = jar
        self._depth = 0  # nesting inside def/async def/class — gates both return and raise rewriting

    # -- depth tracking for nested defs/classes --

    def _visit_nested(self, node: ast.AST) -> ast.AST:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_nested(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._visit_nested(node)

    # -- transformations at depth 0 --

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if self._depth > 0:
            return node
        # return <value> -> raise <return_signal cookie>(<value>)  (cookie -> _JazReturnSignal)
        value = node.value if node.value is not None else ast.Constant(value=None)
        new_node = ast.Raise(
            exc=ast.Call(
                func=self.jar.const_for("return_signal"),
                args=[value],
                keywords=[],
            ),
            cause=None,
        )
        return ast.copy_location(new_node, node)

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        # Left untouched (so it stays ordinary Python, never a deliberate finish):
        #   * a raise inside a def/async def/class body (`_depth > 0`) — the user's own code,
        #   * a bare `raise` (re-raise; `node.exc is None`) — mirrors the reference engine,
        #   * everything when `allow_raise` is off.
        if self._depth > 0 or not self.allow_raise or node.exc is None:
            return node
        # `raise <exc> [from <cause>]` -> `raise _jaz_tag_raise(<exc>) [from <cause>]`.
        # `_jaz_tag_raise` records the exception's identity and returns it, so the REAL
        # exception is raised (not a sentinel wrapper) — the agent's own try/except handles it
        # normally, and the exec boundary only finishes if it escapes. Keeping the `from` clause
        # on the real raise means Python sets `__cause__`/`__suppress_context__` itself.
        new_node = ast.Raise(
            exc=ast.Call(
                func=self.jar.const_for("tag_raise"),
                args=[node.exc],
                keywords=[],
            ),
            cause=node.cause,
        )
        return ast.copy_location(new_node, node)


def _reraise_first_abort() -> None:
    """Re-raise the ``FatalError`` leaf of the in-flight ``except*`` group, *bare*.

    Why not a plain ``raise``: inside an ``except*`` handler that re-raises the enclosing
    ``ExceptionGroup``, so the ``Raise`` the exec boundary produces would carry the *wrapper*.
    Fatality would still be detected — :func:`~jaz.exceptions.is_fatal` treats a group holding
    an ``FatalError`` as fatal — but the exception object is carried onward from there to hooks
    and out of ``jaz.invoke()``, where ``except FatalError`` must match. Against a group it does
    not (measured: ``isinstance(group, FatalError)`` is ``False``), which would defeat the whole
    reason ``FatalError`` was re-rooted under ``Exception``. So this is about preserving the
    exception's *type* across the public boundary, not about classification.

    Reads the group from ``sys.exc_info()`` rather than taking it as an argument so the
    injected handler needs no ``except* ... as <name>`` binding — one less name in the agent's
    scope, and it mirrors the argument-free shape of the ordinary ``except <cookie>: raise``
    guard. Recurses into nested groups, since ``except*`` matching is by leaf.

    Reached only through a cookie (see ._cookies), so the agent can neither name nor shadow it.
    """
    exc = sys.exc_info()[1]
    stack: list[BaseException] = [exc] if exc is not None else []
    while stack:
        current = stack.pop(0)
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
        elif isinstance(current, FatalError):
            raise current
    if exc is not None:  # pragma: no cover - the guard only matches when a leaf exists
        raise exc


class _ExceptGuardTransformer(ast.NodeTransformer):
    """Make the agent's ``try`` / ``except`` unable to swallow the native-``return`` finish, and
    make a bare ``except:`` safe rather than forbidden. Runs whole-tree (every ``ast.Try``, nested
    ``def``s included). Two edits per ``try`` that has ``except`` handlers:

    * **Defense A** — insert, as the *first* handlers, ``except <return_signal cookie>: raise`` and
      ``except <abort cookie>: raise``. The cookies substitute to ``_JazReturnSignal`` and
      :class:`~jaz.exceptions.FatalError`, so a top-level ``return`` — or an abort raised by a tool
      the agent called — that unwinds through this ``try`` is re-raised *before* any of the agent's
      own handlers run. This holds even against an ``except BaseException:`` the agent writes using a
      ``BaseException`` reference handed in as an *input* (the builtins allowlist withholds the name,
      but a caller can pass it): the injected handlers are matched first and re-raise, so both are
      un-swallowable regardless of what the agent can name. The cookie (not an ``ast.Name``) is what
      makes the reference itself un-shadowable and un-nameable — see ._cookies.

      ``FatalError`` needs Defense A specifically because it is ``Exception``-rooted: unlike
      ``_JazReturnSignal`` it is *not* excluded by Defense B's narrowing to ``Exception``, and its
      base class no longer protects it. See :class:`~jaz.exceptions.FatalError` for that call.

    * **Defense B** — rewrite a bare ``except:`` to ``except <exception cookie>:`` (cookie ->
      the real ``Exception``). This narrows the catch to ``Exception``-rooted errors, so a bare
      ``except:`` can no longer swallow the ``BaseException``-rooted finish sentinel (nor any other
      ``BaseException``-rooted signal a caller introduces). Using a cookie for ``Exception`` — rather
      than emitting the *name* ``Exception`` — makes it immune to the agent shadowing ``Exception``
      locally or mutating ``__builtins__['Exception']`` (a plain mutable dict) before the handler is
      reached. This *replaces* the old ``reject_bare_except`` ban (#869): bare ``except:`` is made
      safe instead of rejected.

    ``ast.TryStar`` (``try/except*``) gets an **abort guard only** (:meth:`visit_TryStar`). For
    ``_JazReturnSignal`` no guard is needed: ``except*`` cannot be bare (nothing for Defense B to
    narrow) and cannot catch a naked ``BaseException``-rooted exception, so the finish sentinel is
    immune by construction. That reasoning does *not* extend to ``FatalError``, which is
    ``Exception``-rooted — ``except*`` wraps a lone exception in an implicit ``ExceptionGroup`` and
    matches it, so an unguarded ``try: <aborting call> except* Exception:`` swallowed an abort while
    naming only the allowlisted ``Exception``.

    Two facts make the guard work, both non-obvious enough to be worth stating:

    * ``except*`` clauses **consume** the part of the group they match, and a later clause is offered
      only the remainder. Position 0 therefore takes the ``FatalError`` out before the agent's
      ``except* Exception`` is considered. (It is *not* that only the first clause runs — later
      clauses do run, on what is left; a group of ``[FatalError, ValueError]`` runs both.)
    * The guard body calls :func:`_reraise_first_abort` rather than a plain ``raise``, so what
      reaches the exec boundary is the abort *leaf*, not the ``ExceptionGroup`` wrapper. This is
      about the exception's **type surviving to the caller** — a wrapper is still classified fatal
      by ``is_fatal``, but ``except FatalError`` around ``jaz.invoke()`` would not match it.

    Residual, and irreducible in ``except*`` semantics: when the abort is bundled with unhandled
    exceptions (an ``asyncio.TaskGroup``, say), ``except*`` re-combines the guard's re-raise with the
    remainder, so a group can still reach the boundary. That case is covered on the classification
    side — :func:`~jaz.exceptions.is_fatal` treats a group containing an ``FatalError`` at any depth
    as fatal — and the two halves cover *different* failures rather than overlapping: the guard keeps
    the agent's clause from consuming the abort at all (nothing would reach the boundary to
    classify), and the group rule keeps a bundled abort fatal once it does.

    Two signals are guarded here (``_JazReturnSignal`` and ``FatalError``). The remaining
    must-not-swallow signals are handled differently by design: ``KeyboardInterrupt`` /
    ``SystemExit`` / ``GeneratorExit`` / ``CancelledError`` are absent from the default builtins, so
    the agent can only name — and thus only catch — one when a caller *explicitly* passes it in,
    which is read as intent to let the agent handle it; ``REPLTimeoutError`` is deferred (its
    ``Exception`` rooting makes it catchable by a plain ``except Exception:``, so it needs its own
    treatment, handled separately). ``FatalError`` is *not* in that group: it is reachable via the
    allowlisted ``Exception`` with no caller cooperation, which is why it is force-guarded rather
    than left to the builtins allowlist.
    """

    def __init__(self, jar: CookieJar) -> None:
        self.jar = jar

    def visit_Try(self, node: ast.Try) -> ast.AST:
        # Recurse first so nested try blocks (and try inside def/TryStar) are guarded too.
        self.generic_visit(node)
        # A try with only a `finally` (no handlers) cannot swallow anything — skip it.
        if not node.handlers:
            return node
        # Defense B: narrow any bare handler to the real Exception (via cookie).
        for handler in node.handlers:
            if handler.type is None:
                handler.type = self.jar.const_for("exception")
        # Defense A: prepend un-nameable guards that re-raise the must-not-swallow types.
        #
        # Two separate single-type handlers rather than one `except (<c1>, <c2>):` tuple,
        # deliberately: CPython folds a tuple of constants into a single tuple entry in
        # ``co_consts``, and ``CookieJar.substitute`` walks ``co_consts`` without descending
        # into tuple constants. The cookies would therefore survive unsubstituted *and*
        # undetected by the collision check, leaving `except (b'...', b'...')` to TypeError at
        # runtime. One Constant per handler keeps every cookie a top-level ``co_consts`` entry.
        for role in ("abort", "return_signal"):
            node.handlers.insert(
                0,
                ast.ExceptHandler(
                    type=self.jar.const_for(role), name=None, body=[ast.Raise()]
                ),
            )
        return node

    def visit_TryStar(self, node: ast.TryStar) -> ast.AST:
        # Recurse first so ordinary `try` blocks nested inside the `except*` are guarded too.
        self.generic_visit(node)
        if not node.handlers:
            return node
        # Only the abort needs guarding here: `except*` cannot be bare (nothing for Defense B
        # to narrow) and cannot catch a naked BaseException-rooted exception, so
        # `_JazReturnSignal` is still immune by construction.
        #
        # Position 0 matters because `except*` clauses CONSUME the part of the group they match:
        # a later clause is offered only the remainder. So this guard takes the FatalError out
        # before the agent's `except* Exception` is considered, and that clause never sees it.
        # (Not "only the first clause runs" — later clauses do run, on what is left.)
        #
        # The body calls the cookie-supplied `_reraise_first_abort` rather than a plain `raise`
        # so the exception that reaches the boundary is the abort itself, not the ExceptionGroup
        # wrapper — see that function for why the type, not the classification, is the point.
        node.handlers.insert(
            0,
            ast.ExceptHandler(
                type=self.jar.const_for("abort"),
                name=None,
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=self.jar.const_for("reraise_abort"),
                            args=[],
                            keywords=[],
                        )
                    )
                ],
            ),
        )
        return node


@register_repl("python")
class PythonREPL(BaseREPL):
    """REPL implementation that enforces compile-time safety policies.

    Uses native ``return`` / ``raise`` semantics: a top-level ``return`` / ``raise`` statement
    (not nested inside a function or class) finishes the REPL, producing a ``Return`` / ``Raise``
    result. See the native-machinery block above this class for the AST transform that implements
    this.

    Configuration flags:
        allow_raise: If True (default), a top-level ``raise`` exits the REPL with that exception.
            If False, only ``return`` is available for finishing (a top-level ``raise`` is left
            as ordinary Python and not treated as a finish).
    """

    # The REPL holds NO per-invoke state (#1058): ``initialize`` returns a ``REPLState`` and every
    # stateful method (``exec``/``add_variables``/``drop_variables``) takes it back as an argument.
    # A configured REPL is a pure, reusable template — two invokes cannot bleed into each other
    # because there is no ``self`` for either to write to.
    allow_raise: bool
    allowed_imports: list[str]
    allowed_attributes: list[str]

    # PythonREPL surfaces __history__ in its namespace (see base REPL).
    maintains_repl_history = True

    @property
    def description(self) -> str:
        """Return the appropriate description based on configuration.

        Passes what ``get_description`` reads: the template name and every *live core variable* it
        injects — ``allow_raise``, ``exec_timeout``, ``allow_timeout_pragma`` and the two
        described sandbox allow-lists (imports, file read/write). All must be forwarded from the
        instance here: without them the render falls back to the constructor defaults, so an
        ``allow_raise=False`` REPL would still describe ``raise``, a re-timed one would quote 30 s,
        one with the pragma disabled would still offer it, and one granted imports or file access
        would still tell the agent it has none — in each case disagreeing with both enforcement and
        ``get_description(repl_config)``.
        """
        # ``allowed_attributes`` is deliberately NOT forwarded: it is enforced (secure_compile +
        # the getattr backstops) but no longer described, so passing it would only mislead a reader
        # into thinking the description reflects it. See the template's sandbox-policy comment.
        return self.get_description(
            PythonREPLConfig(
                repl_description_template=self.repl_description_template,
                allow_raise=self.allow_raise,
                exec_timeout=self.exec_timeout,
                allow_timeout_pragma=self.allow_timeout_pragma,
                allowed_imports=self.allowed_imports,
                allowed_read_paths=self.allowed_read_paths,
                allowed_write_paths=self.allowed_write_paths,
            )
        )

    @classmethod
    def get_description(cls, config: PythonREPLConfig | None = None) -> str:
        """Return the appropriate description based on configuration.

        This class method allows getting the description without instantiating the REPL,
        which is useful for displaying in system prompts.

        Args:
            config: Configuration dict with optional keys:
                - repl_description_template: str (default python_repl_description.jinja2)

        Returns:
            The appropriate REPL description string.

        Prompt prose is config-driven, but core renders the template with only *core-owned* vars.
        Five qualify here — ``allow_raise``, ``exec_timeout``, ``allow_timeout_pragma`` and the
        sandbox allow-lists ``allowed_imports`` / ``allowed_read_paths`` / ``allowed_write_paths``
        (one var each, the last two being the two halves of file access) — because core owns and
        enforces all of them, and a description that disagreed with enforcement would be a bug
        (offering ``raise`` as a finish when the REPL forbids it; quoting a wall-clock bound the
        exec does not apply; offering an override the REPL ignores; promising imports or files the
        sandbox denies). So core injects the effective values into the render. This makes the core
        default template (``python_repl_description.jinja2``, gated on ``{% if allow_raise %}``)
        coherent for a direct, non-eval ``jaz.invoke`` caller who sets them via ``repl_configs``.
        The *hook-owned* feature flags (repl_history, exposed inputs, return restriction, …) are an
        eval-configuration concern and are **baked** into the evals description meta-template at
        config time (the harness's ``bake_template``) — never passed at render time. Core
        deliberately has no generic vars escape hatch: a template that references an unbaked toggle
        raises here rather than rendering an empty string, by design.
        """
        # The strictness comes from ``_jinja_env`` being configured with StrictUndefined.
        # The sandbox text used to be *appended* here by three prompt-building helpers, to whatever
        # template was configured; it now lives inside the template, which is what makes that
        # template the whole description rather than most of it. Two consequences, both accepted
        # (user's call) and both spelled out in full in the `{# Sandbox policy #}` comment in
        # python_repl_description.jinja2: a configured `repl_description_template` (the evals
        # variants) no longer inherits the policy text, and `allowed_attributes` is described
        # nowhere — so `_effective_allowed_attributes` feeds enforcement only.
        config = config or {}
        template_name = config.get(
            "repl_description_template", _DEFAULT_REPL_DESCRIPTION_TEMPLATE
        )
        template = _jinja_env.get_template(template_name)
        # The live core variables (see docstring): supply the effective values so the otherwise
        # flag-free core default reflects them for direct callers. Every other var a configured
        # (evals) template needs is baked in at config time, so nothing else here. Each falls back
        # to the constructor default, not to a second literal — `get_description` is a classmethod
        # that may be called with no config at all, and what it states must be what an unconfigured
        # REPL would enforce; for the allow-lists that is what the `_effective_*` helpers return.
        #
        # `allows_everything` rides along as a callable rather than being applied here and passed as
        # a bool, so the template asks the shared predicate (_glob_allowlist) instead of re-spelling
        # "is this trivially allow-all?" in Jinja. Note what that predicate is: a CONSERVATIVE check
        # for the two canonical spellings (`["*"]`/`["**"]`), not an enforcement-side notion of
        # allow-all — enforcement has none, it just runs the matcher. So any other allow-all
        # spelling (`["*", "*"]`, `["**/*"]`) is printed as a literal allow-list while enforcement
        # allows everything. Sound in the safe direction only: the agent under-uses a permission it
        # has, never the reverse.
        return template.render(
            allow_raise=config.get("allow_raise", True),
            exec_timeout=config.get("exec_timeout", _DEFAULT_EXEC_TIMEOUT),
            allow_timeout_pragma=config.get("allow_timeout_pragma", True),
            allowed_imports=_effective_allowed_imports(config),
            allowed_read_paths=_effective_allowed_read_paths(config),
            allowed_write_paths=_effective_allowed_write_paths(config),
            allows_everything=allows_everything,
        )

    @override
    def get_finish_command_hint(self) -> str:
        if self.allow_raise:
            return "`return ...` or `raise ...`"
        return "`return ...`"

    def _format_error(self, exception: BaseException, stdout: str = "") -> str:
        """Format an exception's full output (stdout + filtered traceback) for an
        ERROR result. The agent-facing summary is derived separately by each consumer
        calling ``summarize_exception`` on the stored exception (#566 step C, #590
        follow-up), so it isn't built here.

        How much of the traceback the agent sees is governed by
        ``self.traceback_verbosity`` (a :class:`TracebackVerbosity`); see that enum
        for the level definitions. REPL frames have filenames like
        "<repl_session_id_code_id>" (not ending in .py); library/engine frames
        end in .py.

        ``format_exception`` renders not just ``exception`` but its whole
        ``__cause__``/``__context__`` chain, and each linked exception carries
        its *own* ``__traceback__``. Filtering only the outermost traceback would
        let a chained exception's unfiltered library frames leak — e.g. a
        ``raise X from Y`` (or a bare ``raise`` inside an ``except``, which sets
        ``__context__``) whose linked stack traverses library ``.py`` frames.
        So we filter every exception ``format_exception`` will walk. Because
        ``_filter_traceback`` is documented to leave the original
        ``__traceback__`` intact (it's reused for logging/``summarize_exception``),
        we swap the filtered tracebacks in only transiently and restore the
        originals in a ``finally``.

        Arguments:
            exception: The exception to format.
            stdout: Any stdout output that occurred before the exception.

        Returns:
            formatted_output: Full output including stdout and the filtered traceback.
        """
        chain = self._exception_chain(exception)
        originals = [exc.__traceback__ for exc in chain]
        try:
            for exc in chain:
                exc.__traceback__ = self._filter_traceback(
                    exc.__traceback__, self.traceback_verbosity
                )
            tb_str = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
        finally:
            for exc, original_tb in zip(chain, originals, strict=True):
                exc.__traceback__ = original_tb

        # Combine stdout and traceback. No summary is appended: the rendered traceback already
        # carries strictly more than ``summarize_exception`` would add, in every shape — a
        # ``SyntaxError``'s file/line/caret beat the ``" (<file>, line N)"`` suffix, and an
        # exception group's ``+---`` tree names every child. The two differ only in *format*,
        # so appending would restate the same facts in a second, worse layout.
        return stdout + tb_str

    @staticmethod
    def _exception_chain(exception: BaseException) -> list[BaseException]:
        """The exceptions ``traceback.format_exception`` will render for
        ``exception``: itself plus each linked ``__cause__``/``__context__``.

        Mirrors CPython's ``TracebackException`` traversal — follow ``__cause__``
        when set, else ``__context__`` unless ``__suppress_context__`` (as with
        ``raise X from None``) — and guards against reference cycles by id so a
        self-referential chain can't loop forever.
        """
        chain: list[BaseException] = []
        seen: set[int] = set()
        current: BaseException | None = exception
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(current)
            if current.__cause__ is not None:
                current = current.__cause__
            elif current.__context__ is not None and not current.__suppress_context__:
                current = current.__context__
            else:
                current = None
        return chain

    @staticmethod
    def _filter_traceback(
        tb: TracebackType | None, verbosity: TracebackVerbosity
    ) -> TracebackType | None:
        """Filter ``tb`` for agent display per ``verbosity`` (see TracebackVerbosity).

        ALWAYS strips the leading REPL-engine (.py) frames above the agent's
        code — they are an implementation artifact, never part of the agent's
        error. Then keeps the agent-relevant frames per ``verbosity``. Never mutates
        the input chain (rebuilds with ``TracebackType`` when truncating) so the
        original exception's ``__traceback__`` is left intact for logging/etc.
        """
        if tb is None:
            return None

        if verbosity is TracebackVerbosity.MESSAGE_ONLY:
            return None

        if verbosity is TracebackVerbosity.ALL:
            # Strip the leading engine (.py) frames above the agent's code, then
            # keep the rest: the agent's REPL frame(s) plus any library frames it
            # called into.
            while tb is not None and tb.tb_frame.f_code.co_filename.endswith(".py"):
                tb = tb.tb_next
            return tb

        # REPL_ONLY: keep only the agent's own ``<repl_...>`` frames anywhere in
        # the chain — including code reached via a library callback (e.g. the
        # agent passes its f to an external g and g calls f) — and drop every
        # other frame. We match the ``<repl_`` prefix explicitly rather than
        # "not .py": a plain not-.py test also keeps synthetic engine frames like
        # ``<frozen importlib._bootstrap>``, ``<string>`` and ``<stdin>``, which
        # are engine noise the agent shouldn't see (and which the ALL-level
        # leading-strip, keyed on .py, wouldn't peel either). Frames that become
        # non-adjacent after dropping a frame are re-linked; the message is kept.
        kept: list[TracebackType] = []
        node = tb
        while node is not None:
            if node.tb_frame.f_code.co_filename.startswith("<repl_"):
                kept.append(node)
            node = node.tb_next
        rebuilt: TracebackType | None = None
        for n in reversed(kept):
            rebuilt = TracebackType(rebuilt, n.tb_frame, n.tb_lasti, n.tb_lineno)
        return rebuilt

    def __init__(
        self,
        *,
        exec_timeout: float | None = _DEFAULT_EXEC_TIMEOUT,
        exec_memory_limit: int | None = None,
        allow_timeout_pragma: bool = True,
        allow_raise: bool = True,
        repl_description_template: str = _DEFAULT_REPL_DESCRIPTION_TEMPLATE,
        reject_finish_on_printed_output: bool = True,
        allowed_imports: list[str] | None = None,
        allowed_attributes: list[str] | None = None,
        allowed_read_paths: list[str] | None = None,
        allowed_write_paths: list[str] | None = None,
        traceback_verbosity: TracebackVerbosity | str = TracebackVerbosity.ALL,
    ) -> None:
        """Configure a REPL. The result is **not yet usable** — call :meth:`~PythonREPL.initialize` first.

        Construction takes only *configuration*; the per-invoke state (inputs, the library
        binding, the session id, the history list) is supplied by :meth:`~PythonREPL.initialize`.

        Arguments:
            exec_timeout: Per-exec wall-clock bound in seconds (None = no timeout).
                Defaults to 30 s. Agent code can override it per execution unless
                ``allow_timeout_pragma`` is False.
            exec_memory_limit: Process-wide RSS ceiling in bytes, per exec (None = off).
            allow_timeout_pragma: If True (default), a leading ``# timeout: <seconds>`` comment
                in the code sets that execution's wall-clock bound, overriding ``exec_timeout``
                with no ceiling. Set False to make ``exec_timeout`` a bound the agent cannot
                raise; the comment is then ordinary text and is not described in the prompt.
            allow_raise: If True (default), a top-level ``raise`` exits the REPL with
                that exception. If False, only ``return`` is available for finishing.
            repl_description_template: Jinja2 template name for the REPL description.
            reject_finish_on_printed_output: If True (default), a finishing ``return``/``raise``
                is rejected (not executed) when the same code also printed output — the agent
                must review the output and re-run only the ``return``/``raise``. If False,
                finishing is allowed even with printed output.
            allowed_imports: Import allow-list. ``None`` applies the fail-closed default
                (deny all); ``["*"]`` is the explicit allow-all.
            allowed_attributes: Attribute allow-list. ``None`` applies the secure default.
            allowed_read_paths: gitignore-style globs for file reads. ``None`` applies
                the fail-closed default (deny all).
            allowed_write_paths: gitignore-style globs for file writes. ``None`` applies
                the fail-closed default (deny all).
            traceback_verbosity: How much of the error traceback the agent sees, a
                :class:`TracebackVerbosity` (or its str value). The REPL's engine
                frames above the agent's code are always stripped; this controls
                the agent-relevant remainder: "all" (default; agent frame +
                library frames it called into), "repl_only" (agent frames only —
                drops library frames and their identity-leaking paths), or
                "message_only" (just the exception message).

        Raises:
            ValueError: If ``traceback_verbosity`` is not a known value.
        """
        # EXECUTIVE CALL (user, 2026-08-07): construction takes configuration ONLY, and
        # `initialize` takes only the invoke-time arguments that build REPL *state*. Previously
        # the two were tangled — `__init__` took an already-built namespace (`repl_state_locals`)
        # alongside half the settings, while the other half (`allowed_read_paths`,
        # `allowed_write_paths`) reached the object only as a `config` dict `initialize` folded
        # into a permission policy and never stored.
        #
        # The payoff is that `declared_init_keys(PythonREPL)` is now the structural source of
        # truth for `repl.params`, exactly as it already is for an LLM backend — so
        # `REGISTRY[tag].from_dict(params)` is real for REPLs too, with no per-field
        # special-casing. That row of the design doc's audit table read "no" because
        # `REPL.initialize` mixed an opaque `configs[lang]` bag with named runtime kwargs; the
        # split is what closes it. A configured REPL is a *recipe* (serializable); an
        # initialized one holds per-invoke state, which is explicitly out of scope for
        # serialization ("to_dict is a recipe, not a snapshot").
        #
        # The sandbox defaults are now fail-closed on EVERY path. Before, direct construction
        # defaulted `allowed_imports`/`allowed_attributes` to the permissive `["*"]` while the
        # config path applied the deny-all `_effective_*` defaults — two regimes for one setting,
        # with the safe one reachable only through `initialize`. With one constructor there can
        # only be one default, and it must be the fail-closed one: a REPL built without saying
        # anything about its sandbox must not be the open one.
        # Validated here, not in the config layer. `Config` used to range-check these on the
        # REPL's behalf, which meant a class that owned the setting did not own its contract —
        # and a REPL built directly (a test helper, an embedding caller) got no check at all.
        # The constructor is both the config surface and the only path to an instance, so it is
        # the one place that can promise it.
        if exec_timeout is not None and (
            not isinstance(exec_timeout, numbers.Real) or exec_timeout <= 0
        ):
            raise ValueError("repl.exec_timeout must be positive or None")
        if exec_memory_limit is not None and (
            not isinstance(exec_memory_limit, numbers.Integral)
            or exec_memory_limit <= 0
        ):
            raise ValueError(
                "repl.exec_memory_limit must be a positive integer (bytes) or None"
            )
        # Annotated explicitly: the `numbers.Real` / `numbers.Integral` checks above narrow the
        # parameters, and without these the *attribute* type would be inferred from the narrowed
        # value — leaking `<subclass of float and Real> | <subclass of int and Real> | None` to
        # every reader of `repl.exec_timeout`.
        self.exec_timeout: float | None = exec_timeout
        self.exec_memory_limit: int | None = exec_memory_limit
        # Defaults ON, so the pragma is available without configuration and `exec_timeout` alone
        # is NOT a bound. The knob exists because that is a silent change for an embedder who set
        # `exec_timeout` as a resource guard: without it, upgrading removes a limit they were
        # relying on and the only remedy is to wrap the REPL. Off is the pre-pragma behaviour
        # exactly — the comment is inert AND unadvertised, since a description that offered an
        # override the REPL ignores would be worse than not mentioning it.
        self.allow_timeout_pragma = allow_timeout_pragma
        self.allow_raise = allow_raise
        self.repl_description_template = repl_description_template
        self.reject_finish_on_printed_output = reject_finish_on_printed_output
        # Compiler sandbox (#688): uniform allow-lists, fail-closed when unspecified. `None` means
        # "not configured" and takes the module-level default — deny all imports, the secure
        # attribute set, deny all file access. There is no `None` opt-out: `["*"]` is the explicit
        # allow-all. The module constants are indirected (rather than inlined) so the test suite
        # can relax them for tests that are not about the sandbox; a fresh list is taken so a
        # caller mutating the result cannot corrupt the shared default for later invokes.
        self.allowed_imports = (
            list(_DEFAULT_ALLOWED_IMPORTS)
            if allowed_imports is None
            else allowed_imports
        )
        self.allowed_attributes = (
            list(DEFAULT_ALLOWED_ATTRIBUTES)
            if allowed_attributes is None
            else allowed_attributes
        )
        # Derived, not a constructor parameter: the read/write split exists to undo the *default's*
        # dunder re-admission on the write axis, so a host that states its own `allowed_attributes`
        # gets that one list on both axes rather than a silent subtraction it never asked for.
        self.allowed_write_attributes = (
            list(DEFAULT_ALLOWED_WRITE_ATTRIBUTES)
            if allowed_attributes is None
            else self.allowed_attributes
        )
        self.allowed_read_paths = (
            list(_DEFAULT_ALLOWED_READ_PATHS)
            if allowed_read_paths is None
            else allowed_read_paths
        )
        self.allowed_write_paths = (
            list(_DEFAULT_ALLOWED_WRITE_PATHS)
            if allowed_write_paths is None
            else allowed_write_paths
        )
        try:
            self.traceback_verbosity = TracebackVerbosity(traceback_verbosity)
        except ValueError:
            raise ValueError(
                f"Invalid traceback_verbosity={traceback_verbosity!r}; expected one of "
                f"{[d.value for d in TracebackVerbosity]}."
            ) from None

        # Native ``return`` / ``raise`` compile down to references to ``_JazReturnSignal`` and a
        # per-exec ``tag_raise`` closure. These are injected into the transformed code as compile-time
        # *cookies* (co_consts substitution — see ._cookies and _parse_and_transform), NOT bound in
        # ``__builtins__``: a nameable binding (the old ``builtins["_JazReturnSignal"] = …``) let
        # agent code ``except _JazReturnSignal:`` swallow its own finish, and read/forge the
        # sentinel. Cookies keep them reachable by the compiled bytecode but unnameable by the agent
        # — closing that exposure (this replaced a standing TODO asking exactly that).

    @override
    def initialize(
        self,
        inputs: dict[str, object],
        invoke_tool: InvokeTool | None,
        allowed_builtins: dict[str, object] | None = None,
        session_id: str = "",
        initial_repl_history: list[object] | None = None,
    ) -> REPLState:
        """Build one invoke's :class:`REPLState`; the REPL is left as a reusable config template.

        Takes only the *invoke-time* arguments — everything here belongs to one run rather than
        to the REPL's configuration, which was fixed at construction. Builds the namespace
        (``__builtins__`` under this REPL's sandbox policy, ``__inputs__``, ``__history__``,
        and a binding per input) on a copy, leaving the receiver a clean, reusable template.

        Arguments:
            inputs: The invoke's inputs, already payload-resolved (see ``jaz.inputs``).
            invoke_tool: The recursive sub-invoke primitive to bind as ``invoke``, or None.
            allowed_builtins: Extra builtins to seed before the sandbox policy is applied.
            session_id: Unique session identifier for linecache filenames.
            initial_repl_history: The list object the driver owns and surfaces as
                ``__history__``, or None to start a fresh empty one.
        """
        # This used to bind onto `self` and return it, which made the object single-use: calling
        # it twice silently replaced the first run's namespace, library binding and session id.
        # Tolerable while every caller constructed a fresh REPL per invoke — but it means a
        # *configured* REPL cannot be held and reused, and holding one is exactly what a config
        # that stores components has to do.
        permission_policy = REPLPermissionPolicy(
            allowed_read_paths=self.allowed_read_paths,
            allowed_write_paths=self.allowed_write_paths,
            allowed_imports=self.allowed_imports,
            # Drives the runtime getattr/hasattr/setattr/delattr/vars backstops, which apply the
            # same allow-lists the static AllowedAttributesChecker enforces on literal access —
            # including the read/write split, or `setattr(type(host), "__init__", ...)` would walk
            # straight past a static checker that only sees `obj.attr = ...`.
            allowed_attributes=self.allowed_attributes,
            allowed_write_attributes=self.allowed_write_attributes,
        )
        allowed_builtins = build_allowed_builtins(
            base_builtins=allowed_builtins,
            policy=permission_policy,
        )

        # Import enforcement is two layers (#470/#688): the static ImportPolicyChecker in
        # secure_compile catches `import`/`from` statements AND reserves the `__builtins__` name
        # (rejecting any `ast.Name` reference — e.g. `__builtins__["__import__"]`) at compile time;
        # the policy-bound `__import__` wrapper installed by build_allowed_builtins above is the
        # runtime backstop for every `__import__(...)` call — literal or computed
        # (`__import__("o" + "s")`) — which the static pass does not inspect (#742), the machinery
        # every `import` statement's IMPORT_NAME still calls, and the true enforcement point for any
        # route that reaches the sandbox's `__builtins__` (frame introspection, str.format — which
        # the name reservation cannot close; see ImportPolicyChecker). Reserving the `__builtins__`
        # name statically closes the deferred-execution hole a runtime-only guard left open for the
        # `__builtins__["__import__"]` route: a returned closure over it escapes the sandbox
        # unexecuted. (A follow-up reserves the `__import__` name too, for the bare form.)
        # Both are driven by repl_configs["python"]["allowed_imports"].

        # Agent capability surfaces (#508): what the agent can touch in this REPL
        # comes from two places, NEITHER a builtin bound here — (1) the `jaz`
        # namespace: just `jaz.invoke` (framework helpers like describe/Display are
        # delivered as ordinary inputs by the host, not bound under `jaz.*`),
        # assembled in library/jaz.py and added via the `libraries` list
        # in invoke.py; and (2) import policy: the `allowed_imports` allow-list enforced in
        # `secure_compile` (an AST restriction; #688 moved it back into the compiler from
        # the former ImportPolicyHook). Don't reach for `allowed_builtins` to add either.

        # Inputs arrive already payload-resolved: the `__jaz_get__` protocol (a
        # framework input concern, not a REPL one) is applied by the core agent loop
        # before initialize() — see jaz.inputs.resolve_inputs — so the REPL binds
        # `inputs` directly. This is where a `jaz.Display` directive's payload lands
        # too: #510 folded the display directive onto the same `__jaz_get__` protocol, so the
        # agent-loop resolution unwraps the directive to its real value here, and the
        # REPL never special-cases it. (Direct callers that pass raw wrappers should
        # resolve them first via resolve_inputs.)

        # Initialize magic variables for accessing initialization info. The former
        # `__task__` builtin is gone (#538): the prompt is now an ordinary input, so the
        # agent reaches it as the corresponding input variable (conventionally `task`)
        # via `__inputs__` / its direct binding, not a dedicated magic variable.
        #
        # Inputs are ALWAYS bound now (#568 removed the `expose_inputs_in_repl` flag): both the
        # `__inputs__` snapshot and the direct input-name bindings (below) are created here. "Inputs
        # off" is the evals `WithholdInputsFromREPL` hook, which drops those bindings (and
        # `__inputs__`) via a DropVariables effect on the first turn — the inputs stay *passed to the
        # invoke* (so they still render in the user prompt), they just aren't REPL variables. The
        # prompt's closing sentence is switched to the withheld wording by a config-set
        # `user_prompt_template` (paired with the hook), the same way history pairs with a
        # config-set description.
        allowed_builtins["__inputs__"] = inputs.copy()

        # __history__ is ALWAYS present in the REPL state and owned by the core agent loop
        # (the single writer of iteration entries) — the former `repl_history` on/off flag was
        # removed (#568): history now lives in the core unconditionally, and "history off" is a
        # HOOK (DisableREPLHistory, in evals/) that emits a DropVariables({"__history__"})
        # effect so the binding is removed each turn and agent access raises NameError; the prompt
        # omits the mention via a config-set description (no core `repl_history` flag). The
        # driver builds the list (starting empty — there is no index-0 init snapshot anymore, #568;
        # see repl.history) and passes it as ``initial_repl_history``; the REPL only surfaces it by
        # reference so the loop's appends are visible to agent code. Direct callers (e.g. tests) that
        # pass no list get an equivalent empty default here. See ``agent`` and
        # ``design/design_features/basics.md``.
        # An empty list is the whole initial history — there is no index-0 init snapshot (#568):
        # the old ``REPLHistoryInit(inputs)`` duplicated ``__inputs__`` and leaked withheld inputs.
        allowed_builtins["__history__"] = (
            initial_repl_history if initial_repl_history is not None else []
        )

        init_repl_state_locals: dict[str, object] = {"__builtins__": allowed_builtins}
        # Bound under the bare name `invoke` — there is no `jaz` namespace wrapping it (see
        # jaz._invoke_tool). Deliberately bound BEFORE the inputs below, so an input literally
        # named `invoke` shadows the primitive rather than being refused. That ordering predates
        # this change (an input named `jaz` shadowed the namespace the same way); the bare name
        # only makes the collision likelier, and which names a host may pass is the host's call.
        if invoke_tool is not None:
            init_repl_state_locals["invoke"] = invoke_tool
        # Always bind the inputs as REPL variables (#568); WithholdInputsFromREPL drops them.
        # TODO(#927): unlike the AddInputs/AddVariables effect path (which raises SandboxKeyError on
        # __builtins__), this top-level bind does not refuse a caller-passed __builtins__ input, so
        # it would clobber the sandbox dict. Refuse it here too for a uniform guarantee.
        init_repl_state_locals.update(inputs)

        # Return the per-invoke state as one object (#1058); the REPL itself is untouched. There is
        # no ``copy.copy`` and no per-attribute rebinding: the whole "no mutable attribute is
        # shared" hazard is gone because the REPL holds nothing per-invoke to share. The sandbox
        # allow-lists live only on the config REPL and are read (never mutated) on every exec, so
        # one configured REPL safely serves every invoke; the native-``raise`` tagging scratch is
        # call-local per exec (``_new_raise_tagger``), not an attribute, so concurrent execs from
        # sibling invokes on the shared template don't race on it.
        return REPLState(
            repl_state_locals=init_repl_state_locals,
            invoke_tool=invoke_tool,
            session_id=session_id,
        )

    @override
    def add_inputs(self, state: REPLState, inputs: dict[str, object]) -> None:
        """Add inputs to the REPL environment.

        Applies the same ``__jaz_get__`` payload substitution as initialization, through
        the shared ``jaz.inputs.resolve_inputs`` primitive, so a hook-injected input
        (`AddInputs`) binds the same payload a top-level invoke input would — e.g. an
        injected `Library` binds its root module, not the `Library` wrapper. Both the
        initialize-time and hook-injected binding paths route through that one primitive.

        Updates repl_state_locals directly. Does NOT update __inputs__ magic
        variable as per design decision - that remains a snapshot of initialization.

        Args:
            inputs: Dictionary of variable name -> value to add
        """
        state.repl_state_locals.update(resolve_inputs(inputs))

    @override
    def add_variables(self, state: REPLState, variables: dict[str, object]) -> None:
        """Bind ``variables`` into the REPL namespace **raw** — the per-turn, namespace-level
        counterpart of ``add_inputs``.

        The two differ deliberately (see the class docstrings on ``AddInputs`` vs
        ``AddVariables``): ``add_inputs`` applies ``__jaz_get__`` payload substitution (an
        injected ``Library``/``jaz.wrap`` binds its payload), because an *input* is a
        prompt-level concept that should behave like a top-level invoke input. ``add_variables``
        does **not** resolve — it writes the given objects verbatim into ``repl_state_locals``,
        because a *variable* is a raw namespace binding (a hook that wants a specific object in
        the namespace should get exactly that object, not its ``__jaz_get__`` payload). The
        prompt is untouched either way.

        ``__builtins__`` is refused (it holds the compiler sandbox) — composition
        raises ``SandboxKeyError`` on it via the shared conflict-error rule, so
        it never reaches here through the effect path; this skip is a defensive backstop. Does NOT
        update the ``__inputs__`` snapshot (same as ``add_inputs``).

        Overwriting a name that is **already bound** in the REPL namespace (a top-level input, the
        agent's own variable, or a prior binding) raises ``REPLInputConflictError`` rather than
        silently clobbering it — symmetric with ``AddInputs`` refusing to shadow a caller input, so
        no hook can quietly replace a value the agent (or caller) put there. To *re-bind* a name
        (e.g. to reset it each turn), ``DropVariables`` it first, then ``AddVariables`` — the drop
        makes the intent-to-replace explicit, and the agent loop applies drops before adds so the
        name is free by the time this runs. The drop needs ``allow_missing=True`` if the name may
        not be bound yet. (Names living in ``__builtins__`` — the allowed
        builtins and framework magics — are not checked here; this guards the *variable* namespace.)

        Args:
            variables: Dictionary of variable name -> value to bind (verbatim).
        """
        # Why `__builtins__` is refused: #688/#690 moved the compiler sandbox into that dict,
        # and the composition-time rule that rejects it is ``_coalesce_add``.
        # so binding over it would hand the agent the sandbox itself.
        for name, value in variables.items():
            if name == "__builtins__":
                continue
            if name in state.repl_state_locals:
                raise REPLInputConflictError(
                    f"AddVariables target {name!r} is already bound in the REPL namespace; a "
                    "namespace add must not silently overwrite an existing variable (a top-level "
                    "input, a prior binding, or the agent's own). To re-bind, DropVariables it first."
                )
            state.repl_state_locals[name] = value

    @override
    def drop_variables(
        self,
        state: REPLState,
        names: Iterable[str],
        allow_missing: Iterable[str] = (),
    ) -> None:
        """Remove ``names`` from the REPL namespace (the inverse of ``add_inputs``).

        A name the agent references resolves through Python's chain: ``repl_state_locals``
        (the exec locals/globals) then ``__builtins__``. So to make a name *genuinely*
        undefined we ``pop`` it from **both** — invoke inputs live at the top level
        (``init_repl_state_locals.update(inputs)``), while framework magics like
        ``__history__`` / ``__inputs__`` live inside ``__builtins__``.

        Dropping a name that is **not bound** raises ``MissingDropTargetError`` — a drop of an
        absent name is a hook bug (a typo, or a "re-drop every turn" pattern that should be
        first-turn-only), so it fails loud rather than no-op'ing. This is still order-independent
        and safe for the legitimate *double-drop* case: composition unions the names first (see
        ``dispatcher`` / ``REPLExecContext.dropped_variables``) and this applies the set **once**
        against the pre-drop namespace, so two hooks dropping the same present name coalesce to a
        single deletion and never trip the check. A name in ``allow_missing`` (its ``DropVariables``
        opted into tolerating absence, e.g. ``WithholdInputsFromREPL`` when a ``DropInputs`` may have
        already un-passed the input) is skipped when absent instead of raising.

        ``__builtins__`` *itself* is never removed even if named: it holds the compiler
        sandbox, and unbinding the dict would strip import/attribute/builtins
        policy. (Composition raises ``SandboxKeyError`` on it before reaching here, so via the effect
        path this never runs; this ``continue`` is a defensive backstop.) Removing an individual
        *key inside* ``__builtins__`` (e.g. ``__history__``) is fine — it only hides that
        name, it does not disable the sandbox.
        """
        # The compiler sandbox lives in `__builtins__` as of #688/#690, which is what makes
        # unbinding the dict a policy bypass rather than a mere name removal.
        # ``__builtins__`` is always our sandbox dict (``allowed_builtins``, set at REPL init and
        # never dropped — the loop skips it below), so index + assert directly, matching
        # ``_capture_print``. It is never the CPython builtins *module*: we set the key explicitly,
        # so the interpreter never auto-injects the module form.
        builtins = state.repl_state_locals["__builtins__"]
        assert isinstance(builtins, dict)
        allow_missing = set(allow_missing)
        for name in names:
            if name == "__builtins__":
                continue
            if name not in state.repl_state_locals and name not in builtins:
                if name in allow_missing:
                    continue  # a defensive drop that tolerates the name already being unbound
                raise MissingDropTargetError(
                    f"DropVariables target {name!r} is not bound in the REPL namespace; "
                    "dropping an absent name is a hook bug (drop it only when it is present, "
                    "e.g. once on the first REPLExecEnter)."
                )
            state.repl_state_locals.pop(name, None)
            builtins.pop(name, None)

    @contextmanager
    def _capture_print(
        self, state: REPLState, string_io: StringIO
    ) -> Generator[None, None, None]:
        """Capture this exec's stdout into ``string_io``.

        Two composed layers (see repl/stdout_proxy.py):
          * swap ``builtins["print"]`` for a closure over ``string_io`` — the fast
            lexical path, which also survives into a raw worker thread; and
          * point the ``sys.stdout`` proxy's ``_current_buffer`` at ``string_io`` for
            this context, so ``pprint`` / ``sys.stdout.write`` / prints inside imported
            functions (which reach the *stream*, not the swapped name) are captured too.

        Also sets up the parent output channel so child jaz.invoke() calls can send
        messages back to this REPL's output via parent_print().
        """
        from jaz._parent_output import (
            reset_parent_output_channel,
            set_parent_output_channel,
        )
        from jaz.repl.stdout_proxy import reset_current_buffer, set_current_buffer

        builtins = state.repl_state_locals["__builtins__"]
        assert isinstance(builtins, dict)
        if "print" in builtins:
            old_print = builtins["print"]
            old_print_exists = True
        else:
            old_print = None
            old_print_exists = False
        new_print = _make_print(string_io)
        builtins["print"] = new_print

        # Set parent output channel so child invokes can send messages here
        output_token = set_parent_output_channel(lambda msg: new_print(msg))
        # Route the stdout stream (proxy) to this exec's buffer for this context.
        buffer_token = set_current_buffer(string_io)

        try:
            yield
        finally:
            reset_current_buffer(buffer_token)
            reset_parent_output_channel(output_token)
            if old_print_exists:
                builtins["print"] = old_print

    def _parse_and_transform(
        self, src: str, tag_raise: Callable[[object], object]
    ) -> tuple[ast.Module, CookieJar] | None:
        """Parse *src*, rewrite native ``return`` / ``raise``, and inject the except-guards.

        Returns ``(tree, jar)`` on success, ``None`` on syntax error. The returned :class:`CookieJar`
        carries the return-signal / tag-raise / exception references as co_consts cookies (seeded
        here, where all three are in scope); :func:`secure_compile` substitutes them into the
        compiled code object. ``tag_raise`` is the caller's per-exec tagger (see
        ``_new_raise_tagger``). Parsing — including the module-level-``return`` tolerance and the
        (defensive) def-wrap fallback — is delegated to :func:`parse_allowing_toplevel_return`.
        """
        tree = parse_allowing_toplevel_return(src)
        if tree is None:
            return None
        jar = CookieJar()
        jar.seed("return_signal", _JazReturnSignal)
        jar.seed("tag_raise", tag_raise)
        jar.seed("exception", Exception)
        # FatalError is Exception-rooted, so Defense B's narrowing to `Exception` does NOT
        # protect it and its base class no longer does either — the cookie guard is what keeps
        # the agent's own handlers from swallowing an abort. See jaz.exceptions.FatalError.
        jar.seed("abort", FatalError)
        jar.seed("reraise_abort", _reraise_first_abort)
        _NativeReturnRaiseTransformer(allow_raise=self.allow_raise, jar=jar).visit(tree)
        _ExceptGuardTransformer(jar).visit(tree)
        ast.fix_missing_locations(tree)
        return tree, jar

    def _compile_native(
        self,
        src: str,
        code_id: str,
        filename: str,
        *,
        allow_top_level_await: bool,
        tag_raise: Callable[[object], object],
    ) -> CodeType | ExecResult[Any]:
        """Parse + transform native ``return``/``raise`` and compile. Returns the compiled
        code object, or an error ``ExecResult`` to short-circuit. Shared by the sync ``_exec``
        and async ``_aexec`` - they differ only in ``allow_top_level_await`` and the (sync vs
        async) execution call. ``tag_raise`` is the caller's per-exec tagger (see
        ``_new_raise_tagger``), seeded into the compiled code's cookie jar.

        Return-*type* and return-*value* enforcement do not happen here; they live in hooks
        (``ReturnType`` / ``ValidateReturn``) on the REPLExecExit effect path, so this method
        only parses, transforms, and compiles the code.
        """
        result = self._parse_and_transform(src, tag_raise)
        if result is None:
            return self._handle_syntax_error(src, code_id, filename)
        tree, jar = result

        # ---- Apply policy visitors and compile ----
        try:
            return secure_compile(
                src,
                filename=filename,
                mode="exec",
                allow_top_level_await=allow_top_level_await,
                allowed_imports=self.allowed_imports,
                allowed_attributes=self.allowed_attributes,
                allowed_write_attributes=self.allowed_write_attributes,
                cookie_jar=jar,
                tree=tree,
            )
        # Catch ImportError too, not just SyntaxError: ImportPolicyChecker raises
        # ImportError (AllowedAttributesChecker raises SyntaxError), and a policy violation
        # must be a recoverable Continue here - not an uncaught exception out of _exec. This
        # is a compile-only step (no user code runs, so no foreign-timeout / fatal exception
        # to propagate), so a plain catch suffices.
        except (SyntaxError, ImportError) as e:
            formatted_output = self._format_error(e)
            return Continue(
                output=formatted_output,
                exception=e,
            )

    def _native_exception_result(
        self, e: BaseException, string_io: StringIO, timeout_owner: object
    ) -> Continue:
        """ExecResult for a non-signal exception from native-return exec (shared sync/async).

        A foreign (outer) timeout or fatal exception cannot become a Continue the
        inner agent reacts to: it **re-raises** out of ``exec``,
        so the sub-invoke's spans close and the exception propagates to the parent exec
        that owns the deadline (or, for a fatal, past every agent). Everything else is
        recoverable feedback.
        """
        _reraise_if_fatal(e, timeout_owner)
        formatted_output = self._format_error(e, string_io.getvalue())
        return Continue(
            output=formatted_output,
            exception=e,
        )

    def _handle_native_return(
        self, value: object, string_io: StringIO
    ) -> Return[Any] | Continue | Raise:
        # Reject a finishing ``return`` when this same code also printed output, so the agent
        # reviews the output before finishing (KEEP from the old RETURN-command path). Gated by
        # reject_finish_on_printed_output; when it fires the return is NOT executed and the
        # agent must re-run only the ``return`` statement.
        if self.reject_finish_on_printed_output and string_io.getvalue():
            error = RuntimeError(
                "Your `return` statement was not executed because your code produced "
                "printed output. Please review the output below. If you are "
                "confident and wish to finish, re-run only the `return` statement.\n"
                "⚠️ You cannot continue working in any way on this task after "
                "returning, so make sure you are confident with your work before "
                "returning."
            )
            formatted_output = self._format_error(error, string_io.getvalue())
            return Continue(
                output=formatted_output,
                exception=error,
            )

        # Return-*type* checking and return-*value* validation both moved out of the REPL to
        # hooks (ReturnType / ValidateReturn, on the REPLExecExit effect path) - #528/#568. The
        # REPL no longer knows the return type.
        return Return(return_value=value)

    @staticmethod
    def _new_raise_tagger() -> tuple[list[BaseException], Callable[[object], object]]:
        """Fresh per-exec scratch for native top-level ``raise`` tagging: a ``(tagged, tag_raise)``
        pair, where ``tag_raise`` records an exception by identity into ``tagged`` and returns it so
        ``raise _jaz_tag_raise(exc)`` raises the real exception.

        The exec boundary treats a *tagged* exception that escapes as a finishing ``Raise``; a
        tagged exception the agent's own ``except`` catches simply never reaches the boundary,
        and an untagged one (bare ``raise``, a raise inside a ``def``, an incidental error) is
        never a finish. Tagging a non-exception is harmless — ``raise`` then fails with Python's
        own ``TypeError``, which (being untagged) surfaces as a recoverable error.
        """
        # Call-local, NOT on ``self``: one config REPL is shared across every invoke at a depth
        # (#1058 — there is no per-invoke copy anymore), so a single shared list would be raced by
        # concurrent execs from sibling invokes (e.g. ``asyncio.gather`` over multiple ``ainvoke``
        # under one config) — one exec's reset would wipe another's tag and misclassify its
        # finishing ``raise`` as a recoverable ``Continue``. Per-exec scratch keeps the template
        # holding no exec state.
        tagged: list[BaseException] = []

        def tag_raise(exception: object) -> object:
            if isinstance(exception, BaseException):
                tagged.append(exception)
            return exception

        return tagged, tag_raise

    def _handle_native_raise(
        self,
        exception: BaseException,
        string_io: StringIO,
    ) -> Raise | Continue:
        # ``exception`` is the real, agent-raised exception that escaped its own try/except (it
        # was tagged by ``_tag_raise``). Its ``__cause__`` / ``__context__`` / ``__suppress_context__``
        # were already set by Python's own ``raise`` machinery (we keep the ``from`` clause on the
        # rewritten raise), so no traceback-chaining fix-up is needed here.

        # Reject a finishing ``raise`` when this same code also printed output, so the agent
        # reviews the output before finishing (KEEP from the old RAISE-command path). Gated by
        # reject_finish_on_printed_output; when it fires the finish is turned into a recoverable
        # Continue and the agent must re-run only the ``raise`` statement.
        if self.reject_finish_on_printed_output and string_io.getvalue():
            error = RuntimeError(
                "Your `raise` statement did not finish the task because your code produced "
                "printed output. Please review the output below. If you are "
                "confident and wish to finish, re-run only the `raise` statement.\n"
                "⚠️ You cannot continue working in any way on this task afterwards, "
                "so make sure you are confident with your work before finishing."
            )
            formatted_output = self._format_error(error, string_io.getvalue())
            return Continue(
                output=formatted_output,
                exception=error,
            )
        return Raise(exception=exception)

    def _handle_syntax_error(self, src: str, code_id: str, filename: str) -> Continue:
        """Handle syntax errors (native ``return``/``raise`` model: no old-style command hints)."""
        # TODO: Improve handling of syntax errors
        try:
            # Compile to get the real syntax error
            compile(src, filename, "exec")
        except SyntaxError as e:
            formatted_output = self._format_error(e)
            return Continue(
                output=formatted_output,
                exception=e,
            )

        raise _JazInternalError("Unreachable code in PythonREPL._handle_syntax_error")

    def _repl_filename(self, state: REPLState, code_id: str) -> str:
        # session_id in the filename avoids linecache collisions across sessions
        if state.session_id is not None:
            return f"<repl_{state.session_id}_{code_id}>"
        return f"<repl_{code_id}>"

    # Pre-execution REPL-code validation moved to the ValidateREPLCode hook (#528): it
    # runs the validator at REPLExecSend and supplies a SupplyExecResult (skipping exec) on
    # rejection — the effect-path equivalent of the old _validate_input short-circuit here.

    def _resolve_timeout(
        self, src: str, exec_timeout_override: float | None
    ) -> float | None:
        """This exec's wall-clock bound: the code's own ``# timeout:`` pragma if it has one,
        else the caller's per-exec override, else the configured ``exec_timeout``. With
        ``allow_timeout_pragma=False`` the pragma is skipped entirely — not even parsed, so a
        malformed one cannot fail code the REPL is meant to treat as an ordinary comment.

        Raises:
            REPLTimeoutPragmaError: The code's leading ``# timeout:`` comment is malformed.
        """
        # The pragma outranks the caller's override, not merely the configured default: it is the
        # most specific statement about *this* execution, and the agent that wrote the code is the
        # party that knows the step is a ten-minute build.
        #
        # EXECUTIVE CALL (user, 2026-08-10): the requested value is honored verbatim, with no
        # ceiling and no lower-only rule — which is what the removed `<repl_code timeout="30">`
        # attribute did, and what makes the pragma useful at all (asking for *more* time is the
        # motivating case). Consequence to know before relying on `exec_timeout` as a safety
        # bound: it is not one. An agent can grant itself an arbitrary finite bound, so a host
        # that needs a hard wall-clock limit must impose it outside the REPL. What still holds is
        # an enclosing exec's deadline (a nested invoke cannot outlive its parent's bound) and the
        # cost/iteration budgets, which the pragma does not touch. A host that wants `exec_timeout`
        # back as a real bound sets `allow_timeout_pragma=False`.
        if not self.allow_timeout_pragma:
            return (
                exec_timeout_override
                if exec_timeout_override is not None
                else self.exec_timeout
            )
        pragma_timeout = _parse_timeout_pragma(src)
        if pragma_timeout is not None:
            return pragma_timeout
        if exec_timeout_override is not None:
            return exec_timeout_override
        return self.exec_timeout

    def _exec(
        self,
        state: REPLState,
        src: str,
        code_id: str,
        timeout: float | None,
    ) -> ExecResult[Any]:
        tagged, tag_raise = self._new_raise_tagger()
        filename = self._repl_filename(state, code_id)
        compiled = self._compile_native(
            src, code_id, filename, allow_top_level_await=False, tag_raise=tag_raise
        )
        if not isinstance(compiled, CodeType):
            return compiled

        # ---- Execute ----
        # Tag this exec's deadline (new_owner) so a foreign (outer) timeout firing inside a
        # nested invoke stays distinguishable from our own - see _native_exception_result /
        # _reraise_if_fatal.
        timeout_owner = new_owner()
        string_io = StringIO()
        try:
            with self._capture_print(state, string_io):
                guarded_exec(
                    compiled,
                    state.repl_state_locals,
                    timeout,
                    memory_limit_bytes=self.exec_memory_limit,
                    owner_id=timeout_owner,
                )
        except _JazReturnSignal as sig:
            return self._handle_native_return(sig.value, string_io)
        except BaseException as e:
            # A tagged exception is an agent-authored top-level ``raise`` that escaped its own
            # try/except → finish (Raise). Anything else is incidental/fatal/foreign and goes
            # to _native_exception_result (recoverable Continue, or propagated when fatal).
            if any(e is t for t in tagged):
                return self._handle_native_raise(e, string_io)
            return self._native_exception_result(e, string_io, timeout_owner)
        return Continue(output=string_io.getvalue())

    async def _aexec(
        self,
        state: REPLState,
        src: str,
        code_id: str,
        timeout: float | None,
    ) -> ExecResult[Any]:
        """Async exec: compiles with top-level-await enabled and runs the code as a
        coroutine, so the agent can ``await`` (e.g. ``x = await ainvoke()`` then native
        ``return x``) - #567. Dispatched from ``aexec`` (which owns the shared timeout
        setup). The whole submission runs on the async path; there is no separate command
        syntax to route synchronously.
        """
        tagged, tag_raise = self._new_raise_tagger()
        filename = self._repl_filename(state, code_id)
        compiled = self._compile_native(
            src, code_id, filename, allow_top_level_await=True, tag_raise=tag_raise
        )
        if not isinstance(compiled, CodeType):
            return compiled

        timeout_owner = new_owner()
        string_io = StringIO()
        try:
            with self._capture_print(state, string_io):
                await guarded_aexec(
                    compiled,
                    state.repl_state_locals,
                    timeout,
                    memory_limit_bytes=self.exec_memory_limit,
                    owner_id=timeout_owner,
                )
        except _JazReturnSignal as sig:
            return self._handle_native_return(sig.value, string_io)
        except BaseException as e:
            if any(e is t for t in tagged):
                return self._handle_native_raise(e, string_io)
            return self._native_exception_result(e, string_io, timeout_owner)
        return Continue(output=string_io.getvalue())

    @override
    def exec(
        self,
        state: REPLState,
        src: str,
        code_id: str,
        exec_timeout_override: float | None = None,
    ) -> ExecResult[Any]:
        # TODO: Fix static types
        """Executes REPL code and mutates ``state`` in place.

        Runs the Python code in the REPL. A top-level ``return`` statement exits the REPL
        with that return value; a top-level ``raise`` (when ``allow_raise``) exits with that
        exception. If an error occurs during execution, the error is captured and returned
        back into the REPL as a recoverable :class:`~jaz.repl.Continue` — except a *fatal*
        exception (:class:`~jaz.exceptions.FatalError`, non-``Exception`` signals) or an
        enclosing exec's timeout/memory breach, which **re-raises** out of this call so it
        propagates out of the invoke.

        The code sets its own wall-clock bound with a ``# timeout: <seconds>`` comment among the
        lines that open it (before any real code), which overrides both ``exec_timeout_override``
        and the configured ``exec_timeout`` for this call only. A malformed value there is reported
        as a recoverable :class:`~jaz.repl.Continue` and the code is not executed. Configuring
        ``allow_timeout_pragma=False`` turns that comment back into ordinary text.

        Arguments:
            src: The REPL command to execute
            code_id: The unique identifier for the REPL code

        Returns:
            An :class:`~jaz.repl.ExecResult` — exactly one of :class:`~jaz.repl.Continue`
            (ran, session continues; carries ``output`` and an optional recoverable
            ``exception``), :class:`~jaz.repl.Return` (finished by returning ``return_value``),
            or :class:`~jaz.repl.Raise` (finished by raising ``exception``). The REPL state is
            mutated in place.
        """
        # TODO: Rename code_id to iteration index? (The agent loop passes `str(iteration)`, which
        # is 0-based since #719 — this name predates that and says nothing about the base.)
        try:
            timeout = self._resolve_timeout(src, exec_timeout_override)
        except REPLTimeoutPragmaError as e:
            # A recoverable `Continue`, not a propagated raise: an agent typo in a comment is the
            # same class of mistake as a `SyntaxError`, and the REPL answers those by handing the
            # message back rather than ending the invoke. (`exec` DOES re-raise fatal/foreign
            # exceptions — see `_reraise_if_fatal` — but a pragma typo is neither.)
            #
            # `summarize_exception`, not `_format_error`: no agent frame exists to render (the
            # code never ran), so a traceback would show only stripped engine frames. This is the
            # same rendering the loop used for a parse failure, back when parsing could fail.
            return Continue(output=summarize_exception(e), exception=e)
        # NOTE: __history__ is no longer updated here. It is written exclusively
        # by the core agent loop, once per iteration, from the *finalized* result
        # (after exit-time effect composition); see agent.do_one_repl_iteration. This
        # keeps history a single-writer, core-owned concept and lets a result-override
        # hook (e.g. BudgetForcing) be reflected without the REPL knowing (#448).
        return self._exec(
            state,
            src,
            code_id,
            timeout,
        )

    @override
    async def aexec(
        self,
        state: REPLState,
        src: str,
        code_id: str,
        exec_timeout_override: float | None = None,
    ) -> ExecResult[Any]:
        """Async ``exec``: the code runs as a coroutine on the event loop, so the agent
        can use top-level ``await`` (e.g. ``result = await ainvoke(...)``). The whole
        submission takes the async path (there is no separate command syntax to route
        synchronously); native ``return`` / ``raise`` still finish the REPL.

        The ``# timeout: <seconds>`` pragma applies here exactly as it does to ``exec``.

        Like sync ``exec``, this does not write ``__history__`` — the core agent loop
        is the single writer (see ``exec``'s NOTE).
        """
        # Top-level `await` support is #567. The single-writer rule for `__history__` is
        # #448/#527 — the loop owns the record so a REPL-side write cannot desync it.
        try:
            timeout = self._resolve_timeout(src, exec_timeout_override)
        except REPLTimeoutPragmaError as e:
            return Continue(output=summarize_exception(e), exception=e)  # see `exec`
        return await self._aexec(state, src, code_id, timeout)
