import ast
import math
import numbers
import re
import sys
import threading
import traceback
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from io import StringIO
from types import CodeType, MappingProxyType, TracebackType
from typing import Any, Final, Literal, override

from jaz._component import repr_call
from jaz._string_utils import summarize_exception
from jaz._template_loader import _jinja_env
from jaz.exceptions import (
    FatalError,
    KeyEditDeniedError,
    MissingDropTargetError,
    REPLInputConflictError,
    REPLTimeoutPragmaError,
    _JazInternalError,
    is_fatal,
)
from jaz.llm._types import LLMResponse

from ._ast_utils import parse_or_none
from ._base import BaseREPL
from ._compiler import secure_compile
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
from ._glob_allowlist import Allowlist
from ._permissions import REPLPermissionPolicy, build_allowed_builtins
from ._repl_history import _REPL_HISTORY_DESCRIPTION, REPLHistoryEntry
from ._types import Continue, ExecResult, Raise, Return


class TracebackVerbosity(StrEnum):
    """How much of an error traceback the agent is shown.

    The REPL's own engine frames — the ``.py`` plumbing that ``exec``'s the
    agent's code, sitting *above* the agent's ``<repl_...>`` frame in the stack —
    are ALWAYS stripped: they are an artifact of the REPL implementation, never
    part of the agent's error. These levels say how much of the remaining,
    agent-relevant traceback to keep (most → least): ``ALL`` (``"all"``, the default)
    keeps the agent's own frames plus any library frames its code called into;
    ``REPL_ONLY`` (``"repl_only"``) keeps only the agent's own frames; ``MESSAGE_ONLY``
    (``"message_only"``) keeps no frames — just the exception line.
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


# The allow-list applied when ``repl_configs["python"]`` omits ``allowed_imports``. The production
# value is ``[]`` — deny ALL imports, fail-closed. Exposed as a module constant (rather than an
# inline literal) so the test suite can monkeypatch it to ``["*"]`` (allow all) for tests that
# aren't about the sandbox — incidental ``import`` scaffolding in feature tests then keeps working,
# as a host configuring its own allow-list does. See tests/conftest.py; sandbox tests opt
# into this real default via a pytest marker.
_DEFAULT_ALLOWED_IMPORTS: list[str] = []

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


def _require_allowlist(key: str, value: Any) -> Sequence[str]:
    """Reject a bare ``str`` where a sequence of glob patterns belongs.

    ``str`` IS a ``Sequence[str]``, so ``allowed_read_paths="math"`` would be stored as four
    one-character patterns. Enforcement rejects the string later (``compile_allowlist`` and
    ``secure_open._compile_patterns`` both carry this guard), but the constructor is where a
    setting is validated, and ``description`` reads the stored tuple — without this it would
    describe a file policy nothing enforces.
    """
    if isinstance(value, str):
        raise ValueError(
            f"{key} expects a list of glob patterns, not a bare string {value!r}."
        )
    return value


# Default file-access allowlists — used when the ``repl_configs["python"]`` key is absent (not in
# the config dict at all; a present-but-set value passes through unchanged, see
# ``allowed_read_paths``). The production value is ``[]`` — deny ALL file reads/writes.
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
# THE NEGATION IS ENFORCED ONLY AT `open()`. Like the rest of the list it is both rendered into
# the agent's instructions (verified: the prompt shows "`~/**`, `!~/.jaz/**`") and checked in the
# secure `open` wrapper — but only the second half is a guarantee, and only while `allowed_imports`
# withholds every module that can reach the filesystem another way. Verified: with
# `allowed_imports=["pathlib", "subprocess"]` and exactly the exclusion above, builtin `open` on
# `~/.jaz/credentials.json` raises while `pathlib.Path(...).read_text()` and
# `subprocess.run(["cat", ...])` both return its contents. For credentials specifically an
# instruction is not enough, so keeping them out of a readable file (OS keyring, #1076) is the
# answer that survives a widened `allowed_imports`; the negation is not.
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
      reached. Bare ``except:`` is made safe rather than rejected.

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

    # Defense B replaces #869's `reject_bare_except` ban: making a bare `except:` safe beats
    # rejecting it, since the agent writes them naturally and a rejection costs a whole turn.

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


@dataclass(frozen=True)
class PythonREPLState:
    """The per-invoke state of a :class:`PythonREPL` run — everything one ``initialize()`` builds.

    A REPL is *configured* by construction (its sandbox policy, timeouts, finishing rules) and
    *initialized* per invoke; this holds exactly the per-invoke half. With the state here rather
    than on the REPL, the REPL is pure configuration and one instance safely serves many invokes —
    there is nothing per-run for two of them to share.

    Note the two distinct lifetimes: this is *per-invoke* state. Per-*exec* scratch (the REPL's
    per-turn bookkeeping) does not belong here — it lives for one ``exec`` call and is threaded
    locally, not stored across turns.

    The dataclass is frozen, but that constrains *rebinding*, not contents: a writer mutates
    ``exec_globals`` in place and never reassigns a field.

    Attributes:
        exec_globals: The live globals mapping the agent's code runs against. Hooks see it through
            ``PythonREPL.get_namespace``, a copy with ``__builtins__`` shown read-only.
        session_id: Identifier for the invoke this state belongs to; empty when none was given.
        sandbox_builtins: The dict ``initialize`` installed as ``exec_globals["__builtins__"]``.
            The REPL checks, before it reads or runs anything, that the key still holds this
            very object; agent code that replaced it (through ``globals()``) has corrupted the
            state, and the exec that did so fails with :class:`~jaz.exceptions.KeyEditDeniedError`
            at its end, as does every later read of the sandbox.
        repl_history: The REPL-history list the agent's code reads, one
            :class:`~jaz.repl.REPLHistoryEntry` per turn, appended to by
            ``PythonREPL.update_repl_history``; ``None`` when the history is withheld.
    """

    # Public API (EXECUTIVE CALL, user, 2026-09-14): `initialize()` hands one to a caller and
    # `exec`/`add_to_namespace`/`drop_from_namespace` take it back, which is the rule — a caller is
    # handed one. `BaseREPL` is still generic in its state type, so exporting this obliges no
    # other implementation to name it; it is the shipped REPL's state, advertised for the people
    # who already hold it.
    #
    # Where `__history__` is bound is deliberately NOT in the docstring above: `repl_history` is
    # also `exec_globals["__builtins__"]["__history__"]`, and a shipped docstring promising that
    # layout would make an implementation detail a v1 promise. Recorded here instead, where a
    # maintainer looks.
    #
    # `exec_globals` (EXECUTIVE CALL, user, 2026-09-19): the word "namespace" names exactly one
    # mapping in this package — the agent-visible one `PythonREPL.get_namespace` returns and the
    # `REPLExec*` events carry — and this field is not it: it is the live dict passed to `exec()`
    # as its globals, of which the namespace is a copy with `__builtins__` shown read-only. Naming
    # it `namespace` made the docs and the CHANGELOG call two different mappings by one word. The
    # bare `globals` was not used because it shadows the builtin at every use site.
    #
    # `frozen` makes the "never rebind a field" invariant structural rather than audited: the
    # REPL surfaces these through read-only properties precisely because every writer mutates the
    # *exec_globals dict* in place (which frozen still allows) and never reassigns a field. A fresh
    # `PythonREPLState` is built per invoke, so immutability of the container costs nothing.
    #
    # No `subinvoke` field: `initialize` binds the primitive in the `__builtins__` dict under the
    # bare name `invoke`, which is what makes recursion work, and nothing ever read the field
    # back. Storing it a second time recorded which primitive the run was given, which a shadow
    # can hide (a variable named `invoke` resolves first) but no effect can remove once bound
    # (agent code that reaches `__builtins__` through `globals()` can, TODO(#1720));
    # `DisableRecursion` withholds it before the REPL is built. The record was dead weight: no
    # code consulted it.
    #
    # No `policy_names` field either: it told the sandbox's allowed builtins apart from the
    # framework bindings sharing `__builtins__` with them, when the namespace view merged the
    # latter in. The view is now the agent's `globals()` (see `PythonREPL.get_namespace`), so
    # nothing draws that line.

    exec_globals: dict[str, object] = field(default_factory=dict)
    session_id: str = ""
    # `repr=False`: the same dict already prints as `exec_globals["__builtins__"]`, so showing it
    # again would double a 140-entry mapping in every repr of the state.
    sandbox_builtins: dict[str, object] = field(default_factory=dict, repr=False)
    # `repr=False` for the same reason: when bound, the list prints inside that dict already.
    repl_history: list[object] | None = field(default=None, repr=False)


_ABSENT: Final = object()


def _sandbox(state: PythonREPLState) -> dict[str, object]:
    """The sandbox's builtins dict, after checking that ``exec_globals`` still holds it."""
    # Every read of the sandbox goes through here, and each exec path calls it once the agent's
    # code has run, so the exec that replaced `__builtins__` is the one that fails: the
    # `REPLExec` span and the invoke close `Failed(exc)`, the outcome for machinery that cannot
    # continue, and no LLM query is spent on a state that cannot run code (a check at the next
    # turn's read would bill one whose reply is discarded; found in review). The compiler refuses
    # the bare name in every statement form, so under the default attribute policy the only
    # route left to agent code is a string key through `globals()` / `locals()` / `vars()`
    # (`globals()["__builtins__"] = ...`; allowing `__globals__` opens `f.__globals__[...]`, which
    # this check catches the same way, and a host can edit `exec_globals` directly);
    # replacing `globals` itself was rejected, because a scan of the CodeAct+subagents traces
    # (2026-10-17) found agents keep their state through it in tens of thousands of steps, and
    # no faithful replacement could be shown not to change those runs. So the write is detected,
    # not prevented (the user's call, 2026-10-18), and only by its end state at the end of that
    # exec: a function defined after the write in the same exec already resolves its builtins
    # through the replacement, and a write undone before the exec ends is not seen (it gains
    # nothing, since such a dict holds only what the agent already had). The error ends the
    # invoke rather than becoming feedback, because the state cannot normally be repaired from
    # inside: the agent holds no reference to the dict it displaced unless it saved one first
    # (`saved = globals()["__builtins__"]` is a read, and allowed), so every later turn would
    # fail the same way until the iteration limit; a `return` or
    # finishing `raise` in the same exec is discarded too, computed on a state the sandbox no
    # longer governs (an untagged fatal error re-raises before this check). Not a
    # `Raise` result: that is documented as the agent's own top-level `raise`, and a hook
    # scoring agent behaviour would misattribute it. Not a `FatalError`: that would end the
    # parent of a corrupted sub-invoke too, and the parent's own REPL is intact; with an
    # ordinary exception the parent sees the child's failure as feedback and carries on (a test
    # pins it). `KeyEditDeniedError` because that is the verdict: the agent bound the one key
    # the REPL lets nobody bind, by the one route the compiler cannot refuse. The user's calls
    # (2026-10-18): end only the invoke itself; reuse the existing class. Checking at the end of
    # the exec rather than at the next turn is an implementation choice, not decided by the
    # user. An explicit check rather than an `assert`, which `python -O` removes.
    #
    # Outside the check: an edit of the dict's *entries* through the same string key
    # (`globals()["__builtins__"].pop("__import__")`, `[...]["len"] = ...`, `.clear()`) keeps the
    # identity and is not detected; the rest of the invoke runs against the edited dict. It gains
    # the agent nothing, since it can store only objects it already holds, and a sub-invoke
    # builds its own dict, so the harm is to the agent's own later turns (imports fail, a
    # replaced `__history__` hides the real list from the agent while `update_repl_history`
    # keeps appending to `state.repl_history`). A hook does see it: the namespace view reads
    # the live dict, not a snapshot. Whether to detect it, and how, is TODO(#1765); verified by
    # running each edit.
    builtins = state.exec_globals.get("__builtins__", _ABSENT)
    if builtins is not state.sandbox_builtins:
        state_now = (
            "missing" if builtins is _ABSENT else f"now {type(builtins).__name__}"
        )
        raise KeyEditDeniedError(
            f"The REPL's `__builtins__` is no longer the sandbox's own dict (it is {state_now}), "
            "so this REPL cannot run code; agent code gets at it through a string key on "
            "globals()/locals()/vars(). Nothing restores it; the invoke whose REPL this is "
            "fails here, and an enclosing invoke is unaffected."
        )
    return state.sandbox_builtins


# How to withhold each name the framework puts in `__builtins__`, for a drop that names one.
_WITHHOLD_WITH = {"invoke": "DisableRecursion", "__history__": "WithholdREPLHistory"}


def _missing_note(missing: list[str], builtins: Mapping[str, object]) -> str:
    """What a missing-target error adds about names ``__builtins__`` holds or the framework
    withholds by effect."""
    provided = [name for name in missing if name in builtins]
    withheld = [name for name in missing if name in _WITHHOLD_WITH]
    note = ""
    if provided:
        note += (
            f" {provided!r} {'is' if len(provided) == 1 else 'are'} provided by the sandbox's "
            "builtins, not bound as a variable, so there is nothing to drop."
        )
    if withheld:
        # Named whether or not the name is bound: a drop of `__history__` an earlier
        # `WithholdREPLHistory` already withheld lands here too, and the generic advice above
        # ("drop it only when it is present") would send that hook the wrong way.
        note += (
            " "
            + "; ".join(
                f"to withhold {name!r}, emit {_WITHHOLD_WITH[name]} at InvokeEnter"
                for name in withheld
            ).replace("to", "To", 1)
            + ", not a drop."
        )
    if provided or withheld:
        note += (
            " To withhold a variable of that name that may already be unbound, pass "
            "allow_missing=True."
        )
    return note


class PythonREPL(BaseREPL[PythonREPLState]):
    """REPL implementation that enforces compile-time safety policies.

    Uses native ``return`` / ``raise`` semantics: a top-level ``return`` / ``raise`` statement
    (not nested inside a function or class) finishes the REPL, producing a ``Return`` / ``Raise``
    result. An AST transform rewrites such top-level statements into the finish mechanism.

    Configuration flags:
        allow_raise: If True (default), a top-level ``raise`` exits the REPL with that exception.
            If False, only ``return`` is available for finishing (a top-level ``raise`` is left
            as ordinary Python and not treated as a finish).
    """

    # The REPL holds NO per-invoke state (#1058): ``initialize`` returns a ``PythonREPLState``
    # and every stateful method (``exec``/``add_to_namespace``/``drop_from_namespace``) takes it back as
    # an argument.
    # A configured REPL is a pure, reusable template — two invokes cannot bleed into each other
    # because there is no ``self`` for either to write to.
    allow_raise: bool

    # The five sandbox allow-lists below are frozen tuples in private slots, read through
    # setter-less properties (#741) — which is why they no longer carry a class-level annotation
    # here beside ``allow_raise``. They are the sandbox boundary, and ``secure_compile`` re-reads
    # them on EVERY exec, so a mutation takes effect immediately and for every invoke this REPL
    # serves: one configured REPL is held per config, and a ``ConfigOverride`` that does not
    # override ``repl`` shares that instance by reference with the base config (see
    # ``Config._folded``). Both ways of widening it are therefore closed structurally rather than
    # by convention — the tuple makes ``repl.allowed_imports.append("os")`` an ``AttributeError``
    # (and item assignment a ``TypeError``), the missing setter makes
    # ``repl.allowed_imports = [...]`` an ``AttributeError`` too. A getter returning a defensive
    # copy of a list was rejected: it would turn widening a *sandbox* into a silent no-op, and it
    # would allocate on the per-exec compile path. ``tuple(...)`` in ``__init__`` copies as well as
    # freezes, which closes the older aliasing hole where an explicitly-passed allow-list was
    # stored by reference (only the ``None``-default branch copied) and stayed writable through the
    # caller's own name. Enforcement only iterates and tests membership, both of which a tuple
    # serves.

    @property
    def allowed_imports(self) -> tuple[str, ...]:
        """Module names the agent's code may import (``["*"]`` allows all) — read-only."""
        return self._allowed_imports

    @property
    def allowed_attributes(self) -> tuple[str, ...]:
        """Attribute-name patterns the agent's code may read — read-only."""
        return self._allowed_attributes

    @property
    def allowed_write_attributes(self) -> tuple[str, ...]:
        """Attribute-name patterns the agent's code may write — read-only.

        Derived from ``allowed_attributes`` at construction, not a constructor parameter.
        """
        return self._allowed_write_attributes

    @property
    def allowed_read_paths(self) -> tuple[str, ...]:
        """Glob patterns the agent is told to read files under — read-only.

        Enforced only at the ``open`` builtin; see ``__init__`` for the two layers and how
        ``allowed_imports`` bounds the enforced one.
        """
        return self._allowed_read_paths

    @property
    def allowed_write_paths(self) -> tuple[str, ...]:
        """Glob patterns the agent is told to write files under — read-only.

        Enforced only at the ``open`` builtin; see ``__init__`` for the two layers and how
        ``allowed_imports`` bounds the enforced one.
        """
        return self._allowed_write_paths

    @property
    def repl_history_description(self) -> str:
        """The ``__history__`` description (see :attr:`BaseREPL.repl_history_description`).

        A constant: this REPL's namespace is in-process, so it always binds the list, and it
        describes the :class:`~jaz.repl.REPLHistoryEntry` records it appends.
        """
        # A fact about the REPL, not a setting (executive call, user, 2026-09-19): a `bind_history`
        # constructor flag was tried and removed. A REPL with an in-process Python namespace can
        # always surface the list, and withholding history from an agent is a policy — a hook's
        # `WithholdREPLHistory` effect, per invoke — not a property of the interpreter. `None`
        # exists for REPLs that genuinely cannot bind it.
        return _REPL_HISTORY_DESCRIPTION

    @property
    def description(self) -> str:
        """The agent-facing description of this REPL, rendered from its own settings."""
        # Prompt prose is config-driven, but core renders the template with only *core-owned*
        # variables: `allow_raise`, `exec_timeout`, `allow_timeout_pragma` and the described
        # sandbox allow-lists (imports, file read/write) — core owns and enforces all of them, and
        # a description that disagreed with enforcement would be a bug (offering `raise` as a
        # finish when the REPL forbids it; quoting a wall-clock bound the exec does not apply;
        # offering an override the REPL ignores; promising imports or files the sandbox denies).
        # Reading them off the instance is what makes disagreement impossible: there is no second
        # bag of values to fall out of step with `self`. The *hook-owned* feature flags
        # (exposed inputs, return restriction, …) are a host-configuration concern, baked into
        # the host's own template at config time, never passed at render time. Core deliberately
        # has no generic vars escape hatch: a template
        # that references an unbaked toggle raises here rather than rendering an empty string.
        #
        # The sandbox text used to be *appended* here by three prompt-building helpers, to whatever
        # template was configured; it now lives inside the template, which is what makes that
        # template the whole description rather than most of it. Two consequences, both accepted
        # (user's call): a configured `repl_description_template` no longer inherits the policy
        # text, and `allowed_attributes` is described nowhere — so `allowed_attributes` feeds
        # enforcement only.
        #
        # Each allow-list goes over as an `Allowlist` value — the patterns plus "is this the
        # canonical allow-all", decided here — so the template reads attributes and never calls
        # into the sandbox. See `Allowlist` for what "unrestricted" does and does not mean.
        template = _jinja_env.get_template(self.repl_description_template)
        return template.render(
            allow_raise=self.allow_raise,
            exec_timeout=self.exec_timeout,
            allow_timeout_pragma=self.allow_timeout_pragma,
            allowed_imports=Allowlist.from_patterns(self.allowed_imports),
            allowed_read_paths=Allowlist.from_patterns(self.allowed_read_paths),
            allowed_write_paths=Allowlist.from_patterns(self.allowed_write_paths),
        )

    def _format_error(self, exception: BaseException, stdout: str = "") -> str:
        """Format an exception's full output (stdout + filtered traceback) for an
        ERROR result. The agent-facing summary is derived separately by each consumer
        calling ``summarize_exception`` on the stored exception, so it isn't built here.

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
        # Formatting from the stored exception rather than the live traceback is #566 step C;
        # #590 is the follow-up that keeps the agent's view stable across the protocol seam.
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

    # Defaults are spelled as literals rather than private constants because this signature is
    # published and griffe renders a default as its source expression: a constant's name there
    # is an unimportable name in the reference that hides the value. The sandbox allow-lists are
    # the exception — indirected through module constants so the test suite can relax them.
    def __init__(
        self,
        *,
        exec_timeout: float | None = 30.0,
        exec_memory_limit: int | None = None,
        allow_timeout_pragma: bool = True,
        allow_raise: bool = True,
        repl_description_template: str = "jaz/python_repl_description.jinja2",
        reject_finish_on_printed_output: bool = True,
        allowed_imports: list[str] | None = None,
        allowed_attributes: list[str] | None = None,
        allowed_read_paths: list[str] | None = None,
        allowed_write_paths: list[str] | None = None,
        traceback_verbosity: (
            TracebackVerbosity | Literal["all", "repl_only", "message_only"]
        ) = TracebackVerbosity.ALL,
    ) -> None:
        """Configure a REPL. The result is **not yet usable** — call :meth:`~PythonREPL.initialize` first.

        Construction takes only *configuration*; the per-invoke state (inputs, the subinvoke
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
                (deny all); ``["*"]`` is the explicit allow-all. Granting any module that
                can reach the filesystem — directly, or by spawning a process that does —
                lifts the *enforcement* of ``allowed_read_paths`` / ``allowed_write_paths``
                for that route; the agent is still instructed to respect them.
            allowed_attributes: Attribute allow-list. ``None`` applies the secure default.
            allowed_read_paths: gitignore-style globs for file reads. ``None`` applies
                the fail-closed default (deny all). These patterns do two things, and only
                one of them is enforced: the agent is **told** to respect them — they are
                rendered into its instructions whatever else is configured — and the
                ``open`` builtin **refuses** paths outside them. Nothing checks them on any
                other route to the filesystem, so granting a module that can reach it —
                directly, or by spawning a process that does — leaves the instruction and
                removes the enforcement. Past that point these are a rule the model is
                asked to follow rather than a limit it cannot exceed.
            allowed_write_paths: gitignore-style globs for file writes. ``None`` applies
                the fail-closed default (deny all). Instructed and enforced the same way as
                ``allowed_read_paths``, with the same ``allowed_imports`` caveat.
            traceback_verbosity: How much of the error traceback the agent sees, a
                :class:`TracebackVerbosity` or one of its values as a literal string.
                A string whose value is not known statically — one read from your own
                config — needs wrapping: ``TracebackVerbosity(value)``, which raises
                ``ValueError`` on an unknown one. The REPL's engine
                frames above the agent's code are always stripped; this controls
                the agent-relevant remainder: "all" (default; agent frame +
                library frames it called into), "repl_only" (agent frames only —
                drops library frames and their identity-leaking paths), or
                "message_only" (just the exception message).

        Raises:
            ValueError: If ``traceback_verbosity`` is not a known value.
        """
        # `traceback_verbosity` spells its string arm as a `Literal`, not `| str`. The obvious
        # annotation is redundant: `TracebackVerbosity` is a `StrEnum`, so it is already a `str`
        # subclass and `TracebackVerbosity | str` means exactly `str` to a type checker — every
        # typo passes the check and fails here at run time instead. The `Literal` accepts the
        # same *values* (the enum, or any of its values as a plain string) and rejects
        # `"repl-only"` statically. It accepts strictly fewer *expressions*, which is the real
        # cost: a `str`-typed variable — `v: str = cfg["traceback_verbosity"]` — type-checked
        # before and does not now. That caller wraps it, `TracebackVerbosity(v)`, which is
        # where the `ValueError` they already had to handle comes from. The docstring says so.
        # It also costs a second copy of the values, which
        # `test_the_traceback_verbosity_annotation_lists_exactly_the_enum_values` keeps honest.
        # Config-authored params are unaffected either way: `_instantiate` calls `cls(**params)`
        # off a `Mapping[str, Any]`, so those values are never checked against this signature.
        #
        # EXECUTIVE CALL (user, 2026-09-20) — the path allow-lists are a TWO-LAYER policy, and
        # the docstring says both layers. The alternative, rejected: describe them as pure
        # enforcement, which reads honestly ("worth nothing once you grant `pathlib`") but makes
        # the prompt line a falsehood, since `python_repl_description.jinja2` renders "You may
        # only read files whose path matches: ..." from these patterns on every configuration —
        # verified unchanged under `allowed_imports` of `[]`, `["pathlib"]` and `["*"]`. Under a
        # model-level policy that is the instruction doing its job, not a lie, and a framework
        # whose own posture is "defense-in-depth, not an OS boundary" (secure_open) has no story
        # for why such a policy exists at all if the instruction counts for nothing. So granting
        # a filesystem-capable module degrades these from instructed+enforced to instructed
        # only — weaker, not void — and "a rule the model is asked to follow rather than a limit
        # it cannot exceed" is the distinction a configurer needs. The template is deliberately
        # NOT weakened to match: as an instruction it is correct, and softening it would make
        # agent behaviour worse for no gain. #1537 tracks the enforcement half.
        #
        # EXECUTIVE CALL (user, 2026-08-07): construction takes configuration ONLY, and
        # `initialize` takes only the invoke-time arguments that build REPL *state*. Previously
        # the two were tangled — `__init__` took an already-built namespace (`repl_state_locals`)
        # alongside half the settings, while the other half (`allowed_read_paths`,
        # `allowed_write_paths`) reached the object only as a `config` dict `initialize` folded
        # into a permission policy and never stored.
        #
        # The payoff is that `declared_init_keys(PythonREPL)` is now the structural source of
        # truth for `repl.params`, exactly as it already is for an LLM backend — so the resolved
        # class can be called with the authored params directly, with no per-field
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
        # can relax them for tests that are not about the sandbox.
        #
        # Stored as frozen tuples in private slots, read through the read-only properties on the
        # class (#741 — the rationale is beside them).
        self._allowed_imports: tuple[str, ...] = tuple(
            _DEFAULT_ALLOWED_IMPORTS
            if allowed_imports is None
            else _require_allowlist("allowed_imports", allowed_imports)
        )
        self._allowed_attributes: tuple[str, ...] = tuple(
            DEFAULT_ALLOWED_ATTRIBUTES
            if allowed_attributes is None
            else _require_allowlist("allowed_attributes", allowed_attributes)
        )
        # Derived, not a constructor parameter: the read/write split exists to undo the *default's*
        # dunder re-admission on the write axis, so a host that states its own `allowed_attributes`
        # gets that one list on both axes rather than a silent subtraction it never asked for.
        self._allowed_write_attributes: tuple[str, ...] = tuple(
            DEFAULT_ALLOWED_WRITE_ATTRIBUTES
            if allowed_attributes is None
            else self._allowed_attributes
        )
        self._allowed_read_paths: tuple[str, ...] = tuple(
            _DEFAULT_ALLOWED_READ_PATHS
            if allowed_read_paths is None
            else _require_allowlist("allowed_read_paths", allowed_read_paths)
        )
        self._allowed_write_paths: tuple[str, ...] = tuple(
            _DEFAULT_ALLOWED_WRITE_PATHS
            if allowed_write_paths is None
            else _require_allowlist("allowed_write_paths", allowed_write_paths)
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
    def __repr__(self) -> str:
        return repr_call(
            "PythonREPL",
            ("exec_timeout", self.exec_timeout),
            ("exec_memory_limit", self.exec_memory_limit),
            ("allow_timeout_pragma", self.allow_timeout_pragma),
            ("allow_raise", self.allow_raise),
            ("repl_description_template", self.repl_description_template),
            ("reject_finish_on_printed_output", self.reject_finish_on_printed_output),
            ("allowed_imports", list(self._allowed_imports)),
            ("allowed_attributes", list(self._allowed_attributes)),
            ("allowed_read_paths", list(self._allowed_read_paths)),
            ("allowed_write_paths", list(self._allowed_write_paths)),
            ("traceback_verbosity", self.traceback_verbosity),
        )

    def initialize(
        self,
        inputs: dict[str, object],
        subinvoke: Callable[..., object] | None,
        session_id: str = "",
        repl_history_withheld: bool = False,
    ) -> PythonREPLState:
        """Build one invoke's ``PythonREPLState``; the REPL stays a reusable config template.

        Takes only the *invoke-time* arguments — everything here belongs to one run rather than
        to the REPL's configuration, which was fixed at construction. Builds the namespace
        (``__builtins__`` under this REPL's sandbox policy, ``__history__``, ``invoke``, and a
        binding per input) inside the returned state; the REPL itself is not touched, so one
        configured instance serves every invoke.

        Arguments:
            inputs: The invoke's inputs, bound under their own names.
            subinvoke: The recursive sub-invoke primitive to bind as ``invoke``, or None.
            session_id: Unique session identifier for linecache filenames.
            repl_history_withheld: True to bind no ``__history__`` (the invoke withholds it); the
                state's ``repl_history`` is then ``None`` and nothing is recorded.

        Raises:
            KeyEditDeniedError: An input is named ``__builtins__``, which holds the sandbox.
        """
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
            policy=permission_policy,
        )

        # Import enforcement is two layers (#470/#688): the static ImportPolicyChecker in
        # secure_compile catches `import`/`from` statements, and ReservedNamesChecker beside it
        # refuses every use of the `__builtins__` name (read or bound) at compile time;
        # the policy-bound `__import__` wrapper installed by build_allowed_builtins above is the
        # runtime backstop for every `__import__(...)` call — literal or computed
        # (`__import__("o" + "s")`) — which the static pass does not inspect (#742), the machinery
        # every `import` statement's IMPORT_NAME still calls, and the true enforcement point for any
        # route that reaches the sandbox's `__builtins__` (frame introspection, str.format — which
        # the name reservation cannot close; see ReservedNamesChecker). Reserving the `__builtins__`
        # name statically closes the deferred-execution hole a runtime-only guard left open for the
        # `__builtins__["__import__"]` route: a returned closure over it escapes the sandbox
        # unexecuted. (A follow-up reserves the `__import__` name too, for the bare form.)
        # The two import layers follow the REPL's `allowed_imports` setting; the reservation is
        # unconditional.

        # Agent capability surfaces (#508): what the agent can touch in this REPL
        # comes from two places, NEITHER one of the sandbox's policy builtins — (1) the bare
        # `invoke` tool, a framework binding added to `__builtins__` below (framework helpers are
        # delivered as ordinary inputs by the host, not bound under a `jaz` namespace); and (2)
        # import policy: the
        # `allowed_imports` allow-list enforced in
        # `secure_compile` (an AST restriction; #688 moved it back into the compiler from
        # the former ImportPolicyHook). Don't reach for `allowed_builtins` to add either.

        # Inputs bind as themselves: the REPL binds exactly the object the caller passed, with no
        # substitution step in between.

        # No magic variables for initialization info: the prompt is an ordinary input, reached as
        # its own input variable (conventionally `task`), and there is no `__inputs__` snapshot --
        # an input is reachable by its own name and nothing else.
        #
        # Why no snapshot, even as a convenience: `__inputs__` would be a SECOND path to every
        # input value, making "the agent cannot see input X" two removals rather than one. A hook
        # that unbound the names but forgot the snapshot would leave the values readable as
        # `__inputs__["x"]` while the prompt said they were withheld -- silently measuring the
        # opposite of what it claimed. One binding per input has no such second door.
        #
        # A snapshot kept in `allowed_builtins` rather than the namespace would also survive a
        # namespace-level drop, removable only by a `DropFromNamespace` that knew to name it -- a
        # trap for any hook author who reasoned about the REPL namespace alone.
        #
        # Inputs are ALWAYS bound (there is no `expose_inputs_in_repl` flag); the direct
        # input-name bindings below are the whole surface. "Inputs off" is a hook dropping those
        # bindings on the first turn with `DropFromNamespace` -- the inputs stay *passed to the
        # invoke*, so they still render in the user prompt unless the hook edits that too, as
        # jaz-evals-internal's `CodeAct` does; they just are not REPL variables.

        # `__history__` is this REPL's: the state owns the list, `update_repl_history` appends one
        # entry per turn, and this binding surfaces the same object so agent code sees each
        # append. "History off" is a policy a hook applies with `WithholdREPLHistory`, per invoke,
        # and the loop passes it here as `repl_history_withheld`; it is not this REPL's to decide.
        # The binding also follows `repl_history_description`, which on this class is a constant
        # but which a subclass may override to `None`: the prompt describes the name exactly when
        # that is not `None` and nothing withheld it, and the binding must agree, or a subclass
        # saying "no history" would bind an undescribed name.
        # What the framework provides goes in `__builtins__` beside the allowed builtins
        # (`__history__` and `invoke` here, `_` at exec time): a variable of the same name
        # shadows it, and `del` brings it back, as for `len`.
        repl_history: list[object] | None = (
            None
            if repl_history_withheld or self.repl_history_description is None
            else []
        )
        if repl_history is not None:
            allowed_builtins["__history__"] = repl_history
        # Bound under the bare name `invoke` — there is no `jaz` namespace wrapping it (see
        # jaz._subinvoke). It sits in `__builtins__` with `__history__`, not in the exec globals,
        # which hold the host's inputs and the agent's variables. So an input named `invoke`, or
        # the agent's own `invoke = ...`, shadows the primitive rather than replacing it, and
        # `del invoke` brings it back. Chosen by the user on Claude's recommendation (2026-10-17).
        # A hook's `AddToNamespace({"invoke": ...})` shadows it too, as an assignment would, and
        # a drop removes only such a shadow: the primitive is not a key of the agent's
        # `globals()` (see `get_namespace`), and `DisableRecursion` is what withholds it. Which
        # names a host may pass stays the host's call.
        if subinvoke is not None:
            allowed_builtins["invoke"] = subinvoke

        exec_globals: dict[str, object] = {"__builtins__": allowed_builtins}
        # Always bind the inputs as REPL variables (#568); a hook that withholds them drops them.
        # `__builtins__` is refused as the AddInputs/AddToNamespace effects refuse it: binding it
        # would replace the sandbox dict. `invoke` refuses it first; this is for a caller that
        # initializes the REPL directly.
        if "__builtins__" in inputs:
            raise KeyEditDeniedError(
                "An invoke input cannot be named '__builtins__' — it holds the compiler sandbox; "
                "binding it would reopen import/attribute/builtins policy. Rename the input."
            )
        exec_globals.update(inputs)

        # Return the per-invoke state as one object; the REPL itself is untouched, so there is no
        # "which mutable attribute did we forget to copy" question to get wrong. The sandbox
        # allow-lists live only on the config REPL as frozen tuples behind read-only properties
        # (#741), so "read, never mutated, on every exec" is structural rather than a property of
        # the current call sites, and one configured REPL safely serves every invoke. The
        # native-``raise`` tagging scratch is call-local per exec (``_new_raise_tagger``), not an
        # attribute, so concurrent execs from sibling invokes on the shared template don't race
        # on it.
        return PythonREPLState(
            exec_globals=exec_globals,
            session_id=session_id,
            sandbox_builtins=allowed_builtins,
            repl_history=repl_history,
        )

    @override
    def update_repl_history(
        self,
        state: PythonREPLState,
        llm_response: LLMResponse,
        exec_result: ExecResult[Any],
    ) -> None:
        """Append one turn's :class:`~jaz.repl.REPLHistoryEntry` to ``state.repl_history`` (see
        :meth:`BaseREPL.update_repl_history`); records nothing when the invoke withheld
        ``__history__``.

        The entry records the response *text* (``llm_response.content``; an empty completion
        records ``""``), the REPL ``output``, and the recoverable exception. ``repl_output`` is
        always a ``str``: the ``output`` of a ``Continue``, and ``""`` for a terminal
        ``Return``/``Raise``, which carry none. ``repl_exception`` is the raw exception object of
        a recoverable ``Continue`` and ``None`` otherwise: a terminal ``Raise``'s exception
        travels out of the invoke, not into history.
        """
        # The object rather than a summarized string because that text is already in
        # `repl_output` (an error `Continue`'s `output` ends with the traceback's `Type: message`
        # line); see the field comment on `REPLHistoryEntry`. The whole `LLMResponse` arrives and
        # only its text is kept: a subclass wanting token counts or cost in its record overrides
        # this and reads them off the same object.
        if state.repl_history is None:
            return
        repl_exception = (
            exec_result.exception
            if isinstance(exec_result, Continue) and exec_result.exception is not None
            else None
        )
        state.repl_history.append(
            REPLHistoryEntry(
                llm_response=llm_response.content or "",
                # Only a `Continue` carries `output` (#903 dropped it from the terminal `Return` /
                # `Raise`, whose value/exception — not agent-facing text — is the payload); a
                # terminal turn therefore records `""` — see the field comment on
                # `REPLHistoryEntry` for why the empty string rather than `None`.
                repl_output=exec_result.output
                if isinstance(exec_result, Continue)
                else "",
                repl_exception=repl_exception,
            )
        )

    @override
    def add_to_namespace(
        self, state: PythonREPLState, entries: dict[str, object]
    ) -> None:
        """Bind ``entries`` into the REPL namespace **raw** — the per-turn, namespace-level
        write applied at ``REPLExecEnter``.

        A *variable* is not an *input*: an input renders a prompt block, a variable does not.
        The prompt is untouched either way.

        Naming ``__builtins__`` raises ``KeyEditDeniedError``: it holds the compiler sandbox, and is
        not a namespace entry. Binds the name and nothing else: there is no inputs snapshot to
        mirror.

        Overwriting a name that is **already bound** in the REPL namespace (a top-level input, the
        agent's own variable, or a prior binding) raises ``REPLInputConflictError`` rather than
        silently clobbering it — symmetric with ``AddInputs`` refusing to shadow a caller input, so
        no hook can quietly replace a value the agent (or caller) put there. To *re-bind* a name
        (e.g. to reset it each turn), ``DropFromNamespace`` it first, then ``AddToNamespace`` — the drop
        makes the intent-to-replace explicit, and the agent loop applies drops before adds so the
        name is free by the time this runs. The drop needs ``allow_missing=True`` if the name may
        not be bound yet. (Names living in ``__builtins__`` — the allowed
        builtins and framework magics — are not checked here; this guards the *variable* namespace.)

        Args:
            entries: Dictionary of name -> value to bind (verbatim).
        """
        # Why `__builtins__` is refused: #688/#690 moved the compiler sandbox into that dict, so
        # binding over it would hand the agent the sandbox itself. Composition (``_coalesce_add``)
        # already raises on it, so the effect path never gets here; the REPL raises too rather
        # than skipping because a silent skip on a direct call would hide the caller's bug the
        # same way a silent skip of a policy-name drop would — the two refusals are one rule:
        # sandbox policy is not editable through the namespace, and saying so is never quiet.
        # Checked before the loop so a refused call leaves the namespace untouched.
        if "__builtins__" in entries:
            raise KeyEditDeniedError(
                "AddToNamespace cannot bind '__builtins__' — it holds the compiler sandbox; "
                "binding it would reopen import/attribute/builtins policy."
            )
        for name, value in entries.items():
            if name in state.exec_globals:
                raise REPLInputConflictError(
                    f"AddToNamespace target {name!r} is already bound in the REPL namespace; a "
                    "namespace add must not silently overwrite an existing variable (a top-level "
                    "input, a prior binding, or the agent's own). To re-bind, DropFromNamespace it first."
                )
            state.exec_globals[name] = value

    @override
    def get_namespace(self, state: PythonREPLState) -> Mapping[str, object]:
        """The agent-visible namespace: exactly the keys the agent's ``globals()`` holds.

        That is every variable (an invoke input, a hook's binding, the agent's own), and
        ``__builtins__``, shown as a read-only copy: it holds the allowed builtins and what the
        framework provides, such as ``__history__``, ``invoke`` and the last-expression ``_``. A
        fresh dict per call; the event it lands on freezes it. Other values are the live objects,
        never copies.
        Raises :class:`~jaz.exceptions.KeyEditDeniedError` when agent code has replaced the
        sandbox's ``__builtins__``, as ``exec`` does.
        """
        # The view is the agent's own `globals()`, so a name is bound for a hook exactly when it
        # is bound for the agent's `del`: a framework binding sits in `__builtins__` with `len`,
        # shadowed by an assignment and restored by `del`, and a drop naming one with nothing
        # shadowing it is a missing target, as `del len` is a `NameError`. Hooks still read
        # `namespace["__builtins__"]["__history__"]`, and `__builtins__` is the one key they see
        # but may not edit (`KeyEditDeniedError`). The user's call (2026-10-17). The alternatives
        # it was compared with: a view of variables only; framework bindings merged into the
        # view (the earlier layout, under which a drop could remove one the agent's `del`
        # cannot); and moving `invoke`/`__history__` into the globals behind a compiler check.
        # The sandbox dict is shown as a read-only copy, part of the same call: the event freezes
        # only the top level, so a live dict would let a hook edit the agent's allow-list, and a
        # live read-only view would change under an event a hook kept (`_` moves every turn).
        builtins = _sandbox(state)
        view = dict(state.exec_globals)
        view["__builtins__"] = MappingProxyType(dict(builtins))
        return view

    @override
    def drop_from_namespace(
        self,
        state: PythonREPLState,
        names: Iterable[str],
        allow_missing: Iterable[str] = (),
    ) -> None:
        """Remove ``names`` from the REPL namespace — the namespace counterpart of
        ``add_to_namespace``.

        A drop does what ``del name`` does in the agent's code: it unbinds a key of the agent's
        ``globals()``. Dropping a variable that shadows something in ``__builtins__`` (an allowed
        builtin such as ``len``, or what the framework provides there, such as ``__history__``
        and ``invoke``) removes only the shadow, so the original shows again. Dropping a name
        that is **not bound** raises ``MissingDropTargetError``, as ``del`` raises ``NameError``,
        including a name ``__builtins__`` provides that no variable shadows. A drop of an absent
        name is a hook bug (a typo, or a "re-drop every turn" pattern that should be
        first-turn-only), so it fails loud rather than no-op'ing. This is still order-independent
        and safe for the legitimate *double-drop* case: the framework unions the dropped names
        first, and this applies the set **once** against the pre-drop namespace, so two hooks
        dropping the same present name coalesce to a single deletion and never trip the check.
        A name in ``allow_missing`` (its ``DropFromNamespace`` opted into tolerating absence, e.g.
        a first-turn input drop when a ``DropInputs`` may have already un-passed the input) is
        skipped when absent instead of raising.

        Naming ``__builtins__`` *itself* raises ``KeyEditDeniedError``: it holds the compiler
        sandbox, and unbinding the dict would strip import/attribute/builtins policy. A refused
        call removes nothing.
        """
        # The union happens in composition, on `REPLExecContext.namespace_drops` (see the
        # dispatcher). The docstring says "the framework" instead: both names are private, so a
        # cross-reference to them renders as a dead code span on the published page.
        # The compiler sandbox lives in `__builtins__` as of #688/#690, which is what makes
        # unbinding the dict a policy bypass rather than a mere name removal.
        builtins = _sandbox(state)
        allow_missing = set(allow_missing)
        names = set(names)
        # Composition already refuses this, so the effect path never gets here; the REPL raises
        # too because a silent skip would make the caller's bug invisible on a direct call.
        # Checked before anything is removed.
        if "__builtins__" in names:
            raise KeyEditDeniedError(
                "DropFromNamespace cannot drop '__builtins__' — it holds the compiler sandbox; "
                "dropping it would reopen import/attribute/builtins policy."
            )

        # A name is bound exactly when it is a key of the agent's `globals()`, the view
        # `get_namespace` gives hooks. What `__builtins__` holds (an allowed builtin such as
        # `len`, the import/open wrappers, and the framework's `__history__`, `invoke` and `_`)
        # is not bound until a variable shadows it, and Python's own `del len` raises the same
        # `NameError` there as for a name never assigned. So a drop naming one is a missing
        # target, and `allow_missing` skips it like any other (as when `DropInputs` already
        # un-passed an input named `open`). Treating an unshadowed allowed builtin as missing is
        # the user's call (2026-10-17), on Claude's recommendation, replacing their call of
        # 2026-10-14 to refuse it with `KeyEditDeniedError` as present-but-protected; the
        # framework's names follow from the view being `globals()` (see `get_namespace`).
        # `__builtins__` itself, which `del` does find, stays the protected case above. Checked
        # before any name is removed, so a refused call leaves the namespace untouched.
        missing = sorted(
            name
            for name in names
            if name not in state.exec_globals and name not in allow_missing
        )
        if missing:
            raise MissingDropTargetError(
                f"DropFromNamespace target(s) {missing!r} are not bound in the REPL namespace; "
                "dropping an absent name is a hook bug (drop it only when it is present, "
                "e.g. once on the first REPLExecEnter)."
                + _missing_note(missing, builtins)
            )
        # Only the globals: a shadow goes, and whatever `__builtins__` holds under it shows again.
        for name in names:
            state.exec_globals.pop(name, None)

    @contextmanager
    def _capture_print(
        self, state: PythonREPLState, string_io: StringIO
    ) -> Generator[None, None, None]:
        """Capture this exec's stdout into ``string_io``.

        Two composed layers (see repl/_stdout_proxy.py):
          * swap ``builtins["print"]`` for a closure over ``string_io`` — the fast
            lexical path, which also survives into a raw worker thread; and
          * point the ``sys.stdout`` proxy's ``_current_buffer`` at ``string_io`` for
            this context, so ``pprint`` / ``sys.stdout.write`` / prints inside imported
            functions (which reach the *stream*, not the swapped name) are captured too.
        """
        from jaz.repl._stdout_proxy import reset_current_buffer, set_current_buffer

        # Load-bearing for the sandbox: every exec passes through here before the agent's code
        # runs, and if `__builtins__` were missing Python would run that code with the full
        # builtins, while a module bound there serves its namespace as the builtins (`io`'s
        # holds the real `open`); found by running both through `exec` on Python 3.12. So
        # anything but the sandbox's own dict stops the exec, at the `_sandbox` check each
        # exec path runs before its `try`; this read goes through the same helper.
        builtins = _sandbox(state)
        if "print" in builtins:
            old_print = builtins["print"]
            old_print_exists = True
        else:
            old_print = None
            old_print_exists = False
        new_print = _make_print(string_io)
        builtins["print"] = new_print

        # Route the stdout stream (proxy) to this exec's buffer for this context.
        buffer_token = set_current_buffer(string_io)

        try:
            yield
        finally:
            reset_current_buffer(buffer_token)
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
        ``_new_raise_tagger``). Parsing is delegated to :func:`parse_or_none`, which hands back
        ``None`` on a syntax error so ``_handle_syntax_error`` can position it.
        """
        tree = parse_or_none(src)
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
                "printed output. Do not write a `return` statement after printing."
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
        # Call-local, NOT on ``self``: one config REPL is shared across every invoke at a depth,
        # so a single shared list would be raced by concurrent execs from sibling invokes (e.g.
        # ``asyncio.gather`` over multiple ``ainvoke`` under one config) — one exec's reset would
        # wipe another's tag and misclassify its finishing ``raise`` as a recoverable
        # ``Continue``. Per-exec scratch keeps the template holding no exec state.
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
                "Your `raise` statement was not executed because your code produced "
                "printed output. Do not write a `raise` statement after printing."
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

    def _repl_filename(self, state: PythonREPLState, code_id: str) -> str:
        # session_id in the filename avoids linecache collisions across sessions
        if state.session_id:
            return f"<repl_{state.session_id}_{code_id}>"
        return f"<repl_{code_id}>"

    # Pre-execution REPL-code validation moved to the ValidateREPLCode hook (#528): it
    # runs the validator at REPLExecSend and supplies a SupplyExecResult (skipping exec) on
    # rejection — the effect-path equivalent of the old _validate_input short-circuit here.

    def _resolve_timeout(self, src: str) -> float | None:
        """This exec's wall-clock bound: the code's own ``# timeout:`` pragma if it has one, else
        the configured ``exec_timeout``. With ``allow_timeout_pragma=False`` the pragma is skipped
        entirely — not even parsed, so a malformed one cannot fail code the REPL is meant to treat
        as an ordinary comment.

        Raises:
            REPLTimeoutPragmaError: The code's leading ``# timeout:`` comment is malformed.
        """
        # The pragma outranks the configured default: it is the most specific statement about
        # *this* execution, and the agent that wrote the code is the party that knows the step is
        # a ten-minute build.
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
            return self.exec_timeout
        pragma_timeout = _parse_timeout_pragma(src)
        if pragma_timeout is not None:
            return pragma_timeout
        return self.exec_timeout

    def _exec(
        self,
        state: PythonREPLState,
        src: str,
        code_id: str,
        timeout: float | None,
    ) -> ExecResult[Any]:
        # First, before compiling, as well as after the code ran: a state corrupted before this
        # call (by a loop that ignored the error, or a direct edit of `exec_globals`) must not
        # run code, and the check after the code would miss it when compilation fails, since
        # `_compile_native`'s early return skips everything below.
        _sandbox(state)
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
                    state.exec_globals,
                    timeout,
                    memory_limit_bytes=self.exec_memory_limit,
                    owner_id=timeout_owner,
                )
        except _JazReturnSignal as sig:
            result = self._handle_native_return(sig.value, string_io)
        except BaseException as e:
            # A tagged exception is an agent-authored top-level ``raise`` that escaped its own
            # try/except → finish (Raise). Anything else is incidental/fatal/foreign and goes
            # to _native_exception_result (recoverable Continue, or propagated when fatal).
            if any(e is t for t in tagged):
                result = self._handle_native_raise(e, string_io)
            else:
                result = self._native_exception_result(e, string_io, timeout_owner)
        else:
            result = Continue(output=string_io.getvalue())
        # After the code ran, whatever it did (an untagged fatal error has re-raised above; the
        # agent's finishing `return`/`raise` is a result, discarded here like any other): see
        # `_sandbox` for why the exec that replaced the sandbox is the one that fails.
        _sandbox(state)
        return result

    async def _aexec(
        self,
        state: PythonREPLState,
        src: str,
        code_id: str,
        timeout: float | None,
    ) -> ExecResult[Any]:
        """Async exec: compiles with top-level-await enabled and runs the code as a
        coroutine, so the agent can ``await`` (e.g. ``x = await ainvoke()`` then native
        ``return x``). Dispatched from ``aexec`` (which owns the shared timeout
        setup). The whole submission runs on the async path; there is no separate command
        syntax to route synchronously.
        """
        # The async exec path is #567.
        _sandbox(state)  # as in `_exec`
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
                    state.exec_globals,
                    timeout,
                    memory_limit_bytes=self.exec_memory_limit,
                    owner_id=timeout_owner,
                )
        except _JazReturnSignal as sig:
            result = self._handle_native_return(sig.value, string_io)
        except BaseException as e:
            if any(e is t for t in tagged):
                result = self._handle_native_raise(e, string_io)
            else:
                result = self._native_exception_result(e, string_io, timeout_owner)
        else:
            result = Continue(output=string_io.getvalue())
        _sandbox(state)  # as in `_exec`
        return result

    @override
    def exec(
        self,
        state: PythonREPLState,
        src: str,
        code_id: str,
    ) -> ExecResult[Any]:
        # TODO: Fix static types
        """Executes REPL code and mutates ``state`` in place.

        Runs the Python code in the REPL. A top-level ``return`` statement exits the REPL
        with that return value; a top-level ``raise`` (when ``allow_raise``) exits with that
        exception. If an error occurs during execution, the error is captured and returned
        back into the REPL as a recoverable :class:`~jaz.repl.Continue` — except a *fatal*
        exception (:class:`~jaz.exceptions.FatalError`, non-``Exception`` signals) or an
        enclosing exec's timeout/memory breach, which **re-raises** out of this call so it
        propagates out of the invoke. A :class:`~jaz.exceptions.KeyEditDeniedError` re-raises
        too, before anything is compiled when an earlier exec replaced the sandbox's
        ``__builtins__`` (through ``globals()``) and at the end of the exec that does so: that
        state cannot run code again, so the invoke fails rather than being shown the error.

        The code sets its own wall-clock bound with a ``# timeout: <seconds>`` comment among the
        lines that open it (before any real code), which overrides the configured
        ``exec_timeout`` for this call only. A malformed value there is reported
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
            timeout = self._resolve_timeout(src)
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
        # `__history__` is not written here: the loop calls `update_repl_history` once per
        # iteration with the *finalized* result (after exit-time effect composition), so a
        # result-override hook (e.g. BudgetForcing) is reflected in the record (#448).
        return self._exec(
            state,
            src,
            code_id,
            timeout,
        )

    @override
    async def aexec(
        self,
        state: PythonREPLState,
        src: str,
        code_id: str,
    ) -> ExecResult[Any]:
        """Async ``exec``: the code runs as a coroutine on the event loop, so the agent
        can use top-level ``await`` (e.g. ``result = await ainvoke(...)``). The whole
        submission takes the async path (there is no separate command syntax to route
        synchronously); native ``return`` / ``raise`` still finish the REPL.

        The ``# timeout: <seconds>`` pragma applies here exactly as it does to ``exec``.

        Like sync ``exec``, this does not write ``__history__`` — the agent loop calls
        :meth:`update_repl_history` with the turn's final result — and like it, a replaced
        sandbox ``__builtins__`` re-raises as :class:`~jaz.exceptions.KeyEditDeniedError`.
        """
        # Top-level `await` support is #567. The one-writer rule for `__history__` is
        # #448/#527 — the record follows the final result so an exec-time write cannot desync it.
        try:
            timeout = self._resolve_timeout(src)
        except REPLTimeoutPragmaError as e:
            return Continue(output=summarize_exception(e), exception=e)  # see `exec`
        return await self._aexec(state, src, code_id, timeout)
