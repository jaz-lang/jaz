from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .._glob_allowlist import compile_allowlist, is_allowed


class _ImportPolicy(Protocol):
    @property
    def allowed_imports(self) -> Sequence[str]: ...


def make_secure_import(policy: _ImportPolicy) -> Callable[..., Any]:
    """Create an ``__import__`` wrapper bound to a permission policy (runtime import backstop).

    Runtime complement to the compiler's static ``ImportPolicyChecker`` (#470). The static
    checker (in ``secure_compile``) sees ``import`` / ``from`` *statements* and **reserves the
    ``__builtins__`` name** — any ``ast.Name`` reference (e.g. ``__builtins__["__import__"]``, an
    alias, or a rebind) is rejected at compile time, so an agent cannot reach this wrapper through
    the bare builtins dict. What it does *not* see is ``__import__(...)`` *calls* (neither a
    string-*literal* ``__import__("x")`` nor a *computed* ``__import__("o" + "s")`` / ``__import__(var)``;
    the literal-call detection was dropped in #742 as redundant with this wrapper — a follow-up
    reserves the ``__import__`` name statically too), the ``IMPORT_NAME`` calls that ``import``
    *statements* compile down to, ``importlib.import_module(...)`` (an attribute), and reflective
    reach to ``__builtins__`` via frame/generator introspection or ``str.format`` (which
    name-reservation cannot close — see ``ImportPolicyChecker`` and the Known limitations below).
    Because ``__import__`` is an always-present builtin (every ``import`` statement needs it —
    omitting it makes even allowed imports raise ``ImportError: __import__ not found``), wrapping it
    here — bound to the per-REPL ``allowed_imports`` policy — enforces the allow-list at the moment
    each import actually runs: the true containment point for every route that reaches the sandbox's
    ``__builtins__``.

    History: #742 dropped the static literal-``__import__("x")`` *call* detection as redundant with
    this wrapper, but that reopened a deferred-execution hole — a returned closure over
    ``__builtins__["__import__"]("time")`` (or bare ``__import__("time")``) escapes the sandbox
    unexecuted and fails only in the caller. The static layer now reserves the ``__builtins__``
    *name* (exact, no false positives), closing that gap at definition time for the bare-builtins
    route; a follow-up reserves ``__import__`` for the bare form too. This wrapper remains the
    runtime enforcement point it always was.

    Semantics deliberately mirror ``ImportPolicyChecker`` so static and runtime never disagree:
    the *root* module (``name.split(".")[0]``) must match the gitignore-style allow-list;
    ``["*"]`` allows everything; ``[]`` forbids everything; relative imports (``level > 0``) are not policed
    (they have no top-level module to allowlist — the same reason the static checker only
    inspects ``level == 0`` ``ImportFrom`` nodes). The error type and message match the static
    checker so the agent sees identical feedback whichever layer fires. The two enforcement
    points are intentionally *separate* (compile-time vs. runtime defense-in-depth), so the
    small root-check is duplicated rather than shared across the ``compiler`` / ``_secure_methods``
    boundary — a deliberate, low-churn call over a shared helper.

    Known limitations (this wrapper contains imports only *in concert with* the rest of the
    sandbox — it is not a standalone boundary):
    - ``importlib.import_module`` reaches the C import machinery without going through
      ``__import__``, so it is not caught here. That vector is only reachable if ``importlib``
      itself is allow-listed; with ``importlib`` allowed, *both* a *literal*
      ``importlib.import_module("x")`` and a *computed* ``importlib.import_module(var)`` are gaps
      — the static checker stopped inspecting ``import_module`` calls in #742, and this wrapper
      only sees ``__import__``. (Allow-listing ``importlib`` is itself an escape hatch — see the
      ``"builtins"`` / ``"importlib"`` bullet below.)
    - **The wrapper (this policy-bound ``__import__``) stays reachable — enforcement rides on it,
      not on hiding it.** ``ImportPolicyChecker`` reserves the ``__builtins__`` *name* (and bare
      ``__import__(...)`` calls still land here directly, the runtime backstop's original job), but
      the sandbox's ``__builtins__`` dict (hence this wrapper) is still reachable through routes
      that do not write the name as an ``ast.Name``: ``globals()["__builtins__"]``,
      ``locals()["__builtins__"]``, no-arg ``vars()["__builtins__"]`` (``globals``/``locals`` are
      allow-listed builtins; no-arg ``vars`` is unpoliced), and frame/generator introspection
      (``(x for x in []).gi_frame.f_builtins`` — ``gi_frame``/``f_builtins``/``f_globals`` are
      non-dunder, so the default ``allowed_attributes=["*", "!__*"]`` does not block them). All of
      these land on *this* wrapper, so the allow-list is still enforced at run time — no forbidden
      module is imported; the residual is only that a deferred closure over them can be *built*.
    - **The *real* ``__import__`` (full bypass, not the wrapper) stays reachable via the attribute
      sandbox's documented holes.** ``some_fn.__globals__["__builtins__"]["__import__"]`` needs a
      *Python* function ``some_fn`` (whose ``__globals__`` carry the real builtins) and is closed
      today only because the default ``allowed_attributes`` forbids ``__globals__`` — widening it to
      dunders silently reopens the escape. ``str.format`` is worse: ``"{0.__globals__[__builtins__]}"
      .format(fn)`` does the attribute+index traversal inside a runtime format string, invisible to
      *both* the static ``AllowedAttributesChecker`` and the runtime ``getattr`` wrapper (C-level
      getattr), so no attribute allow-list closes it. Given any Python function in scope (e.g. a
      jaz-library function) this reaches the real ``__import__``.
    - **Allow-listing ``"builtins"`` or ``"importlib"`` is an escape hatch.** Either one hands the
      agent the import machinery directly (``import builtins; builtins.__import__("os")``), so
      those are effectively "open the sandbox" allowlist entries even though they read as ordinary
      module names.
    """
    # Compile the allow-list once: the policy is a frozen dataclass, so its `allowed_imports`
    # never changes for this REPL, and this avoids recompiling the spec on every import.
    spec = compile_allowlist(policy.allowed_imports)

    def _secure_import(
        name: str,
        globals: Any = None,  # noqa: A002 — matches the builtin __import__ signature
        locals: Any = None,  # noqa: A002 — matches the builtin __import__ signature
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        # `level == 0` is an absolute import; relative imports (level > 0) have no root
        # module to allowlist and are passed through, matching the static checker.
        if level == 0 and name:
            root = name.split(".")[0]
            if not is_allowed(spec, root):
                raise ImportError(
                    f"Import of module '{root}' is forbidden in the REPL."
                )
        # Bare `__import__` is the *real* builtin, not this wrapper: `_secure_import` is
        # installed as the REPL's `__import__`, but its own globals are this module's (whose
        # builtins are unsandboxed), so the bare name resolves to genuine import — no recursion.
        return __import__(name, globals, locals, fromlist, level)

    return _secure_import
