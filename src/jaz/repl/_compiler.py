import ast
import linecache
import types
import typing
import warnings
from collections.abc import Sequence
from typing import Literal, cast

from ._cookies import CookieJar
from ._exec_guards import register_code
from ._glob_allowlist import compile_allowlist, is_allowed

ALLOWED_BUILTIN_TYPES: frozenset[type] = frozenset(
    {
        int,
        str,
        float,
        bool,
        list,
        dict,
        tuple,
        type(None),
    }
)


def validate_builtin_type_annotation(
    annotation: object, *, _allow_nonetype: bool = False
) -> None:
    """Recursively validate that a runtime type annotation uses only built-in types.

    ``None`` (the value) is accepted as a type annotation.  ``NoneType``
    (i.e. ``type(None)``) is only allowed as a member of a union
    (e.g. ``str | None``), not as a top-level annotation.  We can't reject
    ``NoneType`` inside unions because Python represents ``str | None`` as
    ``UnionType(str, NoneType)`` at runtime.

    Raises:
        TypeError: If the annotation contains non-builtin types.
    """
    if annotation is None:
        return

    if annotation is type(None):
        if _allow_nonetype:
            return
        raise TypeError(
            "Use None instead of NoneType for type annotations. "
            "For optional types, use e.g. str | None."
        )

    if isinstance(annotation, type):
        if annotation not in ALLOWED_BUILTIN_TYPES:
            raise TypeError(f"Only built-in types are supported. Got: {annotation}")
        return

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Handle Union types (str | int, Optional[str], Union[str, int])
    if origin is types.UnionType or origin is typing.Union:
        for arg in args:
            validate_builtin_type_annotation(arg, _allow_nonetype=True)
        return

    # Handle generic aliases (list[str], dict[str, int])
    if origin is not None:
        if origin not in ALLOWED_BUILTIN_TYPES:
            raise TypeError(f"Only built-in types are supported. Got origin: {origin}")
        for arg in args:
            if arg is Ellipsis:
                continue
            validate_builtin_type_annotation(arg, _allow_nonetype=True)
        return

    raise TypeError(f"Only built-in types are supported. Got: {annotation}")


class PrintLastExprTransformer(ast.NodeTransformer):
    """Auto-print a non-None trailing expression and bind it to the standard REPL ``_``.

    Unconditional: a trailing expression's value is always printed and stored in ``_``, a plain
    Python-REPL convenience.
    """

    # Unconditional as of #568, which removed the ``repl_history`` flag / ``store_last_expr``
    # param that used to gate it, along with the per-turn ``_N`` variables (``_7``, …) — YAGNI,
    # no known use case. Those variables were the only thing that ever needed the REPL code id,
    # so ``secure_compile``'s ``code_id`` parameter has been dead ever since; this change drops it.

    def visit_Module(self, node: ast.Module):
        if len(node.body) == 0 or not isinstance(node.body[-1], ast.Expr):
            return node
        last_expr_value = node.body[-1].value
        # ```
        # x = EXPR
        # if x is not None:
        #     print(repr(x))
        #     __builtins__['_'] = x
        # ```
        # `_` lives in `__builtins__` (not `globals()`), matching the real Python REPL, so
        # that `_ = ...; del _` restores access to the auto-bound value.
        new_last_expr = ast.parse(
            "(lambda x: "
            "    None if x is None else "
            "        print(repr(x)) or "
            "        __builtins__.__setitem__('_', x)"
            ")(EXPR)"
        ).body[0]
        assert isinstance(new_last_expr, ast.Expr)
        new_last_expr_value = new_last_expr.value
        assert isinstance(new_last_expr_value, ast.Call)
        new_last_expr_value.args[0] = last_expr_value
        node.body[-1].value = new_last_expr_value
        return node


# --------------------------------------------------------------------------------------
# Sandbox policy checkers (allowed_imports / allowed_attributes) — uniform gitignore-glob
# allow-lists (see _glob_allowlist), matching the file-path allow-lists in secure_open.
#
# The name surface is no longer policed here by a denylist: the former ForbiddenNamesChecker was
# replaced by the builtins allowlist installed as ``__builtins__`` (#804, see permissions.py) —
# an omitted builtin is simply unbound (``NameError``) rather than statically rejected.
#
# Moved back into the compiler from the former ImportPolicyHook / AllowedAttributesHook (#688):
# they are AST restrictions like the rest of secure_compile,
# so one parse enforces the whole sandbox, and "how locked-down is this REPL?" is answered by
# the ``PythonREPL`` allow-list settings rather than by which hooks happen to be
# active. Enforcement is static /
# whole-source (a violation anywhere in the tree is policed even if that line never runs),
# and — since these raise out of secure_compile, caught by the REPL's exec handler — the
# violation surfaces as a recoverable error the agent can react to, exactly like the former
# hooks' ModifyExecResult. The compiler enforces the sandbox on every exec, but it is NOT
# un-strippable by the agent: the ``repl`` group is an ordinary config setting, so a host that
# hands the agent a config-override surface (e.g. passing ``jaz.ConfigOverride`` in as an input)
# opts into letting the agent reconfigure this sandbox for its sub-invokes.
#
# Why the un-strippability was traded away (the removed protected-config-keys subsystem, an
# executive call — recorded here per CLAUDE.md rather than only in the PR): the subsystem was
# INERT in the default configuration (an agent can't construct a ``ConfigOverride`` unless a host
# opts in — its REPL binds no ``jaz`` and ``import jaz`` is blocked), and even
# when live its guarantee against the *ambient* path (a propagating ``with ConfigOverride`` / a
# ``configure()`` call) never rested on the key check at all — it rested on those primitives being
# unreachable by the agent. So the key check only ever guarded the ConfigOverride-*argument* path,
# a narrow slice, at the cost of a whole subsystem (a per-agent restricted ``ConfigOverride``
# subclass + a server-side arg gate). It was removed in favor of that same unreachability posture.
# The clean way to restore an un-strippable sandbox would be a distinct agent-facing config
# surface, where "the agent can't name the ``repl`` group" is structural, not a key check; no
# issue tracks it.
# --------------------------------------------------------------------------------------


# TODO(#724): these single-pass checkers override visit() to run one ast.walk rather than
# using visit_* dispatch, so the NodeVisitor wrapper buys little; #724 tracks converting
# them to plain functions (PrintLastExprTransformer stays a class — it needs NodeTransformer).
# Names `ReservedNamesChecker` refuses in agent code, read or bound, in any scope. Legitimate
# `import`/`from` statements compile to IMPORT_NAME and never emit an `ast.Name` load of
# `__builtins__`, so reserving it has no false positives.
_RESERVED_NAMES: frozenset[str] = frozenset({"__builtins__"})


def string_bound_names(node: ast.AST) -> list[str]:
    """The names ``node`` binds through a plain string field rather than an ``ast.Name``.

    An ``ast.Name`` in a ``Store`` or ``Del`` context covers assignment, ``del``, ``for``/``with``
    targets, the walrus and ``global``-declared rebinding; these are the other binding forms:
    an import's name or alias, a ``def``/``class`` name, ``except ... as``, a ``match`` capture,
    a parameter, and a type parameter. A ``global``/``nonlocal`` declaration is not one: it
    binds nothing itself, and every rebinding it enables is an ``ast.Name`` or one of these forms.
    """
    # Each of these binds a name Python resolves like any variable, so a check that reads only
    # `ast.Name` lets them through: `def __builtins__(): ...` or `except E as __builtins__:`
    # replaced the sandbox's builtins dict and left every later exec failing, and
    # `from re import __builtins__` gave a function defined after it the real builtins (every
    # pure-Python module carries them under that name), so the real `open`, `eval` and an
    # `__import__` the allow-list never sees, with `re` the only allowed module (found by running
    # each form through `PythonREPL` on Python 3.12). The aliased `import io as __builtins__`
    # reached only what allowing `io` grants anyway (`io.open`).
    match node:
        case ast.Import(names=aliases):
            return [a.asname or a.name.split(".")[0] for a in aliases]
        case ast.ImportFrom(names=aliases):
            return [a.asname or a.name for a in aliases if a.name != "*"]
        case (
            ast.FunctionDef(name=name)
            | ast.AsyncFunctionDef(name=name)
            | ast.ClassDef(name=name)
        ):
            return [name]
        case ast.ExceptHandler(name=str() as name):
            return [name]
        case ast.MatchAs(name=str() as name) | ast.MatchStar(name=str() as name):
            return [name]
        case ast.MatchMapping(rest=str() as name):
            return [name]
        case ast.arg(arg=name):
            return [name]
        # `global X` / `nonlocal X` are deliberately absent. A declaration binds nothing: it only
        # makes a later binding, `del` or read of `X` in that scope act on the outer `X`, and each
        # of those is either an `ast.Name`, which `ReservedNamesChecker` refuses in any scope, or
        # one of the forms listed here (`import io as X`, `def X`, `except E as X` under a `global
        # X` rebind the module global with no `ast.Name`). Listing the declaration would refuse a
        # statement that can do no harm by itself, which is stricter than CPython, which accepts
        # `global __debug__` and refuses only the binding (the user's call, 2026-10-18; verified by
        # compiling the declaration with and without each binding form). `nonlocal X` additionally
        # needs a binding of `X` in an enclosing function, and every form of that binding is
        # refused either here or as an `ast.Name`.
        case (
            ast.TypeVar(name=name)
            | ast.ParamSpec(name=name)
            | ast.TypeVarTuple(name=name)
        ):
            return [name]
    return []


class ReservedNamesChecker(ast.NodeVisitor):
    """Reject every use of the name ``__builtins__`` in agent code, read or bound, in any scope.

    That dict holds the sandbox's whole policy: the allowed builtins and the wrappers that enforce
    the import and file allow-lists. Reading it reaches those wrappers directly. Binding it at the
    top level, in any form (an assignment, ``import x as __builtins__``, ``from x import
    __builtins__``, a ``def``, ``except ... as``, a ``match`` capture; see ``string_bound_names``)
    replaces the policy for code that runs afterwards: ``from x import __builtins__`` binds the
    real builtins, since every pure-Python module carries them under that name, so with any one
    such module allowed the agent had the real ``open`` and an ``__import__`` the allow-list never
    sees. A local binding (a
    parameter) is harmless and refused anyway: no legitimate code names anything ``__builtins__``.
    So the name is refused outright, **independent of every allow-list**. The reservation was
    first motivated by the import route, which the paragraphs below record; the policy it
    protects is wider than imports, which is why it is its own checker rather than part of
    ``ImportPolicyChecker``.

    Refusing the name has no false positives: a real ``import x`` / ``from x import y`` compiles
    to an ``IMPORT_NAME`` opcode and emits no ``ast.Name`` of ``__builtins__``, so legitimate
    imports are untouched, and ``.import_module()`` is an ``ast.Attribute``, not a name.
    """

    # ``SyntaxError``, as CPython raises for its own reserved name: every binding form of
    # ``__debug__`` (``__debug__ = 1``, ``def f(__debug__)``, ``import os as __debug__``,
    # ``del __debug__``) is "SyntaxError: cannot assign to __debug__" at compile time, which is
    # this checker's situation exactly. It is also what the sibling ``AllowedAttributesChecker``
    # raises, and the REPL turns both into the same recoverable feedback for the agent. It was
    # ``ImportError`` while the reservation lived in ``ImportPolicyChecker``, which was history,
    # not a reason. The user's call (2026-10-18), with a jaz policy class a host could catch
    # uniformly left to the wider exception-taxonomy decision deferred until this stack merges.
    #
    # The rationale and history, kept beside the code rather than in the shipped docstring:
    #
    # Why static-reject the name rather than lean solely on the runtime ``make_secure_import``
    # wrapper: the wrapper fires only when an import *executes*, so a function that closes over
    # ``__builtins__["__import__"]("time")`` can be **defined and returned** from an invoke without
    # ever tripping it — the forbidden-intent object escapes the sandbox as a first-class value and
    # only fails when the *caller* runs it. Rejecting the name at *definition* time closes that
    # deferred-execution gap for the bare-``__builtins__`` route.
    #
    # EXECUTIVE CALL (user): reserve the ``__builtins__`` name as a self-contained static defense —
    # it closes the obvious *bare-name* ``__builtins__["__import__"](...)`` route to the wrapper. A
    # follow-up reserves the ``__import__`` name here too (closing the same deferred-execution gap
    # for the bare ``__import__("time")`` form, which this change leaves to the runtime wrapper).
    # Both are **defense-in-depth, not a boundary**: name-reservation cannot close the reflective
    # routes that reach the sandbox ``__builtins__`` without ever writing the name as an
    # ``ast.Name`` — all confirmed to reach the wrapper under the default policy (they land on the
    # sandbox's ``__builtins__``, so the wrapper still enforces the allow-list — see the containment
    # note at the end of this comment):
    #   * ``globals()["__builtins__"]["__import__"]("x")`` — ``globals`` is an allow-listed builtin;
    # ``"__builtins__"``/``"__import__"`` are string keys, not ``ast.Name`` loads. (The builtins-
    # allowlist design doc keeps ``globals``/``locals`` arguing
    # ``globals()["__builtins__"]["eval"]`` "closes itself" because ``eval`` is not in the dict —
    # true for ``eval``/``exec``/``compile``,
    #     but ``__import__`` *must* be in the dict, so the import check is bypassed here.)
    #   * ``locals()["__builtins__"][...]`` — at REPL top level ``locals()`` is the same dict.
    #   * ``vars()["__builtins__"][...]`` — no-arg ``vars()`` returns the caller namespace unpoliced
    #     (``vars(obj)`` *is* policed by the secure wrapper).
    #   * frame/generator introspection — ``(x for x in
    #     []).gi_frame.f_builtins["__import__"]("x")``.
    #     Closed by the shipped default: ``gi_frame``/``f_builtins``/``f_globals`` are
    #     non-dunder and so were allowed by the bare ``["*", "!__*"]``, but
    #     ``DEFAULT_ALLOWED_ATTRIBUTES`` now denies them by name, and the computed-name route
    #     (``getattr(gen, "gi_" + "frame")``) is refused by the runtime backstop. Reopens if a host
    #     widens the allow-list.
    #   * ``str.format`` — ``"{0.__globals__[__builtins__]}".format(fn)`` does attribute+index
    #     traversal inside a runtime format string, invisible to both the static
    #     ``AllowedAttributesChecker`` and the runtime ``getattr`` wrapper (C-level getattr). Given
    #     any *Python* function in scope (e.g. a jaz-library function, whose ``__globals__`` are the
    #     real builtins) this reaches the **real** ``__import__`` — a full policy bypass, not just
    #     the wrapper — and no allow-list can close it.
    #
    # Containment note: fully severing reflective reach to ``__builtins__`` by name/attr bans is
    # unachievable (it is threaded into every frame/function/module, and ``__import__`` must live
    # there for imports to work at all), so this checker deliberately does not attempt it; the
    # import allow-list is enforced at run time by the wrapper, as ``ImportPolicyChecker`` records.

    def visit(self, node: ast.AST) -> None:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in _RESERVED_NAMES:
                raise SyntaxError(
                    f"Use of the name '{n.id}' is forbidden in the REPL "
                    "(it holds the sandbox's policy); for imports, use a plain 'import' statement."
                )
            reserved = sorted(set(string_bound_names(n)) & _RESERVED_NAMES)
            if reserved:
                # At the top level, binding the name in any form replaces the sandbox's
                # builtins dict, and a function defined afterwards in the same exec resolves
                # its builtins through the replacement. This covers statements only: a runtime
                # write through `globals()` is caught before the next exec by the REPL's
                # identity check (`_python_repl._sandbox`). A local binding (a parameter) is
                # harmless but refused too: no legitimate code names anything `__builtins__`,
                # and refusing every scope needs no scope analysis (an implementation choice,
                # not decided by the user).
                raise SyntaxError(
                    f"Binding the '{reserved[0]}' name is forbidden in the REPL "
                    "(it holds the sandbox's policy); choose another name."
                )


class ImportPolicyChecker(ast.NodeVisitor):
    """Reject imports of root modules not in the ``allowed_imports`` allow-list.

    ``allowed_imports`` is a gitignore-style glob allow-list matched against each import's **root**
    module name (``import concurrent.futures`` checks only ``concurrent`` — root-scoped, as before).
    An exact name like ``"math"`` is a glob with no wildcards, so existing allow-lists are
    unchanged; ``["*"]`` allows every import; ``[]`` forbids all. Covers ``import x.y`` and
    ``from x.y import z`` statements (absolute only — relative imports have no top-level module
    to police).

    The containment property for imports — *no forbidden module is ever imported* — is upheld at
    run time by the wrapper, which enforces the allow-list on every route that lands on the
    sandbox's ``__builtins__`` (see ``make_secure_import``'s
    known limitations for the widened-attrs / foreign-frame / ``str.format`` cases that reach the
    *real* ``__import__``). Bare ``__import__(...)`` calls (literal or computed) are likewise not yet
    checked here — the wrapper enforces them at runtime until the ``__import__``-name reservation
    follow-up lands. ``importlib.import_module(...)`` is not checked here either (an ``ast.Attribute``,
    only reachable when a host allow-lists ``importlib``, itself an escape hatch); under a permissive
    allow-list a host needing a computed dynamic import can still reach it that way.
    """

    # The static/runtime split is #470's defence-in-depth: this checker sees statements at compile
    # time, the policy-bound ``__import__`` wrapper catches executed imports. The frame-surface
    # closure named in ``ReservedNamesChecker``'s docstring is the shipped default from #827.

    def __init__(self, allowed_imports: Sequence[str]) -> None:
        self._spec = compile_allowlist(allowed_imports)

    def visit(self, node: ast.AST) -> None:
        roots: set[str] = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                for alias in n.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                if n.level == 0 and n.module:
                    roots.add(n.module.split(".")[0])
        disallowed = sorted(r for r in roots if not is_allowed(self._spec, r))
        if disallowed:
            modules = ", ".join(f"'{m}'" for m in disallowed)
            raise ImportError(f"Import of module {modules} is forbidden in the REPL.")


class AllowedAttributesChecker(ast.NodeVisitor):
    """Reject any ``ast.Attribute`` access whose name is not in the ``allowed_attributes`` allow-list.

    ``allowed_attributes`` is a gitignore-style glob allow-list matched against the accessed
    attribute name. The secure-by-default value is ``DEFAULT_ALLOWED_ATTRIBUTES`` — allow every
    attribute *except* double-underscore-prefixed ones (gitignore negation) and the frame-bearing
    interpreter surface. ``["*"]`` allows everything (incl. dunders);
    ``[]`` forbids all attribute access. This is the allow-only counterpart of the former
    ``ForbiddenAttributesChecker`` — the whole sandbox is now allow-lists (imports/attributes/files),
    no denylists.

    ``allowed_write_attributes`` is the same kind of list applied to *binding* contexts
    (``obj.x = ...``, ``del obj.x``) instead of reads; it defaults to ``allowed_attributes``, i.e. no
    split unless a caller asks for one."""

    # The two lists exist so the shipped default can re-admit `__init__`/`__name__`/`__qualname__`
    # for reads (`super().__init__(...)` is ordinary Python) without making them settable — see
    # DEFAULT_ALLOWED_WRITE_ATTRIBUTES for why the write side is worth denying separately. `n.ctx`
    # was previously ignored entirely, which is what made the read-only audit of those three names
    # wrong: the same spec drove `obj.attr = ...`.

    def __init__(
        self,
        allowed_attributes: Sequence[str],
        allowed_write_attributes: Sequence[str] | None = None,
    ) -> None:
        self._spec = compile_allowlist(allowed_attributes)
        self._write_spec = (
            self._spec
            if allowed_write_attributes is None
            else compile_allowlist(allowed_write_attributes)
        )

    def _rejected(self, node: ast.Attribute) -> bool:
        # ast.Store / ast.Del are the binding contexts; everything else (Load) is a read.
        spec = (
            self._write_spec
            if isinstance(node.ctx, ast.Store | ast.Del)
            else self._spec
        )
        return not is_allowed(spec, node.attr)

    def visit(self, node: ast.AST) -> None:
        found = sorted(
            {
                n.attr
                for n in ast.walk(node)
                if isinstance(n, ast.Attribute) and self._rejected(n)
            }
        )
        if found:
            names = ", ".join(repr(n) for n in found)
            raise SyntaxError(
                f"Access to attribute {names} is not allowed in the REPL."
            )


def apply_policies[TreeT: ast.AST](
    policies: list[ast.NodeVisitor], tree: TreeT
) -> TreeT:
    for policy in policies:
        policy.visit(tree)
    ast.fix_missing_locations(tree)
    return tree


type CompileMode = Literal["exec", "eval", "single"]


def secure_compile(
    src: str,
    *,
    filename: str = "<repl>",
    mode: CompileMode = "exec",
    allow_top_level_await: bool = False,
    allowed_imports: Sequence[str] = ("*",),
    allowed_attributes: Sequence[str] = ("*",),
    allowed_write_attributes: Sequence[str] | None = None,
    cookie_jar: CookieJar | None = None,
    tree: ast.AST | None = None,
):
    """Compile validated REPL source to a code object.

    ``allow_top_level_await`` adds ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` to the final compile so
    a top-level ``await`` is legal (the code object gains ``CO_COROUTINE``) — the async
    ``aexec``/``aeval`` path. The flag only affects the final ``compile()``; the *same*
    policy-validated AST is compiled, so every restriction still holds.

    ``cookie_jar`` (passed by the REPL, ``None`` for direct non-agent callers) carries the
    return-signal / tag-raise / exception references the REPL's transforms embedded as co_consts
    *cookies* (see ._cookies); when given, its placeholders are substituted for the real objects in
    the compiled code object. Direct callers pass no jar and get a plain compile.

    ``allowed_imports`` / ``allowed_attributes`` default to ``("*",)`` (allow-all), NOT the
    fail-closed ``[]``. This is deliberate layering, not a fail-open: ``secure_compile`` is a
    general compile primitive, and a deny-all default would lock out every direct caller (tests,
    tooling) that isn't sandboxing. The fail-closed deny-all default lives one layer up, at
    ``PythonREPL.__init__`` (an unset ``allowed_imports`` → ``[]``, an unset
    ``allowed_attributes`` → the secure default), and every REPL call site passes those stored values — so
    agent code is never compiled under the permissive default. The permissiveness is confined to
    direct, non-agent callers.

    ``allowed_write_attributes`` policies binding contexts (``obj.x = ...``, ``del obj.x``)
    separately from reads; ``None`` (the default) applies ``allowed_attributes`` to both, so a
    caller that does not care about the distinction sees no change.
    """
    # The async path compiles through here too (#567): the flag only changes what the compiler
    # accepts, and the same policy-validated AST is compiled either way.
    # Keyword-only past ``src``. The dropped ``code_id`` used to sit at position 2, so a stale
    # positional caller would otherwise bind its id string to ``filename`` — which feeds
    # ``ast.parse``, the ``linecache`` key, and ``compile()``, so the damage is a wrong filename
    # in every traceback rather than an error. A bare ``*`` makes that a ``TypeError`` at the
    # call instead. Free: every call site already passes these by keyword.
    if tree is None:
        tree = ast.parse(src, filename=filename, mode=mode)
    match mode:
        case "exec":
            tree = cast(ast.Module, tree)
        case "eval":
            tree = cast(ast.Expression, tree)
        case "single":
            tree = cast(ast.Interactive, tree)
        case _:
            raise ValueError(f"Unsupported compile mode {mode!r}.")

    policies: list[ast.NodeVisitor] = []
    # NOTE: secure_compile no longer applies feature-specific compile policies for the Python REPL
    # (#568). Both the ``restrict_invoke_args`` and ``restrict_exec`` (tool-call-only) policies were
    # removed: the former left core for a host hook, taking its AST pass and JSON-literal helpers with
    # it (#905), and the latter was dropped outright as a dead feature no config ever enabled. Only
    # the sandbox policies below remain.
    # Sandbox policies (#688), now uniform allow-lists (imports/attributes/files) — no
    # denylists, no "unrestricted" state. `["*"]` ⇒ allow all; `[]` ⇒ forbid all. Always run
    # (an empty allow-list is meaningful: deny-all); they precede PrintLastExprTransformer.
    policies.append(ReservedNamesChecker())
    policies.append(ImportPolicyChecker(allowed_imports))
    policies.append(
        AllowedAttributesChecker(allowed_attributes, allowed_write_attributes)
    )
    # The native-return/raise sentinels are no longer protected from a bare `except:` by a compile
    # *ban* (the old RejectBareExceptChecker / reject_bare_except). Instead the REPL's
    # _ExceptGuardTransformer makes bare `except:` safe and force-guards the finish sentinel, using
    # co_consts cookies substituted below — see python_repl and ._cookies.
    if mode == "exec":
        policies.append(PrintLastExprTransformer())

    # 1) enforce rules and transform calls
    tree = apply_policies(policies, tree)

    # 2) Add source to linecache so tracebacks can display source code
    # Split source into lines, preserving newlines for linecache
    lines = [line + "\n" for line in src.splitlines()]
    if lines and not src.endswith("\n"):
        # Remove the trailing newline from the last line if original didn't have it
        lines[-1] = lines[-1].rstrip("\n")
    # linecache.cache format: (size, mtime, lines, fullname)
    linecache.cache[filename] = (len(src), None, lines, filename)

    # 3) normal CPython compile (PyCF_ALLOW_TOP_LEVEL_AWAIT only when the async path
    # asks for it, so top-level `await` compiles to a CO_COROUTINE code object).
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT if allow_top_level_await else 0
    with warnings.catch_warnings():
        # The return/raise rewrite emits `raise <cookie-constant>(value)`; the compiler flags a
        # call on a (currently bytes) constant with SyntaxWarning "'bytes' object is not callable".
        # The constant becomes the real class/hook at substitution below, so the call is valid.
        # Match the exact bytes message (not a broad `.*not callable.*`) so agent-authored
        # not-callable warnings on other literal types still surface; a genuine `b'...'()` in agent
        # code is the only residual silenced case, and it still raises TypeError at runtime.
        warnings.filterwarnings(
            "ignore",
            message="'bytes' object is not callable",
            category=SyntaxWarning,
        )
        code = compile(tree, filename, mode=mode, flags=flags)

    # 3b) Swap cookie placeholders in co_consts for the real protected objects (return sentinel,
    # tag-raise hook, Exception) before the code is registered or run. No-op without a jar.
    if cookie_jar is not None:
        code = cookie_jar.substitute(code)

    # 4) register the code-object tree with the timeout deadline registry so
    # LINE events fire in agent code (on any thread it eventually runs on)
    register_code(code)

    return code
