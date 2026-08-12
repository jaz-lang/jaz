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

    Unconditional now (#568): a trailing expression's value is always printed and stored in
    ``_`` — a plain Python-REPL convenience, no longer gated by the removed ``repl_history``
    flag / ``store_last_expr`` param. The former per-input ``_N`` variables (``_7``, …) were
    removed (YAGNI — no known use case), so this no longer needs the REPL input id.
    """

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
# repl_configs rather than by which hooks happen to be active. Enforcement is static /
# whole-source (a violation anywhere in the tree is policed even if that line never runs),
# and — since these raise out of secure_compile, caught by the REPL's exec handler — the
# violation surfaces as a recoverable error the agent can react to, exactly like the former
# hooks' ModifyResult. The compiler enforces the sandbox on every exec, but it is NOT
# un-strippable by the agent: ``repl_configs`` is an ordinary config key, so a host that hands the
# agent a config-override surface (e.g. passing ``jaz.ConfigOverride`` in as an input)
# opts into letting the agent reconfigure this sandbox for its sub-invokes.
#
# Why the un-strippability was traded away (the removed protected-config-keys subsystem, an
# executive call — recorded here per CLAUDE.md rather than only in the PR): the subsystem was
# INERT in the default configuration (an agent can't construct a ``ConfigOverride`` unless a host
# opts in — it isn't in the agent's ``jaz.*`` namespace and ``import jaz`` is blocked), and even
# when live its guarantee against the *ambient* path (a propagating ``with ConfigOverride`` / a
# ``configure()`` call) never rested on the key check at all — it rested on those primitives being
# unreachable by the agent. So the key check only ever guarded the ConfigOverride-*argument* path,
# a narrow slice, at the cost of a whole subsystem (a per-agent restricted ``ConfigOverride``
# subclass + a server-side arg gate). It was removed in favor of that same unreachability posture.
# The clean way to restore an un-strippable sandbox is a distinct agent-facing config surface
# (TODO(#722)) — where "the agent can't name ``repl_configs``" is structural, not a key check.
# --------------------------------------------------------------------------------------


# TODO(#724): these single-pass checkers override visit() to run one ast.walk rather than
# using visit_* dispatch, so the NodeVisitor wrapper buys little; #724 tracks converting
# them to plain functions (PrintLastExprTransformer stays a class — it needs NodeTransformer).
# Import-machinery name reserved by ImportPolicyChecker: any `ast.Name` reference to `__builtins__`
# is rejected at compile time (see the checker's docstring). Legitimate `import`/`from` statements
# compile to IMPORT_NAME and never emit an `ast.Name` load of it, so reserving it has no false
# positives. A follow-up reserves the `__import__` name here too.
_RESERVED_IMPORT_NAMES: frozenset[str] = frozenset({"__builtins__"})


class ImportPolicyChecker(ast.NodeVisitor):
    """Reject imports of root modules not in the ``allowed_imports`` allow-list.

    ``allowed_imports`` is a gitignore-style glob allow-list matched against each import's **root**
    module name (``import concurrent.futures`` checks only ``concurrent`` — root-scoped, as before).
    An exact name like ``"math"`` is a glob with no wildcards, so existing allow-lists are
    unchanged; ``["*"]`` allows every import; ``[]`` forbids all. Covers ``import x.y`` and
    ``from x.y import z`` statements (absolute only — relative imports have no top-level module
    to police).

    The import-machinery **name** ``__builtins__`` is reserved: any ``ast.Name`` referencing it —
    reaching the ``__import__`` wrapper through ``__builtins__["__import__"]``, an alias
    ``b = __builtins__``, or any other bare-name read of the sandbox builtins dict — is rejected
    outright, **independent of ``allowed_imports``**. This is exact and false-positive-free because
    a real ``import x`` / ``from x import y`` statement compiles to an ``IMPORT_NAME`` opcode and
    never emits an ``ast.Name`` load of ``__builtins__`` — so legitimate imports are untouched, and
    the only reason to *reference* the name is to reach the import machinery directly. Rejecting the
    **name** can't false-positive on any ``.import_module()`` (that is an ``ast.Attribute``, not an
    ``ast.Name`` load).

    Why static-reject the name rather than lean solely on the runtime ``make_secure_import`` wrapper
    (#470): the wrapper fires only when an import *executes*, so a function that closes over
    ``__builtins__["__import__"]("time")`` can be **defined and returned** from an invoke without
    ever tripping it — the forbidden-intent object escapes the sandbox as a first-class value and
    only fails when the *caller* runs it. Rejecting the name at *definition* time closes that
    deferred-execution gap for the bare-``__builtins__`` route.

    EXECUTIVE CALL (user): reserve the ``__builtins__`` name as a self-contained static defense — it
    closes the obvious *bare-name* ``__builtins__["__import__"](...)`` route to the wrapper. A
    follow-up reserves the ``__import__`` name here too (closing the same deferred-execution gap for
    the bare ``__import__("time")`` form, which this change leaves to the runtime wrapper). Both are
    **defense-in-depth, not a boundary**: name-reservation cannot close the reflective routes that
    reach the sandbox ``__builtins__`` without ever writing the name as an ``ast.Name`` — all
    confirmed to reach the wrapper under the default policy (they land on the sandbox's
    ``__builtins__``, so the wrapper still enforces the allow-list — see the containment note below):
      * ``globals()["__builtins__"]["__import__"]("x")`` — ``globals`` is an allow-listed builtin;
        ``"__builtins__"``/``"__import__"`` are string keys, not ``ast.Name`` loads. (The builtins-
        allowlist design doc keeps ``globals``/``locals`` arguing ``globals()["__builtins__"]["eval"]``
        "closes itself" because ``eval`` is not in the dict — true for ``eval``/``exec``/``compile``,
        but ``__import__`` *must* be in the dict, so the import check is bypassed here.)
      * ``locals()["__builtins__"][...]`` — at REPL top level ``locals()`` is the same dict.
      * ``vars()["__builtins__"][...]`` — no-arg ``vars()`` returns the caller namespace unpoliced
        (``vars(obj)`` *is* policed by the secure wrapper).
      * frame/generator introspection — ``(x for x in []).gi_frame.f_builtins["__import__"]("x")``;
        ``gi_frame``/``f_builtins``/``f_globals`` are non-dunder, so ``["*", "!__*"]`` does not block
        them. ``getattr`` reaches them with a computed name the static attr checker can't see.
      * ``str.format`` — ``"{0.__globals__[__builtins__]}".format(fn)`` does attribute+index traversal
        inside a runtime format string, invisible to both the static ``AllowedAttributesChecker`` and
        the runtime ``getattr`` wrapper (C-level getattr). Given any *Python* function in scope (e.g. a
        jaz-library function, whose ``__globals__`` are the real builtins) this reaches the **real**
        ``__import__`` — a full policy bypass, not just the wrapper — and no allow-list can close it.
    Fully severing reflective reach to ``__builtins__`` by name/attr bans is unachievable (it is
    threaded into every frame/function/module, and ``__import__`` must live there for imports to work
    at all), so this checker deliberately does not attempt it. The actual containment property — *no
    forbidden module is ever imported* — is upheld at run time by the wrapper, which enforces the
    allow-list on every route that lands on the sandbox's ``__builtins__`` (see ``make_secure_import``'s
    known limitations for the widened-attrs / foreign-frame / ``str.format`` cases that reach the
    *real* ``__import__``). Bare ``__import__(...)`` calls (literal or computed) are likewise not yet
    checked here — the wrapper enforces them at runtime until the ``__import__``-name reservation
    follow-up lands. ``importlib.import_module(...)`` is not checked here either (an ``ast.Attribute``,
    only reachable when a host allow-lists ``importlib``, itself an escape hatch); under a permissive
    allow-list a host needing a computed dynamic import can still reach it that way.
    """

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
            elif isinstance(n, ast.Name) and n.id in _RESERVED_IMPORT_NAMES:
                # Reserve the import-machinery name (see class docstring). Unconditional — not
                # gated on the allow-list — because referencing it by name is the escape hatch
                # regardless of what `allowed_imports` would permit for a statement.
                raise ImportError(
                    f"Direct use of the '{n.id}' name is forbidden in the REPL "
                    "(it exposes the import machinery); use a plain 'import' statement."
                )
        disallowed = sorted(r for r in roots if not is_allowed(self._spec, r))
        if disallowed:
            modules = ", ".join(f"'{m}'" for m in disallowed)
            raise ImportError(f"Import of module {modules} is forbidden in the REPL.")


class AllowedAttributesChecker(ast.NodeVisitor):
    """Reject any ``ast.Attribute`` access whose name is not in the ``allowed_attributes`` allow-list.

    ``allowed_attributes`` is a gitignore-style glob allow-list matched against the accessed
    attribute name. The secure-by-default value is ``["*", "!__*"]`` — allow every attribute
    *except* double-underscore-prefixed ones (gitignore negation), which is exactly the old
    ``forbidden_attributes=["__*"]`` denylist inverted. ``["*"]`` allows everything (incl. dunders);
    ``[]`` forbids all attribute access. This is the allow-only counterpart of the former
    ``ForbiddenAttributesChecker`` — the whole sandbox is now allow-lists (imports/attributes/files),
    no denylists."""

    def __init__(self, allowed_attributes: Sequence[str]) -> None:
        self._spec = compile_allowlist(allowed_attributes)

    def visit(self, node: ast.AST) -> None:
        found = sorted(
            {
                n.attr
                for n in ast.walk(node)
                if isinstance(n, ast.Attribute) and not is_allowed(self._spec, n.attr)
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
    input_id: str,
    filename: str = "<repl>",
    mode: CompileMode = "exec",
    allow_top_level_await: bool = False,
    allowed_imports: Sequence[str] = ("*",),
    allowed_attributes: Sequence[str] = ("*",),
    cookie_jar: CookieJar | None = None,
    tree: ast.AST | None = None,
):
    """Compile validated REPL source to a code object.

    ``allow_top_level_await`` adds ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` to the final compile so
    a top-level ``await`` is legal (the code object gains ``CO_COROUTINE``) — the async
    ``aexec``/``aeval`` path. The flag only affects the final ``compile()``; the *same*
    policy-validated AST is compiled, so every restriction still holds. See #567.

    ``cookie_jar`` (passed by the REPL, ``None`` for direct non-agent callers) carries the
    return-signal / tag-raise / exception references the REPL's transforms embedded as co_consts
    *cookies* (see ._cookies); when given, its placeholders are substituted for the real objects in
    the compiled code object. Direct callers pass no jar and get a plain compile.

    ``allowed_imports`` / ``allowed_attributes`` default to ``("*",)`` (allow-all), NOT the
    fail-closed ``[]``. This is deliberate layering, not a fail-open: ``secure_compile`` is a
    general compile primitive, and a deny-all default would lock out every direct caller (tests,
    tooling) that isn't sandboxing. The fail-closed deny-all default lives one layer up, at
    ``python_repl._effective_allowed_imports`` / ``_effective_allowed_attributes`` (an absent REPL
    config key → ``[]``), and **every** REPL call site passes those explicit effective values — so
    agent code is never compiled under the permissive default. The permissiveness is confined to
    direct, non-agent callers.
    """
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
    # removed: the former moved to the eval-layer ``RestrictInvokeArgs`` hook — whose ``InvokeArgsChecker``
    # AST pass (plus the ``check_json_literal`` / ``check_json_type_hint`` helpers) was relocated out of
    # this module into ``evals/repl_config_hooks.py`` (#905), its sole consumer — and the latter was
    # dropped outright as a dead feature no eval config ever enabled. Only the sandbox policies below
    # remain.
    # Sandbox policies (#688), now uniform allow-lists (imports/attributes/files) — no
    # denylists, no "unrestricted" state. `["*"]` ⇒ allow all; `[]` ⇒ forbid all. Always run
    # (an empty allow-list is meaningful: deny-all); they precede PrintLastExprTransformer.
    policies.append(ImportPolicyChecker(allowed_imports))
    policies.append(AllowedAttributesChecker(allowed_attributes))
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
