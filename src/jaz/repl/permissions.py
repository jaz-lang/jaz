from __future__ import annotations

import builtins
from collections.abc import Sequence
from dataclasses import dataclass, field

from ._secure_methods.secure_attr import (
    make_secure_delattr,
    make_secure_getattr,
    make_secure_hasattr,
    make_secure_setattr,
    make_secure_vars,
)
from ._secure_methods.secure_import import make_secure_import
from ._secure_methods.secure_open import make_secure_open

# --------------------------------------------------------------------------------------
# The REPL builtins allowlist (#804, design/design_features/builtins_allowlist.md).
#
# The exec namespace's ``__builtins__`` is built from ONLY the names enumerated below (plus the
# policy-bound wrappers), NOT ``builtins.__dict__.copy()``. This makes the name surface
# fail-closed — anything not listed is simply unbound (``NameError``), including builtins nobody
# thought to forbid and any *future* Python builtin. It replaces the former
# ``ForbiddenNamesChecker`` denylist, which was near-inert (a 1.85M-submission sweep: fired on
# 0.18% of submissions, entirely ``SystemExit``) while the genuinely dangerous primitives (``eval``/``exec``/
# ``compile``/``__import__``) were exposed by default. Same fail-closed posture as
# ``allowed_imports=[]`` (#797) and ``allowed_attributes=["*", "!__*"]`` (#800).
#
# EXECUTIVE CALL — the exact contents were audited entry-by-entry against ``builtins.__dict__``
# with the user (design-doc Open Question #4), not trusted from the doc's illustrative sketch.
# Rulings that were genuine judgment calls, recorded here so they are not relitigated:
#   * Exception/warning types are DERIVED (``issubclass(x, Exception)``), not hand-listed, so a
#     new stdlib exception is covered automatically. Deriving on ``Exception`` (not
#     ``BaseException``) deliberately EXCLUDES ``BaseException``/``KeyboardInterrupt``/
#     ``SystemExit``/``GeneratorExit``/``BaseExceptionGroup`` — exactly the non-``Exception``-rooted
#     set that ``exceptions.is_fatal`` treats as fatal. This (a) keeps agent code from catching
#     ``BaseException`` and swallowing the REPL's native-``return``/``raise`` sentinels (which
#     subclass ``BaseException`` directly — see python_repl) and (b) preserves the old
#     default of forbidding ``SystemExit``/``exit``/``quit``. (Bare ``except:`` still catches
#     sentinels — it needs no name — and is left to a possible future syntactic ban.)
#   * ``__name__``/``__doc__``/``__package__`` are KEPT (harmless strings; omitting ``__name__``
#     would turn an ``if __name__ == ...`` reference into a ``NameError``), while ``__spec__`` and
#     ``__loader__`` are OMITTED (module-spec / loader objects are import-machinery reach vectors).
#   * ``globals``/``locals``/``__build_class__`` are KEPT: ``globals()["__builtins__"]`` only ever
#     exposes *this* allowlisted dict (installed as ``__builtins__``), so the classic
#     ``globals()["__builtins__"]["eval"]`` escape closes itself; ``__build_class__`` is required by
#     every ``class`` statement. See the design doc §3.
# --------------------------------------------------------------------------------------

# Function/callable builtins safe to expose unchanged.
_SAFE_FUNCTIONS: frozenset[str] = frozenset(
    {
        "abs",
        "aiter",
        "all",
        "anext",
        "any",
        "ascii",
        "bin",
        "callable",
        "chr",
        "dir",
        "divmod",
        "format",
        "globals",
        "hash",
        "hex",
        "id",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "locals",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "repr",
        "round",
        "sorted",
        "sum",
        "__build_class__",  # every `class` statement compiles to a __build_class__ call
    }
)

# Type constructors safe to expose unchanged.
_SAFE_TYPES: frozenset[str] = frozenset(
    {
        "bool",
        "bytearray",
        "bytes",
        "classmethod",
        "complex",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "list",
        "map",
        "memoryview",
        "object",
        "property",
        "range",
        "reversed",
        "set",
        "slice",
        "staticmethod",
        "str",
        "super",
        "tuple",
        "type",
        "zip",
    }
)

# Constants and harmless informational module dunders.
_SAFE_CONSTANTS: frozenset[str] = frozenset(
    {
        "True",
        "False",
        "None",
        "NotImplemented",
        "Ellipsis",
        "__debug__",
        "__doc__",
        "__name__",
        "__package__",
    }
)

# Builtins exposed but wrapped with a policy-bound secure implementation (real object grabbed
# for the base map, then overlaid in build_allowed_builtins).
_WRAPPED_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "__import__",
        "getattr",
        "hasattr",  # same C-level lookup as getattr → an unwrapped __* existence oracle
        "setattr",
        "delattr",
        "vars",
    }
)

# Exception/warning types, DERIVED as the Exception-rooted (recoverable) subset — see the
# executive-call note above for why this excludes the fatal BaseException-rooted types.
_SAFE_EXCEPTIONS: frozenset[str] = frozenset(
    name
    for name, obj in vars(builtins).items()
    if isinstance(obj, type) and issubclass(obj, Exception)
)

# The complete allowlisted name surface. Kept as a module global (rather than inlined) so the
# test suite can monkeypatch it to the full builtins table for feature tests that aren't about
# the sandbox — mirroring _DEFAULT_ALLOWED_IMPORTS. See tests/conftest.py.
_SAFE_BUILTINS: frozenset[str] = (
    _SAFE_FUNCTIONS
    | _SAFE_TYPES
    | _SAFE_CONSTANTS
    | _WRAPPED_BUILTINS
    | _SAFE_EXCEPTIONS
)


@dataclass(frozen=True)
class REPLPermissionPolicy:
    """Permission policy for the REPL's builtin wrappers (file access, imports, attributes)."""

    # Fail-closed file-access allowlists for the secure `open` wrapper (#804 follow-up): gitignore
    # glob lists (anchors `//`=fs, `~/`=home, `/`/`./`=cwd, bare=cwd; a slash anchors, a slash-less
    # name recurses), gated independently for reads vs. write-capable modes. There is NO
    # "unrestricted" state —
    # empty (the default) denies all file access; "allow everything" is explicit (`["**"]` the cwd
    # subtree, `["//**"]` the whole filesystem). See secure_open.make_secure_open.
    allowed_read_paths: Sequence[str] = field(default_factory=tuple)
    allowed_write_paths: Sequence[str] = field(default_factory=tuple)
    # Import allow-list (gitignore globs, matched on the root module name) for the runtime
    # `__import__` backstop (#470). Uniform with every other axis: `[]` (the default) denies all
    # imports, `["*"]` allows all; NO "unrestricted"/None state. Mirrors the static
    # ImportPolicyChecker / `repl.params["allowed_imports"]`.
    allowed_imports: Sequence[str] = field(default_factory=tuple)
    # Attribute allow-list (gitignore globs) for the runtime getattr/hasattr/setattr/delattr/vars
    # backstops. Mirrors `allowed_attributes` / the static AllowedAttributesChecker. `[]` (default)
    # denies all attribute access; `["*"]` allows everything (the wrappers then no-op); the REPL's
    # secure default is `DEFAULT_ALLOWED_ATTRIBUTES` (allow all except dunders and the frame-bearing
    # interpreter surface).
    allowed_attributes: Sequence[str] = field(default_factory=tuple)
    # The same allow-list applied to `setattr`/`delattr` only. `None` means "no split" — the read
    # list governs writes too, which is the behaviour every caller had before the axes were
    # separated. The REPL's secure default DOES split (see DEFAULT_ALLOWED_WRITE_ATTRIBUTES): the
    # three dunders it re-admits are readable but not settable, so a re-admission made for
    # `super().__init__(...)` cannot be turned into `type(host).__init__ = agent_fn`.
    allowed_write_attributes: Sequence[str] | None = None


def build_allowed_builtins(
    *,
    base_builtins: dict[str, object] | None = None,
    policy: REPLPermissionPolicy | None = None,
) -> dict[str, object]:
    """Build the REPL exec ``__builtins__`` from the allowlist plus policy-bound wrappers.

    When ``base_builtins`` is None, the base map is grabbed from ``builtins`` for exactly the
    names in ``_SAFE_BUILTINS`` (fail-closed). A host may pass its own ``base_builtins`` map to
    override the surface entirely (it is copied, then the wrappers are overlaid)."""
    if policy is None:
        policy = REPLPermissionPolicy()

    if base_builtins is not None:
        allowed_builtins: dict[str, object] = base_builtins.copy()
    else:
        allowed_builtins = {
            name: getattr(builtins, name)
            for name in _SAFE_BUILTINS
            if hasattr(builtins, name)
        }

    # File access: a real, heavily-used policy (allowed_read_paths / allowed_write_paths / logging).
    allowed_builtins["open"] = make_secure_open(policy)
    # Runtime import backstop (#470): the compiler's ImportPolicyChecker is static and misses
    # computed names (`__import__("o" + "s")`); this policy-bound `__import__` catches them at
    # run time against the same `allowed_imports` allow-list.
    allowed_builtins["__import__"] = make_secure_import(policy)
    # Runtime attribute backstops: apply the `allowed_attributes` allow-list to the name argument
    # of getattr/hasattr/setattr/delattr/vars, which the static checker cannot see. No-ops (the real
    # builtins) when the allow-list is trivially allow-all (`["*"]`/`["**"]`).
    patterns = policy.allowed_attributes
    # `vars` reads, so it rides the read list; `setattr`/`delattr` bind, so they ride the write one.
    write_patterns = (
        patterns
        if policy.allowed_write_attributes is None
        else policy.allowed_write_attributes
    )
    allowed_builtins["getattr"] = make_secure_getattr(patterns)
    allowed_builtins["hasattr"] = make_secure_hasattr(patterns)
    allowed_builtins["setattr"] = make_secure_setattr(write_patterns)
    allowed_builtins["delattr"] = make_secure_delattr(write_patterns)
    allowed_builtins["vars"] = make_secure_vars(patterns)
    return allowed_builtins
