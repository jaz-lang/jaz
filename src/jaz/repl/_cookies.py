"""Compile-time *cookies*: random byte placeholders that stand in, inside agent AST, for
protected objects (the native-``return`` sentinel, the raise-tagging hook, the real ``Exception``
class) and are swapped for those real objects in a code object's ``co_consts`` — by *identity* —
after ``compile()``.

Why cookies instead of a name binding
-------------------------------------
The REPL rewrites a top-level ``return``/``raise`` into a ``raise`` of an internal sentinel, and
the except-guard transform (#869) injects ``except`` handlers that reference that sentinel and the
real ``Exception`` class. Those references must resolve at runtime *without being nameable by the
agent*. The old approach bound the sentinel in ``__builtins__`` (``builtins["_JazReturnSignal"] =
…``) so a rewritten ``raise _JazReturnSignal(v)`` could resolve the name — but that also let agent
code write ``except _JazReturnSignal:`` to swallow its own finish, and read/forge the sentinel.
Placing the object in ``co_consts`` instead makes it reachable by the compiled bytecode yet
unreachable by any *name* or ``__builtins__`` subscript the agent can write (short of introspecting
a live code object, which the sandbox blocks separately).

Identity, not equality — and why the value must be random
---------------------------------------------------------
Substitution matches placeholders by ``id()`` (``is``), never by value, so it replaces only the
exact object a transformer injected — not an equal-valued literal in the agent's own code. That
distinction is only sound while the placeholder has *no equal peer*: CPython de-duplicates equal
constants within a single ``compile()`` into one ``co_consts`` object, which would alias the
placeholder with (or evict it in favour of) an equal agent literal — collapsing the ``is`` test
into ``==`` or losing the placeholder's identity entirely. So each placeholder is 16 random bytes
minted per compilation: unguessable (the agent cannot know the value to write a colliding literal)
and unforgeable. As belt-and-suspenders, :meth:`CookieJar.substitute` *raises* if any bytes value
*equal* to a minted cookie survives substitution (the collision signature) — a collision is then a
loud compile error, never a silent ``except b'...':`` that would ``TypeError`` at runtime. (A cookie
merely *absent* — legitimately dropped by dead-code elimination — is fine, not a collision.)
"""

import ast
import secrets
import types

# A distinctive, null-laden prefix so a stray cookie is recognizable in a dump; correctness rests on
# per-cookie identity tracking (below), not on this prefix.
_COOKIE_PREFIX = b"\x00\x00__jaz_guard_cookie__\x00"
_COOKIE_RANDOM_BYTES = (
    16  # 128 bits — collision with an agent literal is unreachable in practice
)


class CookieCollisionError(RuntimeError):
    """A bytes value equal to a minted cookie survived substitution — an agent literal of the same
    value aliased/evicted the random placeholder, so identity substitution missed it. Practically
    unreachable with 128-bit random values; raised rather than swallowed so a collision can never
    silently disable a guard."""


class CookieJar:
    """Per-compilation registry of cookie placeholders and the real objects they substitute to.

    Usage: :meth:`seed` the real objects by role, have AST transformers embed
    ``jar.const_for(role)`` where a protected reference is needed, then call :meth:`substitute` on
    the compiled code object. One jar per ``compile()``; never reuse across compilations (the random
    placeholders and identity map are compilation-scoped).
    """

    def __init__(self) -> None:
        self._obj_by_role: dict[str, object] = {}
        self._cookie_by_role: dict[str, bytes] = {}
        self._obj_by_cookie_id: dict[int, object] = {}

    def seed(self, role: str, obj: object) -> None:
        """Register the real object that *role*'s cookie will be substituted with at
        :meth:`substitute` time. Cheap; seeding a role never used mints no cookie."""
        self._obj_by_role[role] = obj

    def cookie_for(self, role: str) -> bytes:
        """Return (minting on first use) the placeholder bytes for *role*. Memoized, so every
        reference to a role shares one placeholder object — the return-rewrite and the finish-guard
        handler thus point at the same ``co_consts`` entry after substitution."""
        cookie = self._cookie_by_role.get(role)
        if cookie is None:
            # Keep a strong reference (in _cookie_by_role) so the placeholder — and hence its id,
            # the substitution key — stays alive from now until substitute() runs.
            cookie = _COOKIE_PREFIX + secrets.token_bytes(_COOKIE_RANDOM_BYTES)
            self._cookie_by_role[role] = cookie
            self._obj_by_cookie_id[id(cookie)] = self._obj_by_role[role]
        return cookie

    def const_for(self, role: str) -> ast.Constant:
        """An ``ast.Constant`` holding *role*'s cookie, for a transformer to drop into the tree
        (as an ``except`` type or a call target); it becomes the real object after substitution."""
        return ast.Constant(value=self.cookie_for(role))

    def substitute(self, code: types.CodeType) -> types.CodeType:
        """Return *code* with every minted cookie replaced (by identity) with its real object,
        recursing into nested code objects (comprehensions, lambdas, nested ``def``s — the guard
        transform runs whole-tree, so cookies can land in nested code).

        Raises :class:`CookieCollisionError` if, after substitution, any bytes value *equal* to a
        minted cookie still remains in ``co_consts`` — the signature of a collision, where an agent
        literal of the same value aliased/evicted the placeholder so identity substitution missed
        it (leaving an ``except b'...':`` / ``raise b'...'(…)`` that would ``TypeError`` at runtime).
        A cookie that is merely *absent* is not a collision — CPython legitimately drops it via
        dead-code elimination (e.g. a ``return`` after ``while True: pass``), and dropping a
        placeholder that guards unreachable code is harmless."""
        if not self._obj_by_cookie_id:
            return code
        new_code = self._replace(code)
        cookie_values = set(self._cookie_by_role.values())
        for const in _walk_consts(new_code):
            if isinstance(const, bytes) and const in cookie_values:
                raise CookieCollisionError(
                    "A REPL guard cookie collided with an equal agent literal and was not "
                    "substituted; refusing to run code with a disabled guard."
                )
        return new_code

    def _replace(self, code: types.CodeType) -> types.CodeType:
        new_consts: list[object] = []
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                new_consts.append(self._replace(const))
            elif id(const) in self._obj_by_cookie_id:
                new_consts.append(self._obj_by_cookie_id[id(const)])
            else:
                new_consts.append(const)
        return code.replace(co_consts=tuple(new_consts))


def _walk_consts(code: types.CodeType):
    """Yield every constant in *code*, recursing into nested code objects."""
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _walk_consts(const)
        else:
            yield const
