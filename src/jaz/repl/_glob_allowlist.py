"""Shared gitignore-style allow-list matching for the REPL sandbox.

Every permission axis — imports, attributes, and file paths — is now the **same shape**: an
allow-list of gitignore-style glob patterns matched with `pathspec`'s `gitwildmatch` patterns via
`PathSpec` — gitignore *as documented*. A name/path is permitted **iff** it matches the list. There
is deliberately **no "unrestricted"/`None` state**: an empty list denies everything, and "allow all"
is the explicit pattern (`*` for a single name segment, `**` for a path). Gitignore negation
re-excludes, so `["*", "!__*"]` = "allow every name except double-underscore-prefixed ones".

`pathspec` ships TWO gitignore dialects: `PathSpec` (gitignore as *documented* — the `gitwildmatch`
pattern factory, `GitWildMatchPattern`) and `GitIgnoreSpec` (replicates Git's *actual* behavior,
which diverges from the docs on edge cases such as re-including files from an excluded directory
while walking a tree). We use **`PathSpec`/`gitwildmatch` on all three axes** — never `GitIgnoreSpec`
— so they share one dialect. The two dialects only diverge when *walking a directory tree* and
pruning excluded dirs; we never walk one — we match each already-resolved absolute path (or bare
name) individually, last-match-wins — so that divergence cannot arise here.

Imports and attributes match a bare name (no `/`), for which `gitwildmatch` degenerates to
per-segment globbing (`*`/`?`/`[seq]`, plus `!` negation), and use this module's single-`PathSpec`
matcher (`compile_allowlist`/`is_allowed`) directly. File paths use the same `gitwildmatch` patterns
and last-match-wins semantics but deliberately do NOT route through here: `secure_open` compiles each
pattern against its own anchor base (`//`=fs, `~/`=home, `/`/`./`=cwd — see
`secure_open._classify`/`_compile_patterns`) and runs its own ordered last-match-wins loop, because a
single `PathSpec` can't carry per-pattern anchor bases. Only paths add that anchoring layer, and it
only changes how a `/` in a pattern is interpreted — import/attribute names never contain a `/`, so
for a slash-less token both code paths produce the identical match and the axes agree on every name.
"""

from __future__ import annotations

from collections.abc import Sequence

import pathspec


def compile_allowlist(patterns: Sequence[str]) -> pathspec.PathSpec:
    """Compile gitignore-style allow patterns. Empty ⇒ a spec that matches nothing (deny-all),
    so the fail-closed default falls out with no special-casing."""
    # A bare ``str`` satisfies ``Sequence[str]`` statically (pyright won't catch it), and while
    # ``PathSpec.from_lines`` does reject it at runtime, it does so with an opaque
    # ``TypeError: ... is not an iterable``. Reject it here with a clear message instead — and
    # mirror ``secure_open._compile_patterns``, which needs the same guard because it iterates the
    # patterns itself (where a bare str WOULD silently fail open, char-by-char).
    if isinstance(patterns, str):
        raise ValueError(
            f"Expected a list of patterns, not a bare string {patterns!r}."
        )
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def is_allowed(spec: pathspec.PathSpec, name: str) -> bool:
    """Whether ``name`` is permitted by the compiled allow-list (last matching pattern wins,
    per gitignore, so a trailing ``!pat`` re-excludes)."""
    return spec.match_file(name)


def allows_everything(patterns: Sequence[str]) -> bool:
    """Cheap fast-path check for a *trivially* allow-all list (``["*"]`` / ``["**"]``), so hot
    call-site wrappers (e.g. the getattr backstop) can skip compiling/matching entirely.

    Conservative by design: only the two canonical spellings short-circuit. Any other allow-all
    list (``["*", "*"]``, ``["**/*"]``, …) still works — it just goes through the matcher — so
    this only affects performance, never correctness."""
    return list(patterns) in (["*"], ["**"])
