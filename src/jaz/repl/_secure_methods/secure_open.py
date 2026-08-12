from __future__ import annotations

import builtins
import enum
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from pathspec.patterns.gitwildmatch import GitWildMatchPattern

from jaz.exceptions import SandboxPermissionError


class _OpenPolicy(Protocol):
    @property
    def allowed_read_paths(self) -> Sequence[str]: ...

    @property
    def allowed_write_paths(self) -> Sequence[str]: ...


def _get_open_mode(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "mode" in kwargs:
        return kwargs["mode"]
    if len(args) > 1:
        return args[1]
    return "r"


def _is_write_mode(mode: str) -> bool:
    # `+` (update) modes can write, so they are gated by the write allowlist. This is why the
    # read/write split keys on _is_write_mode rather than the leading char alone.
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _normalize_file_path(file_arg: Any) -> Path | None:
    """Return the normalized absolute path for path-like args, or None for file descriptors."""
    if isinstance(file_arg, int):
        return None
    fs_path = os.fspath(file_arg)
    if isinstance(fs_path, bytes):
        fs_path = os.fsdecode(fs_path)
    return Path(fs_path).expanduser().resolve(strict=False)


# Anchor vocabulary. A pattern picks a BASE from its prefix, then matches relative to that base by
# one uniform rule — gitignore's own "a separator anchors" rule, applied consistently:
#
#   **A slash-less name (``foo``, ``*.env``) recurses — matches at any depth under its base. ANY
#   slash anchors it to the base's top level: a ``/`` or ``./`` prefix, a ``~/``/``//`` base
#   prefix, or an internal separator (``sub/foo``). Use ``**`` to recurse under an anchored base
#   (``**/*.env``, ``~/**/*.env``, ``//**/*.env``).**
#
#   ``//path``          → filesystem-root base, anchored (absolute)
#   ``~/path`` / ``~``  → home base, anchored (``~`` alone = the whole home tree)
#   ``/path``           → cwd base, anchored to the top level
#   ``path`` / ``./path`` → cwd base; ``./foo`` anchors, bare ``foo`` recurses iff slash-less
#
# This deliberately diverges from *raw* gitignore for the prefix forms — raw gitignore would recurse
# a slash-less tail after ``~/`` and treats ``./foo`` as a dead pattern — but "the foo in my home
# dir" is far less surprising than "any foo under home", and it keeps all four prefixes mutually
# consistent (the presence of a slash always means "anchored"). This also matches Claude Code's
# own permission globs, where ``Read(~/.zshrc)`` grants *the* home-directory ``.zshrc`` — a single
# anchored file, not ``.zshrc`` at any depth under home. Implemented by slicing off the one-char
# prefix marker (``p[1:]`` for ``//``/``~/``/``./``) and keeping the slash at index 1, so every
# slash-carrying prefix yields a leading-slash (anchored) body.
#
# Anchoring is NOT a pathspec feature — a gitwildmatch pattern only matches a path relative to *one*
# base. So we pair each pattern with its own base and match the target relativized to that base,
# evaluating the whole list as one ordered last-match-wins sequence (see _path_allowed). (An earlier
# version prepended the base as a literal prefix into one flattened fs-root spec, which injected a
# slash and silently anchored *every* slash-less pattern to one level; a later one bucketed patterns
# by base and OR'd the buckets, which lost global order and let a positive in one base defeat a later
# negation in another. This ordered per-pattern design fixes both.)
class _Base(enum.StrEnum):
    """The anchor base a pattern resolves against. StrEnum (not a plain Enum) so members keep their
    readable ``"root"``/``"home"``/``"cwd"`` values for debugging and stay ``==`` their string form."""

    ROOT = "root"
    HOME = "home"
    CWD = "cwd"


def _classify(pattern: str) -> tuple[_Base, str]:
    """Map a pattern to (anchor-base key, the gitignore pattern to match relative to that base).

    Preserves a leading ``!`` (negation). A slash-less bare pattern stays slash-less (recursive);
    every prefix form (``./``/``/``/``~/``/``//``) yields a leading-slash (anchored) body, so the
    uniform "any slash ⇒ anchored" rule holds."""
    neg = pattern.startswith("!")
    p = pattern[1:] if neg else pattern
    # For the three slash-carrying prefixes, p[1:] drops the one-char prefix marker (``/``/``~``/
    # ``.``) and keeps the ``/`` at index 1 — the slash that anchors the body to the base top level.
    if p.startswith("//"):  # "//etc/hosts" → "/etc/hosts" (anchored at fs-root)
        base, body = _Base.ROOT, p[1:]
    elif p == "~":
        base, body = _Base.HOME, "**"  # bare "~" = the whole home tree
    elif p.startswith("~/"):
        base, body = _Base.HOME, p[1:]  # "~/foo" → "/foo" (anchored under home)
    elif p.startswith("./"):
        base, body = _Base.CWD, p[1:]  # "./foo" → "/foo" (anchored to cwd top)
    else:
        # cwd base, pattern handed to pathspec verbatim — it does the anchoring itself: a leading
        # "/" already anchors ("/config" → cwd top), a slash-less name recurses ("foo" → any depth).
        # So a leading-slash pattern needs no branch of its own; only "./" (a dead pattern in raw
        # gitignore) must be rewritten to the equivalent "/…" above.
        base, body = _Base.CWD, p
    return base, (f"!{body}" if neg else body)


def _compile_patterns(
    patterns: Sequence[str], cwd: Path, home: Path
) -> list[tuple[Path, GitWildMatchPattern]]:
    """Compile each pattern to (its anchor base, a single gitwildmatch pattern), IN INPUT ORDER.

    Order is preserved because gitignore is *last-match-wins*: a later pattern overrides an earlier
    one, so a trailing ``!secret`` must be able to cancel an earlier ``**``. We deliberately do NOT
    bucket by base and OR the buckets (an earlier design did) — OR-ing loses global order, letting a
    positive in one anchor group defeat a later negation in another (``["//**", "!secret"]`` would
    wrongly allow ``secret``). Instead each pattern keeps its own base and they are evaluated as one
    ordered sequence in ``_path_allowed``. Empty input → no patterns → deny-all falls out.

    Rejects blank/comment and malformed patterns at construction (rather than silently dropping
    them, which in a security allow-list would mask a typo'd rule) — a blank or ``#`` line is a
    valid gitignore *file* line but never a valid allow-list entry."""
    # A bare str is a Sequence[str] that would iterate char-by-char — fail-open, since a slash-less
    # single char silently compiles to a broad grant ("~" → whole home, "*" → cwd tree). Reject it.
    if isinstance(patterns, str):
        raise ValueError(
            f"Expected a list of patterns, not a bare string {patterns!r}."
        )
    bases = {_Base.ROOT: Path("/"), _Base.HOME: home, _Base.CWD: cwd}
    compiled: list[tuple[Path, GitWildMatchPattern]] = []
    for p in patterns:
        base_key, body = _classify(p)
        try:
            pat = GitWildMatchPattern(body)
        except ValueError as e:
            # A malformed pattern (e.g. a bare "!") makes GitWildMatchPattern raise
            # GitWildMatchPatternError, a ValueError subclass. We catch the ValueError *base* rather
            # than importing GitWildMatchPatternError directly: pathspec's exact module for that class
            # shifts across 0.12.x releases, so the deep import trips pyright's reportPrivateImportUsage
            # (a CI failure) — and any ValueError from constructing the pattern means the same thing,
            # the entry is not a usable pattern. from None: our message already folds in str(e); the
            # pathspec exception is a library detail with no nested cause, so chaining adds only noise.
            raise ValueError(f"Invalid file-access pattern {p!r}: {e}") from None
        if pat.include is None:  # blank/whitespace/comment — no regex, matches nothing
            raise ValueError(
                f"Invalid file-access pattern {p!r}: blank or comment line."
            )
        compiled.append((bases[base_key], pat))
    return compiled


def _path_allowed(
    target: Path, compiled: list[tuple[Path, GitWildMatchPattern]]
) -> bool:
    """Whether ``target`` is permitted, by gitignore last-match-wins over the ordered patterns.

    Each pattern is tested against the target relativized to *its own* base (a target outside a base
    can't match that pattern — ``relative_to`` raises → skip it). Among the patterns that do match,
    the LAST one decides: its ``include`` flag (True for a positive, False for a ``!`` negation) is
    the verdict, so a trailing negation overrides earlier positives exactly as in a ``.gitignore``.
    Starts denied (no match → False), giving the fail-closed default. Bases are pre-resolved by the
    caller so a resolved target and a (possibly symlinked) base share one canonical space — else
    ``relative_to`` would spuriously reject a file that is really inside the base."""
    allowed = False
    for base, pat in compiled:
        try:
            rel = target.relative_to(base)
        except ValueError:
            # target and base are both resolved absolute Paths, so relative_to raises ValueError
            # ONLY for "target not under base" → skip this pattern. (A bad-type arg would raise
            # TypeError, which we deliberately don't catch.)
            continue
        if pat.match_file(rel.as_posix()):
            # _compile_patterns rejects blank/comment patterns, so every compiled pattern has a
            # non-None include; assert it to narrow the type for pyright (belt-and-suspenders).
            assert pat.include is not None
            allowed = pat.include
    return allowed


def make_secure_open(policy: _OpenPolicy) -> Callable[..., Any]:
    """Create an ``open`` wrapper bound to a fail-closed read/write path policy (#804 follow-up).

    ``allowed_read_paths`` / ``allowed_write_paths`` are gitignore-style glob lists: an ``open`` is
    permitted only if its resolved path matches the list for its direction (reads vs. write-capable
    modes, see ``_is_write_mode``). The two are **independent**, so a host can grant reading one tree
    and writing another. Patterns use the anchor vocabulary above (``//``=filesystem, ``~/``=home,
    ``/``/``./``=cwd, bare=cwd) with one uniform rule: a **slash-less name recurses (any depth); any
    slash anchors** to the base top level (use ``**`` to recurse under a base). There is deliberately
    **no "unrestricted" state**: an empty list (the default) denies everything, and "allow
    everything" is explicit — ``["**"]`` = the whole cwd subtree, ``["//**"]`` = the whole filesystem.

    This is an in-process, model-level policy — defense-in-depth, not an OS boundary. It raises the
    cost of the obvious escapes; genuinely untrusted code needs OS/process isolation.
    """
    # Resolve the anchor bases once so a resolved target and a (possibly symlinked) base share a
    # canonical space — else target.relative_to(base) would spuriously reject an in-tree file.
    # ``home`` is resolved eagerly (not lazily on first ``~`` pattern), so this assumes HOME is set:
    # on a host with HOME unset ``Path.home()`` raises RuntimeError and REPL init fails — a
    # fail-closed outcome (init errors, no access leaks), and HOME-set is a safe host assumption.
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    read_patterns = _compile_patterns(policy.allowed_read_paths, cwd, home)
    write_patterns = _compile_patterns(policy.allowed_write_paths, cwd, home)

    def _secure_open(*args: Any, **kwargs: Any) -> Any:
        mode = _get_open_mode(args, kwargs)
        file_arg = kwargs.get("file", args[0] if args else None)
        is_write = isinstance(mode, str) and _is_write_mode(mode)
        compiled = write_patterns if is_write else read_patterns
        direction = "write" if is_write else "read"

        if file_arg is None:
            raise TypeError("open() missing required argument 'file'")

        target = _normalize_file_path(file_arg)
        if target is None:
            # An int file descriptor has no path to check against the allowlist → fail closed.
            raise SandboxPermissionError(
                "File descriptor access is not allowed in this REPL environment."
            )

        if not _path_allowed(target, compiled):
            patterns = (
                policy.allowed_write_paths if is_write else policy.allowed_read_paths
            )
            allowed_display = ", ".join(patterns) or "<none>"
            verb = "Writing" if is_write else "Reading"
            raise SandboxPermissionError(
                f"{verb} '{target}' is not allowed in this REPL environment. "
                f"Allowed {direction} paths: {allowed_display}."
            )

        return builtins.open(*args, **kwargs)

    return _secure_open
