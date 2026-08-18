"""Where jaz keeps per-user state on disk.

A single module owns the answer to "which directory?" so that everything persistent —
the user settings file (:mod:`jaz.user_settings`), the console's history file, and whatever
per-user state jaz grows later — agrees on one location, and so a test (or a user with an
unusual home) can redirect all of them at once by setting ``JAZ_CONFIG_DIR``.
"""

# Why a whole module for two functions: the alternative was each consumer spelling
# ``os.path.expanduser("~/.jaz")`` itself, which is how the console's history file came to
# live at ``~/.jaz_repl_history`` — a bare dotfile in the home directory, invisible to any
# future "where does jaz keep my state?" question (it has since moved under this directory,
# as ``~/.jaz/history``). Centralizing makes the directory a
# property of the package rather than a string repeated at each call site, and gives the
# test suite one seam to redirect (see the autouse isolation fixture in tests/conftest.py):
# without that seam, running the test suite on a machine with a real ``~/.jaz/settings.json``
# would silently fold the developer's own settings into every Config built during the run.
#
# EXECUTIVE CALL (user, 2026-08-05): scope for the first cut is USER-scoped state only —
# one directory under the home directory, no project-local (``./.jaz/``) or shared
# checked-in layer. Project scoping is deliberately deferred to a design doc because it
# raises a question this layer does not: ``repl.configs["python"]`` carries the REPL sandbox
# policy (allowed_imports / allowed_read_paths / ...), so an auto-loaded, checked-in project
# file would let a cloned repository rewrite the sandbox that is supposed to contain it.
# A user-scoped file is exactly as trusted as the environment variables it sits beside, so
# that question does not arise here and no key-level trust filtering is needed yet.

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable that relocates the whole per-user directory.
_CONFIG_DIR_ENV = "JAZ_CONFIG_DIR"

#: Default location, expanded at call time (never at import — ``$HOME`` can change).
_DEFAULT_CONFIG_DIR = "~/.jaz"


def config_dir() -> Path:
    """Return the per-user jaz directory: ``~/.jaz``, or ``$JAZ_CONFIG_DIR`` if set.

    The path is returned whether or not it exists — callers that only *read* should handle
    a missing directory rather than create one. Use :func:`ensure_config_dir` to create it.
    """
    # Resolved per call, not cached in a module constant, because both inputs are mutable at
    # runtime: tests point JAZ_CONFIG_DIR at a tmp_path per test, and ``~`` depends on $HOME.
    # An empty/unset variable falls through to the default — an exported-but-empty
    # JAZ_CONFIG_DIR means "not configured", never "use the current directory", which is
    # what ``Path("")`` would otherwise resolve to.
    return Path(os.environ.get(_CONFIG_DIR_ENV) or _DEFAULT_CONFIG_DIR).expanduser()


def ensure_config_dir() -> Path:
    """Create the per-user jaz directory if absent (mode ``0o700``) and return its path.

    Existing directories are left exactly as they are, including their permissions.
    """
    # 0o700 because this is a private per-user state directory: locking it down by default is
    # the safe choice for anything that lands under ``~/.jaz`` — a world-readable *directory*
    # leaks the names of the files in it even when a file inside is itself 0o600. Set via an
    # explicit chmod after creation rather than mkdir(mode=...), which is masked by the process
    # umask (a umask of 022 would silently yield 0o755).
    #
    # Only chmod on the directory WE create. Re-chmodding an existing ~/.jaz would quietly
    # rewrite permissions the user may have chosen deliberately, and doing that as a side
    # effect of an unrelated call is the kind of surprise a config layer should never spring.
    # Parent directories created along the way keep default permissions; the sensitive leaf
    # is the one being locked down.
    directory = config_dir()
    try:
        directory.mkdir(parents=True)
    except FileExistsError:
        return directory
    directory.chmod(0o700)
    return directory
