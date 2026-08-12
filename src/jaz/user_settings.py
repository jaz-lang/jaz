"""The persistent, user-scoped settings file: ``~/.jaz/settings.json``.

The file holds configuration as authored *data* — the nested, grouped shape an eval YAML's
``jaz:`` block uses: ``llm`` / ``repl`` / ``protocol`` groups plus top-level options::

    {
      "llm": {"model": "openai/gpt-5-mini", "temperature": 0.7},
      "repl": {"exec_timeout": 60}
    }

It is **console-scoped**: the ``jaz`` console reads it at startup (see
:func:`jaz.console._apply_user_settings`), compiles it via :func:`jaz.instantiate.build_config`,
and applies it with :func:`jaz.configure` for the session — a persistent baseline for the CLI,
versus a per-run ``configure`` call. Library ``import jaz`` does **not** read it: a Config there
is fully determined by its call site, so code written on top of jaz stays portable and cannot be
changed by a file in whoever's home directory it happens to run under.

Within a console session, settings resolve lowest-to-highest: built-in defaults, **this file**,
startup flags (``--model``), then any in-session :func:`jaz.configure` / :class:`jaz.ConfigOverride`.

API keys do not go in this file. A ``{"llm": {"api_key": "sk-…"}}`` here is accepted — it is
compiled straight onto the backend like any other constructor argument — but this file is
plaintext on disk, which is exactly what to avoid for a credential. Backends read their key from
the environment (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, …) when none is configured; leave it
there.
"""

# EXECUTIVE CALL (user, 2026-08-11): the settings file is scoped to the ``jaz`` console/CLI, NOT
# to library ``import jaz``. This was a review point (uranium11010, #1043): a settings file that
# every ``import jaz`` reads makes software built on jaz non-portable — the same code behaves
# differently on a machine with a different ``~/.jaz`` — and it also made a malformed file fatal
# to *every* jaz process at import (there is no pre-import seam to make ``_default_config =
# Config()`` tolerant). Scoping the read to the console dissolves both: an `import jaz` can't be
# reconfigured or broken by the file, and the one reader (the console) is an interactive entry
# that can report a bad file cleanly. The earlier design read the file in ``Config.__init__``.
#
# Why JSON and not TOML/YAML: the authored config shape is already JSON-native — it is exactly
# what ``build_config`` compiles (an eval YAML's ``jaz:`` block is this same tree), so the file
# format needed no design at all. TOML would need a translation layer for the nested ``params``
# maps, and YAML is not in the base dependency set.
#
# Why no caching. The console reads it once at startup; there is no hot path to cache, and a cache
# would only add a staleness question ("I edited the file, why is my session ignoring it?").

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .paths import config_dir

#: Filename within the per-user jaz directory.
_SETTINGS_FILENAME = "settings.json"

#: Appended to every "this settings file is broken" error (here and in the console reader). The
#: path itself is already in the message prefix; this names the recovery a text editor gives.
SETTINGS_RECOVERY_HINT = (
    "Fix it by editing that file, or delete it to fall back to built-in defaults."
)
# EXECUTIVE CALL (user, 2026-08-11): a broken settings file FAILS FAST — no silent degradation. A
# wrong-but-ignored file is the dangerous case (a typo'd or half-written file that is quietly
# skipped looks identical to one being honoured, and the user debugs the wrong layer). Because the
# file is console-scoped, "fail fast" means the console prints this message and exits non-zero
# (see `jaz.console._apply_user_settings`) rather than starting on a config the user did not write;
# a library `import jaz` is unaffected. Recovery is a text editor. A future console helper-agent
# that can inspect and repair the file in-session is tracked in #1070.


def settings_path() -> Path:
    """Return the path of the user settings file (whether or not it exists)."""
    return config_dir() / _SETTINGS_FILENAME


def load_user_settings() -> Mapping[str, Any]:
    """Read the user settings file and return it as a mapping of ``configure`` groups.

    Returns an empty mapping when the file does not exist — having no settings file is the
    normal case, not an error.

    Raises:
        ValueError: The file exists but does not contain a JSON object (including invalid
            JSON). The message names the offending path.
    """
    # A missing file is indistinguishable from an empty one on purpose. Errors are reserved
    # for a file that exists and is WRONG, because that is the case where silence is
    # dangerous: a typo'd or half-written settings file that is quietly ignored looks
    # identical to one that is being honoured, and the user debugs the wrong layer.
    path = settings_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        # An unreadable-but-present file (bad permissions, a directory, a dangling symlink)
        # is a real misconfiguration and gets the same loud treatment as bad JSON — silently
        # degrading to defaults here is how a machine ends up mysteriously not using the
        # settings its owner believes are in force.
        raise ValueError(
            f"{path}: could not be read: {exc}\n{SETTINGS_RECOVERY_HINT}"
        ) from exc

    if not raw.strip():
        # A zero-byte file is what `touch ~/.jaz/settings.json` leaves behind, and treating
        # that as a JSON error would punish an obvious first step. Empty means no settings.
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: must be valid JSON: {exc}\n{SETTINGS_RECOVERY_HINT}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{path}: must contain a JSON object mapping config groups to settings "
            f"(e.g. {{'llm': {{'model': 'openai/gpt-5-mini'}}}}), "
            f"got {type(parsed).__name__}\n{SETTINGS_RECOVERY_HINT}"
        )
    return parsed
