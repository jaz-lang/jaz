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
is fully determined by its call site, so code written on top of JAZ stays portable and cannot be
changed by a file in whoever's home directory it happens to run under.

Within a console session, settings resolve lowest-to-highest: built-in defaults, **this file**,
startup flags (``--model``), then any in-session :func:`jaz.configure` / :class:`jaz.ConfigOverride`.

Two backend settings are **rejected**, loudly: ``llm.api_key`` and ``llm.base_url``. A key belongs
in ``~/.jaz/credentials.json`` (see ``set_credential``) or the environment (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, …), never in a plaintext settings file; a ``base_url`` reroutes every
request — with your key attached — so it is set per-session via ``jaz.configure`` rather than
persisted here.

Loosening the REPL sandbox from this file (the ``repl.params`` allow-lists — ``allowed_imports`` /
``allowed_read_paths`` / ``allowed_write_paths`` / ``allowed_attributes`` — and ``repl.language``)
**is** allowed, but prints a warning: it changes how agent code is contained in every console
session started on this machine, and that should never happen silently. Note that a widened
``allowed_read_paths`` / ``allowed_write_paths`` reaches this directory — see the note beside
``_DEFAULT_ALLOWED_READ_PATHS`` in :mod:`jaz.repl.python_repl` for how to exclude it.
"""

# EXECUTIVE CALL (user, 2026-08-11): the settings file is scoped to the ``jaz`` console/CLI, NOT
# to library ``import jaz``. This was a review point (uranium11010, #1043): a settings file that
# every ``import jaz`` reads makes software built on JAZ non-portable — the same code behaves
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
import warnings
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

#: LLM backend leaves a settings *file* may never carry — checked at both the flat spelling
#: (``{"llm": {"base_url": …}}``) and under ``params`` (see :func:`_group_leaf`). Rejected, not
#: warned: neither is the user loosening their own sandbox, and a warning would not make them safe.
_BLOCKED_LLM_LEAVES: tuple[str, ...] = ("base_url", "api_key")

#: REPL sandbox axes (they lift into ``repl.params``) that a file MAY set but that loosen how agent
#: code is contained. Setting any of these — or ``repl.language`` — is allowed and warned, never
#: silently honoured.
_REPL_SANDBOX_AXES: tuple[str, ...] = (
    "allowed_imports",
    "allowed_read_paths",
    "allowed_write_paths",
    "allowed_attributes",
)

# What a settings file may and may not carry, and why the line is drawn here.
#
# EXECUTIVE CALL (user, 2026-08-12): a settings file MAY loosen the REPL sandbox — the
# ``repl.params`` allow-lists above, plus ``repl.language`` — but doing so emits a warning. An earlier revision of this PR rejected these outright; that was
# overturned on review (uranium11010, #1047). The file is the user's OWN, console-scoped state, and
# a person who wants a permissive sandbox as their standing CLI default is entitled to one — the
# same policy they could pass to ``jaz.configure()`` at every prompt. The warning keeps the
# loosening from being *silent*, which is the only property worth protecting: no one should contain
# agent code less than they think they do.
#
# The blast radius is a console SESSION, not the machine. Post-#1043 this file is read only by the
# jaz console (:func:`jaz.console._apply_user_settings`), never by library ``import jaz`` — so
# "changes how every later jaz process behaves" is really "changes every later console session".
# Still the interactive entry point where an agent's output lands, hence still worth a warning; but
# it is not the machine-wide, outlives-everything property an earlier draft of this comment leaned on.
#
# ``llm.base_url`` and ``llm.api_key`` are the exception and are REJECTED (user, 2026-08-12, from
# the #1047 [blocker]). They are a different kind of thing from a sandbox knob: a persisted
# ``base_url`` reroutes every request — with the stored key attached — to a host the file-writer
# chose (worse than an import unlock, and persistent the same way), and a persisted ``api_key`` is
# a plaintext credential that belongs in credentials.json / the environment. A warning cannot make
# either safe, so they fail loudly and name the supported channel.


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
    _apply_settings_policy(parsed, path)
    return parsed


#: Sentinel for "the file does not set this leaf". A leaf may legitimately be set to ``None`` /
#: ``False`` / ``[]``, so presence cannot be tested by truthiness.
_UNSET = object()


def _group_leaf(parsed: Mapping[str, Any], group: str, leaf: str) -> Any:
    """Return the value the file gives ``group.leaf`` — flat OR under ``params`` — else ``_UNSET``.

    A group with a ``params`` bag (``llm``, ``repl``) accepts every non-declared leaf in two
    spellings that ``build_config`` folds together: as a direct child of the group (the ergonomic
    authored form, ``{"llm": {"base_url": …}}``) and nested under ``params``
    (``{"llm": {"params": {"base_url": …}}}``). This centralizes that one lift rule
    (:mod:`jaz.instantiate`) so a policy check can ask "did the file set this?" without caring
    which spelling was used — the alternative is mirroring the rule at every call site, which is
    how the pre-restack guard came to check a single dead spelling.
    """
    node = parsed.get(group)
    if not isinstance(node, Mapping):
        return _UNSET
    if leaf in node:
        return node[leaf]
    params = node.get("params")
    if isinstance(params, Mapping) and leaf in params:
        return params[leaf]
    return _UNSET


def _apply_settings_policy(parsed: Mapping[str, Any], path: Path) -> None:
    """Reject leaves a settings file may never carry; warn on ones that loosen the REPL sandbox.

    Raises:
        ValueError: The file sets ``llm.base_url`` or ``llm.api_key`` (in either the flat or the
            ``params`` spelling). The message names the supported per-session channel.

    Warns:
        UserWarning: The file loosens the REPL sandbox (a ``repl.params`` allow-list, or
            ``repl.language``). This is allowed, not fatal.
    """
    # Enforced in the LOADER, not at the one call site in the console, so the rule holds for every
    # future consumer of this file by construction — a second reader (a `jaz config show`, a
    # project-scoped layer) cannot forget to apply it. It runs on the raw parsed data rather than
    # on `build_config`'s output because the policy is about what the FILE set: once compiled and
    # merged with defaults, a built PythonREPL cannot say whether `allowed_imports` came from the
    # file or is the secure default. `_group_leaf` normalizes the flat/params spellings so this
    # does not re-mirror `build_config`'s per-key rules — only its single flat->params lift.

    # Normalize the ``repl=<language>`` shorthand first, exactly as ``Config.update`` does
    # (config.py: ``if key == "repl" and isinstance(value, str): value = {"language": value}``).
    # Without this, ``{"repl": "bash"}`` is the string ``"bash"``, `_group_leaf` sees no Mapping,
    # and the ``repl.language`` warning never fires for the shorthand form.
    if isinstance(parsed.get("repl"), str):
        parsed = {**parsed, "repl": {"language": parsed["repl"]}}

    # Hard block: request-redirect / credential leaves. Fail before the warnings below, so a file
    # carrying both a blocked key and a sandbox tweak reports the fatal one.
    for leaf in _BLOCKED_LLM_LEAVES:
        if _group_leaf(parsed, "llm", leaf) is not _UNSET:
            raise ValueError(
                f"{path}: 'llm.{leaf}' cannot be set from a settings file — a persisted "
                f"{leaf} would redirect or expose every request this machine makes, silently, "
                f"for every later console session. Put a key in ~/.jaz/credentials.json (see "
                f"set_credential) or the environment, and set a base_url per-session with "
                f"jaz.configure(llm=...) in your own code.\n{SETTINGS_RECOVERY_HINT}"
            )

    # Soft warn: sandbox loosening is the user's call, but never silent.
    loosened: list[str] = []
    if _group_leaf(parsed, "repl", "language") is not _UNSET:
        loosened.append("repl.language")
    loosened += [
        f"repl.{axis}"
        for axis in _REPL_SANDBOX_AXES
        if _group_leaf(parsed, "repl", axis) is not _UNSET
    ]
    if loosened:
        warnings.warn(
            f"{path}: this settings file loosens the REPL sandbox ({', '.join(loosened)}). "
            f"Agent code in every console session started on this machine will run with that "
            f"policy. Remove these keys to keep the secure default.",
            UserWarning,
            stacklevel=2,
        )
