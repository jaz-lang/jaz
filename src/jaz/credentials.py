"""The persistent, user-scoped credentials store: ``~/.jaz/credentials.json``.

Keys are stored per provider tag — the same names that select a backend in a model string
(``"openai/gpt-5-mini"``) or in :func:`jaz.configure`::

    {
      "openai": {"api_key": "sk-..."},
      "anthropic": {"api_key": "sk-ant-..."}
    }

The file is created mode ``0o600`` (owner read/write only) inside a ``0o700`` directory.

A backend resolves its key from, in order: an explicit ``api_key`` passed to it, the provider's
environment variable (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``), then this file. So an exported
variable still wins for a one-off run, and the file is the persistent fallback that makes ``jaz``
work in a fresh shell.

The default :class:`~jaz.llm.LiteLLM` backend picks the entry by the provider it routes the model
to, so ``openai/gpt-5-mini`` uses the ``"openai"`` key. It is exact for providers authenticated by
a single API-key variable; for ones needing several (``azure``) or not using an API key at all
(``bedrock``, ``vertex_ai``), configure the environment rather than storing a key here.

Because the environment variable outranks this file, a stale ``OPENAI_API_KEY`` left in a
shell rc silently overrides a key stored here, and the resulting authentication error names
neither. If a key stored here seems to be ignored, check for an exported variable first.
"""

# EXECUTIVE CALL (user, 2026-08-05) — credentials live in their own file, NOT in
# ``settings.json`` and NOT in the ``Config`` object, even though passing ``api_key`` to a
# provider is already a working way to supply a key programmatically. The reason is that a
# configured key is one ``configure(llm=...)`` away from being recorded: the eval harness
# persists each run's config to a ``config.json`` on disk and ships it to ProcessPool
# workers. Keeping credentials out of Config means the secret is never config *input* in the
# first place — resolution happens at provider construction, downstream of Config entirely —
# so "save my API key" cannot become "write my API key into every eval run directory".
#
# This predates the serialization rewrite and its reasoning shifted with it: #1064
# (``8baf86b8``) removed ``Config.to_dict()`` (a configured component is no longer a
# description of itself), #1054 split out the authored-data boundary so the harness now
# records the *authored* config rather than the live one, and ``api_key`` redaction was added
# at the harness egress (``evals/eval_harness.py::_redact_secrets``). All cut the same way
# now, but the cleanest guarantee is still to keep the key off the config surface, which is
# what this file does — it does not have to trust that every future serialization path
# remembers to redact.
#
# Why no cache. NOT for the reason an earlier revision of this comment gave — it claimed
# `_set_credential(...)` followed by an `invoke(...)` "builds a new provider, which re-reads the
# file". It does not: `Agent.__init__` does `self.llm_client = self.config.llm` and reuses the
# configured instance. Freshness comes from the default LiteLLM backend re-resolving on every
# request, and from nowhere else.
#
# So the rule is: resolution must stay uncached because a cache would break the "takes effect from
# the next agent run — no restart needed" promise that `jaz.console.set_credential` makes. The
# failure mode a cache invites ("I set my key and it still says not found") is exactly the one that
# wastes an afternoon.
#
# Do NOT add a cache for speed. Resolution runs immediately before a network round-trip to an LLM
# and is four to five orders of magnitude cheaper than the call it precedes; on this code path
# per-call CPU cost cannot be the binding constraint. A read only happens at all for calls that get
# past the explicit-key and environment-variable checks, so a user who exports their key never
# touches the disk.
#
# KNOWN GAP: the single-vendor `OpenAILLM`/`AnthropicLLM` resolve once in `__init__`, so a reused
# instance never sees a key stored mid-session — for them the no-restart promise is false. Both are
# dormant since #1082 (LiteLLM is the sole registered backend), so nothing reaches this today;
# reviving either means moving their resolution to the request path.
#
# Why not the OS keyring (macOS Keychain / libsecret / Windows credential locker), which is
# what a mature CLI eventually does: it needs a new dependency and per-platform failure
# modes (headless Linux with no secret service is the common one), and the fallback path
# would still be this file. Deferred to the design doc; a 0o600 file is what Claude Code
# itself falls back to on Linux, and it is honest about the protection it offers.

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .paths import config_dir, ensure_config_dir

# Public API of this submodule. `jaz.credentials` is listed in `jaz.__all__`, so this list
# is its published contract and what the docs generator pages (a public submodule with no
# `__all__` is silently skipped — scripts/gen_api_docs.py).
#
# READERS ONLY, deliberately. `_set_credential` is absent because writing a credential is a
# CLI act, not SDK surface — see the rule and its rationale in `jaz/__init__.py` beside
# `"credentials"` in `__all__`. Keep the two in sync: promoting the writer here without
# revisiting that decision would silently undo it.
__all__ = [
    "API_KEY_FIELD",
    "credentials_path",
    "load_credentials",
    "resolve_credential",
]

#: Filename within the per-user jaz directory.
_CREDENTIALS_FILENAME = "credentials.json"

#: Field name under a provider's entry holding its API key.
API_KEY_FIELD = "api_key"


def credentials_path() -> Path:
    """Return the path of the credentials file (whether or not it exists)."""
    return config_dir() / _CREDENTIALS_FILENAME


def load_credentials() -> Mapping[str, Mapping[str, Any]]:
    """Read the credentials file, returning ``{provider: {field: value}}``.

    Returns an empty mapping when the file does not exist.

    Raises:
        ValueError: The file exists but is not a JSON object of per-provider objects. The
            message names the path but never the stored values.
    """
    # Every error message here is written to avoid echoing file *contents*: a credentials
    # file that fails to parse is exactly the file whose bytes must not land in a traceback,
    # a CI log, or a pasted bug report. `json.JSONDecodeError`'s message carries a position
    # and a reason but not the document, so it is safe to forward; the value-shape errors
    # below report only type names.
    path = credentials_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ValueError(f"{path}: could not be read: {exc}") from exc

    if not raw.strip():
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: must be valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{path}: must contain a JSON object mapping provider names to credentials "
            f"(e.g. {{'openai': {{'api_key': '...'}}}}), got {type(parsed).__name__}"
        )
    for provider, entry in parsed.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: entry for {provider!r} must be a JSON object "
                f"(e.g. {{'api_key': '...'}}), got {type(entry).__name__}"
            )
    return parsed


def resolve_credential(provider: str, field: str = API_KEY_FIELD) -> str | None:
    """Return the stored ``field`` for ``provider``, or ``None`` if there is none.

    ``provider`` is the upstream API provider — ``"openai"``, ``"anthropic"``, or whatever name
    a backend looks itself up under.

    A missing file, a missing provider entry, and a missing field are all simply "no
    credential"; only a malformed file raises.

    This is the opt-in hook: the store only takes effect for a backend that calls it. The default
    :class:`~jaz.llm.LiteLLM` backend calls it per request, with the provider it routes the model
    to; the in-tree :class:`~jaz.llm.OpenAILLM` / :class:`~jaz.llm.AnthropicLLM` classes call it
    once in ``__init__``, having one vendor each. A custom ``@register_llm``
    backend that never calls it will not read a key stored under its tag.
    """
    # Non-string values (a number, a nested object) are treated as absent rather than
    # returned: the caller puts this straight into an Authorization header, and a wrong
    # TYPE there fails deep inside the HTTP layer with an unrelated-looking error. Falling
    # through to "not configured" produces the actionable message instead.
    entry = load_credentials().get(provider)
    if not entry:
        return None
    value = entry.get(field)
    return value if isinstance(value, str) and value else None


def _set_credential(provider: str, value: str, field: str = API_KEY_FIELD) -> str:
    """Store ``value`` as ``provider``'s ``field``, returning a human-readable location.

    Creates ``~/.jaz`` (mode ``0o700``) and the file (mode ``0o600``) if absent. Other
    providers' entries are preserved. Takes effect for the next agent run in this process —
    no restart needed.

    There is no remove/list counterpart yet: rotate or delete a key by editing the JSON file
    at :func:`credentials_path` directly.

    Raises:
        ValueError: ``provider`` or ``value`` is empty, or the existing file is malformed.
    """
    # Private on purpose — the underscore is the SDK/CLI boundary, not an implementation
    # detail. `jaz.console.set_credential` is the supported spelling; see the rule beside
    # `"credentials"` in `jaz/__init__.py:__all__` for why writing is a CLI act.
    #
    # Returns a LOCATION STRING rather than the `Path` an earlier revision returned. The only
    # caller uses it to print "Stored openai api_key in <here>", and a `Path` hard-codes "the
    # credential is a file" into the signature — which is precisely what an OS-keyring backend
    # (#1076) stops being true of, where the honest answer is "macOS Keychain (jaz)". Cheap to
    # get right while the name is private; a breaking change once it is not.
    #
    # Read-modify-write rather than append: the file is a single JSON object, and preserving
    # the other providers' entries is the whole point of storing them keyed by name. A
    # malformed existing file deliberately propagates its ValueError instead of being
    # overwritten with a fresh object — silently discarding credentials the user cannot see
    # (they are, by design, not in their shell history) would be unrecoverable.
    if not provider:
        raise ValueError("provider must be a non-empty name, e.g. 'openai'")
    if not value:
        raise ValueError(f"refusing to store an empty {field} for {provider!r}")

    existing = dict(load_credentials())
    entry = dict(existing.get(provider) or {})
    entry[field] = value
    existing[provider] = entry

    ensure_config_dir()
    path = credentials_path()
    _write_private_json(path, existing)
    return str(path)


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` as JSON, owner-readable only, replacing atomically."""
    # Three properties, each deliberate:
    #
    # 1. `os.open(..., 0o600)` rather than `path.write_text()` + `chmod`. The latter creates
    #    the file with default permissions and narrows them a moment later, leaving a window
    #    in which the key is world-readable. A umask cannot loosen the mode we request here,
    #    only tighten it further, so 0o600 is a ceiling that holds on any machine — but the
    #    mode argument only applies when the file is *created*, so we must guarantee creation:
    #    unlink any leftover tmp first and open with `O_EXCL | O_NOFOLLOW`. A stale tmp from a
    #    prior SIGKILL/power loss — a regular file that kept its old mode, or a symlink that
    #    would redirect the write elsewhere — then cannot be reused or followed through.
    # 2. Write to a sibling temp file, `fsync` it, then `os.replace`. The rename is atomic
    #    within a directory, so a crash (or a full disk) mid-write leaves the previous
    #    credentials intact rather than a truncated file — losing a stored key to a failed
    #    write of an UNRELATED provider's key would be a nasty way to find out this wasn't
    #    atomic. The fsync is what extends that guarantee across a power loss: without it the
    #    rename can reach disk before the file's bytes do, leaving the new file present-but-
    #    empty on several filesystems. This atomicity is per-write only, not read-modify-write
    #    serialization: two `_set_credential` calls racing are last-writer-wins (one update is
    #    lost), but neither can corrupt the file. Locking is overkill for a single-user dotfile.
    # 3. The temp file is created in the same directory, not /tmp, both because `os.replace`
    #    cannot rename across filesystems and because /tmp is typically world-readable.
    #
    # O_NOFOLLOW is POSIX-only; it is a no-op where the platform lacks it (e.g. Windows),
    # which is acceptable since the 0o600/0o700 mode discipline is already POSIX-centric.
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
