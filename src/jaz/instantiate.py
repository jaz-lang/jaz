"""The boundary between authored config *data* and the config schema.

Everything in this module deals with mappings that came from **outside the code's control** — an
eval YAML, a persisted settings file, a worker-process handoff, a hand-written ``configure()``
call. Their job is to turn that data into the canonical grouped shape ``Config`` stores, and to
fail loudly and specifically when it cannot.

Three concerns live here, all of which used to sit inside :mod:`jaz.config`:

* **Migration.** Names that moved (``interaction_protocol`` -> ``protocol.name``), leaves that
  moved between groups (``repl.max_output_length`` -> ``protocol``), options that were removed
  outright, and the plural ``repl.configs`` map that ~107 eval YAMLs still write.
* **Normalisation.** The ``repl="python"`` shorthand, and the uniform flat -> ``params`` lift that
  makes the ergonomic authored form and the structural ``{tag, params}`` form the same config.
* **Rejection.** One error builder, so a stale key names its replacement instead of producing a
  bare "Unknown config option".

Keeping them here rather than next to the schema is the point. Migration is a *preprocessing
pass*: it rewrites the input and lets the rest of the machinery stay ignorant of the history.
``_route_relocated_leaves`` already had that shape; the rest has now joined it. The practical
payoff is that :meth:`Config.update` and :meth:`Config.validate_update` share one implementation
of the rewrite instead of two that had to be kept in agreement by hand — a hazard both of them
carried comments about, and which had already drifted once.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: The key each group names its component with, in the authored form. Three spellings for one
#: idea, kept because ~117 eval YAMLs and every existing `configure` call are written that way.
_TAG_KEY = {"llm": "tag", "repl": "language", "protocol": "name"}

#: The REPL a config selects when nothing names one.
_DEFAULT_LANGUAGE = "python"

__all__ = ["build_component", "build_config", "unknown_option_error"]

# `_normalize_group` and `_route_relocated_leaves` are `build_config`'s internals, not surface:
# nothing outside this module calls them. `build_config` is public because the phase-5 data
# sources (eval harness, worker, user settings) will invoke it explicitly, and
# `unknown_option_error` because `config` raises it from its own validation.


# Legacy flat option names -> where they now live. This is NOT a compatibility shim: the
# config surface is nested-only and nothing folds these. It exists so a stale name produces a
# precise error naming its replacement instead of a bare "Unknown config option", because the
# rename is otherwise unguessable in one place -- `interaction_protocol` became `protocol.name`.
#
# The eval YAMLs were migrated to the nested shape rather than folded (#961 review), so this
# maps names no in-tree caller uses; it is purely a migration aid for out-of-tree configs and
# can be deleted once that horizon passes.
_LEGACY_FLAT_NAMES: dict[str, tuple[str, str]] = {
    "llm_client": ("llm", "tag"),
    "model_config": ("llm", "params"),
    "max_llm_attempts": ("llm", "retry_max_attempts"),
    "llm_retry_wait_multiplier": ("llm", "retry_wait_multiplier"),
    "llm_retry_wait_min": ("llm", "retry_wait_min"),
    "llm_retry_wait_max": ("llm", "retry_wait_max"),
    "repl_configs": ("repl", "params"),
    "max_repl_output_length": ("protocol", "max_output_length"),
    "repl_exec_timeout": ("repl", "exec_timeout"),
    "repl_exec_memory_limit": ("repl", "exec_memory_limit"),
    "interaction_protocol": ("protocol", "name"),
    "system_prompt_template": ("protocol", "system_prompt_template"),
    "user_prompt_template": ("protocol", "user_prompt_template"),
    "input_truncation_advice_template": (
        "protocol",
        "input_truncation_advice_template",
    ),
    "repl_output_truncation_advice_template": (
        "protocol",
        "repl_output_truncation_advice_template",
    ),
    "max_input_length": ("protocol", "max_input_length"),
    "truncation_prefix_ratio": ("protocol", "truncation_prefix_ratio"),
}

# Leaves that moved between subconfigs: (old group, leaf) -> (new group, leaf). The authored
# spelling keeps working — routed on the way in — so a settled ownership correction costs no
# config churn. 66 eval YAMLs write ``repl: {max_output_length: N}``.
_RELOCATED_LEAVES: dict[tuple[str, str], tuple[str, str]] = {
    ("repl", "max_output_length"): ("protocol", "max_output_length"),
}

_LIBRARY_DESCRIPTION_REMOVED_HINT = (
    "The JAZ library description is no longer configurable: it is a fixed constant "
    "(jaz.library.jaz._JAZ_LIBRARY_DESCRIPTION). The option only ever selected one template, "
    "which carried no Jinja logic — drop the key."
)

# Legacy names whose migration is NOT a (group, leaf) move, and so cannot be expressed in
# ``_LEGACY_FLAT_NAMES``. Each maps to the whole sentence that follows the rejection.
#
# The library-description rows are here because the option was *deleted*, not moved, so there is
# no (group, leaf) pair to point at. Both spellings are listed since either can appear in an
# out-of-tree config: the pre-nesting flat name and the nested leaf it briefly became. Same
# reasoning as ``provider_configs`` below — a bare "Unknown config option" would leave the reader
# hunting for a new home that does not exist, when the useful thing to say is that the text is
# now fixed and no longer configurable.
#
# ``provider_configs`` needs this because the generic hint would be
# actively wrong rather than merely terse: it was a *plural* backend -> settings map, so
# "this option is now llm.params" reads as
# ``llm={"params": {"openai": {"base_url": ...}}}`` — a param literally named ``openai``,
# which the lift rule then accepts in silence. The right migration names the backend on the
# tag, which no (group, leaf) pair can say. Dropping the row instead (leaving a bare
# "Unknown config option") was the alternative; it was rejected because plural -> singular is
# precisely the rename an out-of-tree config most needs help with (user call, 2026-08-01).
_LEGACY_FLAT_HINTS: dict[str, str] = {
    "jaz_library_description_template": _LIBRARY_DESCRIPTION_REMOVED_HINT,
    "library_description_template": _LIBRARY_DESCRIPTION_REMOVED_HINT,
    "provider_configs": (
        "A config selects exactly one backend: name it on llm['tag'] and put that backend's "
        "settings directly under llm — pass llm={'tag': 'openai', 'base_url': ...}. The "
        "plural backend->settings map is gone; use configure_by_depth(...) if different "
        "depths need different backends."
    ),
}


def unknown_option_error(
    names: str | Iterable[str], *, group: str | None = None
) -> ValueError:
    """A rejection naming *every* unknown option, each pointing at its new home when it is a
    known legacy name.

    Lists all of them rather than the first: these are typically hand-written YAMLs, where
    reporting one stale key at a time costs a fix-and-rerun cycle per key. A hint names a
    single replacement, so hints trail the list (one per name that has one) instead of being
    folded into it.
    """
    where = f"{group} config option" if group else "config option"
    unknown = [names] if isinstance(names, str) else sorted(names)
    # Hint whenever the target differs from what was written -- including inside the right
    # group (``repl={"repl_exec_timeout": ...}``), where the leaf shed its group prefix.
    moves: list[str] = []
    bespoke: list[str] = []
    for name in unknown:
        custom = _LEGACY_FLAT_HINTS.get(name)
        if custom is not None:
            # Attribute the sentence to its name only when there is more than one unknown;
            # with a single one the rejection already named it.
            bespoke.append(custom if len(unknown) == 1 else f"({name!r}) {custom}")
            continue
        moved = _LEGACY_FLAT_NAMES.get(name)
        if moved is not None and moved != (group, name):
            new_group, new_leaf = moved
            moves.append(
                f"{name!r} is now {new_group}.{new_leaf} — pass "
                f"{new_group}={{{new_leaf!r}: ...}}"
            )
    if len(unknown) == 1:
        msg = f"Unknown {where}: {unknown[0]!r}"
    else:
        msg = f"Unknown {where}(s): {unknown!r}"
    tail: list[str] = []
    if moves:
        tail.append(
            "Config is nested by architectural component; " + "; ".join(moves) + "."
        )
    tail.extend(bespoke)
    if tail:
        msg += ". " + " ".join(tail)
    return ValueError(msg)


def _reject_legacy_leaves(unknown: set[str], *, group: str) -> None:
    """Raise the named-replacement error for a legacy spelling written inside a group.

    A group with a ``params`` bag lifts every unrecognised key into it, which is what makes a
    backend's own settings work with no config-layer change — but it also means a *known*
    legacy name would be silently accepted as a param and then ignored. Checked before the lift
    so the half-migrated mistake (``repl={"repl_exec_timeout": ...}``) still names its
    replacement instead of disappearing into the bag.
    """
    # Only a name that actually *moved* is a mistake. `_LEGACY_FLAT_NAMES` also maps names onto
    # themselves within their own group (`max_input_length` -> ("protocol", "max_input_length")),
    # which is the ordinary flat spelling of a param, not something to reject. That distinction
    # used to be free: every such name was a *declared leaf* of its subconfig, so it never
    # reached this check. With the subconfigs gone, a group's whole surface is params, so every
    # setting arrives here and the check has to tell the two cases apart itself.
    legacy = sorted(
        name
        for name in unknown
        if name in _LEGACY_FLAT_HINTS
        or (name in _LEGACY_FLAT_NAMES and _LEGACY_FLAT_NAMES[name] != (group, name))
    )
    if legacy:
        raise unknown_option_error(set(legacy), group=group)


def _fold_legacy_repl_configs(leaves: dict[str, Any], language: str) -> dict[str, Any]:
    """Fold the legacy ``repl.configs`` map into ``params`` for the selected language.

    ``configs`` was a ``{language -> settings}`` map back when a config could describe more
    than one REPL. It is still what ~107 eval YAMLs write, so it is accepted and folded here.
    Entries for other languages are dropped: a config selects one REPL, so they could never
    have taken effect.
    """
    configs = leaves.pop("configs", None)
    if not isinstance(configs, Mapping):
        return leaves
    selected = configs.get(language)
    if isinstance(selected, Mapping):
        # Existing `params` win: an explicit `repl={"exec_timeout": ...}` is the newer, more
        # specific spelling, so it is not overwritten by a stale duplicate in the bag.
        leaves["params"] = {**dict(selected), **leaves.get("params", {})}
    return leaves


def _route_relocated_leaves(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite a grouped update so leaves that moved between subconfigs reach their new home.

    Returns a new mapping; ``updates`` is not mutated.
    """
    out = dict(updates)
    for (src_group, src_leaf), (dst_group, dst_leaf) in _RELOCATED_LEAVES.items():
        src_value = out.get(src_group)
        if not isinstance(src_value, Mapping) or src_leaf not in src_value:
            continue
        src_value = dict(src_value)
        relocated = src_value.pop(src_leaf)
        out[src_group] = src_value
        dst_value = out.get(dst_group)
        dst_value = dict(dst_value) if isinstance(dst_value, Mapping) else {}
        # An explicit write to the NEW name wins over the legacy one, so a config being migrated
        # can carry both spellings during the transition without the old one silently winning.
        dst_value.setdefault(dst_leaf, relocated)
        out[dst_group] = dst_value
    return out


def _normalize_group(
    group: str,
    leaves: dict[str, Any],
    *,
    valid: set[str],
    default_language: str,
) -> dict[str, Any]:
    """One group's authored leaves -> the canonical leaves the schema stores.

    ``valid`` is that group's declared leaf names and ``default_language`` the configured REPL
    language, both passed in so this stays independent of the schema module that imports it.
    Mutates and returns ``leaves``; callers pass a copy.

    Raises ``ValueError`` naming the replacement for a legacy or removed key.
    """
    if group == "repl":
        leaves = _fold_legacy_repl_configs(
            leaves, leaves.get("language", default_language)
        )
    unknown = set(leaves) - valid
    if not unknown:
        return leaves
    # THE uniform rule between the machine-facing structural form and the ergonomic one: they
    # differ only in that the keys under ``params`` are written as *siblings* of the tag. So a
    # key that is not a declared leaf is a param, and gets lifted:
    #
    #     configure(llm={"tag": "openai", "model": "gpt-5-mini", "retry_max_attempts": 3})
    #     ≡ configure(llm={"tag": "openai",
    #                      "params": {"model": "gpt-5-mini", "retry_max_attempts": 3}})
    #
    # The second is the structural form; the first is what a human writes. No per-key
    # special-casing — ``model``, ``base_url``, the retry settings and any backend's own settings
    # all travel the same path, which is why a new backend needs no config-layer change at all.
    #
    # A subconfig with no ``params`` bag (``repl``, ``protocol``) has a closed leaf set, so an
    # unknown key there is still a typo and still raises.
    _reject_legacy_leaves(unknown, group=group)
    if "params" not in valid:
        raise unknown_option_error(unknown, group=group)
    lifted = {leaf: leaves.pop(leaf) for leaf in sorted(unknown)}
    leaves["params"] = {**leaves.get("params", {}), **lifted}
    return leaves


def build_config(
    updates: Mapping[str, Any], *, default_language: str | None = None
) -> dict[str, Any]:
    """Authored grouped config -> canonical kwargs for :meth:`Config.update`.

    This is the single entry point for config that arrives as **data**: an eval YAML's ``jaz:``
    block, a persisted settings file, a worker-process handoff. It applies every migration and
    normalisation rule in this module, so callers hand over what a human wrote and get back what
    the schema stores.

    A group given as a subconfig *instance* passes through untouched — its fields are already the
    declared leaves, so there is nothing to fold, lift or migrate. Only mappings (and the
    ``repl="python"`` shorthand) are rewritten.

    ``default_language`` is the REPL language a legacy ``repl.configs`` map is folded against
    when the update does not name one. It defaults to the schema default; a caller applying an
    update *onto an existing config* passes that config's language instead, so a
    ``configs: {bash: ...}`` map still lands in a session already switched to bash.

    Idempotent: running it on its own output changes nothing, which is what lets
    :meth:`Config.update` apply it unconditionally without caring whether a caller already did.
    """
    # Lazy, and only here: `config` imports this module, so a module-level import back would
    # close a cycle. By the time any config is built the schema is fully defined.
    #
    # The group registry is read from `config` rather than copied, so a group added there is
    # normalised here automatically. A local copy would have failed silently: a fourth group
    # would still be routed and merged by `Config.update`, but would skip the flat->`params`
    # lift and the legacy-name rejection with nothing raising.
    from .config import _SUBCONFIG_TYPES

    if default_language is None:
        # `is None`, not falsiness: the signature says `str | None`, and an empty string should
        # not silently take the fallback even though no empty language is valid today.
        default_language = _DEFAULT_LANGUAGE

    out = _route_relocated_leaves(updates)
    for group in _SUBCONFIG_TYPES:
        if group not in out:
            continue
        value = out[group]
        if isinstance(value, str):
            # Bare-string shorthand, uniform across the groups: `repl="python"` has always been
            # accepted, and `llm="openai"` / `protocol="default"` mean the same thing — name the
            # component, take its defaults.
            value = {_TAG_KEY[group]: value}
        if not isinstance(value, Mapping):
            continue  # already a component, or a bad value `Config.update` rejects by type
        leaves = _normalize_group(
            group,
            dict(value),
            valid={_TAG_KEY[group], "params"},
            default_language=default_language,
        )
        out[group] = build_component(group, leaves, default_language=default_language)
    return out


def build_component(
    group: str,
    spec: Mapping[str, Any],
    *,
    default_language: str | None = None,
) -> Any:
    """A ``{tag, params}`` spec -> the configured component it describes.

    ``spec`` is the canonical form: the group's tag key (``tag`` / ``language`` / ``name``) plus
    a ``params`` mapping. The registry resolves the tag to a class and the params become its
    constructor arguments, so **the component's own signature is the config schema** — a
    backend, REPL or protocol declares what it accepts simply by declaring ``__init__``, and
    this module holds no per-component knowledge.

    Raises ``ValueError`` for an unregistered tag and ``TypeError`` for a param the selected
    component does not accept — both at the call that wrote them, rather than at the first
    invoke.
    """
    # Lazy for the same reason as `build_config`'s import: `config` imports this module.
    from ._llm_client import create_llm
    from .protocol import create_protocol
    from .repl.registry import REPL_LANGUAGE_MAP

    params = dict(spec.get("params") or {})
    tag = spec.get(_TAG_KEY[group])

    # A non-string tag is a *component written in the old place*: `llm={"tag": my_backend}` was
    # how you supplied a pre-built one when config held a tag plus a bag. Rejected loudly rather
    # than coerced, because the coercion is silent and catastrophic — an earlier version fell
    # back to the default tag here, so `llm={"tag": mock}` quietly built a real OpenAI backend
    # and the test suite started making live HTTPS calls instead of using the mock.
    if isinstance(tag, str) and not tag:
        # `""` is not "absent" — `tag or <default>` would coerce it to the default, which is the
        # same silent-fallback shape as the component-in-the-tag-slot case above. An empty tag
        # is a config error, and the useful thing is to say so.
        raise ValueError(
            f"{group}.{_TAG_KEY[group]} must be a non-empty string naming a registered {group}"
        )
    if tag is not None and not isinstance(tag, str):
        key = _TAG_KEY[group]
        raise ValueError(
            f"{group}.{key} must be a string naming a registered {group} "
            f"(got {type(tag).__name__}). A configured component is now passed directly: "
            f"{group}=<component>, not {group}={{{key!r}: <component>}}."
        )

    if group == "llm":
        # No key check: `LLM.__init__` ends in `**request_defaults`, a genuinely open tail (a
        # backend forwards unrecognised kwargs to its API), so there is no closed set to check
        # against. This is the deliberate exception — see `_check_declared`.
        return create_llm(tag or "openai", params)
    if group == "protocol":
        cls = _resolve_protocol(tag or "default")
        _check_declared(group, cls, params)
        return create_protocol(tag or "default", params)

    language = tag or default_language or _DEFAULT_LANGUAGE
    if language not in REPL_LANGUAGE_MAP:
        raise ValueError(
            f"Unsupported REPL language: {language!r}. Known languages: "
            f"{', '.join(sorted(REPL_LANGUAGE_MAP))}. Custom REPLs must be registered "
            "with @register_repl before they are configured."
        )
    cls = REPL_LANGUAGE_MAP[language]
    # Only the *diagnosable* names, unlike the protocol path. `repl.configs[language]` is an
    # open, host-authored bag — an eval YAML may carry a key meant for another REPL, or one
    # whose feature has since been removed — and `REPL.from_dict` documents that it drops those
    # rather than raising. Checking the full declared set here would override that contract and
    # make an unrelated stale key fatal at invoke time: four such keys (`allow_inspect_commands`
    # and friends) sit in ~96 eval configs today. Restricting the check keeps the useful hints
    # for names that genuinely moved while leaving the open bag open.
    _check_declared(group, cls, params, diagnosed_only=True)
    # `from_dict` rather than `cls(**params)`: a REPL may map its config shape onto a different
    # constructor, and this is the same seam `create_llm` / `create_protocol` use.
    return cls.from_dict(params)


def _resolve_protocol(name: str) -> type:
    from .protocol import INTERACTION_PROTOCOL_MAP, validate_protocol_name

    validate_protocol_name(name)
    return INTERACTION_PROTOCOL_MAP[name]


def _check_declared(
    group: str, cls: type, params: Mapping[str, Any], *, diagnosed_only: bool = False
) -> None:
    """Reject a param the selected component does not declare, before its constructor sees it.

    Not redundant with the constructor, which would also reject it: this is what makes the
    *message* useful. A bare ``TypeError: DefaultProtocol.__init__() got an unexpected keyword
    argument 'add_external_thinking'`` says nothing about the option having been *removed*, and
    nothing about where a moved one went — the whole point of `unknown_option_error`'s hints.

    Skipped when the component's ``__init__`` ends in ``**kwargs``: the key set is then open by
    design, which is the case for `LLM` (a backend forwards unrecognised request params to its
    API) and is why the LLM path does not call this.

    ``diagnosed_only`` narrows the rejection to names that actually moved or were removed,
    leaving anything else to the component (see the REPL call site).
    """
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - builtins / C types
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    declared = {name for name in sig.parameters if name != "self"}
    unknown = set(params) - declared
    if diagnosed_only:
        # Only names this module can say something *useful* about — one that moved or was
        # removed. Everything else is a stale key the component is entitled to ignore.
        unknown &= set(_LEGACY_FLAT_NAMES) | set(_LEGACY_FLAT_HINTS)
    if unknown:
        raise unknown_option_error(unknown, group=group)


def language_of(repl: Any) -> str:
    """The registered language name of a configured REPL, for folding a legacy ``configs`` map.

    Reverse-looked-up rather than stored on the REPL: the language is the registry's name for a
    class, not a property of an instance, and duplicating it onto every REPL would be a second
    place for it to be wrong. Falls back to the default for a REPL that is not registered (a
    test double), where a legacy per-language map could not have selected it anyway.
    """
    from .repl.registry import REPL_LANGUAGE_MAP

    # Exact type first, `isinstance` only as a fallback. A registered REPL that *subclasses*
    # another registered one matches both, and an isinstance-first scan would answer with
    # whichever was registered earlier — naming a `dummylang` REPL "python" because it happens
    # to extend `PythonREPL`.
    for name, cls in REPL_LANGUAGE_MAP.items():
        if type(repl) is cls:
            return name
    for name, cls in REPL_LANGUAGE_MAP.items():
        if isinstance(repl, cls):
            return name
    return _DEFAULT_LANGUAGE
