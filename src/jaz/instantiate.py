"""The boundary between authored config *data* and the config schema.

Everything in this module deals with mappings that came from **outside the code's control** — an
eval YAML, a persisted settings file, a worker-process handoff, a hand-written ``configure()``
call. Their job is to turn that data into the canonical grouped shape ``Config`` stores, and to
fail loudly and specifically when it cannot.

Three concerns live here, all of which used to sit inside :mod:`jaz.config`:

* **Migration.** The plural ``repl.configs`` map that ~107 eval YAMLs still write. (Per-leaf
  relocation of renamed options used to live here too, as ``_RELOCATED_LEAVES``; it was removed
  as core cruft — the only remaining consumers are eval YAMLs, so that shim now lives in
  ``evals/`` and rewrites the authored block before it reaches this boundary.)
* **Normalisation.** The ``repl="python"`` shorthand, and the uniform flat -> ``params`` lift that
  makes the ergonomic authored form and the structural ``{tag, params}`` form the same config.
* **Rejection.** One error builder that lists every unknown option.

Keeping them here rather than next to the schema is the point. Migration is a *preprocessing
pass*: it rewrites the input and lets the rest of the machinery stay ignorant of the history.
The practical payoff is that :meth:`Config.update` and :meth:`Config.validate_update` share one
implementation of the rewrite instead of two that had to be kept in agreement by hand — a hazard
both of them carried comments about, and which had already drifted once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: A consumer's tag -> class mapping for one group; returning ``None`` means "unknown name".
Resolver = Callable[[str], Any]
Resolvers = Mapping[str, Resolver]

#: The key each group names its component with, in the authored form.
_SELECTOR_KEY = {"llm": "backend", "repl": "language", "protocol": "name"}

# Not one shared word: each key names the axis that varies within its group — an llm varies by
# `backend`, a repl by `language`, and a protocol's value just is its `name`. #1066 renamed only
# llm's old `tag` (jargon, and a misnomer once a non-vendor backend such as litellm exists — see
# #1082) rather than unifying to a single generic key: the selector leads every authored block,
# so it reads best when it names the axis. `language`/`name` were already right and are unchanged.

#: The REPL a config selects when nothing names one.
_DEFAULT_LANGUAGE = "python"

__all__ = [
    "Resolver",
    "Resolvers",
    "build_component",
    "build_config",
    "unknown_option_error",
]

# `_normalize_group` is `build_config`'s internal, not surface: nothing outside this module calls
# it. `build_config` is public because the phase-5 data sources (eval harness, worker, user
# settings) will invoke it explicitly, and `unknown_option_error` because `config` raises it from
# its own validation.


def unknown_option_error(
    names: str | Iterable[str], *, group: str | None = None
) -> ValueError:
    """A rejection naming *every* unknown option.

    Lists all of them rather than the first: these are typically hand-written YAMLs, where
    reporting one stale key at a time costs a fix-and-rerun cycle per key.
    """
    where = f"{group} config option" if group else "config option"
    unknown = [names] if isinstance(names, str) else sorted(names)
    if len(unknown) == 1:
        msg = f"Unknown {where}: {unknown[0]!r}"
    else:
        msg = f"Unknown {where}(s): {unknown!r}"
    return ValueError(msg)


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

    Raises ``ValueError`` for an unknown key in a group whose leaf set is closed.
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
    #     configure(llm={"backend": "litellm", "model": "openai/gpt-5-mini", "max_retries": 2})
    #     ≡ configure(llm={"backend": "litellm",
    #                      "params": {"model": "openai/gpt-5-mini", "max_retries": 2}})
    #
    # The second is the structural form; the first is what a human writes. No per-key
    # special-casing — ``model``, ``base_url``, the retry settings and any backend's own settings
    # all travel the same path, which is why a new backend needs no config-layer change at all.
    #
    # A subconfig with no ``params`` bag (``repl``, ``protocol``) has a closed leaf set, so an
    # unknown key there is still a typo and still raises.
    if "params" not in valid:
        raise unknown_option_error(unknown, group=group)
    lifted = {leaf: leaves.pop(leaf) for leaf in sorted(unknown)}
    leaves["params"] = {**leaves.get("params", {}), **lifted}
    return leaves


def build_config(
    updates: Mapping[str, Any],
    *,
    default_language: str | None = None,
    resolvers: Resolvers | None = None,
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
    ``configs: {bash: ...}`` map still lands in a session already switched to bash. It selects
    no class — that is ``resolvers``' job — so it affects only which slice of a legacy
    ``configs`` map survives.

    Idempotent: running it on its own output changes nothing, which is what lets
    :meth:`Config.update` apply it unconditionally without caring whether a caller already did.
    """
    # Lazy, and only here: `config` imports this module, so a module-level import back would
    # close a cycle. By the time any config is built the schema is fully defined.
    #
    # The group registry is read from `config` rather than copied, so a group added there is
    # normalised here automatically. A local copy would have failed silently: a fourth group
    # would still be routed and merged by `Config.update`, but would skip the flat->`params`
    # lift and the closed-set rejection with nothing raising.
    from .config import _SUBCONFIG_TYPES

    if default_language is None:
        # `is None`, not falsiness: the signature says `str | None`, and an empty string should
        # not silently take the fallback even though no empty language is valid today.
        default_language = _DEFAULT_LANGUAGE

    out = dict(updates)
    for group in _SUBCONFIG_TYPES:
        if group not in out:
            continue
        value = out[group]
        if isinstance(value, str):
            # Bare-string shorthand, uniform across the groups: `repl="python"` has always been
            # accepted, and `llm="openai"` / `protocol="code_only"` mean the same thing — name the
            # component, take its defaults.
            value = {_SELECTOR_KEY[group]: value}
        if not isinstance(value, Mapping):
            continue  # already a component, or a bad value `Config.update` rejects by type
        leaves = _normalize_group(
            group,
            dict(value),
            valid={_SELECTOR_KEY[group], "params"},
            default_language=default_language,
        )
        out[group] = build_component(group, leaves, resolvers=resolvers)
    return out


def build_component(
    group: str,
    spec: Mapping[str, Any],
    *,
    resolvers: Resolvers | None = None,
) -> Any:
    """A ``{<selector>, params}`` spec -> the configured component it describes.

    ``spec`` is the canonical form: the group's selector key (``backend`` / ``language`` /
    ``name``) plus a ``params`` mapping. ``resolvers`` maps the selector to a class and the params
    become its constructor arguments, so **the component's own signature is the config schema** —
    a backend, REPL or protocol declares what it accepts simply by declaring ``__init__``, and
    this module holds no per-component knowledge.

    Raises ``ValueError`` for a name no resolver answers, ``TypeError`` for a resolver return
    that is not the group's base class or for a param the selected component does not accept —
    all at the call that wrote them, rather than at the first invoke.
    """
    params = dict(spec.get("params") or {})
    tag = spec.get(_SELECTOR_KEY[group])

    # A non-string tag is a *component written in the old place*: `llm={"backend": my_backend}` was
    # how you supplied a pre-built one when config held a tag plus a bag. Rejected loudly rather
    # than coerced, because the coercion is silent and catastrophic — an earlier version fell
    # back to the default tag here, so `llm={"backend": mock}` quietly built a real OpenAI backend
    # and the test suite started making live HTTPS calls instead of using the mock.
    if isinstance(tag, str) and not tag:
        # `""` is not "absent" — `tag or <default>` would coerce it to the default, which is the
        # same silent-fallback shape as the component-in-the-tag-slot case above. An empty tag
        # is a config error, and the useful thing is to say so.
        raise ValueError(
            f"{group}.{_SELECTOR_KEY[group]} must be a non-empty string naming a {group}"
        )
    if tag is not None and not isinstance(tag, str):
        key = _SELECTOR_KEY[group]
        raise ValueError(
            f"{group}.{key} must be a string naming a {group} "
            f"(got {type(tag).__name__}). A configured component is now passed directly: "
            f"{group}=<component>, not {group}={{{key!r}: <component>}}."
        )

    cls = _select_class(group, tag, resolvers)

    if group == "llm":
        # The MODEL is required — no silent default. #1086's *model* half: an implicit model pins
        # one nobody wrote, so a missing `model` is rejected here at the authored-data boundary
        # (not only at `Agent.__init__`'s `get_model` guard) so a YAML/settings block fails at the
        # call that wrote it. The Python path (`configure(llm=LiteLLM(...))`) bypasses this and is
        # guarded by the empty-model default + that same guard.
        if "model" not in params:
            raise ValueError(
                "llm.model is required: e.g. llm={'model': 'openai/gpt-5-mini'}. "
                "An LLM with no model is underspecified, so there is no default."
            )
        # No key check: `BaseLLM.__init__` ends in `**request_defaults`, a genuinely open tail (a
        # backend forwards unrecognised kwargs to its API), so there is no closed set to check
        # against. This is the deliberate exception — see `_check_declared`.
        return cls.from_dict(params)
    if group == "protocol":
        _check_declared(group, cls, params)
        return cls.from_dict(params)

    # No key check on the REPL path, unlike the protocol path. `repl.configs[language]` is an
    # open, host-authored bag — an eval YAML may carry a key meant for another REPL, or one
    # whose feature has since been removed — and `REPL.from_dict` documents that it drops those
    # rather than raising. Checking the declared set here would override that contract and make
    # an unrelated stale key fatal at invoke time: four such keys (`allow_inspect_commands` and
    # friends) sit in ~96 eval configs today.
    # `from_dict` rather than `cls(**params)`: a REPL may map its config shape onto a different
    # constructor.
    return cls.from_dict(params)


def _select_class(group: str, tag: str | None, resolvers: Resolvers | None) -> Any:
    """The class a group's spec names, via the caller's resolver — or the group's default.

    An unnamed component is core's own default (``LiteLLM`` / ``PythonREPL`` /
    ``CodeOnlyProtocol``), which core legitimately knows: a default is not a registry, it is one
    class per group. A *named* one is the caller's resolver's to answer, because the tag -> class
    mapping belongs to whoever defined the config format — only the eval harness knows what
    ``docker_python`` is meant to select.
    """
    # Lazy: `config` imports this module, so a module-level import back would close a cycle.
    from .config import _default_components

    default = _default_components()[group]
    if tag is None:
        return type(default)

    # The resolver is consulted FIRST, and the default's own name is only a fallback for a caller
    # that supplied none. The reverse order silently outranked the caller: a consumer deliberately
    # mapping `litellm` to an instrumented subclass got core's class back with no diagnostic,
    # which is the one silent fallback this module otherwise refuses (see the empty-tag and
    # component-in-the-tag-slot guards above).
    resolver = (resolvers or {}).get(group)
    if resolver is not None:
        cls = resolver(tag)
        if cls is not None:
            return _checked(group, tag, cls)

    if resolver is None:
        raise ValueError(
            f"{group}.{_SELECTOR_KEY[group]}={tag!r} names a component, but no {group} "
            f"resolver was supplied. Pass resolvers={{{group!r}: <tag -> class>}} to map the "
            f"name, or pass the built component directly as {group}=<component>."
        )
    raise ValueError(
        f"Unknown {group}.{_SELECTOR_KEY[group]}: {tag!r}. The supplied {group} resolver "
        "does not know that name."
    )


def _checked(group: str, tag: str, cls: Any) -> Any:
    """Reject a resolver return that is not the group's base class.

    The three ``@register_*`` decorators this replaced each did an ``issubclass`` check at
    registration. Without one the mistake surfaces several frames later as
    ``AttributeError: ... has no attribute 'from_dict'``, naming neither the group nor the
    mapping that produced it.
    """
    from .llm.llm import BaseLLM
    from .protocol.base import BaseProtocol
    from .repl.base import BaseREPL

    base = {"llm": BaseLLM, "repl": BaseREPL, "protocol": BaseProtocol}[group]
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise TypeError(
            f"the {group} resolver returned {cls!r} for {tag!r}; expected a "
            f"{base.__name__} subclass"
        )
    return cls


def _check_declared(group: str, cls: type, params: Mapping[str, Any]) -> None:
    """Reject a param the selected component does not declare, before its constructor sees it.

    Not redundant with the constructor, which would also reject it: this is what makes the
    *message* useful. A bare ``TypeError: CodeOnlyProtocol.__init__() got an unexpected keyword
    argument 'add_external_thinking'`` says nothing about the option being unknown — the whole
    point of `unknown_option_error`.

    Skipped when the component's ``__init__`` ends in ``**kwargs``: the key set is then open by
    design, which is the case for `BaseLLM` (a backend forwards unrecognised request params to its
    API) and is why the LLM path does not call this.
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
    if unknown:
        raise unknown_option_error(unknown, group=group)
