"""The boundary between authored config *data* and the config schema.

Everything in this module deals with mappings that came from **outside the code's control** — a
host's YAML, a persisted settings file, a hand-written ``configure()`` call. Their job is to turn that data into the canonical grouped shape ``Config`` stores, and to
fail loudly and specifically when it cannot.

Four concerns live here (the first three used to sit inside :mod:`jaz._config`):

* **Migration.** The plural ``repl.configs`` map that authored YAML configs still write. (Per-leaf
  relocation of renamed options used to live here too, as ``_RELOCATED_LEAVES``; it was removed
  as core cruft — a host whose configs still carry a renamed key rewrites the authored block
  before it reaches this boundary.)
* **Normalisation.** The ``repl="python"`` shorthand, and the uniform flat -> ``params`` lift that
  makes the ergonomic authored form and the structural ``{tag, params}`` form the same config.
* **Rejection.** One error builder that lists every unknown option.
* **Introspection.** :func:`declared_init_keys` — the schema itself, read off a component's
  ``__init__``; what the console's settings view walks for every component.

Keeping them here rather than next to the schema is the point. Migration is a *preprocessing
pass*: it rewrites the input and lets the rest of the machinery stay ignorant of the history.
The practical payoff is that every path into a :class:`Config` shares one implementation of the
rewrite instead of several that had to be kept in agreement by hand — a hazard they carried
comments about, and which had already drifted once.
"""

from __future__ import annotations

import inspect
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
    "build_config",
    "declared_init_keys",
    "unknown_option_error",
]

# This list is the module's surface *to the rest of the package*: the names another module under
# `src/jaz` imports. `build_config` is here because every source of config-as-data invokes it —
# the console most of all, plus the user-settings file;
# `unknown_option_error` because `config` raises it from its own validation; and
# `declared_init_keys` because the console reads it to render a component's settings.
#
# `_normalize_group` was never listed — it is `build_config`'s internal. `Resolver`, `Resolvers`
# and `build_component` are the same thing and were listed anyway: the two aliases annotate this
# module's own signatures and the builder is `build_config`'s per-group step, and no other module
# imports any of the three. Listing them claimed a boundary that does not exist. Re-list one the
# moment something outside actually needs it — for the aliases that means a caller annotating a
# `resolvers=` argument it passes.


def declared_init_keys(cls: type) -> frozenset[str]:
    """The named (introspectable) parameters of ``cls.__init__``, excluding ``self``.

    Walks the MRO and unions each class's own declared parameters, because the common
    forwarding idiom (``def __init__(self, ..., **retry): super().__init__(**retry)``)
    hides the base's parameters behind a ``**kwargs`` that cannot be introspected. Without
    the walk, settings declared on a base class — such as ``BaseLLM``'s ``retry_policy`` — would be
    invisible on every subclass that forwards them.

    ``*args``/``**kwargs`` are themselves never reported: a class whose settings exist
    *only* behind ``**kwargs`` (declared on no class in the MRO) must name them explicitly.
    """
    # Here rather than under `jaz.llm` because it is not LLM-specific: the console's settings
    # reference walks it to list what a backend "also accepts", and this module is already the
    # one that treats a component's `__init__` as its config schema — `_check_declared`, further
    # down, does the same introspection for the error-message path.
    keys: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        keys.update(
            name
            for name, param in inspect.signature(init).parameters.items()
            if name != "self"
            and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        )
    return frozenset(keys)


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
    than one REPL. It is still what authored YAML configs write, so it is accepted and folded here.
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
    #     configure(llm={"backend": "litellm", "model": "openai/gpt-5-mini", "retry_policy": {"max_retries": 2}})
    #     ≡ configure(llm={"backend": "litellm",
    #                      "params": {"model": "openai/gpt-5-mini", "retry_policy": {"max_retries": 2}}})
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
    """Authored grouped config -> canonical kwargs for building a :class:`Config`.

    This is the single entry point for config that arrives as **data**: a host's YAML, a
    persisted settings file, a hand-written ``configure()`` call. It applies every migration and
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

    Idempotent: running it on its own output changes nothing, which is what lets a caller apply it
    unconditionally without caring whether someone already did.
    """
    # Lazy, and only here: `config` imports this module, so a module-level import back would
    # close a cycle. By the time any config is built the schema is fully defined.
    #
    # The group registry is read from `config` rather than copied, so a group added there is
    # normalised here automatically. A local copy would have failed silently: a fourth group
    # would still be routed and merged by the config layer, but would skip the flat->`params`
    # lift and the closed-set rejection with nothing raising.
    from ._config import _SUBCONFIG_TYPES

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
            continue  # already a component, or a bad value `Config` rejects by type
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
        # so a YAML/settings block fails at the call that wrote it, with a message in the
        # config's own vocabulary. The Python path (`configure(llm=LiteLLM(...))`) bypasses this
        # and is guarded by the backend's own constructor, which rejects an empty model.
        # Only for a backend that HAS a model: `model` is the backend's own setting, not the
        # base's, so a class that declares none (a fixed local server) is built without one —
        # and would refuse `model=` if it were forced through.
        if "model" in declared_init_keys(cls) and "model" not in params:
            raise ValueError(
                "llm.model is required: e.g. llm={'model': 'openai/gpt-5-mini'}. "
                "An LLM with no model is underspecified, so there is no default."
            )
        # No key check: a backend's `__init__` may end in an open `**query_params` tail (a
        # backend forwards unrecognised kwargs to its API), so there is no closed set to check
        # against. This is the deliberate exception — see `_check_declared`.
        return cls(**params)

    # Protocol and REPL share one path: both constructors close over a declared key set, so both
    # get the checked message. The REPL used to be the exception — `BaseREPL.from_dict` filtered
    # `params` to that set instead of checking against it, so a key naming a removed feature was
    # ignored rather than reported. That is a tolerance for stale AUTHORED data, and it lived
    # inside the framework on behalf of the consumer that produces it; dropping stale authored
    # keys is that consumer's job, before `build_config`, and a key that reaches here is a config
    # error like any other.
    _check_declared(group, cls, params)
    return cls(**params)


def _select_class(group: str, tag: str | None, resolvers: Resolvers | None) -> Any:
    """The class a group's spec names, via the caller's resolver — or the group's default.

    An unnamed component is core's own default (``LiteLLM`` / ``PythonREPL`` /
    ``CodeOnlyProtocol``), which core legitimately knows: a default is not a registry, it is one
    class per group. A *named* one is the caller's resolver's to answer, because the tag -> class
    mapping belongs to whoever defined the config format — only the host that wrote
    ``docker_python`` into its configs knows which class it means.
    """
    # Lazy: `config` imports this module, so a module-level import back would close a cycle.
    from ._config import _default_classes

    default = _default_classes()[group]
    if tag is None:
        return default

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
    registration. Without one the mistake surfaces several frames later as a ``TypeError`` from
    calling whatever the mapping returned, naming neither the group nor the mapping that produced
    it.
    """
    from .llm._base import BaseLLM
    from .protocol._base import BaseProtocol
    from .repl._base import BaseREPL

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
