from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from functools import cached_property
from types import MappingProxyType
from typing import Any

from jaz.exceptions import ConfigOverrideActivationError

from .instantiate import unknown_option_error
from .llm import LiteLLM
from .llm.llm import BaseLLM
from .protocol.base import BaseProtocol
from .protocol.code_only import CodeOnlyProtocol
from .repl.base import BaseREPL
from .repl.python_repl import PythonREPL

# Default per-level governance limit (preserves the historical Config default for
# max_repl_iterations). There is no buffer — reaching max_iterations is the hard stop.
_DEFAULT_MAX_ITERATIONS = 10

# REPL sandbox policy (#688) lives in ``repl.params``. Every axis is now a uniform
# gitignore-style **allow-list** (no denylists, no "unrestricted"/None): ``allowed_imports`` /
# ``allowed_attributes`` (compiler-enforced) plus the file-access allow-lists ``allowed_read_paths``
# / ``allowed_write_paths`` (enforced by secure_open) — all un-strippable by the agent
# (``repl.params`` is a protected key, #690). The name surface is policed separately by the builtins
# allowlist installed as ``__builtins__`` (#804, permissions.py), not by a config key. All defaults
# are secure-by-default and live next to enforcement in ``repl/python_repl.py``: ``allowed_imports``
# and the file paths default to ``[]`` (deny all), ``allowed_attributes`` to
# ``DEFAULT_ALLOWED_ATTRIBUTES`` (allow all except dunders and the frame-bearing interpreter
# surface); "allow all" is always the explicit pattern (``["*"]`` for names, ``["//**"]``
# for the whole filesystem — file anchors are ``//``=fs, ``~/``=home, ``/``=cwd-top, bare=cwd). The old
# ``_DEFAULT_*`` reference constants that used to sit here were dead once #688 made the sandbox
# config-driven, so they were removed.


# Keys that may NOT appear in a per-depth partial (rejected by `_validate_depth_overrides`, so a
# clear "self-referential" error fires instead of a bare "unknown field"). `depth_overrides` is
# self-referential — a per-depth override carrying per-depth overrides has no coherent meaning — and
# is not a Config field anyway (#727), so a per-depth partial cannot set it. Everything else a
# partial can set is allowed.
_DEPTH_OVERRIDE_EXCLUDED_FIELDS = frozenset({"depth_overrides"})


# The component each group holds, and how a default one is built. THE group registry: every
# consumer derives from this one mapping rather than restating the names, so a fourth group
# cannot be added in one place and silently skipped in another.
#
# Config stores the *configured components themselves* — a `BaseLLM`, a `BaseREPL`, a
# `BaseProtocol` — not a `{tag, params}` description of them. What that buys, and what it
# cost to get here, is recorded on `Config`.
_SUBCONFIG_TYPES: dict[str, type] = {
    "llm": BaseLLM,
    "repl": BaseREPL,
    "protocol": BaseProtocol,
}
_SUBCONFIGS = tuple(_SUBCONFIG_TYPES)

#: Copy-pasteable "build one of these" hint per group, for the error a caller gets when they
#: pass a dict. Spelled with the import because none of these names is reachable from the `jaz`
#: top level, and this is printed to someone whose config just stopped working. A concrete class
#: rather than the ABC in `_SUBCONFIG_TYPES`: "must be a BaseREPL" does not tell you what to build.
_COMPONENT_HINT = {
    "llm": "from jaz.llm import LiteLLM; jaz.configure(llm=LiteLLM(model=...))",
    "repl": "from jaz.repl import PythonREPL; jaz.configure(repl=PythonREPL(...))",
    "protocol": (
        "from jaz.protocol import CodeOnlyProtocol; "
        "jaz.configure(protocol=CodeOnlyProtocol(...))"
    ),
}


def _default_components() -> dict[str, Any]:
    """A fresh default component per group.

    Built per call rather than held as module constants: a `Config` must not share a component
    with another `Config` by accident. They are treated as values (a fold rebinds whole fields
    and never mutates one in place), but a shared *default* would make that convention the only
    thing standing between two unrelated configs.
    """
    # LiteLLM is the default LLM, as PythonREPL/CodeOnlyProtocol are the default REPL/protocol —
    # v1 ships exactly one backend (design/design_features/litellm_sole_backend_v1.md), so an
    # unconfigured `llm` need not be spelled out. Like OpenAILLM() before it, this instance carries
    # no model (`model=""`), so it is a placeholder that raises at `get_model` until a model is set
    # — the model half of #1086's "no silent default" still holds; only the backend half is retired.
    return {
        "llm": LiteLLM(),
        "repl": PythonREPL(),
        "protocol": CodeOnlyProtocol(),
    }


@dataclass(init=False)
class Config:
    """JAZ framework configuration, grouped by architectural component.

    - ``llm``      (:class:`BaseLLM`) — how JAZ talks to the LLM.
    - ``repl``     (:class:`BaseREPL`) — how agent code executes.
    - ``protocol`` (:class:`BaseProtocol`) — the LLM↔REPL codec
      (prompts + parsing).
    - top-level: genuinely cross-cutting options that belong to none of the three.

    The surface is nested-only — there are no flat ``config.<option>`` aliases. Set via the
    grouped :func:`configure`/:class:`ConfigOverride` shape, passing the configured component:
    ``configure(repl=PythonREPL(exec_timeout=30))``. Configuration written as *data* — a
    settings file, an eval YAML — is compiled into components before it reaches here, so config
    itself only ever holds the built object.

    **Setting a group replaces it.** A group names the whole component, so there is no partial
    update of one component's settings."""

    # This replaced a `{tag, params}` description of each component. Three things went with it:
    #
    # * **Defaults had nowhere to live.** A bag either materialised copies of the component's
    #   defaults (which then drifted — `repl.exec_timeout` said 30 s while the constructor said
    #   unbounded) or omitted them, making "what is in force?" unanswerable from the config.
    # * **Partial merge across the tag was unsound.** `{tag, params}` is a dependent record — the
    #   valid shape of `params` depends on the *value* of `tag` — but the two merged as
    #   independent leaves, so one layer could set the tag and another the params and the fold
    #   produced a combination nobody wrote. Layers now *replace* per group, which is the only
    #   coherent rule for a tagged union.
    # * **Typing stopped at the bag.** `params: dict[str, Any]` cannot be checked; a constructor
    #   call can.

    llm: BaseLLM
    repl: BaseREPL
    protocol: BaseProtocol

    def __init__(self, **kwargs: Any) -> None:
        """Construct a Config from the grouped shape — ``llm=``/``repl=``/``protocol=`` as
        configured components, plus top-level fields — routed through :meth:`update`.

        Explicit arguments are the only source of configuration: there is no ambient layer
        underneath them, so a Config built here is fully determined by its call site.
        """
        for group, component in _default_components().items():
            setattr(self, group, component)
        if kwargs:
            self.update(**kwargs)

    def __copy__(self) -> Config:
        # Components are SHARED by reference, not copied. They are values here: a fold rebinds a
        # whole field and nothing mutates a component in place, so two configs holding the same
        # `PythonREPL` cannot interfere. Copying them instead would be wrong as well as wasteful
        # — a `BaseREPL` template is deliberately reusable (its `initialize` returns a fresh
        # instance per invoke), so duplicating it per fold would multiply objects for nothing.
        #
        # `object.__new__` rather than `Config()`: every field is rebound from the source below,
        # so running `__init__` would build a full set of default components and discard them.
        #
        # Driven off `dataclass_fields` rather than naming the fields, so adding a top-level
        # field cannot silently leave a stale value on every copy — the failure mode of a
        # hand-written copy is invisible (a wrong value, not an error).
        new = object.__new__(Config)
        for f in dataclass_fields(Config):
            setattr(new, f.name, getattr(self, f.name))
        return new

    def _validate_depth_overrides(
        self, value: object
    ) -> dict[int, dict[str, Any]] | None:
        """Validate/normalize ``depth_overrides`` (see :func:`configure_by_depth`).

        A dict keyed by absolute recursion depth (str keys coerced to int for YAML/JSON), each
        value a partial Config. Rejects the excluded fields and validates each partial through
        the SAME path ``Config.update`` uses, so a typo'd field or a bad value raises at
        configure/override time rather than at resolve time. Returns the normalized dict.
        """
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(
                "depth_overrides must be a dict keyed by recursion depth, or None"
            )
        normalized: dict[int, dict[str, Any]] = {}
        for raw_depth, partial in value.items():
            try:
                depth = int(raw_depth)
            except (TypeError, ValueError):
                raise ValueError(
                    "depth_overrides keys must be integer recursion depths, "
                    f"got {raw_depth!r}"
                ) from None
            if depth < 1:
                raise ValueError(
                    "depth_overrides depth must be >= 1 (the top invoke is depth 1), "
                    f"got {depth}"
                )
            if not isinstance(partial, dict):
                raise ValueError(
                    f"depth_overrides[{depth}] must be a dict of Config field overrides, "
                    f"got {type(partial).__name__}"
                )
            excluded = set(partial) & _DEPTH_OVERRIDE_EXCLUDED_FIELDS
            if excluded:
                raise ValueError(
                    f"depth_overrides[{depth}] may not set {sorted(excluded)}: "
                    "depth_overrides is self-referential (a per-depth override cannot itself "
                    "carry per-depth overrides)."
                )
            try:
                self.validate_update(partial)
            except ValueError as e:
                raise ValueError(f"depth_overrides[{depth}]: {e}") from None
            normalized[depth] = dict(partial)
        return normalized

    @classmethod
    def validate_update(cls, updates: Mapping[str, Any]) -> None:
        """Validate a grouped override map without applying it (non-mutating).

        ``ConfigOverride.__enter__`` installs a raw diff as a lazy stack layer and has no
        resolved Config, so it fail-fasts here instead of folding.
        """
        # Identical in structure to `update`'s loop, and now trivially so: with compilation moved
        # out to the caller, both see components and neither can interpret them differently. The
        # two used to fold a legacy `configs` map against *different* languages — `update` the
        # configured one, this the schema default — a divergence that could only be documented,
        # not fixed, while a classmethod with no config had to compile authored data.
        valid_top = set(cls.__dataclass_fields__)
        for key, value in updates.items():
            if key not in valid_top:
                raise unknown_option_error(key)
            # Every top-level field is a configured component (llm/repl/protocol), so validation
            # is a component-type check; each component's own constructor already validated its
            # leaves at the point they were written.
            cls._check_component(key, value)

    @staticmethod
    def _check_component(group: str, value: object) -> None:
        """Reject a group value that is not the component that group holds."""
        expected = _SUBCONFIG_TYPES[group]
        if isinstance(value, expected):
            return
        # A dict or a bare tag string is the one likely mistake here, because both *used* to work:
        # config compiled authored data itself before the boundary moved out. Point at where that
        # data goes now rather than only naming the offending type — a caller who gets
        # "must be a BaseREPL, got dict" back from a config file they have used for months otherwise
        # has no way to know the migration errors moved with it.
        if isinstance(value, Mapping | str):
            raise ValueError(
                f"config option {group!r} must be a {expected.__name__}, got "
                f"{type(value).__name__}. Config holds configured components — build one: "
                f"{_COMPONENT_HINT[group]}"
            )
        raise ValueError(
            f"config option {group!r} must be a {expected.__name__}, got {type(value).__name__}"
        )

    def update(self, **kwargs: Any) -> None:
        """Update config values, grouped by architectural component.

        Each of ``llm=`` / ``repl=`` / ``protocol=`` takes the configured component, plus the
        top-level fields.

        **A group is replaced, not merged.** Setting one names the whole component; there is no
        partial update of a component's settings.
        """
        # Replace rather than merge, which is the visible half of storing components. The old
        # `_merge_leaf` deep-merged a partial into the existing leaf so a caller could set one
        # setting without restating its siblings — convenient, and unsound across the tag: the
        # tag and its params merged independently, so one layer could switch the component while
        # another's params, written for the component that is no longer selected, survived and
        # were silently applied to it. A component states itself completely, so there is nothing
        # left to half-apply.
        for group in _SUBCONFIGS:
            if group not in kwargs:
                continue
            value = kwargs.pop(group)
            self._check_component(group, value)
            setattr(self, group, value)
        # Every dataclass field is a component, popped above, so anything left is an unknown
        # option. (No non-component top-level field exists to setattr — the loop only rejects.)
        for key in kwargs:
            raise unknown_option_error(key)


_default_config = Config()

_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})
_EMPTY_DEPTH_MAP: Mapping[int, Mapping[str, Any]] = MappingProxyType({})


class _LayerBase(ABC):
    """Common interface for a composable stack layer (#727): its per-depth config contribution.

    Two concrete kinds — a plain ``Layer`` (applies at every depth) and a ``DepthLayer`` (applies
    only at matching depths). Kept as a shared base so ``ConfigStack`` folds them uniformly —
    ``partial_for(depth)`` in ``resolve_for_depth`` and ``depthless_partial()`` in the depth-less
    ``get_config()`` fold — with no ``isinstance`` ladder, and so ``DepthLayer`` needn't carry a
    meaningless ``overrides`` field just to satisfy inheritance.
    """

    @abstractmethod
    def applies_at(self, depth: int) -> bool: ...

    @abstractmethod
    def partial_for(self, depth: int) -> Mapping[str, Any]: ...

    def depthless_partial(self) -> Mapping[str, Any]:
        """Contribution to the depth-less ``get_config()`` fold. Default empty (a ``DepthLayer``
        contributes nothing without a depth); a plain ``Layer`` overrides this with its full diff."""
        return _EMPTY_MAPPING


@dataclass(frozen=True)
class Layer(_LayerBase):
    """One propagating :class:`ConfigOverride`'s raw diff — the plain, depth-unconditional layer
    (#727).

    A layer carries only the field→value assignments the override requested
    (``ConfigOverride.overrides``), never a resolved ``Config``. This is what makes it
    **base-independent**: folding a layer onto any base is just replaying those
    ``Config.update`` setattrs (last-wins), so a stack of layers can be re-based onto a
    different ancestor after the fact — the property the worker-thread re-base relies on.
    """

    overrides: Mapping[str, Any]

    def applies_at(self, depth: int) -> bool:
        """A plain layer applies at every recursion depth."""
        return True

    def partial_for(self, depth: int) -> Mapping[str, Any]:
        """The partial this layer contributes at ``depth`` — its full diff, unconditionally."""
        return self.overrides

    def depthless_partial(self) -> Mapping[str, Any]:
        return self.overrides


@dataclass(frozen=True)
class DepthLayer(_LayerBase):
    """A ``ConfigOverrideByDepth``'s ``{depth: partial}`` — a depth-conditional layer (#727).

    Carries only its ``{depth: partial}`` map — no ``overrides`` field (a depth-conditional layer
    has no unconditional diff, so ``depthless_partial`` inherits the empty default). It contributes
    nothing to the depth-less ``get_config()`` fold; it only applies its matching depth's partial
    during ``ConfigStack.resolve_for_depth``.
    """

    depth_map: Mapping[int, Mapping[str, Any]] = _EMPTY_DEPTH_MAP

    def applies_at(self, depth: int) -> bool:
        return depth in self.depth_map

    def partial_for(self, depth: int) -> Mapping[str, Any]:
        return self.depth_map.get(depth, _EMPTY_MAPPING)


def _shallow_fold(partials: Iterable[Mapping[str, Any]]) -> Config:
    """Fold ``partials`` (in order, last-wins) onto a SHALLOW copy of ``_default_config``.

    Shallow (not ``deepcopy``) so a non-copyable instance ``llm_client``/``interaction_protocol``
    stays SHARED, not duplicated per fold (which would lose connection-pool / rate-limit state, and
    could raise on a lock/open-socket). ``Config.update`` rebinds whole fields via ``setattr`` and
    never mutates a container in place, so the copy is leak-safe by convention: overridden fields
    are the partials' own objects; unoverridden fields stay shared-by-reference with the base
    (#741 tracks making that isolation structural). When NO partial actually applies (all empty —
    e.g. a stack of only non-matching ``DepthLayer``s), returns ``_default_config`` by identity, no
    copy — preserving the ``resolve_for_depth(n) is _default_config`` fast path.
    """
    cfg: Config | None = None
    for partial in partials:
        if partial:
            if cfg is None:
                cfg = copy.copy(_default_config)
            cfg.update(**partial)
    return cfg if cfg is not None else _default_config


@dataclass(frozen=True)
class ConfigStack:
    """An ordered stack of propagating :class:`ConfigOverride` diffs relative to the
    ``_default_config`` singleton — the runtime value of the ``_config_stack`` ContextVar (#727).

    Replaces the old resolved-``Config`` snapshot on the ContextVar. Storing raw *ordered
    diffs* (not a flattened ``Config``) is what lets a raw ``ThreadPoolExecutor`` worker
    reconstruct propagated state by **composition** — ``ancestor.layers + local.layers``,
    last-wins — instead of the old either/or that had to pick one side and silently dropped the
    other. The base is implicitly the live ``_default_config`` (so ``configure()`` mutations
    stay visible with no rebuild); everything an operator/agent adds lives in ``layers``.

    The fold is **lazy** (see ``resolved``): a diff is only applied when the effective config is
    *read*, not when the :class:`ConfigOverride` is entered. That laziness is the fix for the
    sandbox-drop bug — a worker's benign override, pushed before its sub-invoke re-bases the
    ancestor, still composes onto that ancestor at the first read once the base is corrected.

    **Layering semantics — why a :func:`configure` inside a :class:`ConfigOverride` block stays
    shadowed until exit.** :func:`configure` and :class:`ConfigOverride` are two different
    *kinds* of state that compose by
    *layering*, not by a single "most-recent-call-wins" timeline: :func:`configure` writes the
    **permanent base** (``_default_config``, unscoped, last-write-wins), while each active
    :class:`ConfigOverride` pushes a **scoped shadow layer** onto this stack (innermost = last =
    highest precedence). Reads fold base-then-layers, so a ``configure(x=...)`` made *inside* a
    ``with ConfigOverride(x=...)`` block does
    **not** win over that override — it updates the base *beneath* the shadow, so the override stays
    effective for the block's whole extent and the new base value surfaces only once the scope
    lifts (the empty-stack ``resolved`` fast path returns the now-updated live
    ``_default_config``). This is the one coherent way to merge a permanent single base with a
    LIFO scoped stack: letting a configure-inside punch through would force block exit to either
    silently discard that :func:`configure` or restore unpredictably, and would break the scoped
    override's guarantee that it deterministically shadows for its whole extent (including
    against a nested ``jaz.invoke``'s mutations). The description system applies the identical
    layering to :func:`jaz.describe` (permanent base) vs. :class:`jaz.DescriptionOverride`
    (scoped shadow) — see ``descriptions.DescriptionOverride``."""

    layers: tuple[_LayerBase, ...] = ()
    # True once a real invoke has established this stack as the ancestor for a subtree. A `with`
    # ConfigOverride never sets it (it only ``push``es, which carries the flag through), so
    # ``not established`` is the reliable "fresh worker thread, ContextVar didn't propagate"
    # signal the worker re-base in library/jaz.py keys on — even after the worker pushes its own
    # layer. Excluded from equality: it's re-establishment bookkeeping, not part of the value.
    established: bool = field(default=False, compare=False)

    @property
    def resolved(self) -> Config:
        """Effective (depth-less) config = ``fold(_default_config, layers)``.

        Empty-stack fast path: return the live ``_default_config`` by identity, **uncached**, so
        both ``configure()`` mutations *and* a rebind of the singleton stay visible (matches the
        pre-#727 ``get_config()``). This matters because the module-default ``ConfigStack`` is
        shared across every context/thread via the ContextVar default — caching ``_default_config``
        on it would freeze whatever the singleton was at first read (e.g. stale after a test's
        ``_reset_global_config`` rebinds it, so a worker reading the shared default stack would see
        the old config). Non-empty stacks memoize their fold (``_folded``) instead — they are
        short-lived (per ``with`` scope) and snapshot the singleton at first read, exactly as the
        old ``ConfigOverride.__enter__`` did.
        """
        if not self.layers:
            return _default_config
        return self._folded

    @cached_property
    def _folded(self) -> Config:
        """The fold for a non-empty stack: SHALLOW-copy ``_default_config`` (via ``_shallow_fold``)
        and apply each layer's diff via ``Config.update`` in push order (last-wins). ``DepthLayer``s
        contribute nothing here (empty unconditional ``overrides``) — they only apply in
        ``resolve_for_depth``.

        Aliasing (load-bearing, #741): shallow — not ``deepcopy`` (which #815 used here, replaced in
        #727) — so a non-copyable instance ``llm_client``/``interaction_protocol`` stays shared
        rather than duplicated per fold (deepcopy would raise on a lock/open socket, or lose
        connection-pool / rate-limit state). The consequence: EVERY ``get_config()`` under a
        non-empty stack — i.e. every ``with ConfigOverride(...)`` — shares its *unoverridden* mutable
        containers (``repl.configs``, ``llm.params``, and the subconfig objects themselves) BY
        REFERENCE with ``_default_config``. Leak-safe only by the "no in-place container mutation;
        every write rebinds the whole field via ``update``/``setattr``" convention: an in-place
        mutation of one of these (``cfg.repl.configs[k] = ...``, ``cfg.<removed>.append(...)``)
        would corrupt the shared global default and every other live config, not a private copy.
        Pinned by ``test_configoverride_resolved_aliases_base_mutable_containers``. (Note the
        asymmetry with ``ConfigOverride.apply_to``, which still deep-copies its mutable fields — see
        ``test_apply_to_deep_copies_mutable_fields``.)

        Cache thread-safety: one ``ConfigStack`` can be shared across worker threads (a contextvar
        copies the reference). ``cached_property`` isn't locked, so two threads may both compute and
        race on the ``__dict__`` write — but the write is a GIL-atomic dict setitem and both results
        are pure and equal, so the worst case is a redundant fold, never corruption. No lock: it
        would serialize this hot read path. (Works on a frozen dataclass because ``cached_property``
        writes ``instance.__dict__`` directly, bypassing the frozen ``__setattr__`` — so
        ``ConfigStack`` must not use ``slots``.)
        """
        return _shallow_fold(layer.depthless_partial() for layer in self.layers)

    def push(self, layer: _LayerBase) -> ConfigStack:
        """Return a new stack with ``layer`` appended (innermost = last = highest precedence)."""
        return ConfigStack(self.layers + (layer,), self.established)

    def resolve_for_depth(self, depth: int) -> Config:
        """Effective config at an ABSOLUTE recursion depth.

        One uniform fold over ``_configured_depth_layers() + self.layers`` — global per-depth base
        layers (``configure_by_depth``) first, then this stack's scoped layers — applied onto
        ``_default_config`` in order, lowest→highest precedence:

        1. ``_default_config`` base fields;
        2. the global per-depth base ``DepthLayer`` (``configure_by_depth``) at this depth — the
           lowest-precedence *override* source (a base layer, no longer a Config field);
        3. this stack's applicable layers in **push (declaration) order** — later wins. A plain
           ``Layer`` re-contributes its full diff (applies at every depth); a ``DepthLayer``
           contributes only its matching depth's partial.

        So a :class:`ConfigOverride` entered *after* a ``ConfigOverrideByDepth`` wins at that
        depth — the #727 precedence flip (was: per-depth wins, applied last). Superset of
        ``resolved``.

        Fast path: when no layer applies at this depth (no global per-depth entry and no applicable
        stack ``DepthLayer``/plain layer), the depth fold equals the depth-less ``resolved`` —
        return it directly (identity to ``_default_config`` when the stack is empty).

        Copy strategy — **shallow** (``_shallow_fold``), carried over from the removed
        ``Config.resolve_for_depth``: an instance ``llm_client``/``interaction_protocol`` may wrap a
        non-copyable resource (a lock, an open HTTP pool) that ``deepcopy`` would raise on or would
        duplicate — losing shared connection-pool / rate-limit state. Shallow copy + whole-field
        rebind keeps the one client shared and is leak-safe by the "no in-place container mutation"
        convention (#741 tracks making that structural).
        """
        layers = _configured_depth_layers() + self.layers
        applicable = [layer for layer in layers if layer.applies_at(depth)]
        if not applicable:
            return self.resolved
        return _shallow_fold(layer.partial_for(depth) for layer in applicable)


# Global per-recursion-depth config (from ``configure_by_depth``), held as a SINGLE base
# ``DepthLayer`` — NOT a field on the unconditional ``_default_config``.
# Per-depth is a *resolution-context rule*, not a property of the default value, so it lives as the
# **bottom layer** of the fold: ``ConfigStack.resolve_for_depth`` folds this first (lowest
# precedence), then the contextvar stack's scoped layers on top. ``None`` = no global per-depth
# config. Not a stack: ``configure`` / ``configure_by_depth`` eagerly compose into this one layer's
# map on every call (merge / key-strip — see those functions), so there is never more than one.
# Contributes NOTHING to the depth-less ``get_config()`` fold (a DepthLayer contributes only at a
# matching depth). (Finishes the #727 migration: scoped ``ConfigOverrideByDepth`` became a stack
# ``DepthLayer`` there; here the global sibling stops being a Config field too.)
_base_config_layer: DepthLayer | None = None


def _configured_depth_layers() -> tuple[_LayerBase, ...]:
    """The global per-depth base layer (``configure_by_depth``), folded under the override stack —
    as a 0-or-1 tuple so ``ConfigStack.resolve_for_depth`` can concatenate it with the stack."""
    return (_base_config_layer,) if _base_config_layer is not None else ()


# Context-local configuration override stack (#727). Holds an ordered stack of composable
# ``ConfigOverride`` diffs; ``get_config()`` folds it onto ``_default_config`` lazily. Replaces
# the old ``_config_override`` ContextVar that held a single pre-resolved ``Config`` snapshot.
# noqa on B039: a shared mutable ContextVar default is the footgun the rule guards against, but
# ConfigStack is a FROZEN (immutable) dataclass with an empty-tuple layer list — sharing the one
# default instance across contexts is safe, and its resolved fast-path returns the live
# _default_config uncached (see ConfigStack.resolved), so no per-context state accretes on it.
_config_stack: ContextVar[ConfigStack] = ContextVar(
    "jaz_config_stack",
    default=ConfigStack(),  # noqa: B039
)


def get_config() -> Config:
    """
    Get the current effective configuration.

    Returns:
        The current Config instance
    """
    # Folds the context-local override stack (set via ``ConfigOverride``) onto the global
    # default, or returns the global default directly when the stack is empty.
    #
    # This is the **depth-less** ambient config: it applies propagating ``ConfigOverride``
    # layers but does NOT apply any per-depth ``DepthLayer`` (``configure_by_depth`` /
    # ``ConfigOverrideByDepth``). Post-#727 the per-depth fold lives in
    # ``ConfigStack.resolve_for_depth(depth)``, driven by the invoke path.
    #
    # Do NOT read this inside execution / enforcement / hook paths to decide something about a
    # *specific* invoke — it silently diverges from the config that invoke actually ran under
    # whenever a per-depth (or per-call) override is in scope (this exact hazard caused the
    # replay divergence bug fixed in #826 §1). Read the depth-resolved config instead:
    # ``event.config`` in a hook, the prehook-resolved ``config`` in ``invoke.py``, or
    # ``resolve_for_depth`` directly. Legitimate uses are ambient/caller-frame reads with no
    # invoke depth to resolve against (e.g. constructing an ``Agent`` outside any invoke, or
    # reading the caller's ancestor policy at a public entry point). See #826 / #531.
    return _config_stack.get().resolved


def get_config_stack() -> ConfigStack:
    """Return the live ``_config_stack`` ContextVar value (the ordered override stack).

    The agent-facing synthesized :func:`invoke` (see ``get_invoke_tool`` in _invoke_tool.py) reads
    this to re-establish propagated config across a raw worker-thread boundary: it composes the
    closure-threaded ancestor stack with any layers the worker pushed locally. ``stack.established``
    distinguishes a fresh worker (ContextVar didn't propagate → needs re-base) from a same-thread
    subtree (already carries the ancestor). Replaces the old ``get_active_config_override()`` (which
    returned a resolved ``Config | None``), matching the #727 stack model.
    """
    return _config_stack.get()


def _set_config_stack(stack: ConfigStack) -> Token[ConfigStack]:
    """Set the ``_config_stack`` ContextVar (internal; returns a reset token).

    Used by ``ConfigOverride.__enter__`` to push a layer, and by the agent-facing synthesized
    ``invoke`` to *re-base* the threaded ancestor stack onto a raw worker thread (where the
    ContextVar didn't propagate). Reset with ``_reset_config_stack``.
    """
    return _config_stack.set(stack)


def _reset_config_stack(token: Token[ConfigStack]) -> None:
    """Reset the ``_config_stack`` ContextVar from a ``_set_config_stack`` token."""
    _config_stack.reset(token)


# Sentinel default for the explicit configure()/ConfigOverride signatures. Config.update keys off
# argument *presence* — naming a field replaces it, omitting it leaves it untouched — and None/False
# are legitimate values for some fields, so "omitted" cannot be spelled None. Typed Any so it
# satisfies every parameter's declared type as a default.
_UNSET: Any = object()


def _configured_fields(**maybe_unset: Any) -> dict[str, Any]:
    """Keep only the fields the caller actually passed (drop ``_UNSET``).

    Preserves ``Config.update``'s presence semantics under an explicit signature: a field is
    applied only when named, never merely because it carries a default.
    """
    return {name: value for name, value in maybe_unset.items() if value is not _UNSET}


def configure(
    *,
    llm: BaseLLM = _UNSET,
    repl: BaseREPL = _UNSET,
    protocol: BaseProtocol = _UNSET,
) -> None:
    """
    Configure the global default JAZ framework settings.

    This modifies the global default configuration that is used when no
    context-local override is active.

    Settings are grouped by component, plus cross-cutting top-level options. Each group takes
    the **configured component itself**, whose constructor is the schema for that group's
    settings.

    **Setting a group replaces it.** A group names the whole component, so you restate it
    completely rather than adjusting one of its settings in place.

    Args:
        llm: How JAZ talks to the model — a :class:`BaseLLM` such as
            ``LiteLLM(model="openai/gpt-5-mini", temperature=0.7)``. ``litellm`` is the sole
            built-in backend and the default; ``@register_llm`` adds your own. The constructor is the whole
            surface: the ``model`` id, the backend's own settings (``base_url``, ``api_key``),
            the retry policy (``max_retries``, ``retry_wait_multiplier``,
            ``retry_wait_min``, ``retry_wait_max``, ``retry_policy``), and any per-request
            default (``temperature``, ``max_tokens``, ``reasoning_effort``, ...) it passes
            through to the provider API. Anything the constructor does not name becomes a
            per-request default, so a backend gains a setting simply by declaring it.
        repl: The REPL execution layer — a :class:`BaseREPL` such as
            ``PythonREPL(exec_timeout=60)``. Its constructor takes ``exec_timeout``,
            ``exec_memory_limit`` (process-RSS ceiling in bytes, enforced independently of
            ``exec_timeout``), and the sandbox allow-lists.
        protocol: The LLM↔REPL codec — a :class:`BaseProtocol` such
            as ``CodeOnlyProtocol(max_invoke_input_length=20000)``. Its constructor takes the
            ``*_template`` prompt-template paths, ``max_invoke_input_length``, ``max_repl_output_length``
            and ``truncation_prefix_ratio``.

    Examples:
        jaz.configure(repl=PythonREPL(exec_timeout=60))
        jaz.configure(llm=LiteLLM(model="openai/gpt-5-mini", temperature=0.7))

    Interaction with :func:`configure_by_depth`: an explicit ``configure`` of a field **overrides**
    any prior per-depth override of that *same* field — the touched keys are stripped from every
    per-depth partial so the new global value wins at all depths (the intuitive "I just set this
    globally" behavior). Per-depth overrides of *other* fields are untouched. (A later
    ``configure_by_depth`` can re-establish a per-depth override.)
    """
    updates = _configured_fields(
        llm=llm,
        repl=repl,
        protocol=protocol,
    )
    _default_config.update(**updates)
    _strip_base_depth_layer_keys(updates.keys())


def _strip_base_depth_layer_keys(keys: Iterable[str]) -> None:
    """Remove ``keys`` from every partial in the global per-depth config (``configure`` calls this
    so an explicit global set wins over stale per-depth overrides of the same field). A depth whose
    partial becomes empty is dropped; if all do, the per-depth config is cleared."""
    global _base_config_layer
    if _base_config_layer is None:
        return
    strip = set(keys)
    new_map: dict[int, Mapping[str, Any]] = {}
    for depth, partial in _base_config_layer.depth_map.items():
        remaining = {k: v for k, v in partial.items() if k not in strip}
        if remaining:
            new_map[depth] = MappingProxyType(remaining)
    _base_config_layer = (
        DepthLayer(depth_map=MappingProxyType(new_map)) if new_map else None
    )


def configure_by_depth(
    overrides_by_depth: dict[int, dict[str, Any]] | None,
) -> None:
    """Set per-depth Config overrides on the global default — the by-depth sibling of
    :func:`configure`.

    ``overrides_by_depth`` maps **absolute recursion depth** (the top invoke is depth 1,
    its first nested invoke depth 2, ...) to a *partial Config* — a mapping of Config-field ->
    value, with each group holding the configured component, exactly as :func:`configure` takes
    it. Each partial is applied over the base Config at that depth via the same validated
    :meth:`Config.update` path as :class:`ConfigOverride`, so there is one
    overriding/validation semantics. A depth with **no** entry uses the base Config
    unchanged — no extrapolation from another depth.

    Each partial holds configured components, exactly as :func:`configure` takes them.

    Excluded field (rejected): ``depth_overrides`` itself (self-referential).

    Like :func:`configure`, this mutates the process-wide default and **composes** with prior
    calls — **field by field within each depth**: for a re-specified depth the new partial is
    merged into that depth's existing one (same key → new value wins, *other keys kept*); new
    depths are added; unmentioned depths are kept. So ``configure_by_depth({2: {a}})`` then
    ``({2: {b}})`` leaves depth 2 with **both** ``a`` and ``b`` (just as ``configure(a=1);
    configure(b=2)`` accumulates fields). Reset all per-depth config with
    ``configure_by_depth(None)``. Interaction with :func:`configure`: a later ``configure(field=…)``
    **strips** ``field`` from every per-depth partial (the global set wins at all depths) — see
    :func:`configure`. For a dynamically-scoped or invoke-local override, use
    :class:`ConfigOverrideByDepth`.

    Stored as a single base ``DepthLayer`` in ``_base_config_layer`` (the bottom of the resolve fold),
    NOT as a field on ``_default_config`` — per-depth is a resolution-context rule, not a property
    of the unconditional default.

    Examples:
        jaz.configure_by_depth({
            1: {"llm": LiteLLM(model="openai/gpt-5")},         # top invoke
            2: {"llm": LiteLLM(model="openai/gpt-5-mini"),     # its sub-invokes
                "repl": PythonREPL(exec_timeout=10.0)},
        })

        })
    """
    global _base_config_layer
    # ``None`` RESETS all global per-depth config — the explicit clear (there is no other way to
    # drop entries under the compose semantics below).
    if overrides_by_depth is None:
        _base_config_layer = None
        return
    # Validate/normalize through the same path ConfigOverrideByDepth uses (str->int keys, excluded
    # + unknown fields, fail-fast value checks).
    normalized = _default_config._validate_depth_overrides(overrides_by_depth) or {}
    # COMPOSE onto any existing per-depth config, FIELD-level within each depth: for a re-specified
    # depth, `{**old_partial, **new_partial}` (same key -> new value wins, other keys kept); new
    # depths are added; unmentioned depths are kept. So configure_by_depth({2: {a}}) then ({2: {b}})
    # leaves depth 2 with BOTH a and b. (An empty map is a no-op; `None` above resets;
    # `configure(field=…)` strips that field from every partial — see configure.)
    existing: Mapping[int, Mapping[str, Any]] = (
        _base_config_layer.depth_map if _base_config_layer is not None else {}
    )
    merged: dict[int, dict[str, Any]] = {
        depth: dict(partial) for depth, partial in existing.items()
    }
    for depth, partial in normalized.items():
        merged.setdefault(depth, {}).update(partial)
    if not merged:
        _base_config_layer = None
        return
    depth_map = MappingProxyType(
        {depth: MappingProxyType(dict(partial)) for depth, partial in merged.items()}
    )
    _base_config_layer = DepthLayer(depth_map=depth_map)


class ConfigOverride:
    """Temporary ``Config`` overrides, usable **two ways** — exactly mirroring how a hook is
    used either as a ``with`` block or a positional hook argument.

    - **As a context manager** (``with ConfigOverride(...):``) it is *dynamically scoped*:
      the overrides apply for the duration of the block and **propagate** to every nested
      ``jaz.invoke()`` in it.
    - **Passed positionally to ``jaz.invoke()``** it is *local* (a "shallow" override): it
      applies to that one invoke only and is **not** propagated to sub-invokes spawned inside
      it — those see the un-overridden ambient config.

    Also like a hook, a single instance is active in **at most one ``with`` scope at a time**:
    re-entering it while already active — nested (``with co: with co:``) or concurrently from
    another thread — raises :class:`ConfigOverrideActivationError`. The usual ``with
    ConfigOverride(...):`` constructs a fresh instance and is unaffected; sequential reuse of
    one instance is fine.

    Contrast with ``jaz.configure(...)``, which sets the process-wide global default.

    Takes the same explicit keyword-only fields as :func:`configure` (``llm`` / ``repl`` /
    ``protocol``); an unknown field is a ``TypeError`` at construction. Field *values* are
    validated by ``Config.update`` when the override is applied.

    Examples:
        # Dynamically scoped — propagates to nested invokes:
        with ConfigOverride(protocol=CodeOnlyProtocol(max_repl_output_length=20000)):
            invoke(...)

        # Local to one invoke — passed POSITIONALLY. A nested invoke the agent makes uses the
        # ambient config, not the override:
        invoke(
            ConfigOverride(protocol=CodeOnlyProtocol(max_repl_output_length=5000)),
            task="task",
        )
    """

    # Why single-activation is an error rather than a no-op: the scoped form is a ContextVar
    # push/reset pair, so overlapping activations of ONE instance would either stack silently or
    # surface as an opaque cross-thread ContextVar reset error. Raising is stricter than strictly
    # necessary — a ConfigOverride has no setup()/teardown(), so an overlap would in fact be
    # harmless — but it keeps the activation model byte-for-byte identical to a hook's, which is
    # the property the two-ways-to-use framing rests on.

    def __init__(
        self,
        *,
        llm: BaseLLM = _UNSET,
        repl: BaseREPL = _UNSET,
        protocol: BaseProtocol = _UNSET,
    ) -> None:
        # Field *values* are still validated lazily by ``Config.update`` when the override is
        # applied — ``apply_to`` for the local form, ``__enter__`` for the scoped form. An
        # *unknown* field is now a ``TypeError`` at construction (the explicit signature, #993)
        # rather than a deferred ``ValueError`` — that eager, statically-checkable rejection is
        # the point of the explicit surface.
        # Exposed read-only (a MappingProxyType view): a ConfigOverride is a value — its
        # override set is fixed at construction.
        self.overrides: Mapping[str, Any] = MappingProxyType(
            _configured_fields(
                llm=llm,
                repl=repl,
                protocol=protocol,
            )
        )
        # A single reset token, non-None only while this instance is active as a `with`
        # block. Entering the *same* instance while it is already active (nested on one
        # thread, or concurrently from another) is forbidden — see `__enter__`. We keep one
        # token rather than a stack because overlapping same-instance reuse is now an error,
        # not a supported case; sequential reuse clears the token on exit and re-enters fine.
        self._token: Token[ConfigStack] | None = None

    def _as_layer(self) -> _LayerBase:
        """This override as a composable stack layer (#727). Plain overrides → a ``Layer``
        carrying the raw diff; ``ConfigOverrideByDepth`` returns a ``DepthLayer`` instead."""
        return Layer(self.overrides)

    def apply_to(self, config: Config) -> Config:
        """Return a SHALLOW copy of ``config`` with these overrides applied.

        The input ``config`` is never *rebound* — ``update`` replaces whole fields via ``setattr``,
        never mutating a container in place — so a shallow copy is leak-safe by the same "no in-place
        container mutation" convention the stack fold relies on.
        Consistent with ``ConfigStack._shallow_fold`` (they were reconciled — both shallow — so
        there is one copy semantics for "resolve overrides onto a base"); the returned config's
        *unoverridden* mutable fields (``repl.configs``/``llm.params``/…) therefore stay shared
        by identity with ``config``. Shallow also avoids ``deepcopy`` raising on / duplicating a
        non-copyable instance ``llm_client``/``interaction_protocol`` (a lock, an open HTTP pool).
        """
        new_config = copy.copy(config)
        if self.overrides:
            new_config.update(**self.overrides)
        return new_config

    def __enter__(self) -> None:
        """Push these overrides as a layer on the context stack for the duration of the block.

        Appends a composable ``Layer`` to the ``_config_stack`` — so the diff
        **propagates** to every nested ``jaz.invoke()`` in the block and composes, last-wins,
        with any ancestor/local layers. Returns ``None``: the effective config is
        depth-dependent and only resolves at an invoke's recursion depth
        (``ConfigStack.resolve_for_depth``), so there is no meaningful depth-less ``Config`` to
        hand back here — a ``ConfigOverrideByDepth`` block especially would return a config with
        its per-depth partials silently absent, which is worse than returning nothing.

        This does **not** eagerly snapshot ``get_config()`` into a resolved ``Config``; the
        diff is folded lazily at read time (see ``ConfigStack.resolved``). The visible
        consequence: a ``configure()`` that mutates the global default *between* this push
        and the block's first config read **is visible inside the block** — the fold happens
        at read time, not at enter time. This laziness is what lets a raw worker's benign
        override compose onto a later-rebased ancestor rather than baking in a sandbox-less
        snapshot.

        Raises ``ConfigOverrideActivationError`` if this exact instance is already active
        as a ``with`` block — a nested re-entry (``with co: with co:``) or a concurrent
        entry from another thread. This mirrors ``Hook``'s activation guard: a single
        instance lives in at most one active scope at a time. The guard is raised before
        the token is taken, so a rejected entry leaves the instance untouched. Distinct
        instances nest freely, and sequential (non-overlapping) reuse of one instance is
        fine — ``__exit__`` clears the token.
        """
        if self._token is not None:
            raise ConfigOverrideActivationError(
                f"{type(self).__name__} instance is already active as a `with` block; "
                "re-entering the same instance (nested, or concurrently from another "
                "thread) is not allowed. Construct a fresh instance per scope."
            )
        # Validate field names/values now (fail-fast at the call site), exactly as the pre-#727
        # eager path did — but without applying: we install the raw diff *layer*, not a resolved
        # snapshot, so the fold stays lazy (see class docstring). ``Config.validate_update`` is a
        # schema-level classmethod (unknown fields, bad values, excluded per-depth keys) — it does
        # NOT read or fold the current stack, so nothing is mutated and the ContextVar is untouched.
        Config.validate_update(self.overrides)
        self._token = _set_config_stack(get_config_stack().push(self._as_layer()))
        # No return value: a depth-less effective config is meaningless here (per-depth layers
        # only resolve at an invoke depth), so `with ConfigOverride(...)` binds nothing.
        return None

    def __exit__(self, *exc_info: object) -> None:
        # A correct `with` pairs one __enter__ (which sets `_token`) with one __exit__, so
        # `_token` is set here. If it is None, __exit__ ran without a live __enter__ — a
        # manual/mismatched call or a double-exit — which is a bug we surface loudly rather
        # than silently no-op'ing. (Unlike a Hook, a ConfigOverride has no plain-instance /
        # manual-teardown lifecycle — it is `with`-only — so there is no legitimate
        # exit-without-enter to tolerate.) A rejected re-entry can't reach here: __enter__
        # raises before taking a token, so the language never calls its __exit__.
        if self._token is None:
            raise RuntimeError(
                "ConfigOverride.__exit__ called with no active override: it was exited "
                "without a matching __enter__, or exited more than once. A ConfigOverride "
                "is entered and exited exactly once, via a `with` block."
            )
        _reset_config_stack(self._token)
        self._token = None

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.overrides.items())
        return f"ConfigOverride({inner})"


class ConfigOverrideByDepth(ConfigOverride):
    """Per-depth Config overrides — the by-depth sibling of :class:`ConfigOverride`, and the
    scoped/local counterpart of :func:`configure_by_depth`.

    Takes a mapping of **absolute recursion depth** (top invoke = depth 1) to a *partial
    Config*, and pushes it as a depth-conditional ``DepthLayer`` on the override stack:

    - **As a context manager** (``with ConfigOverrideByDepth({...}):``) it is dynamically
      scoped and **propagates** to nested ``jaz.invoke()`` calls, which resolve their own
      depth's partial as they recurse. This is the ONLY supported way to use it.
    - **Passed positionally to ``jaz.invoke()``** it is REJECTED (``TypeError``, in
      ``invoke._extract_config_override``): a local override reaches only that one invoke, where
      just its own-depth partial could apply — a degenerate depth-gated plain override — so
      per-depth config, which is meaningful only when it propagates across depths, must use the
      ``with`` form. (A plain :class:`ConfigOverride` may still be passed positionally.)

    Same partial surface as :func:`configure_by_depth`: ``depth_overrides`` is rejected inside a
    partial (self-referential). Keys are absolute depth — one mental model whether set globally
    (``configure_by_depth``) or scoped here.

    Unlike a plain :class:`ConfigOverride`, its diff is a *depth map*, not Config fields — so it
    carries the map separately (``_raw_depth_map``) rather than in the base ``overrides`` (which
    stays empty). Per-depth config is not a Config field, so there is nothing for the base
    ``apply_to`` to set. The map is validated *lazily in* ``_as_layer`` (called by ``__enter__`` and
    by the local-invoke fold), matching the base's validate-on-use — so a bad partial still fails
    fast when the override is entered/used, while ``repr`` of an unentered instance stays cheap.

    Precedence: if a live :class:`ConfigOverride` and a per-depth override set the
    *same* field, precedence follows **declaration nesting** — an inner :class:`ConfigOverride`
    entered *after* a ``ConfigOverrideByDepth`` wins at that depth (the per-depth ``DepthLayer`` is
    folded in push order like any other layer; the global ``configure_by_depth`` base layer is the
    lowest-precedence per-depth source).

    Examples:
        with ConfigOverrideByDepth({2: {"repl": PythonREPL(exec_timeout=10.0)}}):
            invoke(...)   # depth-2 sub-invokes get the 10s timeout; others unchanged
    """

    def __init__(self, overrides_by_depth: dict[int, dict[str, Any]] | None) -> None:
        # Base ``overrides`` stays EMPTY: a per-depth map is not a set of Config fields, so the
        # inherited apply_to/__enter__ field-validation path is a no-op for us. Carry the RAW depth
        # map separately; validation/normalization happens lazily in ``_as_layer`` (on use).
        super().__init__()
        self._raw_depth_map = overrides_by_depth

    def _as_layer(self) -> _LayerBase:
        # A depth-conditional layer: its {depth: partial} map contributes only at a matching
        # recursion depth (see ConfigStack.resolve_for_depth), and nothing to the depth-less
        # get_config() fold. Validate + normalize here (lazy, on use) so entering the override or
        # folding it for a local invoke fails fast on a bad partial, matching ConfigOverride.
        normalized = (
            _default_config._validate_depth_overrides(self._raw_depth_map) or {}
        )
        return DepthLayer(
            depth_map=MappingProxyType(
                {d: MappingProxyType(dict(p)) for d, p in normalized.items()}
            )
        )

    def __repr__(self) -> str:
        return f"ConfigOverrideByDepth({self._raw_depth_map!r})"
