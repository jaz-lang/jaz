from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from functools import cached_property
from types import MappingProxyType
from typing import Any

from jaz._sentinel import UNSET, UnsetType
from jaz.exceptions import ConfigOverrideActivationError

from ._instantiate import unknown_option_error
from .llm import LiteLLM
from .llm._base import BaseLLM
from .protocol._base import BaseProtocol
from .protocol._code_only import CodeOnlyProtocol
from .repl._base import BaseREPL
from .repl._python_repl import PythonREPL

# REPL sandbox policy (#688) is the `PythonREPL` constructor's allow-list kwargs (spelled
# ``repl.params`` in a settings file — see `_user_settings`). Every axis is a uniform
# gitignore-style **allow-list** (no denylists, no "unrestricted"/None): ``allowed_imports`` /
# ``allowed_attributes`` (compiler-enforced) plus the file-access allow-lists ``allowed_read_paths``
# / ``allowed_write_paths`` (enforced by secure_open). The name surface is policed separately by the
# builtins allowlist installed as ``__builtins__`` (``repl/_permissions.py``), not by a config key.
# All defaults are secure-by-default and live next to enforcement in ``repl/_python_repl.py``:
# ``allowed_imports`` and the file paths default to ``[]`` (deny all), ``allowed_attributes`` to
# ``DEFAULT_ALLOWED_ATTRIBUTES`` (allow all except dunders and the frame-bearing interpreter
# surface); "allow all" is always the explicit pattern (``["*"]`` for names, ``["//**"]``
# for the whole filesystem — file anchors are ``//``=fs, ``~/``=home, ``/``=cwd-top, bare=cwd).
# None of it is un-strippable through config: no config key is protected, and a host that
# hands the agent a ``ConfigOverride`` surface opts into letting it reconfigure the sandbox for its
# sub-invokes — the executive call and its reasoning are recorded in ``repl/_compiler.py``.


# The component each group holds, and how a default one is built. THE group registry: every
# consumer derives from this one mapping rather than restating the names, so a fourth group
# cannot be added in one place and silently skipped in another.
#
# Config stores the *configured components themselves* — a `BaseLLM`, a `BaseREPL`, a
# `BaseProtocol` — not a `{tag, params}` description of them. Why is recorded on `Config`.
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


def _default_classes() -> dict[str, type]:
    """The class each group defaults to when a spec names none.

    Distinct from :func:`_default_components`: the LLM *class* has a default (``LiteLLM``, the
    default backend) even though the LLM *component* does not, because a backend cannot be built
    without a model.
    """
    # `LiteLLM` is the default because it reaches every provider LiteLLM routes and prices through
    # LiteLLM's maintained table, so a new model or vendor needs no JAZ release; the other shipped
    # backends cover only what it does not serve (see `jaz.llm`).
    return {"llm": LiteLLM, "repl": PythonREPL, "protocol": CodeOnlyProtocol}


#: How to build the default component for each group that has one.
#:
#: PythonREPL/CodeOnlyProtocol are complete configurations — every setting has a canonical value —
#: so they are real defaults. An LLM is not: it has no default model (#1086), which is why there is
#: no ``llm`` entry here and why a Config cannot be built without one.
#:
#: Factories rather than instances, because ``Config(...)`` builds a FRESH default for each group
#: the caller omits. A component's *containers* are frozen and exposed read-only (#741), but its
#: scalars are ordinary attributes — ``cfg.repl.exec_timeout = 99`` is a plain write — so two
#: independently built configs sharing one component would let a write through either reach the
#: other.
_DEFAULT_COMPONENT_FACTORIES: Mapping[str, Callable[[], Any]] = MappingProxyType(
    {
        "repl": PythonREPL,
        "protocol": CodeOnlyProtocol,
    }
)

#: The default component the AMBIENT state falls back to, built once at import.
#:
#: Shared, unlike a constructed config's. Every fold would otherwise build a new REPL, so two reads
#: of the same unconfigured state would disagree (``get_config().repl is not get_config().repl``)
#: and an invoke's template would change under it.
#:
#: KNOWN LIMITATION: because the ambient default is one object, a write through
#: ``get_config().repl`` reaches the next read. Rebinding a whole group is what ``configure`` /
#: ``ConfigOverride`` are for; writing *into* a component is unsupported.
_AMBIENT_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {group: build() for group, build in _DEFAULT_COMPONENT_FACTORIES.items()}
)


def _default_components() -> Mapping[str, Any]:
    """The component each group falls back to in the ambient state; ``llm`` has none."""
    return _AMBIENT_DEFAULTS


@dataclass(frozen=True, init=False)
class Config:
    """JAZ framework configuration, grouped by architectural component.

    - ``llm``      (:class:`BaseLLM`) — how JAZ talks to the LLM. There is no default model,
      so this one has no default and must be named.
    - ``repl``     (:class:`BaseREPL`) — how agent code executes.
    - ``protocol`` (:class:`BaseProtocol`) — the LLM↔REPL codec
      (prompts + parsing).

    Those three are the whole surface; there is no fourth, top-level group, and naming an
    unknown keyword raises.

    The surface is nested-only — there are no flat ``config.<option>`` aliases. Set via the
    grouped :func:`configure`/:class:`ConfigOverride` shape, passing the configured component:
    ``configure(repl=PythonREPL(exec_timeout=30))``. Configuration written as *data* — a
    settings file, an eval YAML — is compiled into components before it reaches here, so config
    itself only ever holds the built object.

    **Setting a group replaces it.** A group names the whole component, so there is no partial
    update of one component's settings."""

    # Config holds the built component, not a `{tag, params}` description of it, for three reasons:
    #
    # * **Defaults have somewhere to live.** A bag either materialises copies of the component's
    #   defaults (which then drift — `repl.exec_timeout` saying 30 s while the constructor says
    #   unbounded) or omits them, making "what is in force?" unanswerable from the config.
    # * **Partial merge across a tag is unsound.** `{tag, params}` is a dependent record — the
    #   valid shape of `params` depends on the *value* of `tag` — so merging the two as
    #   independent leaves lets one layer set the tag and another the params, folding to a
    #   combination nobody wrote. Layers *replace* per group, the only coherent rule for a tagged
    #   union.
    # * **Typing reaches the leaves.** `params: dict[str, Any]` cannot be checked; a constructor
    #   call can.

    llm: BaseLLM
    repl: BaseREPL[Any]
    protocol: BaseProtocol

    def __init__(
        self,
        *,
        llm: BaseLLM,
        repl: BaseREPL[Any] | UnsetType = UNSET,
        protocol: BaseProtocol | UnsetType = UNSET,
    ) -> None:
        """Construct a Config from the grouped shape, passing the configured component.

        ``llm`` must be named: there is no default model, so there is no config without one.
        ``repl`` and ``protocol`` fall back to their defaults when omitted.

        Explicit arguments are the only source of configuration: there is no ambient layer
        underneath them, so a Config built here is fully determined by its call site.
        """
        # An explicit keyword-only signature rather than a ``**kwargs`` bag, so omitting the
        # backend or misspelling a group is refused by the type checker rather than at runtime.
        # The same call #993 made for `configure` and `ConfigOverride`: an open bag lets `Config()`
        # and `Config(lm=...)` type-check clean and fail only when run (executive call, user,
        # 2026-09-18, on review of #1483).
        #
        # It is also why the "nothing is configured" refusal is NOT here: a call that omits a
        # required argument is a programming error, and `TypeError` says so precisely, whereas
        # "no LLM is configured" is a statement about the *ambient* state. That refusal lives at
        # the fold (`_fold_config`), the only place the ambient state is read.
        named = _configured_fields(llm=llm, repl=repl, protocol=protocol)
        # Validate before assigning, so a rejected argument leaves no half-built config behind.
        # The signature has already rejected an unknown *name*; this is the component-type check.
        self._validate_update(named)
        # A fresh default per omitted group, not the shared ambient one: two configs built here
        # are independent objects, and a write into one's REPL must not reach the other's.
        settings = dict(named)
        for group, build in _DEFAULT_COMPONENT_FACTORIES.items():
            if group not in settings:
                settings[group] = build()
        # `object.__setattr__` because the class is frozen: a config is a resolved value, and a
        # hook holding one through `event.config` must not be able to repoint the invoke's
        # components by assigning to it.
        for group, component in settings.items():
            object.__setattr__(self, group, component)

    @classmethod
    def _validate_update(cls, updates: Mapping[str, Any]) -> None:
        """Validate a grouped override map without applying it (non-mutating).

        ``ConfigOverride.__enter__`` installs a raw diff as a lazy stack layer and has no
        resolved Config, so it fail-fasts here instead of folding.
        """
        # The only validation path there is: `__init__` calls this, so "validate without applying"
        # and "validate then apply" cannot disagree.
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
        # A dict or a bare tag string is the one likely mistake here: it is the shape authored
        # data (a settings file, an eval YAML) takes before it is compiled into a component. Point
        # at where that data goes rather than only naming the offending type — "must be a BaseREPL,
        # got dict" alone leaves the caller with no way to find the right surface.
        if isinstance(value, Mapping | str):
            raise ValueError(
                f"config option {group!r} must be a {expected.__name__}, got "
                f"{type(value).__name__}. Config holds configured components — build one: "
                f"{_COMPONENT_HINT[group]}"
            )
        if value is None:
            # `None` is never something a caller sets: to leave a group unchanged, omit it (the
            # default is `UNSET`). It is not the unconfigured state either — a config holds a
            # component in every group or does not exist.
            raise ValueError(
                f"config option {group!r} must be a {expected.__name__}, got None. To leave "
                f"it unchanged, omit it."
            )
        raise ValueError(
            f"config option {group!r} must be a {expected.__name__}, got {type(value).__name__}"
        )


# The authored global base: what `configure()` has written, and nothing else.
#
# NOT a `Config`. A Config is complete by construction, and "nothing is configured yet" is
# precisely the incomplete state, so the base is a partial mapping — the same shape the override
# layers already speak in. The defaults for `repl`/`protocol` and the refusal for a missing `llm`
# are applied once, where the fold builds a Config. The alternative — a `Config()` base whose
# `llm` is `None` — forces every reader to carry the optional (executive call, user, 2026-09-17).
_base_settings: dict[str, Any] = {}


@dataclass(frozen=True)
class Layer:
    """One :class:`ConfigOverride`'s raw diff.

    Carries only the field→value assignments the override requested
    (``ConfigOverride.overrides``), never a resolved ``Config``.
    """

    # Part of the layered stack model (#727). Carrying the raw diff rather than a resolved
    # ``Config`` is what makes a layer **base-independent**: folding one onto any base is just
    # merging those assignments over it (last-wins), so a stack can be re-based onto a
    # different ancestor after the fact — the property the worker-thread re-base relies on.

    overrides: Mapping[str, Any]


def _fold_settings(partials: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The authored settings: the global base with ``partials`` applied in order, last-wins.

    May be incomplete — ``llm`` is absent until something configures one. Building a
    :class:`Config` from the result is what applies the component defaults and refuses a
    missing LLM.
    """
    # A plain mapping merge, because a layer's diff is already a mapping: there is no base
    # *object* to copy any more, so the "shallow copy, then replay setattrs" dance is gone with
    # it. Components are carried by reference, which is what keeps a non-copyable `llm` /
    # `protocol` (one wrapping a lock or an open HTTP pool) shared rather than duplicated per
    # fold — deepcopy would raise on it, or silently lose connection-pool / rate-limit state.
    # Sharing a component is NOT a guarantee that nothing can write through it: #741 froze each
    # component's containers, not its scalars, so `cfg.repl.exec_timeout = 99` is an ordinary
    # attribute write that every holder of that component sees. See `_AMBIENT_DEFAULTS` for the
    # span over which that is accepted, and why a directly built `Config` is kept out of it.
    settings = dict(_base_settings)
    for partial in partials:
        settings.update(partial)
    return settings


def _fold_config(partials: Iterable[Mapping[str, Any]]) -> Config:
    """:func:`_fold_settings`, resolved into a usable :class:`Config`.

    Raises ``ValueError`` when no LLM is configured — this is the single resolution point, so
    it is the single place that refusal can come from.
    """
    settings = _fold_settings(partials)
    if "llm" not in settings:
        # The DSPy "No LM is loaded" moment: nothing is configured, so nothing can be called.
        #
        # Here rather than in ``Config.__init__`` because this is where the *ambient* state is
        # read, and "no LLM is configured" is a statement about that state. A direct ``Config()``
        # is a call missing a required argument, which its signature reports as a ``TypeError``
        # naming ``llm`` — the precise error for that mistake, and one a type checker catches
        # before it runs.
        #
        # The "No LLM is configured" prefix is load-bearing: the console matches it by substring
        # to print its setup hint (see `_llm_setup_hint` in `_console.py`), there being no
        # narrower shape to match on a plain ValueError. Reword the tail freely; keep the prefix,
        # or the hint silently stops appearing. Guarded by
        # `test_missing_model_error_yields_the_hint`.
        raise ValueError(f"No LLM is configured. Set one: {_COMPONENT_HINT['llm']}")
    # The AMBIENT defaults, passed explicitly, rather than the fresh ones `Config` would build:
    # every ambient read resolves to the same component for a group nobody configured.
    return Config(**{**_AMBIENT_DEFAULTS, **settings})


def _authored_settings() -> dict[str, Any]:
    """What is configured right now, with the defaults that would apply; ``llm`` may be absent.

    The console's view of the ambient state. It asks "is an LLM configured at all?", "which
    backend am I on?" and "what is in force?" *before* anything is configured — which is exactly
    when there is no :class:`Config` to hand it, because :func:`get_config` refuses. Those are
    questions about what has been authored, not about the config an invoke runs under, so they
    are answered from the settings rather than from a config that cannot exist yet.
    """
    settings = _fold_settings(layer.overrides for layer in _config_stack.get().layers)
    return {**_AMBIENT_DEFAULTS, **settings}


@dataclass(frozen=True)
class ConfigStack:
    """An ordered stack of propagating :class:`ConfigOverride` diffs relative to the authored
    ``_base_settings`` — the runtime value of the ``_config_stack`` ContextVar.

    Stores raw *ordered diffs* rather than a flattened ``Config``, so a raw
    ``ThreadPoolExecutor`` worker can reconstruct propagated state by **composition** —
    ``ancestor.layers + local.layers``, last-wins. The base is implicitly the live
    ``_base_settings`` (so ``configure()`` writes stay visible with no rebuild); everything an
    operator/agent adds lives in ``layers``.

    The fold is **lazy** (see ``resolved``): a diff is only applied when the effective config is
    *read*, not when the :class:`ConfigOverride` is entered.

    **Layering semantics — why a :func:`configure` inside a :class:`ConfigOverride` block stays
    shadowed until exit.** :func:`configure` and :class:`ConfigOverride` are two different
    *kinds* of state that compose by
    *layering*, not by a single "most-recent-call-wins" timeline: :func:`configure` writes the
    **permanent base** (``_base_settings``, unscoped, last-write-wins), while each active
    :class:`ConfigOverride` pushes a **scoped shadow layer** onto this stack (innermost = last =
    highest precedence). Reads fold base-then-layers, so a ``configure(x=...)`` made *inside* a
    ``with ConfigOverride(x=...)`` block does
    **not** win over that override — it updates the base *beneath* the shadow, so the override stays
    effective for the block's whole extent and the new base value surfaces only once the scope
    lifts (the empty-stack ``resolved`` path folds the now-updated live
    ``_base_settings``). This is the one coherent way to merge a permanent single base with a
    LIFO scoped stack."""

    # The stack model is #727. Raw ordered diffs rather than a resolved-``Config`` snapshot on the
    # ContextVar, because a snapshot's composition is an either/or that picks one side and silently
    # drops the other. The laziness above is what closes the sandbox-drop hazard — a worker's benign
    # override, pushed before its sub-invoke re-bases the ancestor, still composes onto that
    # ancestor at the first read once the base is corrected.
    #
    # The alternatives-considered half of the layering paragraph: letting a ``configure`` inside an
    # override punch through would force block exit to either silently discard that ``configure``
    # or restore unpredictably, and would break the scoped override's guarantee that it
    # deterministically shadows for its whole extent (including against a nested ``jaz.invoke``'s
    # mutations).

    layers: tuple[Layer, ...] = ()
    # True once a real invoke has established this stack as the ancestor for a subtree. A `with`
    # ConfigOverride never sets it (it only ``push``es, which carries the flag through), so ``not
    # established`` is the reliable "fresh worker thread, ContextVar didn't propagate" signal the
    # worker re-base in ``_subinvoke.py`` keys on — even after the worker pushes its own layer.
    # Excluded from equality: it's re-establishment bookkeeping, not part of the value.
    established: bool = field(default=False, compare=False)

    @property
    def resolved(self) -> Config:
        """Effective config = ``fold(_base_settings, layers)``.

        Empty-stack path: fold the live ``_base_settings``, **uncached**, so a later
        ``configure()`` stays visible. Non-empty stacks memoize their fold instead, snapshotting
        the base at first read.

        Raises ``ValueError`` when no LLM is configured (see :class:`Config`).
        """
        # Uncached on the empty path because the module-default ``ConfigStack`` is shared across
        # every context/thread via the ContextVar default: caching here would freeze whatever was
        # configured at first read (e.g. stale after a test's ``_reset_global_config`` clears the
        # base, so a worker reading the shared default stack would see the old settings). A
        # non-empty stack is short-lived (per ``with`` scope), so memoizing its fold is safe.
        if not self.layers:
            return _fold_config(())
        return self._folded

    @cached_property
    def _folded(self) -> Config:
        """The fold for a non-empty stack: merge ``_base_settings`` with each layer's diff in
        push order (last-wins), then build one :class:`Config`.

        EVERY ``get_config()`` under a non-empty stack — i.e. every ``with ConfigOverride(...)``
        — shares its *unoverridden* component objects (``llm``/``repl``/``protocol``) BY
        REFERENCE with the authored base.
        """
        # Aliasing (load-bearing): components are carried by reference, not ``deepcopy``d, so a
        # non-copyable ``llm``/``protocol`` component stays shared rather than duplicated per fold
        # — deepcopy would raise on a lock/open socket, or lose connection-pool / rate-limit state.
        #
        # The mutation footgun that sharing creates is only PARTLY closed at the component layer
        # (#741), and it is worth being exact about which half. Each component's public
        # *containers* are frozen in its constructor — the LLM's ``retry_policy`` is a frozen
        # value, the REPL's sandbox allow-lists are tuples — so an in-place write
        # (``cfg.repl.allowed_imports.append(...)``) raises, and each is exposed through a
        # setter-less property, so rebinding a whole container (``cfg.repl.allowed_imports =
        # [...]``) is an ``AttributeError``. The LLM's query params have no public container at
        # all. A component's *scalars* are not covered: ``cfg.repl.exec_timeout = 99`` is a plain
        # attribute write, and every config sharing that component sees it. A ``Config`` field
        # cannot be written at all (the class is frozen), so rebinding a whole group through a
        # shared config is closed; reaching through one into a component is not. Pinned by
        # ``test_configoverride_resolved_aliases_base_mutable_containers``.
        # ``ConfigOverride.apply_to`` shares the same shallow semantics, per
        # ``test_apply_to_shallow_aliases_unoverridden_mutable_fields``.
        #
        # Cache thread-safety: one ``ConfigStack`` can be shared across worker threads (a
        # contextvar copies the reference). ``cached_property`` isn't locked, so two threads may
        # both compute and race on the ``__dict__`` write — but the write is a GIL-atomic dict
        # setitem and both results are pure and equal, so the worst case is a redundant fold, never
        # corruption. No lock: it would serialize this hot read path. (Works on a frozen dataclass
        # because ``cached_property`` writes ``instance.__dict__`` directly, bypassing the frozen
        # ``__setattr__`` — so ``ConfigStack`` must not use ``slots``.)
        return _fold_config(layer.overrides for layer in self.layers)

    def push(self, layer: Layer) -> ConfigStack:
        """Return a new stack with ``layer`` appended (innermost = last = highest precedence)."""
        return ConfigStack(self.layers + (layer,), self.established)


# Context-local configuration override stack (#727). Holds an ordered stack of composable
# ``ConfigOverride`` diffs; ``get_config()`` folds it onto ``_base_settings`` lazily.
# noqa on B039: a shared mutable ContextVar default is the footgun the rule guards against, but
# ConfigStack is a FROZEN (immutable) dataclass with an empty-tuple layer list — sharing the one
# default instance across contexts is safe, and its resolved fast-path returns the live
# base settings uncached (see ConfigStack.resolved), so no per-context state accretes on it.
_config_stack: ContextVar[ConfigStack] = ContextVar(
    "jaz_config_stack",
    default=ConfigStack(),  # noqa: B039
)


def get_config() -> Config:
    """
    Get the current effective configuration.

    Returns:
        The current Config instance

    Raises:
        ValueError: if no LLM is configured. A config is complete by construction, so there is
            none to return until :func:`configure` names a backend; the message carries the call
            that does it.
    """
    # Folds the context-local override stack (set via ``ConfigOverride``) onto the global
    # default, or returns the global default directly when the stack is empty.
    #
    # Do NOT read this inside execution / enforcement / hook paths to decide something about a
    # *specific* invoke — it silently diverges from the config that invoke actually ran under
    # whenever a per-call override is in scope (this exact hazard caused the replay divergence
    # bug fixed in #826 §1). Read the invoke's own config instead: ``event.config`` in a hook, or
    # the prehook-resolved ``config`` in ``_invoke.py``. Legitimate uses are ambient/caller-frame
    # reads with no invoke to resolve against (e.g. constructing an ``Agent`` outside any invoke,
    # or reading the caller's ancestor policy at a public entry point). See #826 / #531.
    return _config_stack.get().resolved


def get_config_stack() -> ConfigStack:
    """Return the live ``_config_stack`` ContextVar value (the ordered override stack).

    The agent-facing synthesized :func:`invoke` (see ``get_subinvoke`` in _subinvoke.py) reads
    this to re-establish propagated config across a raw worker-thread boundary: it composes the
    closure-threaded ancestor stack with any layers the worker pushed locally. ``stack.established``
    distinguishes a fresh worker (ContextVar didn't propagate → needs re-base) from a same-thread
    subtree (already carries the ancestor).
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


# The framework's shared sentinel, not a second private `object()`. Configuration keys off
# argument *presence* — naming a group replaces it, omitting it leaves it untouched — and `None`
# is a legitimate value, so "omitted" cannot be spelled `None`.
#
# It is the public `UNSET` because these signatures are PUBLISHED: griffe renders a default as
# its source expression, so a private `_UNSET` put a name callers cannot import into the
# reference page for `configure` and `ConfigOverride`. Naming the shared sentinel is the same
# fix the ecosystem applies (openai-python's `NOT_GIVEN`, attrs' `NOTHING`) — one public,
# documented marker rather than a private one per module.


def _configured_fields(**maybe_unset: Any) -> dict[str, Any]:
    """Keep only the fields the caller actually passed (drop ``UNSET``).

    Preserves the presence semantics of the grouped shape under an explicit signature: a field is
    applied only when named, never merely because it carries a default.
    """
    return {name: value for name, value in maybe_unset.items() if value is not UNSET}


def configure(
    *,
    llm: BaseLLM | UnsetType = UNSET,
    repl: BaseREPL[Any] | UnsetType = UNSET,
    protocol: BaseProtocol | UnsetType = UNSET,
) -> None:
    """
    Configure the global default JAZ framework settings.

    This modifies the global default configuration that is used when no
    context-local override is active.

    Settings are grouped by component. Each group takes
    the **configured component itself**, whose constructor is the schema for that group's
    settings.

    **Setting a group replaces it.** A group names the whole component, so you restate it
    completely rather than adjusting one of its settings in place.

    Args:
        llm: How JAZ talks to the model — a :class:`BaseLLM` such as
            ``LiteLLM(model="openai/gpt-5-mini", temperature=0.7)``. :class:`LiteLLM` is the
            default; :class:`GitHubCopilotLLM` and :class:`SGLangLLM` also ship
            (see :mod:`jaz.llm`),
            and subclassing ``BaseLLM`` adds your own. The constructor is the whole
            surface: the ``model`` id, the backend's own settings (``base_url``, ``api_key``),
            the retry policy (``retry_policy=RetryPolicy(...)``), and any per-request
            default (``temperature``, ``max_tokens``, ``reasoning_effort``, ...) it passes
            through to the provider API. Anything the constructor does not name becomes a
            per-request default, so a backend gains a setting simply by declaring it.
        repl: The REPL execution layer — a :class:`BaseREPL` such as
            ``PythonREPL(exec_timeout=60)``. Its constructor takes ``exec_timeout``,
            ``exec_memory_limit`` (process-RSS ceiling in bytes, enforced independently of
            ``exec_timeout``), the sandbox allow-lists, and the rest of the REPL's settings (see
            :class:`PythonREPL`).
        protocol: The LLM↔REPL codec — a :class:`BaseProtocol` such
            as ``CodeOnlyProtocol(max_invoke_input_length=20000)``. Its constructor takes the
            ``*_template`` prompt-template names, ``max_invoke_input_length``,
            ``max_repl_output_length`` and ``subinvoke_description``.

    Examples:
        jaz.configure(repl=PythonREPL(exec_timeout=60))
        jaz.configure(llm=LiteLLM(model="openai/gpt-5-mini", temperature=0.7))

    """
    updates = _configured_fields(
        llm=llm,
        repl=repl,
        protocol=protocol,
    )
    # Validate exactly as construction would, then write the authored base. There is no `Config`
    # to mutate: the base is the settings mapping, so setting `repl` before any `llm` is named is
    # an ordinary partial write rather than a half-built object.
    Config._validate_update(updates)
    _base_settings.update(updates)


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
    validated when the override is applied and the resulting config is built.

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
        llm: BaseLLM | UnsetType = UNSET,
        repl: BaseREPL[Any] | UnsetType = UNSET,
        protocol: BaseProtocol | UnsetType = UNSET,
    ) -> None:
        # Field *values* are validated outside ``__init__``: ``apply_to`` validates the local form
        # when it builds the resulting config, and ``__enter__`` validates the scoped form eagerly
        # as it installs the layer (it builds no config of its own — the fold does, later, at the
        # first read). An *unknown* field is a ``TypeError`` at construction (the explicit
        # signature, #993) — that eager, statically-checkable rejection is the point of the
        # explicit surface.
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
        # thread, or concurrently from another) is forbidden — see `__enter__`. One token rather
        # than a stack because overlapping same-instance reuse is an error, not a supported case;
        # sequential reuse clears the token on exit and re-enters fine.
        self._token: Token[ConfigStack] | None = None

    def _as_layer(self) -> Layer:
        """This override as a composable stack layer carrying the raw diff."""
        # A seam rather than an inline `Layer(...)` at each push site: it is the single place an
        # override becomes a layer, so a second layer shape would land here and nowhere else.
        return Layer(self.overrides)

    def apply_to(self, config: Config) -> Config:
        """Return ``config`` with these overrides applied.

        The input ``config`` is never mutated. Its *unoverridden* component objects
        (``llm``/``repl``/``protocol``) stay shared by identity with ``config`` rather than being
        duplicated. An override with no fields set returns ``config`` itself, there being nothing
        to apply and nothing a caller could write to the result.
        """
        # Components are carried across by reference, the same semantics as the module-level
        # ``_fold_settings``, so there is one answer to "resolve overrides onto a base"; pinned by
        # ``test_apply_to_shallow_aliases_unoverridden_mutable_fields``. Sharing rather than
        # ``deepcopy`` also avoids raising on, or duplicating, a component wrapping a non-copyable
        # resource (a lock, an open HTTP pool), and is leak-safe since #741 because each component
        # holds its own containers frozen and read-only.
        if not self.overrides:
            return config
        settings = {f.name: getattr(config, f.name) for f in dataclass_fields(Config)}
        settings.update(self.overrides)
        return Config(**settings)

    def __enter__(self) -> None:
        """Push these overrides as a layer on the context stack for the duration of the block.

        Appends a composable ``Layer`` to the ``_config_stack`` — so the diff
        **propagates** to every nested ``jaz.invoke()`` in the block and composes, last-wins,
        with any ancestor/local layers. Returns ``None``: the diff is folded lazily, so there is
        no resolved ``Config`` at this point to hand back.

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
        # Validate field names/values now (fail-fast at the call site) — but without applying: we
        # install the raw diff *layer*, not a resolved
        # snapshot, so the fold stays lazy (see class docstring). ``Config._validate_update`` is a
        # schema-level classmethod (unknown fields, bad values) — it does
        # NOT read or fold the current stack, so nothing is mutated and the ContextVar is untouched.
        Config._validate_update(self.overrides)
        self._token = _set_config_stack(get_config_stack().push(self._as_layer()))
        # No return value: the fold is lazy, so there is nothing resolved here to bind. Binding a
        # config built eagerly at this point would pin the base as it stood at `__enter__`, where
        # the lazy fold pins it at the block's FIRST READ instead — see `ConfigStack._folded`.
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
