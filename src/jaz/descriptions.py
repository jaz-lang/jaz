"""Value-attached input descriptions.

A *description* is prompt/agent metadata that explains what a value is. When a
described value is passed as a top-level input to :func:`jaz.invoke`, its
description renders in the agent's prompt header instead of the auto-stringified
value, and the agent can read it back at runtime via :func:`get_description`.

Why descriptions attach to the *value* rather than to a kwarg name:

- **No brittle name coupling.** The description is a property of the *thing*; its
  meaning doesn't change when you pass it as ``sales=`` vs. ``sales_data=``.
  Renaming the kwarg can't silently drop the description.
- **Persistence across nested invokes.** Because the description lives on the
  object (or in an object-identity-keyed side-table), it travels with the value
  into any nested ``jaz.invoke`` call for free — including the delegate machinery
  that threads REPL state forward to sub-agents.

Storage strategy — **attribute, then weak side-table, then retaining table**:

- We try ``object.__setattr__(value, "__jaz_description__", ...)`` first. This is
  the most discoverable mechanism (it shows up under ``dir(obj)``), survives
  pickling, and works for the vast majority of user-defined classes. The dunder
  name is framework-owned, so it never collides with a user's own attributes.
- For values that reject attribute assignment but are both weak-referenceable and
  hashable (slotted classes, ``frozenset``), we fall back to a
  :class:`weakref.WeakKeyDictionary`. The weak reference means the entry is
  collected when the value is GC'd — no memory cost.
- For what neither path can hold — bare built-ins — we fall back to a global
  ``id()``-keyed table that retains the value. See below.

Semantics that are intentionally *not* magic:

- **``str(value)`` is unchanged.** A description is metadata, not a stringification
  override. ``isinstance`` checks and library operations (pandas, numpy, ...) see
  the original value and type.
- **Identity, not equality.** Two equal-but-distinct objects can carry different
  descriptions ("*this* dataframe is the Q1 report").
- **No propagation through mutation/derivation.** ``df.copy()`` / ``df.query(...)``
  produce new objects that start without a description. Re-:func:`describe` the
  result if it should be described — there is no reliable way to thread a
  description through arbitrary library operations, and pretending otherwise would
  silently produce stale descriptions.

**Bare built-ins.** Built-in primitives and containers (``dict``, ``list``, ``set``,
``tuple``, ``str``, ``int``, ...) can hold neither an attribute nor a weak reference,
so :func:`describe` records them in a process-global table that keeps them alive —
such a value is never garbage-collected.
:class:`~jaz.exceptions.DescriptionRetentionWarning` is emitted once many have
accumulated, so describing built-ins in a loop announces itself. Two ways to avoid
retention when it matters:

- **Composition** (supported) — put the built-in inside a small object of your own and
  describe that. The attribute path applies, so nothing is retained and the description
  still travels into nested invokes. Composition is also how you hide a value from the
  prompt header: only top-level inputs render, so nesting a value inside a described
  composite binds it into the REPL without showing it inline.
- :class:`jaz.Display` (**experimental**) — wrap the value at the ``jaz.invoke`` call
  site. It attaches nothing to the value and so retains nothing, and it is the right
  tool whenever the description is about this call rather than the object itself. It is
  also the only one of the two that can *hide* a top-level input directly. Outside
  ``jaz.__all__``: reachable, but it warns on use and may change or be removed.
"""

# WHY THE THIRD STORAGE PATH RETAINS ITS VALUES — and why `describe` is total at all.
#
# `id()` is an address and CPython recycles addresses aggressively, so an id-keyed table
# that did NOT hold its values would let a brand-new object silently inherit a dead one's
# description (verified with a short repro, not theoretical). Holding the reference is
# what makes the key permanently valid; the leak is the price of that soundness, and
# `_note_retention` is what keeps the price visible.
#
# This is what lets `describe` be total — work on every value, always return the object
# it was given, never change its type — which two earlier designs could not both do:
#
#   - A transparent `wrapt.ObjectProxy` for these values. Kept them collectable but broke
#     identity and type: `describe(x, ...) is x` and `type(x) is list` both became False,
#     the return value became silently load-bearing for one class of input, and C
#     fast-paths switching on exact type (`json.dumps`) rejected the value outright.
#     Those costs landed on callers who never opted into a proxy.
#   - Raising `ValueError` and pointing at the alternatives. Honest, but it left
#     `describe` partial for exactly the large values most worth describing.
#
# Retaining relaxes a fourth axis the design doc's "pick one of {identity, exact-type,
# permanence}" framing missed entirely: memory.
#
# EXECUTIVE CALL (user, 2026-08-05): totality is worth a bounded, observable memory cost.
# The decisive factor was that `DescriptionOverride` — the same id()-keyed mechanism,
# with a deterministic release point — is NOT in `jaz.__all__`, so rejecting built-ins
# left a public-API caller with no propagating option at all.
#
# See `design/design_features/attached_descriptions.md`.

from __future__ import annotations

import inspect
import math
import warnings
import weakref
from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Final

from ._catalog import _CatalogMode, render_catalog
from .exceptions import DescriptionRetentionWarning, DescriptionSharedValueWarning

_DESCRIPTION_ATTR = "__jaz_description__"


class _HiddenInput:
    """Sentinel meaning "render nothing for this input this call".

    Returned by :func:`get_description` (via a ``jaz.Display(value, None)``
    directive's ``__jaz_description__``) and distinct from ``None``: ``None`` means
    "no description — render the value's default"; ``HIDDEN`` means "drop this
    input's prompt block entirely" (it stays bound as a REPL variable). Only the
    prompt builder acts on ``HIDDEN``; every other consumer treats it as "no
    description". A sentinel rather than overloading ``None`` so the two genuinely
    different intents can't be conflated.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "HIDDEN"


HIDDEN: Final = _HiddenInput()

# A *description* is a permanent property of the object: what the thing *is*. It
# is one of: a `str` (literal prompt text); the `jaz.Catalog` render mode (render the
# value as a tool catalog); or a `Callable` evaluated at read time — a
# `__jaz_description__` method on a user class (dispatched like `__str__`) or a
# callable passed to `describe`. It is NOT how to hide/relabel a value for one
# particular invoke — that is a per-call display concern handled by `jaz.Display`
# (see display.py). Callables may optionally accept the input's *bound name* (the
# kwarg it is passed under) as a trailing argument — see `_resolve_stored`.
type DescriptionValue = str | _CatalogMode | Callable[..., str]

# Side-table for values that can't hold a `__jaz_description__` attribute
# (built-in containers, slotted classes). Keyed by object identity; weak so
# entries are collected with their values.
_descriptions: weakref.WeakKeyDictionary[object, DescriptionValue] = (
    weakref.WeakKeyDictionary()
)

# Last-resort table for values that can hold neither an attribute nor a weak reference —
# bare built-ins (`list`, `dict`, `int`, `str`, `None`, ...). Keyed by `id(value)`, and
# the value is RETAINED alongside its description. The retention is not incidental: it is
# what makes the `id()` key sound. CPython recycles addresses aggressively, so an
# id-keyed table that did not hold the value would let a brand-new object inherit a dead
# one's description (verified, not theoretical). Holding the reference makes that
# impossible, at the cost of the entry never being collected — see
# `DescriptionRetentionWarning` for when that cost bites and `_note_retention` for how it
# is surfaced.
#
# This is the same mechanism `DescriptionOverride` uses per-scope (see `_override_stack`);
# the only difference is lifetime — a `with` block pops its frame, this table never does.
#
# Mutation is a plain dict set/get, which is atomic under CPython's per-object locking, so
# no explicit lock: concurrent `describe` calls can interleave but cannot corrupt the dict.
_strong_descriptions: dict[int, tuple[object, DescriptionValue]] = {}

# Retained-value count at which to warn next. Starts at the threshold and doubles after
# each warning, so a runaway loop keeps signalling without emitting on every call.
_RETENTION_WARN_THRESHOLD = 128
_retention_warn_at = _RETENTION_WARN_THRESHOLD


def _note_retention() -> None:
    """Warn when the strong table has grown past the next reporting point.

    Split out of :func:`describe` so the growth policy lives in one place. The counter
    only ever moves forward: entries are never removed, so the table size is monotonic
    and each threshold fires at most once.
    """
    # The remedy leads with composition, not `jaz.Display`, because `Display` is demoted
    # (outside `__all__`, warns on every route to the name). Leading with it made this a
    # warning whose stated fix emits a second warning — the one item in the demotion a user
    # hits without reading any docs. `Display` stays as a flagged aside rather than being
    # cut: it is still the only thing that can *hide* an input, so a reader who needs that
    # should learn it exists and that it is provisional in the same breath.
    global _retention_warn_at
    if len(_strong_descriptions) < _retention_warn_at:
        return
    retained = len(_strong_descriptions)
    _retention_warn_at = retained * 2
    warnings.warn(
        f"jaz.describe has retained {retained} values that cannot hold a description "
        "of their own (bare built-ins such as list/dict/int). These are kept alive for "
        "the lifetime of the process and will not be garbage-collected. If you are "
        "calling describe in a loop or a per-request path, describe a longer-lived "
        "object that contains the value instead, or hoist the describe out of the loop. "
        "(jaz.Display at the call site also retains nothing, but it is experimental and "
        "warns on use.)",
        DescriptionRetentionWarning,
        stacklevel=3,  # through `describe` to the caller's own describe(...) line
    )


# CPython's small-integer cache. Documented in the language reference as an implementation
# detail, but a stable one across every CPython release, and the only int range where sharing
# is a *guarantee* rather than an observation.
_SMALL_INT_CACHE = range(-5, 257)


def _note_shared_value(value: object) -> None:
    """Warn when ``value`` is an object CPython shares across the whole interpreter.

    Split out of :func:`describe` for the same reason as :func:`_note_retention`: the policy
    for what counts as "shared" lives in one place.
    """
    # The one failure `_note_retention` structurally cannot reach. Its threshold prices
    # VOLUME — many retained values — but describing a shared scalar is a scope failure that
    # is already total at count 1: `describe(True, "done")` re-labels every `True` in the
    # process, including inside unrelated libraries. A single call never approaches 128, so
    # without this check the only signal was a sentence in a docstring.
    #
    # `type(value) is int`, not `isinstance`: an `int` SUBCLASS instance is a fresh object and
    # is not cached, so warning on it would be wrong. (Bools are handled by the identity
    # checks above, before the int test, since `isinstance(True, int)` is also true.)
    is_shared = (
        value is None
        or value is True
        or value is False
        or (type(value) is int and value in _SMALL_INT_CACHE)
    )
    if not is_shared:
        return
    warnings.warn(
        f"jaz.describe was called on {value!r}, which CPython shares across the entire "
        "process — every other use of this value now renders this description too, "
        "including inside libraries you did not write and in every nested jaz.invoke. "
        "Describe a small object that contains the value instead, so the description "
        "belongs to something you own. (jaz.Display labels it for one invoke without "
        "attaching anything, but it is experimental and warns on use.)",
        DescriptionSharedValueWarning,
        stacklevel=3,  # through `describe` to the caller's own describe(...) line
    )


# Stack of scoped overrides pushed by `DescriptionOverride`. Each frame maps
# `id(value) -> (value, description)`; the value is retained so its `id` stays
# valid (and unambiguous) for the lifetime of the scope. Propagates to nested
# `jaz.invoke` calls via contextvars, the same mechanism the hook system uses.
_override_stack: ContextVar[tuple[dict[int, tuple[object, DescriptionValue]], ...]] = (
    ContextVar("jaz_description_overrides", default=())
)


def describe[T](value: T, description: DescriptionValue) -> T:
    """Attach a permanent ``description`` to object ``value`` to override its
    default display to the LLM when passed as an input to :func:`jaz.invoke`;
    returns the object itself.

    ``description`` may be a ``str``; the :class:`jaz.Catalog` render mode (for a
    tool namespace / container, so its members render as a catalog); or a callable
    evaluated lazily at read time — given the object, and optionally the bound
    keyword name it was passed under — so the description can reflect the object's
    current state. The description travels with the object into invokes.

    Describing the same object again overwrites. ``value`` is never altered or replaced:
    ``describe(x, ...) is x`` and ``type(x)`` hold for every type.

    An object that can hold the description carries it through ``copy.copy`` and
    ``copy.deepcopy``. A value that is *rebuilt* rather than copied does not, e.g., a derived
    pandas frame (``df.copy()``, ``df.head(1)``, ``df[cols]``) comes back undescribed — nor
    does a copy of a bare builtin, which has nowhere of its own to keep the tag.

    Describing a bare builtin (``list``, ``dict``, ``int``, ...) keeps it alive for the
    lifetime of the process; :class:`~jaz.exceptions.DescriptionRetentionWarning` is emitted
    once many have accumulated, so in a loop or a per-request path describe a longer-lived
    object that *contains* the builtin — the attribute path applies, so nothing is retained.
    Do not describe an interned
    scalar (a small ``int``, ``bool``, ``None``, or identifier-like ``str``): every occurrence
    of one in the process is the same object, so ``describe(42, "row count")`` relabels an
    unrelated ``6 * 7`` too. :class:`~jaz.exceptions.DescriptionSharedValueWarning` is emitted
    for the cases the language guarantees are shared (``None``, ``bool``, small ``int``) — but
    **not** for interned strings, which are shared in practice with no guarantee about which.

    **Experimental.** The description system is an experimental feature; its interfaces may change
    in a future release.
    """

    # To relabel or hide a value for a *single* invoke without attaching anything to it, use
    # `jaz.Display` instead — `describe` is for the object's permanent identity, not per-call
    # display. `Display` is also the only one of the two that can HIDE an input.
    #
    # The docstring used to say a *new* object "starts undescribed" and gave ``df.copy()`` as
    # the example. Half wrong: ``copy.copy``/``copy.deepcopy`` copy ``__dict__``, so the
    # attribute path (the primary one) survives both — measured. What loses the tag is a
    # rebuild, which is why the pandas example held for the wrong reason: ``NDFrame.copy()``
    # goes through ``__finalize__``, which propagates only ``attrs``/``flags``/``_metadata``
    # and not arbitrary instance attributes. Measured on pandas 2.3.3: ``df.copy()``,
    # ``df.head(1)`` and ``df[cols]`` all come back undescribed.
    #
    # The former proxy path had a wart here — ``copy.copy`` of a described bare builtin raised
    # ``NotImplementedError: object proxy must define __copy__()`` from wrapt. The proxy is
    # gone, so that no longer happens: a copy of a bare builtin is just a new object with a
    # new ``id()``, hence undescribed, which is the same rule as any other rebuild.
    #
    # STORAGE, in order — attribute, weak side-table, retaining table:
    #   1. `__jaz_description__` on the value. Most values; survives pickling and copying.
    #   2. `_descriptions`, a WeakKeyDictionary, for values that reject attribute assignment
    #      but are BOTH weak-referenceable and hashable (slotted classes, `frozenset`).
    #      Collected with their values, so no retention.
    #   3. `_strong_descriptions`, keyed by `id(value)` and RETAINING it, for what is left:
    #      bare built-ins, which can hold neither an attribute nor a weak reference (CPython
    #      omits `__weakref__` from the small built-in types).
    #
    # Why path 3 retains: `id()` is an address and CPython recycles addresses aggressively,
    # so an id-keyed table that did NOT hold its values would let a brand-new object inherit
    # a dead one's description (verified, not theoretical). Holding the reference is what
    # makes the key permanently valid. The leak is the price of that soundness, and
    # `_note_retention` is what keeps it observable.
    #
    # EXECUTIVE CALL (user, 2026-08-05): totality is worth a bounded, observable memory cost.
    # Two earlier designs were rejected — a transparent `wrapt.ObjectProxy` (kept values
    # collectable but broke `is`/`type()` and made the return value silently load-bearing,
    # with `json.dumps` rejecting the proxy outright), and raising `ValueError` on bare
    # built-ins (honest, but left `describe` partial for exactly the large values most worth
    # describing). The decisive factor was that `DescriptionOverride` — the same id()-keyed
    # mechanism, with a deterministic release point — is not in `jaz.__all__`, so rejecting
    # built-ins left a public-API caller with no propagating option at all. See
    # `design/design_features/attached_descriptions.md`.
    #
    # Known inconsistency: path 2 keys a WeakKeyDictionary and so matches by `__eq__`/
    # `__hash__` — two equal-but-distinct slotted instances share one entry — while paths 1
    # and 3 match by identity.

    try:
        object.__setattr__(value, _DESCRIPTION_ATTR, description)
    except (AttributeError, TypeError):
        try:
            _descriptions[value] = description
        except TypeError:
            # Unsettable *and* (unweakref-able or unhashable) — a bare built-in. Fall back
            # to the retaining table. Overwriting an existing entry reuses its `id()` key,
            # so re-describing the same value does not grow the table (and `_note_retention`
            # reads the post-write size, so it can never over-count).
            _strong_descriptions[id(value)] = (value, description)
            _note_shared_value(value)
            _note_retention()
    return value


def _max_positional_args(fn: Callable[..., object]) -> float:
    """Largest number of positional args ``fn`` accepts (``math.inf`` if ``*args``).

    Used to decide whether a description callable opted into receiving ``bound_name``
    as a trailing positional argument, without resorting to try/except-on-TypeError
    (which would mask a genuine ``TypeError`` raised *inside* the callable).
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 0
    count = 0
    for p in params:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            count += 1
        elif p.kind is p.VAR_POSITIONAL:
            return math.inf
    return count


def _resolve_stored(
    stored: DescriptionValue, value: object, bound_name: str | None
) -> str | _HiddenInput:
    """Collapse a stored description entry to the ``str`` it represents.

    Cases, in order:

    - The ``jaz.Catalog`` render mode renders ``value`` as a tool catalog rooted at
      ``bound_name`` (the kwarg the agent sees).
    - A callable is evaluated. The base argument is the value for a ``describe``
      callable, or nothing for a *bound* ``__jaz_description__`` method (``self`` is
      implicit, mirroring ``str()``/``__str__``). ``bound_name`` is appended as a
      trailing positional argument **only if the callable accepts it** — so both
      PR #372's name-agnostic forms (``lambda v: ...``, ``def __jaz_description__(self)``)
      and the bound-name-aware forms (``lambda v, name: ...``,
      ``def __jaz_description__(self, bound_name)``) work. (This is the superset that
      keeps #372's zero-arg form working while enabling the catalog use case that
      needs the kwarg name — see design Open Question 3.)
    - A plain ``str`` passes through unchanged.
    """
    if isinstance(stored, _CatalogMode):  # the `jaz.Catalog` render mode
        return render_catalog(value, bound_name)
    if callable(stored):
        base_args: tuple[object, ...] = () if inspect.ismethod(stored) else (value,)
        if _max_positional_args(stored) >= len(base_args) + 1:
            return stored(*base_args, bound_name)
        return stored(*base_args)
    return stored


def resolve_jaz_get(value: object) -> object:
    """Resolve the ``__jaz_get__`` payload-substitution protocol for an input.

    Returns ``value.__jaz_get__()`` when ``value`` carries the protocol (a
    ``jaz.Display`` directive, a ``Library``), else ``value`` unchanged. The
    counterpart to :func:`get_description` (the ``__jaz_description__`` half): the
    REPL binds this payload and prompt rendering uses it for the type label /
    default, while the wrapper itself is consulted for the description. Uses
    ``getattr`` (not ``hasattr`` + attribute access) so the static type narrows.

    A *class* is skipped: ``getattr(SomeClass, "__jaz_get__")`` returns the plain
    (unbound) function, so calling it raises ``TypeError: missing 'self'``. This is the
    same guard `inputs.resolve_inputs` applies (#946) — duplicated here rather than
    centralized because the prompt builder calls *this* function directly, bypassing
    `resolve_inputs`, so the guard has to live at both entry points. It matters because
    framework helper classes are delivered as ordinary inputs
    (``jaz.invoke(Display=jaz.Display, ...)``) and ``Display``/``Library`` are exactly
    the symbols defining ``__jaz_get__``.
    """
    getter = None if isinstance(value, type) else getattr(value, "__jaz_get__", None)
    return getter() if getter is not None else value


def get_description(
    value: object, bound_name: str | None = None
) -> str | None | _HiddenInput:
    """Return the description attached to ``value``, or ``None`` if none is set.

    Lookup order: scoped ``DescriptionOverride`` -> ``__jaz_description__``
    (attribute *or* method) -> side-table. This order *is* the layering rule
    (permanent ``describe`` base + scoped shadow stack on top) documented on
    :class:`DescriptionOverride` — an active override shadows the base for its whole
    extent, even a ``describe`` called inside the block. Used by the prompt builder to
    render top-level inputs and exposed in the REPL (``jaz.get_description``) so agents
    can read descriptions on demand. Callable descriptions are evaluated here, so
    the caller always gets a plain ``str`` (or ``None`` when none is set).

    May also return the ``HIDDEN`` sentinel when ``value`` is a
    ``jaz.Display(value, None)`` directive — "render nothing this call", distinct
    from ``None`` ("no description"). Only the prompt builder distinguishes the two;
    other callers can treat ``HIDDEN`` like ``None``.

    ``bound_name`` is the kwarg name the input is passed under at the ``jaz.invoke``
    call site. The prompt builder threads it in so callable/Catalog descriptions can
    render against the name the agent actually uses (e.g. a tool catalog rooted at
    ``tools.foo`` rather than the value's internal module name). It is ``None`` when
    there is no call-site context (e.g. a bare ``jaz.get_description(value)`` call
    in the REPL, with no kwarg context).
    """
    for frame in reversed(_override_stack.get()):
        entry = frame.get(id(value))
        if entry is not None and entry[0] is value:
            return _resolve_stored(entry[1], value, bound_name)

    desc = getattr(value, _DESCRIPTION_ATTR, None)
    # A *class* is never its own described value. `getattr(SomeClass, "__jaz_description__")`
    # returns the plain (unbound) function the class defines for its INSTANCES, and
    # `_resolve_stored` would then call it with the class itself as `self` — which crashes
    # (`type object 'Display' has no attribute 'text'`) or silently lies. Mirrors the
    # identical guard in `inputs.resolve_inputs` for `__jaz_get__` (#946), and matters for
    # the same reason: framework helper *classes* are delivered as ordinary inputs
    # (`jaz.invoke(Display=jaz.Display, ...)`), and `Display` is exactly the symbol that
    # defines `__jaz_description__`.
    #
    # Only *functions* are skipped, so an explicit `describe(SomeClass, "text")` /
    # `describe(SomeClass, jaz.Catalog)` still resolves normally — those store a
    # non-function value. Known gap: `describe(SomeClass, lambda c: ...)` is indistinguishable
    # from a method definition here and is therefore ignored; describe the class with a
    # string, or wrap it in `jaz.Display`, if you need a computed description for a class.
    if isinstance(value, type) and inspect.isfunction(desc):
        desc = None
    if desc is not None:
        return _resolve_stored(desc, value, bound_name)

    try:
        stored = _descriptions.get(value)
    except TypeError:
        # Not weak-referenceable (int/str/tuple) or unhashable (list/dict): cannot be in
        # the weak side-table — fall through to the strong one.
        stored = None
    if stored is not None:
        return _resolve_stored(stored, value, bound_name)

    # Last: the retaining table for bare built-ins. `entry[0] is value` mirrors the
    # override-stack check — with the value retained an `id()` collision is impossible,
    # so this is a cheap assertion of that invariant rather than a live guard.
    entry = _strong_descriptions.get(id(value))
    if entry is not None and entry[0] is value:
        return _resolve_stored(entry[1], value, bound_name)
    return None


class DescriptionOverride:
    """Scope one or more descriptions to a ``with`` block without mutating the values — the
    scoped counterpart to :func:`describe`, mirroring how :class:`jaz.ConfigOverride` is the
    scoped counterpart to :func:`jaz.configure`.

    Each ``(value, description)`` pair attaches ``description`` to ``value`` (matched by
    object identity — never hashed, copied, or written to the value or side-table) for the
    duration of the block, then removes it on exit. Overrides take precedence over any
    attribute/side-table ``describe`` and propagate into nested ``jaz.invoke`` calls::

        # one value
        with jaz.DescriptionOverride((sales_df, "Q1 sales: ...")):
            invoke(task="...", sales=sales_df)

        # several at once
        with jaz.DescriptionOverride((train_df, "training rows"), (test_df, "held-out rows")):
            invoke(task="...", train=train_df, test=test_df)

    **Layering semantics — and why a ``describe`` *inside* the block does not win over the override.**
    :func:`describe` and ``DescriptionOverride`` are two different *kinds* of state, and they compose
    by *layering*, not by a single "most-recent-call-wins" timeline: ``describe`` writes the value's
    **permanent base** description (one slot, last-write-wins, unscoped), while each active
    ``DescriptionOverride`` pushes a **scoped shadow layer** on top (of nested overrides, the innermost
    wins). :func:`get_description` reads the innermost active override, else the base. The consequence
    is that a ``describe`` call made *inside* an override block does **not** override it: it updates the base
    *beneath* the shadow, so the override still renders for the block's whole extent and the
    interleaved ``describe`` value surfaces only once the override exits. This is the one coherent way
    to merge a permanent single slot with a LIFO scoped stack: the alternative (letting a
    describe-inside punch through the override) would force block exit to either silently discard that
    ``describe`` or restore unpredictably, and would break the scoped override's core guarantee that it
    deterministically shadows for its whole extent — including against a mutation made by a nested
    ``jaz.invoke``. It is exactly how :func:`jaz.configure` behaves under a :class:`jaz.ConfigOverride`
    block (a ``configure`` updates the base singleton but stays shadowed until the scope lifts) — the
    same base-plus-shadow-stack layering, documented on ``config.ConfigStack``; this is the deeper form
    of the ``configure`` ↔ ``ConfigOverride`` parity this class was built to mirror.

    **Why pairs, not a mapping** (``{value: desc}``): the whole point of the scoped form is to
    work for values that reject :func:`describe` — unhashable ones like lists/DataFrames —
    which can't be mapping keys. Identity matching (not equality) also means two equal-but-
    distinct values keep separate descriptions. So the interface is uniformly ``(value,
    description)`` tuples; the earlier ``described(mapping | list-of-pairs)`` and the single-
    value ``description_override`` are folded into this one varargs signature.

    **Scope limitation** — this is context-manager-only. Unlike :class:`jaz.ConfigOverride`
    (which can also be passed positionally to :func:`jaz.invoke` for a *local*,
    non-propagating override), a ``DescriptionOverride`` ``with`` block always propagates to
    nested invokes; there is no per-invoke local form. That parity is a deliberate non-goal
    here — add it as a separate feature if a use case appears. Construct a fresh instance per
    ``with`` block: an instance holds the reset token of its active scope, so re-entering the
    same instance while it is already open is unsupported (the idiomatic ``with
    jaz.DescriptionOverride(...):`` already builds a fresh one).
    """

    def __init__(self, *pairs: tuple[object, DescriptionValue]) -> None:
        # Keyed by id() so unhashable values work as keys; the value is retained alongside the
        # description so `get_description` can confirm `entry[0] is value` and reject a stale
        # id() collision (a new object reusing a GC'd value's address).
        self._frame: dict[int, tuple[object, DescriptionValue]] = {
            id(value): (value, desc) for value, desc in pairs
        }
        self._token: (
            Token[tuple[dict[int, tuple[object, DescriptionValue]], ...]] | None
        ) = None

    def __enter__(self) -> None:
        self._token = _override_stack.set((*_override_stack.get(), self._frame))

    def __exit__(self, *exc_info: object) -> None:
        assert self._token is not None, (
            "DescriptionOverride exited without being entered"
        )
        _override_stack.reset(self._token)
        self._token = None
