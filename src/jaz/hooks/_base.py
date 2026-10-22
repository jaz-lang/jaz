"""Base classes for the hook system.

This module defines the foundational types used throughout the hook system:
- Event: Base class for all events
- ExecutionContext: Base context for all events
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaz._config import Config

    from ._dispatcher import Hook


@dataclass(frozen=True)
class Event:
    """Base class for all events.

    Events represent discrete points in the agent's execution lifecycle where
    hooks can observe and influence behavior.

    Every event carries ``invoke_id`` (the invoke it belongs to), ``depth`` (that invoke's
    nesting depth — ``1`` for a top-level ``invoke``, one more per sub-invoke), ``config``
    (the effective configuration the invoke runs under, overrides applied), ``scope`` (the
    resolved ``jaz.scope`` values in effect for it, as a read-only mapping) and ``hooks``
    (the hook instances in effect for it).

    Every event carries ``timestamp`` — the moment it was emitted to hooks, as a
    **timezone-aware UTC** ``datetime``. Durations are
    consumer arithmetic over two events' timestamps: ``Exit.timestamp - Enter.timestamp``
    is a span's boundary-to-boundary interval, and the ``Send``/``Complete`` stamps give
    the per-stage breakdown. There are no interval fields on any event.

    Every event also carries ``blackboard`` — the per-invoke key-value store hooks use to
    receive caller data and to coordinate with one another, as a **read-only** ``Mapping``.
    Read it like a dict (``event.blackboard.get("label")``); the only way to write is to
    return a :class:`~jaz.hooks.effects.BlackboardWrite` effect, which the dispatcher applies
    after every hook for the event has run. A write is therefore invisible to the event that
    produced it and visible from the next event onward, so producer and consumer hooks
    coordinate without depending on dispatch order.

    **One board per invoke, and a nested invoke gets its own.** "Per-invoke" means one board
    each, not one shared down the tree: a sub-invoke seeds a fresh board before its own
    ``InvokeEnter``, so no *board write* crosses the boundary in either direction — the child
    does not see what the parent wrote, and the parent's board is unchanged by the child.

    Values still cross, just not on the board. Downward, seed them:
    :attr:`Hook.blackboard_seed` runs again for every invoke the hook is active in, which is
    what an ambient ``with MyHook():`` gives you and a per-call ``invoke(MyHook(), ...)``
    deliberately does not. Upward, or across siblings, the carrier is the hook instance's own
    attributes — a hook that stores something while the child runs can write it to the parent's
    board at the parent's next event. That is the supported way to accumulate across invokes;
    the board is not.

    A key that has not been produced is simply **absent**: ``board[key]`` raises ``KeyError``
    and ``board.get(key)`` returns ``None`` (or the supplied default). "Not produced yet" and
    "never produced" read identically on purpose — a consumer must not be able to tell whether
    a producer hook has not run yet or is not active at all. **Read defensively**: prefer
    ``.get(key, default)`` and treat absence as "no value this generation."

    Event objects must be treated as immutable and should never be mutated. Side effects are
    emitted by returning effects from the event handler (see :class:`Hook`).

    That rule is wider than what the framework enforces, so treat it as absolute rather than as
    "whatever raises". Enforcement stops at the container: rebinding a field raises, and the
    mappings and sequences an event carries are copies that reject writes. Past that line
    nothing is promised — what those containers hold is the framework's own state. Some of it
    happens to be immutable (an exec result is a ``NamedTuple``, a ``Config`` is frozen), and the
    rest is live: input and scope values, message dicts, namespace bindings, the components
    inside ``config``, the peer instances in ``hooks``, the ``LLMResponse`` on an
    ``LLMQueryComplete``, the value on an edit record. Mutating one of those changes the run,
    with none of the ordering guarantees that make effects safe. Read through an event; write by
    returning effects.
    """

    # Why the per-invoke board boundary is spelled out rather than left to "per-invoke": the
    # phrase reads as "scoped to the invoke tree" as readily as "one each", and an external
    # write-up of the hook system read this docstring and described the board as "a global
    # message board". The distinction is load-bearing, not trivia — it is exactly what makes an
    # ambient ``with MetaData(...)`` differ from a per-call ``invoke(MetaData(...), ...)``, and
    # collapsing it would remove any way to label one invoke alone. Propagation is the hook
    # activation model's job; the board deliberately does not duplicate it.
    #
    # The downward half above says "seed it", which is in tension with `Hook.blackboard_seed`'s
    # own contract ("a pure function of the hook's own state (its constructor args)"): a seed
    # read once per invoke does let a hook mutate `self` during one invoke and seed the new
    # value into the next, which is not constructor-fixed. #1548 tracks which reading is
    # intended — the prohibition may be on invoke *context* rather than on the hook's state
    # evolving. Worth knowing before leaning on a computed seed.

    # Why immutability is strict, not stylistic: the *same* event object is dispatched to every
    # active hook in turn (the dispatcher does not copy it per hook — see ``HookDispatcher.emit``).
    # If mutation were allowed, a hook ordered earlier would silently change what a hook ordered
    # later sees at the *same* event, making hooks depend on dispatch order. Effects avoid this:
    # the dispatcher collects every hook's effects and applies them only *after* the whole hook
    # loop finishes (``BlackboardWrite`` mints the next generation; result/message effects compose
    # under the effect rules), so one hook's output is invisible to the other hooks at the same
    # event — coordination stays order-independent. The blackboard is the sanctioned cross-hook
    # channel and follows exactly this deferred-write discipline.
    #
    # What enforces it: a ``frozen`` dataclass, so rebinding any field raises
    # ``FrozenInstanceError``; ``inputs`` / ``scope`` are handed out as read-only mappings; and
    # ``hooks`` is an immutable tuple.
    #
    # Some referenced objects are deliberately *live* rather than copies — ``config`` holds the
    # actual effective components and ``hooks`` the live ``Hook`` instances — so that hooks can
    # *read* real runtime state. "Read" is the operative word: mutating *through* them (e.g.
    # ``event.config.llm.some_attr = ...`` or ``event.hooks[i].some_attr = ...``) is unsupported
    # and can corrupt shared/global state or a peer hook. Frozen + read-only mappings close the
    # accidental-mutation footguns that are cheap to close; the rest is this contract.
    #
    # ``config`` is itself a frozen ``Config``, so rebinding a component on it
    # (``event.config.llm = ...``) raises rather than silently repointing the invoke's config —
    # one arm of the TODO below, closed by making a config a value built complete.
    # The docstring above states the boundary as a RULE ("enforcement stops at the container")
    # with examples, not as an inventory: the live surface grows with every payload-carrying
    # field, and an inventory turns each omission into an implied "that one is safe". A reader
    # who takes "frozen, read-only mappings" as the whole contract mutates a value inside
    # ``inputs``, a binding inside ``namespace``, or an edit record's value, and corrupts the
    # run with nothing to warn them. It is phrased as what the framework does not PROMISE rather
    # than as which objects are mutable, because the latter is falsifiable and was in fact false:
    # the first draft said the payload on every ``*Complete`` was live, when only
    # ``LLMResponse`` is — the exec/invoke results are ``NamedTuple``s and refuse the write.
    # TODO: revisit enforcing the remainder — deep ``inputs``/``scope`` value mutation, reaching
    # through ``config`` into a component, live-peer mutation via ``hooks``, and the mutable
    # *payload value objects* some events carry (e.g. ``LLMResponse`` on LLMQueryComplete/Exit,
    # shared with the loop's own copy); the container-level payloads are all coerced read-only.

    # Effective config for the invoke that produced this event (the per-invoke
    # config after ConfigOverride — NOT ambient get_config()).
    # Hooks and the dispatcher read the effective configuration from here so
    # enforcement is correct under overrides and worker
    # threads (where contextvars don't propagate). See #463.
    #
    # kw_only so it stays out of every event's ``__match_args__`` (a positional
    # base field would lead it, silently binding ``config`` to the first var of
    # any positional ``case Event(x, ...)`` pattern) and so subclasses keep full
    # field-ordering freedom (a required kw-only base field never collides with a
    # subclass's defaulted positional field).
    config: Config = field(kw_only=True)

    # The per-invoke blackboard: a generationally-versioned key-value store through
    # which hooks receive per-invoke data and coordinate with one another. The
    # dispatcher stamps the invoke's current generation onto every event at emit time
    # (see ``HookDispatcher.emit`` / ``seed_blackboard``), so the empty default is only
    # what an event constructed outside an invoke (e.g. in a unit test) carries.
    # ``kw_only`` for the same ``__match_args__`` / field-order reasons as ``config``;
    # a default so existing constructions need not pass it.
    #
    # A plain ``MappingProxyType`` over a dict the dispatcher owns, NOT a jaz-defined store
    # type (executive call by the author): the only thing such a class added over a read-only
    # mapping was its write-composition method, and that now lives beside every other effect
    # composition as ``effects._resolve_blackboard_writes``. Dropping it keeps a
    # framework-private type off the public event surface — this field is inherited by every
    # public event and renders in the generated API reference — for no loss of contract.
    #
    # Generational discipline, and why the mapping must be *replaced* rather than updated:
    # ``MappingProxyType`` is a read-only *view*, not a snapshot, so mutating the dict behind a
    # proxy already handed to an event would retroactively change what that event's hooks read
    # — precisely the same-event race the generational rule exists to forbid. The dispatcher
    # therefore builds a **new** dict per generation and stamps a new proxy; no dict is ever
    # mutated after it has been handed out. (``tests/hooks/test_blackboard.py`` pins this: a
    # board captured at an earlier event must never grow a later event's write.)
    #
    # The default is a single shared empty proxy rather than a ``default_factory``: it is
    # immutable, so sharing it across every hand-built event is safe, and it renders in the
    # generated docs as ``MappingProxyType({})`` — a builtin — instead of a private name.
    #
    # KNOWN LIMITATION, pre-dating this field: a ``mappingproxy`` is unpicklable and
    # ``deepcopy`` falls back to pickle, so an ``Event`` holding one cannot be copied. This is
    # NOT introduced here — ``InvokeEnter`` stamps ``inputs`` and ``scope``, and
    # ``InvokeSend`` stamps ``inputs``, the same way (``events/_invoke.py``) — so the board is
    # not the first field of its kind. Nothing in-tree copies an event.
    #
    # ``jaz.llm._base`` settled the same trade the other way for a backend (#741): store a dict,
    # wrap in a view on the way out behind a read-only property. That pattern works here too
    # (verified), but applying it to the board ALONE buys nothing — the other three fields still
    # block the copy — and making ``Event`` genuinely copyable means converting all four, which
    # renames their ``__init__`` parameters (``inputs=`` -> ``_inputs=``) and so breaks every
    # caller that hand-builds an event. That is a deliberate open question, not an oversight —
    # tracked on #831, which is where the immutability-vs-copyability trade for ``Event`` lives
    # and which every option in its list makes worse.
    # ``tests/hooks/test_blackboard.py`` pins the limitation so it stays a decision, not a bug.
    blackboard: Mapping[str, object] = field(kw_only=True, default=MappingProxyType({}))

    # The RESOLVED ambient ``jaz.scope`` for the invoke (``{**parent_scope, **local_scope}``,
    # mirroring how ``config`` is the resolved effective config). A base field, on every
    # event, for the same reason ``config`` and ``hooks`` are (user's executive call,
    # 2026-09-19): it is caller-set, resolved once per invoke, inherited by sub-invokes, and
    # editable by no effect — ``DropInputs`` un-passes inputs only — so it is ambient invoke
    # context, not a per-span proposal. It used to live on ``InvokeEnter`` alone, which made a
    # hook at any later event remember it from there; ``event.config`` and ``event.hooks``
    # were never like that. Stamped by ``HookDispatcher.emit`` from the invoke's ``InvokeEnter``
    # (like ``blackboard``/``hooks``), so a hand-built event outside a dispatch carries the
    # empty default. Disjoint from ``InvokeEnter.inputs`` by construction (a name defined both
    # ways raises at invoke time), so ``{**scope, **inputs}`` — what the REPL binds — is a plain
    # union. Frozen in ``__post_init__`` (subclasses call ``super().__post_init__()``).
    scope: Mapping[str, object] = field(kw_only=True, default=MappingProxyType({}))

    # Uniform per-event identity (#481): every event self-locates in the invocation
    # tree without ad-hoc per-event fields. ``invoke_id`` is the uuid of the invoke
    # that produced the event; ``depth`` its absolute recursion depth (top invoke = 1, its
    # first nested invoke 2, ...; matches the 1-based prehook).
    # ``kw_only`` for the same ``__match_args__`` / field-order reasons as
    # ``config`` (a positional base field would lead every subclass's ``__match_args__``).
    invoke_id: str = field(kw_only=True)
    depth: int = field(kw_only=True)

    # The hooks ACTIVE when this event was dispatched — the resolved active set (propagating ‖
    # local, in dispatch order), stamped by the dispatcher at emit time (like
    # ``blackboard`` / ``config``). This is the effect system's LIVE view of the active set: the
    # actual ``Hook`` instances, so a hook can introspect its peers' runtime state. Mirrors
    # ``config`` being the live resolved ``Config`` — both are live ambient snapshots, hence the
    # naming parity (``hooks`` not ``active_hooks``, as ``config`` isn't ``active_config``).
    # Includes the observer itself. Reflects the set at THIS event (it can change between turns as
    # an agent activates/exits hooks). Empty for an event constructed outside a dispatch (e.g. a
    # unit test). ``kw_only`` for the same reasons as ``config``.
    #
    # READ-ONLY BY CONVENTION: reading a peer's state is the supported use; mutating a peer through
    # this field is unsupported — the event is a passive record, not a back channel between hooks.
    # Observability consumers that want a renderable governance trace call ``repr()`` at
    # their own edge (FileLogger / PrintLogger / TrajectoryRecorder / OTelTracing,
    # all at ``InvokeEnter``): rendering lives at the log boundary, not
    # on this field — which is exactly why the field can hold live, non-serializable hook state
    # (open files, locks, spans) that a pre-serialized field could not.
    hooks: tuple[Hook, ...] = field(kw_only=True, default_factory=tuple)

    # When this event was EMITTED to hooks — stamped by ``HookDispatcher.emit`` immediately
    # before the hook loop (like ``blackboard``/``hooks`` above), overwriting the
    # construction-time default a hand-built event carries outside a dispatch.
    #
    # Emission time, not construction time (user decision, #1156 follow-up — the Kafka
    # CreateTime-vs-LogAppendTime distinction, choosing the append side; DOM's ``timeStamp``
    # chose creation time and it is a recurring confusion there): exit events are
    # constructed measurably before they are dispatched, and "when hooks saw it" is the
    # meaning every derived interval wants. This one uniformly-defined point stamp REPLACED
    # the ``*Exit`` events' ``start_time``/``end_time`` (the interval-on-record model, #1011)
    # and ``LLMQueryExit``'s ``call_start_time``/``call_end_time``: the interval fields'
    # boundary semantics took three caveats to state accurately and had no in-tree interval
    # consumer, and the call fields lost their only consumer when #1159 made BudgetPool a
    # scalar enforcer — unread recorded surface, removed per the same dead-surface standard
    # as ``show_status`` (#719/#1119). Provider-call latency is NOT derivable from event
    # timestamps (``Complete - Send`` includes Send-stage dispatch); if a consumer ever needs
    # it exactly, it must be measured at the call site again and given a consumer in the
    # same change.
    #
    # Timezone-aware UTC (user decision, #1163 review): a recorded timestamp must be an
    # unambiguous absolute time — naive-local is wrong by an hour across a DST fold and
    # meaningless when a trace is read on another machine — and aware values mix safely
    # in arithmetic. Naive-local (the codebase's prior convention) was rejected on the
    # merits, pre-release being the cheapest time; CostTracker's stamps were converted
    # in the same change so no naive wall-clock stamp remains in src/.
    #
    # The default is ``datetime.now`` rather than ``None``: a ``None`` default would make
    # a never-emitted event *detectable*, but hand-built events fired straight at
    # handlers (the unit-test pattern) feed consumers like ATIF's
    # ``timestamp.astimezone(...)`` that need a real datetime — an Optional field would
    # push a None-guard into every consumer to serve only synthetic events.
    timestamp: datetime = field(kw_only=True, default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        # Read-only view over a private shallow copy, like ``InvokeEnter.inputs``: one shared
        # event instance is dispatched to every hook, so a hook must not be able to rebind a
        # scope name that a later-dispatched hook then observes. Subclasses that define their
        # own ``__post_init__`` call this one first.
        object.__setattr__(self, "scope", MappingProxyType(dict(self.scope)))


@dataclass
class ExecutionContext:
    """Base context for all events.

    Holds the effects composed by the dispatcher for a given event. Subclasses
    add fields for the specific effects valid at that event.
    """

    pass
