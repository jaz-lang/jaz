"""Base classes for the hook system.

This module defines the foundational types used throughout the hook system:
- Event: Base class for all events
- ExecutionContext: Base context for all events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .blackboard import Blackboard

if TYPE_CHECKING:
    from jaz.config import Config

    from .dispatcher import Hook


@dataclass(frozen=True)
class Event:
    """Base class for all events.

    Events represent discrete points in the agent's execution lifecycle where
    hooks can observe and influence behavior.

    Every event carries ``timestamp`` — the moment it was emitted to hooks, as a
    **timezone-aware UTC** ``datetime``. Durations are
    consumer arithmetic over two events' timestamps: ``Exit.timestamp - Enter.timestamp``
    is a span's boundary-to-boundary interval, and the ``Send``/``Complete`` stamps give
    the per-stage breakdown. There are no interval fields on any event.

    Event objects must be treated as immutable and should never be mutated. Side effects are
    emitted by returning effects from the event handler (see :class:`Hook`).
    """

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
    # Some referenced objects are deliberately *live* rather than copies — ``config`` is the actual
    # effective ``Config`` and ``hooks`` holds the live ``Hook`` instances — so that hooks can
    # *read* real runtime state. "Read" is the operative word: mutating through them (e.g.
    # ``event.config.baseline_hooks.append(...)`` or ``event.hooks[i].some_attr = ...``) is
    # unsupported and can corrupt shared/global state or a peer hook. Frozen + read-only mappings
    # close the accidental-mutation footguns that are cheap to close; the rest is this contract.
    # TODO: revisit enforcing the remainder — notably that ``config`` is still fully mutable
    # (deep ``inputs``/``scope`` value mutation, live-peer mutation via ``hooks``, and the
    # mutable *payload value objects* some events carry — e.g. ``LLMResponse`` on
    # LLMQueryComplete/Exit, shared with the loop's own copy — are the others; the
    # container-level payloads are all coerced read-only as of #1156).

    # Effective config for the invoke that produced this event (the per-invoke
    # config after ConfigOverride / resolve_for_depth — NOT ambient get_config()).
    # Hooks and the dispatcher read baseline_hooks / model_config from here so
    # enforcement is correct under overrides, depth-specific config, and worker
    # threads (where contextvars don't propagate). See #463.
    # TODO(#481): other ambient state like the set of hooks in effect?
    #
    # kw_only so it stays out of every event's ``__match_args__`` (a positional
    # base field would lead it, silently binding ``config`` to the first var of
    # any positional ``case Event(x, ...)`` pattern) and so subclasses keep full
    # field-ordering freedom (a required kw-only base field never collides with a
    # subclass's defaulted positional field).
    config: Config = field(kw_only=True)

    # The per-invoke blackboard: a generationally-versioned key-value store through
    # which hooks receive per-invoke data and coordinate with one another. The
    # dispatcher stamps the invoke's live board onto every event at emit time (see
    # ``HookDispatcher.emit`` / ``seed_blackboard``), so the ``default_factory``
    # empty board is only what an event constructed outside an invoke (e.g. in a
    # unit test) carries. ``kw_only`` for the same ``__match_args__`` / field-order
    # reasons as ``config``; a default so existing constructions need not pass it.
    blackboard: Blackboard = field(kw_only=True, default_factory=Blackboard)

    # Uniform per-event identity (#481): every event self-locates in the invocation
    # tree without ad-hoc per-event fields. ``invoke_id`` is the uuid of the invoke
    # that produced the event; ``depth`` its absolute recursion depth (top invoke = 1, its
    # first nested invoke 2, ...; matches per-depth config resolution and the 1-based prehook).
    # ``kw_only`` for the same ``__match_args__`` / field-order reasons as
    # ``config`` (a positional base field would lead every subclass's ``__match_args__``).
    invoke_id: str = field(kw_only=True)
    depth: int = field(kw_only=True)

    # The hooks ACTIVE when this event was dispatched — the resolved active set (baseline ‖
    # propagating ‖ local, in dispatch order), stamped by the dispatcher at emit time (like
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
    # their own edge (FileLogger / PrintLogger / ATIFTrace / OTelTracing,
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


@dataclass
class ExecutionContext:
    """Base context for all events.

    Holds the effects composed by the dispatcher for a given event. Subclasses
    add fields for the specific effects valid at that event.
    """

    pass
