"""HookDispatcher orchestrates hooks and composes their effects into execution contexts."""

import json
import logging
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any, ClassVar
from weakref import WeakValueDictionary

from jaz.exceptions import (
    AbortError,
    BlackboardSeedError,
    HookActivationError,
    OverrideConflictError,
    REPLInputConflictError,
    SandboxKeyError,
    _JazInternalError,
)

from .base import Event
from .blackboard import Blackboard
from .context import _reset_hook_context, _set_hook_context, get_current_hooks
from .effects import (
    Abort,
    AddInputs,
    AddMessages,
    AddVariables,
    BlackboardWrite,
    DisableRecursion,
    DropInputs,
    DropMessages,
    DropVariables,
    Effect,
    ModifyResult,
    OverrideResponse,
    OverrideResult,
    resolve_modify_results,
    resolve_override_results,
)

if TYPE_CHECKING:
    from jaz.config import Config
from .events import (
    InvokeContext,
    InvokeEnter,
    InvokeExit,
    InvokeExitContext,
    InvokeSpan,
    LLMQueryContext,
    LLMQueryEnter,
    LLMQueryExit,
    LLMQueryRetry,
    LLMQueryRetryContext,
    LLMQuerySpan,
    REPLExecContext,
    REPLExecEnter,
    REPLExecExit,
    REPLExecExitContext,
    REPLExecSpan,
)

logger = logging.getLogger(__name__)


# Old -> current ``qualified_name`` for renamed hook classes.
#
# ``Hook.to_dict`` stamps ``qualified_name`` into artifacts that OUTLIVE the code that
# wrote them: ATIF traces (the proposed canonical replay/cost-reconstruction format,
# #813), conversation-history records, OTel span attributes, and the log lines
# ``WorkflowReplay`` parses back (#712). Renaming a hook class therefore orphans
# every run recorded before the rename — ``from_dict`` raises "Unknown hook", and any
# tooling that matches the string just silently finds nothing.
#
# Cost of keeping an entry is one dict line; cost of omitting one is unreadable history.
# Add an entry whenever a hook class is renamed.
_LEGACY_HOOK_QUALNAMES: dict[str, str] = {
    "jaz.hooks.builtin.iteration_limit.IterationLimitHook": "jaz.hooks.builtin.iteration_limit.IterationLimit",
    "jaz.hooks.builtin.budget_forcing.BudgetForcingHook": "jaz.hooks.builtin.budget_forcing.BudgetForcing",
    "jaz.hooks.builtin.compaction.CompactionHook": "jaz.hooks.builtin.compaction.Compaction",
    "jaz.hooks.builtin.context_window.ContextWindowHook": "jaz.hooks.builtin.context_window.ContextWindow",
    "jaz.hooks.builtin.conversation_history.ConversationHistoryHook": "jaz.hooks.builtin.conversation_history.ConversationHistory",
    "jaz.hooks.builtin.replay.ReplayHook": "jaz.hooks.builtin.replay.Replay",
    "jaz.hooks.builtin.workflow_replay.WorkflowReplayHook": "jaz.hooks.builtin.workflow_replay.WorkflowReplay",
    "jaz.hooks.builtin.sliding_window.SlidingWindowHook": "jaz.hooks.builtin.sliding_window.SlidingWindow",
    "jaz.hooks.builtin.atif_trace.ATIFTraceHook": "jaz.hooks.builtin.atif_trace.ATIFTrace",
    "jaz.hooks.builtin.otel_tracing.OTelTracingHook": "jaz.hooks.builtin.otel_tracing.OTelTracing",
    "jaz.hooks.builtin.jaeger_tracing.JaegerTracingHook": "jaz.hooks.builtin.jaeger_tracing.JaegerTracing",
    "jaz.hooks.builtin.langfuse_tracing.LangfuseTracingHook": "jaz.hooks.builtin.langfuse_tracing.LangfuseTracing",
}


def _coalesce_add(
    target: dict[str, object],
    additions: dict[str, object],
    *,
    effect: str,
    refuse_sandbox: bool = False,
) -> None:
    """Union ``additions`` into ``target`` with the family's shared **conflict-error** rule
    (``AddInputs`` / ``AddVariables``): two hooks adding the same key with the *same* value
    coalesce; the same key with a *different* value raises ``REPLInputConflictError``. Making a
    divergent add loud (rather than last-write-wins) keeps composition order-independent, exactly
    like ``BlackboardWrite``. ``refuse_sandbox`` **raises** ``SandboxKeyError`` if ``__builtins__``
    is named (mirrors ``DropVariables``), so a namespace/input add can't reopen the compiler sandbox
    — a loud refusal rather than a silent skip that would hide the hook bug.

    Conflict test is identity-first, then a *guarded* ``!=`` (see ``_values_conflict``): two distinct
    objects that don't define ``__eq__`` compare unequal (a genuine conflict), which is the safe
    default for injected values.
    """
    for key, value in additions.items():
        if refuse_sandbox and key == "__builtins__":
            raise SandboxKeyError(
                f"{effect} cannot bind '__builtins__' — it holds the compiler sandbox "
                "(#688/#690); binding it would reopen import/attribute/builtins policy."
            )
        if key in target and _values_conflict(target[key], value):
            raise REPLInputConflictError(
                f"{effect} conflict: '{key}' was added with two different values by "
                f"different hooks (composition must be order-independent)."
            )
        target[key] = value


def _values_conflict(existing: object, incoming: object) -> bool:
    """Whether two same-key adds diverge (raise) or coalesce (skip), crash-free.

    Identity first: the same object never conflicts with itself — cheap, and the common case when
    one shared value reaches composition twice. Otherwise compare with ``!=``, but guard the result:
    a value whose ``__eq__``/``__ne__`` returns a non-bool (e.g. a numpy array, whose ``!=`` yields
    an array and whose truthiness raises ``ValueError``) must not crash composition. Two *distinct*
    such objects can't be proven identical, so the safe default is to treat them as a conflict (fail
    loud) rather than silently coalescing or aborting the whole invoke with an ambiguous-truth error.
    """
    if existing is incoming:
        return False
    try:
        return bool(existing != incoming)
    except (ValueError, TypeError):
        return True


def _is_json_safe(value: object) -> bool:
    """Whether ``value`` survives ``json.dumps`` — the bar for a serialized hook param.

    Used by the generic dataclass ``to_dict`` to decide, per field, whether a value can be
    recorded as-is. A field that isn't JSON-safe and has no ``metadata["to_dict"]`` encoder is
    omitted (and round-trips to its constructor default) rather than crashing serialization.

    This is a *predicate*, NOT a transform: ``to_dict`` records the original Python value, not
    ``json.dumps(value)``. So ``Hook.to_dict``/``Hook.from_dict`` are dict↔object (no JSON *string*
    on either side — that boundary, ``json.dumps``/``json.loads``, belongs to whoever persists the
    dict), and ``from_dict(to_dict(h))`` in memory is exactly symmetric: a value is stored and read
    back as the same object (e.g. a tuple stays a tuple). The one caveat is inherent to JSON, not an
    asymmetry here: a value that is json-safe but not a native JSON *type* (a tuple, an int-keyed
    dict) is retyped by a round-trip *through a JSON string* (tuple→list, int keys→str) — so a
    persisted round-trip can differ from the in-memory one. No built-in hook field hits this (all
    scalars); a field needing exact persisted round-trip should declare a ``metadata`` encoder/
    decoder pair, which controls both directions.
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _log_incomplete_span(exceptional: bool, *, debug_msg: str, error_msg: str) -> None:
    """Log a span that closed without ``complete()``, at the level its cause warrants.

    Shared by all three ``span_*`` context managers so a fourth span can't grow its own
    hand-rolled copy of this debug-vs-error split (each still passes its own messages).

    An *exceptional* exit — the span body unwound on an in-flight exception (LLM/provider
    failure, ``KeyboardInterrupt``) — is expected, not a bug: the caller never had a result
    to ``complete()`` with, so the loud "you must call complete()" text points at the wrong
    culprit (observed live: Ctrl-C in the interactive console printed it mid-interrupt). It
    is logged at debug and the propagating exception is left to be the real signal. A
    *clean* exit that simply forgot to ``complete()`` is the genuine programming error and
    stays at error.
    """
    if exceptional:
        logger.debug(debug_msg)
    else:
        logger.error(error_msg)


class Hook:
    """Base class for hooks — subclass it and override the handlers you care about.

    Every event has a handler named after it (:class:`LLMQueryEnter` → ``on_llm_query_enter``).
    A handler observes the event and returns a list of effects; returning effects is the only
    way a hook influences the run, and which effects are honored depends on the event — each
    event class documents its own set. A cross-cutting observer (logging, tracing) overrides
    ``on_any`` instead, which runs *in addition to* the matched typed handler, so the two
    compose.

    There are two ways to activate a hook, and **they differ in whether nested invokes see
    it.** A hook activated either way must not be activated the other way at the same time.

    ``with`` — propagating::

        with MyHook():
            invoke(...)           # active here AND in every invoke nested inside

        with Hook1(), Hook2():    # several at once
            invoke(...)

    It reaches nested invokes, functions called inside the block, async code, and threads
    (each gets its own context copy).

    Leading positional argument — one invoke only::

        invoke(MyHook(), task=...)   # active for THIS invoke, NOT for invokes it nests

    A hook passed this way applies to that invoke alone — the invokes it nests never see it, so
    anything it accumulates or enforces (a budget, a limit, a log) covers the outer invoke's
    own turns and nothing below them. Use ``with`` when sub-invokes must be included.

    For a list of hooks known only at runtime, use ``contextlib.ExitStack``, which unwinds in
    reverse order and hands the in-flight exception to each hook's ``__exit__`` exactly as
    nested ``with`` blocks do::

        with ExitStack() as stack:
            for hook in hook_list:
                stack.enter_context(hook)
            invoke(...)

    Examples:
        from jaz.hooks import Hook
        from jaz.hooks.effects import Abort, Effect
        from jaz.hooks.events import LLMQueryEnter

        class StopAfterFiftyTurns(Hook):
            def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
                if event.iteration > 50:
                    return [Abort(error=RuntimeError("Limit exceeded"))]
                return []

        with StopAfterFiftyTurns():
            result = invoke(ReturnType(str), task="task")
    """

    # Maintainers: ``hooks/README.md`` is the contributor guide — design philosophy, the
    # contextvar/event-flow architecture, ``HookDispatcher``, and how to add a new event,
    # effect or hook. It is deliberately not linked from the docstring above: it is a repo
    # path a reader of the published reference cannot follow, and its content is for people
    # changing the hook system rather than using it.
    #
    # The docstring links nowhere else either. The website's Hooks guide
    # (``/docs/guides/hooks``) was the obvious candidate and is currently unusable: it lists six
    # hooks that do not exist (``WorkflowStrategyHook``, ``MemoryStoreHook``,
    # ``BudgetTrackingHook``, ``JaegerTracingHook``, ``LangfuseTracingHook``, plus
    # ``WorkflowReplayHook`` under a stale name), and every code sample is broken — the removed
    # ``return_type=`` keyword, a positional prompt instead of ``task=``, an ``on_event`` handler
    # that no longer exists, and effects/events (``ReplIterationEnter``, ``AddInstructionPrompt``)
    # that were never in this API. Restore the link once that page is rewritten; until then it
    # would send readers somewhere strictly worse than this docstring.

    # Auto-populated registry of every Hook subclass, keyed by fully-qualified name for
    # reconstruction by :meth:`from_dict` (#727 follow-up). ``__init_subclass__`` records each class.
    # The registry is only ever *matched* against a serialized ``qualified_name`` — it NEVER imports
    # by name (arbitrary import-by-name is a footgun), so a class must already be imported to be
    # resolvable. Built-in hooks are covered because importing ``jaz.hooks`` imports
    # ``jaz.hooks.builtin`` (which imports all of them); a custom hook is resolvable once its module
    # has been imported. Resolution is by qualified name ONLY — the short ``class`` name that
    # ``to_dict`` also emits is an informational label, not a reconstruction key (every dict this
    # code produces carries ``qualified_name``, so there is no short-name fallback to maintain).
    #
    # A ``WeakValueDictionary`` so a subclass that is no longer referenced elsewhere (one defined
    # inside a function or a test) is evicted when garbage-collected rather than pinned for the life
    # of the process; module-level hook classes are held alive by their module and stay registered.
    _registry_by_qualname: ClassVar["WeakValueDictionary[str, type[Hook]]"] = (
        WeakValueDictionary()
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Hook._registry_by_qualname[f"{cls.__module__}.{cls.__qualname__}"] = cls

    def _dispatch_event(self, event: Event) -> list[Effect]:
        """The framework event router: dispatch an event to its typed ``on_<event>``
        handler (the override points below).

        **Do not override this.** It is the dispatcher's internal router; overriding it
        replaces the routing, so any ``on_<event>`` handlers would silently never run
        (#597). To extend a hook, override the typed handlers below — you get typed
        attribute access and implement only the events you care about, with no
        ``isinstance``/``match`` boilerplate and no "remember to ``return []``" footgun.

        For a genuinely *cross-cutting* observer that treats every event uniformly (loggers,
        tracing), override :meth:`on_any` instead: the dispatcher calls it *in addition to*
        the matched typed handler, so it composes with — rather than replaces — the
        per-event dispatch, and a hook may use both.

        Returns the effects expressing how to influence execution (empty for none).
        """
        match event:
            case InvokeEnter():
                return self.on_invoke_enter(event)
            case InvokeExit():
                return self.on_invoke_exit(event)
            case REPLExecEnter():
                return self.on_repl_exec_enter(event)
            case REPLExecExit():
                return self.on_repl_exec_exit(event)
            case LLMQueryEnter():
                return self.on_llm_query_enter(event)
            case LLMQueryExit():
                return self.on_llm_query_exit(event)
            case LLMQueryRetry():
                return self.on_llm_query_retry(event)
            case _:
                # Unreachable for a correctly-wired build: the event taxonomy is closed
                # and every type is enumerated above. Hitting this means a new Event type
                # was added without wiring its `case` + typed handler. Raising a
                # _JazInternalError surfaces the gap loudly for *every* hook: emit()'s
                # `except _JazInternalError: raise` re-raises it ahead of the `except Exception`
                # that would otherwise log-and-continue, so it propagates (fails tests / aborts
                # the invoke) rather than being no-op'd (the footgun typed handlers exist to
                # kill). That clause is narrowed to the internal error on purpose: this is the
                # dispatcher's own invariant, not a hook electing to abort — a hook doing that
                # must emit `Abort(error=...)`, which raising deliberately no longer substitutes
                # for (see emit's handler).
                raise _JazInternalError(
                    f"Hook._dispatch_event has no dispatch case for event type "
                    f"{type(event).__name__!r}; add a `case` and an `on_<event>` handler."
                )

    # Typed per-event handlers — the primary extension API. Override the ones you care
    # about; each defaults to a no-op. (Hooks that override `_dispatch_event` bypass these.)
    def on_invoke_enter(self, event: InvokeEnter) -> list[Effect]:
        """An invoke is starting, before any LLM query. Default no-op."""
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        """An invoke has finished (terminal result produced). Default no-op."""
        return []

    def on_repl_exec_enter(self, event: REPLExecEnter) -> list[Effect]:
        """REPL code is about to execute (parsed code available). Default no-op."""
        return []

    def on_repl_exec_exit(self, event: REPLExecExit) -> list[Effect]:
        """REPL code has executed (result available, at effect composition). Default no-op."""
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        """An LLM query is about to be issued. Default no-op."""
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        """An LLM query has returned. Default no-op."""
        return []

    def on_llm_query_retry(self, event: LLMQueryRetry) -> list[Effect]:
        """An LLM query is being retried after a failure. Default no-op."""
        return []

    def on_any(self, event: Event) -> list[Effect]:
        """Catch-all called for *every* event, in addition to the matched typed handler.

        Override this for cross-cutting observers (loggers, tracing, replay) that treat all
        events uniformly, where N separate typed handlers would be worse than one ``match``.
        It *composes* with the typed handlers rather than replacing them — the effects of both
        are collected — so a hook may implement specific ``on_<event>`` handlers *and* an
        ``on_any`` without either clobbering the other. Default no-op.
        """

        # The router it composes with is :meth:`_dispatch_event`; overriding *that* replaces the
        # routing and silently disables every ``on_<event>`` handler (#597), which is why this
        # exists as the supported extension point for uniform observers.
        return []

    # Per-invoke blackboard contract (see ``hooks/blackboard.py``). Two declarative
    # slots, both default-empty, that the core consults — it never introspects
    # arbitrary hook fields. Hooks that read or seed the blackboard override these.

    #: Keys this hook *reads* from ``event.blackboard``, mapped to a one-line help
    #: string. This is the discoverable menu of per-invoke keys that seed/write linting
    #: is checked against: a key seeded or written by a producer but declared by *no*
    #: active hook's ``blackboard_consumes`` is a likely typo/orphan and *warns* (not
    #: raises — see :meth:`HookDispatcher._warn_orphan_keys`). The reverse is **not**
    #: checked or enforced: declaring a key here that no hook ever produces is silent,
    #: and reading it just yields absence (``.get`` -> ``None`` / ``[]`` -> ``KeyError``;
    #: see :class:`~jaz.hooks.blackboard.Blackboard`). This attribute documents read
    #: intent and drives producer-side linting — it does not require a producer to exist.
    blackboard_consumes: ClassVar[dict[str, str]] = {}

    def blackboard_seed(self) -> dict[str, object]:
        """Return this hook's contribution to the blackboard's generation 0.

        Merged into the seed *before* :class:`InvokeEnter` is dispatched, so seeded keys
        are visible to every event including the first. Must be a pure function of
        the hook's **own state** (its constructor args) — it receives no invoke
        context (prompt/config/inputs). A value that must be *computed from the
        invoke's parameters* is not seedable and belongs at :class:`InvokeEnter` as an
        effect; keeping seeds own-state-only is what avoids a pre-:class:`InvokeEnter`
        dispatch (and the generation-regress it would imply). Default: no seed.
        """
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this hook to a JSON-safe identity + parameters dict.

        Named ``to_dict`` (a plain hook→dict serializer, like ``InvokeNode.to_dict``) — this is
        **not** related to Python's *descriptor protocol*.
        Two uses:

        - **Observability**: the observability hooks serialize each active hook via this at their
          own edge (at :class:`InvokeEnter`) to record *what governance applied here* into their
          trace. ``Event.hooks`` itself carries the LIVE hook objects, not pre-serialized dicts,
          so serialization happens at the log boundary — here — not on the event field.
        - **Round-trip partner**: the serialized form is what :meth:`from_dict` reconstructs, by
          matching ``class`` (or ``qualified_name``) against the subclass registry (never imports
          ``qualified_name`` — import-by-name is a footgun). ``Config.baseline_hooks`` was the live
          caller of that reconstruction until #959 removed it; see :meth:`from_dict` for why the
          round-trip is kept regardless.

        Returns ``{"class": <short name>, "qualified_name": <module.qualname>, "params": {...}}``.
        ``class`` is the reconstruction key (kept short for back-compat with existing serialized
        configs); ``qualified_name`` disambiguates same-named hooks across modules.

        **Params are derived automatically for a hook that is a dataclass** — the common,
        boilerplate-free path (a hook opts in simply by being ``@dataclass(eq=False)``). Each
        ``init`` field is recorded: a field with a ``metadata["to_dict"]`` encoder is passed through
        it (a ``None`` result *omits* the key — how a non-round-trippable value falls back to its
        default, e.g. a callable ``must_exit_warning``); an unadorned JSON-safe value is recorded
        as-is; a non-JSON-safe value with no encoder is omitted (it will round-trip to its
        constructor default). ``field(init=False)`` runtime state is skipped. A *non-dataclass* hook
        falls back to :meth:`_to_dict_params` (the explicit escape hatch, still supported).

        Recomputed on every call — deliberately **NOT** memoized. The dict is a pure function of the
        construction params only *if* those params are never mutated after ``__init__``, which the
        base does not enforce (nothing freezes ``self.max_iterations`` etc.); a cache would go
        silently stale the moment a caller mutated a param. It is cheap and no longer on any hot path
        — ``Event.hooks`` is the live set stamped as a plain reference tuple, so only the
        :class:`InvokeEnter`-time observability consumers call this.
        """
        params = (
            self._dataclass_params() if is_dataclass(self) else self._to_dict_params()
        )
        return {
            "class": type(self).__name__,
            "qualified_name": f"{type(self).__module__}.{type(self).__qualname__}",
            "params": params,
        }

    def _dataclass_params(self) -> dict[str, Any]:
        """Derive serialized params from this hook's dataclass fields (see :meth:`to_dict`)."""
        params: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]  # guarded by is_dataclass at the call site
            if not f.init:
                continue  # field(init=False) = runtime state, not a construction param
            value = getattr(self, f.name)
            encoder = f.metadata.get("to_dict")
            if encoder is not None:
                encoded = encoder(value)
                if encoded is None:
                    continue  # encoder signals "omit" (value not round-trippable → default)
                params[f.name] = encoded
            elif _is_json_safe(value):
                params[f.name] = value
            else:
                # No encoder and not JSON-safe: omit rather than crash. It round-trips to the
                # field's constructor default. A hook that needs the value preserved declares a
                # ``metadata={"to_dict": ..., "from_dict": ...}`` encoder pair on the field.
                logger.debug(
                    "Hook %s: omitting non-JSON-safe field %r from to_dict() (no encoder)",
                    type(self).__name__,
                    f.name,
                )
        return params

    def _to_dict_params(self) -> dict[str, Any]:
        """Escape-hatch params for a NON-dataclass hook's :meth:`to_dict` (default: none).

        Only consulted when the hook is not a dataclass (a dataclass derives its params from fields
        automatically). Override to declare the constructor arguments worth recording — JSON-safe
        values only, and only *frozen* params (not mutable runtime state). Prefer making the hook a
        ``@dataclass(eq=False)`` instead, which gets this for free.
        """
        return {}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Hook":
        """Reconstruct a hook from its :meth:`to_dict` form.

        Resolves ``d["qualified_name"]`` against the subclass registry — the class must already be
        imported (never imported by name). ``qualified_name`` is required (every dict :meth:`to_dict`
        produces carries it; the short ``class`` name is informational only). Reads the params from
        ``d["params"]`` (a missing ``params`` is treated as no params). For a dataclass target, each
        field's ``metadata["from_dict"]`` decoder is applied to its param before construction (the
        symmetric partner of the ``to_dict`` encoder — e.g. ``must_exit_warning``). Then
        ``target(**params)`` builds the instance.

        Raises ``ValueError`` for a missing/unknown ``qualified_name`` or params the constructor
        rejects.

        **No in-tree caller — kept deliberately.** The only call site was ``Config._validate``'s
        ``baseline_hooks`` branch, deleted when that field was removed. Even before then the
        branch was unreachable in practice: the eval harness unconditionally overwrote
        ``kwargs["baseline_hooks"]`` with freshly-constructed :class:`Hook` instances before
        ``jaz.configure()``, so validation always took the ``isinstance(entry, Hook)`` path
        and never the serialized-dict one.

        Maintainer's call (2026-07-27), recorded because it is not recoverable from the code:
        **``from_dict`` stays for exactly as long as ``to_dict`` exists.** The two are one
        contract, not two functions — ``to_dict`` writes hook identity + params into artifacts
        that outlive the process (ATIF traces, conversation-history records, OTel span
        attributes, replayed log lines), and a serialization format nothing can read back is a
        write-only format. The alternative considered and rejected was deleting it as dead code
        under the YAGNI pass that removed ``max_repl_invoke_calls`` and ``baseline_hooks``: that
        would strand every already-written trace and quietly convert ``to_dict`` from a
        round-trippable record into a debug string. So: if ``to_dict`` is ever dropped, drop this
        with it; while ``to_dict`` lives, "no callers" is not grounds for removal. The same
        reasoning is why :data:`_LEGACY_HOOK_QUALNAMES` exists — a rename must not orphan
        artifacts this method is meant to be able to read.
        """
        name = d.get("class")
        target = cls._resolve_registered(name, d.get("qualified_name"))
        params = dict(d.get("params", {}))
        if is_dataclass(target):
            field_meta = {f.name: f.metadata for f in fields(target)}
            for key in list(params):
                meta = field_meta.get(key)
                decoder = meta.get("from_dict") if meta is not None else None
                if decoder is not None:
                    params[key] = decoder(params[key])
        try:
            return target(**params)
        except TypeError as e:
            raise ValueError(f"Invalid params for hook {name!r}: {e}") from e

    @classmethod
    def _resolve_registered(
        cls, name: str | None, qualified_name: str | None
    ) -> type["Hook"]:
        """Map a serialized ``qualified_name`` to a registered subclass (no import).

        ``name`` (the short ``class``) is used only for clearer error messages.
        """
        if qualified_name is None:
            raise ValueError(
                f"serialized hook dict (class {name!r}) is missing 'qualified_name', which is "
                f"required to reconstruct it."
            )
        target = Hook._registry_by_qualname.get(qualified_name)
        if target is None:
            # Renamed since the trace was written? `to_dict` stamps the class name into
            # durable artifacts (ATIF traces, conversation-history records, OTel span
            # attributes, log lines), so a rename silently orphans every run recorded
            # before it. Consulting the alias map keeps old runs reconstructible; add an
            # entry here whenever a hook class is renamed.
            renamed = _LEGACY_HOOK_QUALNAMES.get(qualified_name)
            if renamed is not None:
                target = Hook._registry_by_qualname.get(renamed)
        if target is None:
            raise ValueError(
                f"Unknown hook {qualified_name!r}. Is its module imported?"
            )
        return target

    def setup(self) -> None:  # noqa: B027 - optional override point, intentionally a no-op default
        """Acquire resources for this hook (open files, start providers, ...).

        Default no-op. Override in stateful hooks instead of overriding
        ``__enter__``. This runs on whichever activation path is used:

        - the ``with MyHook():`` context-manager path (via ``__enter__``), and
        - the per-invoke ``local_hooks=[...]`` path (driven explicitly by
          ``jaz.invoke``), which deliberately does NOT touch the contextvar.

        A given instance is active via exactly ONE path at a time — activating it
        via both at once raises :class:`HookActivationError` — so ``setup()``
        runs once per activation and a stateful hook need not make it idempotent.
        Separating resource lifecycle from contextvar registration is what lets
        ``local_hooks`` run cleanup without re-introducing propagation.
        """

    def teardown(self, exc: BaseException | None = None) -> None:  # noqa: B027 - optional override point, intentionally a no-op default
        """Release resources for this hook (flush/close files, end spans, ...).

        Default no-op. Override in stateful hooks instead of overriding
        ``__exit__`` (and do NOT call ``super().__exit__`` — contextvar
        de-registration is handled by the base ``__exit__``). Runs on both the
        ``with`` and ``local_hooks`` activation paths.

        ``exc`` is the exception that propagated out of the scope (or ``None``
        on clean exit), mirroring the exception ``__exit__`` would have seen.
        """

    def __enter__(self):
        """Enter hook context - runs setup() and adds this hook to active hooks.

        Raises ``HookActivationError`` if this exact instance is already active —
        either on the contextvar via ``with`` (a nested ``with h: with h:``), or as a
        still-live ``local_hooks`` instance in an enclosing invoke (``local_active``) —
        which would otherwise run ``setup()`` twice and double-register it. The check is
        by identity, so distinct instances of the same class nest freely. Raised
        *before* ``setup()`` and before the contextvar token is taken, so a rejected
        activation leaves no partial state.
        """
        current = get_current_hooks()
        if any(h is self for h in current.hooks):
            raise HookActivationError(
                f"{type(self).__name__} instance is already active via `with`; "
                "re-entering the same instance would run setup()/teardown() twice. "
                "Activate a hook instance via exactly one scope at a time."
            )
        if id(self) in current.local_active:
            raise HookActivationError(
                f"{type(self).__name__} instance is already active via local_hooks in "
                "an enclosing invoke; entering it via `with` would run setup()/teardown() "
                "twice while its local lifecycle is still live. Use one scope at a time."
            )
        # The `with` path reaches setup() THROUGH __enter__ (and registers on the contextvar
        # below → propagates). The local_hooks path calls setup() directly (no __enter__, no
        # contextvar); baseline never calls setup(). RecursionLimit keys on this distinction to
        # accept `with`/baseline and reject local_hooks — do not route local activation through
        # __enter__ without updating that guard (see _activate_local_hooks in invoke.py).
        self.setup()
        new_context = get_current_hooks().with_hook(self)
        self._token = _set_hook_context(new_context)
        return self

    def __exit__(self, _exc_type, exc_val, _exc_tb):
        """Exit hook context - restores previous context, then runs teardown().

        De-registration happens before teardown(); ordering is immaterial since
        resource cleanup does not depend on contextvar membership.
        """
        # `_token` is only set by __enter__. A hook used as a plain instance and
        # exited manually (or never entered) has nothing to de-register.
        if hasattr(self, "_token"):
            _reset_hook_context(self._token)
        self.teardown(exc_val)
        return False


class HookDispatcher:
    """Orchestrator that reads hooks and composes their effects into execution contexts.

    The HookDispatcher is responsible for:
    1. Reading active hooks (from contextvar and from its bound local hooks)
    2. Emitting events to active hooks
    3. Collecting effects from all hooks
    4. Composing effects into a typed ExecutionContext with explicit rules
    5. Resolving supply/transform/terminate effects (OverrideResult / ModifyResult / Abort) at REPL execution
    6. Managing span lifecycle for enter/exit event pairs

    Design principles:
    - Hooks are called in registration order (first registered, first called)
    - Composition rules are explicit and documented
    - Invalid effects for an event are ignored with a warning
    - Each event type has a specific ExecutionContext type

    Two channels of hooks
    ---------------------
    Two channels of hooks flow into every `emit()`:

    1. **Propagating hooks** — read from the `_hook_context` contextvar on
       every `emit()`. Set by `with SomeHook(...)` blocks. These naturally
       propagate to nested invokes via contextvar semantics.

    2. **Local hooks** — bound to a specific dispatcher *instance* at
       construction via `local_hooks=...`. Used to scope hooks to a single
       `jaz.invoke()` call without propagating to nested invokes.

    Dispatcher lifetime
    -------------------
    - `Agent` now *always* constructs its own per-invoke `HookDispatcher`
      (regardless of whether `local_hooks` is empty), because the instance carries
      the invoke's seeded blackboard — per-invoke state a shared singleton cannot
      hold. Each Agent owns its dispatcher for the lifetime of its `_invoke()` call.
    - The `get_dispatcher()` singleton is retained only for external / direct
      callers (tests, and the README examples) that emit without an Agent; it has
      no local hooks and no seeded board.

    Why per-invoke instances rather than a second contextvar (Candidate C)
    ---------------------------------------------------------------------
    Two alternative designs were considered:

    - **Second contextvar.** Add a `_local_hook_context` contextvar and
      reset it at every `_invoke()` entry to prevent leakage to nested
      invokes. Pros: dispatcher API unchanged. Cons: non-propagation depends
      on a fragile "always reset on entry" trick that any new `_invoke()`
      entry point would have to remember; contextvars semantically suggest
      propagation, which is the opposite of what local hooks want.

    - **Thread `local_hooks` through every emit/span call.** Add a
      `local_hooks=` parameter to `emit()` and all `span_*` methods. Pros:
      fully explicit data flow. Cons: touches ~5 method signatures plus
      every call site in `Agent`; each new dispatcher method has to remember
      the parameter.

    - **Per-hook `recurse=` flag.** Instead of two channels, give every hook
      a propagation flag — `PrintLogger(recurse=False)` for local,
      `PrintLogger(recurse=True)` for deep — so a single `with` (or list)
      form covers both and switching shallow↔deep is just toggling a bool.
      Pros: more unified surface, arguably neater at the call site. Cons:
      every hook has to thread the flag through `__init__`/`__enter__`, and
      propagation becomes per-hook state rather than a property of how the
      hook is activated; the dispatcher would still need to partition hooks
      by the flag at emit time. Worth revisiting if the two-channel split
      proves awkward in practice.

    The per-invoke-instance approach won because:

    - **Structural non-propagation.** A nested `jaz.invoke()` re-enters
      `_invoke()` → constructs a fresh `Agent` → constructs a fresh
      `HookDispatcher` with the nested call's own `local_hooks` (or the
      singleton if none). The outer invoke's local hooks are not in scope
      because they live on a different object. No reset trick required.

    - **Thread- and async-safe by construction.** `_local_hooks` is set
      once at construction and is an immutable tuple. Each invoke owns its
      own dispatcher, so two concurrent invokes (in threads or asyncio
      Tasks) never share local-hook state. The propagating contextvar
      channel uses standard `contextvars` semantics: an asyncio Task copies
      the enclosing context when it is created, but a raw `threading.Thread`
      does NOT inherit it — the new thread starts from the default context
      unless the caller explicitly runs the target via
      `contextvars.copy_context().run(...)`.

    - **No API churn.** Only `__init__` and `emit()` change; every span_*
      method and call site stays as-is.

    Limitation worth knowing
    ------------------------
    Code that calls `get_dispatcher().emit(...)` directly (bypassing the
    Agent's `self.dispatcher`) will not see local hooks or a seeded blackboard,
    because the singleton has neither. `Agent` no longer calls `get_dispatcher()`
    at all (it always builds a per-invoke instance); the singleton is now reached
    only by tests and the README examples. If a future component needs
    invoke-bound local hooks visible to arbitrary call sites within an
    invoke's call stack, it should reach for `agent.dispatcher` rather than
    the singleton (or we should reconsider the second-contextvar design at
    that time).

    Future async/multi-thread notes
    -------------------------------
    - Per-Agent dispatcher instances make concurrent invokes trivially
      isolated: spawning `jaz.invoke()` in multiple threads / asyncio Tasks
      gives each its own Agent, its own dispatcher, its own local hooks.
    - Propagating hooks flow into asyncio Tasks automatically (a Task
      snapshots the context at creation). They do NOT flow into a raw
      `threading.Thread`, which starts from the default context; code that
      spawns worker threads must carry hooks across with
      `contextvars.copy_context().run(...)`.
    - If we ever introduce parallel sub-invokes within a single Agent
      (e.g., `asyncio.gather` of nested `jaz.invoke()` calls), each child
      `_invoke()` still constructs its own Agent + dispatcher, so isolation
      holds. The only shared state across the parallel children would be
      the propagating contextvar, which is the desired behavior.

    TODO: Add setting to change how dispatcher handles invalid effects (warn vs. error)
    """

    def __init__(self, local_hooks: tuple["Hook", ...] = ()) -> None:
        # Frozen at construction. Agent always constructs a per-invoke instance (it
        # carries the seeded blackboard); get_dispatcher() returns the shared singleton
        # only for external/direct callers with no local hooks or seeded board.
        self._local_hooks: tuple[Hook, ...] = local_hooks
        # The invoke's per-call blackboard, seeded before InvokeEnter via
        # seed_blackboard(). None on the shared singleton / before seeding: emit()
        # then leaves each event's own (empty) default board untouched.
        self._blackboard: Blackboard | None = None
        # The union of active hooks' blackboard_consumes, snapshotted at seed time so
        # emit() can lint writes against the same set without re-walking hooks per
        # event. None until seeded — an unseeded dispatcher skips the write lint, as it
        # skips the board stamp above.
        self._accepted_keys: set[str] | None = None
        # Orphan keys already warned about this invoke — lint each at most once (writes
        # recur per generation; seeds run once). Per-invoke, since the dispatcher is.
        self._warned_orphan_keys: set[str] = set()

    def _active_hooks(self) -> list[Hook]:
        """The hooks active for an invoke, in dispatch order (baseline, then
        propagating (contextvar), then local).

        Baseline is resolved from the given EFFECTIVE config — NOT ambient
        get_config() — so a ConfigOverride / depth-specific config / worker-thread
        invoke uses the baseline it actually configured (#463).

        No dedup is needed across propagating and local: a hook instance can be active
        in at most one live scope. Every overlap shape raises ``HookActivationError`` at
        activation time (see ``Hook.__enter__`` and ``invoke._activate_local_hooks``,
        #533 / #540) — ``with``+``with``, ``with``+local, ``[h, h]`` in one call, AND a
        ``with``/local re-activation nested inside an invoke where the instance is
        already a live local (this last caught via the ``HookContext.local_active`` id
        set the contextvar carries for a local's dynamic extent). So the streams are
        disjoint by construction and this stays a plain concatenation; the old identity
        filter here was a symptom of tolerating that overlap.
        """
        propagating = list(get_current_hooks().hooks)
        local = list(self._local_hooks)
        return propagating + local

    def seed_blackboard(
        self, config: "Config", caller_metadata: dict[str, object] | None = None
    ) -> Blackboard:
        """Assemble the invoke's blackboard (generation 0) and bind it to this
        dispatcher, so subsequent ``emit()`` calls stamp it onto every event.

        Called once per invoke, *before* :class:`InvokeEnter` is dispatched. The seed is
        built in plain code (it is NOT an event — that would need a generation to
        read from, a regress) from two own-state-only sources:

        - each active hook's ``blackboard_seed()`` payload, and
        - ``caller_metadata`` (the caller-supplied per-invoke data), which wins on
          key conflicts.

        Two hooks seeding one key with different values raises
        :class:`BlackboardSeedError` — an irresolvable, order-dependent ambiguity. A
        seed key no active hook consumes only *warns* (see :meth:`_warn_orphan_keys` for
        why orphan producers are a lint, not a failure); writes are linted identically
        in :meth:`emit`.
        """
        active = self._active_hooks()
        accepted = {k for h in active for k in h.blackboard_consumes}
        # Snapshot so emit() lints writes against the same consumes set (see __init__).
        self._accepted_keys = accepted

        seed: dict[str, object] = {}
        for hook in active:
            for key, value in hook.blackboard_seed().items():
                if key in seed and seed[key] != value:
                    raise BlackboardSeedError(
                        f"Conflicting blackboard seed for key {key!r}: "
                        f"{seed[key]!r} != {value!r}. Two hooks seed the same key "
                        "with different values; this cannot be resolved without "
                        "relying on hook order."
                    )
                seed[key] = value

        # Caller metadata overlays hook seeds (the caller is authoritative).
        if caller_metadata:
            seed.update(caller_metadata)

        self._warn_orphan_keys(set(seed), accepted, origin="seed")

        self._blackboard = Blackboard(seed)
        return self._blackboard

    def _warn_orphan_keys(
        self, keys: set[str], accepted: set[str], origin: str
    ) -> None:
        """Warn (do not raise) about board keys no active hook consumes.

        Executive decision (thread on #542): orphan producers are a **lint, not a
        failure**, applied *identically* to seeds and writes. The earlier design
        validated seed orphans by raising but left writes unchecked — an asymmetry with
        no principled basis, since "require a live consumer" couples a producer to the
        ambient hook set the same way whether it seeds or writes. Producing board data
        whose consumer isn't active is legitimate (an optional or not-yet-activated
        consumer), so failing loud would be wrong for both; warning catches the real
        target — a typo'd / dead key — without aborting the invoke or blocking a
        standalone producer. Value *conflicts* (two different values for one key) still
        raise: that is an irresolvable ambiguity, not an orphan.

        Each orphan key is warned at most once per invoke (writes recur every
        generation), tracked on ``self._warned_orphan_keys``.
        """
        unknown = sorted(keys - accepted - self._warned_orphan_keys)
        if not unknown:
            return
        self._warned_orphan_keys.update(unknown)
        menu = ", ".join(sorted(accepted)) or "(none)"
        logger.warning(
            "Blackboard %s key(s) %s are not consumed by any active hook (declared via "
            "Hook.blackboard_consumes: %s) — a likely typo or orphan datum. This is a "
            "lint, not an error: producing board data whose consumer isn't active "
            "(optional or not-yet-activated) is allowed. Seeds and writes are treated "
            "identically; only value conflicts for a key raise.",
            origin,
            unknown,
            menu,
        )

    def emit(self, event: Event) -> list[Effect]:
        """Emit an event to all active hooks (baseline + propagating + local) and collect effects.

        Args:
            event: The event to emit

        Returns:
            List of all effects returned by hooks, EXCLUDING ``BlackboardWrite``
            (consumed here, see below). Order is: baseline hooks (from
            Config.baseline_hooks), then propagating hooks in registration order,
            then local hooks in constructor order. Order is irrelevant for
            commutative effects.

        Blackboard stamping + generational write barrier:
            If this dispatcher has a seeded blackboard, it is stamped onto the event
            so every hook reads the same per-invoke board. ``BlackboardWrite``
            effects are collected during the hook loop and applied to the board only
            *after* the loop — so they are invisible to this event's reads and
            surface to the next event (the next generation). This is what keeps
            producer/consumer hooks order-independent.
        """
        # Stamp the invoke's live board onto the event (overriding its empty default)
        # so reads during this event see the current generation. Event is frozen (hooks must
        # treat it as immutable); object.__setattr__ is the sanctioned in-framework escape hatch
        # for the dispatcher's own stamps — the one place allowed to write an event's fields.
        if self._blackboard is not None:
            object.__setattr__(event, "blackboard", self._blackboard)

        effects: list[Effect] = []
        blackboard_writes: list[tuple[str, object]] = []

        active = self._active_hooks()
        # Stamp the LIVE active hook set (dispatch order) onto the event so hooks in the loop below —
        # and any observability consumer — can see the governance active at this event (Event.hooks).
        # The actual Hook instances (#727), not serialized dicts: Event.hooks is the effect system's
        # view of the active set; consumers that log it call Hook.to_dict() at their edge. Keeping it
        # a plain reference tuple means no per-event serialization on this hot path. object.__setattr__
        # because Event is frozen (see the blackboard stamp above).
        object.__setattr__(event, "hooks", tuple(active))
        for hook in active:
            try:
                # Route to the matched typed handler AND the cross-cutting `on_any`
                # catch-all; their effects compose (#597). ``or []`` tolerates observers
                # (e.g. loggers) that fall off the end returning ``None``.
                #
                # `on_any` is dispatched HERE, as a peer of `_dispatch_event`, rather than
                # nested inside the default `_dispatch_event` router — so it still fires even
                # when a hook overrides `_dispatch_event` (which replaces the router). Nesting it
                # would make `on_any` silently vanish under exactly the override footgun
                # this split fixes (#529). This also matches how event frameworks invoke a
                # catch-all (Laravel `*`, Symfony's wildcard dispatcher, Node's
                # EventEmitter): from the dispatcher, alongside the specific handlers —
                # not from within one of them.
                hook_effects = (hook._dispatch_event(event) or []) + (
                    hook.on_any(event) or []
                )
            # A framework invariant violation is not a hook's exception and is always fatal:
            # it is raised by the dispatcher itself while routing (see _dispatch_event's
            # unreachable-event-type case), so it must not be filed under "buggy hook".
            except _JazInternalError:
                raise
            except Exception as e:
                # A plain Exception from a hook is a non-fatal bug: log + continue, so one
                # buggy hook (observer, tracer, user control hook, or even a governance
                # hook) can't abort the run. This replaces the former
                # baseline-vs-non-baseline split (baseline hooks used to fail loud); with
                # the default baseline empty (#465) that rationale is gone.
                #
                # EXECUTIVE CALL (user, 2026-07-30): **raising is not an abort channel.** An
                # `AbortError` from a hook handler is caught here like any other exception and
                # logged, NOT propagated. The point of the hook system is to contain side
                # effects in a small, finite, composable effect set rather than admitting
                # arbitrary code or side effects; letting a hook author reach for `raise` to
                # terminate a run defeats that, because it is a control-flow side effect the
                # effect algebra cannot see, order-independently compose, or validate.
                # `Abort(error=...)` is the supported way, and it is strictly better behaved:
                # it composes with other effects, and it resolves to a terminal `Raise`
                # *through* the span, so `span.complete()` runs and `InvokeExit` still fires
                # (a raise propagates around both — see the AbortError docstring).
                #
                # Trade-off, deliberately accepted (Option B on #562): a governance hook
                # whose _dispatch_event has a *bug* — raising a plain Exception instead of
                # terminating via an effect — is swallowed here, silently disabling its
                # enforcement, rather than failing loud.
                # TODO(#612): make log-vs-propagate for uncaught hook exceptions configurable.
                if isinstance(e, AbortError):
                    # Distinct message: without it, a hook author who used the (now
                    # unsupported) raise channel sees only a generic "raised exception" line
                    # and no hint that the abort was dropped or what to use instead.
                    logger.error(
                        f"Hook {hook.__class__.__name__} raised {type(e).__name__} "
                        f"processing {type(event).__name__}: {e}. Raising is NOT an abort "
                        f"channel — the run was NOT aborted. Emit Abort(error=...) instead.",
                        exc_info=True,
                    )
                else:
                    logger.error(
                        f"Hook {hook.__class__.__name__} raised exception "
                        f"processing {type(event).__name__}: {e}",
                        exc_info=True,
                    )
                continue
            # Partition BlackboardWrite out of the composed-effect stream: it is
            # applied at the boundary below, not by the per-context _compose_*
            # methods (which would warn on an unrecognized effect).
            for effect in hook_effects:
                if isinstance(effect, BlackboardWrite):
                    blackboard_writes.append((effect.key, effect.value))
                else:
                    effects.append(effect)

        # Generational barrier: apply this event's writes only after every hook has
        # run, minting the next generation that the NEXT event will observe. Applied
        # to ``event.blackboard`` — which IS this dispatcher's seeded board (stamped
        # above) during a real invoke, or the event's own throwaway default board on
        # an unseeded dispatcher (a hand-built event in a test), where it is harmless.
        if blackboard_writes:
            # Lint writes against the seed-time consumes set, symmetric with seeds (only
            # possible once seeded; a hand-built event on an unseeded dispatcher has no
            # consumes snapshot). Conflicting values still raise, inside _apply_writes.
            if self._accepted_keys is not None:
                self._warn_orphan_keys(
                    {k for k, _ in blackboard_writes},
                    self._accepted_keys,
                    origin="write",
                )
            event.blackboard._apply_writes(blackboard_writes)

        return effects

    # Context composition helpers (shared across context types)

    def _warn_invalid_repl_result_effects(self, effects: list[Effect]) -> None:
        """Warn that the exec-result effects (OverrideResult / ModifyResult) are not valid
        here.

        ``OverrideResult`` is valid only at ``REPLExecEnter`` (supply a result) and
        ``ModifyResult`` only at ``REPLExecExit`` (transform the result). ``Abort`` is
        deliberately NOT warned: termination is valid at every live event (it is collected,
        not ignored, at LLMQueryEnter). Called from ``_compose_llm_query`` to catch a hook
        that emits a result-scoped effect at the LLM-query boundary, where no execution
        result exists to supply or transform.
        """
        for effect in effects:
            if isinstance(effect, OverrideResult | ModifyResult):
                logger.warning(
                    f"{type(effect).__name__} is not valid for this event (OverrideResult at "
                    "REPLExecEnter, ModifyResult at REPLExecExit; use Abort to terminate "
                    "anywhere), ignoring"
                )

    # Context-specific composition methods

    def _compose_repl_exec(self, effects: list[Effect]) -> REPLExecContext:
        """Compose effects for REPL execution *enter* events.

        Composition rules:
        - OverrideResult (supply a result) / Abort (terminate): collected to (possibly)
          short-circuit execution (see resolve_override_results)
        - AddVariables (bind namespace names): unioned into ``added_variables`` via the shared
          conflict-error rule (identical coalesces, divergent raises); ``__builtins__`` refused.
        - DropVariables (drop namespace names): unioned into ``dropped_variables``
          (order-independent, like DropMessages). ``__builtins__`` — the compiler sandbox
          (#688/#690) — is refused, so a stray drop can't reopen it.

        ``AddInputs``/``DropInputs`` are **not** collected here — they are ``InvokeEnter``-only
        (#481): inputs are per-invoke setup, applied once before the prompt renders. The per-turn
        namespace counterpart is ``AddVariables``/``DropVariables`` here.
        """
        ctx = REPLExecContext()

        # Apply REPL-specific effects
        for effect in effects:
            match effect:
                case OverrideResult() as override:
                    ctx.override_effects.append(override)

                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case AddVariables(variables=variables):
                    # Bind namespace names (raw, no __jaz_get__). Shared conflict-error rule;
                    # __builtins__ refused (mirrors DropVariables) so an add can't reopen the
                    # sandbox. ``Agent`` applies drops FIRST, so a name also in dropped_variables
                    # resolves to the *add* — and drop-then-add of a bound name is a re-bind.
                    _coalesce_add(
                        ctx.added_variables,
                        variables,
                        effect="AddVariables",
                        refuse_sandbox=True,
                    )

                case DropVariables(names=names, allow_missing=allow_missing):
                    # Union of names (order-independent). Never drop the sandbox key:
                    # __builtins__ carries the un-strippable import/attr/builtins allow-lists
                    # (#688/#690), so a DropVariables that names it is refused loudly
                    # (``SandboxKeyError``) rather than reopening the sandbox — a silent skip would
                    # hide the hook bug.
                    if "__builtins__" in names:
                        raise SandboxKeyError(
                            "DropVariables cannot drop '__builtins__' — it holds the compiler "
                            "sandbox (#688/#690); dropping it would reopen import/attribute/"
                            "builtins policy."
                        )
                    names = set(names)
                    ctx.dropped_variables |= names
                    # Track per-name allow_missing so the REPL's missing-target check can exempt
                    # exactly the names a defensive drop opted in (one opt-in exempts the name).
                    if allow_missing:
                        ctx.dropped_variables_allow_missing |= names

                case ModifyResult():
                    logger.warning(
                        "ModifyResult is not valid at REPLExecEnter (it transforms an "
                        "existing result; use OverrideResult to supply one), ignoring"
                    )

        return ctx

    def _compose_repl_exec_exit(self, effects: list[Effect]) -> REPLExecExitContext:
        """Compose effects for REPL execution *exit* events.

        Composition rules:
        - ModifyResult (transform the result) / Abort (terminate): collected to (possibly)
          override the execution result (see resolve_modify_results)
        - All other effects are invalid at exit and ignored with a warning
        """
        ctx = REPLExecExitContext()

        for effect in effects:
            match effect:
                case ModifyResult() as modify:
                    ctx.modify_effects.append(modify)

                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    logger.warning(
                        f"{type(effect).__name__} not valid for REPLExecExit, ignoring"
                    )

        return ctx

    def _compose_invoke_exit(self, effects: list[Effect]) -> InvokeExitContext:
        """Compose effects for invoke *exit* events (#568).

        Symmetric with ``_compose_repl_exec_exit``: collect ``ModifyResult`` (transform the
        terminal result) / ``Abort`` (terminate); every other effect is invalid at this boundary
        and ignored with a warning. Resolved against the terminal result in ``span_invoke`` via
        ``resolve_modify_results`` — this is what lets ``ReturnType`` / ``ValidateReturn`` downgrade
        a wrong-typed / invalid final ``Return`` to a ``Raise`` (the backstop for another hook's
        ``REPLExecExit`` override reinstating a ``Return``).
        """
        ctx = InvokeExitContext()

        for effect in effects:
            match effect:
                case ModifyResult() as modify:
                    ctx.modify_effects.append(modify)

                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    logger.warning(
                        f"{type(effect).__name__} not valid for InvokeExit, ignoring"
                    )

        return ctx

    def _compose_llm_query(self, effects: list[Effect]) -> LLMQueryContext:
        """Compose effects for LLM query events.

        Composition rules:
        - Abort: exceptions collected into ``abort_errors`` — this is the always-present
          per-turn point where loop/budget hard-stops terminate the invoke; resolved to a
          ``Raise`` in ``span_llm_query`` (before the LLM call).
        - DropMessages: indices are unioned (order-independent)
        - OverrideResponse: order-independent — identical overrides compose to
          one; two distinct overrides raise OverrideConflictError (rather than
          resolving the conflict by hook/with-nesting order)
        """
        from jaz._llm_client import LLMResponse

        ctx = LLMQueryContext()

        self._warn_invalid_repl_result_effects(effects)

        # Track the winning override effect so a later one can be compared for
        # equality — the resolution must not depend on arrival order.
        override_effect: OverrideResponse | None = None

        for effect in effects:
            match effect:
                case Abort(error=error):
                    ctx.abort_errors.append(error)
                case DropMessages(indices=indices, persistent=persistent):
                    edits = ctx.message_edits
                    target = (
                        edits.persistent_drops if persistent else edits.transient_drops
                    )
                    target.update(indices)
                case AddMessages(persistent=persistent) as add:
                    edits = ctx.message_edits
                    bucket = (
                        edits.persistent_adds if persistent else edits.transient_adds
                    )
                    bucket.append(add)
                case OverrideResponse() as override:
                    if override_effect is not None and override != override_effect:
                        raise OverrideConflictError(
                            "Multiple conflicting OverrideResponse effects for "
                            "the same LLM query. Override composition is "
                            "order-independent: distinct overrides cannot be "
                            "resolved without relying on hook registration / "
                            "with-nesting order. Ensure only one hook overrides a "
                            "given query, or have them produce identical overrides."
                        )
                    if override_effect is None:
                        override_effect = override
                        ctx.override_response = LLMResponse(
                            content=override.content,
                            prompt_tokens=override.prompt_tokens,
                            completion_tokens=override.completion_tokens,
                            total_tokens=override.total_tokens,
                            cost=override.cost,
                        )

        return ctx

    def _compose_invoke(self, effects: list[Effect]) -> InvokeContext:
        """Compose effects for invoke enter events.

        Composition rules:
        - Abort: exceptions collected into ``abort_errors`` to (possibly) terminate the whole
          invoke before its first iteration (resolved to a ``Raise`` in ``span_invoke``).
          The result-scoped effects (OverrideResult / ModifyResult) are not valid here — an
          invoke has no execution result to supply or transform; ``Finish`` (a graceful
          Return-terminate) is deliberately excluded (#481, YAGNI: no built-in forces a
          Return; governance hooks Abort or push the agent to return via AddMessages).
        - AddInputs: unioned into ``added_inputs`` via the shared conflict-error rule (identical
          coalesces, divergent raises) — order-independent, no silent overwrite.
        - DropInputs: unioned into ``dropped_inputs`` (order-independent). Both are applied to the
          invoke's inputs BEFORE the prompt renders, so an added input shows in the prompt and a
          dropped input vanishes from it. The agent applies drops FIRST, so a key that is both
          added and dropped resolves to the *add* — and a drop-then-add of a caller input is a
          replacement rather than a collision.
        - DisableRecursion: idempotent OR into ``recursion_disabled`` — any emitter suppresses
          this invoke's ``jaz.invoke`` tool (primitive binds ``jaz_library=None``).
        """
        ctx = InvokeContext()

        # Apply invoke-specific effects
        for effect in effects:
            match effect:
                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case AddInputs(inputs=inputs):
                    # Refuse __builtins__ here too (mirrors AddVariables): an added *input* is bound
                    # into the same namespace, so binding the sandbox key would reopen it.
                    _coalesce_add(
                        ctx.added_inputs,
                        inputs,
                        effect="AddInputs",
                        refuse_sandbox=True,
                    )

                case DropInputs(keys=keys, allow_missing=allow_missing):
                    key_set = set(keys)
                    ctx.dropped_inputs |= key_set
                    if allow_missing:
                        ctx.dropped_inputs_allow_missing |= key_set

                case DisableRecursion():
                    ctx.recursion_disabled = True

        return ctx

    def _compose_llm_query_retry(self, effects: list[Effect]) -> LLMQueryRetryContext:
        """Compose effects for LLM retry events.

        This is a read-only context - LLM retries are informational.
        No effects are allowed.
        """
        ctx = LLMQueryRetryContext()

        # Warn about invalid effects
        for effect in effects:
            logger.warning(
                f"{type(effect).__name__} not valid for LLMQueryRetry, ignoring"
            )

        return ctx

    # Span-based context managers for enter/exit event pairs

    @contextmanager
    def span_repl_exec(self, enter_event: REPLExecEnter):
        """Context manager for REPL execution span.

        Usage:
            enter_event = REPLExecEnter(...)
            with dispatcher.span_repl_exec(enter_event) as span:
                if span.enter_override is not None:
                    exec_result = span.enter_override  # hook supplied a result
                else:
                    exec_result = repl.exec(...)
                span.complete(exec_result=exec_result)
            # exit-time ModifyResult / Abort is applied here:
            exec_result = span.get_final_exec_result()

        The span must be completed before exiting, or an error is raised.

        Layering of enter-time supply and exit-time transform (both intentional):
          1. When `enter_override` (an OverrideResult supply, or an Abort) short-circuits
             execution, `REPLExecExit` still fires with that override as `exec_result`.
             Observability hooks therefore record an outcome for code that never ran — they
             care about what happened from the agent's perspective, not whether `repl.exec`
             was called.
          2. Exit-time `ModifyResult`s compose on top of whatever result the enter branch
             produced, and an exit `Abort` escalates any prior result to a terminal `Raise`.
             This is the composition rule that lets `BudgetForcing` transform a `Return`
             into a `Continue` at exit, and it deliberately does not check the origin of the
             prior result. The origin-agnostic rule cuts both ways: because the result *kind*
             is decided by the effects present at *this* (exit) boundary, an exit
             `ModifyResult(Continue(...))` composed onto an enter terminal (`Return` /
             `Raise`) *downgrades* it to a recoverable `Continue` — the exit boundary's own
             effects are the last word (see `resolve_modify_results`).
        """
        # Fire enter event and compose context
        effects = self.emit(enter_event)
        ctx = self._compose_repl_exec(effects)

        # Create span
        span = REPLExecSpan(ctx=ctx)

        # Enter-time OverrideResult (supply) / Abort short-circuit execution: no result has
        # been produced yet, so the supply names THE result outright (no original to fold).
        span.enter_override = resolve_override_results(
            override_effects=ctx.override_effects,
            abort_errors=ctx.abort_errors,
        )

        exceptional_exit = False
        try:
            yield span
        except BaseException:
            # Body unwound on an in-flight exception — see _log_incomplete_span for why
            # this stays quiet rather than logging the forgot-to-complete error.
            exceptional_exit = True
            raise
        finally:
            # Ensure span was completed
            if not span.is_completed():
                _log_incomplete_span(
                    exceptional_exit,
                    debug_msg=(
                        f"REPLExec span for execution {enter_event.iteration} exited "
                        "on an in-flight exception before completion — no REPLExecExit fires."
                    ),
                    error_msg=(
                        f"REPLExec span for execution {enter_event.iteration} was not "
                        "completed. You must call span.complete(exec_result=...) before exiting."
                    ),
                )
                # Don't return - let exceptions propagate
            else:
                original = span.get_exec_result()
                # Fire exit event only if span was completed
                exit_event = REPLExecExit(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    iteration=enter_event.iteration,
                    exec_result=original,
                    depth=enter_event.depth,
                )
                exit_effects = self.emit(exit_event)
                exit_ctx = self._compose_repl_exec_exit(exit_effects)
                # Exit-time ModifyResult transforms the actual result (original kept if none).
                override = resolve_modify_results(
                    modify_effects=exit_ctx.modify_effects,
                    abort_errors=exit_ctx.abort_errors,
                    original=original,
                )
                span.set_final_exec_result(
                    override if override is not None else original
                )

    @contextmanager
    def span_llm_query(self, enter_event: LLMQueryEnter):
        """Context manager for LLM query span.

        Usage:
            enter_event = LLMQueryEnter(...)
            with dispatcher.span_llm_query(enter_event) as span:
                if span.abort is not None:
                    return span.abort          # Abort @ LLMQueryEnter: terminate the invoke
                response = llm.query(messages, **base_config)
                span.complete(
                    response_content=response,
                    prompt_tokens=...,
                    completion_tokens=...,
                    cost=...
                )

        :class:`LLMQueryEnter` is the always-present per-turn boundary, so :class:`Abort` is honored
        here (the loop/budget hard-stops). When a hook aborts, ``span.abort`` is a terminal
        :class:`Raise` and the caller must return it *without* querying or completing the span: an
        aborted query never happened, so **no :class:`LLMQueryExit` fires** (mirroring the loop
        pseudocode, where ``if q.abort: return`` short-circuits before the exit event).
        """
        # Fire enter event and compose context
        effects = self.emit(enter_event)
        ctx = self._compose_llm_query(effects)

        # Create span
        span = LLMQuerySpan(ctx=ctx)

        # Abort @ LLMQueryEnter resolves to a terminal Raise (no Continue/Return here, so
        # original=None). Set on the span; the caller returns it to terminate the invoke.
        span.abort = resolve_override_results(
            override_effects=[],
            abort_errors=ctx.abort_errors,
        )

        exceptional_exit = False
        try:
            yield span
        except BaseException:
            # Body unwound on an in-flight exception (LLM/provider failure, KeyboardInterrupt)
            # — see _log_incomplete_span for why this stays quiet rather than logging the
            # forgot-to-complete error.
            exceptional_exit = True
            raise
        finally:
            if span.abort is not None:
                # Aborted at enter: the LLM call never happened and the span is not
                # completed by design — fire no LLMQueryExit and log no "not completed"
                # error (there is no response to report).
                pass
            elif not span.is_completed():
                _log_incomplete_span(
                    exceptional_exit,
                    debug_msg=(
                        "LLMQuery span exited on an in-flight exception before "
                        "completion — no LLMQueryExit fires."
                    ),
                    error_msg=(
                        "LLMQuery span was not completed. "
                        "You must call span.complete(...) before exiting."
                    ),
                )
                # Don't return - let exceptions propagate
            else:
                # Fire exit event only if span was completed
                exit_event = LLMQueryExit(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    response=span.get_response(),
                    model=enter_event.model,
                    iteration=span.get_iteration(),
                    depth=enter_event.depth,
                    start_time=span.get_start_time(),
                    end_time=span.get_end_time(),
                    message_edits=ctx.message_edits,
                )
                self.emit(exit_event)

    @contextmanager
    def span_invoke(self, enter_event: InvokeEnter):
        """Context manager for invoke span.

        Usage:
            enter_event = InvokeEnter(...)
            with dispatcher.span_invoke(enter_event) as span:
                if span.enter_override is not None:
                    result = span.enter_override  # a hook aborted the invoke
                else:
                    # Use span.ctx.added_inputs / dropped_inputs, etc.
                    result = agent._invoke_internal(...)
                span.complete(result=result)
        """
        # Fire enter event and compose context
        effects = self.emit(enter_event)
        ctx = self._compose_invoke(effects)

        # Create span
        span = InvokeSpan(ctx=ctx)

        # Enter-time Abort short-circuits the whole invoke (no execution has happened yet).
        span.enter_override = resolve_override_results(
            override_effects=[],
            abort_errors=ctx.abort_errors,
        )

        exceptional_exit = False
        try:
            yield span
        except BaseException:
            # Body unwound on an in-flight exception — see _log_incomplete_span for why
            # this stays quiet rather than logging the forgot-to-complete error.
            exceptional_exit = True
            raise
        finally:
            # Ensure span was completed
            if not span.is_completed():
                _log_incomplete_span(
                    exceptional_exit,
                    debug_msg=(
                        "Invoke span exited on an in-flight exception before "
                        "completion — no InvokeExit fires."
                    ),
                    error_msg=(
                        "Invoke span was not completed. "
                        "You must call span.complete(result=...) before exiting."
                    ),
                )
                # Don't return - let exceptions propagate
            else:
                # Fire the exit event and honor exit-time ModifyResult / Abort — the invoke's
                # terminal result is a transform boundary now (#568), symmetric with
                # span_repl_exec. A ReturnType / ValidateReturn hook downgrades a wrong-typed /
                # invalid final Return to a Raise here (the backstop for another hook's
                # REPLExecExit override reinstating a Return). The (possibly overridden) result is
                # stored back on the span; the invoke loop reads span.get_result() AFTER the span
                # closes and returns/raises from that.
                #
                # Known limitation (#906): the InvokeExit *event* carries the PRE-transform
                # ``original``, but the invoke actually returns/raises the POST-transform
                # ``span.get_result()``. So a *passive* observer keyed on ``event.result`` (ATIF
                # trace, conversation history, OTel span) records the pre-backstop result and misses a
                # ReturnType/ValidateReturn InvokeExit downgrade. This is inherent to the event being
                # the transform's *input*; a hook that wants to affect the final value emits
                # ModifyResult here (folded by resolve_modify_results below), not read the event.
                original = span.get_result()
                exit_event = InvokeExit(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    result=original,
                    depth=enter_event.depth,
                )
                exit_ctx = self._compose_invoke_exit(self.emit(exit_event))
                override = resolve_modify_results(
                    modify_effects=exit_ctx.modify_effects,
                    abort_errors=exit_ctx.abort_errors,
                    original=original,
                )
                if override is not None:
                    span.set_final_result(override)

    # Simple events (no enter/exit pair, just fire and compose)

    def process_llm_query_retry(self, event: LLMQueryRetry) -> LLMQueryRetryContext:
        """Process an LLM retry event.

        Args:
            event: The LLM retry event to process

        Returns:
            LLMQueryRetryContext (read-only, can only record metrics)
        """
        effects = self.emit(event)
        return self._compose_llm_query_retry(effects)


# Default singleton dispatcher — used by every Agent that does NOT pass
# local_hooks. Stateless w.r.t. invokes (it has no local hooks bound),
# so it is safe to share across all such Agents. Agents that DO pass
# local_hooks construct their own per-invoke HookDispatcher; see the
# HookDispatcher class docstring for the design rationale.
_dispatcher = HookDispatcher()


def get_dispatcher() -> HookDispatcher:
    """Return the global default dispatcher (no local hooks bound).

    This singleton has an empty `_local_hooks` tuple and behaves identically
    to the pre-`local_hooks` design: it only dispatches to propagating hooks
    read from the contextvar. Agents that want local hooks should construct
    their own `HookDispatcher(local_hooks=...)` instead.
    """
    return _dispatcher
