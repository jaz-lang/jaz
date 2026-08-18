"""HookDispatcher orchestrates hooks and composes their effects into execution contexts."""

import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

from jaz.exceptions import (
    BlackboardSeedError,
    BlackboardWriteConflictError,
    FatalError,
    HookActivationError,
    InvalidEffectError,
    LLMResponseConflictError,
    REPLInputConflictError,
    SandboxKeyError,
    _JazInternalError,
)
from jaz.repl.types import Raise, Return

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
    ModifyExecResult,
    SupplyExecResult,
    SupplyLLMResponse,
    _combine_exceptions,
    _resolve_edit_index,
    resolve_modify_results,
    resolve_supply_results,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jaz.config import Config
    from jaz.llm import MessageDict
from .events import (
    InvokeComplete,
    InvokeCompleteContext,
    InvokeContext,
    InvokeEnter,
    InvokeExit,
    InvokeSend,
    InvokeSendContext,
    InvokeSpan,
    LLMQueryComplete,
    LLMQueryCompleteContext,
    LLMQueryContext,
    LLMQueryEnter,
    LLMQueryExit,
    LLMQueryRetry,
    LLMQueryRetryContext,
    LLMQuerySend,
    LLMQuerySendContext,
    LLMQuerySpan,
    MessageAddRecord,
    MessageDropRecord,
    REPLExecComplete,
    REPLExecCompleteContext,
    REPLExecContext,
    REPLExecEnter,
    REPLExecExit,
    REPLExecSend,
    REPLExecSendContext,
    REPLExecSpan,
    ResolvedMessageEdits,
)
from .events.base import Aborted, Completed, Failed
from .events.llm_query import MessageEdits

logger = logging.getLogger(__name__)


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


def _log_incomplete_span(error_msg: str) -> None:
    """Log the loud ERROR for a span whose body exited *cleanly* without ``complete()``.

    Shared by all three ``span_*`` context managers so a fourth span can't grow its own
    hand-rolled copy (each still passes its own message). This is the genuine
    forgot-to-``complete()`` programming error — an early ``return``/``break`` out of the
    ``with`` body that skipped the unconditional trailing ``complete()`` — and it stays a
    loud tripwire; synthesizing a close would mask the bug. An *exceptional* unwind never
    reaches here: the span CMs' abnormal arm closes the span with a ``Failed``/``Aborted``
    outcome instead (the former debug arm of this helper, deleted as unreachable).
    """
    logger.error(error_msg)


class Hook:
    """Base class for hooks — subclass it and override the handlers you care about.

    Every event has a handler named after it (:class:`LLMQueryEnter` → ``on_llm_query_enter``).
    A handler observes the event and returns a list of effects; returning effects is the only
    way a hook influences the run, and which effects are accepted depends on the event — each
    event class documents its own set, and an effect outside it raises
    :class:`~jaz.exceptions.InvalidEffectError` rather than being silently dropped. A
    cross-cutting observer (logging, tracing) overrides
    ``on_any`` instead, which runs *in addition to* the matched typed handler, so the two
    compose.

    There are two ways to activate a hook, and **they differ in whether nested invokes see
    it.** A hook activated either way must not be activated the other way at the same time.
    (Snippets below assume ``from jaz import invoke``.)

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
        from jaz import invoke
        from jaz.hooks import Hook
        from jaz.hooks.effects import Abort, Effect
        from jaz.hooks.events import LLMQueryEnter

        class StopAfterFiftyTurns(Hook):
            def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
                if event.iteration >= 50:  # iterations are 0-based
                    return [Abort(error=RuntimeError("Limit exceeded"))]
                return []

        with StopAfterFiftyTurns():
            result = invoke(task="task")
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

    def _dispatch_event(self, event: Event) -> list[Effect]:
        """The framework event router: dispatch an event to its typed ``on_<event>``
        handler (the override points below).

        **Do not override this.** It is the dispatcher's internal router; overriding it
        replaces the routing, so any ``on_<event>`` handlers would silently never run
        (#597). To extend a hook, override the typed handlers below — you get typed
        attribute access and implement only the events you care about, with no
        ``isinstance``/``match`` boilerplate and no "remember to ``return []``" footgun.

        For a genuinely *cross-cutting* observer that treats every event uniformly (loggers,
        tracing), override :meth:`~Hook.on_any` instead: the dispatcher calls it *in addition to*
        the matched typed handler, so it composes with — rather than replaces — the
        per-event dispatch, and a hook may use both.

        Returns the effects expressing how to influence execution (empty for none).
        """
        match event:
            case InvokeEnter():
                return self.on_invoke_enter(event)
            case InvokeSend():
                return self.on_invoke_send(event)
            case InvokeComplete():
                return self.on_invoke_complete(event)
            case InvokeExit():
                return self.on_invoke_exit(event)
            case REPLExecEnter():
                return self.on_repl_exec_enter(event)
            case REPLExecSend():
                return self.on_repl_exec_send(event)
            case REPLExecComplete():
                return self.on_repl_exec_complete(event)
            case REPLExecExit():
                return self.on_repl_exec_exit(event)
            case LLMQueryEnter():
                return self.on_llm_query_enter(event)
            case LLMQuerySend():
                return self.on_llm_query_send(event)
            case LLMQueryComplete():
                return self.on_llm_query_complete(event)
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

    def on_invoke_send(self, event: InvokeSend) -> list[Effect]:
        """An invoke's input set has committed (post add/drop). Default no-op."""
        return []

    def on_invoke_complete(self, event: InvokeComplete) -> list[Effect]:
        """An invoke's raw terminal result exists (transform boundary). Default no-op."""
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        """An invoke's span has closed (see ``event.outcome``). Default no-op."""
        return []

    def on_repl_exec_enter(self, event: REPLExecEnter) -> list[Effect]:
        """REPL code is about to execute (parsed code available). Default no-op."""
        return []

    def on_repl_exec_send(self, event: REPLExecSend) -> list[Effect]:
        """An execution's input has committed (namespace deltas applied). Default no-op."""
        return []

    def on_repl_exec_complete(self, event: REPLExecComplete) -> list[Effect]:
        """An execution's raw result exists (transform boundary). Default no-op."""
        return []

    def on_repl_exec_exit(self, event: REPLExecExit) -> list[Effect]:
        """A REPL execution's span has closed (see ``event.outcome``). Default no-op."""
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        """An LLM query is about to be issued. Default no-op."""
        return []

    def on_llm_query_send(self, event: LLMQuerySend) -> list[Effect]:
        """A query's input has committed (the post-edit shown list). Default no-op."""
        return []

    def on_llm_query_complete(self, event: LLMQueryComplete) -> list[Effect]:
        """A query's raw response exists (response control boundary). Default no-op."""
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list[Effect]:
        """An LLM query's span has closed (see ``event.outcome``). Default no-op."""
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

    # Static write-authority declarations (see ``HookDispatcher._reject_declared_write_conflicts``).
    # Each of the effects below is *slot-scoped* with a single authority per slot: the turn for
    # ``SupplyLLMResponse``, the key/name for the others. Two active hooks declaring the same slot
    # is an irresolvable, order-dependent authority conflict, so it is rejected at seed time (before
    # the first ``InvokeEnter``) rather than mid-run. These are OPT-IN and default-empty: a hook that
    # writes a slot WITHOUT declaring it is not seen by the seed-time check but is still caught
    # dynamically by the same conflict error at composition time.
    #
    # The seed-time predicate is deliberately STRICTER than the runtime one, not merely earlier. At
    # runtime, two writes to one slot with *identical values* COALESCE and only *divergent* values
    # conflict (``_coalesce_add`` / ``_values_conflict`` for AddInputs/AddVariables, the same-key
    # coalesce in ``blackboard.py`` for BlackboardWrite, the ``supplied != response_effect`` guard
    # for SupplyLLMResponse). This check instead rejects any two *declarers* of a slot regardless of
    # value — a coalescing duplicate is still two authorities for one slot, which under "one
    # authority per slot" is a config smell worth surfacing (the same executive call made for two
    # identical ``Replay``s: a pointless duplicate should fail loudly, not silently coalesce). So the
    # runtime check remains the looser backstop for an UNDECLARED write, while a DECLARED
    # identical-value pair is intentionally rejected here even though runtime would have coalesced
    # it. Consequence for a defensive author: adding a declaration to a composition that previously
    # coalesced turns it into a seed error. (Same-slot always rejects — there is no
    # reducer/accumulator carve-out in this design.)

    #: Whether this hook may emit :class:`~jaz.hooks.effects.SupplyLLMResponse` (claim authorship
    #: of the model's turn). ``SupplyLLMResponse`` is a *sole-authority* effect — one response per
    #: turn — so two active hooks both declaring this are rejected at activation with
    #: :class:`~jaz.exceptions.LLMResponseConflictError`.
    supplies_llm_response: ClassVar[bool] = False

    #: Input keys this hook may bind via :class:`~jaz.hooks.effects.AddInputs`. Two active hooks
    #: declaring the same key are rejected at activation with
    #: :class:`~jaz.exceptions.REPLInputConflictError`.
    adds_inputs: ClassVar[frozenset[str]] = frozenset()

    #: REPL namespace names this hook may bind via :class:`~jaz.hooks.effects.AddVariables`. Two
    #: active hooks declaring the same name are rejected at activation with
    #: :class:`~jaz.exceptions.REPLInputConflictError`.
    adds_variables: ClassVar[frozenset[str]] = frozenset()

    #: Blackboard keys this hook may write via :class:`~jaz.hooks.effects.BlackboardWrite`. Two
    #: active hooks declaring the same key are rejected at activation with
    #: :class:`~jaz.exceptions.BlackboardWriteConflictError`.
    writes_blackboard: ClassVar[frozenset[str]] = frozenset()

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

    def __repr__(self) -> str:
        """``ClassName()`` — override to show the configuration this hook applies."""
        # Overriding the object default is not cosmetic. Observability consumers render the
        # active hook set with `repr()`, and `object.__repr__` embeds the instance's memory
        # address, so the default would stamp `<...BudgetPool object at 0x10331f380>` into every
        # trace, log line and span attribute — noise that also differs run to run, defeating any
        # diff of two traces. The bare class name is stable and says the one thing every hook can
        # say honestly; a hook whose configuration is worth recording says more by overriding.
        #
        # Known loss against the serialized record this replaced: it carried a `qualified_name`,
        # so two hooks with the same class name from different modules — a user's own
        # `Compaction`, or one subclassed inside a test — were distinguishable in a trace and are
        # not now. Deliberate: module-qualifying every hook would make every built-in's string
        # longer for a collision that is rare, and a hook that expects to collide can qualify
        # itself in its own override. Reach for that before concluding a trace is lying to you.
        #
        # A dataclass hook needs nothing: @dataclass generates its own __repr__ on the subclass,
        # which wins over this by MRO and already lists the fields.
        return f"{type(self).__name__}()"

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
    5. Resolving supply/transform/abort effects (SupplyExecResult / ModifyExecResult / Abort) at REPL execution
    6. Managing span lifecycle for enter/exit event pairs

    Design principles:
    - Hooks are called in registration order (first registered, first called)
    - Composition rules are explicit and documented
    - An effect returned at an event that does not accept it raises
      :class:`~jaz.exceptions.InvalidEffectError` (out-of-stage effects are hook bugs
      and fail loudly, never silently no-op)
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
        # The exception objects THIS dispatcher's span CMs raised while resolving an
        # ``Abort`` effect, recorded by identity. This is the abort-classification record
        # (see ``_unwind_outcome``): the dispatcher is per-invoke state (Agent constructs
        # one per invoke), so "is this unwinding exception my own invoke's abort?" is a
        # plain identity test against this list — no global registry, no exception
        # wrapping/marking that would perturb the propagating object. On the shared
        # ``get_dispatcher()`` singleton the record spans whatever spans are opened on it;
        # that singleton is test/README-only surface (see the class docstring's
        # "Limitation worth knowing"). Bounded in practice: an invoke resolves at most a
        # handful of aborts before it ends, and the list dies with the dispatcher.
        self._own_abort_exceptions: list[BaseException] = []

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

    def _reject_declared_write_conflicts(self, active: list[Hook]) -> None:
        """Reject, before the first :class:`InvokeEnter`, two active hooks that statically declare
        writing the same sole-authority slot.

        Each of ``SupplyLLMResponse`` / ``AddInputs`` / ``AddVariables`` / ``BlackboardWrite`` has a
        single authority per slot — the turn for the response, the key/name for the others — so two
        hooks declaring the same slot is an irresolvable, order-dependent conflict, surfaced here at
        turn 0 instead of mid-run. Keyed on the hooks' static declarations
        (``supplies_llm_response`` / ``adds_inputs`` / ``adds_variables`` / ``writes_blackboard``);
        a hook that writes a slot WITHOUT declaring it is invisible here and is still caught
        dynamically by the *same* conflict error at composition time. This is deliberately
        *stricter* than the runtime check, not merely earlier: runtime coalesces two writes carrying
        identical values and conflicts only on divergent ones, whereas this rejects any two declarers
        of a slot regardless of value (see the class-level note on ``Hook.supplies_llm_response`` for
        why). Raised errors reuse the runtime conflict types, so ``except REPLInputConflictError``
        (etc.) catches both the early and the late form.
        """
        # SupplyLLMResponse — one sole authority for the whole turn (no key).
        responders = [type(h).__name__ for h in active if h.supplies_llm_response]
        if len(responders) > 1:
            raise LLMResponseConflictError(
                f"{len(responders)} active hooks ({', '.join(responders)}) each declare they may "
                "supply the LLM response (supplies_llm_response=True). SupplyLLMResponse is a "
                "sole-authority effect — at most one hook may author the model's turn — so this is "
                "rejected at activation. Activate only one."
            )
        # Key-scoped effects — one authority per key/name.
        self._reject_key_authority_collision(
            active, "adds_inputs", REPLInputConflictError, "AddInputs input key"
        )
        self._reject_key_authority_collision(
            active,
            "adds_variables",
            REPLInputConflictError,
            "AddVariables namespace name",
        )
        self._reject_key_authority_collision(
            active,
            "writes_blackboard",
            BlackboardWriteConflictError,
            "BlackboardWrite key",
        )

    @staticmethod
    def _reject_key_authority_collision(
        active: list[Hook],
        attr: str,
        error_cls: type[Exception],
        label: str,
    ) -> None:
        """Raise ``error_cls`` if two active hooks declare (via the class attr ``attr``) writing the
        same slot. Two instances of one class declaring the same key still collide — they are two
        authorities for one slot — which is why the message lists class names, not identities."""
        declarers_by_slot: dict[str, list[str]] = {}
        for hook in active:
            for slot in getattr(hook, attr):
                declarers_by_slot.setdefault(slot, []).append(type(hook).__name__)
        for slot, declarers in declarers_by_slot.items():
            if len(declarers) > 1:
                raise error_cls(
                    f"{len(declarers)} active hooks ({', '.join(declarers)}) each declare (via "
                    f"{attr}) that they write the same {label} {slot!r}. This is a sole-authority "
                    "slot — at most one hook may write it — so it is rejected at activation. "
                    "Have one hook own the slot, or activate only one."
                )

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
        # Reject statically-declared write-authority collisions before the first InvokeEnter.
        self._reject_declared_write_conflicts(active)
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

        Timestamp stamping:
            ``event.timestamp`` is set here to the emission time, immediately before
            the hook loop — a pre-set value (a hand-built event's construction-time
            default included) is overwritten, so a direct ``emit()`` caller must not
            expect its own stamp to survive.
        """
        # Stamp the invoke's live board onto the event (overriding its empty default)
        # so reads during this event see the current generation. Event is frozen (hooks must
        # treat it as immutable); object.__setattr__ is the sanctioned in-framework escape hatch
        # for the dispatcher's own stamps — the one place allowed to write an event's fields.
        if self._blackboard is not None:
            object.__setattr__(event, "blackboard", self._blackboard)

        # Stamp the emission time — the one timing field every event carries (see
        # Event.timestamp: emission, not construction, so the stamp is taken here, the
        # single place every event passes through immediately before the hook loop).
        object.__setattr__(event, "timestamp", datetime.now(UTC))

        effects: list[Effect] = []
        blackboard_writes: list[tuple[str, object]] = []

        active = self._active_hooks()
        # Stamp the LIVE active hook set (dispatch order) onto the event so hooks in the loop below —
        # and any observability consumer — can see the governance active at this event (Event.hooks).
        # The actual Hook instances (#727), not rendered strings: Event.hooks is the effect system's
        # view of the active set; consumers that log it call ``repr()`` at their edge. Keeping it
        # a plain reference tuple means no per-event rendering on this hot path. object.__setattr__
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
                # EXECUTIVE CALL (user, 2026-07-30): **raising is not an abort channel.** A
                # `FatalError` from a hook handler is caught here like any other exception and
                # logged, NOT propagated. The point of the hook system is to contain side
                # effects in a small, finite, composable effect set rather than admitting
                # arbitrary code or side effects; letting a hook author reach for `raise` to
                # terminate a run defeats that, because it is a control-flow side effect the
                # effect algebra cannot see, order-independently compose, or validate.
                # `Abort(error=...)` is the supported way, and it is strictly better behaved:
                # it composes with other effects, and it resolves to a terminal `Raise`
                # *through* the span, so `span.complete()` runs and `InvokeExit` still fires
                # (a raise propagates around both — see the FatalError docstring).
                #
                # Trade-off, deliberately accepted (Option B on #562): a governance hook
                # whose _dispatch_event has a *bug* — raising a plain Exception instead of
                # terminating via an effect — is swallowed here, silently disabling its
                # enforcement, rather than failing loud.
                # TODO(#612): make log-vs-propagate for uncaught hook exceptions configurable.
                if isinstance(e, FatalError):
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

    @staticmethod
    def _reject_invalid_effect(effect: Effect, event_name: str, valid: str) -> NoReturn:
        """Raise :class:`InvalidEffectError` for an effect returned at a stage that does not
        accept it.

        One shared raise site so every ``_compose_*`` rejects out-of-stage effect kinds with
        the same loud contract (this replaced a warn-and-ignore patchwork, plus one composer
        that silently dropped — the final stage of span_event_lifecycle.md: a stale hook
        must fail its first run, not silently no-op). ``BlackboardWrite`` never reaches
        composition — ``emit()`` partitions it out and applies it at the generational
        barrier — so it stays valid everywhere without a carve-out here.
        """
        raise InvalidEffectError(
            f"{type(effect).__name__} is not a valid effect at {event_name} "
            f"(effects accepted there: {valid}). Supplies compose at the *Send events, "
            "transforms at the *Complete events, and Abort at every control event; the "
            "*Exit events are observation-only and accept no effects."
        )

    # Context-specific composition methods

    def _compose_repl_exec(self, effects: list[Effect]) -> REPLExecContext:
        """Compose effects for REPL execution *enter* events.

        Composition rules:
        - Abort (abort): exceptions collected into ``abort_errors``; ``span_repl_exec``
          raises the combined exception (the span closes ``Aborted``).
        - AddVariables (bind namespace names): unioned into ``added_variables`` via the shared
          conflict-error rule (identical coalesces, divergent raises); ``__builtins__`` refused.
        - DropVariables (drop namespace names): unioned into ``dropped_variables``
          (order-independent, like DropMessages). ``__builtins__`` — the compiler sandbox
          (#688/#690) — is refused, so a stray drop can't reopen it.
        - Any other effect raises ``InvalidEffectError`` — notably ``SupplyExecResult``,
          which composes at ``REPLExecSend`` (a supplier must see the committed input).

        ``AddInputs``/``DropInputs`` are **not** collected here — they are ``InvokeEnter``-only
        (#481): inputs are per-invoke setup, applied once before the prompt renders. The per-turn
        namespace counterpart is ``AddVariables``/``DropVariables`` here.
        """
        ctx = REPLExecContext()

        # Apply REPL-specific effects
        for effect in effects:
            match effect:
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

                case _:
                    self._reject_invalid_effect(
                        effect, "REPLExecEnter", "Abort, AddVariables, DropVariables"
                    )

        return ctx

    def _compose_repl_exec_send(self, effects: list[Effect]) -> REPLExecSendContext:
        """Compose effects for REPL execution *send* events — the supply boundary.

        Composition rules:
        - SupplyExecResult (supply a result for the committed input): collected to
          (possibly) short-circuit execution (see resolve_supply_results).
        - Abort: collected; ``span_repl_exec`` raises the combined exception (the span
          closes ``Aborted``), superseding any supply.
        - Any other effect raises ``InvalidEffectError``.
        """
        ctx = REPLExecSendContext()

        for effect in effects:
            match effect:
                case SupplyExecResult() as supplied:
                    ctx.supply_effects.append(supplied)

                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    self._reject_invalid_effect(
                        effect, "REPLExecSend", "SupplyExecResult, Abort"
                    )

        return ctx

    def _compose_repl_exec_complete(
        self, effects: list[Effect]
    ) -> REPLExecCompleteContext:
        """Compose effects for REPL execution *complete* events — the transform boundary.

        Composition rules:
        - ModifyExecResult (transform the result): collected to (possibly) override the
          execution result (see resolve_modify_results).
        - Abort: collected; ``span_repl_exec`` raises the combined exception (the span
          closes ``Aborted``), superseding any transform.
        - Any other effect raises ``InvalidEffectError``.
        """
        ctx = REPLExecCompleteContext()

        for effect in effects:
            match effect:
                case ModifyExecResult() as modify:
                    ctx.modify_effects.append(modify)

                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    self._reject_invalid_effect(
                        effect, "REPLExecComplete", "ModifyExecResult, Abort"
                    )

        return ctx

    def _compose_invoke_send(self, effects: list[Effect]) -> InvokeSendContext:
        """Compose effects for invoke *send* events.

        Composition rules:
        - Abort (abort): exceptions collected into ``abort_errors``; ``span_invoke``'s send
          emitter raises the combined exception (the span closes ``Aborted``). No invoke
          supplier effect exists, so abort is the only control kind here.
        - Any other effect raises ``InvalidEffectError``.
        """
        ctx = InvokeSendContext()

        for effect in effects:
            match effect:
                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    self._reject_invalid_effect(effect, "InvokeSend", "Abort")

        return ctx

    def _compose_invoke_complete(self, effects: list[Effect]) -> InvokeCompleteContext:
        """Compose effects for invoke *complete* events (#568).

        Symmetric with ``_compose_repl_exec_complete``: collect ``ModifyExecResult`` (transform
        the terminal result) / ``Abort`` (abort — ``span_invoke`` raises the combined
        exception and the span closes ``Aborted``); any other effect raises
        ``InvalidEffectError``. Resolved against the terminal result in
        ``span_invoke`` via ``resolve_modify_results`` — this is what lets ``ReturnType`` /
        ``ValidateReturn`` downgrade a wrong-typed / invalid final ``Return`` to a ``Raise`` (the
        backstop for another hook's ``REPLExecComplete`` override reinstating a ``Return``).
        """
        ctx = InvokeCompleteContext()

        for effect in effects:
            match effect:
                case ModifyExecResult() as modify:
                    ctx.modify_effects.append(modify)

                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    self._reject_invalid_effect(
                        effect, "InvokeComplete", "ModifyExecResult, Abort"
                    )

        return ctx

    def _compose_llm_query(self, effects: list[Effect]) -> LLMQueryContext:
        """Compose effects for LLM query *enter* events.

        Composition rules:
        - Abort: exceptions collected into ``abort_errors`` — this is the
          always-present per-turn point where loop/budget hard-stops terminate the invoke;
          ``span_llm_query`` raises the combined exception (the span closes ``Aborted``,
          before the LLM call).
        - DropMessages: indices are unioned (order-independent)
        - AddMessages: accumulated (within-slot order resolved by the fold)
        - Any other effect raises ``InvalidEffectError`` — notably ``SupplyLLMResponse``,
          which composes at ``LLMQuerySend`` (a supplier must see the committed post-edit
          list).
        """
        ctx = LLMQueryContext()

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
                case _:
                    self._reject_invalid_effect(
                        effect, "LLMQueryEnter", "Abort, DropMessages, AddMessages"
                    )

        return ctx

    def _compose_llm_query_send(self, effects: list[Effect]) -> LLMQuerySendContext:
        """Compose effects for LLM query *send* events — the supply boundary.

        Composition rules:
        - SupplyLLMResponse: order-independent — identical overrides compose to
          one; two distinct overrides raise LLMResponseConflictError (rather than
          resolving the conflict by hook/with-nesting order)
        - Abort: exceptions collected into ``abort_errors``; ``span_llm_query``'s send
          emitter raises the combined exception (the committed-input veto — the span closes
          ``Aborted``).
        - Any other effect raises ``InvalidEffectError``.
        """
        from jaz._llm_client import LLMResponse

        ctx = LLMQuerySendContext()

        # Track the winning supply effect so a later one can be compared for
        # equality — the resolution must not depend on arrival order.
        response_effect: SupplyLLMResponse | None = None

        for effect in effects:
            match effect:
                case Abort(error=error):
                    ctx.abort_errors.append(error)
                case SupplyLLMResponse() as supplied:
                    if response_effect is not None and supplied != response_effect:
                        raise LLMResponseConflictError(
                            "Multiple conflicting SupplyLLMResponse effects for "
                            "the same LLM query. Supply composition is "
                            "order-independent: distinct supplied responses cannot be "
                            "resolved without relying on hook registration / "
                            "with-nesting order. Ensure only one hook supplies a "
                            "given query, or have them produce identical responses."
                        )
                    if response_effect is None:
                        response_effect = supplied
                        ctx.supplied_response = LLMResponse(
                            content=supplied.content,
                            prompt_tokens=supplied.prompt_tokens,
                            completion_tokens=supplied.completion_tokens,
                            cost_usd=supplied.cost_usd,
                        )
                case _:
                    self._reject_invalid_effect(
                        effect, "LLMQuerySend", "SupplyLLMResponse, Abort"
                    )

        return ctx

    def _compose_invoke(self, effects: list[Effect]) -> InvokeContext:
        """Compose effects for invoke enter events.

        Composition rules:
        - Abort: exceptions collected into ``abort_errors`` to (possibly) abort the
          whole invoke before its first iteration (``span_invoke`` raises the combined
          exception; the span closes ``Aborted``).
          The result-scoped effects (SupplyExecResult / ModifyExecResult) are not valid here — an
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
          this invoke's ``invoke`` tool (primitive binds ``invoke_tool=None``).
        - Any other effect raises ``InvalidEffectError`` (this composer used to drop them
          silently — the worst variant of the pre-loud-rejection patchwork).
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

                case _:
                    self._reject_invalid_effect(
                        effect,
                        "InvokeEnter",
                        "Abort, AddInputs, DropInputs, DisableRecursion",
                    )

        return ctx

    def _compose_llm_query_retry(self, effects: list[Effect]) -> LLMQueryRetryContext:
        """Compose effects for LLM retry events.

        This is a read-only context - LLM retries are informational.
        Any effect raises ``InvalidEffectError``.
        """
        ctx = LLMQueryRetryContext()

        for effect in effects:
            self._reject_invalid_effect(effect, "LLMQueryRetry", "none (observational)")

        return ctx

    def _compose_llm_query_complete(
        self, effects: list[Effect]
    ) -> LLMQueryCompleteContext:
        """Compose effects for LLM query *complete* events.

        Composition rules:
        - Abort (abort): exceptions collected into ``abort_errors``; ``span_llm_query``
          raises the combined exception once the complete event has been dispatched (the
          span closes ``Aborted``).
        - Any other effect raises ``InvalidEffectError``. (No response-modify effect
          exists yet — see the comment on ``LLMQueryComplete``.)
        """
        # Why abort is the one control kind that survives here: see `LLMQueryCompleteContext`.
        ctx = LLMQueryCompleteContext()

        for effect in effects:
            match effect:
                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    self._reject_invalid_effect(effect, "LLMQueryComplete", "Abort")

        return ctx

    # Send / abnormal-close plumbing shared by the span context managers

    @staticmethod
    def _resolve_message_edits(
        snapshot: "Sequence[MessageDict]", edits: MessageEdits
    ) -> ResolvedMessageEdits:
        """Resolve a query's composed edits into self-contained records against the
        enter-time ``snapshot``.

        The resolver lives HERE, not on the record dataclasses: the events package must
        not import from the effects layer (cycle hazard — see events/invoke.py's
        ``InvokeCompleteContext.modify_effects`` comment), and the dispatcher is the one place
        that holds both the snapshot and the raw edits at emission time. Resolving once and
        shipping records (rather than bare indices) is what spares every edit-consuming
        observer its own snapshot-and-index-math machinery (span_event_lifecycle.md,
        Defect 5). Reuses ``_resolve_edit_index`` — the same resolver
        ``apply_message_edits`` folds with — so the records agree with the buffer by
        construction; the agent folds the same indices over an identical-content list
        *before* Send fires, so resolution here cannot be the first to see an
        out-of-range index. ("Identical content", not "the same object": the enter
        event's ``messages`` is an immutable enter-time snapshot of the buffer — the
        tuple coercion is what guarantees no hook can desync it from the agent's fold
        between Enter and Send.)
        """
        n = len(snapshot)
        # Distinct raw indices can name one message (-1 and n-1): dedupe by resolved
        # position, persistent winning — matching the buffer, where one persistent drop
        # removes the message for good.
        persistent_by_position: dict[int, bool] = {}
        for raw in edits.all_drops:
            position = _resolve_edit_index(raw, n, upper=n - 1, kind="DropMessages")
            persistent_by_position[position] = persistent_by_position.get(
                position, False
            ) or (raw in edits.persistent_drops)
        drops = [
            MessageDropRecord(message=snapshot[position], persistent=persistent)
            for position, persistent in sorted(persistent_by_position.items())
        ]
        adds = [
            MessageAddRecord(
                # Uncopied — the record's __post_init__ tuple-coerces (owns the severing).
                messages=add.messages,
                position=_resolve_edit_index(
                    add.index if add.index is not None else n,
                    n,
                    upper=n,
                    kind="AddMessages",
                ),
                persistent=add.persistent,
            )
            for add in edits.all_adds
        ]
        return ResolvedMessageEdits(drops=drops, adds=adds)

    @staticmethod
    def _reject_observation_only_effects(effects: list[Effect], event: Event) -> None:
        """Raise ``InvalidEffectError`` for any effect returned at an observation-only
        emission.

        The ``*Exit`` events accept NO effects on any arm: the outcome they carry is
        already final (the transform boundary is ``*Complete``, the supply boundary
        ``*Send``), so an effect here can only be a stale or buggy hook — rejected loudly
        (this upgraded the stage-1 interim ignore-with-debug-log guard of
        span_event_lifecycle.md). On the happy path the raise propagates and fails the
        invoke; on the guarded closes (unwind / ahead of an abort's own raise) it is caught
        and logged by ``_emit_guarded_exit``, because nothing there may replace the
        exception that must win. ``BlackboardWrite`` never reaches this check — ``emit()``
        partitions it out — so observation-time board writes stay legal.
        """
        if effects:
            kinds = ", ".join(type(e).__name__ for e in effects)
            raise InvalidEffectError(
                f"{len(effects)} effect(s) ({kinds}) were returned at "
                f"{type(event).__name__}, which is observation-only: the outcome is "
                "already final, so no effect is valid at any *Exit event. Supplies "
                "compose at the *Send events, transforms at the *Complete events, and "
                "Abort at every control event."
            )

    def _emit_guarded_exit(self, exit_event: Event) -> None:
        """Emit a span's ``*Exit`` on a path where an exception must win the unwind.

        Used for the abnormal close (the span body unwound — ``Failed``/``Aborted``) and
        for the ``Aborted`` close the CM emits just before raising an abort's carried
        exception. Composes nothing: returned effects go through the loud
        observation-only rejection, but here the raise is *caught and logged* — the close
        runs while the span's real exception unwinds (or immediately before the CM raises
        it), and nothing on that path may replace it. (Per-hook exceptions are already
        caught inside ``emit``; this catch covers the dispatcher's own machinery and the
        rejection itself.)
        """
        try:
            self._reject_observation_only_effects(self.emit(exit_event), exit_event)
        except Exception:
            logger.error(
                "Emitting %s on the span's guarded close failed; the span's own "
                "exception continues to propagate.",
                type(exit_event).__name__,
                exc_info=True,
            )

    def _record_own_abort(self, abort_errors: list[Exception]) -> BaseException:
        """Combine a stage's composed ``Abort`` exceptions and record the result as THIS
        invoke's own abort.

        Multiple aborts at one stage keep the grouping law the legacy resolution had
        (``ExceptionGroup``; a single abort raises its exception bare). The returned object
        is recorded by identity on ``_own_abort_exceptions`` so ``_unwind_outcome`` can
        classify the raise as this invoke's own control plane speaking — the same object
        the span CM is about to raise, unwrapped, so the caller/parent receives the carried
        exception exactly (NOT wrapped in FatalError: wrapping would promote every child
        budget-stop into a tree-wide kill; how far the stop travels belongs to the carried
        exception's own category, see the ``Abort`` effect docstring).
        """
        # ExceptionGroup's message is agent-facing; keep it generic (see _combine_exceptions).
        exc = _combine_exceptions(list(abort_errors), "Multiple errors occurred")
        self._own_abort_exceptions.append(exc)
        return exc

    def _unwind_outcome(self, exc: BaseException) -> Aborted | Failed:
        """Classify a span's non-completed close: ``Aborted`` iff ``exc`` is THIS invoke's
        own resolved abort (by identity), else ``Failed``.

        This is the ONLY place :class:`Aborted` is constructed, and the rule is
        **control-plane association** (user decision, final — it supersedes the archived
        design doc's category-based rule; review of the implementation plan sharpened
        this): a span closes ``Aborted`` iff its OWN invoke's hook control plane issued the
        ``Abort`` effect — the stage CM that resolved it closes ``Aborted`` directly, and
        the same invoke's enclosing spans (the invoke span, when its inner query/exec span
        aborted) match the recorded exception by identity on the unwind, because the effect
        is named ``Abort``-the-*invoke*: it is this invoke's control plane speaking,
        whichever stage it composed at. EVERY other non-completion is ``Failed``: a
        ``FatalError`` unwinding from below, a child invoke's propagated abort-carried
        exception (a foreign control plane's decision — the enclosing invoke's spans close
        ``Failed`` with it), ``KeyboardInterrupt``, provider errors, machinery bugs.

        Why association and not exception category: intent is unknowable at the escalation
        channel — ``Abort(error=...)`` carries governance stops (a budget ceiling) and
        breakage reports (a tool's ``FatalError``) alike, which is the same muddiness that
        renamed the category to ``FatalError`` (a property claim, not an intent claim).
        .NET draws exactly this line: an ``OperationCanceledException`` moves a task to
        ``Canceled`` ONLY when its token matches the task's own ``CancellationToken``;
        the same exception with a foreign token means ``Faulted``. Escalation scope —
        ``FatalError`` being unrenderable/uncontainable by agents, the authorship boundary
        — is an ORTHOGONAL axis to this outcome: an ``Aborted`` invoke may carry a
        containable ``BudgetExhaustedError`` or an uncontainable ``FatalError``; the
        outcome says whose stop it was, the category says how far it travels.
        """
        if any(exc is recorded for recorded in self._own_abort_exceptions):
            return Aborted(exc)
        return Failed(exc)

    # Span-based context managers for enter/exit event pairs

    @contextmanager
    def span_repl_exec(self, enter_event: REPLExecEnter):
        """Context manager for REPL execution span.

        Usage:
            enter_event = REPLExecEnter(...)
            with dispatcher.span_repl_exec(enter_event) as span:
                # ...apply span.ctx namespace deltas...
                span.send()  # REPLExecSend: suppliers compose here (an Abort raises)
                if span.supplied is not None:
                    exec_result = span.supplied  # a hook supplied the result
                else:
                    exec_result = repl.exec(...)
                span.complete(exec_result=exec_result)
            # REPLExecComplete-composed ModifyExecResult is applied here:
            exec_result = span.get_final_exec_result()

        The span must be completed before exiting cleanly — a body that skips
        ``complete()`` trips a loud ERROR log and fires no exit (the
        ``_log_incomplete_span`` tripwire; synthesizing a close would mask the bug).

        An ``Abort`` composed at any of this span's control stages raises its carried
        exception out of the ``with`` statement (from ``span.send()`` for a Send-time one),
        after the span closes with an ``Aborted`` outcome — there is no result for the
        loop to read on that path.

        Layering of Send-time supply and Complete-time transform (both intentional):
          1. When `supplied` (a Send-composed SupplyExecResult supply)
             short-circuits execution, `REPLExecComplete` and `REPLExecExit` still fire with
             that result. Observability hooks therefore record an outcome for code that
             never ran — they care about what happened from the agent's perspective, not
             whether `repl.exec` was called (a supplied result is a result).
          2. Complete-time `ModifyExecResult`s compose on top of whatever result the
             producer made. This is the composition rule that lets `BudgetForcing`
             transform a `Return` into a `Continue`, and it deliberately does not check the
             origin of the prior result. The origin-agnostic rule cuts both ways: because
             the result *kind* is decided by the effects present at *this* (transform)
             boundary, a `ModifyExecResult(Continue(...))` composed onto a supplied terminal
             (`Return` / `Raise`) *downgrades* it to a recoverable `Continue` — the
             transform boundary's own effects are the last word (see
             `resolve_modify_results`). An Abort is exempt from that rule by construction:
             it raises before the fold, so no transform can revoke it.
        """
        # Fire the enter event. No CM-side timing: every event carries its own
        # emission ``timestamp`` (stamped in emit()); intervals are consumer arithmetic.
        effects = self.emit(enter_event)
        # The try opens IMMEDIATELY after the enter-emit returns (close-then-propagate,
        # #892): once any observer has opened its span, an exception anywhere — including
        # enter-effect COMPOSITION raising a deliberate conflict error — must close the
        # span (Exit fires on the abnormal arm below) and then propagate. Composition
        # errors are never absorbed: catching-and-logging them (the way emit() guards a
        # buggy hook) would silently void the conflict contract; the fix is closing the
        # span on the way out, never swallowing the exception.
        span: REPLExecSpan | None = None
        # True once this CM has emitted the span's Exit — set by _abort (Aborted arm) and
        # by the normal close — so the abnormal arm below can't double-close.
        closed = False

        def _exit_event(outcome: Completed | Aborted | Failed) -> REPLExecExit:
            return REPLExecExit(
                config=enter_event.config,
                invoke_id=enter_event.invoke_id,
                iteration=enter_event.iteration,
                depth=enter_event.depth,
                outcome=outcome,
            )

        def _abort(abort_errors: list[Exception]) -> NoReturn:
            # A composed Abort at any of this span's control stages (Enter/Send/Complete):
            # close the span Aborted, then raise the carried exception — the abort
            # propagates instead of being laundered into a terminal Raise result, so no
            # downstream transform boundary exists at which another hook could revoke it
            # (span_event_lifecycle.md Defect 3, closed structurally). The guarded
            # emission-then-raise ordering is what distinguishes this close from the
            # generic abnormal arm below, which stays Failed.
            nonlocal closed
            exc = self._record_own_abort(abort_errors)
            closed = True
            self._emit_guarded_exit(_exit_event(self._unwind_outcome(exc)))
            raise exc

        try:
            ctx = self._compose_repl_exec(effects)
            if ctx.abort_errors:
                _abort(ctx.abort_errors)
            span = REPLExecSpan(ctx=ctx)

            def _emit_send() -> None:
                # REPLExecSend payload is fully known here (code from the enter event,
                # namespace deltas from composition); the loop just marks the commit point.
                send_event = REPLExecSend(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    depth=enter_event.depth,
                    iteration=enter_event.iteration,
                    code=enter_event.code,
                    # Passed uncopied on purpose: the event's __post_init__ owns the
                    # severing (copy) and freezing (proxy/frozenset) — a call-site copy
                    # here would be a second, redundant one. Same at InvokeSend.
                    added_variables=ctx.added_variables,
                    dropped_variables=ctx.dropped_variables,
                )
                send_ctx = self._compose_repl_exec_send(self.emit(send_event))
                # Termination trumps supply: a Send-time Abort raises out of span.send()
                # before any supplied result is even resolved.
                if send_ctx.abort_errors:
                    _abort(send_ctx.abort_errors)
                # Send is the supply boundary: a supply names THE result outright (no
                # original to fold); pure supply — aborts never launder in here anymore.
                assert span is not None
                span.supplied = resolve_supply_results(
                    supply_effects=send_ctx.supply_effects,
                )

            span._emit_send = _emit_send
            yield span

            # ---- Normal close (the body finished cleanly) ----
            if not span.is_completed():
                # The forgot-to-complete programming error — loud ERROR tripwire, no exit.
                _log_incomplete_span(
                    f"REPLExec span for execution {enter_event.iteration} was not "
                    "completed. You must call span.complete(exec_result=...) before exiting."
                )
            else:
                original = span.get_exec_result()
                # Transform boundary first: REPLExecComplete fires with the RAW result and
                # its ModifyExecResult effects compose into the final one; an Abort here
                # supersedes the fold and raises (the span closes Aborted).
                complete_event = REPLExecComplete(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    iteration=enter_event.iteration,
                    exec_result=original,
                    depth=enter_event.depth,
                )
                complete_ctx = self._compose_repl_exec_complete(
                    self.emit(complete_event)
                )
                if complete_ctx.abort_errors:
                    _abort(complete_ctx.abort_errors)
                transformed = resolve_modify_results(
                    modify_effects=complete_ctx.modify_effects,
                    original=original,
                )
                final = transformed if transformed is not None else original
                span.set_final_exec_result(final)
                # Exit carries the POST-transform result (#906 fixed by the Complete→Exit
                # ordering): observers of the close finally record what the turn actually
                # produced, not composition's input. Observation-only — returned effects
                # are rejected loudly. `closed` is set before the emit so a raise from the
                # emission (or the rejection itself) propagates without a double-close.
                exit_event = _exit_event(Completed(final))
                closed = True
                self._reject_observation_only_effects(self.emit(exit_event), exit_event)
        except BaseException as exc:
            # Abnormal close (#892): the span opened, so it must close exactly once, here,
            # with the true outcome — then the exception propagates untouched. `closed`
            # means this CM already emitted the exit (an Aborted close whose raise is now
            # passing through, or a failure after the Completed exit fired).
            # _unwind_outcome classifies: Aborted iff the exception is this invoke's own
            # resolved abort (e.g. raised by the LLM-query span's CM and unwinding through
            # the loop); every other unwind is Failed.
            if not closed:
                self._emit_guarded_exit(_exit_event(self._unwind_outcome(exc)))
            raise

    @contextmanager
    def span_llm_query(self, enter_event: LLMQueryEnter):
        """Context manager for LLM query span.

        Usage:
            enter_event = LLMQueryEnter(...)
            with dispatcher.span_llm_query(enter_event) as span:
                span.send(shown_messages)      # LLMQuerySend: suppliers compose here
                if span.supplied_response is not None:
                    response = span.supplied_response
                else:
                    response = llm.query(shown_messages, **base_config)
                span.complete(response=response)
            return result

        :class:`LLMQueryEnter` is the always-present per-turn boundary, so :class:`Abort`
        is honored here (the loop/budget hard-stops). An abort at any of this span's
        control stages raises its carried exception: from the ``with`` statement itself for
        an enter-time one (before the query), from ``span.send(...)`` for the
        committed-input veto, and from the block's close for a :class:`LLMQueryComplete`
        one — in every case the span closes first with an ``Aborted`` outcome on
        :class:`LLMQueryExit` (every opened span closes, on every path).

        A Complete-time abort does *not* un-do the query — it completed, was paid for, and
        its ``LLMQueryComplete`` fired and was observed by every hook. What the abort stops
        is everything downstream: the raise propagates before the caller reads the result,
        so the agent never acts on the response and the code the model proposed is not
        executed.
        """
        # Fire the enter event. No CM-side timing: every event carries its own
        # emission ``timestamp`` (stamped in emit()); intervals are consumer arithmetic.
        effects = self.emit(enter_event)
        # The try opens IMMEDIATELY after the enter-emit (close-then-propagate, #892): an
        # enter-composition conflict error must close the span, never be absorbed — see
        # span_repl_exec for the full rationale.
        span: LLMQuerySpan | None = None
        # True once this CM has emitted the span's Exit (see span_repl_exec).
        closed = False

        def _exit_event(outcome: Aborted | Failed) -> LLMQueryExit:
            return LLMQueryExit(
                config=enter_event.config,
                invoke_id=enter_event.invoke_id,
                model=enter_event.model,
                iteration=enter_event.iteration,
                depth=enter_event.depth,
                outcome=outcome,
            )

        def _abort(abort_errors: list[Exception]) -> NoReturn:
            # Composed Abort: close the span Aborted, then raise the carried exception —
            # propagation, not laundering (see span_repl_exec._abort for the rationale).
            nonlocal closed
            exc = self._record_own_abort(abort_errors)
            closed = True
            self._emit_guarded_exit(_exit_event(self._unwind_outcome(exc)))
            raise exc

        try:
            ctx = self._compose_llm_query(effects)
            # Abort @ LLMQueryEnter (iteration / budget hard-stop): the query never
            # happens — the span closes Aborted and the exception raises out of the
            # `with` statement before the body runs.
            if ctx.abort_errors:
                _abort(ctx.abort_errors)
            span = LLMQuerySpan(ctx=ctx)

            def _emit_send(shown_messages: "list[MessageDict]") -> None:
                # Resolve the edit records HERE — the dispatcher holds both the enter
                # snapshot and the composed edits (see _resolve_message_edits).
                send_event = LLMQuerySend(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    depth=enter_event.depth,
                    messages=shown_messages,
                    model=enter_event.model,
                    iteration=enter_event.iteration,
                    edits=self._resolve_message_edits(
                        enter_event.messages, ctx.message_edits
                    ),
                )
                # Send is the supply boundary: a SupplyLLMResponse (e.g. Replay) answers
                # here with the committed list in view, overriding the default producer;
                # an Abort declines the input (raising out of span.send).
                send_ctx = self._compose_llm_query_send(self.emit(send_event))
                if send_ctx.abort_errors:
                    _abort(send_ctx.abort_errors)
                assert span is not None
                span.supplied_response = send_ctx.supplied_response

            span._emit_send = _emit_send
            yield span

            # ---- Normal close (the body finished cleanly) ----
            if not span.is_completed():
                # The forgot-to-complete programming error — loud ERROR tripwire, no exit.
                _log_incomplete_span(
                    "LLMQuery span was not completed. "
                    "You must call span.complete(...) before exiting."
                )
            else:
                # Response control boundary first: LLMQueryComplete fires with the raw
                # response and its Abort effects compose (no response-modify effect
                # exists yet — the slot is documented on the event). An abort here closes
                # the span Aborted and raises out of the `with` close — after the query
                # was observed, before the caller reads its content.
                complete_event = LLMQueryComplete(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    response=span.get_response(),
                    model=enter_event.model,
                    # From the enter event, the family's single iteration source — the
                    # span's own (mismatchable) channel was removed (audit C3).
                    iteration=enter_event.iteration,
                    depth=enter_event.depth,
                )
                complete_ctx = self._compose_llm_query_complete(
                    self.emit(complete_event)
                )
                if complete_ctx.abort_errors:
                    _abort(complete_ctx.abort_errors)
                # Exit closes the observation record — after Complete, so observers see
                # the final state of the turn. Observation-only: returned effects are
                # rejected loudly; `closed` is set before the emit so a raise from the
                # emission propagates without a double-close.
                exit_event = LLMQueryExit(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    model=enter_event.model,
                    iteration=enter_event.iteration,
                    depth=enter_event.depth,
                    outcome=Completed(span.get_response()),
                )
                closed = True
                self._reject_observation_only_effects(self.emit(exit_event), exit_event)
        except BaseException as exc:
            # Abnormal close (#892): fire the exit with the classified outcome
            # (_unwind_outcome: Aborted iff this invoke's own resolved abort, else
            # Failed), then let the exception propagate untouched.
            if not closed:
                self._emit_guarded_exit(_exit_event(self._unwind_outcome(exc)))
            raise

    @contextmanager
    def span_invoke(self, enter_event: InvokeEnter):
        """Context manager for invoke span.

        Usage:
            enter_event = InvokeEnter(...)
            with dispatcher.span_invoke(enter_event) as span:
                # Use span.ctx.added_inputs / dropped_inputs, etc.
                result = agent._invoke_internal(...)
                span.complete(result=result)

        An :class:`Abort` composed at :class:`InvokeEnter` raises its carried exception
        from the ``with`` statement itself (the span closes ``Aborted`` first); one at
        :class:`InvokeSend` raises out of ``span.send(...)``; one at
        :class:`InvokeComplete` raises from the block's close. The exception propagates
        out of ``jaz.invoke()`` bare — the top-level caller contract.
        """
        # Fire the enter event. No CM-side timing: every event carries its own
        # emission ``timestamp`` (stamped in emit()); intervals are consumer arithmetic.
        effects = self.emit(enter_event)
        # The try opens IMMEDIATELY after the enter-emit (close-then-propagate, #892): an
        # enter-composition conflict error must close the span, never be absorbed — see
        # span_repl_exec for the full rationale.
        span: InvokeSpan | None = None
        # True once this CM has emitted the span's Exit (see span_repl_exec).
        closed = False

        def _exit_event(outcome: Completed | Aborted | Failed) -> InvokeExit:
            return InvokeExit(
                config=enter_event.config,
                invoke_id=enter_event.invoke_id,
                depth=enter_event.depth,
                outcome=outcome,
            )

        def _abort(abort_errors: list[Exception]) -> NoReturn:
            # Composed Abort: close the span Aborted, then raise the carried exception —
            # propagation, not laundering (see span_repl_exec._abort for the rationale).
            nonlocal closed
            exc = self._record_own_abort(abort_errors)
            closed = True
            self._emit_guarded_exit(_exit_event(self._unwind_outcome(exc)))
            raise exc

        try:
            ctx = self._compose_invoke(effects)
            # Enter-time Abort short-circuits the whole invoke (no execution has happened
            # yet): the span closes Aborted and the carried exception raises out of the
            # `with` statement before the loop starts.
            if ctx.abort_errors:
                _abort(ctx.abort_errors)
            span = InvokeSpan(ctx=ctx)

            def _emit_send(inputs: "Any") -> None:
                # The committed inputs come from the loop (it owns the input fold); the
                # add/drop payloads come from composition here.
                send_event = InvokeSend(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    depth=enter_event.depth,
                    inputs=inputs,
                    # Uncopied — the event's __post_init__ owns severing + freezing
                    # (see the REPLExecSend construction).
                    added_inputs=ctx.added_inputs,
                    dropped_inputs=ctx.dropped_inputs,
                )
                # Send composes abort only (no invoke supplier effect exists): the
                # combined exception raises out of span.send(), declining the committed
                # input set.
                send_ctx = self._compose_invoke_send(self.emit(send_event))
                if send_ctx.abort_errors:
                    _abort(send_ctx.abort_errors)

            span._emit_send = _emit_send
            yield span

            # ---- Normal close (the body finished cleanly) ----
            if not span.is_completed():
                # The forgot-to-complete programming error — loud ERROR tripwire, no exit.
                _log_incomplete_span(
                    "Invoke span was not completed. "
                    "You must call span.complete(result=...) before exiting."
                )
            else:
                # Transform boundary first: InvokeComplete fires with the RAW terminal
                # result and its ModifyExecResult effects compose (#568); an Abort here
                # supersedes the fold and raises (the span closes Aborted). A
                # ReturnType / ValidateReturn hook downgrades a wrong-typed / invalid final
                # Return to a Raise here (the backstop for another hook's REPLExecComplete
                # override reinstating a Return). The (possibly overridden) result is stored
                # back on the span; the invoke loop reads span.get_result() AFTER the span
                # closes and returns/raises from that.
                original = span.get_result()
                complete_event = InvokeComplete(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    result=original,
                    depth=enter_event.depth,
                )
                complete_ctx = self._compose_invoke_complete(self.emit(complete_event))
                if complete_ctx.abort_errors:
                    _abort(complete_ctx.abort_errors)
                transformed = resolve_modify_results(
                    modify_effects=complete_ctx.modify_effects,
                    original=original,
                )
                if transformed is not None:
                    span.set_final_result(transformed)
                # Exit carries the POST-transform result — the #906 fix, delivered by the
                # Complete→Exit ordering: observers of the close finally record what the
                # invoke actually returned/raised, backstop downgrades included, instead of
                # composition's input. Observation-only — returned effects are rejected
                # loudly; `closed` is set before the emit so a raise from the emission
                # propagates without a double-close.
                final_result = span.get_result()
                # An invoke only ever exits on a *terminal* result (the loop completes the
                # span solely from its Return/Raise arm; a Continue appends an observation
                # and iterates), which is what lets InvokeOutcome's Completed payload be
                # ``Return | Raise`` rather than the full ExecResult. Asserted here — the
                # one place the payload enters the outcome — so a leaked Continue fails at
                # the source instead of in whichever observer matches the union first.
                assert isinstance(final_result, Return | Raise)
                exit_event = _exit_event(Completed(final_result))
                closed = True
                self._reject_observation_only_effects(self.emit(exit_event), exit_event)
        except BaseException as exc:
            # Abnormal close (#892): fire the exit with the classified outcome, then let
            # the exception propagate untouched. _unwind_outcome is where an invoke span
            # closes Aborted for its OWN invoke's aborts resolved at an inner span's stage
            # (the LLM-query or REPL-exec CM recorded the exception, and it is unwinding
            # through this loop now); any foreign exception — a child invoke's propagated
            # abort, a FatalError from below, KeyboardInterrupt — closes Failed.
            if not closed:
                self._emit_guarded_exit(_exit_event(self._unwind_outcome(exc)))
            raise

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
