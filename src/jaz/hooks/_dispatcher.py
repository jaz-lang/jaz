"""HookDispatcher orchestrates hooks and composes their effects into execution contexts."""

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NoReturn, assert_never

from jaz._property_contract import require_properties
from jaz.exceptions import (
    BlackboardSeedConflictError,
    BlackboardWriteConflictError,
    FatalError,
    HookActivationError,
    InvalidEffectError,
    KeyEditDeniedError,
    LLMResponseConflictError,
    REPLInputConflictError,
    ReturnValueConflictError,
    _JazInternalError,
)
from jaz.repl._types import Raise, Return

from ._base import Event
from ._context import (
    _pop_activation_token,
    _push_activation_token,
    _reset_hook_context,
    _set_hook_context,
    get_current_hooks,
)
from .effects import (
    UNSET,
    Abort,
    AddInputs,
    AddMessages,
    AddToNamespace,
    BlackboardWrite,
    DeleteCode,
    DisableRecursion,
    DropFromNamespace,
    DropInputs,
    DropMessages,
    Effect,
    InsertCode,
    ModifyExecResult,
    ModifyInvokeResult,
    ModifyLLMResponse,
    SupplyExecResult,
    SupplyInvokeResult,
    SupplyLLMResponse,
    WithholdREPLHistory,
    _apply_code_edits,
    _combine_exceptions,
    _resolve_blackboard_writes,
    _resolve_code_offset,
    _resolve_edit_index,
    _resolve_invoke_modify_results,
    _resolve_invoke_supply_results,
    _resolve_llm_modify_results,
    _resolve_modify_results,
    _resolve_supply_results,
    _uncomparable_note,
    _values_conflict,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jaz._messages import MessageDict
from .events import (
    CodeDeleteRecord,
    CodeInsertRecord,
    InputAddRecord,
    InputDropRecord,
    InvokeComplete,
    InvokeEnter,
    InvokeExit,
    InvokeSend,
    LLMQueryComplete,
    LLMQueryEnter,
    LLMQueryExit,
    LLMQueryRetry,
    LLMQuerySend,
    MessageAddRecord,
    MessageDropRecord,
    NamespaceAddRecord,
    NamespaceDropRecord,
    REPLExecComplete,
    REPLExecEnter,
    REPLExecExit,
    REPLExecSend,
    ResolvedCodeEdits,
    ResolvedInputEdits,
    ResolvedMessageEdits,
    ResolvedNamespaceEdits,
)
from .events._base import Aborted, Completed, Failed
from .events._invoke import (
    InvokeCompleteContext,
    InvokeContext,
    InvokeSendContext,
    InvokeSpan,
)
from .events._llm_query import (
    LLMQueryCompleteContext,
    LLMQueryContext,
    LLMQueryRetryContext,
    LLMQuerySendContext,
    LLMQuerySpan,
    MessageEdits,
)
from .events._repl_execution import (
    REPLExecCompleteContext,
    REPLExecContext,
    REPLExecSendContext,
    REPLExecSpan,
)

logger = logging.getLogger(__name__)


#: The effects that compose through ``_coalesce_add`` — the "add" family.
type _AddEffect = Literal["AddInputs", "AddToNamespace"]


def _reservation_attr(effect: _AddEffect) -> str:
    """The ``Hook`` reservation that would have caught this effect's conflict at activation."""
    # A ``match`` with ``assert_never`` rather than a dict lookup so the type checker enforces
    # what this mapping claims: widening ``_AddEffect`` without adding a case here fails
    # ``pyright``. A dict would type-check clean and surface the miss at runtime as a ``KeyError``
    # raised from inside the conflict branch — masking the ``REPLInputConflictError`` the author
    # was supposed to see, which is a worse failure than the one this replaced.
    match effect:
        case "AddInputs":
            return "reserved_input_names"
        case "AddToNamespace":
            return "reserved_namespace_names"
        case _:
            assert_never(effect)


def _coalesce_add(
    target: dict[str, object],
    additions: dict[str, object],
    *,
    effect: _AddEffect,
    refuse_protected_keys: bool = False,
) -> None:
    """Union ``additions`` into ``target`` with the family's shared **conflict-error** rule
    (``AddInputs`` / ``AddToNamespace``): two hooks adding the same key with the *same* value
    coalesce; the same key with a *different* value raises ``REPLInputConflictError``. Making a
    divergent add loud (rather than last-write-wins) keeps composition order-independent, exactly
    like ``BlackboardWrite``. ``refuse_protected_keys`` **raises** ``KeyEditDeniedError`` if ``__builtins__``
    is named (mirrors ``DropFromNamespace``), so a namespace/input add can't reopen the compiler sandbox
    — a loud refusal rather than a silent skip that would hide the hook bug.

    Conflict test is identity-first, then a *guarded* ``!=`` (see ``_values_conflict``): two distinct
    objects that don't define ``__eq__`` compare unequal (a genuine conflict), which is the safe
    default for injected values.
    """
    for key, value in additions.items():
        if refuse_protected_keys and key == "__builtins__":
            raise KeyEditDeniedError(
                f"{effect} cannot bind '__builtins__' — it holds the compiler sandbox "
                "(#688/#690); binding it would reopen import/attribute/builtins policy."
            )
        if key in target and _values_conflict(target[key], value):
            # Name the reservation: this error fires whenever the two writes first meet, which may
            # be many paid turns in. Reserving the key on each hook moves the same rejection to
            # activation (``HookDispatcher._reject_declared_write_conflicts``).
            attr = _reservation_attr(effect)
            raise REPLInputConflictError(
                f"{effect} conflict: '{key}' was added with two different values by "
                f"different hooks (composition must be order-independent). Set "
                f"{attr} = frozenset({{{key!r}}}) on each hook that writes it to have this "
                f"pairing rejected at activation instead, before the run spends anything."
                + _uncomparable_note(target[key], value)
            )
        target[key] = value


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
    (each gets its own context copy). Across a process boundary it follows the start method
    (https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods): under ``fork``
    the child inherits the activation, and with it the parent's instance holding the parent's
    open files, locks and spans; under ``spawn`` and ``forkserver`` the hook is simply absent.
    Activate hooks inside the worker.

    Leading positional argument — one invoke only::

        invoke(MyHook(), task=...)   # active for THIS invoke, NOT for invokes it nests

    A hook passed this way applies to that invoke alone — the invokes it nests never see it, so
    anything it accumulates or enforces (a budget, a limit, a log) covers the outer invoke's
    own turns and nothing below them. Use ``with`` when sub-invokes must be included.

    **Re-activating one instance.** A hook may be activated again after its previous scope
    ended — the same ``FileLogger`` or ``BudgetPool`` passed to several invokes in turn. Only
    *overlapping* activation is rejected. A hook that carries per-run state must therefore pick
    one of two policies, because nothing else will:

    - **Clean**, when everything the run produced lives in the hook's own memory. Clear that
      state so each activation starts empty. Clear it in ``setup`` if a caller is meant to read
      the finished run *after* the scope ends, in ``teardown`` if promptly releasing it matters
      more.
    - **Reject**, when a second run would *destroy* what the first one produced — typically a
      file the hook rewrites from scratch each time. Cleaning memory does not un-write a file.
      Call :meth:`forbid_reactivation` at the top of ``setup``.

    Producing output outside the hook does not by itself call for rejecting: what matters is
    whether reuse overwrites it. A span exporter appends, so re-activating a tracing hook adds
    the second run's spans beside the first's and needs neither policy.

    A hook with no per-run state, such as a validator that only inspects each event, needs
    neither and can be reused freely.

    **Declaring what you write.** The ``reserved_*`` / ``reserves_*`` attributes below let a hook
    declare, up front, the slots it intends to write. They are entirely optional: a hook that
    writes a slot without declaring it works exactly as before. What a declaration buys is *when*
    a conflict surfaces — two hooks that both claim one slot are rejected at seed time, before the
    first event of the next :func:`jaz.invoke`, instead of when their two writes happen to meet,
    which can be deep into a paid-for run. Declare when the slot is one only your hook should own;
    skip it otherwise. One caveat, because it can turn a working composition into an error: the
    seed-time check is *stricter* than the runtime one — it rejects two declarers of a slot even
    when they would write identical values, where runtime would have coalesced them.

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

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        require_properties(cls, ("blackboard_seed",))

    # Maintainers: ``hooks/README.md`` is the contributor guide — design philosophy, the
    # contextvar/event-flow architecture, ``HookDispatcher``, and how to add a new event,
    # effect or hook. It is deliberately not linked from the docstring above: it is a repo
    # path a reader of the published reference cannot follow, and its content is for people
    # changing the hook system rather than using it.
    #
    # The docstring links nowhere else either. The website's Hooks guide
    # (``/docs/guides/hooks``) was the obvious candidate and is currently unusable: it lists six
    # hooks that do not exist (``WorkflowStrategyHook``, ``MemoryStoreHook``,
    # ``BudgetTrackingHook``, ``JaegerTracingHook``, ``LangfuseTracingHook``,
    # ``WorkflowReplayHook``), and every code sample is broken — the removed
    # ``return_type=`` keyword, a positional prompt instead of ``task=``, an ``on_event`` handler
    # that no longer exists, and effects/events (``ReplIterationEnter``, ``AddInstructionPrompt``)
    # that were never in this API. Restore the link once that page is rewritten; until then it
    # would send readers somewhere strictly worse than this docstring.

    def _dispatch_event(self, event: Event) -> list[Effect]:
        """The framework event router: dispatch an event to its typed ``on_<event>``
        handler (the override points below).

        **Do not override this.** It is the dispatcher's internal router; overriding it
        replaces the routing, so any ``on_<event>`` handlers would silently never run.
        To extend a hook, override the typed handlers below — you get typed
        attribute access and implement only the events you care about, with no
        ``isinstance``/``match`` boilerplate and no "remember to ``return []``" footgun.

        For a genuinely *cross-cutting* observer that treats every event uniformly (loggers,
        tracing), override :meth:`~Hook.on_any` instead: the dispatcher calls it *in addition to*
        the matched typed handler, so it composes with — rather than replaces — the
        per-event dispatch, and a hook may use both.

        Returns the effects expressing how to influence execution (empty for none).
        """
        # The typed-handler dispatch replaced a single untyped on_event in #597.
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
        """An execution's code has committed (namespace deltas applied). Default no-op."""
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

    # Static write-authority reservations (see ``HookDispatcher._reject_declared_write_conflicts``).
    # Writes, not seeds. Every slot here is decided DURING a run, which is what makes an
    # unreserved collision expensive — it surfaces whenever the two writers happen to meet.
    # A seed is settled before the first event fires, so its collision is already free and
    # needs no declaration; ``blackboard_seed`` is deliberately absent from this family. Reserving
    # a seeded key is worse than pointless: it cannot stop an undeclared writer, and if the writer
    # does declare, the pair is refused and the invoke never starts.
    # Each of the effects below is *slot-scoped* with a single authority per slot: the turn for
    # ``SupplyLLMResponse``, the key/name for the others. Two active hooks reserving the same slot
    # is an irresolvable, order-dependent authority conflict, so it is rejected at seed time (before
    # the first ``InvokeEnter``) rather than mid-run. These are OPT-IN and default-empty: a hook that
    # writes a slot WITHOUT reserving it is not seen by the seed-time check but is still caught
    # dynamically by the same conflict error at composition time — the expensive path, because a
    # conflict that first fires deep into a run kills the invoke with the run's spend already gone.
    # The runtime conflict errors therefore name the reservation that would have caught them here.
    #
    # TWO FORMS, split by arity (EXECUTIVE CALL, user, 2026-09-19). A slot named by a string is a
    # key set and takes the participle — ``reserved_input_names``, ``reserved_blackboard_keys`` —
    # which is how every other collection-valued declaration in this package reads
    # (``extra_message_keys``, ``added_inputs``, ``expected_outputs``). A slot
    # with exactly one instance per boundary is a flag and takes the verb —
    # ``reserves_llm_response``, ``reserves_exec_return`` — matching the package's other predicate
    # attributes (``MessageEdits.has_persistent``, ``TrajectoryReplay.is_exhausted``). The uniform
    # ``reserves_*`` spelling this replaced made the grammar promise a boolean in four of the seven
    # members, and was the only leading-``verbs_`` collection in ``src/jaz``.
    #
    # The cost of the split, stated because it is not derivable the way the property/``get_`` rule
    # is: a reader who meets one member cannot tell the others are spelled differently. Arity is
    # the discriminator anyway — it is caller-visible, and a member changing arity is already a
    # breaking type change, so the rename it forces is correct rather than churn. A future
    # declaration that is neither a key set nor a flag (a ``dict`` of key to reason, say) has no
    # obvious inflection: give it the participle, since it is a collection.
    #
    # Named ``reserv*`` (EXECUTIVE CALL, user, 2026-09-19), not ``adds_*``/``writes_*``: this is a
    # claim of exclusive ownership, not a description of behavior — two reservers of a slot are
    # rejected whether or not either ever writes it. ``priority_*`` was weighed and rejected for the
    # opposite reason: priority implies conflict *resolution* (this hook wins, the other's write is
    # dropped), and silently dropping a write is what the sole-authority rule exists to prevent — the
    # losing hook would go on believing it bound a value the agent never saw.
    #
    # Deliberately NOT ``ClassVar``: a hook parameterized by key name (``MemoryHook(key=...)``) can
    # only reserve in ``__init__``, and the check already reads the reservation off the instance
    # (``getattr(hook, attr)``), so per-instance reservation works at runtime. The ``ClassVar``
    # annotation was the one thing forbidding it — the type checker rejected the very assignment the
    # pre-flight honors, which kept the hooks most likely to collide out of the early check.
    #
    # The seed-time predicate is deliberately STRICTER than the runtime one, not merely earlier. At
    # runtime, two writes to one slot with *identical values* COALESCE and only *divergent* values
    # conflict (``effects._values_conflict``, the comparison every composition check uses). This check
    # instead rejects any two *declarers* of a slot regardless of value — a coalescing duplicate is
    # still two authorities for one slot, which under "one authority per slot" is a config smell worth
    # surfacing (the same executive call made for two identical ``TrajectoryReplay``s: a pointless
    # duplicate should fail loudly, not silently coalesce). So the runtime check remains the looser
    # backstop for an UNDECLARED write, while a DECLARED identical-value pair is intentionally
    # rejected here even though runtime would have coalesced it. Consequence for a defensive author:
    # adding a declaration to a composition that previously coalesced turns it into a seed error.
    # (Same-slot always rejects — there is no reducer/accumulator carve-out in this design.)

    # The key sets first, then the flags — the two forms the comment above describes.
    # Attribute docstrings, not ``#:`` comments: griffe reads a string literal below the
    # assignment and returns ``None`` for ``#:``, so the comment form reaches neither ``help()``
    # nor the generated API reference — the text would be written for users and shipped to
    # nobody.
    reserved_input_names: frozenset[str] = frozenset()
    """Input keys this hook reserves for its :class:`~jaz.hooks.effects.AddInputs` writes.

    Two active hooks reserving the same key are rejected at seed time — when the next
    :func:`jaz.invoke` assembles the blackboard, not when the hook is activated — with
    :class:`~jaz.exceptions.REPLInputConflictError`. Optional, like every declaration here: an
    unreserved collision still raises, but only once the two writes meet. Set it on the class, or
    per instance in ``__init__`` when the key is a constructor argument.
    """

    reserved_namespace_names: frozenset[str] = frozenset()
    """REPL namespace names this hook reserves for its
    :class:`~jaz.hooks.effects.AddToNamespace` writes.

    Two active hooks reserving the same name are rejected at seed time with
    :class:`~jaz.exceptions.REPLInputConflictError`. Set it on the class, or per instance in
    ``__init__`` when the name is a constructor argument.
    """

    reserved_llm_response_fields: frozenset[str] = frozenset()
    """:class:`~jaz.llm.LLMResponse` fields this hook reserves for its
    :class:`~jaz.hooks.effects.ModifyLLMResponse` writes (``"content"``, ``"cost_usd"``, ...).

    Authority is per *field*: two active hooks reserving the same field are rejected at seed time
    with :class:`~jaz.exceptions.LLMResponseConflictError`, while hooks that set disjoint fields
    compose. Set it on the class, or per instance in ``__init__``. Distinct from
    ``reserves_llm_response`` despite the name: that one is authorship of the turn at
    ``LLMQuerySend``, this one is per-field editing of the response at ``LLMQueryComplete``, and
    neither is checked against the other.
    """

    reserved_blackboard_keys: frozenset[str] = frozenset()
    """Blackboard keys this hook reserves for its :class:`~jaz.hooks.effects.BlackboardWrite`
    writes.

    Two active hooks reserving the same key are rejected at seed time with
    :class:`~jaz.exceptions.BlackboardWriteConflictError`.  Set it on the class, or per instance
    in ``__init__`` when the key is a constructor argument.

    **Writes only — do not reserve a key you merely seed.** Seeds need no reservation: they are
    assembled before the first event, so two hooks seeding one key with different values already
    raise :class:`~jaz.exceptions.BlackboardSeedConflictError` there, at no cost. Reserving a
    seeded key cannot stop an undeclared writer, and against one that *does* declare it is
    actively harmful: seeding a key and writing it at a later event is legal — the later
    generation overwrites — but two declarers of one key are refused whatever they meant to do
    with it, so the invoke never starts.
    """

    reserves_llm_response: bool = False
    """Whether this hook reserves authorship of the model's turn — whether it may emit
    :class:`~jaz.hooks.effects.SupplyLLMResponse`.

    That effect is *sole-authority* — one response per turn — so two active hooks both reserving
    it are rejected at seed time with :class:`~jaz.exceptions.LLMResponseConflictError`. Set it on
    the class, or per instance in ``__init__``.
    """

    reserves_exec_return: bool = False
    """Whether this hook reserves the *return value* of a REPL execution — whether it may carry a
    :class:`~jaz.repl.Return` in a :class:`~jaz.hooks.effects.SupplyExecResult` or
    :class:`~jaz.hooks.effects.ModifyExecResult`.

    NOT "may emit that effect": several hooks carrying :class:`~jaz.repl.Continue` at one boundary
    is normal and composes (outputs concatenate, exceptions group), and two
    :class:`~jaz.repl.Raise` results group their exceptions. Only a ``Return`` is sole-authority,
    because two distinct return values cannot merge — which is why the name says ``return`` and
    not ``result``. Two active hooks reserving it are rejected at seed time with
    :class:`~jaz.exceptions.ReturnValueConflictError`. Set it on the class, or per instance in
    ``__init__``.
    """

    reserves_invoke_return: bool = False
    """Whether this hook reserves the *return value* of an invoke — whether it may carry a
    :class:`~jaz.repl.Return` in a :class:`~jaz.hooks.effects.SupplyInvokeResult` or
    :class:`~jaz.hooks.effects.ModifyInvokeResult`.

    The invoke-boundary twin of ``reserves_exec_return``, and the most expensive slot to get
    wrong: it is decided at :class:`~jaz.hooks.events.InvokeComplete`, after the agent has
    finished, so an unreserved collision kills the run with its whole cost already spent. Two
    active hooks reserving it are rejected at seed time with
    :class:`~jaz.exceptions.ReturnValueConflictError`. Set it on the class, or per instance in
    ``__init__``.
    """

    @property
    def blackboard_seed(self) -> dict[str, object]:
        """Return this hook's contribution to the blackboard's initial contents.

        The *blackboard* is a small shared scratch space (a key/value store) that every hook
        active for an invoke can read and write, used to pass per-invoke data between hooks and
        the caller; "generation 0" is its initial contents, assembled from these seeds before the
        run starts.

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
        # so two hooks with the same class name from different modules — a user's own hook, or one
        # subclassed inside a test — were distinguishable in a trace and are not now. Deliberate:
        # module-qualifying every hook would make every built-in's string longer for a collision
        # that is rare, and a hook that expects to collide can qualify itself in its own override. Reach for that before concluding a trace is lying to you.
        #
        # A dataclass hook needs nothing: @dataclass generates its own __repr__ on the subclass,
        # which wins over this by MRO and already lists the fields.
        return f"{type(self).__name__}()"

    def setup(self) -> None:  # noqa: B027 - optional override point, intentionally a no-op default
        """Acquire resources for this hook (open files, start providers, ...).

        Default no-op. Override in stateful hooks instead of overriding
        ``__enter__``. This runs on whichever activation path is used:

        - the ``with MyHook():`` context-manager path (via ``__enter__``), and
        - the per-invoke path (a hook passed positionally to ``jaz.invoke``), which
          deliberately does NOT touch the contextvar.

        A given instance is active via exactly ONE path at a time — activating it
        via both at once raises :class:`HookActivationError` — so ``setup()``
        runs once per activation and a stateful hook need not make it idempotent.
        Separating resource lifecycle from contextvar registration is what lets
        per-invoke hooks run cleanup without re-introducing propagation.

        ``setup()`` may run more than once on one instance, at different times: re-activating a
        hook whose previous scope has ended is allowed. See the class docstring for the two
        policies a stateful hook must choose between, and :meth:`forbid_reactivation`.
        """

    #: Set by :meth:`forbid_reactivation` on the first activation; reserved on every hook so the
    #: read is a plain attribute access. A subclass ``@dataclass`` does not collect it as a field —
    #: only the subclass's own annotations are.
    _jaz_activated_before: bool = False

    def forbid_reactivation(self) -> None:
        """Refuse a second activation of this instance. Call from ``setup`` if reuse is unsafe.

        Raises :class:`~jaz.exceptions.HookActivationError` on every activation after the first,
        whether or not the earlier one has ended. An activation whose ``setup`` went on to fail
        still counts as the first: this runs before the rest of ``setup`` and cannot know how it
        turns out, so a retry after a transient failure is refused too.
        """
        # Offered rather than imposed, because the right policy is not uniform — it varies by hook
        # and even by CONFIGURATION of one hook. `TrajectoryRecorder(output_path=None)` must clean
        # rather than refuse: `get_trajectories()` is documented readable after the scope ends, so
        # refusing reuse would be gratuitous. The same class with `output_path` set must refuse:
        # the second run's `teardown` overwrites the first run's file, and no amount of clearing
        # in-memory state un-writes it. `IterationLimit` and `RecursionLimit` hold no per-run state
        # at all and want neither. Only the hook knows which it is, so the base class supplies the
        # mechanism and a uniform message, and leaves the decision where the knowledge is.
        #
        # Deliberately NOT keyed on whether the previous scope is still open. That weaker check is
        # already the dispatcher's, in `__enter__`; this one exists for hooks whose objection is to
        # the second run existing at all.
        if self._jaz_activated_before:
            raise HookActivationError(
                f"{type(self).__name__} instance has already been activated and cannot be "
                "reused; construct a new one for each activation."
            )
        # `object.__setattr__`, because a hook may be a FROZEN dataclass — three built-ins are
        # dataclasses already — and a plain assignment would raise `FrozenInstanceError` on the
        # FIRST activation, naming an attribute the hook's author never wrote.
        object.__setattr__(self, "_jaz_activated_before", True)

    def teardown(self, exc: BaseException | None = None) -> None:  # noqa: B027 - optional override point, intentionally a no-op default
        """Release resources for this hook (flush/close files, end spans, ...).

        Default no-op. Override in stateful hooks instead of overriding
        ``__exit__`` (and do NOT call ``super().__exit__`` — contextvar
        de-registration is handled by the base ``__exit__``). Runs on both the
        ``with`` and per-invoke activation paths.

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
                f"{type(self).__name__} instance is already passed to an enclosing invoke; "
                "entering it via `with` would run setup()/teardown() twice while its "
                "per-invoke lifecycle is still live. Use one scope at a time."
            )
        # The `with` path reaches setup() THROUGH __enter__ (and registers on the contextvar
        # below → propagates). The local_hooks path calls setup() directly (no __enter__, no
        # contextvar). RecursionLimit keys on this distinction to accept `with` and reject
        # local_hooks — do not route local activation through __enter__ without updating that
        # guard (see _activate_local_hooks in invoke.py).
        self.setup()
        new_context = get_current_hooks().with_hook(self)
        _push_activation_token(self, _set_hook_context(new_context))
        return self

    def __exit__(self, _exc_type, exc_val, _exc_tb):
        """Exit hook context - restores previous context, then runs teardown().

        De-registration happens before teardown(); ordering is immaterial since
        resource cleanup does not depend on contextvar membership.
        """
        # A token is recorded only by __enter__, and only in the context that entered. `None` back
        # therefore means one of three things — never entered, already exited, or entered in a
        # DIFFERENT context — and none of them is de-registerable from here, so all three are
        # swallowed. That is a deliberate widening: while the token sat on the instance, a second
        # `__exit__` raised `RuntimeError: Token has already been used once`.
        #
        # The cost is that `teardown()` below still runs on those calls, so a doubled `__exit__`
        # tears down twice for one activation. Left as-is because the alternative — raising, or
        # tracking "has been exited" separately — reintroduces per-instance lifecycle state, which
        # is the thing that was wrong: a hook can be live in two contexts at once, so "already
        # exited" is not a property of the instance. A hook for which a repeated teardown is
        # destructive should refuse reuse outright (`forbid_reactivation`).
        token = _pop_activation_token(self)
        if token is not None:
            _reset_hook_context(token)
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
    - There is no shared singleton. Every field on this class is per-invoke state
      (`_local_hooks`, `_blackboard`, `_own_abort_exceptions`), so an instance shared
      across invokes would leak all of them; a direct caller that emits without an Agent
      constructs its own.

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
      `HookDispatcher` with the nested call's own `local_hooks` (an empty tuple
      if none). The outer invoke's local hooks are not in scope
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
    A dispatcher that was never seeded carries an empty blackboard rather than a
    missing one, so `BlackboardWrite` effects accumulate there exactly as they do
    after `seed_blackboard()` — an unseeded dispatcher is just one whose hooks
    contributed no seed keys. If a future component needs invoke-bound local hooks
    visible to arbitrary call sites within an invoke's call stack, it should reach
    for `agent.dispatcher` (or we should reconsider the second-contextvar design at
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
        # Frozen at construction. Agent always constructs a per-invoke instance, because
        # every field below is per-invoke state — there is no shared dispatcher to reuse.
        self._local_hooks: tuple[Hook, ...] = local_hooks
        # The invoke's current blackboard generation — a read-only mapping, seeded from the
        # active hooks at InvokeEnter via seed_blackboard() and REPLACED (never mutated) at
        # each event's write barrier.
        #
        # Starts EMPTY, not None. An unseeded dispatcher therefore behaves like a seeded one
        # whose hooks contributed no seeds, which is what it is: `seed_blackboard()` adds the
        # hooks' seed keys, and with no hooks it produces exactly this empty board. The
        # `is not None` guards this replaced were doing one real thing — discarding writes on
        # an unseeded dispatcher — and that was wrong, because "unseeded" is reachable in
        # production: `Agent.do_one_repl_iteration()` emits `REPLExecEnter` without seeding, so
        # a `BlackboardWrite` there vanished silently even though later events in the same call
        # would have observed it. Writes now accumulate on every dispatcher, which is the
        # generational rule applied uniformly rather than switched off for one shape.
        self._blackboard: Mapping[str, object] = MappingProxyType({})
        # The invoke's resolved ``jaz.scope``, stamped onto every event (see ``emit``) the way
        # ``blackboard`` and ``hooks`` are; set once, from the ``InvokeEnter`` that opens the
        # invoke. One dispatcher serves one invoke (an ``Agent`` builds its own), so a
        # sub-invoke's scope never reaches its parent's events.
        self._scope: Mapping[str, object] = MappingProxyType({})
        # The exception objects THIS dispatcher's span CMs raised while resolving an
        # ``Abort`` effect, recorded by identity. This is the abort-classification record
        # (see ``_unwind_outcome``): the dispatcher is per-invoke state (Agent constructs
        # one per invoke), so "is this unwinding exception my own invoke's abort?" is a
        # plain identity test against this list — no global registry, no exception
        # wrapping/marking that would perturb the propagating object. Bounded in practice:
        # an invoke resolves at most a handful of aborts before it ends, and the list dies
        # with the dispatcher. A dispatcher shared across invokes would accumulate here
        # across all of them, which is one of the reasons there is no longer a shared
        # singleton to reach for.
        self._own_abort_exceptions: list[BaseException] = []

    def _active_hooks(self) -> list[Hook]:
        """The hooks active for an invoke, in dispatch order (propagating
        (contextvar), then local).

        No dedup is needed across propagating and local: a hook instance can be active
        in at most one live scope. Every overlap shape raises ``HookActivationError`` at
        activation time (see ``Hook.__enter__`` and ``invoke._activate_local_hooks``) —
        ``with``+``with``, ``with``+local, ``[h, h]`` in one call, AND a
        ``with``/local re-activation nested inside an invoke where the instance is
        already a live local (this last caught via the ``HookContext.local_active`` id
        set the contextvar carries for a local's dynamic extent). So the streams are
        disjoint by construction and this stays a plain concatenation; the old identity
        filter here was a symptom of tolerating that overlap.
        """
        # The duplicate-instance rejection covers every route into one live scope (#533 / #540).
        propagating = list(get_current_hooks().hooks)
        local = list(self._local_hooks)
        return propagating + local

    def _reject_declared_write_conflicts(self, active: list[Hook]) -> None:
        """Reject, before the first :class:`InvokeEnter`, two active hooks that statically reserve
        the same sole-authority slot.

        Each of ``SupplyLLMResponse`` / ``AddInputs`` / ``AddToNamespace`` / ``BlackboardWrite`` has a
        single authority per slot — the turn for the response, the key/name for the others — so two
        hooks declaring the same slot is an irresolvable, order-dependent conflict, surfaced here at
        turn 0 instead of mid-run. Keyed on the hooks' static reservations
        (``reserves_llm_response`` / ``reserved_input_names`` / ``reserved_namespace_names`` /
        ``reserved_blackboard_keys``), read off the instance so a hook that reserves in ``__init__``
        counts; a hook that writes a slot WITHOUT reserving it is invisible here and is still caught
        dynamically by the *same* conflict error at composition time. This is deliberately
        *stricter* than the runtime check, not merely earlier: runtime coalesces two writes carrying
        identical values and conflicts only on divergent ones, whereas this rejects any two declarers
        of a slot regardless of value (see the class-level note on ``Hook.reserves_llm_response`` for
        why). Raised errors reuse the runtime conflict types, so ``except REPLInputConflictError``
        (etc.) catches both the early and the late form.
        """
        # SupplyLLMResponse — one sole authority for the whole turn (no key).
        responders = [type(h).__name__ for h in active if h.reserves_llm_response]
        if len(responders) > 1:
            raise LLMResponseConflictError(
                f"{len(responders)} active hooks ({', '.join(responders)}) each reserve the right to "
                "supply the LLM response (reserves_llm_response=True). SupplyLLMResponse is a "
                "sole-authority effect — at most one hook may author the model's turn — so this is "
                "rejected at activation. Activate only one."
            )
        # Key-scoped effects — one authority per key/name.
        self._reject_key_authority_collision(
            active,
            "reserved_input_names",
            REPLInputConflictError,
            "AddInputs input key",
        )
        self._reject_key_authority_collision(
            active,
            "reserved_namespace_names",
            REPLInputConflictError,
            "AddToNamespace namespace name",
        )
        self._reject_key_authority_collision(
            active,
            "reserved_blackboard_keys",
            BlackboardWriteConflictError,
            "BlackboardWrite key",
        )
        self._reject_key_authority_collision(
            active,
            "reserved_llm_response_fields",
            LLMResponseConflictError,
            "ModifyLLMResponse field",
        )
        # Booleans, not key sets: the slot is "the return value at this boundary", of which
        # there is one per boundary rather than one per key. NOT the terminal result — two
        # ``Raise`` results group and two ``Continue`` results concatenate, so only the return
        # value is sole-authority.
        self._reject_sole_authority_flag(
            active,
            "reserves_exec_return",
            ReturnValueConflictError,
            "carry a Return at a REPL-exec boundary",
        )
        self._reject_sole_authority_flag(
            active,
            "reserves_invoke_return",
            ReturnValueConflictError,
            "carry a Return at an invoke boundary",
        )

    @staticmethod
    def _reject_sole_authority_flag(
        active: list["Hook"],
        attr: str,
        error_cls: type[Exception],
        what: str,
    ) -> None:
        """Raise ``error_cls`` if two active hooks set the boolean reservation ``attr``.

        The slot is the whole boundary rather than a key, so the declaration is a flag.
        """
        # `reserves_llm_response` keeps its own inline check because its message names the
        # effect (`SupplyLLMResponse`) rather than an action, which reads better for the one slot
        # whose authority is "authored this turn" rather than "may carry a value".
        claimants = [type(h).__name__ for h in active if getattr(h, attr, False)]
        if len(claimants) > 1:
            raise error_cls(
                f"{len(claimants)} active hooks ({', '.join(claimants)}) each reserve (via "
                f"{attr}) the right to {what}. That is a sole-authority slot — two distinct "
                "return values cannot merge — so it is rejected at activation. Have one hook "
                "own the boundary, or activate only one."
            )

    @staticmethod
    def _reject_key_authority_collision(
        active: list[Hook],
        attr: str,
        error_cls: type[Exception],
        label: str,
    ) -> None:
        """Raise ``error_cls`` if two active hooks reserve (via the attr ``attr``) the same slot.

        Read with ``getattr`` off the *instance*, so a hook that reserves a constructor-supplied key
        in ``__init__`` is covered. Two instances of one class reserving the same key still collide —
        they are two authorities for one slot — which is why the message lists class names."""
        declarers_by_slot: dict[str, list[str]] = {}
        for hook in active:
            for slot in getattr(hook, attr):
                declarers_by_slot.setdefault(slot, []).append(type(hook).__name__)
        for slot, declarers in declarers_by_slot.items():
            if len(declarers) > 1:
                raise error_cls(
                    f"{len(declarers)} active hooks ({', '.join(declarers)}) each reserve (via "
                    f"{attr}) the same {label} {slot!r}. This is a sole-authority "
                    "slot — at most one hook may write it — so it is rejected at activation. "
                    "Have one hook own the slot, or activate only one."
                )

    def seed_blackboard(
        self, caller_metadata: dict[str, object] | None = None
    ) -> Mapping[str, object]:
        """Assemble the invoke's blackboard (generation 0) and bind it to this
        dispatcher, so subsequent ``emit()`` calls stamp it onto every event.

        Called once per invoke, *before* :class:`InvokeEnter` is dispatched. The seed is
        built in plain code (it is NOT an event — that would need a generation to
        read from, a regress) from two own-state-only sources:

        - each active hook's ``blackboard_seed`` payload, and
        - ``caller_metadata`` (the caller-supplied per-invoke data), which wins on
          key conflicts.

        Two hooks seeding one key with different values raises
        :class:`BlackboardSeedConflictError` — an irresolvable, order-dependent ambiguity. Nothing
        else about a seed key is checked: which keys exist on the board is the producers'
        business, and no consumer has to be active for a key to be legitimate.
        """
        active = self._active_hooks()
        # Reject statically-declared write-authority collisions before the first InvokeEnter.
        self._reject_declared_write_conflicts(active)
        seed: dict[str, object] = {}
        for hook in active:
            for key, value in hook.blackboard_seed.items():
                if key in seed and _values_conflict(seed[key], value):
                    raise BlackboardSeedConflictError(
                        f"Conflicting blackboard seed for key {key!r}: "
                        f"{seed[key]!r} != {value!r}. Two hooks seed the same key "
                        "with different values; this cannot be resolved without "
                        "relying on hook order." + _uncomparable_note(seed[key], value)
                    )
                seed[key] = value

        # Caller metadata overlays hook seeds (the caller is authoritative).
        if caller_metadata:
            seed.update(caller_metadata)

        # ``seed`` is local and never mutated again, so the proxy over it is effectively a
        # snapshot: generation 0. Every later generation is a fresh dict (see ``emit``).
        self._blackboard = MappingProxyType(seed)
        return self._blackboard

    def emit(self, event: Event) -> list[Effect]:
        """Emit an event to all active hooks (propagating + local) and collect effects.

        Args:
            event: The event to emit

        Returns:
            List of all effects returned by hooks, EXCLUDING ``BlackboardWrite``
            (consumed here, see below). Order is: propagating hooks in
            registration order, then local hooks in constructor order. Order is
            irrelevant for commutative effects.

        Blackboard stamping + generational write barrier:
            If this dispatcher has a seeded blackboard, its current generation is stamped
            onto the event so every hook reads the same per-invoke board.
            ``BlackboardWrite`` effects are collected during the hook loop and folded only
            *after* the loop, into a **new** generation the dispatcher stamps onto
            subsequent events — so they are invisible to this event's reads and surface to
            the next one. This is what keeps producer/consumer hooks order-independent.
            On an unseeded dispatcher (a hand-built event emitted directly) there is no
            board to advance, so writes are still folded (a conflict raises) and dropped.

        Timestamp stamping:
            ``event.timestamp`` is set here to the emission time, immediately before
            the hook loop — a pre-set value (a hand-built event's construction-time
            default included) is overwritten, so a direct ``emit()`` caller must not
            expect its own stamp to survive.

        Scope stamping:
            ``event.scope`` is set here to the invoke's resolved ``jaz.scope``, taken from
            the ``InvokeEnter`` that opened the invoke — like the blackboard and the
            timestamp, a value the event was constructed with is overwritten, and on a
            dispatcher that has opened no invoke it is the empty mapping.
        """
        # Stamp the invoke's current blackboard generation onto the event (overriding its
        # empty default) so every hook this event reaches reads the same, fixed generation —
        # the barrier below stamps a *new* mapping rather than mutating this one, so this
        # event's board cannot change under its hooks. Event is frozen (hooks must
        # treat it as immutable); object.__setattr__ is the sanctioned in-framework escape hatch
        # for the dispatcher's own stamps — the one place allowed to write an event's fields.
        object.__setattr__(event, "blackboard", self._blackboard)
        object.__setattr__(event, "scope", self._scope)

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
                # hook) can't abort the run. One rule for every hook, whichever channel
                # activated it.
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

        # Generational barrier: fold this event's writes only after every hook has run,
        # minting the next generation that the NEXT event will observe.
        if blackboard_writes:
            # A same-event conflict raises whether or not this dispatcher was seeded: a
            # conflict is a hook-composition bug, and hiding it on the unseeded path would make
            # a direct-``emit()`` test pass where a real invoke fails.
            batch = _resolve_blackboard_writes(blackboard_writes)
            # REPLACE, never update: the mapping handed to this (and every earlier) event is a
            # read-only VIEW, so mutating the dict behind it would retroactively rewrite what
            # those events' hooks read — the same-event race the generational rule forbids.
            # Building a new dict makes each generation an immutable snapshot. Do not
            # "optimise" this into an update of the existing dict.
            self._blackboard = MappingProxyType({**self._blackboard, **batch})

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
            "transforms at the *Complete events, and Abort at every control event and at "
            "LLMQueryExit / REPLExecExit; the other *Exit events accept no effects."
        )

    # Context-specific composition methods

    def _compose_repl_exec(self, effects: list[Effect]) -> REPLExecContext:
        """Compose effects for REPL execution *enter* events.

        Composition rules:
        - Abort (abort): exceptions collected into ``abort_errors``; ``span_repl_exec``
          raises the combined exception (the span closes ``Aborted``).
        - AddToNamespace (bind namespace names): unioned into ``namespace_adds`` via the shared
          conflict-error rule (identical coalesces, divergent raises); ``__builtins__`` refused.
        - DropFromNamespace (drop namespace names): unioned into ``namespace_drops``
          (order-independent, like DropMessages). ``__builtins__`` — the compiler sandbox
          is refused, so a stray drop can't reopen it.
        - InsertCode / DeleteCode (edit the proposed code): accumulated into ``code_inserts`` /
          ``code_deletes`` (no conflict at collection, like AddMessages/DropMessages); folded
          against the enter code in ``span_repl_exec`` via ``_apply_code_edits``.
        - Any other effect raises ``InvalidEffectError`` — notably ``SupplyExecResult``,
          which composes at ``REPLExecSend`` (a supplier must see the committed code).

        ``AddInputs``/``DropInputs`` are **not** collected here — they are ``InvokeEnter``-only:
        inputs are per-invoke setup, applied once before the prompt renders. The per-turn
        namespace counterpart is ``AddToNamespace``/``DropFromNamespace`` here.
        """
        # Refusing a drop of the sandbox builtins keeps #688/#690's allow-list closed. The
        # inputs-at-InvokeEnter-only split is #481: inputs are per-invoke setup, namespace entries are per-turn.
        ctx = REPLExecContext()

        # Apply REPL-specific effects
        for effect in effects:
            match effect:
                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case AddToNamespace(entries=entries):
                    # Bind namespace names. Shared conflict-error rule;
                    # __builtins__ refused (mirrors DropFromNamespace) so an add can't reopen the
                    # sandbox. ``Agent`` applies drops FIRST, so a name also in namespace_drops
                    # resolves to the *add* — and drop-then-add of a bound name is a re-bind.
                    _coalesce_add(
                        ctx.namespace_adds,
                        entries,
                        effect="AddToNamespace",
                        refuse_protected_keys=True,
                    )

                case DropFromNamespace(names=names, allow_missing=allow_missing):
                    # Union of names (order-independent). Never drop the sandbox key:
                    # __builtins__ carries the un-strippable import/attr/builtins allow-lists
                    # (#688/#690), so a DropFromNamespace that names it is refused loudly
                    # (``KeyEditDeniedError``) rather than reopening the sandbox — a silent skip would
                    # hide the hook bug.
                    if "__builtins__" in names:
                        raise KeyEditDeniedError(
                            "DropFromNamespace cannot drop '__builtins__' — it holds the compiler "
                            "sandbox (#688/#690); dropping it would reopen import/attribute/"
                            "builtins policy."
                        )
                    names = set(names)
                    ctx.namespace_drops |= names
                    # Track per-name allow_missing so the REPL's missing-target check can exempt
                    # exactly the names a defensive drop opted in (one opt-in exempts the name).
                    if allow_missing:
                        ctx.namespace_drops_allow_missing |= names

                case InsertCode():
                    # Accumulate (no conflict at collection, like AddMessages) — same-offset
                    # ordering + delete overlaps are resolved by _apply_code_edits in the CM.
                    ctx.code_inserts.append(effect)

                case DeleteCode():
                    ctx.code_deletes.append(effect)

                case _:
                    self._reject_invalid_effect(
                        effect,
                        "REPLExecEnter",
                        "Abort, AddToNamespace, DropFromNamespace, InsertCode, DeleteCode",
                    )

        return ctx

    def _compose_repl_exec_send(self, effects: list[Effect]) -> REPLExecSendContext:
        """Compose effects for REPL execution *send* events — the supply boundary.

        Composition rules:
        - SupplyExecResult (supply a result for the committed code): collected to
          (possibly) short-circuit execution (see _resolve_supply_results).
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
          execution result (see _resolve_modify_results).
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
        """Compose effects for invoke *send* events — the supply boundary.

        The invoke-boundary twin of ``_compose_repl_exec_send``:
        - SupplyInvokeResult (supply the invoke's terminal result for the committed inputs):
          collected to (possibly) short-circuit the whole agent loop (see
          ``_resolve_invoke_supply_results``).
        - Abort: collected; ``span_invoke``'s send emitter raises the combined exception (the
          span closes ``Aborted``), superseding any supply.
        - Any other effect raises ``InvalidEffectError``.
        """
        ctx = InvokeSendContext()

        for effect in effects:
            match effect:
                case SupplyInvokeResult() as supplied:
                    ctx.supply_effects.append(supplied)

                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    self._reject_invalid_effect(
                        effect, "InvokeSend", "SupplyInvokeResult, Abort"
                    )

        return ctx

    def _compose_invoke_complete(self, effects: list[Effect]) -> InvokeCompleteContext:
        """Compose effects for invoke *complete* events.

        The invoke-boundary twin of ``_compose_repl_exec_complete``: collect ``ModifyInvokeResult``
        (transform the terminal result) / ``Abort`` (abort — ``span_invoke`` raises the combined
        exception and the span closes ``Aborted``); any other effect raises ``InvalidEffectError``.
        Unlike the REPL boundary it takes ``ModifyInvokeResult``, not ``ModifyExecResult``: an
        invoke completes only on a terminal result, so its transform effect carries a terminal one
        (``Return`` | ``Raise``) — a ``Continue`` is rejected at the effect's construction rather
        than surfacing here. Resolved against the terminal result in ``span_invoke`` via
        ``_resolve_invoke_modify_results`` — this is what lets ``ReturnType`` / ``ValidateReturn``
        downgrade a wrong-typed / invalid final ``Return`` to a ``Raise`` (the backstop for another
        hook's ``REPLExecComplete`` override reinstating a ``Return``).
        """
        # Complete events are #568's seam for the return-type contract.
        ctx = InvokeCompleteContext()

        for effect in effects:
            match effect:
                case ModifyInvokeResult() as modify:
                    ctx.modify_effects.append(modify)

                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    self._reject_invalid_effect(
                        effect, "InvokeComplete", "ModifyInvokeResult, Abort"
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
        - AddMessages: accumulated; same-slot adds apply in hook order, persistent before transient
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
        from jaz.llm._types import LLMResponse

        ctx = LLMQuerySendContext()

        # Track the winning supply effect so a later one can be compared for
        # equality — the resolution must not depend on arrival order.
        response_effect: SupplyLLMResponse | None = None

        for effect in effects:
            match effect:
                case Abort(error=error):
                    ctx.abort_errors.append(error)
                case SupplyLLMResponse() as supplied:
                    if response_effect is not None and _values_conflict(
                        response_effect, supplied
                    ):
                        raise LLMResponseConflictError(
                            "Multiple conflicting SupplyLLMResponse effects for "
                            "the same LLM query. Supply composition is "
                            "order-independent: distinct supplied responses cannot be "
                            "resolved without relying on hook registration / "
                            "with-nesting order. Ensure only one hook supplies a "
                            "given query, or have them produce identical responses. "
                            "Setting reserves_llm_response = True on each such hook has "
                            "this rejected at activation instead, before the run spends "
                            "anything." + _uncomparable_note(response_effect, supplied)
                        )
                    if response_effect is None:
                        response_effect = supplied
                        ctx.supplied_response = LLMResponse(
                            content=supplied.content,
                            prompt_tokens=supplied.prompt_tokens,
                            completion_tokens=supplied.completion_tokens,
                            cached_tokens=supplied.cached_tokens,
                            cost_usd=supplied.cost_usd,
                            # LLMResponse.extra is a non-optional dict; a supply that omits it
                            # (None) gets the empty-dict default rather than a null.
                            extra=supplied.extra if supplied.extra is not None else {},
                            raw_response=supplied.raw_response,
                            tokens=supplied.tokens,
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
          Return-terminate) is deliberately excluded (YAGNI: no built-in forces a
          Return; governance hooks Abort or push the agent to return via AddMessages).
        - AddInputs: unioned into ``added_inputs`` via the shared conflict-error rule (identical
          coalesces, divergent raises) — order-independent, no silent overwrite.
        - DropInputs: unioned into ``dropped_inputs`` (order-independent). Both are applied to the
          invoke's inputs BEFORE the prompt renders, so an added input shows in the prompt and a
          dropped input vanishes from it. The agent applies drops FIRST, so a key that is both
          added and dropped resolves to the *add* — and a drop-then-add of a caller input is a
          replacement rather than a collision.
        - DisableRecursion: idempotent OR into ``recursion_disabled`` — any emitter suppresses
          this invoke's ``invoke`` tool (primitive binds ``subinvoke=None``).
        - WithholdREPLHistory: idempotent OR into ``repl_history_withheld`` — any emitter withholds
          this invoke's ``__history__`` (the REPL binds no list, and the prompt omits its section).
        - Any other effect raises ``InvalidEffectError`` (this composer used to drop them
          silently — the worst variant of the pre-loud-rejection patchwork).
        """
        # Return-terminate excluded as YAGNI in #481.
        ctx = InvokeContext()

        # Apply invoke-specific effects
        for effect in effects:
            match effect:
                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case AddInputs(inputs=inputs):
                    # Refuse __builtins__ here too (mirrors AddToNamespace): an added *input* is bound
                    # into the same namespace, so binding the sandbox key would reopen it.
                    _coalesce_add(
                        ctx.added_inputs,
                        inputs,
                        effect="AddInputs",
                        refuse_protected_keys=True,
                    )

                case DropInputs(names=names, allow_missing=allow_missing):
                    name_set = set(names)
                    ctx.dropped_inputs |= name_set
                    if allow_missing:
                        ctx.dropped_inputs_allow_missing |= name_set

                case DisableRecursion():
                    ctx.recursion_disabled = True

                case WithholdREPLHistory():
                    ctx.repl_history_withheld = True

                case _:
                    self._reject_invalid_effect(
                        effect,
                        "InvokeEnter",
                        "Abort, AddInputs, DropInputs, DisableRecursion, WithholdREPLHistory",
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
        """Compose effects for LLM query *complete* events — the response-transform boundary.

        The LLM-query twin of ``_compose_repl_exec_complete``:
        - ModifyLLMResponse (transform the response): collected to (possibly) override the
          response (see ``_resolve_llm_modify_results``).
        - Abort (abort): exceptions collected into ``abort_errors``; ``span_llm_query`` raises
          the combined exception once the complete event has been dispatched (the span closes
          ``Aborted``), superseding any transform.
        - Any other effect raises ``InvalidEffectError``.
        """
        # Why abort survives here as a control kind: see `LLMQueryCompleteContext`.
        ctx = LLMQueryCompleteContext()

        for effect in effects:
            match effect:
                case ModifyLLMResponse() as modify:
                    ctx.modify_effects.append(modify)

                case Abort(error=error):
                    ctx.abort_errors.append(error)

                case _:
                    self._reject_invalid_effect(
                        effect, "LLMQueryComplete", "ModifyLLMResponse, Abort"
                    )

        return ctx

    # Send / abnormal-close plumbing shared by the span context managers

    @staticmethod
    def _resolve_code_edits(code: str, ctx: REPLExecContext) -> ResolvedCodeEdits:
        """Resolve an execution's composed code edits into self-contained records.

        Mirrors ``_resolve_message_edits``: offsets are resolved with the same
        ``_resolve_code_offset`` the fold uses, inserts are listed in the fold's application
        order (by offset, hook order within one), and deletes are unioned into the
        disjoint runs the fold removes, one record per run carrying the deleted source — so
        the records agree with the committed code by construction; the fold ran on this same
        ``code`` just before, so resolution here cannot be the first to see an out-of-range
        offset.
        """
        n = len(code)
        by_offset: dict[int, list[InsertCode]] = {}
        for ins in ctx.code_inserts:
            at = _resolve_code_offset(ins.at, n, kind="InsertCode at")
            by_offset.setdefault(at, []).append(ins)
        inserts = [
            CodeInsertRecord(text=ins.text, position=at)
            for at in sorted(by_offset)
            for ins in by_offset[at]
        ]
        # One record per removed run, not per effect: the fold removes the UNION of the
        # resolved ranges, so overlapping or adjacent effects remove one run between them, and
        # per-effect records would report the same source removed twice (three effects over a
        # four-character run would carry nine characters between them). A record is "what was
        # removed" — the criterion ``_resolve_message_edits`` applies when it dedupes drops by
        # resolved position. An empty range removes nothing and so leaves no record. Sorting
        # by (start, end) is what lets one left-to-right pass merge: a range overlaps or abuts
        # the current run iff it starts no later than the run's end.
        ranges = sorted(
            (
                _resolve_code_offset(d.start, n, kind="DeleteCode start"),
                _resolve_code_offset(d.end, n, kind="DeleteCode end"),
            )
            for d in ctx.code_deletes
        )
        runs: list[list[int]] = []
        for start, end in ranges:
            if runs and start <= runs[-1][1]:
                runs[-1][1] = max(runs[-1][1], end)
            else:
                runs.append([start, end])
        deletes = [
            CodeDeleteRecord(start=start, end=end, text=code[start:end])
            for start, end in runs
            if start < end
        ]
        return ResolvedCodeEdits(inserts=inserts, deletes=deletes)

    @staticmethod
    def _resolve_namespace_edits(
        before: "Mapping[str, object]", ctx: REPLExecContext
    ) -> ResolvedNamespaceEdits:
        """Resolve an execution's composed namespace edits into self-contained records.

        ``before`` is ``REPLExecEnter.namespace`` — the namespace as the Enter event
        snapshotted it, before the drops were applied — which is where a drop's unbound value
        is read from (``UNSET`` where the name was absent, or the REPL reports nothing). The
        same snapshot-at-Enter source ``_resolve_message_edits`` reads its drops from.
        """
        adds = [
            NamespaceAddRecord(name=name, value=value)
            for name, value in sorted(ctx.namespace_adds.items())
        ]
        drops = [
            NamespaceDropRecord(name=name, value=before.get(name, UNSET))
            for name in sorted(ctx.namespace_drops)
        ]
        return ResolvedNamespaceEdits(adds=adds, drops=drops)

    @staticmethod
    def _resolve_message_edits(
        snapshot: "Sequence[MessageDict]", edits: MessageEdits
    ) -> ResolvedMessageEdits:
        """Resolve a query's composed edits into self-contained records against the
        enter-time ``snapshot``.

        The resolver lives HERE, not on the record dataclasses: the dispatcher is the one place
        that holds both the snapshot and the raw edits at emission time. Resolving once and
        shipping records (rather than bare indices) is what spares every edit-consuming
        observer its own snapshot-and-index-math machinery (span_event_lifecycle.md,
        Defect 5). Reuses ``_resolve_edit_index`` — the same resolver
        ``_apply_message_edits`` folds with — so the records agree with the buffer by
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
            MessageDropRecord(
                message=snapshot[position], position=position, persistent=persistent
            )
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
        # Stable, so within one slot the records keep the fold's application order.
        adds.sort(key=lambda r: r.position)
        return ResolvedMessageEdits(drops=drops, adds=adds)

    @staticmethod
    def _reject_observation_only_effects(effects: list[Effect], event: Event) -> None:
        """Raise ``InvalidEffectError`` for any effect returned at an observation-only
        emission.

        Used for ``InvokeExit`` (observation-only on both arms — see
        :meth:`_resolve_completed_exit_effects` for why the invoke's terminal Exit is the one
        ``*Exit`` that stays pure) and for any other genuinely observation-only emission. The
        per-turn ``*Exit`` events (``LLMQueryExit`` / ``REPLExecExit``) do NOT use this — they
        accept an ``Abort`` on every arm — so "no effect is valid" below is stated for the
        observation-only callers, not for those. The outcome carried here is already final (the
        transform boundary is ``*Complete``, the supply boundary ``*Send``), so an effect can only
        be a stale or buggy hook — rejected loudly (this upgraded the stage-1 interim
        ignore-with-debug-log guard of span_event_lifecycle.md). On the happy path the raise
        propagates and fails the invoke; on the guarded closes (unwind / ahead of an abort's own
        raise) it is caught and logged by ``_emit_guarded_exit``, because nothing there may replace
        the exception that must win. ``BlackboardWrite`` never reaches this check — ``emit()``
        partitions it out — so observation-time board writes stay legal.
        """
        if effects:
            kinds = ", ".join(type(e).__name__ for e in effects)
            raise InvalidEffectError(
                f"{len(effects)} effect(s) ({kinds}) were returned at "
                f"{type(event).__name__}, which is observation-only: the outcome is "
                "already final, so no effect is valid there. Supplies compose at the "
                "*Send events, transforms at the *Complete events, and Abort at every "
                "control event and at LLMQueryExit / REPLExecExit."
            )

    def _resolve_completed_exit_effects(
        self, effects: list[Effect], event: Event
    ) -> None:
        """Handle effects returned at a *per-turn* ``*Exit`` (``LLMQueryExit`` /
        ``REPLExecExit``) on the ``Completed`` arm — the post-transform, pre-act boundary.

        An ``Abort`` is honored here: the transform boundary (``*Complete``) composes the final
        result *before* any hook can observe it (a hook never sees another hook's same-event
        effect), so a hook that must stop the run on the basis of the FINAL, post-``Modify*``
        outcome has nowhere earlier to do it — this is that point. The ``*Exit`` event has ALREADY
        fired ``Completed`` (the turn genuinely completed), so it is NOT re-emitted; the abort is
        recorded as this invoke's own and raised, and the enclosing invoke span records ``Aborted``
        on the unwind ("turn completed, invoke aborted"). A non-``Abort`` effect is still rejected
        loudly — only ``Abort`` is valid at a ``*Exit``.

        This handles the ``Completed`` (normal-close) arm; the abnormal arms (``Aborted`` /
        ``Failed``) have a twin, :meth:`_emit_guarded_exit_honoring_abort`, so ``Abort`` at a
        per-turn ``*Exit`` is honored on EVERY arm — the uniform contract. The two differ only in
        how they preserve close-then-propagate when an exception is already unwinding: there is none on the
        ``Completed`` arm, so the abort simply raises; on the abnormal arms it is *folded into* the
        in-flight exception via an ``ExceptionGroup`` rather than replacing it. ``InvokeExit``
        deliberately uses neither (it keeps ``_reject_observation_only_effects`` on both arms): an
        invoke aborting at its own terminal Exit would make that Exit's payload diverge from the
        caller's exception, with no enclosing span to record the abort — and ``InvokeComplete``
        already gives full control over the final invoke result. (This is the deliberate, narrow
        reversal of "Exit is pure observation" — traded for letting a hook abort on the
        *post-transform* result/response, e.g. ``BudgetPool`` aborting on the final, re-priced
        cost.)
        """
        # Close-then-propagate is #892's contract. Honouring Abort at Exit reverses #1059's "Exit is
        # pure observation", traded deliberately for letting a hook abort on the terminal turn.
        abort_errors: list[Exception] = []
        invalid: list[Effect] = []
        for effect in effects:
            if isinstance(effect, Abort):
                abort_errors.append(effect.error)
            else:
                invalid.append(effect)
        if invalid:
            kinds = ", ".join(type(e).__name__ for e in invalid)
            raise InvalidEffectError(
                f"{len(invalid)} effect(s) ({kinds}) were returned at "
                f"{type(event).__name__}: only Abort is valid at a per-turn *Exit. "
                "Supplies compose at *Send, transforms at *Complete."
            )
        if abort_errors:
            # Exit already fired Completed; do NOT re-emit — just raise the recorded abort so the
            # span CM's except arm (closed already True) re-raises it and the enclosing invoke
            # records Aborted on the unwind.
            raise self._record_own_abort(abort_errors)

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

    def _emit_guarded_exit_honoring_abort(
        self, exit_event: Event, in_flight: BaseException
    ) -> BaseException | None:
        """Emit a *per-turn* span's ``*Exit`` on the abnormal close (``Aborted`` / ``Failed``),
        honoring an ``Abort`` composed there by *combining* it with the in-flight exception.

        This is the abnormal-arm twin of :meth:`_resolve_completed_exit_effects`: ``Abort`` at a
        per-turn ``*Exit`` is valid on EVERY arm, not just ``Completed``. But the span is already
        unwinding ``in_flight`` here, so the hook's abort cannot simply *replace* it — that would
        drop the terminating exception, violating close-then-propagate. Instead the two are
        folded into one group via :meth:`_record_own_abort` and returned for the CM to raise in
        place of the bare re-raise: ``in_flight`` stays present (so ``is_fatal`` still sees any
        ``FatalError`` in the group and a tree-kill survives), and the group is recorded as this
        invoke's own abort so the enclosing invoke records ``Aborted`` on the unwind. Returns
        ``None`` when no ``Abort`` was composed — the CM then re-raises ``in_flight`` unchanged.

        A non-``Abort`` effect is invalid at any ``*Exit`` but cannot be rejected loudly here
        (raising would replace ``in_flight``); it is dropped with an error log, mirroring
        ``_emit_guarded_exit``. A machinery failure in the emission itself is likewise caught and
        logged — nothing on this path may replace the exception that must win, except an abort
        deliberately folded *into* it.

        Edge: if ``in_flight`` is a *foreign* ``FatalError`` (a child's, unwinding from below), the
        folded group is still recorded as this invoke's own abort, so the enclosing invoke labels
        the close ``Aborted`` rather than ``Failed``. The two events then report DIFFERENT labels on
        this path — this per-turn ``*Exit`` fires ``Failed`` (classified from ``in_flight`` before
        the hook's abort), while the enclosing ``InvokeExit`` reads ``Aborted`` — and that
        divergence is the intended contract (user decision): the events describe different scopes,
        the per-turn ``*Exit`` reporting what the *span* did (it Failed) and the invoke reporting
        what its *control plane* did (a hook Aborted). It is cosmetic for propagation regardless:
        ``is_fatal`` walks the group and drives how far it travels, not the outcome label.
        """
        # Dropping the terminating exception would violate #892's close-then-propagate, so the abort
        # and the in-flight exception are combined rather than one replacing the other.
        try:
            effects = self.emit(exit_event)
        except Exception:
            logger.error(
                "Emitting %s on the span's guarded close failed; the span's own "
                "exception continues to propagate.",
                type(exit_event).__name__,
                exc_info=True,
            )
            return None
        abort_errors = [e.error for e in effects if isinstance(e, Abort)]
        invalid = [e for e in effects if not isinstance(e, Abort)]
        if invalid:
            kinds = ", ".join(type(e).__name__ for e in invalid)
            logger.error(
                "%d effect(s) (%s) were returned at %s on an abnormal close: only Abort is "
                "valid at a per-turn *Exit, and it cannot be rejected loudly while an exception "
                "unwinds — dropping them.",
                len(invalid),
                kinds,
                type(exit_event).__name__,
            )
        if not abort_errors:
            return None
        # Fold the hook's abort(s) INTO the in-flight exception (never replace it): the group keeps
        # in_flight present for is_fatal, and recording it makes the enclosing invoke read Aborted.
        # `in_flight` may itself already be a recorded own-abort (the _abort route passes the
        # composed stage-abort here), so it ends up on _own_abort_exceptions twice — harmless: only
        # the returned group is raised and identity-matched by _unwind_outcome; the stray entry is
        # never re-raised.
        return self._record_own_abort([in_flight, *abort_errors])

    def _record_own_abort(
        self, abort_errors: "Sequence[BaseException]"
    ) -> BaseException:
        """Combine a stage's composed ``Abort`` exceptions and record the result as THIS
        invoke's own abort.

        Also used by :meth:`_emit_guarded_exit_honoring_abort` to fold a hook's ``*Exit``-time
        abort into an already-unwinding exception (the ``in_flight`` exception is passed as the
        first element), which is why the parameter is typed ``BaseException`` rather than
        ``Exception``. Multiple aborts at one stage keep the grouping law the legacy resolution had
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
                span.send(repl.get_namespace(state))  # REPLExecSend
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
             `_resolve_modify_results`). An Abort is exempt from that rule by construction:
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
            # generic abnormal arm below, which stays Failed. The Aborted Exit fired here honors
            # an Abort on that (abnormal) arm just like the unwind path: it is folded into `exc`
            # rather than replacing it — see _emit_guarded_exit_honoring_abort.
            nonlocal closed
            exc = self._record_own_abort(abort_errors)
            closed = True
            combined = self._emit_guarded_exit_honoring_abort(
                _exit_event(self._unwind_outcome(exc)), exc
            )
            raise combined if combined is not None else exc

        try:
            ctx = self._compose_repl_exec(effects)
            if ctx.abort_errors:
                _abort(ctx.abort_errors)
            span = REPLExecSpan(ctx=ctx)
            # Fold enter-time InsertCode / DeleteCode onto the proposed code once, here, so the
            # committed code is a single string both REPLExecSend observers (ValidateREPLCode,
            # suppliers) and the executor read — no Send/exec desync. Reuse the enter code
            # verbatim when nothing edits it (the common path), avoiding a needless rebuild.
            span.committed_code = (
                enter_event.code
                if not (ctx.code_inserts or ctx.code_deletes)
                else _apply_code_edits(
                    enter_event.code, ctx.code_inserts, ctx.code_deletes
                )
            )

            def _emit_send(namespace: Mapping[str, object]) -> None:
                # The committed code, the edit effects and the Enter snapshot are known here;
                # only the resulting namespace comes from the loop (it holds the REPL state;
                # the dispatcher never does) — the InvokeSend split, where the loop passes the
                # committed inputs.
                send_event = REPLExecSend(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    depth=enter_event.depth,
                    iteration=enter_event.iteration,
                    code=span.committed_code,
                    code_edits=self._resolve_code_edits(enter_event.code, ctx),
                    namespace_edits=self._resolve_namespace_edits(
                        enter_event.namespace, ctx
                    ),
                    namespace=namespace,
                )
                send_ctx = self._compose_repl_exec_send(self.emit(send_event))
                # Termination trumps supply: a Send-time Abort raises out of span.send()
                # before any supplied result is even resolved.
                if send_ctx.abort_errors:
                    _abort(send_ctx.abort_errors)
                # Send is the supply boundary: a supply names THE result outright (no
                # original to fold); pure supply — aborts never launder in here anymore.
                assert span is not None
                span.supplied = _resolve_supply_results(
                    supply_effects=send_ctx.supply_effects,
                )

            span._emit_send = _emit_send
            yield span

            # ---- Normal close (the body finished cleanly) ----
            if not span.is_completed:
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
                transformed = _resolve_modify_results(
                    modify_effects=complete_ctx.modify_effects,
                    original=original,
                )
                final = transformed if transformed is not None else original
                span.set_final_exec_result(final)
                # Exit carries the POST-transform result (#906 fixed by the Complete→Exit
                # ordering): observers of the close finally record what the turn actually
                # produced, not composition's input. No transform here — anything other
                # than an `Abort` is rejected loudly. `closed` is set before the emit so a
                # raise from the emission (or the rejection itself) propagates without a
                # double-close.
                exit_event = _exit_event(Completed(final))
                closed = True
                # Per-turn *Exit accepts an Abort on the Completed arm (post-transform, pre-act);
                # a non-Abort effect is still rejected. See _resolve_completed_exit_effects.
                self._resolve_completed_exit_effects(self.emit(exit_event), exit_event)
        except BaseException as exc:
            # Abnormal close (#892): the span opened, so it must close exactly once, here,
            # with the true outcome — then the exception propagates untouched. `closed`
            # means this CM already emitted the exit (an Aborted close whose raise is now
            # passing through, or a failure after the Completed exit fired).
            # _unwind_outcome classifies: Aborted iff the exception is this invoke's own
            # resolved abort (e.g. raised by the LLM-query span's CM and unwinding through
            # the loop); every other unwind is Failed. A hook may still Abort at this Exit on
            # the abnormal arm: it is folded INTO `exc` (never replacing it) and raised in
            # place of the bare re-raise — see _emit_guarded_exit_honoring_abort.
            if not closed:
                combined = self._emit_guarded_exit_honoring_abort(
                    _exit_event(self._unwind_outcome(exc)), exc
                )
                if combined is not None:
                    # `combined` already contains `exc` (folded in as a group member), so suppress
                    # the redundant implicit __context__ chaining `raise` would add.
                    raise combined from None
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
                    response = llm.query(to_wire(shown_messages), **base_config)  # egress
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
                iteration=enter_event.iteration,
                depth=enter_event.depth,
                outcome=outcome,
            )

        def _abort(abort_errors: list[Exception]) -> NoReturn:
            # Composed Abort: close the span Aborted, then raise the carried exception —
            # propagation, not laundering (see span_repl_exec._abort for the rationale). An Abort
            # on the Aborted Exit fired here is folded into `exc` (never replaces it).
            nonlocal closed
            exc = self._record_own_abort(abort_errors)
            closed = True
            combined = self._emit_guarded_exit_honoring_abort(
                _exit_event(self._unwind_outcome(exc)), exc
            )
            raise combined if combined is not None else exc

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
                    iteration=enter_event.iteration,
                    message_edits=self._resolve_message_edits(
                        enter_event.messages, ctx.message_edits
                    ),
                )
                # Send is the supply boundary: a SupplyLLMResponse (e.g. TrajectoryReplay) answers
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
            if not span.is_completed:
                # The forgot-to-complete programming error — loud ERROR tripwire, no exit.
                _log_incomplete_span(
                    "LLMQuery span was not completed. "
                    "You must call span.complete(...) before exiting."
                )
            else:
                # Response control boundary first: LLMQueryComplete fires with the RAW response
                # and its ModifyLLMResponse / Abort effects compose. An abort here closes the span
                # Aborted and raises out of the `with` close — after the query was observed, before
                # the caller reads its content. A ModifyLLMResponse transforms the response (see
                # _resolve_llm_modify_results); an Abort supersedes it (raised before the write).
                complete_event = LLMQueryComplete(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    response=span.get_response(),
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
                # Authoritative field-wise merge written back onto the span: get_response() below
                # (Exit) and the agent's post-close read then see the SAME post-transform object,
                # so content, observers, and cost agree. None = no override → keep the raw object
                # (its identity is what the agent's post-close is-check relies on).
                transformed = _resolve_llm_modify_results(
                    modify_effects=complete_ctx.modify_effects,
                    original=span.get_response(),
                )
                if transformed is not None:
                    span.set_response(transformed)
                # Exit closes the observation record — after Complete, so observers see
                # the final state of the turn. No transform here: anything other than an
                # `Abort` is rejected loudly; `closed` is set before the emit so a raise
                # from the emission propagates without a double-close.
                exit_event = LLMQueryExit(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    iteration=enter_event.iteration,
                    depth=enter_event.depth,
                    outcome=Completed(span.get_response()),
                )
                closed = True
                # Per-turn *Exit accepts an Abort on the Completed arm (post-transform, pre-act) —
                # e.g. BudgetPool aborting on the final, re-priced cost; non-Abort still rejected.
                self._resolve_completed_exit_effects(self.emit(exit_event), exit_event)
        except BaseException as exc:
            # Abnormal close (#892): fire the exit with the classified outcome
            # (_unwind_outcome: Aborted iff this invoke's own resolved abort, else
            # Failed), then let the exception propagate. A hook may Abort at this Exit on the
            # abnormal arm too: it is folded INTO `exc` (never replacing it) and raised in
            # place of the bare re-raise — see _emit_guarded_exit_honoring_abort.
            if not closed:
                combined = self._emit_guarded_exit_honoring_abort(
                    _exit_event(self._unwind_outcome(exc)), exc
                )
                if combined is not None:
                    # `combined` already contains `exc` (folded in as a group member), so suppress
                    # the redundant implicit __context__ chaining `raise` would add.
                    raise combined from None
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
        # Every event of this invoke carries its resolved scope; the Enter event is where the
        # loop hands it over, so it is stashed before that event is itself stamped and emitted.
        self._scope = enter_event.scope
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

            def _emit_send(inputs: "Mapping[str, object]") -> None:
                # The committed inputs come from the loop (it owns the input fold); the edit
                # records come from composition here, with each drop's un-passed value read off
                # the Enter snapshot (``enter_event.inputs`` — the proposal the drop applied
                # to) and UNSET where the name was absent (an allow_missing drop). An add's
                # value is the effect's own: the fold binds it verbatim, so it IS what
                # ``inputs`` holds under that name.
                send_event = InvokeSend(
                    config=enter_event.config,
                    invoke_id=enter_event.invoke_id,
                    depth=enter_event.depth,
                    inputs=inputs,
                    input_edits=ResolvedInputEdits(
                        adds=[
                            InputAddRecord(name=name, value=value)
                            for name, value in sorted(ctx.added_inputs.items())
                        ],
                        drops=[
                            InputDropRecord(
                                name=name, value=enter_event.inputs.get(name, UNSET)
                            )
                            for name in sorted(ctx.dropped_inputs)
                        ],
                    ),
                    recursion_disabled=ctx.recursion_disabled,
                    repl_history_withheld=ctx.repl_history_withheld,
                )
                # Send composes SupplyInvokeResult (short-circuit) + Abort. An Abort supersedes
                # any supply: the combined exception raises out of span.send(), declining the
                # committed input set (the span closes Aborted) before the supply is read.
                # Otherwise the resolved terminal result lands on span.supplied and the agent
                # skips the whole loop — the invoke twin of span_repl_exec's supply short-circuit.
                send_ctx = self._compose_invoke_send(self.emit(send_event))
                if send_ctx.abort_errors:
                    _abort(send_ctx.abort_errors)
                span.supplied = _resolve_invoke_supply_results(
                    supply_effects=send_ctx.supply_effects
                )

            span._emit_send = _emit_send
            yield span

            # ---- Normal close (the body finished cleanly) ----
            if not span.is_completed:
                # The forgot-to-complete programming error — loud ERROR tripwire, no exit.
                _log_incomplete_span(
                    "Invoke span was not completed. "
                    "You must call span.complete(result=...) before exiting."
                )
            else:
                # Transform boundary first: InvokeComplete fires with the RAW terminal
                # result and its ModifyInvokeResult effects compose (#568); an Abort here
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
                # ``original`` is terminal (the loop completes the span only from its Return/Raise
                # arm), and ``ModifyInvokeResult`` carries only terminal results, so the fold is
                # terminal-in ⇒ terminal-out — a ``Continue`` can no longer reach the invoke's
                # outcome through this path (that gap was the reason for the assert below).
                transformed = _resolve_invoke_modify_results(
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
                # ``Return | Raise`` rather than the full ExecResult. This is now guaranteed at
                # construction — a Complete transform can only be a ``ModifyInvokeResult``, which
                # rejects a ``Continue`` — so a Continue can no longer be *introduced* here; the
                # assert is retained as a cheap defense-in-depth backstop against a regression in
                # that chain (and to document the invariant at the one place the payload enters the
                # outcome). It is no longer the sole guard, so its being stripped under ``-O`` no
                # longer risks a silently leaked Continue.
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
