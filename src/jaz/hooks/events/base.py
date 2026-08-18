"""Shared event-payload vocabulary for the span events: the outcome variants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Completed[T]:
    """The span produced its normal result, carried as ``result``.

    One of the three outcome variants every ``*Exit`` event's ``outcome`` field holds
    (``Completed[...] | Aborted | Failed``). The variant *is* the span status: match on the
    type to branch on how the span ended, and read the payload off the matched variant —
    there is no parallel status enum to keep in sync.

    ``T`` is the span's payload type: ``LLMResponse`` for the LLM query,
    :class:`~jaz.repl.types.ExecResult` for the REPL execution, ``Return | Raise`` for the
    invoke.
    """

    # A pure tagged union by user decision: a derived status property (`event.outcome.status`)
    # and payload sugar (e.g. `event.result` forwarding into the Completed arm) were both
    # considered and deliberately deferred until real users demand them — every consumer so
    # far matches on the variant anyway, and sugar would re-create the parallel-field shape
    # this reshape deletes.
    #
    # The two levels are deliberately separate: the STATUS layer (these variants) wraps the
    # RESULT layer (Continue/Return/Raise, LLMResponse), so `case Completed(Raise(exc))` —
    # an agent-produced raise, captured as data — is structurally distinct from
    # `Failed(exc)` — the span's machinery unwinding without a result. That is the design's
    # Raise-as-result-vs-machinery-failure line (span_event_lifecycle.md), expressed in the
    # type rather than in consumer conventions.

    result: T


@dataclass(frozen=True)
class Aborted:
    """The span was deliberately stopped by an :class:`~jaz.hooks.effects.Abort`;
    ``exception`` carries the abort's error.

    See :class:`Completed` for the outcome-union contract.
    """

    # Constructed in exactly one place: ``HookDispatcher._unwind_outcome``, the
    # classification site — a span closes Aborted iff its OWN invoke's hook control plane
    # issued the Abort effect (control-plane association; the full rationale, including
    # the .NET token-match precedent and why exception *category* was rejected as the
    # rule, lives at that site). Every other non-completion is Failed.
    # ``Completed | Aborted | Failed`` mirrors .NET's TaskStatus trio
    # (RanToCompletion | Canceled | Faulted) — readers arrive pre-trained on the
    # deliberate-vs-fault distinction.

    exception: BaseException


@dataclass(frozen=True)
class Failed:
    """The span's machinery unwound on an in-flight exception before a result existed;
    ``exception`` carries the propagating error (which still propagates after the exit
    event fires).

    See :class:`Completed` for the outcome-union contract.
    """

    exception: BaseException
