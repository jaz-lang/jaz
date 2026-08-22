"""The return contract as hooks: ``ReturnType`` and ``ValidateReturn``.

Return typing and validation are hooks, passed positionally so the keyword namespace
belongs entirely to ``**inputs``:

- :class:`ReturnType` carries the expected return type as a self-contained hook — ``invoke``,
  ``agent``, and the REPL know nothing about return types. It drives the ``<return_type>``
  prompt block (a persistent user message at the first ``LLMQueryEnter``) and enforces the type
  with a recoverable ``Continue`` at ``REPLExecComplete`` (``ModifyExecResult``) and a terminal
  ``Raise`` backstop at ``InvokeComplete`` (``ModifyInvokeResult``). Passed as the FIRST positional argument, a typed
  overload keyed on ``ReturnType[T]`` recovers the static ``-> T`` return type. As a context
  manager it propagates into nested / delegated invokes.

- :class:`ValidateReturn` runs return-*value* validation through the effect system, mirroring
  :class:`~BudgetForcing`: at ``REPLExecComplete`` it runs the
  validator on a ``return``'s value and, if rejected, downgrades the terminal ``Return`` to a
  recoverable ``Continue`` (via ``ModifyExecResult``) so the agent sees the error and keeps working —
  until a per-invoke failure cap is reached, at which point it becomes a terminal ``Raise``
  carrying the validator's exception unchanged.

Both hooks check at ``REPLExecComplete`` **and** re-check at ``InvokeComplete``: the second,
terminal check is a backstop against *another* hook's ``REPLExecComplete`` ``ModifyExecResult``
reinstating a ``Return`` past the per-turn check, so a wrong-typed or invalid value can never
escape the invoke to its caller. (The transform boundaries are the ``*Complete`` events —
``*Exit`` is observation-only; each hook keeps a small ``on_invoke_exit`` purely for per-invoke
state cleanup, which must run on abnormal outcomes too.)

The return *type* check (``ReturnType``) is distinct from the return *value* predicate
(``ValidateReturn``): a type mismatch is a contract violation on the shape of the result; a value
rejection is a predicate on an already-well-typed result.

Why these hooks stay result-transforms (``ModifyExecResult`` at ``REPLExecComplete``,
``ModifyInvokeResult`` at ``InvokeComplete``) even at their failure caps — where
``ValidateREPLCode`` escalates to ``Abort``: a return validator only ever reshapes a
*terminal the agent already produced* (its ``Return`` becomes a ``Continue`` refusal, or a
``Raise`` at the cap), so the run was ending either way and a transform expresses that
faithfully. ``ValidateREPLCode``'s cap instead ends a run that would otherwise continue —
the agent submitted more work — which is control-plane authority and belongs to ``Abort``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, overload

from beartype.door import is_bearable
from beartype.roar import BeartypeException

from jaz.exceptions import ReturnTypeError
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import AddMessages, Effect, ModifyExecResult, ModifyInvokeResult
from jaz.hooks.events import (
    InvokeComplete,
    InvokeExit,
    LLMQueryEnter,
    REPLExecComplete,
)
from jaz.repl.types import Continue, Raise, Return
from jaz.string_utils import summarize_exception

if TYPE_CHECKING:
    from typing_extensions import TypeForm


def _format_type_name(t: object) -> str:
    """Human-readable name for a type or type form: ``__name__`` for a concrete type (``int``),
    else ``str()`` (``list[int]``). Local to this module — after #568 moved return-type rendering
    off ``prompts.py`` into ``ReturnType``, this hook is its only consumer.
    """
    return t.__name__ if isinstance(t, type) else str(t)


# A value used only to probe hint *validity* — never type-checked for real. Any object works: a
# usable hint makes ``is_bearable`` return a bool (this object matches or not), while an invalid
# hint raises regardless of the value.
_HINT_PROBE = object()


def _validate_return_type_hint(return_type: TypeForm | None) -> None:
    """Fail fast at invoke *entry* if ``return_type`` is not a hint beartype can check (#891).

    The return-type contract is enforced at ``return`` time by ``is_bearable`` (see
    ``_return_type_error``). If ``return_type`` is not a usable hint — most commonly the builtin
    ``callable`` (a *function*, not ``collections.abc.Callable``) — that ``is_bearable`` call raises
    a ``BeartypeException`` deep in beartype internals, with a trace that points at beartype rather
    than at the caller's annotation, and only *after* the agent has done a full turn's work. This
    check moves the failure to ``ReturnType.__init__`` (which runs when the hook is constructed at
    invoke entry), turning that late, cryptic crash into an immediate, actionable ``TypeError`` that
    names the offending annotation.

    It probes with the **same** ``is_bearable`` mechanism the return check uses, against a sentinel
    value: a usable hint returns a bool without raising; an invalid hint raises ``BeartypeException``
    independent of the value. Using the identical mechanism (rather than a separate validator)
    guarantees we accept *exactly* the hints the return check will — no "passes at entry, rejected at
    return" skew. ``None`` (``ReturnType(None)`` — "no return value") is a valid hint to beartype
    (``NoneType``) and passes. This is deliberately *usability*-only; it does not enforce the
    builtin-only restriction (that is the evals ``RestrictReturnValue``'s job) — ``Callable``, custom classes,
    ``list[int]``, etc. all pass.
    """
    try:
        is_bearable(_HINT_PROBE, return_type)  # type: ignore[arg-type]
    except BeartypeException as e:
        raise TypeError(
            f"Invalid return type {_format_type_name(return_type)}: it is not a usable type hint. "
            f"Pass a type or typing form (e.g. str, list[int], collections.abc.Callable)."
        ) from e


def _return_type_error(
    return_value: object, return_type: TypeForm | None
) -> ReturnTypeError | None:
    """A ``ReturnTypeError`` if ``return_value`` doesn't satisfy ``return_type``, else ``None``.

    Uses beartype's ``is_bearable`` (the check the REPL's removed ``_check_type`` used). This is the
    return-*type* contract, enforced by the ``ReturnType`` hook — distinct from the return-*value*
    predicate a ``ValidateReturn`` runs on an already-well-typed result.
    """
    if is_bearable(return_value, return_type):  # type: ignore[arg-type]
        return None
    return ReturnTypeError(
        f"Expected to return object of type {_format_type_name(return_type)}. "
        f"Got object of type {type(return_value)} instead."
    )


def _terminal_type_error(exc: Exception, return_value: object) -> BaseException:
    """Build the terminal exception for a return value that failed its declared *type*,
    augmenting the message with the offending value. Falls back to ``RuntimeError`` when the
    exception type rejects a str-only constructor. Used by ``ReturnType`` on both terminal paths:
    the ``InvokeComplete`` backstop, and the ``max_failures`` cap in ``on_repl_exec_complete``.

    ``ValidateReturn`` deliberately does NOT go through this: the exception there comes from a
    caller-supplied validator, which owns its own message. This helper augments a
    ``ReturnTypeError`` that this module built itself, so there is nothing of anyone else's to
    preserve.
    """
    msg = f"{exc}\nReturn value: {return_value!r}"
    try:
        return type(exc)(msg)
    except Exception:  # noqa: BLE001 — some exceptions reject a str-only constructor
        # Fall back to a plain RuntimeError rather than force-constructing the original type via
        # ``type(exc).__new__(type(exc))`` + ``Exception.__init__``: bypassing ``__init__`` yields
        # a half-initialized instance whose ``str()`` can silently DROP the message (e.g.
        # UnicodeDecodeError, whose text is built from C-level attrs ``__init__`` never set — it
        # renders empty) or RAISE (a subclass whose ``__str__`` reads attributes ``__init__`` was
        # meant to set). Preserving the exact type isn't worth surfacing a blank/broken error; the
        # message the agent needs is what a guaranteed-constructible RuntimeError carries.
        return RuntimeError(msg)


class ReturnType[ReturnT](Hook):
    """Declare an invoke's return type — the agent is told it, and the value is checked.

    Pass it as the *first* positional argument so the type is also inferred statically::

        x: str = invoke(ReturnType(str), task=...)

    A returned value of the wrong type is shown to the agent as a recoverable error and it keeps
    working; a wrong-typed value never escapes to the caller. By default there is no cap — the
    agent retries until it returns the right type or hits another limit (iteration/budget). Set
    ``max_failures`` to instead abort terminally once the agent has produced that many wrong-typed
    returns and one more — what you want when the type may be impossible for the agent to construct
    (e.g. a custom class it cannot build): the caller then gets a type-specific error rather than a
    generic budget-exhausted one.

    Passed positionally — the usual form, and the only one that types the call statically —
    it applies to that invoke alone. Under ``with`` it also applies to every invoke nested
    inside, holding each of them to the same return type.

    Args:
        return_type: The expected return type. Any type annotation works (``None``, ``str``,
            ``list``, ``dict[str, int]``, ...).
        max_failures: How many wrong-typed returns the agent may make and recover from before the
            next one aborts the invoke — a tolerance, not a total (``0`` aborts on the first wrong
            type; ``2`` allows two recoveries, aborting on the third). The abort raises a
            type-specific error naming the offending value. Only *type* mismatches count, and they
            are counted **cumulatively across the whole invoke** (a valid return ends it; a later
            wrong-typed one still draws on the same budget). ``None`` (the default) never escalates
            — the agent may keep retrying until it returns the right type, so ``IterationLimit`` /
            ``BudgetPool`` is the backstop, not this hook. Must be ``>= 0``. Mirrors
            ``ValidateReturn.max_failures``.

    Raises:
        TypeError: At construction, if ``return_type`` is not a proper type annotation.
        ValueError: At construction, if ``max_failures`` is negative.
    """

    # The whole return-*type* contract lives in this hook, driven entirely by events +
    # effects — ``invoke``, ``agent`` and the REPL know nothing about return types.
    #
    # Mechanics: handles ``LLMQueryEnter`` (injects the ``<return_type>`` block as a
    # *persistent* user message on the first turn only, so it carries forward without being
    # repeated), ``REPLExecComplete`` (the per-turn check — a mismatched ``Return`` is downgraded
    # via ``ModifyExecResult`` to a ``Continue`` carrying a ``ReturnTypeError``) and
    # ``InvokeComplete`` (a terminal backstop that downgrades a still-mismatched final ``Return``
    # to a ``Raise`` via ``ModifyInvokeResult``, message augmented with the offending value, matching
    # ``ValidateReturn``'s terminal path). ``on_invoke_exit`` only releases per-invoke state — cleanup lives at Exit
    # (fires on every outcome) while the transform lives at Complete (fires iff a result exists).
    #
    # The backstop exists because another hook's ``REPLExecComplete`` override can reinstate a
    # ``Return`` past the per-turn check. The common no-override path skips it cheaply: when
    # the invoke's final ``Return`` is the *same object* the per-turn check approved,
    # ``is_bearable`` is not re-run.
    #
    # Validation at construction (rather than at ``return`` time) is deliberate: an unusable hint fails
    # with a clear message naming it instead of crashing deep inside beartype (#891).

    # Overloaded purely for inference: the implementation signature below is unchanged, and at
    # runtime nothing here matters.
    #
    # The culprit is the ``| None`` in that implementation signature. Presented to callers as a
    # single ``return_type: TypeForm[ReturnT] | None``, the parameter is a *union*, and pyright
    # then solves ``ReturnT`` for neither arm — so BOTH the documented ``ReturnType(None)``
    # spelling and every union spelling (``ReturnType(int | None)``, ``Optional[int]``, ...)
    # infer ``ReturnType[Unknown]``. ``Unknown`` is ``Any``-like, which makes the failure silent
    # and worse than no narrowing: ``x: str = jaz.invoke(ReturnType(None), ...)`` type-checks
    # clean, quietly disabling the very checking this hook exists to provide.
    #
    # Splitting the two cases apart gives each a non-union parameter, which pyright can solve:
    # ``None`` binds ``ReturnT`` to ``None``, and a ``TypeForm`` binds it to whatever it denotes.
    # The two overloads therefore reach ``ReturnT = None`` by different routes — ``ReturnType(None)``
    # via the first (bound by ``self: ReturnType[None]``, since the parameter carries no type
    # variable), ``ReturnType(NoneType)`` via the second — so the test asserting they agree covers
    # two code paths, not one twice.
    #
    # An alternative fix was measured and rejected: ``enableExperimentalFeatures = true`` in
    # ``[tool.pyright]`` (which makes pyright honor PEP 747 ``TypeForm``) repairs both symptoms
    # equally well. These overloads are preferred because they don't make the build depend on an
    # experimental, version-volatile type-checker setting, and they work for checkers other than
    # pyright. Adding both would be redundant — and would leave neither pinned by a test, since
    # removing either one alone would still type-check.
    @overload
    def __init__(
        self: ReturnType[None], return_type: None, *, max_failures: int | None = None
    ) -> None: ...

    @overload
    def __init__(
        self, return_type: TypeForm[ReturnT], *, max_failures: int | None = None
    ) -> None: ...

    def __init__(
        self, return_type: TypeForm[ReturnT] | None, *, max_failures: int | None = None
    ) -> None:
        # Validate at construction (invoke entry) so an unusable hint (e.g. the builtin `callable`)
        # fails here with a clear message, not at RETURN inside beartype. See #891.
        _validate_return_type_hint(return_type)
        self.return_type: TypeForm[ReturnT] | None = return_type
        # ``max_failures`` is a count of *tolerated* wrong-typed returns (``count > max_failures``
        # aborts), so ``0`` is valid and meaningful — abort on the first wrong type. Only a
        # negative is junk: it would abort before any failure, behaving identically to ``0``.
        # Reject negatives; ``None`` remains "never abort". (Same guard on the sibling
        # ``ValidateReturn`` / ``ValidateREPLCode``.)
        if max_failures is not None and max_failures < 0:
            raise ValueError(f"max_failures must be >= 0, got {max_failures!r}")
        self.max_failures = max_failures
        # invoke_id -> the Return the per-turn REPLExecComplete check already type-checked OK.
        # The InvokeComplete backstop skips re-running is_bearable when that same object is the
        # invoke's final result (the overwhelmingly-common no-override path), so a side-effecting
        # scenario isn't double-hit. Keyed by invoke_id because a propagating `with ReturnType(T):`
        # reaches nested invokes; released at InvokeExit (which fires on every outcome, so an
        # invoke that FAILED before completing still releases its slot).
        self._checked_ok: dict[str, Return] = {}
        # invoke_id -> count of wrong-typed RETURNs so far, for the ``max_failures`` cap. Mirrors
        # ``ValidateReturn._failures``; released at InvokeExit like ``_checked_ok``.
        self._failures: dict[str, int] = {}

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # First turn only: inject the <return_type> block once, as a persistent user message so it
        # carries forward (mirrors the old opener placement; #568 moved it off the opener so
        # prompts.py no longer names return types).
        # Iterations are 0-based (#719), so the first turn is 0.
        if event.iteration != 0:
            return []
        block = (
            f"<return_type>\n\n{_format_type_name(self.return_type)}\n\n</return_type>"
        )
        return [
            AddMessages(
                messages=[{"role": "user", "content": block}],
                persistent=True,
            )
        ]

    def on_repl_exec_complete(self, event: REPLExecComplete) -> list[Effect]:
        result = event.exec_result
        if isinstance(result, Return):
            error = _return_type_error(result.return_value, self.return_type)
            if error is not None:
                count = self._failures.get(event.invoke_id, 0) + 1
                self._failures[event.invoke_id] = count
                if self.max_failures is not None and count > self.max_failures:
                    # Cap reached: end the invoke terminally instead of looping forever. Uses the
                    # SAME terminal error as the InvokeComplete backstop (``_terminal_type_error``,
                    # value-augmented) so both terminal paths surface identically. Without a cap an
                    # unconstructible return type (a custom class the agent can't build) would
                    # retry until the iteration/budget limit, giving the caller a generic
                    # BudgetExhaustedError rather than this type-specific error. Mirrors
                    # ``ValidateReturn``'s ``max_failures``.
                    return [
                        ModifyExecResult(
                            result=Raise(
                                exception=_terminal_type_error(
                                    error, result.return_value
                                ),
                            )
                        )
                    ]
                # Recoverable: downgrade the mismatched RETURN to a Continue so the agent sees the
                # error and keeps working. This is the branch for ``max_failures=None`` (the
                # default — never escalates) AND for a set cap not yet exceeded — the failure is
                # within tolerance (e.g. the first mismatch at ``max_failures=1``, which tolerates
                # one before the next aborts).
                #
                # ``output`` carries the agent-facing text (the ``Continue`` contract, #875-A),
                # rendered with ``summarize_exception`` — i.e. ``"Type: message"``, not bare
                # ``str(error)``.
                #
                # **Why type + message, and why no traceback.** #928 makes ``output`` the *only*
                # surface the agent sees (the protocol stopped rendering ``exception`` separately),
                # so dropping the type name would lose information the old ``<error>`` block
                # carried. ``repr`` was rejected as the other direction — it renders Python
                # constructor syntax (``ReturnTypeError('...')``) rather than prose. A traceback is
                # not an option at all here: this exception is *constructed* by the check and never
                # raised, so its ``__traceback__`` is ``None`` — there is no stack, and a synthetic
                # one would point at framework internals rather than at the agent's code. Same
                # choice at every hook downgrade site, matching
                # the ``"…failed with {type}: {exc}"`` shape ``ValidateReturn`` already used.
                return [
                    ModifyExecResult(
                        result=Continue(
                            output=summarize_exception(error), exception=error
                        )
                    )
                ]
            # Passed: record identity so the InvokeComplete backstop can skip the redundant
            # re-check.
            self._checked_ok[event.invoke_id] = result
        return []

    def on_invoke_complete(self, event: InvokeComplete) -> list[Effect]:
        result = event.result
        already_ok = self._checked_ok.pop(event.invoke_id, None)
        # Skip when this exact Return already passed the per-turn check (no other hook overrode
        # it). A *different* object means another hook's REPLExecComplete ModifyExecResult
        # reinstated a Return past that check — re-verify it terminally, which is the backstop's
        # whole purpose.
        if isinstance(result, Return) and result is not already_ok:
            error = _return_type_error(result.return_value, self.return_type)
            if error is not None:
                return [
                    ModifyInvokeResult(
                        result=Raise(
                            exception=_terminal_type_error(error, result.return_value),
                        )
                    )
                ]
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        # Leak-prevention cleanup ONLY — the transform logic lives at on_invoke_complete.
        # Exit fires on every outcome (the outcome union, #892), Complete only when a
        # terminal result exists, so an invoke that unwound abnormally releases its
        # ``_checked_ok`` slot here rather than pinning it for the hook's lifetime. On the
        # completed path Complete already popped it and this is a no-op.
        self._checked_ok.pop(event.invoke_id, None)
        self._failures.pop(event.invoke_id, None)
        return []


class ValidateReturn[ReturnT](Hook):
    """Validate an invoke's ``return`` value, rejecting it as a recoverable error.

    A rejected value does not end the invoke: the agent is shown the validator's exception and
    keeps going, so it can correct itself and return again. The agent may be rejected
    ``max_failures`` times and recover; the *next* rejection ends the invoke by raising, so
    ``max_failures=0`` makes the very first rejection terminal.

    Passed positionally it validates that one invoke. Under ``with`` the *same* validator is
    applied to every invoke in scope, nested ones included — usually not what you want, since
    a sub-invoke returns to its caller, not to you, and is rarely subject to the same rule.
    Either way each invoke counts its own rejections against ``max_failures`` rather than
    drawing down its parent's.

    Args:
        validator: Called with the return value; it should raise iff the value is invalid. The
            exception it raises is shown to the agent as a recoverable error, except for the one
            past the cap, which propagates out of the invoke unchanged — include the offending
            value in the message if the caller needs to see it.
        max_failures: How many rejected returns the agent may recover from before the next
            rejection ends the invoke — a tolerance, not a total (``0`` raises on the first
            rejection; ``2`` allows two recoveries, raising on the third). Counted **cumulatively
            across the whole invoke** (a valid return ends it; a later rejected one still draws on
            the same budget). ``None`` (the default) never escalates — the agent may keep retrying
            until it satisfies the validator, so ``IterationLimit`` / ``BudgetPool`` is the
            backstop, not this hook. Must be ``>= 0``.

    Raises:
        ValueError: At construction, if ``max_failures`` is negative.
    """

    # A validator that needs prior REPL iterations reads ``__history__`` from the agent's
    # REPL namespace — the framework does not inject it. Out of the docstring: it reads as a
    # note about the *validator's* signature, which is the one thing that paragraph is not
    # about.
    #
    # Mechanics: handles ``REPLExecComplete`` (per-turn check on the produced ``Return``),
    # ``InvokeComplete`` (the terminal backstop) and ``InvokeExit`` (releasing the per-invoke
    # state — cleanup lives at Exit, which fires on every outcome, while the transforms live
    # at the Complete boundaries); emits ``ModifyExecResult`` at ``REPLExecComplete`` and
    # ``ModifyInvokeResult`` at ``InvokeComplete``.
    # Behavior mirrors ``BudgetForcing``: a rejected value is downgraded to a recoverable
    # ``Continue`` whose output leads with a ``Return value validation failed with <Type>:
    # <message>`` header and carries the raw exception. The core loop records the FINAL,
    # post-override result in ``__history__``, so the rejected RETURN is logged as the
    # error rather than as a clean finish. At the cap it emits a terminal ``Raise`` carrying the
    # validator's exception object itself. Counts are keyed by ``invoke_id`` and released at
    # ``InvokeExit`` (every outcome).
    #
    # That exception is passed through **unmodified**, deliberately. It used to be rebuilt as
    # ``type(exc)(f"{exc}\nReturn value: {value!r}")`` (inherited from the REPL's old
    # ``_handle_validation_failure``), which cost more than it bought: the caller caught a
    # *different object* than the validator raised, any custom attributes were lost, an
    # exception type whose ``__init__`` rejects a lone string silently became a
    # ``RuntimeError``, and a validator that already named the value said it twice. Rendering
    # the value is the validator's call — it is the only party that knows whether the value is
    # small enough to print, or already implied by its own message.

    def __init__(
        self,
        validator: Callable[[ReturnT], None],
        *,
        max_failures: int | None = None,
    ) -> None:
        self.validator = validator
        # ``max_failures`` is a count of *tolerated* rejections (``count > max_failures`` raises),
        # so ``0`` is valid and meaningful — raise on the first rejection. Only a negative is junk:
        # it would raise before any rejection, behaving identically to ``0``. Reject negatives;
        # ``None`` remains "never escalate". (Same guard on ``ReturnType`` / ``ValidateREPLCode``.)
        if max_failures is not None and max_failures < 0:
            raise ValueError(f"max_failures must be >= 0, got {max_failures!r}")
        self.max_failures = max_failures
        # Per-invoke failure counts (keyed by invoke_id); see BudgetForcing for why a
        # single shared counter would be wrong across nested invokes.
        self._failures: dict[str, int] = {}
        # invoke_id -> the Return whose value the per-turn check already validated OK, so the
        # InvokeComplete backstop can skip re-running the (possibly side-effecting) validator when
        # that same object is the invoke's final result. Released at InvokeComplete (or, on an
        # abnormal outcome where Complete never fires, by the InvokeExit cleanup).
        self._validated_ok: dict[str, Return] = {}

    def on_invoke_complete(self, event: InvokeComplete) -> list[Effect]:
        # Backstop: re-validate the invoke's FINAL Return. Another hook's REPLExecComplete
        # override may have reinstated a Return past our per-turn check, so this is the last
        # chance to reject an invalid value — terminally (Raise), since the invoke is already
        # finishing (mirrors the terminal branch of on_repl_exec_complete). InvokeComplete is
        # the invoke's transform boundary (#568 / span_event_lifecycle.md).
        effects: list[Effect] = []
        result = event.result
        already_ok = self._validated_ok.pop(event.invoke_id, None)
        # Skip when this exact Return already passed the per-turn validator (no override) — a pure
        # predicate would just repeat, and a side-effecting one shouldn't fire twice. A *different*
        # object slipped past that check via another hook's ModifyExecResult — re-validate it.
        if isinstance(result, Return) and result is not already_ok:
            try:
                self.validator(result.return_value)
            except Exception as exc:  # noqa: BLE001 — any validator exception is a rejection
                effects = [ModifyInvokeResult(result=Raise(exception=exc))]
        return effects

    def on_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        # Leak-prevention cleanup ONLY — the transform/backstop logic lives at the Complete
        # boundaries. Exit fires on every outcome, so the per-invoke maps are released even
        # when the invoke unwound abnormally and Complete never fired.
        self._failures.pop(event.invoke_id, None)
        self._validated_ok.pop(event.invoke_id, None)
        return []

    def on_repl_exec_complete(self, event: REPLExecComplete) -> list[Effect]:
        result = event.exec_result
        # Only a RETURN carries a value to validate; regular code / RAISE / parse-failure
        # turns produce no return value, so there is nothing to check.
        if not isinstance(result, Return):
            return []
        try:
            self.validator(result.return_value)
        except Exception as exc:  # noqa: BLE001 — any validator exception is a rejection
            count = self._failures.get(event.invoke_id, 0) + 1
            self._failures[event.invoke_id] = count
            if self.max_failures is not None and count > self.max_failures:
                # Terminal: transform the return into a Raise carrying the validator's exception
                # *unchanged*, so the caller catches exactly what the validator raised.
                # ``ModifyExecResult`` with a Raise wins the resolve fold (Raise > Return), ending
                # the invoke.
                return [ModifyExecResult(result=Raise(exception=exc))]
            # Recoverable: downgrade the RETURN to a Continue carrying the validator's error,
            # so the agent sees it and keeps working (the fold prepends the original output).
            # The output leads with an explicit header naming the exception type + message, so
            # the agent gets a self-describing signal rather than relying solely on the core
            # loop's rendering of the attached exception.
            # TODO(#880): render this as a traceback underlining the offending RETURN statement.
            return [
                ModifyExecResult(
                    result=Continue(
                        # ``summarize_exception`` rather than ``{type}: {exc}``: for an
                        # exception *group* (a validator aggregating several complaints)
                        # ``str()`` renders only a sub-exception count, dropping every child's
                        # text; the summary expands them.
                        output=(
                            "Return value validation failed with "
                            f"{summarize_exception(exc)}"
                        ),
                        exception=exc,
                    )
                )
            ]
        # Passed: record identity so the InvokeComplete backstop skips re-running the validator.
        self._validated_ok[event.invoke_id] = result
        return []
