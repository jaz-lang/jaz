"""JAZ exception taxonomy — the errors raised during agent execution.

Every error JAZ raises derives from :class:`JazException`, which derives from ``Exception``.

Exceptions raised by hooks are unstable API: the hook system is experimental, so the errors it
raises may be renamed, re-rooted, or removed in a future release.
"""

# There are no exceptions to the rule above, so ``try: jaz.invoke(...) except Exception:``
# catches everything JAZ can surface — including :class:`FatalError`, which used to escape it
# (see that class's docstring for why the ``BaseException`` rooting was reversed). Catch
# ``jaz.exceptions.JazException`` for a framework-wide catch-all, or a specific subclass for a
# specific failure mode (e.g. a hook enforcing a budget raises ``BudgetExhaustedError``).
#
# Naming rules, so the next audit does not re-litigate them (#806):
#
# * ``Jaz`` prefixes only the taxonomy root, :class:`JazException` (plus the private
#   :class:`_JazInternalError`). Specific errors are unprefixed and descriptive; one that would
#   clash with a builtin takes a domain word instead of the prefix —
#   :class:`SandboxPermissionError`, not ``JazPermissionError``.
# * Conflict errors are ``<subject>ConflictError`` (:class:`REPLInputConflictError`,
#   :class:`LLMResponseConflictError`, …) — one shape for one contract.
# * The :class:`LLMError` subtree keeps LiteLLM's exception vocabulary verbatim; see that class's
#   docstring for the provenance and for why an ``LLM`` prefix was considered and dropped.
#
# Decision — NO ``REPLError`` grouping base (#1005). The REPL-layer errors — e.g.
# ``SandboxPermissionError``, ``REPLInputConflictError``, ``REPLTimeoutPragmaError``,
# ``MissingDropTargetError`` here; ``REPLTimeoutError`` / ``REPLMemoryError`` in ``repl/_exec_guards.py``;
# ``CookieCollisionError`` in ``repl/_cookies.py`` — are deliberately NOT unified under a common base,
# unlike the :class:`LLMError` subtree. ``LLMError`` earns its base because its members share one payload
# and one coherent handling story; these share neither, so a caller writing ``except REPLError`` would
# have no single failure mode to act on. ``except JazException`` is already the framework-wide catch-all,
# and three of these errors are not ``JazException`` at all (they subclass plain ``Exception`` /
# ``RuntimeError`` and live in other modules), so introducing the base would force MRO changes and widen
# what ``except JazException`` catches — cost, for no caller benefit.

# Deprecated aliases for every pre-rename spelling live next to their replacements. They are
# plain assignments, kept out of ``__all__``, and removable at the next major bump. Maintainer
# detail rather than shipped text: a user reading the reference sees each alias documented at
# its own definition, and the removal schedule is a decision about this module, not something
# the reader of a caught exception acts on.

__all__ = [
    "JazException",
    "FatalError",
    "NonPublicAPIWarning",
    # Public so callers can filter it: `jaz.describe` on bare built-ins retains them,
    # and this is how that cost announces itself. Silencing it must be possible by name.
    "DescriptionRetentionWarning",
    # Same reason, different failure: `jaz.describe` on a process-shared value (None, bools,
    # small ints) mislabels every other use of it. Filterable separately because the remedy
    # differs — this one is never fixed by describing fewer values.
    "DescriptionSharedValueWarning",
    "UnknownRequestParamWarning",
    # Execution / runtime failures
    "ReturnTypeError",
    "SandboxPermissionError",
    "BudgetExhaustedError",
    "IterationLimitExhaustedError",
    "BudgetPoolExhaustedError",
    "RecursionLimitError",
    "BudgetForcingError",
    "REPLInputConflictError",
    "REPLTimeoutPragmaError",
    "SandboxKeyError",
    # Rollout export failures (jaz.hooks.builtin.rollout)
    "NonMonotoneRolloutError",
    # API-misuse failures (hooks / config composition)
    "InvalidEffectError",
    "LLMResponseConflictError",
    "ReturnValueConflictError",
    "BlackboardSeedError",
    "BlackboardWriteConflictError",
    "HookActivationError",
    "ModelPricingUnavailableError",
    "ConfigOverrideActivationError",
    "MessageEditIndexError",
    "MissingDropTargetError",
    # LLM provider / transport errors (subtree under LLMError; all JazException)
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "NotFoundError",
    "PermissionDeniedError",
    "UnsupportedParamsError",
    "APIError",
]


class JazException(Exception):
    """Base exception class for JAZ-related errors."""


class FatalError(JazException):
    """Class of exceptions that cause :func:`invoke` to error out instead of being shown to the
    agent as recoverable feedback; exempt from being caught by agent-written ``except`` clauses.

    An ordinary exception in REPL code is caught and shown to the agent as recoverable feedback
    (a :class:`Continue`) so it can retry. A ``FatalError`` instead ends the REPL session and
    causes the :func:`invoke` to error out with that error. If this
    invoke is a subinvoke of a parent invoke and the ``FatalError`` occurs in the REPL of that
    parent :func:`invoke`, then that parent :func:`invoke` also errors out. Thus, usually, the
    entire :func:`invoke` tree fails and the top-level :func:`invoke` raises the
    ``FatalError``.

    Raise ``FatalError`` when a fatal error occurs that the invoke tree cannot recover from. For
    example, write ``raise FatalError()`` inside a tool definition in paths that should take
    down the entire invoke tree. Errors from third-party libraries that you'd like to convert to
    a fatal error should be caught and re-raised as ``FatalError``. A hook that would like to
    take down the entire invoke tree should emit an ``Abort(error=FatalError(...))`` effect.
    (Emitting ``Abort(error=exc)`` where ``exc`` is not a ``FatalError`` aborts the in-progress
    :func:`invoke` but does not propagate to the parent :func:`invoke`.)

    Who can decline a fatal error, since that is the whole point of the class:

    ====================  ==========================================================================
    **Agent REPL code**   **No** — a ``FatalError`` is exempt from being caught by agent-written
                          ``except`` and ``except*`` clauses. Two known exceptions, both under
                          "Known limitations".
    **User code**         Yes, by catching it — a tool or the caller is host code, and one that
                          catches its own fatal error has decided not to fail.
    **A hook**            **No** — a fatal exception propagates during unwind, and no transform
                          boundary fires on the unwind path, so there is no effect a hook could
                          return to convert it back into normal control flow.
    ====================  ==========================================================================

    The purpose of this class is *routing*, i.e. who the error is addressed to. An ordinary error
    is addressed to the agent; a fatal error is addressed to the **caller**: "this failure is not
    something the agent can fix by retrying."

    Known limitations
    -----------------
    * **A suppressing context manager still swallows it**, in the default configuration and
      with no imports::

          class S:
              def __enter__(self): return self
              def __exit__(self, *a): return True   # truthy -> suppresses anything
          with S():
              tool_that_raises_fatal()              # fatal error swallowed; the run continues

      A known limitation, not a boundary.
    * **A group-wrapped fatal error is catchable.** An ``ExceptionGroup`` containing a
      ``FatalError`` is really an ``ExceptionGroup``, so both ``except Exception`` and
      ``except ExceptionGroup`` catch it. Such a group *is* still exempt from ``except*``, which
      surfaces the ``FatalError`` leaf. A group that goes uncaught still takes down the invoke
      tree.
    """

    # Renamed from the abort-stem error name (span_event_lifecycle.md naming, revised by user
    # decision): the effect/category stem-conflation was resolved by moving THIS name, not the
    # effect's — DOM ``AbortController.abort(reason)`` is the ``Abort`` effect's exact precedent
    # (an operation-scoped stop with a reason payload, containable by the parent), so the
    # mechanism keeps the word "abort", while "fatal" is what this module's own ``is_fatal``
    # predicate already called this category — the rename makes the pairing tautological. The
    # deeper reason the old name had to go: it baked an INTENT claim into the type name
    # ("someone chose to abort") — but the channel carries unknowable intent (governance stops
    # and breakage reports alike), which is what caused real conflation. "FatalError" claims
    # only a PROPERTY: the error is fatal, hence it propagates past every agent. Property-naming
    # keeps the category intent-agnostic, which the span-outcome classification requires
    # (outcome is decided by control-plane association, never by reading intent off an
    # exception type). No back-compat alias — pre-release.
    #
    # EXECUTIVE CALL (user, 2026-07-29): **rooted at ``JazException`` (an ``Exception``)**, so
    # every error JAZ raises is an ``Exception`` and one rule describes the taxonomy. This
    # reverses the earlier ``BaseException`` rooting (#559/#561), for two reasons:
    #
    # 1. The caller is the audience, and ``BaseException`` rooting hid it from them. The natural
    #    way to handle a failed run is ``try: jaz.invoke(...) except Exception:``, and under the
    #    old rooting the one error explicitly routed *to* the caller was the one that handler
    #    missed — the exact inversion of this class's purpose. (Contrast ``KeyboardInterrupt``
    #    escaping ``except Exception``, which is desirable precisely because it is *not*
    #    addressed to you.)
    # 2. The guarantee that rooting bought was not real. ``BaseException`` rooting was chosen so
    #    no routine handler could swallow a fatal error. It does stop ``except Exception`` — but it is
    #    defeated in the *default* configuration by four lines of agent code (the suppressing
    #    context manager under "Known limitations"). So the property being protected was a strong
    #    default, not a boundary, and it is now upheld by mechanism rather than class identity.
    #
    # Mechanism behind the two known limitations, kept out of the docstring as implementation
    # detail. Suppressing context manager: ``__exit__`` is invoked at the C level and a ``with``
    # block has no ``except`` clause for the cookie transform to guard, so neither the transform
    # nor any base class can stop it. This defeated the old ``BaseException`` rooting identically,
    # so it is not a regression, but it does mean an adversarial agent can decline a fatal error.
    # Group-wrapped fatal error: the cookie guard matches ``FatalError``, and a group *containing* one
    # is not an ``FatalError``, so the guard does not fire. Reachable whenever anything in the
    # call path bundles exceptions — a tool using ``asyncio.TaskGroup``/``gather``, or
    # ``hooks.effects._combine_exceptions``. Guarding it would mean intercepting groups and
    # unwrapping, which a plain injected handler cannot do without also stealing ordinary
    # ``ExceptionGroup``s from the agent's own handlers. #977 tracks the fatality *latch* on the
    # invoke context, which closes both routes at once.


class ReturnTypeError(TypeError, JazException):
    """Raised when the agent's returned value doesn't satisfy the declared return type.

    Produced by the return-validation hooks (``jaz.hooks.ReturnType`` / :class:`ValidateReturn`, see
    ``hooks/builtin/return_hooks.py``): a mismatched ``return`` is downgraded to a :class:`Continue`
    carrying this error, so the agent is told what it got wrong and can retry.
    """


#: Deprecated alias for the pre-rename spelling — kept for the same reason as the other aliases
#: in this module: the old name was in ``__all__``, i.e. documented surface, so dropping it
#: outright would give out-of-tree ``except jaz.exceptions.CommandTypeError`` a bare
#: ``NameError``. An alias, not a subclass, so ``except`` keeps matching exactly. Deliberately
#: absent from ``__all__``.
CommandTypeError = ReturnTypeError


class SandboxPermissionError(JazException):
    """Raised when the REPL filesystem sandbox denies a file operation."""

    # Raised only by ``repl._secure_methods.secure_open`` — when agent code opens a path outside
    # the configured readable/writable roots, or opens a readable-only path for writing. The
    # message names the offending path and the permission it lacked.
    #
    # Named for the sandbox, not the framework: the earlier spelling ``JazPermissionError``
    # carried the ``Jaz`` prefix purely to dodge the builtin ``PermissionError``, which made it
    # the only prefixed *leaf* in the taxonomy. ``Sandbox`` disambiguates just as well while
    # saying what actually denied the operation, and pairs it with ``SandboxKeyError``.


#: Deprecated alias for the pre-rename spelling — see :data:`BudgetForcingException`
#: for the full rationale. An alias, not a subclass, so ``except`` keeps matching exactly.
#: Deliberately absent from ``__all__``.
JazPermissionError = SandboxPermissionError


class BudgetExhaustedError(JazException):
    """Raised when a budget-enforcing hook stops the run because its budget is spent.

    The base of two distinct causes — catch this to handle either, or the subclass to tell
    them apart: :class:`IterationLimitExhaustedError` (the turn cap) and
    :class:`BudgetPoolExhaustedError` (a cost/calls pool). The distinction matters to a caller
    that can *do* something different about each — e.g. offer more turns for the former but not
    the latter, which more turns cannot refill.
    """


class IterationLimitExhaustedError(BudgetExhaustedError):
    """Raised by :class:`jaz.hooks.IterationLimit` when the per-invoke turn cap is reached.

    A subclass of :class:`BudgetExhaustedError`, so ``except BudgetExhaustedError`` still
    matches; catch this specifically to react to the turn cap alone (granting more turns can
    let the run finish, unlike a spent cost/calls pool).
    """


class BudgetPoolExhaustedError(BudgetExhaustedError):
    """Raised by :class:`jaz.hooks.BudgetPool` when its pooled cost or calls ceiling is reached.

    A subclass of :class:`BudgetExhaustedError`, so ``except BudgetExhaustedError`` still
    matches; catch this specifically to distinguish a spent cost/calls pool from the turn cap
    (:class:`IterationLimitExhaustedError`) — more turns cannot refill a pool.
    """


class NonMonotoneRolloutError(JazException):
    """Raised by ``Rollout.to_flat()`` when the trajectory's context was not append-only.

    The model saw genuinely different prefixes on different turns, so no single flat
    masked token sequence represents the run. Routine causes: a reasoning model under
    ``SGLangLLM``'s hidden-trace interface (each turn's ``<think>…</think>`` is stripped
    from the next turn's context, so *every* multi-turn reasoning rollout is non-monotone
    by design), a transient hook nudge (e.g. :class:`jaz.hooks.IterationLimit`'s warning),
    or a history compaction. Use ``to_pieces()`` (flat samples per monotone span) or
    ``to_turn_samples()`` (fully general, per-turn) instead.
    """


class RecursionLimitError(JazException):
    """Raised when a ``jaz.invoke`` exceeds an :func:`invoke` recursion limit set by
    :class:`jaz.hooks.RecursionLimit`.
    """

    # A recursion cap is enforced by removing the recursive tool at the cap leaf, so an agent
    # normally cannot exceed it. The *public* ``jaz.invoke`` (reached from a human-written tool, or
    # a deliberately injected real ``jaz``) has no such lever, so a child that enters past the cap
    # anyway raises this instead — the over-cap backstop.
    #
    # The cap is imposed by the opt-in ``RecursionLimit`` hook, whose ``Abort`` carries this error
    # when a child enters past the cap. The framework itself imposes no recursion cap, so this is
    # only raised when a ``RecursionLimit`` is installed.


#: Deprecated alias for the pre-rename spelling (#806 question 2: align the exception with
#: the ``RecursionLimit`` hook and ``DisableRecursion`` effect that raise it). Kept for the
#: same reason as :data:`BudgetForcingException` — the old name was in ``__all__``, i.e.
#: documented surface, so `except jaz.exceptions.DepthLimitError` out-of-tree would otherwise
#: become a bare ``NameError``. An alias rather than a subclass, so existing `except` clauses
#: keep matching the raised type exactly. Deliberately absent from ``__all__``: reachable,
#: unsupported, and removable at the next major bump.
DepthLimitError = RecursionLimitError


class _JazInternalError(FatalError):
    """Internal framework error — **always fatal**.

    Subclasses :class:`FatalError` so a framework bug / invariant violation fails
    fast (propagates out of the invoke) rather than being hidden from the caller as
    agent feedback. Framework code that must end the run raises this (not a bare ``assert``,
    which is deliberately non-fatal — see :func:`is_fatal`)."""


class BudgetForcingError(JazException):
    """Raised when the :class:`jaz.hooks.BudgetForcing` hook blocks an early
    ``return``/``raise`` finish.
    """


#: Deprecated alias for the pre-rename spelling. Kept because the old name was in
#: ``jaz.exceptions.__all__`` — i.e. it was *documented* surface, not merely reachable —
#: so `except jaz.exceptions.BudgetForcingException` in out-of-tree code would otherwise
#: become a bare ``NameError`` with no hint of the replacement. It is an alias, not a
#: subclass, so existing `except` clauses keep matching the raised type exactly. Deliberately
#: absent from ``__all__``: reachable, unsupported, and removable at the next major bump.
BudgetForcingException = BudgetForcingError


class REPLInputConflictError(ValueError, JazException):
    """Raised when an added name conflicts with a binding that already exists.

    Two effects adding the same name with *different* values, an :class:`AddInputs` key that
    collides with a caller's own input or scope entry, or an :class:`AddVariables` name already
    bound in the REPL namespace. Identical values coalesce instead of raising."""

    # ``ValueError`` and not ``KeyError``: nothing failed to look up here — the name resolves
    # fine, it is the *request* that is inappropriate, which is what ``ValueError`` means
    # ("an argument of the right type but an inappropriate value"). ``KeyError`` is for a key
    # absent from a mapping, the opposite of this failure. No ``except ValueError`` sits on
    # either raise path (``_compose_add_variables`` in the dispatcher, ``Agent``'s input
    # resolution, ``PythonREPL.add_variables``), so the wider base cannot swallow it silently.


#: Deprecated alias for the pre-rename spelling — see :data:`BudgetForcingException`
#: for the full rationale (old name was in ``__all__``, so dropping it outright would be a
#: bare ``NameError``). An alias, not a subclass, so ``except`` keeps matching exactly.
#: Deliberately absent from ``__all__``.
DuplicateREPLInputError = REPLInputConflictError


class REPLTimeoutPragmaError(ValueError, JazException):
    """Raised when the code's leading ``# timeout:`` comment names no positive number of seconds.

    An agent sets a per-execution wall-clock bound with a ``# timeout: <seconds>`` comment among
    the lines opening its REPL code. The value must parse as a positive, finite number, so
    ``# timeout: 120s`` and ``# timeout: 0`` are rejected; a trailing ``#`` comment after the
    number is allowed. The offending code is not executed — the agent sees the error as ordinary
    REPL feedback and can correct it on the next iteration."""

    # ``ValueError`` (following :class:`REPLInputConflictError`): nothing failed to resolve — the
    # line is a well-formed comment, and it is the *value* that is inappropriate.
    #
    # Defined here rather than beside the parser in ``jaz.repl.python_repl`` because this module is
    # the taxonomy: it opens by promising that every error JAZ raises derives from ``JazException``,
    # and ``__all__`` is what the generated exceptions reference publishes. A ``JazException``
    # subclass living outside it would be uncatchable by import from the documented module. (The
    # two REPL exceptions that *do* live outside — ``REPLTimeoutError`` / ``REPLMemoryError`` in
    # ``repl/_exec_guards.py`` — derive from plain ``Exception``, so they are not precedent.)
    #
    # EXECUTIVE CALL (user, 2026-08-10): a malformed value is rejected, not ignored. Ignoring it is
    # the silent failure — the agent proceeds believing it has the timeout it asked for and is cut
    # off at the configured bound by a bare ``TimeoutError``, with the typo nowhere in sight.
    # Rejecting costs one turn and names the problem. It is raised (rather than reported inline) so
    # that the REPL, not the protocol, owns pragma parsing: ``PythonREPL.exec`` converts it to a
    # recoverable ``Continue`` for the agent loop, and a direct caller of the REPL sees a normal
    # exception. The pattern is deliberately narrow (case-sensitive, colon adjacent to the keyword)
    # so that ordinary prose in a leading comment cannot trip this — see ``_TIMEOUT_PRAGMA_REGEX``.


class SandboxKeyError(ValueError, JazException):
    """Raised when a hook effect names the compiler-sandbox key ``__builtins__``.

    ``__builtins__`` holds the sandbox allow-lists, so binding or dropping it would reopen the
    sandbox. The Add/Drop family refuses it loudly at composition rather than skipping it."""

    # ``ValueError`` despite the name: this is not a failed lookup — ``__builtins__`` is present
    # and resolves fine; it is a *forbidden* argument, which is ``ValueError``'s definition.
    # ``KeyError`` means "key absent from a mapping" and would say the wrong thing. The class
    # name describes the subject (a sandbox key), not the builtin base.

    # A silent skip hid a real hook bug (the author believes they bound/dropped ``__builtins__``
    # but nothing happened); raising surfaces it at composition, where the offending effect is
    # emitted. The individual REPL methods keep a defensive skip as a backstop.

    # TODO: revise docstring


class InvalidEffectError(JazException):
    """Raised when a hook returns an effect at an event that does not accept it.

    Every event documents the effects valid at its stage (supplies at ``*Send``, transforms
    at ``*Complete``, :class:`Abort` at every control event, the ``*Exit`` events accept
    none). An out-of-stage effect is a hook bug — emitting it can only mean the hook expects
    an influence the stage cannot deliver — so composition rejects it loudly rather than
    ignoring it. The message names the offending effect kind and the event.
    """

    # Follows the conflict-error convention (a dedicated JazException per composition
    # contract): a plain rejection, NOT FatalError-category — like a conflict error it
    # propagates out of the invoke whose composition hit it, and a parent invoke may
    # legitimately contain a child's hook-config bug the way it contains any other
    # child failure. This replaced the warn-and-ignore patchwork (and the silent drop in
    # the invoke-enter composer) in the final stage of span_event_lifecycle.md: a stale
    # hook must fail its first run, not silently no-op.
    #
    # ``BlackboardWrite`` never reaches this check: ``HookDispatcher.emit`` partitions it
    # out of the effect stream and applies it at the generational barrier, so it remains
    # valid at every event, observation-only ``*Exit`` included.


class LLMResponseConflictError(JazException):
    """Raised when hooks compose conflicting LLM-response values for the same query.

    Two cases, both order-independence violations: two distinct :class:`SupplyLLMResponse` effects
    supplying a response at :class:`LLMQuerySend`, or two :class:`ModifyLLMResponse` effects setting
    the *same* field to different values at :class:`LLMQueryComplete`. Identical values coalesce;
    distinct ones raise, because resolving by arrival order would make composition depend on hook
    registration or ``with``-nesting order.
    """


#: Deprecated aliases for the pre-rename spellings — see :data:`BudgetForcingException`
#: for the full rationale (the old names were public / in ``__all__``, so dropping them outright
#: would be a bare ``NameError`` for any code that ``except``s or imports them). Aliases, not
#: subclasses, so ``except`` keeps matching exactly. Deliberately absent from ``__all__``.
#: ``OverrideConflictError`` was this class's own former name (renamed with the effect
#: ``OverrideResponse`` -> ``SupplyLLMResponse``); ``ConflictingOverrideError`` was an even older
#: alias of that.
OverrideConflictError = LLMResponseConflictError
ConflictingOverrideError = LLMResponseConflictError


class ReturnValueConflictError(JazException):
    """Raised when effects carrying *distinct* :class:`Return` values fold at one boundary.

    Applies wherever results fold — :class:`SupplyExecResult` at :class:`REPLExecSend`,
    :class:`ModifyExecResult` at :class:`REPLExecComplete`, and :class:`ModifyInvokeResult` at
    :class:`InvokeComplete`. Carried
    :class:`Continue` results merge (outputs concatenate, exceptions group) and identical return
    values coalesce, but two *different* return values cannot be combined without picking a
    winner by hook order.
    """


#: Deprecated alias for the pre-rename spelling — see :data:`BudgetForcingException`
#: for the full rationale (old name was in ``__all__``, so dropping it outright would be a
#: bare ``NameError``). An alias, not a subclass, so ``except`` keeps matching exactly.
#: Deliberately absent from ``__all__``.
ConflictingReturnValueError = ReturnValueConflictError


class ResultConflictError(JazException):
    """Deprecated and never raised — kept only so an existing import keeps resolving.

    Callers migrating off it want :class:`ReturnValueConflictError`, which covers distinct
    carried :class:`Return` values. Unsupported, and removable at the next major bump.
    """

    # Once raised when two hooks supplied *distinct* whole results at the ``REPLExecEnter``
    # supply boundary, on the theory that a supply "names THE result" and so must agree. That
    # contract was dropped: the supply boundary now folds its ``SupplyExecResult``s by the same
    # precedence the exit boundary uses, so independent enter-point vetoes compose instead of
    # crashing.
    #
    # Removed from ``__all__`` (2026-07-31). It had remained *advertised* public API — frozen
    # into the v1 surface by #955 — while being unreachable: zero raise sites, so an out-of-tree
    # ``except ResultConflictError`` could never fire. Advertising a handler that cannot run is
    # worse than not offering one, because it reads as coverage. It concerned a *supply* effect
    # (the enter-boundary ``SupplyExecResult``, then spelled ``OverrideResult``); the surviving
    # ``LLMResponseConflictError`` concerns the LLM-query supply ``SupplyLLMResponse``. Nothing
    # replaces it directly, so no alias was added.


#: Deprecated alias for the pre-rename spelling — see :data:`BudgetForcingException`
#: for the full rationale (old name was in ``__all__``, so dropping it outright would be a
#: bare ``NameError``). An alias, not a subclass, so ``except`` keeps matching exactly.
#: Deliberately absent from ``__all__``.
ConflictingResultError = ResultConflictError


class BlackboardSeedError(JazException):
    """Raised while assembling a per-invoke blackboard's generation 0.

    Either a seed key that no active hook declares in ``blackboard_consumes`` — a likely typo, or
    an orphan datum no consumer will read — or two hooks seeding the same key with different
    values. Both surface before :class:`InvokeEnter` is dispatched.
    """

    # TODO: revise docstring


class BlackboardWriteConflictError(JazException):
    """Raised when two :class:`BlackboardWrite` effects in one event write the same key
    with different values.

    Identical writes coalesce; distinct ones raise, since within an event the write order across
    hooks is irrelevant by design and has no order-independent resolution. Writes in *different*
    events are fine — the later generation overwrites.
    """

    # TODO: revise docstring


def is_fatal(e: BaseException) -> bool:
    """Whether ``e`` should end the whole agent run instead of becoming feedback.

    Consulted at every REPL exec boundary to decide whether
    a caught exception re-raises out of ``exec`` and propagates out of the invoke (fatal)
    or is rendered as a recoverable ``Continue``.

    Two rules, in this order:

    1. **Anything rooted outside ``Exception`` is fatal** — ``KeyboardInterrupt`` /
       ``SystemExit`` / ``GeneratorExit`` / ``asyncio.CancelledError``. This leans on the
       recoverable-vs-fatal split the language already encodes (``Exception`` = "all
       non-system-exiting exceptions") rather than a hand-maintained list; enumerating them
       by hand is what silently omitted ``GeneratorExit`` before.
    2. **:class:`FatalError` (incl. :class:`_JazInternalError`) is fatal**, explicitly —
       including when it is nested anywhere inside a ``(Base)ExceptionGroup``.

    Rule 2 is a deliberate carve-out for an ``Exception``-rooted fatal, added when
    :class:`FatalError` was re-rooted under :class:`JazException` (see its docstring for the
    executive call). It is safe in a way a hand-list is not: it is a single ``isinstance``
    against a type JAZ owns, so it covers every present and future subclass automatically and
    cannot silently omit a member the way enumerating ``BaseException`` leaves did.

    It remains the **only** such carve-out. Notably ``MemoryError`` is still *not*
    special-cased: CPython roots it under ``Exception`` precisely because OOM can be
    recoverable (free a large object and retry), so an agent hitting it gets recoverable
    feedback (the iteration / budget caps backstop a retry loop). Adding per-type carve-outs
    for exceptions JAZ does *not* own would reintroduce the hand-list that hid
    ``GeneratorExit`` — if a third-party error should be fatal, rewrap it as a :class:`FatalError`
    at the raise site rather than extending this function.

    Why classify here at all, rather than ``except Exception`` and let fatals propagate
    (the "why not the fully Python-native shape" the reviewer asked to capture, #561):
    the exec boundary is *forced* to catch ``BaseException`` because the agent's
    ``return``/``raise`` compile to ``BaseException``-rooted control-flow signals (see
    ``python_repl``) — once everything is caught it must be sorted — and every exec
    outcome is a typed :class:`ExecResult` (spans / hooks / cost / captured stdout), so a fatal
    is surfaced as a :class:`Raise` carrying the original exception *and* partial output
    rather than a raw unwind those layers can't see. (It is NOT because the timeout is a
    ``BaseException`` — ``_exec_guards.TimeoutError`` is an ``Exception``.)

    ``AssertionError`` is deliberately *not* fatal (it is an ``Exception``): an agent's own
    ``assert`` stays recoverable; framework asserts that must be fatal raise
    :class:`_JazInternalError` instead.

    Exception groups
    ----------------
    A group is fatal if it contains a :class:`FatalError` **anywhere**, at any nesting depth.
    Judgment call, recorded because the alternative is defensible: a fatal error bundled with
    ordinary errors could be treated as recoverable on the grounds that the group "is mostly
    recoverable". It is not, because a fatal error is not negotiable — allowing a co-occurring
    ``ValueError`` to neutralize it would make fatality depend on what else happened to fail
    at the same moment, and agent code can *construct* such a group deliberately
    (``asyncio.TaskGroup``, or the remainder ``except*`` re-raises alongside a guard's leaf).
    So containment, not majority.

    This is what makes the ``except*`` guard complete: ``except*`` can re-bundle a re-raise with
    an unhandled remainder, so for a mixed group the exec boundary can still see a group — the
    guard alone cannot guarantee a bare :class:`FatalError` there. The guard remains necessary for a
    different reason (it keeps the agent's own clause from consuming the fatal error, and it preserves
    its type across the public boundary, which this rule does not); the two cover
    different failures rather than one subsuming the other. See
    ``repl.python_repl._ExceptGuardTransformer``.
    """
    if not isinstance(e, Exception) or isinstance(e, FatalError):
        return True
    if isinstance(e, BaseExceptionGroup):
        return any(is_fatal(sub) for sub in e.exceptions)
    return False


class HookActivationError(JazException):
    """Raised when one :class:`Hook` instance is activated by more than one path at a time.

    A hook instance is active via exactly one path: the propagating ``with MyHook():``, or a
    single positional argument to one invoke. Overlapping them (``with h: jaz.invoke(h, ...)``),
    listing it twice (``jaz.invoke(h, h, ...)``), or re-entering it while active would run its
    ``setup()``/``teardown()`` more than once and clobber whatever it acquired.
    """

    # The overlaps are also redundant: a ``with``-active hook already propagates to nested
    # same-thread ``invoke()`` calls via the contextvar, so passing it again as a local hook adds
    # nothing. Forbidden loudly rather than silently deduped, so "one instance, one lifecycle
    # scope" is a property of activation itself (#533 / #540).

    # TODO: revise docstring


class ModelPricingUnavailableError(JazException):
    """Raised when a ``cost_budget`` is set but no per-call cost is available to enforce it.

    The usual cause is a model missing from the bundled price table; refresh it (see
    ``jaz/llm/pricing.py``) or use ``calls_budget``, which counts LLM calls and needs no
    rates. Custom or self-hosted models reached through a ``base_url`` never appear in that
    table, so ``calls_budget`` is the way to cap them. Nothing is spent before this is raised,
    unless a custom backend's :meth:`~BaseLLM.can_report_cost` promised a cost
    it then failed to deliver.
    """

    # Raised from either of two checks in `BudgetPool`, whichever fires first:
    #
    # * PRE-FLIGHT (`on_llm_query_enter`), before the first query: the backend answers
    #   `LLM.can_report_cost` with False, so the budget is known unenforceable in advance and
    #   NO LLM call is made. The client is asked rather than the price table consulted here
    #   because only the client knows where its cost comes from — a table lookup would
    #   false-abort `MockLLMClient`, which supplies its own cost and whose default model name
    #   is not in the table.
    # * BACKSTOP (`on_llm_query_complete`, #1071): a turn actually reported no cost despite
    #   that promise (a stale rate, a provider returning no usage). ONE TURN IS SPENT — the
    #   uncosted call itself — but the abort composes at LLMQueryComplete, the very turn the
    #   condition is observed, so the agent never executes the code from that turn and no
    #   further query is made.
    #
    # So the pre-flight check is exact for clients that implement it and the backstop covers
    # the rest; only the backstop path costs a turn.
    #
    # Why this exists: the bundled price table is a filtered snapshot of LiteLLM's data
    # (see llm/pricing.py), so it goes stale whenever a model family ships. Before
    # this error, an unpriced model made `BudgetPool` raise an AssertionError on every
    # LLMQueryExit; the dispatcher swallows hook exceptions, so the only symptom was a log
    # line and a budget that silently enforced nothing. That is the worst failure shape for
    # a spending control — the caller believes they are capped and they are not. Observed
    # live: the GPT-5.6 family (luna/sol) ran uncapped, one of them to 164 turns.
    #
    # Surfaced as an `Abort` effect, NOT by raising: the dispatcher never propagates a hook's
    # exception ("Raising is NOT an abort channel — emit Abort(error=...) instead"), so
    # raising here would be swallowed exactly like the assert it replaces.
    #
    # Historical shape of the hole, kept because it explains why BOTH layers exist. The
    # backstop was originally detect-at-exit / abort-at-NEXT-enter, so an invoke that
    # finished on the uncosted turn never reached another LLMQueryEnter and completed
    # silently — a harness running single-turn invokes under one BudgetPool got no signal at
    # all. The pre-flight was added to close exactly that case for honest clients
    # (`test_unpriced_model_aborts_before_any_llm_call` asserts zero LLM calls), and the
    # backstop's move to LLMQueryComplete (#1071) closed the deferral itself: the abort now
    # composes on the observed turn, last turn included. What remains is only the single
    # uncosted call — an over-promise by a client whose `can_report_cost` answered True,
    # unreachable for the built-in backends, whose answer is the same lookup that produces
    # the cost.
    #
    # Distinct from #1034, where a cost IS reported (0.0, from a table entry with no rates) and
    # so neither layer fires.


class ConfigOverrideActivationError(JazException):
    """Raised when one :class:`ConfigOverride` instance is entered while already active.

    An instance is active within exactly one live scope at a time. Re-entering the *same* one —
    nested on a thread or concurrently from another — is forbidden, mirroring
    :class:`HookActivationError`. Distinct instances nest freely and sequential reuse is fine;
    only overlapping reuse of one instance trips the guard.
    """

    # Unlike a hook there is no ``setup()``/``teardown()`` to double-run, so the overlap is
    # harmless in effect — re-applying the same overrides is a no-op. It is forbidden anyway to
    # keep the API identical to hooks and, concretely, to turn the cross-thread reset hazard
    # (resetting a ``ContextVar`` token from a different ``contextvars.Context`` raises an opaque
    # ``ValueError``) into a clear, actionable error.

    # TODO: revise docstring


class MessageEditIndexError(IndexError, JazException):
    """Raised when a :class:`DropMessages` / :class:`AddMessages` index is out of range.

    Indices follow list semantics against the snapshot the hook saw on :class:`LLMQueryEnter`: a
    negative index counts from the end, and append is ``index == len``. Anything outside
    ``[-len, len]`` for an add or ``[-len, len-1]`` for a drop raises rather than being ignored
    or clamped.

    Both an ``IndexError`` and a :class:`JazException`, so ``except IndexError`` catches it as the
    out-of-range failure it is, while ``except JazException`` still catches everything JAZ raises.
    """

    # Every index is computed against that exact snapshot (no live-buffer race), so out-of-range
    # is a hook bug; and for an add, silently dropping the content would be unrecoverable loss.
    #
    # ``IndexError`` and not ``LookupError``/``ValueError``: the failure is literally a sequence
    # index outside its range, which is the one thing ``IndexError`` means. Nothing on the raise
    # path catches ``IndexError`` or ``LookupError``, so the wider base cannot swallow this
    # silently. Unlike ``KeyError``, ``IndexError`` inherits ``Exception.__str__``, so the message
    # renders unquoted in a traceback and no ``__str__`` override is needed.

    # TODO: revise docstring


class CodeEditOffsetError(IndexError, JazException):
    """Raised when an :class:`InsertCode` / :class:`DeleteCode` character offset is out of range.

    The character-offset twin of :class:`MessageEditIndexError`: offsets follow list semantics
    against the code the hook saw on :class:`REPLExecEnter` (a negative offset counts from the
    end; end-of-string is ``offset == len``). Anything outside ``[-len, len]`` — or a
    :class:`DeleteCode` whose resolved ``start`` exceeds its ``end`` — raises rather than being
    ignored or clamped, since every offset is computed against that exact snapshot and so an
    out-of-range one is a hook bug.

    Both an ``IndexError`` and a :class:`JazException`, mirroring :class:`MessageEditIndexError`.
    """


class MissingDropTargetError(KeyError, JazException):
    """Raised when a :class:`DropVariables` / :class:`DropInputs` names a target that isn't present.

    Dropping a name that isn't there is a hook bug — a typo, or a re-drop-every-turn pattern that
    should be first-turn-only — so it fails loud instead of no-op'ing. Set ``allow_missing=True``
    for a deliberately defensive drop."""

    # ``KeyError`` renders its message through ``repr(args[0])``, so a ``KeyError``-rooted
    # exception prints *quoted* in a traceback ("DropVariables target 'x' is not bound…") — and
    # these messages are full sentences, not bare key names. Inheriting ``Exception.__str__``
    # restores the plain rendering while keeping ``except KeyError`` working. This is why
    # ``MessageEditIndexError`` needs no such override: ``IndexError`` has no custom ``__str__``.
    __str__ = Exception.__str__

    # Nothing on either raise path (``PythonREPL.drop_variables``, ``Agent``'s ``DropInputs``
    # resolution) sits inside an ``except KeyError``/``except LookupError``, so the wider base
    # cannot silently swallow this. Worth re-checking if that ever changes: ``except KeyError``
    # around a dict lookup is common enough in caller code that a swallowed hook bug is the
    # realistic failure mode of this rooting.

    # Safe against the legitimate double-drop case: composition unions the names/keys first and
    # applies the set once against the pre-drop snapshot, so two hooks dropping the same present
    # name coalesce to one deletion and never trip this. ``DropVariables`` naming ``__builtins__``
    # raises ``SandboxKeyError`` at composition first, so it is never treated as a missing target.

    # TODO: revise docstring


# --- LLM provider / transport errors -----------------------------------------
# Raised by the provider layer (the code that calls OpenAI/Anthropic/litellm) and
# normalized into this taxonomy. Defined HERE — the central exception module — rather than
# in jaz/llm/exceptions.py so they share the `JazException` base (one catch-all) and
# are importable from the single public home, `jaz.exceptions`, without exposing the
# internal `jaz.llm` package. `jaz.llm.exceptions` re-exports them for
# back-compat (and hosts the provider-only `is_content_policy_violation` helper).


class LLMError(JazException):
    """Base class for the provider/transport errors raised around an LLM call.

    Catch it to handle any provider failure — rate limit, auth, context window, content policy,
    404, 403, bad parameters, or a generic API error — without naming each subclass. Each
    subclass documents whether it is retryable.

    The subtree keeps LiteLLM's exception names verbatim, so anyone arriving from LiteLLM can
    translate one-for-one.
    """

    # Provenance, because it is not recoverable from the code and keeps getting re-litigated:
    # these eight names are not a jaz design. jaz originally called LiteLLM, and ``3ef1b828``
    # ("Retry for LLM calls") listed ``litellm.exceptions.{NotFound,PermissionDenied,
    # ContextWindowExceeded,UnsupportedParams,API,Authentication}Error`` directly as its retry
    # allowlist. When ``3aab3884`` replaced LiteLLM with direct httpx calls, the classes were
    # re-created here under the same names so the retry list and every call site survived the
    # swap untouched. The vocabulary outlived the dependency.
    #
    # An ``LLM`` prefix on all eight was implemented and then WITHDRAWN (user, 2026-07-30).
    # Recorded so it is not proposed a third time, with the argument and the counter-argument:
    #
    # - For: unprefixed, these are generic enough to collide with an openai/anthropic/litellm
    #   import in the same module, where ``from jaz.exceptions import RateLimitError`` silently
    #   shadows the SDK's identically-named class and ``except RateLimitError`` then catches the
    #   wrong taxonomy with no error.
    # - Against, and decisive: jaz is not expected to be used alongside a raw provider SDK, so
    #   the collision is hypothetical, and nothing in-tree exhibits it (the provider layer is
    #   httpx-only and imports no SDK). Tracebacks are unambiguous regardless, since they qualify
    #   with the defining module. And for the two names whose stem is a bare state rather than an
    #   LLM-domain noun, the prefix actively misread as the *subject* — ``LLMNotFoundError``
    #   parses as "an LLM was not found" rather than "the provider returned 404".
    #   ``LLMAPIError`` also stuttered two acronyms.
    #
    # What the prefix would have bought and is genuinely lost: consistency between this base and
    # its children, and standalone informativeness (``except NotFoundError:`` does not say what
    # was not found). Both judged not worth eight verbose renames.

    # TODO: revise docstring

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        llm_provider: str | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.llm_provider = llm_provider
        super().__init__(message)


class RateLimitError(LLMError):
    """Raised when the API rate limit is exceeded.

    This is a retryable error - the request can be retried after a delay.
    """

    pass


class AuthenticationError(LLMError):
    """Raised when API authentication fails (invalid API key, etc.).

    This is NOT retryable - the API key or credentials need to be fixed.
    """

    pass


class ContextWindowExceededError(LLMError):
    """Raised when the LLM input exceeds the model's context window.

    This is NOT retryable with the same input.
    """

    pass


class ContentPolicyViolationError(LLMError):
    """Raised when a prompt is rejected by the provider's content/usage policy.

    This is NOT retryable with the same input.
    """

    # Providers return this as a 400 (e.g. OpenAI ``invalid_prompt`` / "flagged as potentially
    # violating our usage policy"). It is a distinct, more granular sibling of
    # ``UnsupportedParamsError`` (mirroring LiteLLM's ``ContentPolicyViolationError`` — the
    # upstream name has no ``LLM`` prefix; that is this taxonomy's addition, see ``LLMError``):
    # the request is well-formed, so "fix your parameters" does not apply — the *content* was
    # blocked.
    #
    # Retrying identical flagged input is expected to be blocked again, matching LiteLLM's default
    # of 0 retries for this class. The distinct type exists so callers can special-case it — e.g.
    # an opt-in bounded retry for the mild nondeterminism of moderation, or a fallback to a model
    # with different safety filters.
    #
    # Limitation: this covers only content policy surfaced as an HTTP *error* (the 4xx handled in
    # each provider's ``_handle_error_response``). Anthropic (Claude 4+) also declines some content
    # on a **200** response via ``stop_reason="refusal"`` — a normal response body, not an error —
    # which this type does not represent, since the error path never runs for a 2xx. Handling that
    # (and OpenAI's ``finish_reason="content_filter"`` equivalent) in the success-path parser is
    # tracked in #669.

    pass


class NotFoundError(LLMError):
    """Raised when the LLM provider returns 404 for the completion request.

    jaz posts to exactly one endpoint per provider (``{base_url}/chat/completions``,
    ``/v1/messages``), so a 404 has two realistic causes:

    - the **model name** does not exist — by far the common case, and the first thing to check;
    - the **base URL** is wrong, so the route itself is missing. Not hypothetical:
      ``OPENAI_API_BASE`` / ``ANTHROPIC_API_BASE`` are honored, so anyone pointed at vLLM, a
      LiteLLM proxy, Ollama, or Azure can 404 on the path rather than the model. The message
      carries the URL, which is what tells the two apart.

    NOT retryable — the model name or base URL has to be corrected.
    """

    pass


class PermissionDeniedError(LLMError):
    """Raised when the LLM provider returns 403 for the completion request.

    The party being denied is your API key, not the model: the key is valid — an invalid one is
    401, :class:`AuthenticationError` — but is not permitted this operation or this model (wrong
    project/org scope, a model the account cannot access, a restricted key).

    This is NOT retryable — permissions have to be granted.
    """

    # TODO: revise docstring

    pass


class UnsupportedParamsError(LLMError):
    """Raised when unsupported parameters are passed to the API.

    This is NOT retryable — the parameters have to be corrected.
    """

    pass


class APIError(LLMError):
    """Raised for a server-side API error with no more specific class.

    May or may not be retryable, depending on the underlying error.
    """

    pass


class DescriptionRetentionWarning(UserWarning):
    """Emitted when :func:`jaz.describe` has retained many un-collectable values.

    Bare built-ins (``list``, ``dict``, ``int``, ...) can hold no description of their
    own, so :func:`jaz.describe` keeps them alive for the lifetime of the process; such a
    value is never garbage-collected. A handful of describes on long-lived values costs
    nothing, but describing built-ins in a loop or a per-request path grows without
    bound. This warning marks that case: it is emitted once the retained count crosses a
    threshold, then again at each doubling, so a runaway loop keeps signalling without
    warning on every call.

    The fix is usually one of: describe a longer-lived object that *contains* the built-in
    rather than the built-in itself, or hoist the ``describe`` out of the loop.
    :class:`jaz.Display` at the call site also attaches nothing to the value, but it is
    experimental and warns on use. Filter it like any category::

        import warnings, jaz
        warnings.simplefilter("ignore", jaz.exceptions.DescriptionRetentionWarning)
    """

    # The retention is load-bearing, not an oversight: the strong reference is what keeps
    # the entry's `id()` key valid. Without it CPython could recycle the address and a
    # brand-new object would silently inherit a dead one's description. See the comment
    # under `jaz.descriptions`' module docstring for the full rationale and the two
    # rejected alternatives.


class DescriptionSharedValueWarning(UserWarning):
    """Emitted when :func:`jaz.describe` is called on a value CPython shares process-wide.

    ``None``, ``True``/``False`` and the small integers are *one object each* for the whole
    interpreter, so describing one describes every occurrence of it — in your code, in your
    dependencies, and in every nested ``jaz.invoke``::

        describe(42, "row count")
        invoke(task="…", limit=42)   # `limit` now renders as "row count"

    Unlike :class:`DescriptionRetentionWarning`, this fires on the **first** call: the problem
    is scope, not volume, and a single describe is already the whole failure.

    Describe a small object that *contains* the value instead, so the description belongs to
    something you own rather than to a process-wide singleton. :class:`jaz.Display` labels the
    input for one invoke without attaching anything, but it is experimental and warns on use.
    Filter it like any category::

        import warnings, jaz
        warnings.simplefilter("ignore", jaz.exceptions.DescriptionSharedValueWarning)
    """

    # Scoped to what CPython *guarantees* is shared: the `None`/`True`/`False` singletons and
    # the small-int cache (-5..256). Deliberately NOT extended to interned strings, `""` or
    # `()`, even though those are shared in practice: string interning has no guaranteed set
    # (it varies by literal form, compiler folding, and version), so a warning keyed on it
    # would fire inconsistently across runs of the same code — worse than not warning, because
    # it teaches callers the check is unreliable. The boundary is "documented language
    # guarantee", not "observed identity".
    #
    # Warn rather than raise (executive call, user, 2026-08-07): rejecting these values would
    # make `describe` partial again, which is exactly what this PR's totality decision ruled
    # out. See `_note_shared_value`.


class NonPublicAPIWarning(UserWarning):
    """Emitted when a name absent from a module's ``__all__`` is accessed via the package.

    The name is reachable, but it is not part of the official API and the interface is unstable
    because, e.g., the feature is experimental.

    Filter it like any warning category::

        import warnings, jaz
        warnings.simplefilter("ignore", jaz.NonPublicAPIWarning)   # opt out
        warnings.simplefilter("error", jaz.NonPublicAPIWarning)    # make it fatal
    """

    # Defined here rather than beside the ``__getattr__`` machinery in ``jaz._warnings`` because
    # it is public: ``jaz.exceptions`` is the documented home for the exception/warning taxonomy,
    # and the docs generator pages each public symbol by its defining module — a class defined in
    # a private module would publish a page titled ``jaz._warnings``.
    #
    # What a caller does about it: import the name from its defining module, or use a name in
    # ``__all__``.
    #
    # Rooted at ``UserWarning``, NOT ``DeprecationWarning``, because the category is about
    # *default visibility*, not about lifecycle. ``DeprecationWarning`` is ignored by default
    # outside ``__main__``, so a user calling ``jaz.Agent`` from their own module would never see
    # it — which defeats the entire purpose of warning at the point of access. Python's own split
    # is by audience: ``DeprecationWarning`` targets other Python *developers* (hidden by
    # default), ``FutureWarning`` targets *end users* (shown). This warning targets end users.
    #
    # An earlier version of this docstring justified the rooting with "a demoted name is not
    # scheduled for removal". That was wrong, and ``_warn_nonpublic`` says as much in the message
    # it emits ("may change or be removed in the near future") — the deprecated aliases in this
    # module are demoted *and* removable at the next major bump. It does not change the rooting:
    # this category spans names with different fates (some to be promoted, some replaced, some
    # removed), so no lifecycle-typed category fits all of them, and a name that acquires a real
    # removal date wants its own targeted ``DeprecationWarning``/``FutureWarning`` at that site
    # rather than a re-rooting of the whole category. ``FutureWarning`` is the one defensible
    # alternative — also shown by default — if "will be removed" ever becomes the dominant case;
    # switching is cheap either way, since user filters name this class, not its base.


class UnknownRequestParamWarning(UserWarning):
    """Emitted when a backend construction kwarg looks like a misspelled request parameter.

    A backend (e.g. :class:`jaz.llm.LiteLLM`) forwards any keyword it does not name as a
    per-request default onto the wire, so a typo like ``temperatrue=0.7`` is otherwise sent
    verbatim and silently ignored by the provider. When an unrecognized kwarg closely matches a
    known request parameter, this warns with the suggested spelling — but the kwarg is still
    forwarded, since arbitrary provider-specific extras are supported by design.

    Filter it like any warning category::

        import warnings, jaz
        warnings.simplefilter("ignore", jaz.exceptions.UnknownRequestParamWarning)
        warnings.simplefilter("error", jaz.exceptions.UnknownRequestParamWarning)
    """

    # Warns only on a CLOSE match to a known param, not on every unrecognized kwarg: the
    # ``**request_defaults`` tail is a deliberate open passthrough (vLLM/router extras a JAZ
    # release cannot enumerate), so warning on every unknown key would train users to ignore the
    # category. A near-miss of a real param name is the signal that means "typo", which is the
    # failure this catches. Rooted at ``UserWarning`` (shown by default) for the same
    # audience reason as :class:`NonPublicAPIWarning`.
