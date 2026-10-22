"""JAZ exception taxonomy — the errors raised during agent execution.

Every error defined here derives from :class:`JazException`, which derives from ``Exception``, so
``except JazException`` around :func:`jaz.invoke` catches any of them. The module also publishes
one warning category, :class:`UnknownRequestParamWarning`, which is a ``UserWarning`` rather than
an error.

Configuring JAZ is not agent execution and does not go through this taxonomy: calling a setup
function wrongly raises the builtin that fits (``ValueError``, ``TypeError``), and a component
validating input with someone else's library raises that library's error — registering a template
loader surfaces ``jinja2.TemplateSyntaxError``. Each such function documents what it raises.

Exceptions raised by hooks are unstable API: the hook system is experimental, so the errors it
raises may be renamed, re-rooted, or removed in a future release.
"""

# Whether a new error belongs in this taxonomy is two questions, and it needs both: can a caller of
# `jaz.invoke` write a meaningful `except` for it, and is JAZ the thing that decides it happened?
#
# The second is what keeps registration-time validation out. `add_template_loader` rejecting an
# inheritance loop is JAZ's *detection*, but Jinja's *verdict* — the template is invalid by Jinja's
# rules — so it raises `jinja2.TemplateAssertionError` and a caller catches one family, not two.
# The `LLMError` subtree is the same test answered the other way: JAZ normalises across backends
# and owns that contract, so those re-root here while keeping the vendor SDKs' and LiteLLM's vocabulary.
#
# The first keeps out plumbing a caller does not handle — `REPLTimeoutError`,
# `CookieCollisionError` and friends live in the private modules that raise them and are
# translated before they surface from the outermost `jaz.invoke`, so advertising handlers for
# them would promise an `except` clause nobody should write. (Not *never* observable: a
# sub-invoke's guard failure propagates out of that inner `jaz.invoke` into the exec that ran
# it — see `_exec_guards`, where the exemption from the echo rule below is argued.)
#
# Activation errors are in, though they fire while an invoke or a scope is being set up, which
# the docstring's rule would send to a builtin: :class:`HookActivationError` and
# :class:`ConfigOverrideActivationError` report JAZ's own lifecycle rule (one instance, one live
# scope), not a bad argument, and for ``ConfigOverride`` the builtin it would otherwise surface
# as is contextvars' opaque ``ValueError`` on a cross-thread reset. A caller can act on the class
# and could not on that.
#
# Catch `jaz.exceptions.JazException` for a framework-wide catch-all, or a specific subclass for a
# specific failure mode (e.g. a hook enforcing a budget raises `BudgetExhaustedError`). Either
# catches :class:`FatalError` too (see the comment on that class for why it is rooted at
# `Exception` rather than `BaseException`).
#
# Naming rules, so the next audit does not re-litigate them (#806):
#
# * ``Jaz`` prefixes only the taxonomy root, :class:`JazException` (plus the private
#   :class:`_JazInternalError`). Specific errors are unprefixed and descriptive; one that would
#   clash with a builtin never resolves it with the ``Jaz`` prefix. The rule is: **a name may
#   end in a builtin exception's name only if it subclasses that builtin.** :class:`ReturnTypeError`
#   and :class:`MessageEditIndexError` earn their suffixes; a class that does not subclass the
#   builtin must not borrow its word either, and is named for its subject and verdict instead:
#   :class:`KeyEditDeniedError`, a ``ValueError``, not ``ProtectedKeyError``.
#   :class:`FileAccessDeniedError` subclasses ``PermissionError`` and still has a
#   subject-and-verdict name: the rule bars the echo without the base, not a base without the
#   echo. A domain qualifier like ``REPL`` narrows scope; it does not license the echo. Enforced by ``tests/test_exceptions.py`` over
#   every public module's ``__all__``, not this module's alone — an exception advertised from
#   ``jaz.repl`` is just as catchable. Its ``_ECHO_EXEMPT`` names the two deliberate exceptions
#   — the REPL guards, argued where they are defined — so an exemption is a written decision
#   rather than an accident of which module a class lives in.
# * **A policy refusal that agent code can meet mimics the builtin Python raises for the
#   ordinary failure**, so the agent's usual handling applies, as it would outside the sandbox:
#   a refused file is a ``PermissionError`` (:class:`FileAccessDeniedError`), a refused dynamic
#   attribute (``getattr``) an ``AttributeError``, a refused ``__import__`` an ``ImportError``.
#   A refusal decided before the code runs fails the whole block, so no handler in it can see
#   one: an ``import`` statement is an ``ImportError``, a syntactic attribute access or a
#   reserved name a ``SyntaxError``. The user's
#   call (2026-10-21, #1777), over giving every refusal a class outside the builtin so that no
#   handler written for the real failure could absorb the sandbox's verdict. Of the in-process
#   Python sandboxes surveyed on #1777, RestrictedPython raises a bare ``AttributeError`` and
#   zope.security's ``ForbiddenAttribute`` subclasses ``AttributeError``; Jinja's sandbox makes
#   a refused attribute ``Undefined``, as a missing one is, and raises its own ``SecurityError``
#   only on use; smolagents raises one ``InterpreterError(ValueError)`` for every refusal. None
#   puts a refusal out of reach of the code's own handlers. Platforms that use a distinct class
#   (Deno's ``NotCapable``, Java's ``SecurityException``) give reasons about the host granting
#   a permission, not about code inside the sandbox. The cost: an agent that writes a fallback
#   for the real failure takes it on a refusal too and does not see the policy. For files and
#   imports the allow-list is in the default REPL description, and its multi-step workflow
#   instructions say not to wrap code in ``try``; the attribute allow-list is described
#   nowhere, and describing it would change the default prompt, which stays fixed so published
#   results reproduce. TODO(#1787): tell the agent the attribute rule without moving the
#   default prompt.
# * Conflict errors are ``<subject>ConflictError`` (:class:`REPLInputConflictError`,
#   :class:`LLMResponseConflictError`, …) and derive from ``JazException`` alone, since no builtin
#   exception names "two writers disagree about one slot" — one shape for one contract.
#   ``tests/test_exceptions.py`` enforces the base.
# * The :class:`LLMError` subtree takes the vendor SDKs' names, then LiteLLM's, then another
#   framework's (:class:`ServerError`, google-genai's), else its own (:class:`QuotaExceededError`);
#   see the comment on that class for the provenance and for why an ``LLM`` prefix was considered
#   and dropped.
#
# Decision — NO ``REPLError`` grouping base (#1005). The REPL-layer errors — e.g.
# ``FileAccessDeniedError``, ``REPLInputConflictError``, ``REPLTimeoutPragmaError``,
# ``MissingDropTargetError`` here; ``REPLTimeoutError`` / ``REPLMemoryError`` in ``repl/_exec_guards.py``;
# ``CookieCollisionError`` in ``repl/_cookies.py`` — are deliberately NOT unified under a common base,
# unlike the :class:`LLMError` subtree. ``LLMError`` earns its base because its members share one payload
# and one coherent handling story; these share neither, so a caller writing ``except REPLError`` would
# have no single failure mode to act on. ``except JazException`` is already the framework-wide catch-all,
# and three of these errors are not ``JazException`` at all (they subclass plain ``Exception`` /
# ``RuntimeError`` and live in other modules), so introducing the base would force MRO changes and widen
# what ``except JazException`` catches — cost, for no caller benefit.

# No deprecated aliases for renamed errors. Every old name dropped so far was dropped while the
# latest release was an alpha (0.2.0a1 through 0.2.0a4), and alphas carry no compatibility
# promise (#1367), so the old name goes outright rather than keeping a guard whose only job is a
# better error for a caller that cannot exist. Not because the old names were short-lived: this
# module had no ``__all__`` until 2026-07-28, so several of them were importable for the whole
# life of v0.1.0 and v0.1.1, and most were renamed before 0.2.0a1 and kept aliases until #1380
# dropped them. Whether a name dropped after 0.2.0b1 keeps an alias depends on what the betas
# promise, which is not yet decided.

import math
from collections.abc import Mapping
from typing import Any

__all__ = [
    "JazException",
    "FatalError",
    # Public so callers can filter it by name: a misspelled request parameter is forwarded to
    # the provider rather than rejected, so this is the only signal a caller gets.
    "UnknownRequestParamWarning",
    # Execution / runtime failures
    "ReturnTypeError",
    "FileAccessDeniedError",
    "BudgetExhaustedError",
    "IterationLimitExhaustedError",
    "BudgetPoolExhaustedError",
    "RecursionLimitError",
    "InputRenderError",
    "BudgetForcingError",
    "REPLInputConflictError",
    "REPLTimeoutPragmaError",
    "KeyEditDeniedError",
    # API-misuse failures (hooks / config composition)
    "InvalidEffectError",
    "LLMResponseConflictError",
    "ReturnValueConflictError",
    "BlackboardSeedConflictError",
    "BlackboardWriteConflictError",
    "HookActivationError",
    "ModelPricingUnavailableError",
    "ConfigOverrideActivationError",
    "CodeEditOffsetError",
    "MessageEditIndexError",
    "MissingDropTargetError",
    # LLM provider / transport errors (subtree under LLMError; all JazException)
    "LLMError",
    "RateLimitError",
    "QuotaExceededError",
    "AuthenticationError",
    "NotFoundError",
    "PermissionDeniedError",
    "BadRequestError",
    "UnsupportedParamsError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "ServerError",
    "InternalServerError",
    "OverloadedError",
    "APIResponseValidationError",
    "APIConnectionError",
    "APITimeoutError",
    "RequestAbortedError",
    # The fatality predicate consulted at every exec boundary
    "is_fatal",
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

    It is an ``Exception`` (through :class:`JazException`), not a ``BaseException`` like
    ``KeyboardInterrupt``, because the caller is its audience: ``except Exception`` around
    :func:`invoke` catches it, as it catches every other error JAZ raises. The agent's own
    ``except Exception`` does not, as the table above says; that is the REPL's doing, not the
    base class's.

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

    # Named ``FatalError``, not after the ``Abort`` effect (user decision): the effect and the
    # error category must not share a stem, and the conflict is resolved on THIS name, not the
    # effect's — DOM ``AbortController.abort(reason)`` is the ``Abort`` effect's exact precedent
    # (an operation-scoped stop with a reason payload, containable by the parent), so the
    # mechanism keeps the word "abort", while "fatal" is what this module's own ``is_fatal``
    # predicate calls this category, so the pairing is tautological. The deeper reason an
    # abort-stem name is wrong here: it bakes an INTENT claim into the type name ("someone chose
    # to abort"), but the channel carries unknowable intent (governance stops and breakage
    # reports alike), which is what caused real conflation. "FatalError" claims only a PROPERTY:
    # the error is fatal, hence it propagates past every agent. Property-naming keeps the
    # category intent-agnostic, which the span-outcome classification requires (outcome is
    # decided by control-plane association, never by reading intent off an exception type).
    #
    # EXECUTIVE CALL (user, 2026-07-29): **rooted at ``JazException`` (an ``Exception``)**, so
    # every error JAZ raises is an ``Exception`` and one rule describes the taxonomy — rather
    # than at ``BaseException`` (the rooting #559/#561 argued for), for two reasons:
    #
    # 1. The caller is the audience, and ``BaseException`` rooting hides it from them. The natural
    #    way to handle a failed run is ``try: jaz.invoke(...) except Exception:``, and under that
    #    rooting the one error explicitly routed *to* the caller is the one that handler
    #    misses — the exact inversion of this class's purpose. (Contrast ``KeyboardInterrupt``
    #    escaping ``except Exception``, which is desirable precisely because it is *not*
    #    addressed to you.)
    # 2. The guarantee that rooting would buy is not real. The case for ``BaseException`` is that
    #    no routine handler could swallow a fatal error. It does stop ``except Exception`` — but it is
    #    defeated in the *default* configuration by four lines of agent code (the suppressing
    #    context manager under "Known limitations"). So the property being protected is a strong
    #    default, not a boundary, and it is upheld by mechanism rather than class identity.
    #
    # The docstring states the reason for the rooting, the caller being its audience, though
    # rationale normally lives here: the user asked for it there (2026-10-21), since it is what a
    # caller needs to know to catch it. The mechanism stays here and below.
    #
    # This departs from the field's default, deliberately. Every library surveyed on #1777 that
    # needs an error to get past ``except Exception`` roots it at ``BaseException`` and says so:
    # smolagents' ``FinalAnswerException`` ("to prevent being caught by generic ``except
    # Exception`` clauses in agent-generated code"), Modal's ``InputCancellation``, trio's
    # ``Cancelled``, asyncio's ``CancelledError`` (bpo-32528) and pytest's outcomes; PEP 8
    # reserves ``BaseException`` for "exceptions where catching them is almost always the
    # wrong thing to do". Those signals are addressed to the code that would catch them; this
    # one is addressed to the caller, whose catching it is the point, which is reason 1.
    # (``_JazReturnSignal``, a control-flow signal like theirs, is ``BaseException``-rooted.)
    #
    # Mechanism behind the two known limitations, kept out of the docstring as implementation
    # detail. Suppressing context manager: ``__exit__`` is invoked at the C level and a ``with``
    # block has no ``except`` clause for the cookie transform to guard, so neither the transform
    # nor any base class can stop it. It defeats ``BaseException`` rooting identically, so the
    # rooting is not what is at stake, but it does mean an adversarial agent can decline a
    # fatal error.
    # Group-wrapped fatal error: the cookie guard matches ``FatalError``, and a group *containing* one
    # is not an ``FatalError``, so the guard does not fire. Reachable whenever anything in the
    # call path bundles exceptions — a tool using ``asyncio.TaskGroup``/``gather``, or
    # ``hooks.effects._combine_exceptions``. Guarding it would mean intercepting groups and
    # unwrapping, which a plain injected handler cannot do without also stealing ordinary
    # ``ExceptionGroup``s from the agent's own handlers. #977 tracks the fatality *latch* on the
    # invoke context, which closes both routes at once.


class ReturnTypeError(TypeError, JazException):
    """Raised when the agent's returned value doesn't satisfy the declared return type.

    Produced by :class:`jaz.hooks.ReturnType`: a mismatched ``return`` is downgraded to a
    :class:`Continue` carrying this error, so the agent is told what it got wrong and can retry.
    Past ``ReturnType``'s ``max_failures`` it is raised out of the invoke instead, with the
    offending value in the message. (:class:`jaz.hooks.ValidateReturn` shows the agent its
    validator's own exception instead.)
    """

    # The hook lives in ``hooks/_builtin/return_hooks.py``.
    #
    # A ``TypeError`` (implementation choice, not decided by the user): the value is of the wrong
    # type, which is the builtin's definition, and where a caller meets it uncaught — raised out
    # of the invoke past ``max_failures``, or by the ``InvokeComplete`` backstop when another
    # hook reinstated the return — an ``except TypeError`` around ``jaz.invoke`` catches an
    # agent that could not produce the declared type, as it would a
    # function that returned the wrong one. Before that it is agent feedback: the check runs at
    # the exec boundary after the code returned, so agent code cannot catch it.


class FileAccessDeniedError(PermissionError, JazException):
    """Raised when the REPL refuses a file operation its policy does not allow.

    A :class:`PermissionError`, so agent code handling a denied ``open`` the usual way
    (``except PermissionError`` or ``except OSError``) handles this one too, as it would outside
    the sandbox. What refuses it is the REPL's own policy, not the operating system — usually
    ``allowed_read_paths`` / ``allowed_write_paths``, and the message then names the offending
    path and the allowed patterns; ``errno`` is ``None``. Opening a raw file *descriptor* is
    refused outright instead, with no path to report, since there is nothing to check an
    allow-list against.
    """

    # Raised only by ``repl._secure_methods.secure_open`` — a path outside the configured
    # readable/writable roots, a readable-only path opened for writing, or a file descriptor
    # (no path to check, so it fails closed).
    #
    # NOT ``SandboxPermissionError`` (EXECUTIVE CALL, user, 2026-09-20), which it was, and not
    # ``JazPermissionError``, which it was before that. Two problems with the old name, both of
    # which the new one avoids by naming the subject rather than the mechanism. "Permission"
    # promises OS semantics — chmod, EACCES — where the denial is a configured allow-list the
    # filesystem knows nothing about; and it ends in ``PermissionError`` while NOT subclassing
    # the builtin, so ``except PermissionError`` does not catch it. "Sandbox" is the other
    # problem: this repo has a Docker-backed REPL, so on a file operation that word reads as
    # "the container refused", when in fact this fires before the container sees the call.
    #
    # Of those reasons, two stand: "Permission" in the name would still promise an OS cause where
    # the denial is a configured allow-list, and "Sandbox" would still point at the container.
    # The echo reason no longer applies: the class is a ``PermissionError`` since the user's call
    # of 2026-10-21 (#1777), so every policy refusal agent code meets follows one rule — mimic
    # the builtin Python raises for the ordinary failure (see the module header). It was kept off
    # ``PermissionError`` before so an agent's ``except PermissionError`` could not absorb a
    # policy denial; the message still says the REPL's policy refused.


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


class _JazInternalError(FatalError):
    """Internal framework error — **always fatal**.

    Subclasses :class:`FatalError` so a framework bug / invariant violation fails
    fast (propagates out of the invoke) rather than being hidden from the caller as
    agent feedback. Framework code that must end the run raises this (not a bare ``assert``,
    which is deliberately non-fatal — see :func:`is_fatal`)."""


class InputRenderError(JazException):
    """Raised when an :func:`invoke` input cannot be rendered into the agent's prompt.

    Wraps whatever rendering the value raised -- its own ``__str__``/``__repr__`` failing, or
    anything else -- and names the input; the original is chained as ``__cause__``. Rendering
    happens before the invoke makes a single LLM call, so nothing has been spent.

    It is never fatal: in a sub-invoke it reaches the parent agent as recoverable feedback,
    so the parent can call again without the input. One case it cannot recover from on its
    own: a value inherited from an enclosing :func:`jaz.scope` is not the parent's to remove,
    so every sub-invoke under that scope fails the same way until the scope changes.
    ``KeyboardInterrupt`` and other non-``Exception`` signals are not wrapped and still end
    the run.
    """

    # Executive call (user, 2026-10-06): failing to render an input never warrants ending the
    # invoke tree, so every `Exception` from rendering becomes this, a `FatalError` raised
    # inside a value's `__repr__` included. Rendering is read-only and local: a failure means
    # "this input cannot be shown", not that the run is in a state it cannot recover from --
    # which holds for a bug in the renderer too, since it keeps no state. The scoped-value stall
    # above is the accepted cost, bounded by the iteration and budget caps.
    #
    # Raised by the prompt builder around its single render call (`_render_input_block`), not
    # inside the renderer: the renderer then raises the original exception to any direct
    # caller, and every failure -- the value's own code or a renderer bug -- names its input.
    # Not swallowed into a placeholder: a traceback naming the input is more actionable than an
    # `[unrenderable Foo]` buried in a prompt no one reads.


class BudgetForcingError(JazException):
    """Raised when the :class:`jaz.hooks.BudgetForcing` hook blocks an early
    ``return``/``raise`` finish.
    """


class REPLInputConflictError(JazException):
    """Raised when an added name conflicts with a binding that already exists.

    Two effects adding the same name with *different* values, an :class:`AddInputs` key that
    collides with a caller's own input or scope entry, or an :class:`AddToNamespace` name already
    bound in the REPL namespace. Identical values coalesce instead of raising."""

    # No builtin base, like the other conflict classes below (``LLMResponseConflictError``,
    # ``ReturnValueConflictError``, the two blackboard ones): each says two writers disagree
    # about one slot, which no builtin exception names. It was a ``ValueError`` until the user's
    # call (2026-10-17), on Claude's recommendation, to make the family uniform the smaller way,
    # rather than give all five ``ValueError`` or add a shared base no caller catches.


class REPLTimeoutPragmaError(ValueError, JazException):
    """Raised when the code's leading ``# timeout:`` comment names no positive number of seconds.

    An agent sets a per-execution wall-clock bound with a ``# timeout: <seconds>`` comment among
    the lines opening its REPL code. The value must parse as a positive, finite number, so
    ``# timeout: 120s`` and ``# timeout: 0`` are rejected; a trailing ``#`` comment after the
    number is allowed. The offending code is not executed — the agent sees the error as ordinary
    REPL feedback and can correct it on the next iteration."""

    # ``ValueError``: nothing failed to resolve — the line is a well-formed comment, and it is the
    # *value* that is inappropriate ("an argument of the right type but an inappropriate value").
    #
    # Defined here rather than beside the parser in ``jaz.repl._python_repl`` because this module is
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


class KeyEditDeniedError(ValueError, JazException):
    """Raised when an invoke input, or a namespace or input edit, names a key the REPL refuses
    to let it bind or drop.

    Today that is ``__builtins__``, which holds the REPL compiler's allow-lists (which builtins,
    imports and attributes the agent's code may reach), so binding or dropping it would reopen
    them. The framework refuses it, rather than skipping it, wherever it binds or drops a name
    for a caller or a hook: an input passed to :func:`~jaz.invoke` or :func:`~jaz.scope`, or an
    Add/Drop effect, for example. The REPL also raises it at the end of an exec whose code
    replaced ``__builtins__`` (through ``globals()``, which the compiler cannot refuse), and on
    every later read of the sandbox; that invoke then fails with it, and a parent invoke sees the
    failure as ordinary feedback. The message names the key."""

    # ``ValueError`` despite the name: this is not a failed lookup — ``__builtins__`` is present
    # and resolves fine; it is a *forbidden* argument, which is ``ValueError``'s definition.
    # ``KeyError`` means "key absent from a mapping" and would say the wrong thing. The class
    # name describes the subject and the verdict, like its sibling ``FileAccessDeniedError``,
    # and deliberately does NOT end in ``KeyError``: this is a ``ValueError``, and a name
    # trailing a builtin exception it does not subclass sends a reader to write the ``except``
    # that will not catch it. ``ProtectedKeyError``, the previous spelling, had exactly that
    # defect, and ``MissingDropTargetError`` in this same module IS a ``KeyError``, so the file
    # would have taught both conventions at once. NOT
    # "sandbox" (EXECUTIVE CALL, user, 2026-09-19): in this repo that word also means a container
    # or remote executor — there is a Docker-backed REPL — so a reader meeting it in a traceback
    # would look in the wrong place. ``KeyEditDeniedError`` over ``BuiltinsKeyError`` for the same
    # reason the name matters at all: it says what the rule IS (this key is protected from
    # writes) rather than which key happens to be the only member today, and the message already
    # names the key, so precision is not lost by the class being about the category.

    # A silent skip hid a real hook bug (the author believes they bound/dropped ``__builtins__``
    # but nothing happened); raising surfaces it at composition, where the offending effect is
    # emitted. The individual REPL methods raise it too on a direct call, for the same reason.


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
    # child failure. Loud rather than warn-and-ignore (or a silent drop) because a stale hook
    # must fail its first run, not silently no-op.
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


class ReturnValueConflictError(JazException):
    """Raised when effects carrying *distinct* :class:`Return` values fold at one boundary.

    Applies wherever results fold — :class:`SupplyExecResult` at :class:`REPLExecSend`,
    :class:`ModifyExecResult` at :class:`REPLExecComplete`, and :class:`ModifyInvokeResult` at
    :class:`InvokeComplete`. Carried
    :class:`Continue` results merge (outputs concatenate, exceptions group) and identical return
    values coalesce, but two *different* return values cannot be combined without picking a
    winner by hook order.
    """


class BlackboardSeedConflictError(JazException):
    """Raised while assembling a per-invoke blackboard's generation 0.

    Two hooks seeding the same key with different values: the board cannot pick one without
    depending on hook order. Surfaces before :class:`InvokeEnter` is dispatched.
    """


class BlackboardWriteConflictError(JazException):
    """Raised when two :class:`BlackboardWrite` effects in one event write the same key
    with different values.

    Identical writes coalesce; distinct ones raise, since within an event the write order across
    hooks is irrelevant by design and has no order-independent resolution. Writes in *different*
    events are fine — the later generation overwrites.
    """


def is_fatal(e: BaseException) -> bool:
    """Whether ``e`` should end the whole agent run instead of becoming feedback.

    Consulted at every REPL exec boundary: a caught exception either re-raises out of ``exec``
    and propagates out of the invoke (fatal) or is rendered as a recoverable ``Continue``.

    Two rules, in this order:

    1. Anything rooted outside ``Exception`` is fatal — ``KeyboardInterrupt``, ``SystemExit``,
       ``GeneratorExit``, ``asyncio.CancelledError``. This is the recoverable-vs-fatal split the
       language already encodes, not a hand-maintained list.
    2. :class:`FatalError` is fatal, explicitly — including when it is nested anywhere inside a
       ``(Base)ExceptionGroup``, at any depth. Containment, not majority: a fatal error bundled
       with ordinary ones stays fatal.

    That is the only carve-out. ``MemoryError`` is not special-cased (CPython roots it under
    ``Exception`` because OOM can be recoverable — the iteration and budget caps backstop a
    retry loop), and neither is ``AssertionError`` — an agent's own ``assert`` stays
    recoverable. To make a third-party error fatal, rewrap it as a :class:`FatalError` at the
    raise site rather than extending this function.
    """
    # Lives here, not in `jaz.repl`, by executive call (user, 2026-10-08): its only caller is the
    # REPL exec boundary (and any `BaseREPL` implementation's), but what it decides -- which
    # exceptions end the run -- is a property of the exception taxonomy this module defines, so
    # the semantics belong beside `FatalError`. Moving it to `jaz.repl` to signal "for REPL
    # implementers only" was considered and rejected.
    #
    # Rule 1 leans on the language's split rather than a list because enumerating
    # `BaseException` leaves by hand is what silently omitted `GeneratorExit` before.
    #
    # Rule 2 is a deliberate carve-out for an `Exception`-rooted fatal, added when `FatalError`
    # was re-rooted under `JazException` (the executive call is recorded on `FatalError`). It is
    # safe in a way a hand-list is not: a single `isinstance` against a type JAZ owns covers
    # every present and future subclass — `_JazInternalError` included, which is what framework
    # asserts raise instead of `AssertionError`. Per-type carve-outs for exceptions JAZ does NOT
    # own would reintroduce the hand-list.
    #
    # Why classify here at all, rather than `except Exception` and let fatals propagate (the
    # "why not the fully Python-native shape" question): the exec boundary is FORCED to catch
    # `BaseException`, because the agent's `return`/`raise` compile to `BaseException`-rooted
    # control-flow signals (see `_python_repl`) — once everything is caught it must be sorted —
    # and every exec outcome is a typed `ExecResult` (spans / hooks / cost / captured stdout),
    # so a fatal is surfaced as a `Raise` carrying the original exception AND partial output
    # rather than a raw unwind those layers can't see. (It is NOT because the timeout is a
    # `BaseException` — `_exec_guards.REPLTimeoutError` is an `Exception`.)
    #
    # Group containment is a judgment call, recorded because the alternative is defensible: a
    # fatal bundled with ordinary errors could be called recoverable on the grounds that the
    # group "is mostly recoverable". It is not, because a fatal error is not negotiable —
    # letting a co-occurring `ValueError` neutralize it would make fatality depend on what else
    # happened to fail at the same moment, and agent code can CONSTRUCT such a group
    # deliberately (`asyncio.TaskGroup`, or the remainder `except*` re-raises alongside a
    # guard's leaf). This is also what makes the `except*` guard complete: `except*` can
    # re-bundle a re-raise with an unhandled remainder, so for a mixed group the exec boundary
    # can still see a group — the guard alone cannot guarantee a bare `FatalError` there. The
    # guard remains necessary for a different reason (it keeps the agent's own clause from
    # consuming the fatal error, and preserves its type across the public boundary, which this
    # rule does not); the two cover different failures. See `_ExceptGuardTransformer` in
    # `jaz.repl._python_repl`.
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


class ModelPricingUnavailableError(JazException):
    """Raised when a ``cost_budget`` is set but no per-call cost is available to enforce it.

    The usual cause is a model the backend cannot price — LiteLLM has no entry for it, or the
    backend does not set ``LLMResponse.cost_usd`` at all. Use ``calls_budget``, which counts LLM
    calls and needs no rates. Custom or self-hosted models reached through a ``base_url`` are
    rarely priced, so ``calls_budget`` is the way to cap them. Nothing is spent before this is
    raised, unless the backend's ``model_info.can_report_cost`` (see :class:`jaz.llm.ModelInfo`)
    promised a cost that a call then came back without: :class:`jaz.llm.GitHubCopilotLLM` does
    when GitHub leaves a call's credit figure out, and a custom backend can.
    """

    # Raised from either of two checks in `BudgetPool`, whichever fires first:
    #
    # * PRE-FLIGHT (`on_llm_query_enter`), before the first query: the backend's
    #   `model_info.can_report_cost` is False, so the budget is known unenforceable in advance and
    #   NO LLM call is made. The backend is asked rather than any table consulted here
    #   because only the client knows where its cost comes from — a table lookup would
    #   false-abort `MockLLM`, which supplies its own cost and whose default model name
    #   is not in the table.
    # * BACKSTOP (`on_llm_query_exit`, on the Completed arm): a turn actually reported no cost
    #   despite that promise (a stale rate, a provider returning no usage). ONE TURN IS SPENT —
    #   the uncosted call itself — but the abort composes at the Completed LLMQueryExit, the very
    #   turn the condition is observed and still before the agent acts on the response, so the
    #   agent never executes the code from that turn and no further query is made. It reads the
    #   FINAL post-transform cost, so a `ModifyLLMResponse(cost_usd=...)` re-price rescues the
    #   turn instead of tripping this.
    #
    # So the pre-flight check is exact for clients that implement it and the backstop covers
    # the rest; only the backstop path costs a turn.
    #
    # Why this exists: a price map goes stale whenever a model family ships, and LiteLLM's
    # is no exception. Before
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
    # (`test_unpriced_model_aborts_before_any_llm_call` asserts zero LLM calls), and moving the
    # backstop onto the observed turn closed the deferral itself: the abort now
    # composes on the observed turn, last turn included. What remains is only the single
    # uncosted call — a client whose `can_report_cost` answered True returning `cost_usd=None`.
    # `LiteLLM` cannot, since its answer is the same lookup that produces the cost;
    # `GitHubCopilotLLM` can, when a call's usage event lacks GitHub's credit figure.
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
    """Raised when a :class:`DropFromNamespace` / :class:`DropInputs` names a target that isn't present.

    As with Python's ``del``, a builtin the sandbox allows (``len``) that no variable shadows is
    not present: it is the sandbox's policy, not a variable. Dropping a name that isn't there is
    a hook bug — a typo, or a re-drop-every-turn pattern that should be first-turn-only — so it
    fails loud instead of no-op'ing. Set ``allow_missing=True`` for a deliberately defensive
    drop."""

    # ``KeyError`` renders its message through ``repr(args[0])``, so a ``KeyError``-rooted
    # exception prints *quoted* in a traceback ("DropFromNamespace target(s) ['x'] are not bound…") — and
    # these messages are full sentences, not bare key names. Inheriting ``Exception.__str__``
    # restores the plain rendering while keeping ``except KeyError`` working. This is why
    # ``MessageEditIndexError`` needs no such override: ``IndexError`` has no custom ``__str__``.
    __str__ = Exception.__str__

    # Nothing on either raise path (``PythonREPL.drop_from_namespace``, ``Agent``'s ``DropInputs``
    # resolution) sits inside an ``except KeyError``/``except LookupError``, so the wider base
    # cannot silently swallow this. Worth re-checking if that ever changes: ``except KeyError``
    # around a dict lookup is common enough in caller code that a swallowed hook bug is the
    # realistic failure mode of this rooting.

    # Safe against the legitimate double-drop case: composition unions the names/keys first and
    # applies the set once against the pre-drop snapshot, so two hooks dropping the same present
    # name coalesce to one deletion and never trip this. A ``DropFromNamespace`` naming
    # ``__builtins__`` is refused with ``KeyEditDeniedError`` instead: the key is present (Python's
    # ``del __builtins__`` finds it) but protected. An unshadowed allowed builtin is missing,
    # as ``del len`` finds nothing to delete (see ``PythonREPL.drop_from_namespace``).


# --- LLM provider / transport errors -----------------------------------------
# Raised by the `jaz.llm` backends (the code that calls litellm, Copilot, or an SGLang
# server) and normalized into this taxonomy. Defined HERE — the central exception module — rather
# than in `jaz/llm/_exceptions.py` so they share the `JazException` base (one catch-all) and have
# a single public home, `jaz.exceptions`; `jaz.llm._exceptions` is the in-package import path the
# backends use.


class LLMError(JazException):
    """Base class for the provider/transport errors raised around an LLM call.

    Catch it to handle any provider failure without naming each subclass. Each subclass
    documents whether it is retryable.

    The names follow the ``openai`` and ``anthropic`` SDKs where they define the error
    (:class:`BadRequestError`, :class:`APITimeoutError`, :class:`APIConnectionError`, ...),
    LiteLLM's where only LiteLLM does (:class:`ContextWindowExceededError`, ...), and otherwise
    another framework's (:class:`ServerError`, google-genai's) or JAZ's own
    (:class:`QuotaExceededError`). Unlike those SDKs' ``APIError``, which every API error
    subclasses, the root here is ``LLMError``.

    Every attribute but ``message`` is ``None`` when the backend did not see it: no response
    came back (a timeout, a refused connection), or the backend does not expose it.

    Attributes:
        message: The error message, the provider's where it gave one.
        status_code: The HTTP status the provider answered with.
        body: The provider's response body, decoded from JSON where it is JSON, else its
            text: its error body, or on :class:`jaz.llm.SGLangLLM` a 200 it could not use or
            whose request was aborted.
        headers: The provider's response headers, with lowercase names.
        code: The provider's machine-readable error code, such as
            ``"credit_balance_exhausted"`` or ``"rate_limit_error"``: the first string among
            the error object's ``code``, ``type`` and ``status`` fields (OpenRouter's
            ``metadata.error_type`` after them), else Bedrock's ``x-amzn-ErrorType`` header;
            on :class:`jaz.llm.GitHubCopilotLLM`, the Copilot runtime's error code.
        request_id: The provider's request id (the ``x-request-id``, ``request-id`` or
            ``x-amzn-RequestId`` header; on :class:`jaz.llm.GitHubCopilotLLM`, GitHub's
            ``x-github-request-id``, else the Copilot service's), to quote when reporting a
            problem to the provider.
        retry_after: How long the provider asked the caller to wait before retrying, in
            seconds, from its ``retry-after-ms`` or ``retry-after`` header, else the
            ``retryDelay`` Gemini and Vertex put in the body. A value that is not a finite,
            non-negative number is stored as ``None``. A backend's retry waits this long plus
            jitter; a zero waits the usual backoff, and one longer than
            ``RetryPolicy.retry_after_max`` stops the retries.
        is_retryable: Whether to retry it whatever its class, as the provider said in an
            ``x-should-retry`` header (read by :class:`jaz.llm.LiteLLM` and
            :class:`jaz.llm.SGLangLLM`; :class:`jaz.llm.GitHubCopilotLLM`'s runtime passes no
            response headers); ``None`` leaves it to the class, and only ``True`` or
            ``False`` is stored. Retry caps still apply, a ``retry_after`` longer than
            ``RetryPolicy.retry_after_max`` stops the retries even when it is ``True``.
    """

    # Sources for the docstring's naming paragraph: the `openai` SDK's classes
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_exceptions.py#L106-L161),
    # LiteLLM's `ContextWindowExceededError`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L535),
    # google-genai's `ServerError`
    # (https://github.com/googleapis/python-genai/blob/v2.29.0/google/genai/errors.py#L299),
    # and the SDKs' root `APIError`
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_exceptions.py#L46,
    # https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_exceptions.py#L35).
    #
    # Provenance, because it is not recoverable from the code and keeps getting re-litigated:
    # the original eight names are not a jaz design. jaz originally called LiteLLM, and ``3ef1b828``
    # ("Retry for LLM calls") listed ``litellm.exceptions.{NotFound,PermissionDenied,
    # ContextWindowExceeded,UnsupportedParams,API,Authentication}Error`` directly as its retry
    # allowlist. When ``3aab3884`` replaced LiteLLM with direct httpx calls, the classes were
    # re-created here under the same names so the retry list and every call site survived the
    # swap untouched. The vocabulary outlived the dependency.
    #
    # On 2026-10-10 the user chose to follow the vendor SDKs over LiteLLM where they disagree, after
    # a comparison with 24 SDKs, gateways and agent frameworks found three names that meant
    # something else in the SDKs: `APIError` (their root, renamed `ServerError`), `Timeout` (their
    # HTTP client's re-exported settings object, renamed `APITimeoutError`) and
    # `UnsupportedParamsError` (a client-side refusal elsewhere, now a subclass of the SDKs'
    # `BadRequestError`).
    #
    # An ``LLM`` prefix on all eight was implemented and then WITHDRAWN (user, 2026-07-30).
    # Recorded so it is not proposed a third time, with the argument and the counter-argument:
    #
    # - For: unprefixed, these are generic enough to collide with an openai/anthropic/litellm
    #   import in the same module, where ``from jaz.exceptions import RateLimitError`` silently
    #   shadows the SDK's identically-named class and ``except RateLimitError`` then catches the
    #   wrong taxonomy with no error.
    # - Against, and decisive: jaz is not expected to be used alongside a raw provider SDK, so
    #   the collision is hypothetical, and nothing in-tree exhibits it (the built-in backends
    #   reach their SDK's exception classes through the SDK's namespace, never by bare import).
    #   Tracebacks are unambiguous regardless, since they qualify with the defining module. And for the two names whose stem is a bare state rather than an
    #   LLM-domain noun, the prefix actively misread as the *subject* — ``LLMNotFoundError``
    #   parses as "an LLM was not found" rather than "the provider returned 404".
    #   ``LLMAPIError`` also stuttered two acronyms.
    #
    # What the prefix would have bought and is genuinely lost: consistency between this base and
    # its children, and standalone informativeness (``except NotFoundError:`` does not say what
    # was not found). Both judged not worth eight verbose renames.
    #
    # What a built-in backend translates into this subtree, and what it lets through: a failure
    # becomes an `LLMError` when it is the outcome of the call — the provider answered with an
    # error, the connection to it failed, its response was unusable, the request was rejected as
    # invalid (by the provider, by LiteLLM, or by the backend at call time, as `LiteLLM`'s
    # `UnsupportedParamsError` for a process-wide LiteLLM setting), or a check the caller set up
    # in the client blocked it (LiteLLM's `max_budget`, a LiteLLM guardrail). Those are what a
    # caller acts on: retry, back off, change the request, pay. The rest passes through as
    # itself: a backend rejects bad arguments passed to it with the builtin that fits
    # (`TypeError`, `ValueError`), as configuration does (see the module header); a bug keeps its
    # class and traceback; and a cancellation (`asyncio.CancelledError`, `KeyboardInterrupt`) is
    # never caught. The line matters because retries go by class: a bug translated into a
    # retryable class is retried, and a provider failure left untranslated fails fast. A custom
    # backend is not bound to the rule (`BaseLLM.retryable_exceptions` takes any class). LiteLLM
    # wraps every exception it does not recognise, its own bugs included, in its
    # `APIConnectionError`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L2684-L2691);
    # `LiteLLM` lets one through when a programming error is what failed,
    # and so also a response LiteLLM could not read that way (see `_BUG_CLASSES` in
    # `llm/_litellm.py`). Where a built-in backend falls short: `GitHubCopilotLLM` lets through an
    # SDK error that came without a session error event, such as the `JsonRpcError` for an
    # unknown model (TODO(#1361)).

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
        code: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        is_retryable: bool | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.body = body
        self.headers = headers
        self.code = code
        self.request_id = request_id
        self.retry_after = (
            float(retry_after)
            if isinstance(retry_after, int | float)
            and not isinstance(retry_after, bool)
            and math.isfinite(retry_after)
            and retry_after >= 0
            else None
        )
        self.is_retryable = is_retryable if isinstance(is_retryable, bool) else None
        super().__init__(message)

    # `is_retryable` is the Vercel AI SDK's `isRetryable`
    # (https://github.com/vercel/ai/blob/ai%407.0.137/packages/ai/src/util/retry-with-exponential-backoff.ts#L85-L86)
    # and the `openai`/`anthropic` SDKs' `x-should-retry` handling
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L896-L904):
    # one error's verdict, which a class cannot carry.
    #
    # The fields the `openai` SDK (`body`, `code`, `request_id`;
    # https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_exceptions.py#L69-L103),
    # the `anthropic` SDK (`body`, `type`, `request_id`;
    # https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_exceptions.py#L54-L92),
    # pydantic-ai (`body`, `headers`, `retry_after`;
    # https://github.com/pydantic/pydantic-ai/blob/v2.55.0/pydantic_ai_slim/pydantic_ai/exceptions.py#L547-L548)
    # and DSPy's lm15 (`request_id`, `retry_after`, `code`;
    # https://github.com/stanfordnlp/dspy/blob/3.4.0/dspy/_vendor/lm15/errors.py#L59-L74)
    # carry, so a handler can read the provider's answer without walking `__cause__`, which
    # only `LiteLLM` sets. Not `provider` or `model`: the caller knows which backend and model
    # it configured, and an `llm_provider` keyword was removed for that reason. Keyword-only
    # and defaulted, so `cls(message, status_code=...)` everywhere keeps working, and they
    # survive pickling as instance attributes. A bad `retry_after` or `is_retryable` from a custom
    # backend becomes `None` rather than raising: raising here would replace the error being
    # reported, the retry wait would otherwise crash on a non-number, and the retry reads any
    # `is_retryable` other than `None` or `False` as "retry", so `0` or `"false"` would
    # otherwise retry.


class RateLimitError(LLMError):
    """Raised when the provider refuses the call for now, in a way that clears by retrying without
    anyone acting: a per-minute or other rolling-window limit, a 429, a 402 that says how long to
    wait and names no quota, or on :class:`jaz.llm.GitHubCopilotLLM` a per-session spending cap that
    a new session resets. A limit that needs someone to act, or lasts until a calendar reset, is a
    :class:`QuotaExceededError` instead, and the provider's quota code says so at a 429 too.

    Retried by default, after the wait the provider asks for (``retry_after``); a wait longer
    than ``RetryPolicy.retry_after_max`` is not retried, and ``retry_after`` says when to try
    again.
    """

    # This class and `QuotaExceededError` are told apart by what clears the limit (the user's
    # call, 2026-10-17): retrying alone (a rolling window, a stated wait, a fresh session), or
    # someone acting or a calendar reset. The evidence per signal is beside the code that reads
    # it: a 402's stated wait in `_litellm._classified`, Copilot's codes at
    # `_copilot._QUOTA_CODES`, the per-day limits at `_litellm._QUOTA_CODES`.

    pass


class QuotaExceededError(LLMError):
    """Raised when the account's or project's quota, credit or spend limit is used up, so the
    call cannot succeed until someone acts (adds credit, raises a budget, enables billing) or
    the quota resets on the calendar (a daily quota, a monthly allowance or spend cap). A limit
    that clears by retrying without anyone acting (a rolling window, a stated wait, a new
    session) is a :class:`RateLimitError` instead.

    Not retried by default, since resending cannot fix it, unless the provider's
    ``x-should-retry`` header says to. To keep retrying while someone tops up the account, opt
    in: ``RetryPolicy(extra_retryable_exceptions=[QuotaExceededError])``.

    Every backend raises it for a 402 status unless the 402 says how long to wait and names no
    quota, or, on :class:`jaz.llm.GitHubCopilotLLM`, is the per-session spending cap on a reused
    session that spent on earlier turns; and for the provider's own quota codes at any other status.
    :class:`jaz.llm.LiteLLM` also recognizes a provider's documented quota message, and raises it
    when ``litellm.max_budget`` is spent. An error a backend does not recognize as a quota surfaces
    as whatever class its status maps to, typically :class:`RateLimitError` or
    :class:`BadRequestError`.
    """

    # OpenRouter's in-flight-budget 402 is transient and carries a `Retry-After`
    # (https://openrouter.ai/docs/api/reference/limits#in-flight-spending-budget).
    #
    # Named "Exceeded", not "Exhausted" (the user's call, 2026-10-11): the providers say
    # "exceeded" (OpenAI's `organization_spend_limit_exceeded`
    # (https://developers.openai.com/api/docs/guides/error-codes),
    # AWS `ServiceQuotaExceeded`, Copilot `quota_exceeded`
    # (https://github.com/github/copilot-sdk/blob/v1.0.19/nodejs/src/generated/session-events.ts#L2128),
    # LiteLLM's `BudgetExceededError`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L998));
    # "exhausted" would suggest Gemini's `RESOURCE_EXHAUSTED` lands here, though that also covers
    # per-minute limits (https://ai.google.dev/gemini-api/docs/rate-limits),
    # which are a `RateLimitError`; and it keeps the provider's refusal apart from JAZ's own
    # `BudgetExhaustedError`.
    #
    # A sibling of `RateLimitError`, not a subclass (an implementation choice, not decided by
    # the user): nesting it would follow
    # precedent rather than any principle, and no consumer catches `RateLimitError` to handle a
    # quota. As a sibling, `limits` caps the two apart and `except RateLimitError` keeps
    # meaning "slow down".
    #
    # Not retried by default (the user's call, reversing an earlier "retried" so that users
    # switching from another framework are not surprised). Surveyed 2026-10-09: the frameworks
    # that classify quota on purpose fail fast (DSPy's `LMBillingError` is outside its retryable
    # set (https://github.com/stanfordnlp/dspy/blob/3.4.0/dspy/utils/exceptions.py#L263);
    # the Vercel AI SDK marks a streamed `insufficient_quota` non-retryable
    # (https://github.com/vercel/ai/blob/ai%407.0.137/packages/openai/src/openai-stream-error.ts#L303-L312)),
    # and the openai/anthropic SDKs, the LiteLLM Router, LangChain, pydantic-ai and openai-agents
    # fail fast on a billing 400/402/403
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L907-L922,
    # https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_base_client.py#L876-L891,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L7165-L7188,
    # https://github.com/langchain-ai/langchain/blob/langchain-openai%3D%3D1.7.1/libs/partners/openai/langchain_openai/chat_models/base.py#L1470-L1471,
    # https://github.com/openai/openai-agents-python/blob/v0.23.1/src/agents/models/_openai_retry.py#L92-L93).
    # They do retry a quota 429 (OpenAI `insufficient_quota`, Anthropic's spend cap), but only
    # because they retry by status and never read the code; Anthropic documents its spend-cap
    # 429 as failing "until access resumes"
    # (https://platform.claude.com/docs/en/api/errors#http-errors).
    # Retrying cost the default budget (11 calls, about 5 minutes) on an account nobody may top up.
    # The rejected alternative was matching the SDKs by status, which would make the same
    # out-of-credit condition retry on one provider and fail on another. Detection and the
    # provider markers live in `_litellm.py`; the question was raised on #1621.

    pass


class AuthenticationError(LLMError):
    """Raised when API authentication fails (invalid API key, etc.).

    Not retried by default - the API key or credentials need to be fixed.
    """

    pass


class BadRequestError(LLMError):
    """Raised when what was asked has to change, so resending it unchanged usually fails the
    same way: the provider rejects the request, the backend refuses to send it, or (as
    :class:`ContentPolicyViolationError`) the provider blocks the prompt or the response. Its
    subclasses name a cause, such as :class:`ContextWindowExceededError` for an input too long
    for the model. Where nothing more is known the status decides: any 4xx with no class of its
    own raises it, registered in HTTP or not (an invalid parameter, a request too large, Groq's
    498), except 408, 409 and 499, which are retried as :class:`ServerError` (on
    :class:`jaz.llm.SGLangLLM` a 499 is :class:`RequestAbortedError`). Where the backend
    recognizes them, an account out of credit raises :class:`QuotaExceededError` instead, and a
    wrong model name or address :class:`NotFoundError`.

    Not retried by default. The message carries the reason, the provider's where it sent one.
    """

    # This class and `NotFoundError` are assigned by what has to change, the status deciding only
    # where nothing else says (the user's call, 2026-10-17): this one for what was asked,
    # `NotFoundError` for where it was sent. The `openai` and `anthropic` SDKs bind the same names
    # to the status instead.
    #
    # The `openai`/`anthropic` SDKs' and LiteLLM's name for a generic 4xx. Every 4xx
    # without its own class lands here (`_class_for_status`), a 422 included, though the SDKs
    # give a 422 its own `UnprocessableEntityError`, a sibling of their `BadRequestError`
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_exceptions.py#L153,
    # https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_client.py#L882-L883).
    # Kept as one class rather than one per cause.
    #
    # TODO(#1671): Groq documents its 422 ("semantic errors or model hallucinations") as possibly
    # worth retrying (https://console.groq.com/docs/errors), yet it fails fast here, as in the
    # `openai` SDK.

    pass


class UnsupportedParamsError(BadRequestError):
    """Raised when a request parameter or backend setting is not supported, before anything is
    sent: :class:`jaz.llm.LiteLLM` raises it for a parameter LiteLLM cannot translate for the
    provider (set ``drop_params=True`` to drop such parameters instead), and for a LiteLLM
    global the backend refuses, such as ``litellm.num_retries``.

    A :class:`BadRequestError`, so ``except BadRequestError`` catches it. Not retried by
    default.
    """

    # Narrowed to LiteLLM's meaning (its `UnsupportedParamsError` is raised client-side, before
    # sending; https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L3783-L3785),
    # which any-llm's `UnsupportedParameterError` shares
    # (https://github.com/mozilla-ai/any-llm/blob/1.33.1/src/any_llm/exceptions.py#L172).
    # A provider's own 400 for a bad parameter is a plain `BadRequestError`: nothing in the
    # status tells the two apart.

    pass


class ContextWindowExceededError(BadRequestError):
    """Raised when the LLM input exceeds the model's context window.

    A :class:`BadRequestError`, so ``except BadRequestError`` catches it. Not retried by
    default, since the same input will not fit.
    """

    # A subclass, as in LiteLLM
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L535),
    # DSPy (`LMInvalidRequestError`;
    # https://github.com/stanfordnlp/dspy/blob/3.4.0/dspy/utils/exceptions.py#L218)
    # and lm15 (`InvalidRequestError`;
    # https://github.com/stanfordnlp/dspy/blob/3.4.0/dspy/_vendor/lm15/errors.py#L344):
    # a handler for rejected requests should not miss the commonest one. It also makes a 413
    # consistent across backends: the shared status rule calls it a `BadRequestError`, and
    # `GitHubCopilotLLM`, whose SDK documents its 413 as a context limit
    # (https://github.com/github/copilot-sdk/blob/v1.0.19/nodejs/src/generated/session-events.ts#L625),
    # the narrower class.

    pass


class ContentPolicyViolationError(BadRequestError):
    """Raised when the provider blocks a prompt or response under its content or usage policy,
    or the model refuses it: as an error status, or, on :class:`jaz.llm.LiteLLM` and
    :class:`jaz.llm.GitHubCopilotLLM`, as a refusal or content filter signaled on a successful
    response, where ``status_code`` is ``None`` since no error status came back. Models LiteLLM
    routes through its Responses API bridge (such as ``gpt-5-pro`` and the ``-codex`` models)
    lose a refusal, which arrives as an ordinary response; their content filter does raise. On
    :class:`jaz.llm.LiteLLM` it is also raised, with ``status_code`` ``None``, when a guardrail
    LiteLLM runs blocks the call. LiteLLM runs guardrails on async calls (``jaz.ainvoke``), and on
    sync calls only for the models on that Responses API bridge.

    A :class:`BadRequestError`, so ``except BadRequestError`` catches it. Not retried by
    default. The same input can pass when resent, so to retry it add it to
    ``RetryPolicy.extra_retryable_exceptions``.
    """

    # LiteLLM runs guardrails on async calls: its `wrapper_async` calls them
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L2006)
    # and the sync `wrapper` does not.
    #
    # Providers return this as a 400 (e.g. OpenAI ``invalid_prompt``). A subclass of
    # `BadRequestError` (the user's call, 2026-10-11), as in LiteLLM (its
    # `ContentPolicyViolationError(BadRequestError)`;
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L619)
    # and the `openai` SDK (which raises its `BadRequestError` for the 400;
    # https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_client.py#L867),
    # so a switcher's `except BadRequestError` still catches it. It was a sibling, on the argument
    # that the request was well-formed and only the content was blocked, and that a refusal on a
    # 200 is no rejected request; matching the peers won over that.
    #
    # Fail-fast by default, the convention (LiteLLM, OpenAI's SDK), since some prompts are blocked
    # every time. But OpenAI's `invalid_prompt` is often not deterministic: in the official AppWorld
    # baseline (gpt-5.4-nano, 3x417 tasks) 15 of 15 flagged calls passed on the first identical
    # retry, and in mini-swe-agent SWE-bench runs (gpt-5-mini) at least 88 passed on an identical
    # retry while 7 failed all 10 attempts (counted from one machine's run logs when #1616 was
    # written; no committed record holds the counts. jaz-evals-internal's
    # `checklists/run_appworld_jaz.md` records the related kill rates, 15.7% vs 0.9%). The OpenAI
    # message LiteLLM quotes says as much: "your request may succeed if retried"
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L620). So retry is an
    # opt-in, with a `limits` cap to bound the cost of a prompt that is blocked for good. Falling
    # back to a model with different safety filters is the other way to special-case it.
    #
    # On a refusal raised from a successful response `status_code` stays `None` rather than the
    # 2xx, because LiteLLM does not expose the status of a response it parsed.
    #
    # A refusal or filter on a 200 raises this class too, and so fails fast by the same
    # default; `_raise_if_refused` in `llm/_litellm.py` and `_filtered_error` in
    # `llm/_copilot.py` have the signals they read.

    pass


class NotFoundError(LLMError):
    """Raised when the request went to the wrong place, so the model name or the base URL has
    to change.

    Usually a 404, with two realistic causes:

    - the **model name** does not exist, or names a model the account is not entitled to — by
      far the common case, and the first thing to check. Most providers answer both with a 404
      (OpenAI's ``model_not_found`` reads "does not exist or you do not have access to it");
      some send a 403 for the entitlement (:class:`PermissionDeniedError`), Anthropic and
      Bedrock among them;
    - the **base URL** is wrong, so the route itself is missing: anyone pointed at vLLM, a
      LiteLLM proxy, Ollama, or Azure can 404 on the path rather than the model.

    On Azure the model name is the *deployment* name, and a wrong one is a 404 too.

    Also raised for a redirect (a 3xx) the backend does not follow, and for redirects that never
    end, and for OpenAI's ``model_not_found`` code whatever the status. A provider that answers
    an unknown model with a plain 400 raises :class:`BadRequestError` instead. On ``LiteLLM``, a
    301, 302 or 303 is followed as a GET, as browsers and httpx do, so it surfaces as whatever
    the new address answers.

    Not retried by default — the model name or base URL has to be corrected.
    """

    # Azure: Microsoft Learn ties a 404 to a base URL missing `/openai/v1/` and to an invalid
    # deployment name on the v1 surface, where the deployment goes in `model`
    # (https://learn.microsoft.com/en-us/azure/foundry/openai/supported-languages, "For a 404
    # response, confirm that the base URL ends in /openai/v1/ and that model contains a valid
    # deployment name"; https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/embeddings
    # says the same with "endpoint includes"; Learn does not version its URLs, read 2026-10-21).
    # `LiteLLM`'s `azure/` route takes that surface only for an api_version of `v1`, `latest` or
    # `preview` (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/llms/azure/common_utils.py#L859-L862,
    # `_is_azure_v1_api_version`); with a dated one the deployment goes in the URL path, and that
    # route's 404 rests on the Q&A bodies below. Azure's error code for the deployment case,
    # `DeploymentNotFound`, is named, with no status, in Learn's hosted-agent troubleshooting
    # table ("ResourceNotFound or DeploymentNotFound | Verify the endpoint URL and model deployment
    # name", https://learn.microsoft.com/en-us/azure/foundry/agents/quickstarts/quickstart-hosted-agent);
    # it appears with a 404 in bodies users posted on Microsoft Q&A
    # (https://learn.microsoft.com/en-us/answers/questions/1479158/how-to-fix-deploymentnotfound-error),
    # and LiteLLM maps it to its `NotFoundError`
    # (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/litellm_core_utils/exception_mapping_utils.py#L1981-L1988).
    # Since no Microsoft page gives the code and the status together, the docstring does not
    # name the code. Whether a wrong `api-version` is a 404 is not documented, and is not claimed.
    #
    # Assigned by what has to change, not by the 404 (see `BadRequestError`). A redirect is the
    # address being wrong, as a 404 is, and the same on every attempt (`_class_for_status`).
    # A 404 whose fix is neither (Bedrock's use-case form) is taken out by its error type
    # (`_litellm._AWS_ACCOUNT_STANDING_ERRORS`). The providers that answer an entitlement with
    # a 403 are cited under `PermissionDeniedError`. TODO(#1744): a wrong model name sent as a
    # 400 with no code (Bedrock, the Copilot CLI) is still a `BadRequestError`.

    pass


class PermissionDeniedError(LLMError):
    """Raised when the provider answers 403 and nothing in its response says otherwise: the key
    is valid — an invalid one is 401, :class:`AuthenticationError` — but is not permitted the
    request.

    What a 403 covers differs by provider (a key outside the project's scope, Bedrock's model
    access not enabled, OpenAI's unsupported region), so the class is the status's verdict
    rather than one fix. A 403 whose body the backend recognizes as a content block, a quota or
    a bad credential raises :class:`ContentPolicyViolationError`, :class:`QuotaExceededError` or
    :class:`AuthenticationError` instead; OpenRouter's ``permission_denied`` covers a key's
    missing permission and a guardrail block alike, and arrives as this. A model the account is
    not entitled to is a 404 (:class:`NotFoundError`) or a 400 on most providers, not a 403.
    Also raised where the provider says the account's standing is the problem whatever the
    status, such as a Gemini response stopped with ``PUP_LIMITED_DISABLED`` or Bedrock's 404 for
    an account that has not submitted Anthropic's use-case form.

    Not retried by default — a permission has to be granted, or the account's standing
    restored.
    """

    # Status-bound with named exceptions because the providers' docs agree on nothing finer:
    # model access is a 403 on Anthropic (`permission_error`, "does not have permission to use
    # the specified resource", https://platform.claude.com/docs/en/api/errors) and Bedrock
    # (`AccessDeniedException`,
    # https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html#ts-access-denied)
    # but a 404 or 400 on most (OpenAI's 404 "Requested resource does not exist",
    # https://developers.openai.com/api/docs/guides/error-codes; Gemini's 404 `model_not_found`,
    # https://ai.google.dev/gemini-api/docs/api-errors); a region block is a 403 on OpenAI
    # ("Country, region, or territory not supported", same page) and a 400 `FAILED_PRECONDITION`
    # on Gemini (community-sourced: "User location is not supported for the API use"; Gemini's
    # page gives disabled billing as its `failed_precondition` example); and Together's 403 is a
    # context overflow (`_litellm._together_overflow`). So any single fix the class promised
    # would be wrong somewhere.

    pass


class ServerError(LLMError):
    """Raised when the provider fails to answer the request properly: a server error (5xx), a
    408 (Request Timeout) or 409 (Conflict), which the `openai` and `anthropic` SDKs also retry,
    a 499 outside :class:`jaz.llm.SGLangLLM`, a response that is not a valid completion, or a
    failure no more specific class describes (such as a status outside 4xx and 5xx). A failed
    transport, a timeout included, raises :class:`APIConnectionError` instead.

    A 5xx raises the subclass :class:`InternalServerError`; on :class:`jaz.llm.SGLangLLM`, a
    success status whose response is not a valid completion raises
    :class:`APIResponseValidationError`.

    Retried by default, with its subclasses (it is in ``BaseLLM.retryable_exceptions``). On
    :class:`jaz.llm.LiteLLM` the original LiteLLM exception is chained as ``__cause__``.
    """

    # The docstring's 408/409 retry: the `openai` SDK
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L907-L915)
    # and the `anthropic` SDK
    # (https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_base_client.py#L876-L881).
    #
    # The name of google-genai's
    # (https://github.com/googleapis/python-genai/blob/v2.29.0/google/genai/errors.py#L294-L299)
    # and DSPy's (lm15's;
    # https://github.com/stanfordnlp/dspy/blob/3.4.0/dspy/_vendor/lm15/errors.py#L367)
    # class for the provider's own failure. Not `APIError`: in the `openai`/`anthropic`/`groq`
    # SDKs and google-genai, `APIError` is the root of every API error
    # (https://github.com/groq/groq-python/blob/v1.7.0/src/groq/_exceptions.py#L25;
    # it catches a 429 or a 401; LiteLLM's own `APIError` does not, its classes sharing only
    # `openai.APIError`;
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L811),
    # the meaning `LLMError` has here, so a class by that name that did not catch them silently
    # narrowed a switcher's `except APIError` (the user's call, 2026-10-10, after a comparison with
    # 24 SDKs and frameworks).
    #
    # 408 and 409 land here, not in `BadRequestError`, because the `openai` SDK and LiteLLM's
    # Router retry them (https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/router.py#L7948);
    # google-genai's `ServerError` is 5xx only
    # (https://github.com/googleapis/python-genai/blob/v2.29.0/google/genai/errors.py#L201-L203),
    # so the name is a slight stretch.

    pass


class InternalServerError(ServerError):
    """Raised when the provider answers with a server error status (5xx). A 529, Anthropic's
    "overloaded", raises the subclass :class:`OverloadedError`.

    Retryable, as a :class:`ServerError`; key ``RetryPolicy.limits`` on it to cap 5xx answers
    apart from invalid responses.
    """

    # The `openai`/`anthropic` SDKs' and LiteLLM's name for a 5xx
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_client.py#L888-L889,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L760).
    # A subclass rather than the status alone so `except` and `limits` can tell a provider's own
    # failure from a response that did not parse; the status stays in `status_code`.

    pass


class OverloadedError(InternalServerError):
    """Raised when the provider answers that it is overloaded (HTTP 529, as Anthropic sends).

    Retryable, as an :class:`InternalServerError`.
    """

    # The `anthropic` SDK's class for its 529
    # (https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_client.py#L582).
    # Only that status: a 503 can mean overloaded too, but providers also send it for maintenance
    # or a model being unavailable (https://console.groq.com/docs/errors). Nested under
    # `InternalServerError`, unlike the `anthropic` SDK, where the two are siblings
    # (https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_exceptions.py#L167-L179):
    # the `openai` SDK raises its `InternalServerError` for a 529, so `except InternalServerError`
    # keeps catching it for a switcher from either.

    pass


class APIResponseValidationError(ServerError):
    """Raised when the provider answers with a success status but the response is not a valid
    completion, such as a body that is not JSON, one missing a field the backend needs, an error
    object in place of a completion, a tool call when no tools were offered, or a response that
    fails a JSON schema the caller asked the backend to enforce
    (``litellm.enable_json_schema_validation`` on :class:`jaz.llm.LiteLLM`). ``status_code`` is
    the success status where the backend saw it, else ``None``. :class:`jaz.llm.LiteLLM`
    detects only a body that is not JSON; JSON of the wrong shape surfaces as whatever class
    LiteLLM's own error maps to, which on some providers is not retried.

    Retryable, as a :class:`ServerError`: a garbled response is usually transient, and another
    sample may pass the schema.
    """

    # TODO(#1641): LiteLLM's garbled 200s that are JSON are not detected as this class.
    #
    # The `openai`/`anthropic` SDKs' name. They do not retry it
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_base_client.py#L753), since
    # for them it means the response failed their own schema, which resending will not change; here
    # a garbled answer from the server is retried, as the user chose (#1622). A response failing the
    # caller's schema on LiteLLM is retried too (the user's call, 2026-10-11): another sample often
    # passes, it was retried as a plain `ServerError` before this class existed, and LiteLLM's
    # Router and `completion(num_retries=...)` both retry it (status 500;
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L906,
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/utils.py#L1920). The cost is a full
    # generation per retry; failing fast was the alternative, which would have needed a subclass or
    # a retry-rule exception to keep SGLang's garbled responses retried.

    pass


class APIConnectionError(LLMError, ConnectionError):
    """Raised when the transport fails, rather than the server answering with an error: the
    endpoint cannot be reached, the request cannot be sent, or no usable response comes back
    (the connection drops, what arrives cannot be parsed or decoded, such as a body whose
    compression is broken), or the call times out (the subclass :class:`APITimeoutError`).

    Retried by default; key ``RetryPolicy.limits`` on ``APIConnectionError`` to cap it, timeouts
    included. If it persists, check the network and the endpoint the backend talks to (its base
    URL, or for a CLI-backed backend, the CLI). It is also a builtin ``ConnectionError``, so
    ``except ConnectionError`` (and ``except OSError``) catches it.
    """

    # The `openai`/`anthropic`/`groq` SDKs' and LiteLLM's name
    # (https://github.com/groq/groq-python/blob/v1.7.0/src/groq/_exceptions.py#L69).
    # Its own class, apart from `ServerError`, because the transport failed rather than a server
    # answering with an error status jaz could read (implementation choice, not decided by the
    # user): as in the `openai` SDK, where it is not an `InternalServerError`
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_exceptions.py#L106),
    # and lm15, where `TransportError` sits beside `ServerError`
    # (https://github.com/stanfordnlp/dspy/blob/3.4.0/dspy/_vendor/lm15/errors.py#L78).
    # So it is in `BaseLLM.retryable_exceptions` by name.
    #
    # A builtin `ConnectionError` (the user's call, "split, builtin bases", 2026-10-09): the
    # naming rule in this module's header requires it, and a handler written for the builtin
    # then catches it.

    pass


class APITimeoutError(APIConnectionError, TimeoutError):
    """Raised when an LLM call times out before the provider answers.

    An :class:`APIConnectionError`, so ``except APIConnectionError`` and
    ``limits={APIConnectionError: n}`` cover it too; key ``RetryPolicy.limits`` on
    ``APITimeoutError`` to cap timeouts on their own, or raise the backend's timeout. Retried by
    default. It is also a builtin ``TimeoutError`` (and so an ``OSError``), so
    ``except TimeoutError`` catches it.
    """

    # The `openai`/`anthropic`/`groq` SDKs' name. Not `Timeout`, LiteLLM's name and this class's
    # earlier one: in those SDKs `Timeout` is the HTTP client's settings object they re-export
    # (`httpx.Timeout`, or `httpx2.Timeout` in the latest `openai` and `anthropic`;
    # https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_types.py#L36,
    # https://github.com/anthropics/anthropic-sdk-python/blob/v1.13.0/src/anthropic/_types.py#L36).
    # The user's call, 2026-10-10, with the `ServerError` rename.
    #
    # A subclass of `APIConnectionError` (the user's call, 2026-10-11), as in the `openai` and
    # `anthropic` SDKs
    # (https://github.com/openai/openai-python/blob/v3.29.0/src/openai/_exceptions.py#L111)
    # and LiteLLM (`Timeout(openai.APITimeoutError)`;
    # https://github.com/BerriAI/litellm/blob/v1.105.0/litellm/exceptions.py#L356),
    # so a switcher's `except APIConnectionError` still catches a timeout. It was a sibling, so that
    # a cap on connection failures would not also cap timeouts; the cost of nesting is exactly that,
    # and a cap on timeouts alone still works.
    #
    # Its own class because its remedy differs: a longer timeout, or fewer retries on an
    # expensive generation. A builtin `TimeoutError` (the user's call, "split, builtin bases",
    # 2026-10-09), so a handler written for the builtin catches it. The cost is the reverse: code
    # around an LLM call that catches the builtin for its own deadline must check it is its own
    # (the async REPL exec guard does, via `asyncio.Timeout.expired()`).

    pass


class RequestAbortedError(LLMError):
    """Raised when the server reports the request as aborted: on :class:`jaz.llm.SGLangLLM`,
    SGLang 0.5.17's 499 for a request aborted before it was scheduled, or a request SGLang
    aborted (a pause with ``mode="abort"``, a weight update, ``/abort_request``) and answered
    with a 200 carrying the partial generation, which is then in :attr:`body`. SGLang answers a
    grammar failure the same way, so that raises it too, though nobody cancelled it.

    Not retried by default, since an abort is normally meant to stop the request. To retry it,
    add it to ``RetryPolicy.extra_retryable_exceptions``; the retry re-sends the original prompt
    from scratch and does not continue from the partial generation. A server's
    ``x-should-retry`` header still decides where it sends one. Raised in a sub-invoke, it
    reaches the parent's code as an ordinary exception, which can catch it and go on querying.
    """

    # The two limitations above are the ones an RL trainer meets: continuing from the partial
    # after the pause lifts is what the frameworks that resume do (TODO(#1768)), and a parent
    # that swallows a child's abort keeps the server busy through the trainer's drain
    # (TODO(#1767); making the class fatal is part of the deferred taxonomy decision).
    #
    # Its own class, not a `ServerError` with a "do not retry" verdict, so that a caller can opt
    # it back in by class, as with `QuotaExceededError` (the user's call, 2026-10-13, after a
    # review found the verdict-only version could not be opted in). Named after SGLang 0.5.17's
    # `RequestAbortedError`, which raises it
    # (https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/tokenizer_manager.py#L198-L199;
    # SGLang reverted it in sgl-project/sglang#34160):
    # neither vendor SDK has a class for a 499, and the taxonomy takes another framework's name
    # before inventing one (see the naming rules in the module header).

    pass


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
    # ``**query_params`` tail is a deliberate open passthrough (vLLM/router extras a JAZ
    # release cannot enumerate), so warning on every unknown key would train users to ignore the
    # category. A near-miss of a real param name is the signal that means "typo", which is the
    # failure this catches. Rooted at ``UserWarning`` (shown by default) so a typo surfaces
    # without the caller having to opt in.
