import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import get_origin, overload

from ._agent import Agent
from ._config import (
    ConfigOverride,
    ConfigStack,
    _reset_config_stack,
    _set_config_stack,
    get_config_stack,
)
from ._scope import _scope_var
from ._subinvoke import Subinvoke, get_subinvoke

# `normalize_inputs` lowers any t-string INPUT VALUES (a Template passed as an input —
# `task` is an ordinary input, not a positional prompt, #538) into text
# plus sibling bindings. `Template` itself is imported only for the positional-argument
# guard below, which names a t-string prompt as such; on <3.14 it is that module's dummy
# stand-in, against which `isinstance` is simply always False (no t-string can exist there).
from ._templates import Template, normalize_inputs
from .exceptions import (
    HookActivationError,
    KeyEditDeniedError,
)
from .hooks import Hook, ReturnType
from .hooks._context import (
    HookContext,
    _reset_hook_context,
    _set_hook_context,
    get_current_hooks,
)
from .repl._stdout_proxy import capturing as _stdout_capturing

logger = logging.getLogger(__name__)


# The prehook for the children of the currently-running invoke, published while an
# agent runs (see `_invoke` / `_ainvoke`). The *public* `jaz.invoke` reads this to
# inherit the enclosing invocation's depth / invoke-calls accounting when it is
# reached from inside a running agent (e.g. via a human-written tool that calls
# `jaz.invoke`), instead of minting a fresh root. Mirrors the config
# (`_config_stack`) and hook (`_hook_context`) ContextVars: live ContextVar is
# primary; unset means a true top-level call. On a raw worker thread the agent
# facade re-establishes this var from its captured ancestor before it calls in
# (see get_subinvoke in _subinvoke.py, alongside the symmetric config/hook
# re-establishment), so a public `jaz.invoke` reached from inside a worker
# inherits the enclosing depth rather than minting a fresh root.
_current_prehook: ContextVar["Prehook | None"] = ContextVar(
    "jaz_current_prehook", default=None
)


def get_active_prehook() -> "Prehook | None":
    """Return the live child-prehook ContextVar value, or None if unset.

    None means a true top-level invoke — no enclosing agent. A raw worker thread the
    agent spawned does not inherit the var, but the agent facade re-establishes it from
    its captured ancestor around each sub-invoke call, so code running under one still
    sees the enclosing invoke. The public `invoke` prefers this over minting a fresh
    root `Prehook(repl_depth=1)`.
    """
    return _current_prehook.get()


def _set_current_prehook(prehook: "Prehook") -> "Token[Prehook | None]":
    """Publish the child prehook on the ContextVar (internal; returns a reset token)."""
    return _current_prehook.set(prehook)


def _reset_current_prehook(token: "Token[Prehook | None]") -> None:
    """Reset the child-prehook ContextVar from a `_set_current_prehook` token."""
    _current_prehook.reset(token)


def _teardown_in_order(hooks: list[Hook], exc: BaseException | None) -> None:
    """Tear ``hooks`` down first to last, each in the previous one's ``finally``."""
    # Nesting each teardown in the previous one's `finally` is what nested `__exit__` calls
    # are, so Python gives exactly `with`'s result without this code building any of it: every
    # teardown runs even when one raises, and the one that propagates is the last raised, with
    # every earlier one -- and anything each raised internally -- on its `__context__` chain.
    # Holding the exceptions and re-raising them afterwards cannot match it: a `raise` inside a
    # `finally` re-sets the raised exception's own `__context__`.
    #
    # Each teardown is handed the exception actually in flight, as an outer `__exit__` is: the
    # invoke's own, or an earlier teardown's interrupt. Passed explicitly rather than read from
    # `sys.exception()`, which also returns an exception the CALLER is handling -- `invoke`
    # called inside an `except` would hand every teardown of a clean run that exception.
    if not hooks:
        return
    try:
        _teardown_logging_bugs(hooks[0], exc)
    except BaseException as interrupt:  # noqa: BLE001 - handed to the next teardown, re-raised
        exc = interrupt
        raise
    finally:
        _teardown_in_order(hooks[1:], exc)


def _teardown_logging_bugs(hook: Hook, exc: BaseException | None) -> None:
    try:
        hook.teardown(exc)
    except Exception:
        # A teardown bug is logged and dropped; a `BaseException` propagates as under `with`.
        logger.exception("Local hook %s teardown failed", type(hook).__name__)


@contextmanager
def _activate_local_hooks(hooks: tuple[Hook, ...]) -> Iterator[None]:
    """Run setup()/teardown() around a single invoke for its local hooks.

    Unlike the `with MyHook():` path, this does NOT register the hook *instances* on
    the contextvar — that is what keeps local hooks from propagating (being dispatched)
    to nested invokes. It records only their *ids* in ``HookContext.local_active`` (for
    collision detection, below) and drives the resource lifecycle, passing any exception
    that propagates out of the invoke to teardown() so stateful hooks (e.g. tracing) see
    the same exception they would via __exit__.

    Enforces the "one instance, one live scope" invariant before running
    any setup(). A second activation that would run setup()/teardown() twice on one live
    instance raises ``HookActivationError`` (by identity, so distinct instances of the
    same class are fine):
    - already active via `with` on the contextvar
      (`with h: jaz.invoke(h, ...)`) — also redundant, since the contextvar
      already propagates `h` to this same-thread invoke;
    - already a live local in an *enclosing* invoke (an invoke passed `h` positionally
      whose agent code, at any nesting depth, invokes again passing `h`) —
      caught via ``local_active``, which the contextvar carries for exactly that
      enclosing invoke's dynamic extent (its setup..teardown window);
    - listed twice in one call (`jaz.invoke(h, h, ...)`).

    The worker-thread restore path is unaffected: it re-establishes the ancestor
    *propagating* set and passes the sub-invoke's own (disjoint) local hooks, so no
    instance lands in both.
    """
    # The invariant and every route into it (with+with, with+local, a repeated instance in one
    # call) are #533 / #540.
    current = get_current_hooks()
    propagating = current.hooks
    local_active = current.local_active
    seen: list[Hook] = []
    for hook in hooks:
        if any(hook is p for p in propagating):
            raise HookActivationError(
                f"{type(hook).__name__} instance is active via `with` and is also passed to "
                "this invoke; this would run setup()/teardown() twice and is redundant (a "
                "`with`-active hook already propagates to nested invokes). Use exactly one "
                "path."
            )
        if id(hook) in local_active:
            raise HookActivationError(
                f"{type(hook).__name__} instance is already passed to an enclosing invoke; "
                "passing it again would run setup()/teardown() twice while its per-invoke "
                "lifecycle is still live. Use one scope at a time."
            )
        if any(hook is s for s in seen):
            raise HookActivationError(
                f"{type(hook).__name__} instance is passed to this invoke more than once; "
                "this would run setup()/teardown() twice. List each hook instance at most "
                "once."
            )
        seen.append(hook)
    # Record these locals as live on the contextvar for the invoke's whole dynamic
    # extent (detection only — their *ids*, not the instances: they stay off `hooks`, so
    # they are NOT dispatched to and do NOT propagate into nested invokes). This is what
    # lets a nested `with h:` / positional `h` see that h's lifecycle is still live
    # and refuse to re-run it. Reset in the outermost finally so even a setup() failure
    # can't leak the token.
    token = _set_hook_context(current.with_local_active(id(h) for h in hooks))
    try:
        # Local hooks run setup() DIRECTLY here — NOT via ``Hook.__enter__`` (the ``with`` path).
        # ``RecursionLimit`` relies on this exact routing to reject the local-hooks channel (its
        # ``setup()`` raises unless reached through ``__enter__``); if this ever went through
        # ``__enter__``, that rejection would silently stop firing and the under-cap footgun
        # would return. Keep it direct.
        #
        # Teardown covers the hooks whose `setup()` RETURNED, not the whole list, and the
        # bookkeeping is what makes the difference: the setup loop is not inside the `try` whose
        # `finally` tears down, so without it a `setup()` that raises would skip teardown for every
        # hook already started beside it. Reachable since a hook can refuse activation outright
        # (`Hook.forbid_reactivation`) — `jaz.invoke(fresh, already_used, ...)` set up the first
        # and then stranded it, leaving its file handle or span open for the process's life.
        #
        # The raiser itself is deliberately absent from `started`: a `setup()` that did not
        # complete has nothing to undo, matching `with`, where a throwing `__enter__` means no
        # `__exit__`.
        started: list[Hook] = []
        exc: BaseException | None = None
        try:
            for hook in hooks:
                hook.setup()
                started.append(hook)
            yield
        except BaseException as e:  # noqa: BLE001 - captured to forward to teardown, then re-raised
            exc = e
            raise
        finally:
            # Reverse order, as nested `with` blocks exit.
            _teardown_in_order(list(reversed(started)), exc)
    finally:
        # A SECOND `finally`, wrapping the one above, rather than both jobs in one block: a
        # `BaseException` propagating out of a teardown would otherwise skip the reset
        # written after it, leaking this invoke's `local_active` entry for the life of the
        # process. Splitting them makes the token reset unconditional on anything teardown does.
        _reset_hook_context(token)


@dataclass(kw_only=True, frozen=True)
class PrehookOutput:
    repl_depth: int
    parent_invoke_id: str | None = None
    parent_repl_iteration: int | None = None


class Prehook:
    """Callable prehook that tracks parent invoke context.

    The parent_repl_iteration field is mutable and updated by the Agent
    at each REPL iteration, so nested invokes know which iteration spawned them.
    """

    def __init__(
        self,
        repl_depth: int,
        parent_invoke_id: str | None = None,
    ) -> None:
        self.repl_depth = repl_depth
        self.parent_invoke_id = parent_invoke_id
        self.parent_repl_iteration: int | None = None
        # Snapshot the ancestor hook context at construction time (parent's thread).
        # Closure-threaded alongside the prehook so the ancestor hook chain survives a raw
        # worker thread, where the _hook_context ContextVar doesn't propagate: get_subinvoke
        # re-bases this snapshot's hooks UNDER any the worker activates locally (#727 composition,
        # not either/or). An established live ContextVar stays primary (same-thread and
        # copy_context paths); only a fresh worker falls back to this snapshot.
        #
        # Residual limitation (STRUCTURAL, not a snapshot-timing bug — option A, #727): captured
        # before the agent's REPL runs, so a hook the agent activates MID-REPL is not reflected in
        # a *raw* worker thread it then spawns (the ContextVar doesn't cross a ThreadPoolExecutor,
        # and the framework has no parent-thread interception point at pool.submit to re-snapshot).
        # This is inherent to contextvars, not fixable here. The supported way to carry mid-REPL
        # hook state into a worker is `contextvars.copy_context().run(lambda: jaz.invoke(...))`,
        # which copies the live (established) context across — see hooks/README.md and the
        # TestHookWorkerRebase tests. (Fully mirroring mid-REPL state via a shared mutable holder
        # was rejected: it introduces a parent-teardown-vs-worker exit race — see #727.)
        self.parent_hooks: HookContext = get_current_hooks()

    def __call__(self) -> PrehookOutput:
        return PrehookOutput(
            repl_depth=self.repl_depth,
            parent_invoke_id=self.parent_invoke_id,
            parent_repl_iteration=self.parent_repl_iteration,
        )


@dataclass(kw_only=True)
class _InvokeSetup:
    """Pre-computed infrastructure shared between the sync and async invoke paths.

    Does not carry ``return_type`` or a return validator: the return type is a positional
    ``ReturnType`` hook and return-value validation is the ``ValidateReturn`` hook, so nothing
    here knows about either.
    """

    agent: Agent
    repl_depth: int
    invoke_id: str
    parent_invoke_id: str | None
    parent_repl_iteration: int | None
    subinvoke: Subinvoke
    # No `config`: the resolved config reaches the Agent through its constructor, and a second
    # copy here would have no reader.
    child_prehook: Prehook
    # Explicit `**inputs` kwargs and resolved ambient `jaz.scope`, kept as SEPARATE
    # provenance channels all the way into Agent.(a)invoke (#727). The agent merges
    # them into the single REPL namespace itself, and the prompt renders scope vs.
    # explicit inputs in distinct sections straight from these two dicts — no merge
    # here and no re-split downstream. They are DISJOINT (a name defined both ways
    # raises above), so keeping them apart loses nothing.
    inputs: dict[str, object]
    scope: dict[str, object]


def _build_invoke_setup(
    *,
    local_hooks: list[Hook] | None = None,
    prehook: Prehook,
    config_override: ConfigOverride | None = None,
    parent_scope: dict[str, object] | None = None,
    inputs: dict[str, object],
) -> _InvokeSetup:
    """Resolve all infrastructure shared between the sync and async paths.

    Pure Python — no I/O.  The caller is responsible for the actual
    ``agent.invoke``/``agent.ainvoke`` call.

    ``parent_scope`` carries the ambient scope (see ``jaz.scope``) snapshotted by
    the parent invoke, closed over as plain data by the agent-facing ``invoke``
    primitive rather than read from the ``_scope_var`` ContextVar at this leaf — mirroring
    how ``config`` propagates, so scope survives a thread/task hop (a bare
    ContextVar read inside a ``ThreadPoolExecutor`` worker would see an empty
    context). It is merged with same-context scope here; the resolved scope and the
    explicit kwargs are exposed as the separate ``scope`` / ``inputs`` fields on the
    returned setup (kept apart, not pre-merged — see ``_InvokeSetup``).
    """
    # Resolve the effective config for THIS invoke from the propagating override STACK (#727).
    # The effective config is folded from `ancestor_stack` below, once `repl_depth` is known —
    # nothing config-shaped is threaded in from the caller frame.
    # A local (non-propagating) `config_override` argument is folded as an
    # INNERMOST layer on a throwaway copy of the stack — so it wins over propagating overrides
    # (declaration-nesting: innermost wins, #727) and applies to THIS invoke only. It is NOT
    # pushed onto the live stack, so it does not propagate to sub-invokes (which inherit the
    # ancestor `ancestor_stack` — see get_subinvoke below). Both the public `invoke`/`ainvoke` and
    # the synthesized agent-facing `invoke` route through that helper, so `config_override` here is
    # always a plain `ConfigOverride`.
    # The agent-facing synthesized invoke accepts a `config_override` too (see get_subinvoke in
    # _subinvoke.py), so an agent can pass one to its own sub-invokes. To cap a subtree's
    # recursion depth it wraps the sub-invoke in
    # `with jaz.hooks.RecursionLimit(max_depth=...):` (local_hooks would NOT work — it doesn't
    # propagate, and RecursionLimit rejects that channel).
    ancestor_stack = get_config_stack()
    effective_stack = ancestor_stack
    if config_override is not None:
        effective_stack = ancestor_stack.push(config_override._as_layer())

    # Resolve ambient scope (see `jaz.scope`) for this invoke. `parent_scope`
    # arrives as plain data closed over by the agent-facing `invoke` (snapshot-and-bind,
    # like `config`); merge it with anything newly set via `with jaz.scope(...)`
    # in *this* same context (the `_scope_var` read is only a same-context
    # bridge). Inner same-context scope shadows the inherited parent scope.
    scoped = {**(parent_scope or {}), **(_scope_var.get() or {})}

    if scoped:
        # No process-boundary guard here: a "MainProcess only" check could not see the case it
        # names (under spawn the child's scope is empty, so this branch never runs) while firing
        # on the supported pattern of a worker that sets its own scope and invokes. What a child
        # actually inherits, and why nothing is checked, is in `jaz.scope`'s module comment.
        #
        # TODO(scope): support process boundaries via explicit serialization once there is a real
        # use case and a serialization story.
        # TODO(#475): config/hooks ride the same snapshot-and-bind path and may warrant analogous
        # process-boundary handling for consistency.

        # Conflict detection is loud, not silent: a value can't be both ambient
        # (via jaz.scope) and passed explicitly to this same invoke. Silent
        # precedence would let a distant scope override a local kwarg (or vice
        # versa) and ship the wrong behavior to production.
        conflicts = sorted(set(scoped) & set(inputs))
        if conflicts:
            # Show every conflicting name in the shadowing example, not just the
            # first — a multi-conflict case shouldn't read as if only one name is
            # the problem.
            shadow_example = ", ".join(f"{k}=..." for k in conflicts)
            raise ValueError(
                f"Inputs {conflicts} conflict: each is defined both via "
                f"jaz.scope(...) and passed as an explicit kwarg to jaz.invoke(). "
                f"Remove the explicit kwarg; to override the scoped value for this "
                f"call, shadow it with a nested `with jaz.scope({shadow_example}):` "
                f"block instead."
            )

    # Refused here, as the `AddInputs` effect refuses it, rather than only where `PythonREPL`
    # binds the inputs: a hook that supplies the result skips that, every hook sees the inputs
    # first, and another REPL need not check. The REPL checks again for a direct caller.
    # TODO(#1754): the name is `PythonREPL`'s; the REPL should declare what it reserves.
    if "__builtins__" in scoped or "__builtins__" in inputs:
        raise KeyEditDeniedError(
            "An invoke input cannot be named '__builtins__' — it holds the compiler sandbox; "
            "binding it would reopen import/attribute/builtins policy. Rename the input."
        )

    # `scoped` (resolved ambient scope) and `inputs` (explicit kwargs) are carried
    # SEPARATELY from here on (#727): the agent binds the merged namespace itself and
    # the prompt renders the two kinds from the two dicts, so no merge/re-split round
    # trip is needed. They are disjoint (the conflict check above guarantees it).

    # No task-length guard here: there is no distinguished task string to bound — the prompt is
    # an ordinary input (#538), subject to the same `max_invoke_input_length` per-input rendering
    # cap as every other input.

    prehook_output = prehook()

    # The framework itself imposes no recursion cap: the opt-in RecursionLimit hook does
    # (affordance-removal via DisableRecursion at the cap leaf; an over-cap Abort backstops the
    # public jaz.invoke path). With no RecursionLimit installed, nesting is unbounded. The prehook
    # tracks `repl_depth` (feeding event.depth) but carries no cap to guard against here.

    # Fold the effective config from the override stack (#727): authored base ⊕ applicable
    # propagating layers ⊕ the local override layer, in declaration order (later wins).
    # Folded here, once, at the top of the invoke: everything downstream — the Agent, every
    # event, every hook — receives a complete `Config`, so nothing re-asks whether an LLM is
    # configured. A missing one is refused by the fold.
    config = effective_stack.resolved

    invoke_id = str(uuid.uuid4())
    parent_invoke_id = prehook_output.parent_invoke_id
    parent_repl_iteration = prehook_output.parent_repl_iteration

    # Iteration limits are owned by the opt-in IterationLimit hook and cost / LLM-call budgets by
    # the opt-in BudgetPool hook (jaz.hooks); nothing here tracks either.

    # The prehook a child of THIS invoke would run under: depth+1. The Prehook's
    # parent_repl_iteration is updated by Agent at each iteration. There is no framework
    # recursion cap to carry — the RecursionLimit hook (if installed) enforces one via
    # effects on the child's InvokeEnter.
    child_prehook = Prehook(
        repl_depth=prehook_output.repl_depth + 1,
        parent_invoke_id=invoke_id,
    )

    # Build the recursive sub-invoke primitive the agent calls as the bare name `invoke`. Built
    # *unconditionally*: the framework imposes no recursion cap, and the one thing that withholds
    # the tool — a DisableRecursion effect at InvokeEnter (RecursionLimit at the cap) — is only
    # known once the Agent is running, so the Agent nulls it out then rather than choosing here
    # between two prebuilt variants (see `get_subinvoke` for why "the invoke tool without invoke"
    # is just `None`). User tool namespaces are ordinary inputs, not passed here. Passed as named
    # arguments rather than a shared kwargs dict: a `dict[str, Any]` splat costs the type checking
    # of these kwargs against `get_subinvoke`'s signature.
    subinvoke: Subinvoke = get_subinvoke(
        _invoke,
        child_prehook,
        # Snapshot of this level's scope, bound as data so it propagates to
        # nested invokes across thread/task hops (see `parent_scope` above).
        parent_scope=scoped,
        # Nested invokes inherit the ancestor override STACK (not the local override), so a
        # local ConfigOverride applies to this level only and does not propagate, while a
        # propagating `with ConfigOverride` (already on the stack) reaches them — composed,
        # last-wins, even across a raw worker-thread boundary (#727).
        config_stack=ancestor_stack,
    )

    # local_hooks is bound to this Agent's dispatcher only; nested jaz.invoke() calls construct
    # their own Agent and therefore don't see these hooks. See HookDispatcher class docstring for
    # the design rationale.
    local = tuple(local_hooks or ())

    agent = Agent(
        prehook=child_prehook,
        config=config,
        local_hooks=local,
    )

    return _InvokeSetup(
        agent=agent,
        repl_depth=prehook_output.repl_depth,
        invoke_id=invoke_id,
        parent_invoke_id=parent_invoke_id,
        parent_repl_iteration=parent_repl_iteration,
        subinvoke=subinvoke,
        child_prehook=child_prehook,
        inputs=inputs,
        scope=scoped,
    )


@contextmanager
def _established_config_stack() -> Iterator[None]:
    """Mark the propagating config stack "established" for this invoke's dynamic extent.

    The agent-facing synthesized ``invoke`` (``get_subinvoke``) re-bases the ancestor stack
    onto a raw worker thread by detecting an *unestablished* stack — the module-default stack a
    fresh worker sees because the ``_config_stack`` ContextVar didn't propagate. Marking the stack
    established here means a SAME-thread sub-invoke (where the ContextVar *did* propagate) is not
    mistaken for a worker and does not re-base (which would double-apply the ancestor). Twin of
    ``_established_hook_context``; ``push`` carries the flag through, so a nested ``with
    ConfigOverride`` stays established. No-op when already established (a nested same-thread invoke
    inherits the parent's established stack). Layers are unchanged; only the marker flips.
    """
    # "Established" is the #727 stack model's marker for "this stack is the ancestor for a subtree",
    # which is how a fresh worker thread tells a missing ContextVar from an empty one.
    stack = get_config_stack()
    if stack.established:
        yield
        return
    token = _set_config_stack(ConfigStack(stack.layers, established=True))
    try:
        yield
    finally:
        _reset_config_stack(token)


@contextmanager
def _established_hook_context() -> Iterator[None]:
    """Mark the propagating hook context "established" for this invoke's dynamic extent.

    Twin of ``_established_config_stack`` — the hook half of the same worker-detection mechanism.
    The agent-facing synthesized ``invoke`` re-bases the ancestor hook chain onto a raw worker
    thread by detecting an *unestablished* context (``not get_current_hooks().established``); a fresh
    worker sees that because the ``_hook_context`` ContextVar didn't propagate. Marking it here means
    a SAME-thread sub-invoke isn't mistaken for a worker and doesn't re-base (which would
    double-dispatch the ancestor chain). ``with_hook``/``with_local_active`` carry the flag through,
    so a mid-tree ``with Hook()`` stays established. No-op when already established.
    """
    ctx = get_current_hooks()
    if ctx.established:
        yield
        return
    token = _set_hook_context(ctx.with_established())
    try:
        yield
    finally:
        _reset_hook_context(token)


def _invoke(
    *,
    local_hooks: list[Hook] | None = None,
    prehook: Prehook,
    config_override: ConfigOverride | None = None,
    parent_scope: dict[str, object] | None = None,
    inputs: dict[str, object],
) -> object:
    """Same as public ``invoke`` but with the prehook specified explicitly.

    The return type is not a parameter: it is declared by a positional
    ``ReturnType(...)`` hook that dispatches on its own (prompt + type check), so ``agent`` /
    the REPL know nothing about it and this returns whatever the agent RETURNed as an untyped
    ``object``. The public ``invoke``'s typed overloads provide the static ``-> ReturnT``.
    """
    # Lower any t-string INPUT VALUES into text + sibling bindings (#538): after `task`
    # became an ordinary input, a t-string is passed as an input value rather than a
    # positional prompt, so the lowering walks the inputs dict instead of a single task.
    # Done here (not in the public entry) so the agent-facing synthesized invoke, which
    # routes straight to `_invoke`, gets t-string support for free. Plain values pass
    # through untouched.
    inputs = normalize_inputs(inputs)
    s = _build_invoke_setup(
        local_hooks=local_hooks,
        prehook=prehook,
        config_override=config_override,
        parent_scope=parent_scope,
        inputs=inputs,
    )
    # Publish the child prehook so the *public* jaz.invoke reached from inside this
    # agent (same thread — e.g. a human-written tool that calls jaz.invoke) inherits
    # this invocation's depth / invoke-calls accounting instead of minting a fresh
    # root. The token-based reset restores the parent's value, so same-thread nested
    # invokes stack correctly (child@2 -> child@3 -> ...). Mirrors the config/hook
    # ContextVars; a raw worker thread does not inherit it, which is why the agent facade
    # re-establishes it from its captured ancestor (see get_subinvoke).
    _prehook_token = _set_current_prehook(s.child_prehook)
    try:
        # NOTE: For static type checking, we need to call the overloads explicitly
        # TODO: Once pyright gets better, we no longer need separate calls because overloads will no longer be needed
        local = tuple(local_hooks or ())
        # Install the stdout proxy for the invoke tree (refcounted): during any exec,
        # sys.stdout routes to the exec's capture buffer, catching pprint / sys.stdout
        # writes / imported-function prints that the lexical print-swap misses. Outside
        # all invokes sys.stdout is left untouched. See repl/_stdout_proxy.py.
        # `_established_config_stack` / `_established_hook_context` mark the propagating override
        # stack and hook context established for this invoke's extent, so a same-thread sub-invoke
        # isn't mistaken for a raw worker (#727). The hook marker precedes `_activate_local_hooks`
        # so its `local_active` records onto the established context.
        with (
            _established_config_stack(),
            _established_hook_context(),
            _activate_local_hooks(local),
            _stdout_capturing(),
        ):
            return s.agent.invoke(
                subinvoke=s.subinvoke,
                depth=s.repl_depth,
                invoke_id=s.invoke_id,
                parent_invoke_id=s.parent_invoke_id,
                parent_repl_iteration=s.parent_repl_iteration,
                inputs=s.inputs,
                scope=s.scope,
            )
    finally:
        _reset_current_prehook(_prehook_token)


async def _ainvoke(
    *,
    local_hooks: list[Hook] | None = None,
    prehook: Prehook,
    config_override: ConfigOverride | None = None,
    parent_scope: dict[str, object] | None = None,
    inputs: dict[str, object],
) -> object:
    """Async version of ``_invoke`` (see it: the return type is a ``ReturnType`` hook)."""
    # See `_invoke`: lower any t-string input values into text + sibling bindings (#538).
    inputs = normalize_inputs(inputs)
    s = _build_invoke_setup(
        local_hooks=local_hooks,
        prehook=prehook,
        config_override=config_override,
        parent_scope=parent_scope,
        inputs=inputs,
    )

    _prehook_token = _set_current_prehook(s.child_prehook)
    try:
        # NOTE: For static type checking, we need to call the overloads explicitly
        # TODO: Once pyright gets better, we no longer need separate calls because overloads will no longer be needed
        local = tuple(local_hooks or ())
        # Install the stdout proxy for the invoke tree (refcounted): during any exec,
        # sys.stdout routes to the exec's capture buffer, catching pprint / sys.stdout
        # writes / imported-function prints that the lexical print-swap misses. Outside
        # all invokes sys.stdout is left untouched. See repl/_stdout_proxy.py.
        # `_established_config_stack` / `_established_hook_context` mark the propagating override
        # stack and hook context established for this invoke's extent, so a same-thread sub-invoke
        # isn't mistaken for a raw worker (#727). The hook marker precedes `_activate_local_hooks`
        # so its `local_active` records onto the established context.
        with (
            _established_config_stack(),
            _established_hook_context(),
            _activate_local_hooks(local),
            _stdout_capturing(),
        ):
            return await s.agent.ainvoke(
                subinvoke=s.subinvoke,
                depth=s.repl_depth,
                invoke_id=s.invoke_id,
                parent_invoke_id=s.parent_invoke_id,
                parent_repl_iteration=s.parent_repl_iteration,
                inputs=s.inputs,
                scope=s.scope,
            )
    finally:
        _reset_current_prehook(_prehook_token)


def _extract_config_override(
    args: tuple["Hook | ConfigOverride", ...],
) -> tuple[ConfigOverride | None, tuple[Hook, ...]]:
    """Split a positional ``ConfigOverride`` out of the leading invoke arguments.

    The public ``invoke`` / ``ainvoke`` accept a local ``ConfigOverride`` POSITIONALLY, in the
    same ``*local_config_hooks`` channel as hooks, so the keyword namespace belongs entirely to
    ``**inputs``. Returns ``(config_override, hooks)`` where ``hooks`` is the
    remaining positional list with any ``ConfigOverride`` removed (its order preserved).

    Because ``ConfigOverride`` is NOT a ``Hook``, this split is unambiguous and lossless — a
    ``ReturnType`` (which *is* a ``Hook``) stays in ``hooks`` and dispatches like any other.

    At most ONE ``ConfigOverride`` may be given. Unlike hooks — which are independent effects
    that compose in any order — a config override is a single scalar layer of Config-field
    overrides, so merging several would require a defined precedence order between them. Rather
    than pick one silently, we reject the ambiguity: ``TypeError`` on more than one, and the
    caller composes the fields into one override (or moves the shared part into a propagating
    ``with jaz.ConfigOverride(...):`` block).
    """
    overrides: list[ConfigOverride] = []
    hooks: list[Hook] = []
    for arg in args:
        if isinstance(arg, ConfigOverride):
            overrides.append(arg)
        else:
            hooks.append(arg)
    if len(overrides) > 1:
        raise TypeError(
            "invoke() got more than one ConfigOverride positional argument; an invoke takes a "
            "single local config override. Pass exactly one ConfigOverride (compose the "
            "overrides into one, or put the shared part in a `with jaz.ConfigOverride(...):` "
            "block)."
        )
    return (overrides[0] if overrides else None), tuple(hooks)


def _non_hook_positional_message(arg: object) -> str:
    """The ``TypeError`` text for a positional argument that is not a hook.

    A hook/config class needs instantiation, a prompt (``str``/t-string) belongs in a
    keyword input, a return type belongs in ``ReturnType(...)``, and data or tools are
    also keyword inputs.
    """
    # Each message states the rule AND the corrected call, because the mistake is a mental
    # model ("invoke takes a prompt") rather than a typo — a message that only said what is
    # wrong would leave the user to guess `task=`, which is an ordinary input name, not a
    # parameter they can find in the signature.
    kind = type(arg).__name__
    if isinstance(arg, (str, Template)):
        # `task` is an example input name, not a required keyword. A t-string cannot be
        # quoted back (its interpolations are already evaluated), so it gets a schematic form.
        if isinstance(arg, str):
            excerpt = arg if len(arg) <= 40 else arg[:37] + "..."
            corrected = f"invoke(task={excerpt!r})"
            if len(arg) > 40:
                corrected += " (prompt truncated)"
        else:
            corrected = 'invoke(task=t"...")'
        return (
            f"invoke() got the prompt as a positional argument ({kind}). The prompt is an "
            f"input, passed by keyword, e.g., {corrected}. invoke()'s positional arguments are "
            "hooks and an optional ConfigOverride; every input — the prompt, data, tools — "
            "is a keyword argument."
        )
    if isinstance(arg, type) and issubclass(arg, (Hook, ConfigOverride)):
        shown = arg.__name__
        return (
            f"invoke() got the {shown} class rather than an instance. Hooks and "
            f"ConfigOverride are passed as instances, e.g., invoke({shown}(...), task=...)."
        )
    if isinstance(arg, type) or get_origin(arg) is not None:
        shown = arg.__name__ if isinstance(arg, type) else repr(arg)
        return (
            f"invoke() got a type ({shown}) as a positional argument. A return type is "
            f"declared with the ReturnType hook, e.g., invoke(ReturnType({shown}), task=...). "
            "invoke()'s positional arguments are hooks and an optional ConfigOverride."
        )
    return (
        f"invoke() got a positional argument of type {kind}, which is not a hook. Pass data "
        f"and tools as keyword inputs, e.g., invoke(task=..., my_data=...). invoke()'s positional "
        "arguments are hooks and an optional ConfigOverride."
    )


def _resolve_invoke_hooks(
    local_hooks: tuple[Hook, ...],
) -> list[Hook]:
    """Resolve the effective hook list for an invoke.

    Shared by ``invoke``, ``ainvoke``, and the agent-facing ``invoke`` closure.
    ``ReturnType`` is **not special-cased**: it is a genuine, self-contained
    hook (it renders the ``<return_type>`` prompt and enforces the return type via effects), so it
    stays in the hook list and dispatches like any other. ``invoke`` / ``agent`` / the REPL know
    nothing about return types.

    Return-value validation (``ValidateReturn``) and REPL-code validation (``ValidateREPLCode``)
    are ordinary hooks too — positional, or propagating via ``with``. Nothing is synthesized from
    config: this resolver only guards and returns the hook list.

    Guards against a positional argument that is not a hook (the prompt passed positionally
    is the common one) and against more than one ``ReturnType`` (an invoke has a single
    return type).

    Semantic note: omitting ``ReturnType`` entirely means **no return-type contract** — the
    agent may ``return`` any value and it is returned as ``object``, unchecked. This is deliberately
    *not* the same as ``ReturnType(None)``, which still enforces a ``None`` return.
    """
    # The non-hook guard exists because #538's split — positional = hooks/ConfigOverride,
    # keyword = inputs — makes `invoke("do the thing")` a *type* error that nothing caught:
    # a stray positional stayed in the hook list all the way to `_activate_local_hooks`, which
    # failed with `AttributeError: 'str' object has no attribute 'setup'` from inside a
    # contextmanager, several frames from the call. That traceback names neither `invoke` nor
    # the argument, and it is the first thing a new user meets, since `invoke("prompt")` is the
    # shape every other agent library takes (and the shape an LLM writes from memory in its own
    # REPL — this resolver guards the agent-facing `invoke` too, where the message is fed
    # back as the observation it self-corrects from).
    for hook in local_hooks:
        if not isinstance(hook, Hook):
            raise TypeError(_non_hook_positional_message(hook))
    # The "no hook means no contract" reading is deliberate (#568): an absent hook means the
    # feature is off, consistent with every other flag that became a hook, rather than an implicit
    # `ReturnType(None)`.
    if sum(isinstance(h, ReturnType) for h in local_hooks) > 1:
        raise TypeError(
            "invoke() got more than one ReturnType(...) hook; an invoke has a single "
            "return type. Pass exactly one ReturnType(...)."
        )
    return list(local_hooks)


# Two overloads, keyed on whether a ``ReturnType(...)`` hook leads the positional arguments:
#   1. ``ReturnType[ReturnT]`` first  -> ``-> ReturnT`` (static ``x: T = jaz.invoke(ReturnType(T), ...)``)
#   2. no leading ``ReturnType``      -> ``-> object``
# There is no ``return_type=`` keyword: the return type is declared solely via a positional
# ``ReturnType(...)`` hook, and a stray ``return_type=`` lands in ``**inputs`` and binds as an
# ordinary agent input rather than typing the call. Static typing works because overload 1 keys on
# the *leading position* — the runtime extraction accepts a ``ReturnType`` anywhere among the
# positional hooks, but only the leading form is type-inferred.
#
# The variadic is ``*local_config_hooks: Hook | ConfigOverride``: a local ``ConfigOverride`` is
# passed POSITIONALLY in the same channel as hooks, and ``_extract_config_override`` splits it out
# at runtime. ConfigOverride is not a Hook, so the
# union is unambiguous and a leading ``ReturnType`` still infers as usual.
@overload
def invoke[ReturnT](
    return_type: ReturnType[ReturnT],
    /,
    *local_config_hooks: Hook | ConfigOverride,
    **inputs: object,
) -> ReturnT: ...


# Overload 2 is NOT made redundant by the concrete ``def invoke`` signature below: under
# ``@overload``, a type checker surfaces ONLY the overload signatures to callers (the
# implementation signature is invisible), so this overload is what types a call with no
# leading ``ReturnType``. Without it such calls would match no overload.
#
# Returns ``object``, not ``None``: this overload matches BOTH a call with no ``ReturnType`` at
# all (the agent may ``return`` any value, and it comes back unchecked) AND one that passes a
# ``ReturnType(...)`` in a NON-leading position (the extraction accepts it anywhere, so the
# runtime returns that declared type). Neither arm is ``None``-typed, so ``None`` would mistype
# a real ``int`` result and make legitimate use like ``x + 1`` a false type error. ``object`` is
# the honest common supertype. Lead with ``ReturnType(...)`` (overload 1) whenever the typed
# result is needed.
@overload
def invoke(
    *local_config_hooks: Hook | ConfigOverride,
    **inputs: object,
) -> object: ...


def invoke(
    *local_config_hooks: Hook | ConfigOverride,
    **inputs: object,
) -> object:
    """Run a REPL-based agent on the given inputs.

    A call ``invoke(name1=input1, name2=input2, ...)`` (with arbitrary names) shows the agent
    the description of each ``input_i`` along with its ``name_i``, and binds each ``input_i``
    to a variable ``name_i`` in the REPL. Inputs typically include the prompt (e.g. ``instructions="..."``
    or ``task="..."``), input data (e.g. ``df=pd.DataFrame(...)``), and tools (e.g.
    ``def web_search(...): ...; invoke(..., web_search=web_search)``).

    An input's description is its ``str()``, with these exceptions. A function is described by
    its signature and docstring; an object with ``__call__`` by those and its public members; a
    class by its constructor signature, docstring and public members; a module by its docstring
    and public names; and an object that defines neither ``__str__`` nor ``__repr__``, or a
    ``SimpleNamespace``, by its class docstring and public attributes. A value inside a list, a
    dict or a dataclass shows as its ``repr()``. So to change what the agent sees, write the
    docstring of a function, class, module or callable object (whose ``__str__`` is not used),
    define ``__str__`` on any other object, and define ``__repr__`` on one that goes inside a
    container.

    Pass a :class:`ReturnType` hook as the first positional arg (e.g.
    ``invoke(ReturnType(int), ...)``) to narrow the static return type and enforce it at
    runtime.

    To allow the agent to write async code with top-level ``await`` in its REPL, use
    :func:`ainvoke` instead.

    Args:
        **inputs: The inputs passed to the agent. Note that ``inputs`` do not propagate to
            sub-invokes — use the :func:`scope` context manager to pass in variables that
            propagate to sub-invokes.
        *local_config_hooks: Optional hooks and config override as leading positional
            arguments. If a :class:`ConfigOverride` is present, there can be at most one.
            Any hooks and config override passed as arguments to :func:`invoke` are *local*:
            they apply only to that :func:`invoke` and do not propagate to sub-invokes. Use a
            hook or config override as a context manager around :func:`invoke` for it to
            propagate to sub-invokes.

    Returns:
        The value the agent returned from the REPL session.
    """
    # Docs idiom (user decisions in the #1165 review, 2026-08-15). The *calling surface*
    # is imported bare in examples — ``from jaz import invoke, scope`` (`ainvoke` rides with
    # `invoke`): these are the verbs written in the flow of agent-calling code, and bare they
    # read as language primitives. Every other `jaz` name (configure/ConfigOverride, the hooks
    # namespaces) stays package-qualified in examples — the qualified spelling is itself the
    # signal of facility rather than primitive. Carve-outs: inline prose name-references stay
    # qualified (they double as fully qualified names); agent-REPL-side examples spell the bare
    # ``invoke``, which is the name the primitive is bound to inside an agent's own REPL;
    # multi-line runnable examples carry the import in-snippet, while one-line schematics rely
    # on the convention. A comment, not the docstring: this is authoring policy, not user-facing
    # API documentation.
    # Split a positional ConfigOverride out of the positional args, then resolve the hooks from
    # what remains. Order matters: a ReturnType is a Hook and stays in `local_hooks` for
    # _resolve_invoke_hooks to guard.
    config_override, local_hooks = _extract_config_override(local_config_hooks)

    # ReturnType is a self-contained hook (#568): it stays in local_hooks and drives its own
    # return-type contract, so this just guards + returns the hook list.
    hooks = _resolve_invoke_hooks(local_hooks)

    # Prefer the prehook of the enclosing invocation, if any. When this public
    # `invoke` is reached from inside a running agent on the same thread (e.g. via a
    # human-written tool that calls `jaz.invoke`), `_current_prehook` holds the child
    # prehook published by the enclosing `_invoke`, so this sub-invoke inherits the
    # tree's recursion cap and per-level invoke-calls budget instead of escaping them.
    # Unset (None) means a true top-level call — or a raw worker thread, where the
    # ContextVar didn't propagate and there is no closure fallback on the public path
    # (the documented gap) — so we mint a fresh root: recursion depth 1, no parent,
    # hence no parent-level invoke-calls budget and a no-op budget check at the top.
    prehook = get_active_prehook() or Prehook(repl_depth=1)

    # Call _invoke with prehook
    return _invoke(
        local_hooks=hooks,
        prehook=prehook,
        config_override=config_override,
        # Top-level invoke has no parent scope. Inputs travel as one ``inputs``
        # dict (not **splat), so an input named "parent_scope" stays a dict key
        # and can never collide with this internal plumbing parameter.
        parent_scope=None,
        inputs=inputs,
    )


# See `invoke` for the overload rationale: two overloads keyed on a leading ``ReturnType(...)`` —
# the return type via a positional ``ReturnType(...)`` and a local override via a positional
# ``ConfigOverride``; neither has a keyword form.
@overload
async def ainvoke[ReturnT](
    return_type: ReturnType[ReturnT],
    /,
    *local_config_hooks: Hook | ConfigOverride,
    **inputs: object,
) -> ReturnT: ...


# Overload 2 returns ``object``, not ``None`` — see `invoke`'s overload 2: a call with no
# ``ReturnType`` returns whatever the agent returned, and a non-leading ``ReturnType(...)`` returns
# its declared type; ``object`` is the sound common supertype. Lead with ``ReturnType(...)`` for a
# typed result.
@overload
async def ainvoke(
    *local_config_hooks: Hook | ConfigOverride,
    **inputs: object,
) -> object: ...


async def ainvoke(
    *local_config_hooks: Hook | ConfigOverride,
    **inputs: object,
) -> object:
    """Async version of :func:`invoke`. Runs the agent loop without blocking the event loop,
    enabling concurrent execution via ``asyncio.gather`` or similar patterns. It also lets the
    agent write async code with top-level ``await`` in its REPL, which :func:`invoke` does not.

    Arguments are identical to :func:`invoke` — see it for full documentation.
    """
    # Split the positional ConfigOverride out, then resolve the hooks from the remaining
    # positional arguments. See `invoke`.
    config_override, local_hooks = _extract_config_override(local_config_hooks)
    hooks = _resolve_invoke_hooks(local_hooks)

    prehook = get_active_prehook() or Prehook(repl_depth=1)

    return await _ainvoke(
        local_hooks=hooks,
        prehook=prehook,
        config_override=config_override,
        # Top-level invoke has no parent scope. Inputs travel as one ``inputs``
        # dict (not **splat), so an input named "parent_scope" stays a dict key
        # and can never collide with this internal plumbing parameter.
        parent_scope=None,
        inputs=inputs,
    )
