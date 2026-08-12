import logging
import multiprocessing
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, overload

from ._agent import Agent
from .config import (
    Config,
    ConfigOverride,
    ConfigOverrideByDepth,
    ConfigStack,
    _reset_config_stack,
    _set_config_stack,
    get_config_stack,
)
from .exceptions import (
    HookActivationError,
)
from .hooks import Hook, ReturnType
from .hooks.context import (
    HookContext,
    _reset_hook_context,
    _set_hook_context,
    get_current_hooks,
)
from .instantiate import language_of
from .library import Library, get_jaz_library
from .repl.stdout_proxy import capturing as _stdout_capturing
from .scope import _scope_var

# `normalize_inputs` lowers any t-string INPUT VALUES (a Template passed as an input,
# now that `task` is an ordinary input rather than a positional prompt — #538) into text
# plus sibling bindings. Nothing in this module annotates against `Template` anymore, so
# only the lowering helper is imported.
from .templates import normalize_inputs

logger = logging.getLogger(__name__)


# The prehook for the children of the currently-running invoke, published while an
# agent runs (see `_invoke` / `_ainvoke`). The *public* `jaz.invoke` reads this to
# inherit the enclosing invocation's depth / invoke-calls accounting when it is
# reached from inside a running agent (e.g. via a human-written tool that calls
# `jaz.invoke`), instead of minting a fresh root. Mirrors the config
# (`_config_stack`) and hook (`_hook_context`) ContextVars: live ContextVar is
# primary; unset means a true top-level call. On a raw worker thread the agent
# facade re-establishes this var from its captured ancestor before it calls in
# (see get_jaz_library in library/jaz.py, alongside the symmetric config/hook
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


@contextmanager
def _activate_local_hooks(hooks: tuple[Hook, ...]) -> Iterator[None]:
    """Run setup()/teardown() around a single invoke for its local hooks.

    Unlike the `with MyHook():` path, this does NOT register the hook *instances* on
    the contextvar — that is what keeps local hooks from propagating (being dispatched)
    to nested invokes. It records only their *ids* in ``HookContext.local_active`` (for
    collision detection, below) and drives the resource lifecycle, passing any exception
    that propagates out of the invoke to teardown() so stateful hooks (e.g. tracing) see
    the same exception they would via __exit__.

    Enforces the "one instance, one live scope" invariant (#533 / #540) before running
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
    current = get_current_hooks()
    propagating = current.hooks
    local_active = current.local_active
    seen: list[Hook] = []
    for hook in hooks:
        if any(hook is p for p in propagating):
            raise HookActivationError(
                f"{type(hook).__name__} instance is active via both `with` and "
                "local_hooks at once; this would run setup()/teardown() twice and is "
                "redundant (a `with`-active hook already propagates to nested invokes). "
                "Pass it via exactly one path."
            )
        if id(hook) in local_active:
            raise HookActivationError(
                f"{type(hook).__name__} instance is already active via local_hooks in "
                "an enclosing invoke; activating it again would run setup()/teardown() "
                "twice while its local lifecycle is still live. Use one scope at a time."
            )
        if any(hook is s for s in seen):
            raise HookActivationError(
                f"{type(hook).__name__} instance appears more than once in "
                "local_hooks; this would run setup()/teardown() twice. List each hook "
                "instance at most once."
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
        # Local hooks run setup() DIRECTLY here — NOT via ``Hook.__enter__`` (the ``with``
        # path) and NOT at all for baseline hooks. ``RecursionLimit`` relies on this exact
        # routing to reject the local-hooks channel (its ``setup()`` raises unless reached
        # through ``__enter__``); if this ever went through ``__enter__``, that rejection would
        # silently stop firing and the under-cap footgun would return. Keep it direct.
        for hook in hooks:
            hook.setup()
        exc: BaseException | None = None
        try:
            yield
        except BaseException as e:  # noqa: BLE001 - captured to forward to teardown, then re-raised
            exc = e
            raise
        finally:
            # Reverse order mirrors nested-context-manager teardown semantics.
            for hook in reversed(hooks):
                try:
                    hook.teardown(exc)
                except Exception:
                    logger.exception(
                        "Local hook %s teardown failed", type(hook).__name__
                    )
    finally:
        _reset_hook_context(token)


# Common misspellings or misuses of invoke() kwargs.
# Keys are invalid kwarg names that users may pass; values are suggestions.
#
# Scope: this guard fires only at the *public* invoke()/ainvoke() boundary. The
# in-REPL nested jaz.invoke (the closure in library/jaz.py) routes through
# _invoke and deliberately does NOT consult this map, so an agent that passes a
# removed kwarg (e.g. additional_repls=...) inside a REPL binds it as an ordinary
# input variable instead of getting this guidance. Extending the check there was
# considered and declined: the nested closure's **inputs are agent-supplied task
# data, where names like `model`/`temperature`/`repl` can legitimately be inputs,
# so raising on them would be a false positive on real data.
_REPL_HINT = (
    "from jaz.repl.python_repl import PythonREPL; jaz.configure(repl=PythonREPL({}))"
)
_LLM_HINT = "from jaz.providers.openai import OpenAILLM; jaz.{}(llm=OpenAILLM({}))"

_INVOKE_KWARG_TYPOS: dict[str, str] = {
    # Config knobs are no longer invoke() kwargs: set them via a local
    # ConfigOverride (or jaz.ConfigOverride(...) / jaz.configure(...)).
    "max_iterations": "with jaz.hooks.IterationLimit(max_iterations=...): jaz.invoke(...)",
    "max_repl_iterations": "with jaz.hooks.IterationLimit(max_iterations=...): jaz.invoke(...)",
    "max_depth": "with jaz.hooks.RecursionLimit(max_depth=...): jaz.invoke(...)",
    "max_cost_budget": "with jaz.hooks.BudgetPool(cost_budget=...): jaz.invoke(...)",
    "max_llm_calls_budget": "with jaz.hooks.BudgetPool(calls_budget=...): jaz.invoke(...)",
    # Kept (repointed, not deleted) now that the per-level nested-invoke cap is gone:
    # the hint is most valuable right after a removal, when callers still reach for the
    # old name and the generic TypeError would name no replacement.
    "max_invoke_calls": "removed (YAGNI — the per-level nested-invoke cap): use "
    "with jaz.hooks.BudgetPool(calls_budget=...) to bound work, or "
    "with jaz.hooks.RecursionLimit(max_depth=...) to bound depth",
    "max_repl_invoke_calls": "removed (YAGNI — the per-level nested-invoke cap): use "
    "with jaz.hooks.BudgetPool(calls_budget=...) to bound work, or "
    "with jaz.hooks.RecursionLimit(max_depth=...) to bound depth",
    "context_window_fraction": "with jaz.hooks.ContextWindow(context_window_fraction=...): jaz.invoke(...)",
    # Spelled with the import, because neither name is reachable from the `jaz` top level and
    # these strings are printed to someone who just hit a typo — a hint they cannot paste is
    # only half a hint.
    "allowed_imports": _REPL_HINT.format("allowed_imports=[...]"),
    "allowed_attributes": _REPL_HINT.format("allowed_attributes=[...]"),
    "repls": "jaz.ConfigOverride(repl=...)",
    "repl": "jaz.ConfigOverride(repl=...)",
    "additional_repls": "jaz.ConfigOverride(repl=...) (multiple REPLs are no longer supported)",
    "model": _LLM_HINT.format("configure", "model=..."),
    "temperature": _LLM_HINT.format("ConfigOverride", "temperature=..."),
    "max_tokens": _LLM_HINT.format("ConfigOverride", "max_tokens=..."),
    # task_name moved off the core signature onto the per-invoke blackboard: it is
    # now hook metadata (consumed by WorkflowReplay / tracing), seeded by a
    # MetaData carrier hook rather than a kwarg.
    "task_name": "jaz.invoke(jaz.hooks.MetaData(task_name=...), ...) (or with jaz.hooks.MetaData(task_name=...): ...)",
    # input_descriptions was replaced by value-attached descriptions: attach a
    # permanent description to a value with jaz.describe(value, text), or control
    # per-call rendering (relabel/hide) with jaz.Display(value, text|None).
    # Leads with `describe` because `Display` is demoted (outside __all__, warns on use):
    # a rejection message naming it first sends the caller from an error straight into a
    # warning. Kept as a flagged second option since it is the only one that can hide.
    "input_descriptions": "jaz.describe(value, text), or jaz.Display(value, text|None) per input (experimental — warns on use)",
    # local_hooks is no longer a keyword: hooks are leading POSITIONAL arguments now, so
    # a stray `local_hooks=[...]` would silently become an ordinary input (pyright can't
    # catch it — any keyword is a valid `**inputs`). Redirect it to the positional form.
    "local_hooks": "pass hooks positionally: jaz.invoke(MyHook(), OtherHook(), ..., task=...)",
    # The return_type= keyword was removed (#528): the return type is now declared with a
    # positional ReturnType(...) hook. The keyword lived on for a while as a thin shim; that
    # shim is gone, so a stray `return_type=` would silently become an ordinary `**inputs`
    # variable. Guard it and point at the hook. Static typing (`x: T = jaz.invoke(...)`) still
    # works when ReturnType(T) is the FIRST positional argument (see the typed overload).
    "return_type": "pass a ReturnType(...) hook positionally (first, for static typing): "
    "jaz.invoke(ReturnType(T), ..., task=...)",
    # config_override= is no longer a keyword: a local ConfigOverride is now passed POSITIONALLY,
    # in the same `*local_config_hooks` channel as hooks (it composes with them the same way — a
    # local, non-propagating override for this one invoke). A stray `config_override=` would
    # otherwise become an ordinary `**inputs` variable, so guard it and point at the positional
    # form. (The agent-facing synthesized jaz.invoke still takes config_override= — it forwards to
    # the internal plumbing keyword — but that surface routes through _invoke, not this guard.)
    "config_override": "pass the ConfigOverride positionally: "
    "jaz.invoke(ConfigOverride(...), ..., task=...)",
    # Return-value and REPL-input validation moved to positional hooks (#528). The old
    # return_validator=/repl_input_validator= keywords are gone; a stray one would silently
    # become an ordinary `**inputs` variable, so guard it and point at the hook. (A
    # process-wide input validator is a propagating `with ValidateREPLInput(fn):` context
    # manager — the config-level `repl_input_validator` field was removed too.)
    "return_validator": "pass a ValidateReturn(fn) hook positionally: "
    "jaz.invoke(ReturnType(T), jaz.hooks.ValidateReturn(fn), ..., task=...)",
    "repl_input_validator": "pass a ValidateREPLInput(fn) hook positionally: "
    "jaz.invoke(jaz.hooks.ValidateREPLInput(fn), ..., task=...)  "
    "(or process-wide via `with jaz.hooks.ValidateREPLInput(fn):`)",
}


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
        # worker thread, where the _hook_context ContextVar doesn't propagate: get_jaz_library
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

    Does not carry ``return_type`` — it stays in the generic ``_invoke[ReturnT]`` /
    ``_ainvoke[ReturnT]`` callers so pyright can track ``ReturnT`` through to
    ``agent.invoke`` / ``agent.ainvoke``. (Return-value validation is no longer a
    parameter at all: it flows through the ``ValidateReturn`` hook — #528.)
    """

    agent: Agent
    repl_depth: int
    invoke_id: str
    parent_invoke_id: str | None
    parent_repl_iteration: int | None
    jaz_library: Library | None
    # Cap-leaf variant of `jaz_library` — same opted-in `jaz.*` helpers minus `jaz.invoke`
    # (#635). The Agent binds this instead of `jaz_library` when a DisableRecursion effect
    # fires at InvokeEnter. `None` when nothing is opted in (cap leaf then gets no library).
    jaz_library_no_invoke: Library | None
    config: Config
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
    the parent invoke, bound as plain data on the agent-facing ``jaz`` object
    rather than read from the ``_scope_var`` ContextVar at this leaf — mirroring
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
    # INNERMOST layer on a throwaway copy of the stack — so it wins over propagating overrides and
    # per-depth layers (declaration-nesting: innermost wins, #727) and applies to THIS invoke
    # only. It is NOT pushed onto the live stack, so it does not propagate to sub-invokes (which
    # inherit the ancestor `ancestor_stack` — see get_jaz_library below). A local
    # `ConfigOverrideByDepth` argument never reaches here: `_extract_config_override` rejects one
    # passed positionally with a TypeError (see its docstring), precisely because locally it would
    # collapse to a degenerate depth-gated plain override. Both the public `invoke`/`ainvoke` and
    # the synthesized agent-facing `invoke` route through that helper, so `config_override` here is
    # always a plain `ConfigOverride`.
    # The agent-facing synthesized invoke accepts a `config_override` too (see get_jaz_library in
    # library/jaz.py), so an agent can pass one to its own sub-invokes. To cap a subtree's
    # recursion depth it wraps the sub-invoke in
    # `with jaz.hooks.RecursionLimit(max_depth=...):` (the cap is no longer a config key;
    # local_hooks would NOT work — it doesn't propagate, and RecursionLimit rejects that channel).
    ancestor_stack = get_config_stack()
    effective_stack = ancestor_stack
    if config_override is not None:
        effective_stack = ancestor_stack.push(config_override._as_layer())

    # Resolve ambient scope (see `jaz.scope`) for this invoke. `parent_scope`
    # arrives as plain data bound on the agent-facing `jaz` (snapshot-and-bind,
    # like `config`); merge it with anything newly set via `with jaz.scope(...)`
    # in *this* same context (the `_scope_var` read is only a same-context
    # bridge). Inner same-context scope shadows the inherited parent scope.
    scoped = {**(parent_scope or {}), **(_scope_var.get() or {})}

    if scoped:
        # Process boundaries are forbidden for now: the bound `jaz` (and many
        # scoped values — a live APIClient, a connection pool) don't survive the
        # pickle into a child process, so scope would be either silently empty or
        # a cryptic PicklingError. Refuse loudly instead of shipping that footgun.
        # TODO(scope): support process boundaries via explicit serialization once
        # there is a real use case and a serialization story.
        # TODO(#475): config/hooks ride the same snapshot-and-bind path and may
        # warrant analogous process-boundary guards for consistency.
        if multiprocessing.current_process().name != "MainProcess":
            raise RuntimeError(
                "jaz.scope does not cross process boundaries: jaz.invoke() with an "
                "active scope was called outside the main process (e.g. in a "
                "multiprocessing / ProcessPoolExecutor worker). Set the scope "
                "inside the worker, or thread the value explicitly as a kwarg."
            )

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

    # `scoped` (resolved ambient scope) and `inputs` (explicit kwargs) are carried
    # SEPARATELY from here on (#727): the agent binds the merged namespace itself and
    # the prompt renders the two kinds from the two dicts, so no merge/re-split round
    # trip is needed. They are disjoint (the conflict check above guarantees it).

    # The former `max_task_length` guard is gone (#538): there is no distinguished task
    # string to bound — the prompt is now an ordinary input, subject to the same
    # `max_input_length` per-input rendering cap as every other input, not a separate
    # length ceiling here.

    prehook_output = prehook()

    # The recursion cap is no longer enforced by the framework: it moved to the opt-in
    # RecursionLimit hook (affordance-removal via DisableRecursion at the cap leaf; an
    # over-cap Abort backstops the public jaz.invoke path). With no RecursionLimit installed,
    # nested jaz.invoke is unbounded. The prehook still tracks `repl_depth` (feeding
    # event.depth and depth_overrides) but no longer carries a cap to guard against here.

    # Fold the effective config for this ABSOLUTE recursion depth from the override stack (#727):
    # base ⊕ global depth field ⊕ applicable propagating layers ⊕ the local override layer, in
    # declaration order (later wins). A depth with no per-depth source returns the depth-less
    # resolved config (identity to _default_config when the stack is empty), so per-depth
    # mutations can't leak back into the base Config (the fold works on a deep copy).
    config = effective_stack.resolve_for_depth(prehook_output.repl_depth)

    invoke_id = str(uuid.uuid4())
    parent_invoke_id = prehook_output.parent_invoke_id
    parent_repl_iteration = prehook_output.parent_repl_iteration

    # Iteration limits are owned by an opt-in IterationLimit (activated via `with`).
    # Cost / LLM-calls budget tracking + enforcement lives in the opt-in BudgetPool
    # (jaz.hooks). jaz.invoke no longer creates a CostTracker.

    # The single REPL language for this invocation, named by the registry rather than stored on
    # the config: the config holds the configured REPL itself now.
    agent_repl = language_of(config.repl)

    # The prehook a child of THIS invoke would run under: depth+1. The Prehook's
    # parent_repl_iteration is updated by Agent at each iteration. There is no framework
    # recursion cap to carry — the RecursionLimit hook (if installed) enforces one via
    # effects on the child's InvokeEnter.
    child_prehook = Prehook(
        repl_depth=prehook_output.repl_depth + 1,
        parent_invoke_id=invoke_id,
    )

    # Build the framework `jaz_library` — the synthesized JAZ library, whose sole member is the
    # recursive `jaz.invoke` tool. Bound *unconditionally* now: the framework imposes no
    # recursion cap. Two variants are built and handed to the Agent, which picks between them
    # when it honors a DisableRecursion effect at InvokeEnter (a RecursionLimit at the cap):
    #   - `jaz_library`          — the full surface, incl. the recursive `jaz.invoke` tool.
    #   - `jaz_library_no_invoke`— the cap-leaf surface: WITHOUT `jaz.invoke`. Since `jaz.invoke`
    #     is now the namespace's only member, this variant has no members and is always `None`
    #     (get_jaz_library's `if not all_tools: return None`), so an at-cap agent has no `jaz`
    #     library at all. Framework helpers (ReturnType/Display/…) reach an agent as ordinary
    #     inputs the host passes, not via `jaz.*` (#635 kept the leaf variant only for exports,
    #     which are gone).
    # Both are built here (not in Agent) because the DisableRecursion decision is only known at
    # InvokeEnter — after Agent already holds both. User tool namespaces are ordinary inputs bound
    # via `__jaz_get__` (see library_as_input.md), not passed here.
    new_prehook = child_prehook
    # The `config.allow_config_hooks_in_subinvoke` read below intentionally uses the overridden
    # `config`: it describes how THIS agent sees its sub-invoke surface (full-vs-minimal invoke
    # signature), not how the sub-invocations execute. A local ConfigOverride of it is therefore
    # scoped to this level only (non-propagating). `config_stack=` below is the *ancestor* stack
    # (WITHOUT this invoke's local override layer) precisely because it controls how the nested
    # invokes execute, which IS supposed to be non-propagating — the closure re-bases this stack
    # onto a raw worker thread (#727).
    _jaz_library_kwargs: dict[str, Any] = dict(
        allow_config_hooks_in_subinvoke=config.allow_config_hooks_in_subinvoke,
        # Snapshot of this level's scope, bound as data so it propagates to
        # nested invokes across thread/task hops (see `parent_scope` above).
        parent_scope=scoped,
        # Nested invokes inherit the ancestor override STACK (not the local override), so a
        # local ConfigOverride applies to this level only and does not propagate, while a
        # propagating `with ConfigOverride` (already on the stack) reaches them — composed,
        # last-wins, even across a raw worker-thread boundary (#727).
        config_stack=ancestor_stack,
    )
    jaz_library: Library | None = get_jaz_library(
        _invoke, new_prehook, include_invoke=True, **_jaz_library_kwargs
    )
    # The cap-leaf variant is now ALWAYS None: with `jaz.invoke` the namespace's only member,
    # `get_jaz_library(include_invoke=False)` has no tools and short-circuits to None
    # unconditionally. Written as a literal rather than that guaranteed-None call so the reader
    # doesn't wonder whether the second build can ever differ. The `jaz_library_no_invoke`
    # plumbing (through _InvokeSetup / Agent) is retained deliberately: it lets a future PR
    # re-introduce a helper-only cap-leaf surface without re-threading five call sites — and
    # `get_jaz_library` keeps its `include_invoke` parameter for that.
    jaz_library_no_invoke: Library | None = None

    # Invoke agent in REPL.
    # local_hooks is bound to this Agent's dispatcher only; nested
    # jaz.invoke() calls construct their own Agent and therefore don't
    # see these hooks. See HookDispatcher class docstring for the
    # design rationale. (Cost/LLM-call budget tracking now lives in the
    # opt-in BudgetPool, so jaz.invoke no longer manages a CostTracker.)
    local = tuple(local_hooks or ())

    agent = Agent(
        repl=agent_repl,
        prehook=new_prehook,
        config=config,
        local_hooks=local,
    )

    return _InvokeSetup(
        agent=agent,
        repl_depth=prehook_output.repl_depth,
        invoke_id=invoke_id,
        parent_invoke_id=parent_invoke_id,
        parent_repl_iteration=parent_repl_iteration,
        jaz_library=jaz_library,
        jaz_library_no_invoke=jaz_library_no_invoke,
        config=config,
        child_prehook=child_prehook,
        inputs=inputs,
        scope=scoped,
    )


@contextmanager
def _established_config_stack() -> Iterator[None]:
    """Mark the propagating config stack "established" for this invoke's dynamic extent (#727).

    The agent-facing synthesized ``jaz.invoke`` (``get_jaz_library``) re-bases the ancestor stack
    onto a raw worker thread by detecting an *unestablished* stack — the module-default stack a
    fresh worker sees because the ``_config_stack`` ContextVar didn't propagate. Marking the stack
    established here means a SAME-thread sub-invoke (where the ContextVar *did* propagate) is not
    mistaken for a worker and does not re-base (which would double-apply the ancestor). Twin of
    ``_established_hook_context``; ``push`` carries the flag through, so a nested ``with
    ConfigOverride`` stays established. No-op when already established (a nested same-thread invoke
    inherits the parent's established stack). Layers are unchanged; only the marker flips.
    """
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
    """Mark the propagating hook context "established" for this invoke's dynamic extent (#727).

    Twin of ``_established_config_stack`` — the hook half of the same worker-detection mechanism.
    The agent-facing synthesized ``jaz.invoke`` re-bases the ancestor hook chain onto a raw worker
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

    The return type is no longer a parameter (#568): it is declared by a positional
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
    # re-establishes it from its captured ancestor (see get_jaz_library).
    _prehook_token = _set_current_prehook(s.child_prehook)
    try:
        # NOTE: For static type checking, we need to call the overloads explicitly
        # TODO: Once pyright gets better, we no longer need separate calls because overloads will no longer be needed
        local = tuple(local_hooks or ())
        # Install the stdout proxy for the invoke tree (refcounted): during any exec,
        # sys.stdout routes to the exec's capture buffer, catching pprint / sys.stdout
        # writes / imported-function prints that the lexical print-swap misses. Outside
        # all invokes sys.stdout is left untouched. See repl/stdout_proxy.py.
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
                jaz_library=s.jaz_library,
                jaz_library_no_invoke=s.jaz_library_no_invoke,
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
    """Async version of ``_invoke`` (see it: the return type is a ``ReturnType`` hook now, #568)."""
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
        # all invokes sys.stdout is left untouched. See repl/stdout_proxy.py.
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
                jaz_library=s.jaz_library,
                jaz_library_no_invoke=s.jaz_library_no_invoke,
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
    same ``*local_config_hooks`` channel as hooks — the ``config_override=`` keyword was removed
    so the keyword namespace belongs entirely to ``**inputs`` (mirroring the ``local_hooks`` and
    ``return_type`` migrations). Returns ``(config_override, hooks)`` where ``hooks`` is the
    remaining positional list with any ``ConfigOverride`` removed (its order preserved).

    Because ``ConfigOverride`` is NOT a ``Hook``, this split is unambiguous and lossless — a
    ``ReturnType`` (which *is* a ``Hook``) stays in ``hooks`` for ``_extract_return_type`` to
    pull later.

    At most ONE ``ConfigOverride`` may be given. Unlike hooks — which are independent effects
    that compose in any order — a config override is a single scalar layer of Config-field
    overrides, so merging several would require a defined precedence order between them. Rather
    than pick one silently, we reject the ambiguity: ``TypeError`` on more than one, and the
    caller composes the fields into one override (or moves the shared part into a propagating
    ``with jaz.ConfigOverride(...):`` block).

    A ``ConfigOverrideByDepth`` (a ``ConfigOverride`` subclass) is REJECTED here rather than
    accepted like its base class: passed positionally it is *local* (non-propagating), so it
    reaches only this one invoke, where only the partial for this invoke's OWN depth could ever
    apply. That collapses it to a depth-gated plain override — almost never the intent, since
    per-depth config is meaningful precisely when it PROPAGATES to sub-invokes at other depths.
    We raise ``TypeError`` pointing at the propagating ``with jaz.ConfigOverrideByDepth(...):``
    form instead of silently applying a degenerate slice of it.
    """
    overrides: list[ConfigOverride] = []
    hooks: list[Hook] = []
    for arg in args:
        # ConfigOverrideByDepth IS-A ConfigOverride, so this reject must precede the base check.
        if isinstance(arg, ConfigOverrideByDepth):
            raise TypeError(
                "invoke() got a ConfigOverrideByDepth positional argument. A per-depth override "
                "only makes sense when it propagates to sub-invokes at other depths; passed "
                "locally to one invoke, only that invoke's own depth would apply — a degenerate "
                "depth-gated plain override. Use a propagating `with jaz.ConfigOverrideByDepth(...):` "
                "block instead, or pass a plain `ConfigOverride(...)` positionally."
            )
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


def _resolve_invoke_hooks(
    local_hooks: tuple[Hook, ...],
) -> list[Hook]:
    """Resolve the effective hook list for an invoke.

    Shared by ``invoke``, ``ainvoke``, and the agent-facing ``jaz.invoke`` closure.
    ``ReturnType`` is **no longer special-cased**: as of #568 it is a genuine, self-contained
    hook (it renders the ``<return_type>`` prompt and enforces the return type via effects), so it
    stays in the hook list and dispatches like any other. ``invoke`` / ``agent`` / the REPL no
    longer know anything about return types.

    Return-value validation (``ValidateReturn``) and REPL-input validation (``ValidateREPLInput``)
    are ordinary positional hooks too. The ``return_validator=`` / ``repl_input_validator=``
    keywords were removed (#528), and so was the config-level input validator
    (``jaz.configure(repl_input_validator=fn)`` + ``max_repl_input_validation_failures``): a
    propagating ``ValidateREPLInput`` context manager is the direct replacement, so this resolver
    no longer synthesizes any hook from config — it just guards and returns the hook list.

    Guards against more than one ``ReturnType`` (an invoke has a single return type).

    Semantic note (#568): omitting ``ReturnType`` entirely means **no return-type contract** — the
    agent may ``return`` any value and it is returned as ``object``, unchecked. This is deliberately
    *not* the same as ``ReturnType(None)``, which still enforces a ``None`` return. (Pre-#568 the
    default ``return_type=None`` was threaded to the REPL, which rejected any non-``None`` return,
    so the two were equivalent; now "no hook" means "no contract", consistent with every other
    migrated #568 flag — a deliberate decision, not an oversight.)
    """
    if sum(isinstance(h, ReturnType) for h in local_hooks) > 1:
        raise TypeError(
            "invoke() got more than one ReturnType(...) hook; an invoke has a single "
            "return type. Pass exactly one ReturnType(...)."
        )
    return list(local_hooks)


# Two overloads, keyed on whether a ``ReturnType(...)`` hook leads the positional arguments:
#   1. ``ReturnType[ReturnT]`` first  -> ``-> ReturnT`` (static ``x: T = jaz.invoke(ReturnType(T), ...)``)
#   2. no leading ``ReturnType``      -> ``-> object``
# The ``return_type=`` keyword shim is GONE (#528): the return type is declared solely via a
# positional ``ReturnType(...)`` hook. A stray ``return_type=`` now trips the ``_INVOKE_KWARG_TYPOS``
# guard (it lands in ``**inputs``) rather than silently typing the call. Static typing survives
# because overload 1 keys on the *leading position* — the runtime extraction still accepts a
# ``ReturnType`` anywhere among the positional hooks, but only the leading form is type-inferred.
#
# The variadic is ``*local_config_hooks: Hook | ConfigOverride``: a local ``ConfigOverride`` is
# passed POSITIONALLY in the same channel as hooks (the ``config_override=`` keyword was removed),
# and ``_extract_config_override`` splits it out at runtime. ConfigOverride is not a Hook, so the
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
# all (runtime returns ``None``) AND one that passes a ``ReturnType(...)`` in a NON-leading
# position (the extraction accepts it anywhere, so the runtime returns that declared type — an
# arbitrary value). ``None`` would be precise for the first but a lie for the second (typing a
# real ``int`` result as ``None``, so legitimate use like ``x + 1`` becomes a false type error).
# ``object`` is the honest common supertype of both: sound in either case, at the cost of the
# no-return call no longer inferring the precise ``None`` (a result that is normally discarded
# anyway). Lead with ``ReturnType(...)`` (overload 1) whenever the typed result is needed.
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
    ``def web_search(...): ...; jaz.invoke(..., web_search=web_search)``).

    Pass a :class:`ReturnType` hook as the first positional arg (e.g.
    ``jaz.invoke(ReturnType(int), ...)``) to narrow the static return type and enforce it at
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
    # Catch common kwarg misspellings before they silently become input variables
    for key in inputs:
        if key in _INVOKE_KWARG_TYPOS:
            raise TypeError(
                f"invoke() got unexpected keyword argument '{key}'. "
                f"Did you mean: {_INVOKE_KWARG_TYPOS[key]}"
            )

    # Split a positional ConfigOverride (the config_override= keyword was removed) out of the
    # positional args, then resolve the return type / hooks from what remains. Order matters: a
    # ReturnType is a Hook and stays in `local_hooks` for _resolve_invoke_hooks to extract.
    # (The protected-key guard's ancestor policy is no longer captured here — it is computed inside
    # `_build_invoke_setup`, depth-resolved at the parent's depth. #826.)
    config_override, local_hooks = _extract_config_override(local_config_hooks)

    # ReturnType is a self-contained hook (#568): it stays in local_hooks and drives its own
    # return-type contract, so this just guards + returns the hook list. The return_type= /
    # validator keywords and the config-level input-validator shim were all removed (#528).
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


# See `invoke` for the overload rationale: two overloads keyed on a leading ``ReturnType(...)``,
# with the ``return_type=`` and ``config_override=`` keyword shims removed (#528) — the return
# type via a positional ``ReturnType(...)`` and a local override via a positional ``ConfigOverride``.
@overload
async def ainvoke[ReturnT](
    return_type: ReturnType[ReturnT],
    /,
    *local_config_hooks: Hook | ConfigOverride,
    **inputs: object,
) -> ReturnT: ...


# Overload 2 returns ``object``, not ``None`` — see `invoke`'s overload 2: it also matches a
# non-leading ``ReturnType(...)`` (runtime returns that declared type), which ``None`` would
# mistype; ``object`` is the sound common supertype. Lead with ``ReturnType(...)`` for a typed result.
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
    for key in inputs:
        if key in _INVOKE_KWARG_TYPOS:
            raise TypeError(
                f"ainvoke() got unexpected keyword argument '{key}'. "
                f"Did you mean: {_INVOKE_KWARG_TYPOS[key]}"
            )

    # Split the positional ConfigOverride out (config_override= keyword removed, #528), then
    # resolve the return type / hooks from the remaining positional hooks. See `invoke`.
    # (The guard's ancestor policy is computed inside `_build_invoke_setup`, not here. #826.)
    config_override, local_hooks = _extract_config_override(local_config_hooks)

    # No return_type= keyword shim anymore (#528) — pass None; the type is declared solely by a
    # positional ReturnType(...) hook, which _resolve_invoke_hooks extracts from local_hooks.
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
