"""The agent-facing recursive sub-invoke primitive, bound in the REPL as bare ``invoke``.

This module builds the single callable an agent uses to spawn a sub-agent. It is NOT the
public host-side :func:`jaz.invoke` — it is a per-invoke closure over that primitive which
re-bases the ancestor config / hook / prehook context across thread boundaries before
delegating.
"""

# Formerly ``jaz.library.jaz.get_jaz_library``, which wrapped this callable in a ``Library``
# named "JAZ" whose sole root module was ``jaz``, so the agent called ``jaz.invoke(...)``. Once
# #635 stopped auto-binding the framework helpers (``ReturnType`` / ``Display`` / ``Library`` /
# …) under ``jaz.*`` and made them ordinary inputs, that namespace had exactly one member and
# the wrapper bought nothing: a one-member "library" whose prompt card advertised a "Library
# name", a "Description" and a "Top-level module" for what is really a single function. The
# namespace is gone — the primitive binds as the bare name ``invoke`` — and with it the
# ``Library`` dependency, so ``Library`` is now purely a user-facing tool-namespace builder
# (see ``design/design_features/library_as_input.md``).
#
# Consequences of dropping the namespace, both deliberate:
#   - The prompt block is ``<invoke type="function">`` rather than ``<jaz_library>``, and
#     renders one catalog entry instead of a library card. The library card's "Description"
#     line — a blurb about the JAZ framework — went with it, replaced by a one-line lead-in
#     naming what the function is for.
#   - The bare name is far more collidable than ``jaz`` was. An input literally named
#     ``invoke`` shadows the primitive in the REPL (inputs bind after it — see
#     ``PythonREPL.initialize``). That was already true of an input named ``jaz``; it is now
#     merely likelier. Left as shadowing rather than an error because an input name is the
#     host's call, and refusing one would be a new failure mode on a surface that has never
#     had one.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from jaz.config import ConfigOverride, ConfigStack
    from jaz.hooks import Hook
    from jaz.invoke import Prehook


# The recursive sub-invoke primitive as the rest of the framework passes it around: an opaque
# callable, or ``None`` when this invoke may not recurse (a ``DisableRecursion`` effect at the
# RecursionLimit cap leaf). Spelled as an alias so the ~10 signatures that merely forward it
# say what they carry; nothing introspects it beyond binding and rendering.
type InvokeTool = Callable[..., object]


def get_invoke_tool(
    invoke_function: Callable,
    prehook: Prehook,
    config_stack: ConfigStack,
    parent_scope: dict[str, object] | None = None,
) -> InvokeTool:
    """Build the ``invoke`` callable to bind into one agent's REPL."""
    # No ``include_invoke=False`` variant anymore. It existed to build a cap-leaf library that
    # kept the ``jaz.*`` helpers but dropped ``jaz.invoke`` (#635); once the helpers left the
    # namespace that variant had no members and always returned ``None``, and with the namespace
    # itself gone "the invoke tool without invoke" is just ``None``. The cap leaf is therefore
    # expressed where the decision is made — ``Agent.invoke`` passes ``None`` when
    # ``recursion_disabled`` — instead of by asking this factory for an empty surface.

    # Lazy import, hoisted out of the per-call `invoke` closure below: this module must not
    # import jaz.config at module top (config.py builds a default Config() at import time).
    # Safe here — get_invoke_tool only runs at runtime, well past the circular-import-sensitive
    # window — and runs once per tool build rather than once per sub-invoke call.
    from jaz.config import (
        ConfigStack,
        _reset_config_stack,
        _set_config_stack,
        get_config_stack,
    )

    def invoke(
        # Hooks — and, optionally, a single ConfigOverride — passed POSITIONALLY, exactly
        # like the public jaz.invoke (#528). There is no positional `prompt` (#538) and no
        # `return_type=`/`config_override=` keyword: the agent declares a return type with a
        # positional `ReturnType(T)` hook and a local override with a positional
        # `ConfigOverride(...)`, both extracted at runtime below. These hook classes are only
        # reachable when the host passes them in as inputs; referenced by the bare name the
        # host bound them under.
        *local_config_hooks: Hook | ConfigOverride,
        # Deliberately NO `task_name` typo-guard here (unlike the public `invoke`,
        # which redirects a stray `task_name=` to MetaData): on this agent-facing
        # surface `task_name` is an *ordinary input variable* — the caller means "put
        # `task_name` in my sub-agent's input blocks and REPL namespace," so it
        # correctly flows into `**inputs`. The plumbing label is a separate concept
        # (the blackboard, seeded via MetaData); an agent that wants to *label* a
        # sub-invoke passes a `MetaData(task_name=...)` hook positionally, kept distinct
        # from the `task_name` input var. Guarding `task_name` here would wrongly reject a
        # legitimate input name. (Design decision, #546.)
        **inputs: object,
    ) -> object:
        # TODO: Remove `object` looseness once pyright gets better at this surface.
        """Invoke a REPL-based sub-agent on arbitrary inputs.

        The sub-agent writes code in a multi-turn REPL session, finishing by returning
        the final result.

        Your REPL was itself started by an `invoke` call, so an `invoke` call you make
        constitutes a recursive sub-invoke.

        Arguments:
            *local_config_hooks: Zero or more hooks — and, optionally, a single
                config-override object — passed as leading positional arguments.
            **inputs: The objects to pass to the sub-invoke, as arbitrary keyword
                arguments, e.g., ``invoke(task="...", data=data, tool=tool)``. Inputs
                may include instructions for the sub-agent, tools the sub-agent is
                allowed to call, and any other objects it needs access to. Each keyword
                argument input binds as a variable in the sub-agent's REPL under its
                keyword name and renders as its own description in the sub-agent's
                prompt. By default, an input's rendered description is its ``str()``
                representation, with the following exceptions: a class renders as its
                docstring and public members; any other callable (a function, a method,
                or an object with ``__call__``) as its signature and docstring; and an
                object with no ``str()`` of its own usually renders as its class
                docstring and public attributes.

        Returns:
            The return value from the sub-invoke's REPL session
        """

        # Split the positional args exactly like the public jaz.invoke: pull out the
        # (at most one) ConfigOverride, then resolve the return type from a positional
        # ReturnType(...) hook and add the config-level ValidateREPLInput hook. There is no
        # return_type=/config_override= keyword on this surface anymore (#528) — a ReturnType
        # is a Hook so it stays in `hooks_only` for _resolve_invoke_hooks to extract, and
        # value/input validators reach here as ordinary positional ValidateReturn /
        # ValidateREPLInput hooks. Same helpers as the host `invoke`. Lazy import: jaz.invoke
        # is fully loaded by the time this closure runs, but a module-level import would
        # participate in the jaz.invoke -> jaz._invoke_tool import cycle.
        #
        # Done FIRST, ahead of every ContextVar re-base below. Both helpers raise on
        # agent-controllable input (`TypeError` on >1 ConfigOverride / >1 ReturnType), while
        # each re-base sets a token that is only reset in the `finally` around
        # `invoke_function`. Between a token set and that `try` there must therefore be no
        # raise path, or the token leaks — and these run on POOLED worker threads, so a
        # stranded `_current_prehook` would silently re-parent a later, unrelated top-level
        # invoke on the same thread (and, under a `RecursionLimit`, could refuse it against a
        # dead ancestor's depth). Neither helper reads a ContextVar, so hoisting them past
        # the re-bases is behavior-preserving. Measured before the hoist: a sub-invoke call
        # with two ConfigOverrides left the ancestor prehook stranded on the pool thread.
        from jaz.invoke import (
            _extract_config_override,
            _reset_current_prehook,
            _resolve_invoke_hooks,
            _set_current_prehook,
            get_active_prehook,
        )

        config_override, hooks_only = _extract_config_override(local_config_hooks)
        resolved_hooks = _resolve_invoke_hooks(hooks_only)

        # Re-base the propagating config override stack across a raw worker-thread
        # boundary (#727). The `_config_stack` ContextVar doesn't propagate into a
        # ThreadPoolExecutor worker, so a fresh worker sees the unestablished
        # module-default stack. Detect that via `not get_config_stack().established`
        # and re-base: compose the closure-threaded ancestor stack UNDER the layers the
        # worker already pushed locally (`ancestor.layers + live_stack.layers`, last-wins),
        # and mark it established. "Pushed locally" means the worker's own propagating
        # `with ConfigOverride()` pushes (which belong in the re-based stack) — NOT the
        # non-propagating per-invoke `config_override=` parameter, which stays off this
        # contextvar stack and is forwarded separately (twin of the `local_hooks` vs
        # `with Hook()` split in the hook re-base below). Because the fold is lazy, a benign
        # override the worker pushed *before* this call still composes onto the ancestor at
        # read time — the fix for the old either/or that dropped the ancestor's propagating
        # config (e.g. a compiler sandbox) whenever the worker had any local override. A
        # SAME-thread sub-invoke sees an established stack (the ContextVar propagated) and
        # is left untouched — re-basing it would double-apply the ancestor. Reset in
        # `finally`.
        #
        # This re-base is what the sub-invoke's config reads depend on: `_build_invoke_setup`
        # folds its EXECUTING config per-depth off this stack, read from the ContextVar rather
        # than threaded in. On a raw worker that read sees the composed ancestor stack only
        # because it happens inside the window this block opens; the caller frame passes
        # nothing config-shaped.
        _cfg_token = None
        live_stack = get_config_stack()
        if not live_stack.established:
            _cfg_token = _set_config_stack(
                ConfigStack(config_stack.layers + live_stack.layers, established=True)
            )

        # Re-base the ancestor hook context across a raw worker-thread boundary (#727,
        # mirroring the config re-base above). The _hook_context ContextVar doesn't
        # propagate into a ThreadPoolExecutor worker, so a fresh worker sees an
        # unestablished context. Detect that via `not get_current_hooks().established`
        # (a `with Hook()` only appends and carries the flag through, so the signal stays
        # reliable even after the worker activates its own hook — the old "is the ContextVar
        # None?" signal was corrupted by that `with`, which set the var and made the
        # sub-invoke SKIP restoring the ancestor → ancestor chain DROPPED). On a fresh
        # worker, compose `ancestor.hooks + local.hooks` (the worker's own `with`-hooks kept
        # ON TOP) and mark it established; a SAME-thread sub-invoke sees an established context
        # and is left untouched (re-basing would double-dispatch the ancestor). Lazy import:
        # hooks.effects imports this module's caller graph, so importing hooks at module top
        # would cycle.
        from jaz.hooks.context import (
            HookContext,
            _reset_hook_context,
            _set_hook_context,
            get_current_hooks,
        )

        # `parent_hooks` is always a HookContext (never None).
        parent_hooks = prehook.parent_hooks
        _hook_token = None
        live_hooks = get_current_hooks()
        if not live_hooks.established:
            # `live_hooks.hooks` holds ONLY the worker's own `with Hook()` activations —
            # which are *propagating* by design and so belong in the re-based ancestor set.
            # It is NOT the `local_hooks=` parameter: those never land on `.hooks` (they're
            # id-recorded in `local_active` and dispatched per-invoke only — see
            # invoke._activate_local_hooks), and this sub-invoke's own `local_hooks` are
            # forwarded separately below. So threading these through propagates exactly the
            # hooks that a same-thread sub-invoke would have inherited via the contextvar.
            #
            # Identity-dedup the one genuinely unchecked merge: the worker activated its
            # `with`-hooks while the ancestor chain was invisible (unestablished), so
            # Hook.__enter__'s collision check couldn't see the ancestor. Drop any worker
            # hook that IS the same instance as an ancestor propagating hook — otherwise it
            # would be dispatched twice. Distinct instances of the same class stay fine.
            kept_worker_hooks = tuple(
                h
                for h in live_hooks.hooks
                if not any(h is a for a in parent_hooks.hooks)
            )
            if len(kept_worker_hooks) != len(live_hooks.hooks):
                logger.warning(
                    "invoke worker re-base: dropped %d hook(s) already active in the "
                    "ancestor chain (same instance re-activated in a worker thread) to avoid "
                    "double dispatch.",
                    len(live_hooks.hooks) - len(kept_worker_hooks),
                )
            _hook_token = _set_hook_context(
                HookContext(
                    hooks=parent_hooks.hooks + kept_worker_hooks,
                    # Merge only the local-hook *id* markers (collision detection), not
                    # instances — this carries "these locals are still live" across the
                    # boundary without dispatching them to the sub-invoke.
                    local_active=parent_hooks.local_active | live_hooks.local_active,
                    established=True,
                )
            )
        # Re-base the ancestor prehook across a raw worker-thread boundary, mirroring the
        # config and hook re-bases above. `_current_prehook` is published by `_invoke` on the
        # thread running the agent, so a ThreadPoolExecutor worker sees it unset; without
        # this the sub-invoke's *own* nested calls (a public `jaz.invoke` reached from a
        # helper inside the worker) would mint a fresh depth-1 root. Establishing it from the
        # closure's captured ancestor makes all three ContextVars behave alike.
        _prehook_token = None
        if get_active_prehook() is None:
            _prehook_token = _set_current_prehook(prehook)

        # The parent to attribute this sub-invoke to. The LIVE value wins: this closure is
        # bound to one invoke, but nothing stops that invoke from wrapping it in a function
        # and handing the function to a sub-invoke, in which case the calling invoke — not
        # the binding one — is the real parent. Inside the binding invoke's own REPL the two
        # are the same object, so this only changes the cross-invoke case. On a raw worker
        # the block above has just established the captured value, so it resolves there too.
        effective_prehook = get_active_prehook() or prehook

        try:
            return invoke_function(
                local_hooks=resolved_hooks,
                config_override=config_override,
                prehook=effective_prehook,
                # Propagate this REPL's scope (snapshotted by the parent and
                # bound here as data) to the sub-invoke, so `jaz.scope` reaches
                # nested invokes even across thread/task hops. See `jaz.scope`.
                parent_scope=parent_scope,
                inputs=inputs,
            )
        finally:
            if _prehook_token is not None:
                _reset_current_prehook(_prehook_token)
            if _hook_token is not None:
                _reset_hook_context(_hook_token)
            if _cfg_token is not None:
                _reset_config_stack(_cfg_token)

    # The agent-facing sub-invoke is a single full signature: the `invoke` closure itself —
    # `(*local_config_hooks, **inputs)`, the same positional hooks + ConfigOverride surface as
    # the public jaz.invoke. The agent declares a return type / validators / metadata with the
    # ReturnType / ValidateReturn / ValidateREPLInput / MetaData hooks and a local override with
    # ConfigOverride — the hook classes must be reachable in the REPL (passed in as inputs by the
    # host). Binding inputs as REPL variables is a SEPARATE concern (not gated here): #568 removed
    # `expose_inputs_in_repl`, inputs always bind, and withholding them is a hook (evals'
    # `WithholdInputsFromREPL`, via a `DropVariables` effect).
    #
    # (There is no minimal `(**inputs)`-only variant: the `allow_config_hooks_in_subinvoke`
    # toggle that selected it was removed — the positional channel is always open. The former
    # `simplify_recursive_invoke` toggle and the string-based `restrict_invoke_args` "code
    # strings" lever were removed earlier for the same reason.)
    #
    # Expose the full closure directly — its signature/docstring IS the agent-facing tool
    # description.
    # TODO(#1223): under `restrict_invoke_args` the advertised positional-hook channel is one the
    # AST guard vetoes at runtime — make the rendered signature consistent with that guard.
    return invoke
