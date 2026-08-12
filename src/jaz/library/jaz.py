from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from .core import Library

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from jaz.config import ConfigOverride, ConfigStack
    from jaz.hooks import Hook
    from jaz.invoke import Prehook


# The static description of the JAZ library shown to agents. Inlined here from the former
# ``jaz_library_description_simple.jinja2`` template — which was selected by the (now removed)
# ``protocol.library_description_template`` option (``jaz_library_description_template`` before
# the config was nested by component). That template carried no Jinja logic (it
# ignored the ``expose_inputs_in_repl`` / ``restrict_invoke_args`` vars it was rendered with) and
# was the only value the option ever pointed at, so a config-selectable template file bought
# nothing a module constant doesn't. Kept as a named constant (rather than inlined at the
# ``Library(...)`` call site) so the wording stays easy to find and edit. The trailing newline is
# deliberate — it reproduces the template's output byte-for-byte (the Jinja env used
# ``keep_trailing_newline=True``).
_JAZ_LIBRARY_DESCRIPTION = (
    "JAZ is a framework for building LLM agents that solve tasks by writing and running\n"
    "code in a REPL loop, iterating until the task is complete.\n"
)


def get_jaz_library(
    invoke_function: Callable | None = None,
    prehook: Callable | None = None,
    config_stack: ConfigStack | None = None,
    allow_config_hooks_in_subinvoke: bool = False,
    parent_scope: dict[str, object] | None = None,
    include_invoke: bool = True,
) -> Library | None:  # TODO: proper type hinting
    # The agent-facing ``jaz`` namespace holds exactly one member: the synthesized recursive
    # ``jaz.invoke`` tool. Framework helpers (``ReturnType`` / ``Display`` / ``Library`` / …) are
    # no longer auto-bound under ``jaz.*``; a host that wants an agent to use them passes them as
    # ordinary inputs (``jaz.invoke(ReturnType=jaz.ReturnType, ...)``), so they bind as top-level
    # REPL names. Consequently ``include_invoke=False`` (the recursion-cap leaf, #635) now has no
    # members at all and returns ``None`` — an at-cap agent simply has no ``jaz`` library — via the
    # ``if not all_tools`` short-circuit below. Callers that want the recursive tool pass
    # ``include_invoke=True`` (the default).
    #
    # TRADEOFF of input-delivery vs. the old ``jaz.*`` export: a ``jaz.*`` member was invisible to
    # the prompt, but an input is NOT — each helper passed in renders its own ``<name type=...>``
    # block (docstring and all) into the user message, on every turn, and counts against
    # ``max_input_length`` / truncation. So handing an agent N helpers costs N rendered blocks per
    # turn, and any run compared against a pre-migration baseline now has a different prompt for a
    # reason unrelated to what it measures. A host that wants a helper bound but not shown can wrap
    # it at the call site with ``jaz.Display(value, None)`` (the standard hide-from-prompt directive,
    # which binds the payload as a plain REPL var) — itself a helper that must be passed in.
    if include_invoke:
        # TODO: Clean up all of this mess lmao
        assert invoke_function is not None, (
            "include_invoke=True requires invoke_function"
        )
        assert prehook is not None
        assert config_stack is not None

        # Lazy import, hoisted out of the per-call `invoke` closure below: library.jaz
        # must not import jaz.config at module top (config.py builds a default Config()
        # at import time which needs this module). Safe here — get_jaz_library only runs
        # at runtime, well past the circular-import-sensitive window — and now runs once
        # per library build rather than once per sub-invoke call.
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
            # reachable when the host passes them in as inputs (they are no longer bound under
            # `jaz.*`); referenced by the bare name the host bound them under.
            *local_config_hooks: Hook | ConfigOverride,
            # Deliberately NO `task_name` typo-guard here (unlike the public `invoke`,
            # which redirects a stray `task_name=` to MetaData): on this agent-facing
            # surface `task_name` is an *ordinary input variable* — the caller means "put
            # `task_name` in my sub-agent's input blocks and REPL namespace," so it
            # correctly flows into `**inputs`. The plumbing label is a separate concept
            # (the blackboard, seeded via MetaData); an agent that wants to *label* a
            # sub-invoke passes `jaz.MetaData(task_name=...)` positionally (now exposed,
            # #545), kept distinct from the `task_name` input var. Guarding `task_name`
            # here would wrongly reject a legitimate input name. (Design decision, #546.)
            **inputs: object,
        ) -> object:
            # TODO: Remove `object` looseness once pyright gets better at this surface.
            """Invoke a sub-agent with the given inputs to enter a REPL session and
            return the response.

            This REPL was itself started by a `jaz.invoke` call, so each
            `jaz.invoke` you make here is a recursive (nested) sub-invoke.

            Arguments:
                *local_config_hooks: Zero or more hooks — and, optionally, a single
                    config-override object — passed as leading POSITIONAL arguments, e.g.
                    ``jaz.invoke(ReturnType(int), task=...)``. Use them to declare the
                    sub-invoke's return type (a ReturnType hook — omit it for no return
                    value), validate its return value or REPL input, label it for the
                    blackboard, or apply a local, non-propagating config override for THIS
                    sub-invoke (at most one override). WHICH hook / override classes are
                    available depends on the host: they are passed in as inputs (bound as
                    top-level names) — a class you can't reference simply wasn't provided.
                    Any other hooks are scoped to THIS sub-invoke only and do not
                    propagate to its own nested invokes.
                inputs: The inputs to pass to the sub-agent, as keyword arguments (e.g.
                    ``jaz.invoke(ReturnType(str), task="...", data=data)``). There is no
                    positional prompt; the sub-agent's task is just an input (conventionally
                    ``task=...``). Each input binds as a REPL variable and renders as its own
                    ``<name type="...">`` block.

            Returns:
                The return value from the REPL session
            """

            # Split the positional args exactly like the public jaz.invoke: pull out the
            # (at most one) ConfigOverride, then resolve the return type from a positional
            # ReturnType(...) hook and add the config-level ValidateREPLInput hook. There is no
            # return_type=/config_override= keyword on this surface anymore (#528) — a ReturnType
            # is a Hook so it stays in `hooks_only` for _resolve_invoke_hooks to extract, and
            # value/input validators reach here as ordinary positional ValidateReturn /
            # ValidateREPLInput hooks. Same helpers as the host `invoke`. Lazy import: jaz.invoke
            # is fully loaded by the time this closure runs, but a module-level import would
            # participate in the jaz.invoke -> jaz.library import cycle.
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
                    ConfigStack(
                        config_stack.layers + live_stack.layers, established=True
                    )
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
            # hooks.effects imports jaz.library, so importing hooks at module top would cycle.
            from jaz.hooks.context import (
                HookContext,
                _reset_hook_context,
                _set_hook_context,
                get_current_hooks,
            )

            # `prehook` is asserted non-None at the top of this block, and `parent_hooks` is
            # always a HookContext (never None).
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
                        "jaz.invoke worker re-base: dropped %d hook(s) already active in the "
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
                        local_active=parent_hooks.local_active
                        | live_hooks.local_active,
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
                # ``prehook`` is annotated ``Callable | None`` on this factory (a long-standing
                # loose hint — see the TODO on the return type) but is always a ``Prehook``;
                # it is asserted non-None above and used as one at ``prehook.parent_hooks``.
                _prehook_token = _set_current_prehook(cast("Prehook", prehook))

            # The parent to attribute this sub-invoke to. The LIVE value wins: this closure is
            # bound to one invoke, but nothing stops that invoke from wrapping it in a function
            # and handing the function to a sub-invoke, in which case the calling invoke — not
            # the binding one — is the real parent. Inside the binding invoke's own REPL the two
            # are the same object, so this only changes the cross-invoke case. On a raw worker
            # the block above has just established the captured value, so it resolves there too.
            effective_prehook = get_active_prehook() or prehook

            config_override, hooks_only = _extract_config_override(local_config_hooks)
            resolved_hooks = _resolve_invoke_hooks(hooks_only)
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

        # Two agent-facing shapes (#528 follow-up), selected by the dedicated
        # `allow_config_hooks_in_subinvoke` config toggle (threaded in as a param) — deliberately
        # SEPARATE from whether inputs bind as REPL variables — the two concerns were previously
        # conflated. That half is no longer config at all: #568 removed `expose_inputs_in_repl`,
        # inputs always bind, and withholding them is a hook (evals' `WithholdInputsFromREPL`,
        # via a `DropVariables` effect).
        #   - MINIMAL (`allow_config_hooks_in_subinvoke=False`, DEFAULT): `(**inputs)` — the agent
        #     may pass only named inputs, with no hook/override surface. Off by default keeps the
        #     agent-facing sub-invoke narrow and composes with `restrict_invoke_args` (the
        #     literal-only jaz.invoke AST guard — see the evals ``RestrictInvokeArgs`` hook — which is
        #     INCOMPATIBLE with positional hooks like `jaz.ReturnType(int)`, a call not a literal).
        #   - FULL (`allow_config_hooks_in_subinvoke=True`): the `invoke` closure itself —
        #     `(*local_config_hooks, **inputs)`, the same positional hooks + ConfigOverride surface
        #     as the public jaz.invoke. The agent declares a return type / validators / metadata
        #     with the ReturnType / ValidateReturn / ValidateREPLInput / MetaData hooks and a local
        #     override with ConfigOverride — but this flag only OPENS the positional channel; the
        #     hook classes must be reachable in the REPL (passed in as inputs by the host).
        #
        # (The former `simplify_recursive_invoke` toggle and the string-based `restrict_invoke_args`
        # "code strings" lever were removed; there is a single full signature now.)
        if allow_config_hooks_in_subinvoke:
            # Expose the full closure directly — its signature/docstring IS the agent-facing
            # full tool description.
            invoke_name_and_func = [("jaz.invoke", invoke)]
        else:

            def _invoke_minimal(**inputs: object) -> object:
                """Invoke a sub-agent to solve a task in its own REPL session.

                This REPL was itself started by a `jaz.invoke` call, so each
                `jaz.invoke` you make here is a recursive (nested) sub-invoke.

                Arguments:
                    **inputs: Named inputs to pass to the sub-agent (conventionally
                        `task=...` for the sub-agent's task). There is no positional prompt.
                        These bind as REPL variables in the sub-agent and also render in
                        its prompt.

                Returns:
                    The return value from the sub-agent's REPL session.
                """
                return invoke(**inputs)

            invoke_name_and_func = [("jaz.invoke", _invoke_minimal)]
    else:
        # Cap-leaf variant: no recursive jaz.invoke tool (see include_invoke docstring).
        invoke_name_and_func = []

    # `jaz.invoke` (invoke_name_and_func) is the sole member of the `jaz` namespace. Framework
    # helpers are delivered as ordinary inputs by the host, not bound under `jaz.*`.
    all_tools = invoke_name_and_func

    # No members at all (invoke-less AND nothing opted in) → no library. Returning an empty
    # "JAZ" library would render a member-less `<jaz_library>` block; None renders nothing,
    # matching the pre-#635 cap-leaf UX (the leaf variant now has no members at all).
    if not all_tools:
        return None

    # Build module description. The namespace's sole member is the recursive `jaz.invoke`.
    module_desc = "Contains `jaz.invoke`."

    # The JAZ library description is a static constant (see ``_JAZ_LIBRARY_DESCRIPTION``). The
    # former template was rendered with the ancestor's effective ``repl.configs``
    # (``expose_inputs_in_repl`` / ``restrict_invoke_args``) but ignored those vars entirely, so the
    # description never actually varied — dropping that dead plumbing along with the template.
    jaz_library = Library(
        "JAZ",
        _JAZ_LIBRARY_DESCRIPTION,
        [("jaz", module_desc)],
        all_tools,
    )
    return jaz_library
