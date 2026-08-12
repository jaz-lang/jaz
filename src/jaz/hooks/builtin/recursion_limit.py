"""Opt-in recursion-depth cap, expressed as a hook.

This is the hook-driven successor to the framework's former ``config.max_depth`` field.
The framework itself imposes **no** recursion limit: nested ``jaz.invoke`` is unbounded
unless a ``RecursionLimit`` is installed via a propagating channel —
``with RecursionLimit(max_depth=...): jaz.invoke(...)`` — which caps an invoke and its
nested invokes. The cap is a policy an operator opts into rather than a built-in
floor — governance/security are opt-in in jaz —
because (a) models don't reach for ``jaz.invoke`` on their own (they recurse only when a
task explicitly calls for it, and then only to the task's natural depth), (b) the real
runaway harm is *cost*, bounded by the opt-in ``BudgetPool``, not depth, and (c) CPython's
own recursion limit backstops a literal infinite loop on the default synchronous path.

## What it reproduces

The old ``config.max_depth`` did two structurally different things; this hook does both,
each via the effect matched to it, keyed on the invoke's ``event.depth`` (1-based: the top
agent is depth 1):

- **Affordance removal at the cap leaf** (``depth == max_depth``): emit ``DisableRecursion``,
  which the primitive honors by binding the REPL and rendering the system prompt WITHOUT the
  recursive ``jaz.invoke`` tool. The at-cap agent never sees the tool — the same UX the old
  ``jaz_library=None`` gate gave, not a "tool present but the call fails" model.
- **Backstop for an over-cap child** (``depth > max_depth``): emit ``Abort`` carrying a
  ``RecursionLimitError``. Affordance removal means an honest agent never gets here, but the
  *public* ``jaz.invoke`` (reached from a human-written tool, or a deliberately injected real
  ``jaz`` module) has no such lever — so a child that enters past the cap anyway is terminated,
  reproducing the old primitive ``RecursionLimitError`` guard. ``RecursionLimitError`` is reused so
  caller/test expectations are unchanged.

## Installation: propagating vs local (IMPORTANT)

``RecursionLimit`` bounds a *subtree* only when it is installed via a **propagating** channel
— i.e. one that rides the hook-context contextvar (or the config) into every nested invoke:

- ``with RecursionLimit(max_depth=N): jaz.invoke(...)`` — the context-manager form; propagates
  to every nested invoke for the duration of the block. This is the general subtree cap, and
  the successor to the old ``ConfigOverride(max_depth=k)`` (that config field is gone). An
  agent tightens its own subtree the same way — ``with rl: jaz.invoke(...)`` — provided the
  host passes a ``RecursionLimit`` in to it as an ordinary input; it is not in the agent
  facade by default.
To cap a whole run / subtree, activate it via ``with`` at the top (e.g. an ``ExitStack`` of
governance hooks around the run) — it then propagates to every nested invoke.

Passing it as a positional local hook (``jaz.invoke(RecursionLimit(...), ...)``) is
**rejected with ``HookActivationError``** — it does not silently under-cap. Local hooks are dispatched to *only
the one invoke they are attached to* and are deliberately kept off the contextvar (see
``_activate_local_hooks`` in ``invoke.py``), so they never reach nested invokes; a local
``RecursionLimit`` would therefore fail to cap the subtree (an uncapped grandchild would
recurse without limit). Rather than let that footgun ship silently, ``RecursionLimit`` refuses
the local-hooks channel and points the caller at the ``with`` form.

### How the rejection works without touching the core

No framework change is needed: the two install channels reach a :class:`Hook` through *different*
lifecycle entry points, and ``RecursionLimit`` distinguishes them in its own overrides —

- ``with`` → ``Hook.__enter__`` → ``setup()`` (and registers on the contextvar → propagates),
- ``local_hooks`` → ``setup()`` **directly** (no ``__enter__``, no contextvar → no propagation).

So ``setup()`` reached *without* going through ``__enter__`` means a local-hooks activation —
and that is exactly the case ``setup()`` raises on. ``__enter__`` marks itself so the nested
``setup()`` it triggers is allowed; ``baseline`` never trips ``setup()``. See ``__enter__`` /
``setup`` below.

## Composition (tighten-only, for free)

For a *propagating* install, stacking a *stricter* ``RecursionLimit`` unions the effects — the
stricter one fires (disables / aborts) at a shallower depth and can't be loosened — so
tighten-only composition falls out with no special plumbing, exactly as ``IterationLimit``
documents for its Aborts.

## Why ``on_invoke_enter``, not ``on_llm_query_enter``

Recursion availability is fixed when the REPL is built, and ``DisableRecursion`` is only
coherent at ``InvokeEnter`` (before the library is bound). ``InvokeEnter`` is also where the
child's depth is first known, so both the leaf-disable and the over-cap-abort decisions live
in one handler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jaz.exceptions import HookActivationError, RecursionLimitError
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import Abort, DisableRecursion, Effect
from jaz.hooks.events import InvokeEnter


@dataclass(eq=False, kw_only=True)
class RecursionLimit(Hook):
    """Cap ``jaz.invoke`` recursion at ``max_depth`` levels.

    ``max_depth`` is the maximum number of invokes on the stack, so ``max_depth=1`` allows a
    top-level invoke and no sub-invokes at all.

    Depth is taken from the invoke a call runs inside, not from where the call is written. A
    function of yours that calls ``jaz.invoke`` produces a depth-1 invoke when your own code
    calls it, and a depth-2 invoke when an agent calls it from within an invoke.

    Must be activated with ``with RecursionLimit(max_depth=N):``. Passing it positionally
    raises :class:`~jaz.exceptions.HookActivationError`: a positional hook does not reach
    nested invokes, so it could not cap them.

    An agent at ``max_depth`` runs normally but is not given the ``jaz.invoke`` tool. It can
    still reach one indirectly — through a function you passed it that invokes internally — so
    a sub-invoke that lands past the cap is refused with
    :class:`~jaz.exceptions.RecursionLimitError`.

    **Known gap.** Under the above definition of "depth", a sub-invoke is considered the sub-invoke of
    the invoke that *calls* it, including when an agent wraps its bound ``jaz.invoke`` in a function
    and passes that function down. However, the current implementation does not obey this when the
    wrapper's ``jaz.invoke`` call runs on a thread that did not inherit the calling invoke's context —
    either because the caller ran the wrapper on a thread it spawned (a ``ThreadPoolExecutor`` worker,
    say), or because the wrapper spawns one itself and invokes inside it. The invoke is then counted at
    the *binding* invoke's depth instead of the caller's, so a stack deeper than ``max_depth`` can be
    built that way. (Note that this gap requires ``allowed_imports`` to be set to allow agents
    to spawn threads.)

    Args:
        max_depth: Deepest allowed invoke depth, 1-based — a top-level invoke is depth 1, its
            sub-invokes depth 2, sub-sub-invokes depth 3, etc. (User code outside any invoke is
            at depth 0.)

    Raises:
        ValueError: If ``max_depth`` is less than 1.
    """

    # Mechanics: handles ``InvokeEnter``; emits ``DisableRecursion`` at the cap leaf (which is
    # what "is not given the jaz.invoke tool" means), and ``Abort`` carrying a
    # ``RecursionLimitError`` as the backstop when a child enters past the cap anyway (which is
    # what "refused" means).
    #
    # Withholding the tool is an affordance, not an enforcement: the agent can still reach a
    # nested invoke through any callable it was handed that invokes internally. The backstop
    # covers that because ``event.depth`` comes from the ambient ``_current_prehook``
    # ContextVar, which the REPL exec runs inside — so an invoke made from a helper inherits
    # the enclosing invoke's depth + 1 exactly as the tool would. Measured: with
    # ``max_depth=1``, an agent calling a helper that invokes produces a depth-2
    # ``InvokeEnter`` that exits as a ``Raise`` carrying ``RecursionLimitError``.
    #
    # That inheritance holds even when an agent wraps its bound ``jaz.invoke`` in a function and
    # passes the function to a sub-invoke: the facade resolves the parent from the live
    # ``_current_prehook``, so the invoke belongs to whichever invoke *calls* the wrapper. It
    # did not always — the closure used to pass its captured ``prehook`` unconditionally, which
    # attributed such an invoke to the binding invoke and let a stack deeper than ``max_depth``
    # run unrefused. See ``TestDepthFollowsTheCallingInvoke``.
    #
    # The gap needs BOTH the wrapper-passing and the thread hop, which is easy to misread as one
    # condition. The facade picks a parent as ``get_active_prehook() or prehook`` — a live signal
    # (who is calling) and a captured one (who bound the closure) — and each ingredient breaks a
    # different one. Passing the wrapper down corrupts the CAPTURE: it is fixed at bind time, so it
    # names the binder no matter who ends up holding the function. Running on a raw worker erases
    # the LIVE value: ``Executor.submit`` does not copy context. Either alone is survivable because
    # the other source covers it — wrapper passed down but called on the caller's own thread
    # resolves live and is correct; a worker with no wrapper-passing (an agent calling its OWN bound
    # ``jaz.invoke`` on a thread it spawned) falls back to a capture that genuinely is its parent.
    # Only together is there nothing left to read.
    #
    # TODO(#1033): tracked there, together with the direction that would actually close it — a
    # jaz-owned context-propagating parallel primitive (``copy_context()`` before dispatch), which
    # would also close #440 (the *public* entry reached from a worker, which has no closure to fall
    # back on and so mints a fresh depth-1 root instead of landing on the binder) and folds into the
    # propagation unification in #439.
    #
    # WHICH party supplies the thread is irrelevant — only that one is between the caller and the
    # ``jaz.invoke`` call. The docstring names both forms because the second is easy to miss and is
    # the stronger one: a wrapper that spawns its own worker and invokes inside it carries the hop
    # with it, so the caller does nothing unusual and cannot tell. Measured on the depth-1 →
    # depth-2 tree used by ``TestDepthFollowsTheCallingInvoke``: the caller invoking such a wrapper
    # DIRECTLY still produces ``InvokeEnter`` depths [1, 2, 2] with the third parented to the first,
    # and ``max_depth=2`` does not refuse it — identical to the pre-fix behaviour, where the same
    # tree without any thread now correctly gives [1, 2, 3] and is refused.
    #
    # The documented gap is what is left of that after the fix, and it is STRUCTURAL rather than
    # an oversight. ``_current_prehook`` does not cross a ``ThreadPoolExecutor`` boundary, so on
    # a raw worker the closure's captured ancestor is the only ancestor there is — the facade
    # establishes it (that is what keeps a public ``jaz.invoke`` reached from inside the worker
    # attached to the tree at all) and then reads it back as the effective parent. Nothing
    # distinguishes "a worker of the invoke that bound this closure" from "a worker of some other
    # invoke that was handed the closure", because the only thing that survived the boundary is
    # the capture itself. The alternative — refusing to fall back on a worker — trades this
    # narrow under-count for the much wider one it replaced (every worker sub-invoke minting a
    # fresh depth-1 root, detaching the whole subtree from the cap), so the fallback stays.
    #
    # A host that needs the cap to hold across a thread it controls can carry the live context
    # over explicitly: ``contextvars.copy_context().run(lambda: jaz.invoke(...))`` — the same
    # escape hatch the hook re-base documents for mid-REPL hook state (see ``Prehook`` in
    # invoke.py and hooks/README.md).
    #
    # The docstring states the two observable outcomes rather than the effects.

    # A ``@dataclass(eq=False)`` so ``to_dict``/``from_dict`` derive ``{max_depth}`` from the
    # field automatically (no ``_to_dict_params`` boilerplate). ``eq=False`` keeps identity
    # semantics (hooks are deduped by ``is``, not value). ``kw_only`` preserves the keyword-only
    # constructor.

    max_depth: int

    # True only for the dynamic extent of ``__enter__`` (the ``with`` install path), so the
    # ``setup()`` that ``__enter__`` triggers knows it is a legitimate propagating install.
    # ``init=False`` so it is per-instance lifecycle bookkeeping, kept out of the constructor AND
    # out of serialization (to_dict skips ``init=False`` fields), not a construction param.
    _entering_via_context: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        # 1-based depth: max_depth < 1 would forbid even the top-level invoke, which is
        # incoherent (there must always be a root). Mirrors the old config validation that
        # rejected max_depth <= 0.
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth!r}")

    def __enter__(self):
        # Mark the `with` path so the setup() the base __enter__ calls next is recognized as a
        # propagating install. Cleared in `finally` so a failed __enter__ (e.g. the base's
        # double-activation guard raising before setup()) never leaves the flag set — which
        # would let a later local_hooks activation slip through unrejected.
        self._entering_via_context = True
        try:
            return super().__enter__()
        finally:
            self._entering_via_context = False

    def setup(self) -> None:
        # Reject the local_hooks install channel. setup() runs on exactly two paths — `with`
        # (via __enter__, which set the flag above) and local_hooks (called directly, flag
        # unset). So reaching setup() with the flag unset means local_hooks, which does NOT
        # propagate to nested invokes and therefore cannot
        # cap a subtree. Fail loud rather than silently under-cap. This lives entirely in the
        # hook — no change to the core Hook base, the dispatcher, or _activate_local_hooks.
        # TODO(#808): this infers the install channel from the __enter__/setup routing; consider
        # making the channel an explicit signal to setup() so the guard can't be silently broken.
        if not self._entering_via_context:
            raise HookActivationError(
                "RecursionLimit cannot be installed via local_hooks: local hooks are not "
                "dispatched to nested invokes, so it could not cap the subtree. Install it "
                "with `with RecursionLimit(max_depth=...):` instead."
            )
        super().setup()

    def on_invoke_enter(self, event: InvokeEnter) -> list[Effect]:
        # Backstop: a child that entered past the cap anyway (public jaz.invoke via a human
        # tool, or an injected real jaz) — terminate it. On the structural path (affordance
        # removed at the leaf) this never fires.
        if event.depth > self.max_depth:
            return [
                Abort(
                    error=RecursionLimitError(
                        f"jaz.invoke reached recursion depth {event.depth}, "
                        f"exceeding the RecursionLimit maximum of {self.max_depth}."
                    )
                )
            ]
        # Affordance removal on the cap leaf: this agent runs, but its REPL is built without
        # the recursive jaz.invoke tool (a sub-invoke would be depth max_depth + 1 > cap).
        if event.depth == self.max_depth:
            return [DisableRecursion()]
        return []
