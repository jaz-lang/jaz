"""Context variable management for hooks.

This module provides the context variable-based hook management system that allows
hooks to be scoped using context managers and automatically propagate to nested invokes.

Design Philosophy
-----------------
The context-based design uses Python's `contextvars` module to manage hook lifecycle.
This provides several key benefits:

1. **Automatic Propagation**: Hooks naturally propagate to nested invoke() calls
2. **Scope-Based Lifecycle**: Hooks are active exactly within their `with` block
3. **Thread-Safe**: Each thread gets its own context copy
4. **Async-Safe**: Works correctly across `await` boundaries
5. **No Parameter Passing**: No need to pass dispatcher through function calls

Basic Usage
-----------
    from jaz import invoke
    from jaz.hooks import PrintLogger
    import logging

    # Activate a hook
    with PrintLogger(level=logging.INFO):
        result = invoke(ReturnType(str), task="task")

Architecture
------------
The context system maintains an immutable HookContext containing the tuple of
currently active Hook instances (in registration order). Each time a hook is
entered/exited, a new HookContext is created and set in the context variable.
This immutability ensures thread safety and makes context propagation predictable.

Design: built-in defaults and no generic disable
-------------------------------------------------
The default global hook context is **empty**. Future essential behaviors
(budget enforcement, permissions, etc.) will be auto-installed by `invoke()`
as regular `Hook` instances. Each such built-in ships with its own dedicated,
behavior-specific opt-out (e.g., a hypothetical `bypass_budget()`).

There is deliberately **no** generic disable/clear mechanism — neither a
class-keyed `disable_hook(SomeClass)` nor a blanket `clear_all_hooks()` (#466
removed both). Hook lifecycle is managed purely by lexical `with` scoping:

1. Multiple instances of the same `Hook` class can be active simultaneously
   (e.g., two `PrintLogger`s at different log levels). A class-keyed disable
   silently kills all of them, conflating class with instance identity.
2. A blanket clear is a footgun: it silently strips governance / safety
   baseline hooks (e.g. the loop's only termination guarantee), and its effect
   can't even be honored in raw worker threads (they restore the ancestor
   context) — so it gives a false sense of a clean slate.
3. Named, behavior-specific opt-outs document intent at the call site and are
   far easier to audit than a generic kill switch.

See Also
--------
- hooks/README.md: Comprehensive documentation of the hook system
- Hook base class: Implements context manager protocol for easy activation
- get_dispatcher(): Get the singleton dispatcher that reads from context
"""

from collections.abc import Iterable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dispatcher import Hook

# Context variable holding the current hook context
_hook_context: ContextVar["HookContext"] = ContextVar("hook_context")


@dataclass(frozen=True)
class HookContext:
    """Immutable snapshot of active hooks at a point in execution.

    NOTE: Hooks are currently called in registration order (first registered,
    first called). In the future, we may add options to control hook ordering (e.g.,
    priority-based).
    """

    hooks: tuple["Hook", ...] = field(default_factory=tuple)
    # Ids of hook instances active via `local_hooks` in this invoke or any enclosing
    # one. Carried on the contextvar purely for *collision detection*, NOT dispatch:
    # local hooks must not propagate to nested invokes (being local is the whole
    # point), so they stay off `hooks`. But a local hook's lifecycle IS live for its
    # invoke's entire dynamic extent (setup at entry, teardown at exit — see
    # invoke._activate_local_hooks), which includes every nested invoke. Recording the
    # id here lets `Hook.__enter__` / `_activate_local_hooks` see a still-live local and
    # refuse to re-activate it (via `with` or a nested `local_hooks`), which would run
    # setup()/teardown() twice. This makes the "one instance, one live scope" invariant
    # (#533 / #540) total rather than only covering `with`-activated hooks.
    local_active: frozenset[int] = frozenset()
    # True once a real invoke has established this context as the ancestor for a subtree
    # (#727, mirroring ConfigStack.established). A `with` activation only ``with_hook``s,
    # which carries the flag through but never sets it true — so ``not established`` is the
    # reliable "fresh worker thread, contextvar didn't propagate" signal the worker re-base in
    # library/jaz.py keys on, even after the worker runs its own `with Hook()`. This replaces
    # the old "is the contextvar unset (None)?" signal, which a worker's own `with` corrupted
    # (it set the var, so the sub-invoke skipped restoring the ancestor → ancestor DROPPED).
    # Excluded from equality: it's re-establishment bookkeeping, not part of the value.
    established: bool = field(default=False, compare=False)

    def with_hook(self, hook: "Hook") -> "HookContext":
        """Return new context with hook added at the end (carrying local_active/established)."""
        return HookContext(
            hooks=self.hooks + (hook,),
            local_active=self.local_active,
            established=self.established,
        )

    def with_local_active(self, ids: Iterable[int]) -> "HookContext":
        """Return new context recording these local-hook instance ids as live."""
        return HookContext(
            hooks=self.hooks,
            local_active=self.local_active | frozenset(ids),
            established=self.established,
        )

    def with_established(self) -> "HookContext":
        """Return this context marked established (the ancestor for a subtree). See
        ``established`` and ``invoke._established_hook_context``."""
        return HookContext(
            hooks=self.hooks, local_active=self.local_active, established=True
        )


def get_current_hooks() -> HookContext:
    """Get current hook context.

    Returns an empty context if none has been set.
    """
    try:
        return _hook_context.get()
    except LookupError:
        return HookContext()


def _set_hook_context(ctx: HookContext) -> Token[HookContext]:
    """Set the hook context and return a token for resetting.

    Internal function - users should use context managers instead.
    """
    return _hook_context.set(ctx)


def _reset_hook_context(token: Token[HookContext]) -> None:
    """Reset the hook context using a token.

    Internal function - users should use context managers instead.
    """
    _hook_context.reset(token)
