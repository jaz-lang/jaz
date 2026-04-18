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
    from jaz.hooks import PrintLogger, disable_hook, isolated_hooks
    import logging

    # Activate a hook
    with PrintLogger(level=logging.INFO):
        result = invoke("task", return_type=str)

    # Disable a hook type
    with disable_hook(PrintLogger):
        result = invoke("quiet task", return_type=str)

    # Isolated context (no parent hooks)
    with isolated_hooks():
        result = invoke("clean slate", return_type=str)

Architecture
------------
The context system maintains an immutable HookContext that contains:
- hooks: tuple of currently active Hook instances (in registration order)
- disabled_types: set of hook types that are temporarily disabled

Each time a hook is entered/exited, a new HookContext is created and set in the
context variable. This immutability ensures thread safety and makes context
propagation predictable.

Default Global Context
----------------------
By default, the global context includes PrintLogger(level=logging.INFO).
This means all invoke() calls have basic logging unless explicitly disabled
with isolated_hooks().

See Also
--------
- hooks/README.md: Comprehensive documentation of the hook system
- Hook base class: Implements context manager protocol for easy activation
- get_dispatcher(): Get the singleton dispatcher that reads from context
"""

from contextlib import contextmanager
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
    disabled_types: frozenset[type] = field(default_factory=frozenset)
    is_implicit_default: bool = False

    def with_hook(self, hook: "Hook") -> "HookContext":
        """Return new context with hook added at the end."""
        return HookContext(
            hooks=self.hooks + (hook,),
            disabled_types=self.disabled_types,
            is_implicit_default=False,
        )

    def with_disabled(self, hook_type: type) -> "HookContext":
        """Return new context with hook type disabled."""
        return HookContext(
            hooks=self.hooks,
            disabled_types=self.disabled_types | {hook_type},
            is_implicit_default=False,
        )

    def get_active_hooks(self) -> list["Hook"]:
        """Get all hooks that aren't disabled, in registration order."""
        return [h for h in self.hooks if type(h) not in self.disabled_types]


def get_current_hooks() -> HookContext:
    """Get current hook context.

    Returns the default global context if none has been set.
    The default context includes PrintLogger(level=logging.INFO).
    """
    try:
        return _hook_context.get()
    except LookupError:
        # Return default global context
        return _get_default_context()


def _get_default_context() -> HookContext:
    """Get the default global hook context.

    This is lazily initialized to include PrintLogger(level=logging.INFO).
    The is_implicit_default flag indicates this is the default context,
    which will be replaced by an empty context when the first user hook is added.
    """
    # Import here to avoid circular dependency
    import logging

    from .builtin.print_logger import PrintLogger

    return HookContext(
        hooks=(PrintLogger(level=logging.INFO),), is_implicit_default=True
    )


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


@contextmanager
def disable_hook(*hook_types: type):
    """Context manager to temporarily disable specific hook types.

    Usage:
        with PrintLogger(level=logging.INFO):
            do_something()  # PrintLogger active

            with disable_hook(PrintLogger):
                do_other()  # PrintLogger disabled

                with PrintLogger(level=logging.WARNING):
                    do_third()  # New PrintLogger active (WARNING level)

    Args:
        *hook_types: Hook types to disable
    """
    current = get_current_hooks()
    new_context = current
    for hook_type in hook_types:
        new_context = new_context.with_disabled(hook_type)

    token = _set_hook_context(new_context)
    try:
        yield
    finally:
        _reset_hook_context(token)


@contextmanager
def isolated_hooks():
    """Context manager that clears all parent hooks.

    Usage:
        with PrintLogger():
            invoke(...)  # Has PrintLogger

            with isolated_hooks():
                invoke(...)  # No hooks at all
    """
    current = get_current_hooks()
    empty_context = HookContext()
    _token = _set_hook_context(empty_context)  # Token unused - we restore directly
    try:
        yield
    finally:
        _set_hook_context(current)
