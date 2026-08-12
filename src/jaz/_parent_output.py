"""Parent output channel for child-to-parent communication.

This module provides a general mechanism for code running inside a child
jaz.invoke() to write messages that appear in the parent's REPL output.

Usage:
    from jaz._parent_output import parent_print

    # In a tool, hook, or library function:
    parent_print("Processing item 1 of 10...")

The REPL automatically sets up the channel during code execution, so
parent_print() will work correctly in any code called from the REPL.
"""

from collections.abc import Callable
from contextvars import ContextVar, Token

# Context variable holding the parent's print function.
# Set by PythonREPL._capture_print() during code execution.
_parent_output_channel: ContextVar[Callable[[str], None] | None] = ContextVar(
    "parent_output_channel", default=None
)


def parent_print(message: str, *, fallback_to_stdout: bool = False) -> bool:
    """Print a message to the parent's REPL output.

    If called from within a child jaz.invoke(), the message appears in
    the parent's output.

    Args:
        message: The message to print to the parent.
        fallback_to_stdout: If True and no parent channel exists, print to
            stdout instead (treating the user as the parent). If False (default),
            silently do nothing at the top level.

    Returns:
        True if the message was printed (to parent or stdout),
        False if no output occurred.

    Examples:
        from jaz._parent_output import parent_print

        def my_tool_function():
            # Silent at top level:
            parent_print("Starting processing...")

            # Always visible (prints to stdout if no parent):
            parent_print("Progress: 50%", fallback_to_stdout=True)
    """
    channel = _parent_output_channel.get()
    if channel is not None:
        channel(message)
        return True
    if fallback_to_stdout:
        print(message)
        return True
    return False


def get_parent_output_channel() -> Callable[[str], None] | None:
    """Get the current parent output channel, or None if at top level.

    Use this for advanced cases where you need the raw channel.
    For simple printing, prefer parent_print().
    """
    return _parent_output_channel.get()


def set_parent_output_channel(
    channel: Callable[[str], None] | None,
) -> Token[Callable[[str], None] | None]:
    """Set the parent output channel. Returns token for reset.

    This is called by PythonREPL._capture_print() to set up the channel.
    Users should not normally call this directly.
    """
    return _parent_output_channel.set(channel)


def reset_parent_output_channel(
    token: Token[Callable[[str], None] | None],
) -> None:
    """Reset the parent output channel using a token.

    This is called by PythonREPL._capture_print() to restore the channel.
    Users should not normally call this directly.
    """
    _parent_output_channel.reset(token)
