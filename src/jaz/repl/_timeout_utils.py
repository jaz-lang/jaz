"""
Timeout utilities for exec and eval operations to prevent infinite loops.

Uses sys.settrace() for timeout enforcement without creating new threads.
This is critical for avoiding deadlocks with reentrant locks in recursive calls.

NOTE: sys.settrace() is thread-local in Python, so each thread has its own
trace function and they don't interfere with each other.

TODO: This doesn't work when REPL input launches new threads, because they don't
inherit the trace.
"""

import sys
import time
from types import CodeType


class TimeoutError(Exception):
    """Raised when execution exceeds the specified timeout."""

    pass


def exec_with_timeout(
    code: str | CodeType,
    globals_dict: dict[str, object],
    timeout: float | None,
) -> None:
    """
    Execute code with a timeout using sys.settrace().

    This implementation does not create any new threads, making it safe to use
    with reentrant locks and recursive calls.

    Args:
        code: Compiled code object to execute
        globals_dict: Global namespace for execution
        timeout: Timeout in seconds (None = no timeout)

    Raises:
        TimeoutError: If execution exceeds timeout
    """
    # TODO: This is best-effort and not guaranteed to work,
    # e.g., if the code spawns new threads that don't inherit the trace,
    # or does `subprocess.run`
    if timeout is None:
        exec(code, globals_dict)
        return

    start_time = time.time()
    old_trace = sys.gettrace()

    def trace_timeout(frame, event, arg):  # noqa: ARG001
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Execution exceeded timeout of {timeout} seconds. "
                f"This may indicate an infinite loop."
            )
        return trace_timeout

    try:
        sys.settrace(trace_timeout)
        exec(code, globals_dict)
    finally:
        sys.settrace(old_trace)


def eval_with_timeout(
    code: str | CodeType,
    globals_dict: dict[str, object],
    timeout: float | None,
) -> object:
    """
    Evaluate code with a timeout using sys.settrace().

    This implementation does not create any new threads, making it safe to use
    with reentrant locks and recursive calls.

    Args:
        code: Compiled code object to evaluate
        globals_dict: Global namespace for evaluation
        timeout: Timeout in seconds (None = no timeout)

    Returns:
        Result of evaluation

    Raises:
        TimeoutError: If evaluation exceeds timeout
    """
    # TODO: This is best-effort and not guaranteed to work,
    # e.g., if the code spawns new threads that don't inherit the trace,
    # or does `subprocess.run`
    if timeout is None:
        return eval(code, globals_dict)

    start_time = time.time()
    old_trace = sys.gettrace()

    def trace_timeout(frame, event, arg):  # noqa: ARG001
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Evaluation exceeded timeout of {timeout} seconds. "
                f"This may indicate an infinite loop."
            )
        return trace_timeout

    try:
        sys.settrace(trace_timeout)
        result = eval(code, globals_dict)
        return result
    finally:
        sys.settrace(old_trace)
