"""LLM retry event.

This is a simple observational event with no enter/exit pair.
Fired when an LLM call fails and tenacity retries it.
"""

from dataclasses import dataclass

from ..base import Event, ExecutionContext


@dataclass
class LLMRetry(Event):
    """Fired when an LLM call fails and is about to be retried.

    This is informational only - hooks can observe retries and record metrics
    but should not halt here.
    """

    invoke_id: str
    model: str
    attempt_number: int
    exception: Exception
    wait_seconds: float
    iteration: int | None
    cur_recursion_depth: int | None


@dataclass
class LLMRetryContext(ExecutionContext):
    """Context for LLM retry events.

    This is a read-only context - LLM retries are informational.
    Hooks can only record metrics.
    """

    pass
