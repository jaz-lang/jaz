"""Budget update event.

This is a simple observational event with no enter/exit pair.
"""

from dataclasses import dataclass

from ..base import Event, ExecutionContext


@dataclass
class BudgetUpdate(Event):
    """Fired after budget status changes.

    This is informational only - hooks can record metrics but should not
    halt here. Halting should happen at the next action event (e.g., ReplIterationEnter).
    """

    budget_status: str
    cur_recursion_depth: int | None


@dataclass
class BudgetUpdateContext(ExecutionContext):
    """Context for budget update events.

    This is a read-only context - budget updates are informational.
    Hooks can only record metrics. Halting should happen at the next
    action event (e.g., ReplIterationEnter) rather than here.
    """

    pass
