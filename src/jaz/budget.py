import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

from jaz.exceptions import BudgetExhaustedError


class LLMCallRecord(TypedDict):
    """Record of a single LLM API call"""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    recursion_depth: int
    iteration: int | None
    start_time: str
    end_time: str
    duration: float


class CostTrackerData(TypedDict):
    """Data structure for cost tracking that can be serialized to JSON"""

    # Local (this level only)
    llm_cost: float
    llm_prompt_tokens: int
    llm_completion_tokens: int
    n_llm_calls: int
    n_invoke_calls: int

    # Cumulative (across entire subtree)
    total_llm_cost: float
    total_llm_prompt_tokens: int
    total_llm_completion_tokens: int
    total_llm_calls: int
    total_invoke_calls: int

    # Budgets and limits
    llm_cost_budget: float | None
    llm_cost_buffer: float
    llm_calls_budget: int | None
    llm_calls_buffer: int
    invoke_calls_budget: int | None

    # Lists
    llm_calls: list[LLMCallRecord]
    invoke_calls: list["CostTrackerData"]

    # Metadata
    recursion_depth: int

    # Timing
    start_time: str
    end_time: str
    duration: float


class CostTracker:
    """
    Tracks LLM API costs and invoke calls with hierarchical support for recursive
    invocations
    """

    def __init__(
        self,
        llm_cost_budget: float | None = None,
        llm_cost_buffer: float = 0.10,
        llm_calls_budget: int | None = None,
        llm_calls_buffer: int = 1,
        invoke_calls_budget: int | None = None,
        recursion_depth: int = 1,
        parent: "CostTracker | None" = None,
    ):
        """
        Initialize a cost tracker.

        Args:
            llm_cost_budget: Maximum LLM API cost in USD (None = no limit)
            llm_cost_buffer: Buffer amount in USD before hard error
            llm_calls_budget: Maximum number of LLM API calls (None = no limit)
            llm_calls_buffer: Buffer LLM calls before hard error
            invoke_calls_budget: Maximum number of invoke children at this level
                (None = no limit)
            recursion_depth: Current recursion depth
            parent: Parent tracker for nested invocations
        """
        # Budgets and limits
        self.llm_cost_budget = llm_cost_budget
        self.llm_cost_buffer = llm_cost_buffer
        self.llm_calls_budget = llm_calls_budget
        self.llm_calls_buffer = llm_calls_buffer
        self.invoke_calls_budget = invoke_calls_budget

        # Metadata
        self.recursion_depth = recursion_depth
        self.parent = parent

        # Local (this level only)
        self.llm_cost = 0.0
        self.llm_prompt_tokens = 0
        self.llm_completion_tokens = 0

        # Cumulative (across entire subtree)
        self.total_llm_cost = 0.0
        self.total_llm_prompt_tokens = 0
        self.total_llm_completion_tokens = 0
        self.total_llm_calls = 0
        self.total_invoke_calls = 0

        # Lists
        self.llm_calls: list[LLMCallRecord] = []
        self.invoke_calls: list[CostTracker] = []

        # Timing
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.duration: timedelta | None = None

    def get_root(self) -> "CostTracker":
        """Get the root tracker (the topmost parent in the hierarchy)"""
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    def start(self) -> None:
        """Mark the start time of this invoke call"""
        if self.start_time is not None:
            raise ValueError("Already started")
        self.start_time = datetime.now()

    def end(self) -> None:
        """Mark the end time of this invoke call and calculate duration"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        self.end_time = datetime.now()
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time

    def is_llm_cost_budget_exhausted(self) -> bool:
        """Check if LLM cost budget has been exhausted"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.llm_cost_budget is None:
            return False
        return self.total_llm_cost >= self.llm_cost_budget

    def is_llm_cost_buffer_exhausted(self) -> bool:
        """Check if LLM cost budget + buffer has been exhausted"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.llm_cost_budget is None:
            return False
        return self.total_llm_cost >= self.llm_cost_budget + self.llm_cost_buffer

    def get_llm_cost_buffer_exhausted_error(self) -> Exception | None:
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if (
            self.llm_cost_budget is None
            or self.total_llm_cost < self.llm_cost_budget + self.llm_cost_buffer
        ):
            return None
        return BudgetExhaustedError(
            f"LLM cost budget exhausted: "
            f"${self.total_llm_cost:.6f} spent, "
            f"budget was ${self.llm_cost_budget:.6f} "
            f"(buffer: ${self.llm_cost_buffer:.6f})"
        )

    def get_remaining_llm_cost_budget(self) -> float | None:
        """Get the remaining LLM cost budget (llm_cost_budget - total_llm_cost)"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.llm_cost_budget is None:
            return None
        return self.llm_cost_budget - self.total_llm_cost

    def is_llm_calls_budget_exhausted(self) -> bool:
        """Check if LLM calls budget has been exhausted"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.llm_calls_budget is None:
            return False
        return self.total_llm_calls >= self.llm_calls_budget

    def is_llm_calls_buffer_exhausted(self) -> bool:
        """Check if LLM calls budget + buffer has been exhausted"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.llm_calls_budget is None:
            return False
        return self.total_llm_calls >= self.llm_calls_budget + self.llm_calls_buffer

    def get_llm_calls_buffer_exhausted_error(self) -> Exception | None:
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if (
            self.llm_calls_budget is None
            or self.total_llm_calls < self.llm_calls_budget + self.llm_calls_buffer
        ):
            return None
        return BudgetExhaustedError(
            f"LLM calls budget exhausted: "
            f"{self.total_llm_calls} calls made, "
            f"budget was {self.llm_calls_budget} "
            f"(buffer: {self.llm_calls_buffer})"
        )

    def get_remaining_llm_calls_budget(self) -> int | None:
        """Get the remaining LLM calls budget (llm_calls_budget - total_llm_calls)"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.llm_calls_budget is None:
            return None
        return self.llm_calls_budget - self.total_llm_calls

    def get_llm_calls_budget_status(self, include_root: bool = False) -> str:
        """Get a formatted string describing LLM calls budget status.

        Args:
            include_root: If True and this is not the root tracker, also include
                the root tracker's budget status.

        Returns:
            Formatted budget status string.
        """
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.llm_calls_budget is None:
            status = (
                f"Total LLM calls so far: {self.total_llm_calls} (no budget limit set)"
            )
        else:
            remaining = self.llm_calls_budget - self.total_llm_calls
            percent_used = (
                (self.total_llm_calls / self.llm_calls_budget * 100)
                if self.llm_calls_budget > 0
                else 0
            )

            status = (
                f"LLM calls: {self.total_llm_calls} / {self.llm_calls_budget} "
                f"({percent_used:.1f}% used, {remaining} remaining)"
            )

            if self.is_llm_calls_budget_exhausted():
                over_budget = self.total_llm_calls - self.llm_calls_budget
                status += f"\n⚠️  BUDGET EXHAUSTED by {over_budget} calls!"
                if self.is_llm_calls_buffer_exhausted():
                    status += (
                        " Buffer also exhausted - will error unless the REPL exits!"
                    )
                else:
                    status += f" Within buffer of {self.llm_calls_buffer}."

        # Include root tracker status if requested and this is not the root
        if include_root and self.parent is not None:
            root = self.get_root()
            status += f"; ROOT {root.get_llm_calls_budget_status(include_root=False)}"

        return status

    def get_llm_cost_budget_status(self, include_root: bool = False) -> str:
        """Get a formatted string describing LLM cost budget status.

        Args:
            include_root: If True and this is not the root tracker, also include
                the root tracker's budget status.

        Returns:
            Formatted budget status string.
        """
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.llm_cost_budget is None:
            status = (
                f"Total cost so far: ${self.total_llm_cost:.6f} (no budget limit set)"
            )
        else:
            remaining = self.llm_cost_budget - self.total_llm_cost
            percent_used = (
                (self.total_llm_cost / self.llm_cost_budget * 100)
                if self.llm_cost_budget > 0
                else 0
            )

            status = (
                f"Budget: ${self.total_llm_cost:.6f} / ${self.llm_cost_budget:.6f} "
                f"({percent_used:.1f}% used, ${remaining:.6f} remaining)"
            )

            if self.is_llm_cost_budget_exhausted():
                over_budget = self.total_llm_cost - self.llm_cost_budget
                status += f"\n⚠️  BUDGET EXHAUSTED by ${over_budget:.6f}!"
                if self.is_llm_cost_buffer_exhausted():
                    # This should never happen, but just in case
                    status += (
                        " Buffer also exhausted - will error unless the REPL exits!"
                    )
                else:
                    status += f" Within buffer of ${self.llm_cost_buffer:.6f}."

        # Include root tracker status if requested and this is not the root
        if include_root and self.parent is not None:
            root = self.get_root()
            status += f"; ROOT {root.get_llm_cost_budget_status(include_root=False)}"

        return status

    def add_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost: float,
        iteration: int | None,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Record an LLM API call.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total number of tokens
            cost: Cost in USD
            iteration: REPL iteration number, if call was made inside REPL
            start_time: Start time of the LLM call
            end_time: End time of the LLM call
        """
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        record: LLMCallRecord = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "recursion_depth": self.recursion_depth,
            "iteration": iteration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
        }
        self.llm_calls.append(record)
        self.llm_cost += cost
        self.llm_prompt_tokens += prompt_tokens
        self.llm_completion_tokens += completion_tokens
        self.total_llm_cost += cost
        self.total_llm_prompt_tokens += prompt_tokens
        self.total_llm_completion_tokens += completion_tokens
        self.total_llm_calls += 1

        # Propagate cost, tokens, and count to all parents up the chain
        current = self.parent
        while current is not None:
            current.total_llm_cost += cost
            current.total_llm_prompt_tokens += prompt_tokens
            current.total_llm_completion_tokens += completion_tokens
            current.total_llm_calls += 1
            current = current.parent

    def get_invoke_calls_status(self) -> str:
        """Get a formatted string describing invoke calls status"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        num_invoke_calls = len(self.invoke_calls)

        if self.invoke_calls_budget is None:
            return (
                f"jaz.invoke calls made at this level: {num_invoke_calls} "
                f"(no budget limit set)"
            )

        remaining = self.invoke_calls_budget - num_invoke_calls

        status = (
            "jaz.invoke calls made at this level: "
            f"{num_invoke_calls} / {self.invoke_calls_budget} "
            f"({remaining} remaining)"
        )

        if num_invoke_calls >= self.invoke_calls_budget:
            status += "\n⚠️  Per-level budget reached! Further calls will raise a BudgetExhaustedError."

        return status

    def add_invoke_call(
        self,
        invoke_calls_budget: int,
        llm_cost_budget: float | None = None,
        llm_calls_budget: int | None = None,
    ) -> "CostTracker":
        """
        Create a child tracker for a nested invocation.

        This also records the invoke call in the parent tracker.

        Args:
            llm_cost_budget: LLM cost budget for the child tracker. If None, uses
                parent's remaining budget.
            llm_calls_budget: LLM calls budget for the child tracker. If None, uses
                parent's remaining budget.
            invoke_calls_budget: Per-level invoke children budget for the child.
                If None, inherits from parent.

        Raises:
            BudgetExhaustedError: If the per-level invoke children budget is exhausted.
        """
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        # Default to remaining budget if not specified
        if llm_cost_budget is None:
            llm_cost_budget = self.get_remaining_llm_cost_budget()
        if llm_calls_budget is None:
            llm_calls_budget = self.get_remaining_llm_calls_budget()

        num_invoke_calls = len(self.invoke_calls)

        # Check per-level invoke children budget BEFORE creating child
        if self.invoke_calls_budget is not None:
            if num_invoke_calls >= self.invoke_calls_budget:
                raise BudgetExhaustedError(
                    "jaz.invoke calls budget exhausted at recursion depth "
                    f"{self.recursion_depth}: "
                    f"{num_invoke_calls} calls made, "
                    f"budget was {self.invoke_calls_budget}"
                )

        # Increment cumulative invoke calls counter
        self.total_invoke_calls += 1

        # Propagate cumulative count to all parents up the chain
        current = self.parent
        while current is not None:
            current.total_invoke_calls += 1
            current = current.parent

        # Create child tracker
        child = CostTracker(
            llm_cost_budget=llm_cost_budget,
            llm_cost_buffer=self.llm_cost_buffer,
            llm_calls_budget=llm_calls_budget,
            llm_calls_buffer=self.llm_calls_buffer,
            invoke_calls_budget=invoke_calls_budget,
            recursion_depth=self.recursion_depth + 1,
            parent=self,
        )
        self.invoke_calls.append(child)

        return child

    def to_dict(self) -> CostTrackerData:
        """Convert tracker data to dictionary for JSON serialization"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.end_time is None:
            raise ValueError(
                "Must call end() before calling to_dict() or save_to_json()"
            )
        assert self.duration is not None
        return {
            # Local (this level only)
            "llm_cost": self.llm_cost,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "n_llm_calls": len(self.llm_calls),
            "n_invoke_calls": len(self.invoke_calls),
            # Cumulative (across entire subtree)
            "total_llm_cost": self.total_llm_cost,
            "total_llm_prompt_tokens": self.total_llm_prompt_tokens,
            "total_llm_completion_tokens": self.total_llm_completion_tokens,
            "total_llm_calls": self.total_llm_calls,
            "total_invoke_calls": self.total_invoke_calls,
            # Budgets and limits
            "llm_cost_budget": self.llm_cost_budget,
            "llm_cost_buffer": self.llm_cost_buffer,
            "llm_calls_budget": self.llm_calls_budget,
            "llm_calls_buffer": self.llm_calls_buffer,
            "invoke_calls_budget": self.invoke_calls_budget,
            # Lists
            "llm_calls": self.llm_calls,
            "invoke_calls": [child.to_dict() for child in self.invoke_calls],
            # Metadata
            "recursion_depth": self.recursion_depth,
            # Timing
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": self.duration.total_seconds(),
        }

    def save_to_json(self, path: str | Path) -> None:
        """Save cost tracking data to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Cost tracking data saved to {path}")
