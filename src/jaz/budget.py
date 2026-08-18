import threading
from datetime import UTC, datetime, timedelta
from typing import TypedDict


class LLMCallRecord(TypedDict):
    """Record of a single LLM API call"""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    depth: int
    iteration: int | None
    start_time: str
    end_time: str
    duration: float


class CostTrackerData(TypedDict):
    """Data structure for cost tracking that can be serialized to JSON.

    Schema note: this shape is written to disk (``BudgetPool(json_path=...)``), so run
    directories on disk outlive changes to it. The ``invoke_calls_budget`` field was
    dropped when the per-level nested-invoke cap was removed from ``Config`` — cost
    accounting is LLM-focused, and pool-level limits live on ``BudgetPool``. Anything
    reading *historical* cost trees (the analysis scripts under ``scripts/``) therefore
    has to tolerate both shapes: read it with ``.get(...)``, never ``[...]``.
    """

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

    # Lists
    llm_calls: list[LLMCallRecord]
    invoke_calls: list["CostTrackerData"]

    # Metadata
    depth: int

    # Timing
    start_time: str
    end_time: str
    duration: float


class CostTracker:
    """A per-invoke cost-accounting node in a BudgetPool's forest.

    Pure accounting: it records LLM calls, propagates local + subtree-cumulative
    spend up its tree, and serializes to the JSON cost tree. It owns **no budget** —
    the one budget, the running aggregate, and all enforcement live on
    :class:`~BudgetPool` (the pool). Each
    top-level invoke under the hook is its own root; nested invokes chain off their
    real parent (so the hook holds a *forest*, not a single tree).
    """

    def __init__(
        self,
        depth: int = 1,
        parent: "CostTracker | None" = None,
    ):
        """
        Initialize a cost-accounting node.

        Args:
            depth: Current recursion depth
            parent: Parent node for nested invocations
        """
        # Metadata
        self.depth = depth
        self.parent = parent

        # One lock shared across this node's tree (root creates it; descendants share
        # the root's). Serializes the cumulative read-modify-write accounting in
        # add_llm_call / add_invoke_call, which walk up to and mutate shared ancestors
        # — otherwise parallel sub-agents (running under a BudgetPool restored
        # into worker threads) would lose updates and under-count the tree. The hook's
        # enforcement reads (it sums the roots' totals) are intentionally NOT locked:
        # enforcement is best-effort / eventually-consistent, and a single-attribute
        # read is atomic under the GIL.
        #
        # A plain (non-reentrant) Lock suffices: add_llm_call / add_invoke_call are the
        # only acquirers and don't re-enter.
        self._lock: threading.Lock = (
            parent._lock if parent is not None else threading.Lock()
        )

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

    def start(self) -> None:
        """Mark the start time of this invoke call"""
        if self.start_time is not None:
            raise ValueError("Already started")
        self.start_time = datetime.now(UTC)

    def end(self) -> None:
        """Mark the end time of this invoke call and calculate duration"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        self.end_time = datetime.now(UTC)
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time

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
            "depth": self.depth,
            "iteration": iteration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
        }
        # Lock the cumulative accounting + parent-chain propagation: shared
        # ancestors are mutated by every descendant, so parallel sub-agents must
        # not interleave these read-modify-writes (one tree-wide lock).
        with self._lock:
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

    def add_invoke_call(self) -> "CostTracker":
        """
        Create and link a child accounting node for a nested invocation.

        Records the invoke call on this node (and propagates the cumulative count up
        the chain) and returns the child, which shares this tree's lock.
        """
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        # Lock the cumulative increment + parent propagation + child append as one
        # atomic step (shared state mutated by every parallel child).
        with self._lock:
            # Increment cumulative invoke calls counter
            self.total_invoke_calls += 1

            # Propagate cumulative count to all parents up the chain
            current = self.parent
            while current is not None:
                current.total_invoke_calls += 1
                current = current.parent

            # Create child node (shares this tree's lock via parent=self)
            child = CostTracker(
                depth=self.depth + 1,
                parent=self,
            )
            self.invoke_calls.append(child)

        return child

    def to_dict(self) -> CostTrackerData:
        """Convert tracker data to dictionary for JSON serialization"""
        if self.start_time is None:
            raise ValueError("Must call start() before calling any other methods")
        if self.end_time is None:
            raise ValueError("Must call end() before calling to_dict()")
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
            # Lists
            "llm_calls": self.llm_calls,
            "invoke_calls": [child.to_dict() for child in self.invoke_calls],
            # Metadata
            "depth": self.depth,
            # Timing
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": self.duration.total_seconds(),
        }
