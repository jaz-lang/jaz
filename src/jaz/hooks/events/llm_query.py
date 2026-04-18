"""LLM query event, context, and span."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from jaz.providers import MessageDict

from ..base import Event, HaltableContext


@dataclass
class LLMQueryEnter(Event):
    """Fired before making an LLM API call.

    Hooks can halt execution or override model parameters.
    """

    invoke_id: str
    messages: list[MessageDict]
    model: str
    iteration: int | None
    cur_recursion_depth: int | None


@dataclass
class LLMQueryExit(Event):
    """Fired after receiving LLM response.

    This is observational only - the query is complete.
    Hooks can record metrics but cannot influence execution.
    """

    invoke_id: str
    response_content: str | None
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: float | None
    iteration: int | None
    cur_recursion_depth: int | None
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class LLMQueryContext(HaltableContext):
    """Context for LLM query events.

    Hooks can:
    - Halt execution (prevent the query)
    """

    pass


class LLMQuerySpan:
    """Span for LLM query.

    Usage:
        with dispatcher.span_llm_query(...) as span:
            if span.ctx.should_halt:
                _raise_halt_errors(span.ctx.halt_errors)  # ExceptionGroup if multiple

            response = llm.query(messages, **base_config)
            span.complete(
                response_content=response,
                prompt_tokens=...,
                completion_tokens=...,
                cost=...
            )
    """

    def __init__(self, ctx: LLMQueryContext) -> None:
        self.ctx = ctx
        self._completed: bool = False
        self._response_content: str | None = None
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None
        self._cost: float | None = None
        self._iteration: int | None = None
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

    def complete(
        self,
        *,
        response_content: str | None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost: float | None = None,
        iteration: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        """Complete span with LLM response data.

        Args:
            response_content: The LLM's response text
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            cost: Cost of the query in dollars
            iteration: The REPL iteration number
            start_time: When the LLM call started
            end_time: When the LLM call finished

        Raises:
            RuntimeError: If span was already completed
        """
        if self._completed:
            raise RuntimeError("Span already completed")
        self._response_content = response_content
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cost = cost
        self._iteration = iteration
        self._start_time = start_time
        self._end_time = end_time
        self._completed = True

    def is_completed(self) -> bool:
        """Check if span was completed."""
        return self._completed

    def get_response_content(self) -> str | None:
        """Get the LLM response content.

        Returns:
            The response content provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        return self._response_content

    def get_prompt_tokens(self) -> int | None:
        """Get the prompt tokens count.

        Returns:
            The prompt tokens provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        return self._prompt_tokens

    def get_completion_tokens(self) -> int | None:
        """Get the completion tokens count.

        Returns:
            The completion tokens provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        return self._completion_tokens

    def get_cost(self) -> float | None:
        """Get the query cost.

        Returns:
            The cost provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        return self._cost

    def get_iteration(self) -> int | None:
        """Get the iteration number.

        Returns:
            The iteration provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        return self._iteration

    def get_start_time(self) -> datetime | None:
        """Get the LLM call start time.

        Returns:
            The start_time provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        return self._start_time

    def get_end_time(self) -> datetime | None:
        """Get the LLM call end time.

        Returns:
            The end_time provided to complete()

        Raises:
            RuntimeError: If span was not completed
        """
        if not self._completed:
            raise RuntimeError("Span not completed")
        return self._end_time
