"""Event definitions for the hook system.

Events are organized by type:
- repl_iteration: REPL iteration events (LLM query + execution)
- repl_execution: REPL execution events
- llm_query: LLM API call events
- invoke: Agent invocation events
- message: Message events
- budget: Budget tracking events
"""

from .budget import BudgetUpdate, BudgetUpdateContext
from .invoke import InvokeContext, InvokeEnter, InvokeExit, InvokeSpan
from .llm_query import LLMQueryContext, LLMQueryEnter, LLMQueryExit, LLMQuerySpan
from .llm_retry import LLMRetry, LLMRetryContext
from .message import Message, MessageContext
from .repl_execution import (
    ReplExecutionContext,
    ReplExecutionEnter,
    ReplExecutionExit,
    ReplExecutionSpan,
)
from .repl_iteration import (
    ReplIterationContext,
    ReplIterationEnter,
    ReplIterationExit,
    ReplIterationSpan,
)

__all__ = [
    # REPL iteration
    "ReplIterationEnter",
    "ReplIterationExit",
    "ReplIterationContext",
    "ReplIterationSpan",
    # REPL execution
    "ReplExecutionEnter",
    "ReplExecutionExit",
    "ReplExecutionContext",
    "ReplExecutionSpan",
    # LLM query
    "LLMQueryEnter",
    "LLMQueryExit",
    "LLMQueryContext",
    "LLMQuerySpan",
    # Invoke
    "InvokeEnter",
    "InvokeExit",
    "InvokeContext",
    "InvokeSpan",
    # Message
    "Message",
    "MessageContext",
    # Budget
    "BudgetUpdate",
    "BudgetUpdateContext",
    # LLM retry
    "LLMRetry",
    "LLMRetryContext",
]
