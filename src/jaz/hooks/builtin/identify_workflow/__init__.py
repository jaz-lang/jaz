"""Workflow strategy hooks for multi-select decision points.

This module provides the WorkflowStrategyHook which enables agents to select
from multiple combinable workflow strategies at various decision points.

Decision Points:
    1. Invoke Start: At the beginning of each jaz.invoke
    2. Step Strategy: At every step (including 0th)  # TODO: Implement this properly
    3. Context Filter: At every step (excluding 0th)  # TODO: Implement this properly
    4. Post Invoke: After every jaz.invoke (observational)  # TODO: Implement this properly

Example:
    >>> from jaz.hooks import WorkflowStrategyHook
    >>> hook = WorkflowStrategyHook(
    ...     enable_invoke_start=True,
    ...     enable_step_strategy=True,
    ... )
    >>> with hook:
    ...     result = invoke(MyAgent, prompt="Build a web app")
"""

from .options import (
    CONTEXT_FILTER_OPTIONS,
    INVOKE_START_OPTIONS,
    POST_INVOKE_OPTIONS,
    STEP_STRATEGY_OPTIONS,
)
from .prompts import (
    CONTEXT_FILTER_PROMPTS,
    INVOKE_START_PROMPTS,
    POST_INVOKE_PROMPTS,
    STEP_STRATEGY_PROMPTS,
    compose_prompts,
)
from .workflow_strategy_hook import InvokeState, WorkflowStrategyHook

# Backward compatibility alias
IdentifyWorkflowHook = WorkflowStrategyHook

__all__ = [
    # Main exports
    "WorkflowStrategyHook",
    "InvokeState",
    # Backward compatibility
    "IdentifyWorkflowHook",
    # Options
    "INVOKE_START_OPTIONS",
    "STEP_STRATEGY_OPTIONS",
    "CONTEXT_FILTER_OPTIONS",
    "POST_INVOKE_OPTIONS",
    # Prompts
    "INVOKE_START_PROMPTS",
    "STEP_STRATEGY_PROMPTS",
    "CONTEXT_FILTER_PROMPTS",
    "POST_INVOKE_PROMPTS",
    "compose_prompts",
]
