"""WorkflowStrategyHook - Multi-select decision points for workflow strategies."""

from __future__ import annotations

from dataclasses import dataclass

from jaz.exceptions import _JazInternalError
from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import AddInstructionPrompt, AddReplInput, Effect
from jaz.hooks.events import InvokeEnter, InvokeExit, ReplIterationEnter
from jaz.interactive_objects import MultiSelect

from .options import (
    CONTEXT_FILTER_OPTIONS,
    INVOKE_START_OPTIONS,
    STEP_STRATEGY_OPTIONS,
)
from .prompts import (
    CONTEXT_FILTER_PROMPTS,
    INVOKE_START_PROMPTS,
    STEP_STRATEGY_PROMPTS,
    compose_prompts,
)


@dataclass
class InvokeState:
    """State tracked per invoke for decision point handling."""

    invoke_id: str
    iteration: int = 0
    invoke_start_selection: MultiSelect | None = None
    invoke_start_prompt_added: bool = False
    step_strategy_selection: MultiSelect | None = None
    context_filter_selection: MultiSelect | None = None


class WorkflowStrategyHook(Hook):
    """Hook that provides multi-select decision points for workflow strategies.

    This hook supports 4 decision points with combinable options:

    1. **Invoke Start (DP1)**: At the beginning of each jaz.invoke
       - Options: simple, search, extra_guidance, decompose, refinement
       - simple is exclusive; others are combinable

    2. **Step Strategy (DP2)**: At every step (including 0th)  # TODO: Implement this properly
       - Options: simple, search, extra_guidance
       - simple is exclusive; others are combinable

    3. **Context Filter (DP3)**: At every step (excluding 0th)  # TODO: Implement this properly
       - Options: context_filter, no_filter
       - Yes/no decision

    4. **Post Invoke (DP4)**: After every jaz.invoke completes  # TODO: Implement this properly
       - Options: refinement, accept
       - Observational only (cannot add effects)

    Attributes:
        enable_invoke_start: Enable decision point 1 (default: True)
        enable_step_strategy: Enable decision point 2 (default: False)
        enable_context_filter: Enable decision point 3 (default: False)
        enable_post_invoke: Enable decision point 4 (default: False)

    Example:
        >>> hook = WorkflowStrategyHook(
        ...     enable_invoke_start=True,
        ...     enable_step_strategy=True,
        ... )
        >>> with hook:
        ...     result = invoke(MyAgent, prompt="Build a web app")
    """

    def __init__(
        self,
        enable_invoke_start: bool = True,
        enable_step_strategy: bool = False,
        enable_context_filter: bool = False,
        enable_post_invoke: bool = False,
    ) -> None:
        """Initialize the hook.

        Args:
            enable_invoke_start: Enable decision point 1 (invoke start)
            enable_step_strategy: Enable decision point 2 (step strategy)
            enable_context_filter: Enable decision point 3 (context filter)
            enable_post_invoke: Enable decision point 4 (post invoke, observational)
        """
        self.enable_invoke_start = enable_invoke_start
        self.enable_step_strategy = enable_step_strategy
        self.enable_context_filter = enable_context_filter
        self.enable_post_invoke = enable_post_invoke

        # Track state per invoke_id to handle nested invokes
        self._invoke_states: dict[str, InvokeState] = {}

    def on_event(self, event: Event) -> list[Effect]:
        """Handle events and inject workflow choices at decision points."""
        match event:
            case InvokeEnter():
                return self._handle_invoke_enter(event)
            case ReplIterationEnter():
                return self._handle_repl_iteration_enter(event)
            case InvokeExit():
                return self._handle_invoke_exit(event)
            case _:
                return []

    def _create_state(self, invoke_id: str) -> InvokeState:
        """Create state for an invoke."""
        assert invoke_id not in self._invoke_states, (
            f"State for invoke {invoke_id} already exists"
        )
        self._invoke_states[invoke_id] = InvokeState(invoke_id=invoke_id)
        return self._invoke_states[invoke_id]

    def _get_state(self, invoke_id: str) -> InvokeState:
        """Get state for an invoke."""
        return self._invoke_states[invoke_id]

    def _handle_invoke_enter(self, event: InvokeEnter) -> list[Effect]:
        """Handle InvokeEnter - Decision Point 1."""
        # Only activate if we can still recurse
        if event.cur_recursion_depth >= event.max_recursion_depth:
            return []

        # Always create state for tracking (needed by DP2, DP3, DP4 even if DP1 is disabled)
        state = self._create_state(event.invoke_id)

        if not self.enable_invoke_start:
            return []

        # Create the MultiSelect for invoke start
        state.invoke_start_selection = MultiSelect(
            prompt="Select workflow strategies for this invoke:",
            options=INVOKE_START_OPTIONS,
            exclusive_keys=frozenset({"simple"}),
        )

        # Build the instruction prompt
        prompt_text = f"""
IMPORTANT: Before doing anything else, select your workflow strategy.

{str(state.invoke_start_selection)}

Use these methods to make your selection:
- `workflow.select("key")` or `workflow.select(index)` to select options
- `workflow.select("key1", "key2")` to select multiple at once
- `workflow.deselect("key")` to remove a selection
- `workflow.confirm()` to confirm your selection and proceed

Note: Selecting "simple" will clear other selections. Selecting any other option will clear "simple".
"""
        # TODO: Refactor the last `Note` on exclusivity into __str__ method of MultiSelect

        return [
            AddReplInput(
                key="workflow",
                value=state.invoke_start_selection,
                reason="Workflow strategy selection for invoke",
            ),
            AddInstructionPrompt(text=prompt_text),
        ]

    def _handle_repl_iteration_enter(self, event: ReplIterationEnter) -> list[Effect]:
        """Handle ReplIterationEnter - Decision Points 1 (prompt), 2, and 3."""
        # Only activate if we can still recurse
        if event.cur_recursion_depth >= event.max_recursion_depth:
            return []

        effects: list[Effect] = []
        state = self._get_state(event.invoke_id)

        # Track iteration
        state.iteration = event.iteration

        # DP1: Add invoke_start prompt if selection was confirmed but prompt not added yet
        if self.enable_invoke_start and not state.invoke_start_prompt_added:
            dp1_effects = self._add_invoke_start_prompt(state)
            effects.extend(dp1_effects)

        # DP2: Step strategy (every step)
        # First add prompt from previous step's confirmed selection, then inject new
        # TODO: Implement this properly
        if self.enable_step_strategy:
            dp2_prompt_effects = self._add_step_strategy_prompt(state)
            effects.extend(dp2_prompt_effects)
            dp2_inject_effects = self._inject_step_strategy(state, event)
            effects.extend(dp2_inject_effects)

        # DP3: Context filter (every step except 0th)
        # First add prompt from previous step's confirmed selection, then inject new
        # TODO: Implement this properly
        if self.enable_context_filter and event.iteration > 0:
            dp3_prompt_effects = self._add_context_filter_prompt(state)
            effects.extend(dp3_prompt_effects)
            dp3_inject_effects = self._inject_context_filter(state, event)
            effects.extend(dp3_inject_effects)

        # TODO: DP4 post-invoke prompt

        return effects

    def _add_invoke_start_prompt(self, state: InvokeState) -> list[Effect]:
        """Add the workflow prompt based on invoke start selection."""
        if state.invoke_start_selection is None:
            raise _JazInternalError(
                "invoke_start_selection should exist from InvokeEnter prompt hook when enable_invoke_start is True, but is None"
            )

        # Check if selection is confirmed
        if not state.invoke_start_selection.is_confirmed:
            return []

        selected = state.invoke_start_selection.selected_keys
        if not selected:
            return []

        # Compose the prompt from selected strategies
        prompt = compose_prompts(INVOKE_START_PROMPTS, selected)

        state.invoke_start_prompt_added = True
        return [AddInstructionPrompt(text=prompt)]

    def _add_step_strategy_prompt(self, state: InvokeState) -> list[Effect]:
        """Add the strategy prompt based on previous step's confirmed selection."""
        # TODO: Implement this properly
        if state.step_strategy_selection is None:
            return []

        # Check if selection is confirmed
        if not state.step_strategy_selection.is_confirmed:
            return []

        selected = state.step_strategy_selection.selected_keys
        if not selected:
            return []

        # Compose the prompt from selected strategies
        prompt = compose_prompts(STEP_STRATEGY_PROMPTS, selected)
        if not prompt:
            return []

        return [AddInstructionPrompt(text=prompt)]

    def _inject_step_strategy(
        self, state: InvokeState, event: ReplIterationEnter
    ) -> list[Effect]:
        """Inject step strategy selection for this step."""
        # TODO: Implement this properly
        state.step_strategy_selection = MultiSelect(
            prompt=f"Select strategy for step {event.iteration}:",
            options=STEP_STRATEGY_OPTIONS,
            exclusive_keys=frozenset({"simple"}),
        )

        prompt_text = f"""
## Step {event.iteration} Strategy

{str(state.step_strategy_selection)}

Select your strategy for this step using `step_strategy.select(...)` and `step_strategy.confirm()`.
"""

        return [
            AddReplInput(
                key="step_strategy",
                value=state.step_strategy_selection,
                reason=f"Step strategy selection for step {event.iteration}",
            ),
            AddInstructionPrompt(text=prompt_text),
        ]

    def _add_context_filter_prompt(self, state: InvokeState) -> list[Effect]:
        """Add the context filter prompt based on previous step's confirmed selection."""
        # TODO: Implement this properly
        if state.context_filter_selection is None:
            return []

        # Check if selection is confirmed
        if not state.context_filter_selection.is_confirmed:
            return []

        selected = state.context_filter_selection.selected_keys
        if not selected:
            return []

        # Compose the prompt from selected options
        prompt = compose_prompts(CONTEXT_FILTER_PROMPTS, selected)
        if not prompt:
            return []

        return [AddInstructionPrompt(text=prompt)]

    def _inject_context_filter(
        self, state: InvokeState, event: ReplIterationEnter
    ) -> list[Effect]:
        """Inject context filter selection for this step."""
        # TODO: Implement this properly
        state.context_filter_selection = MultiSelect(
            prompt=f"Apply context filtering for step {event.iteration}?",
            options=CONTEXT_FILTER_OPTIONS,
            exclusive_keys=frozenset({"context_filter", "no_filter"}),
        )

        prompt_text = f"""
## Context Filtering for Step {event.iteration}

{str(state.context_filter_selection)}

Select whether to filter context using `context_filter.select(...)` and `context_filter.confirm()`.
"""

        return [
            AddReplInput(
                key="context_filter",
                value=state.context_filter_selection,
                reason=f"Context filter selection for step {event.iteration}",
            ),
            AddInstructionPrompt(text=prompt_text),
        ]

    def _handle_invoke_exit(self, event: InvokeExit) -> list[Effect]:
        """Handle InvokeExit - Decision Point 4 (observational only).

        Note: InvokeExit is observational-only, so we cannot return effects.
        This method logs information for potential external refinement handling.
        """
        # TODO: Implement post-invoke properly

        if not self.enable_post_invoke:
            # Clean up state
            self._invoke_states.pop(event.invoke_id, None)
            return []

        state = self._invoke_states.get(event.invoke_id)
        if state:
            # Log refinement opportunity (observational)
            # External code can inspect state.invoke_start_selection to see if
            # refinement was requested and implement the refinement pattern
            pass

        # Clean up state for this invoke
        self._invoke_states.pop(event.invoke_id, None)
        return []


__all__ = ["WorkflowStrategyHook", "InvokeState"]
