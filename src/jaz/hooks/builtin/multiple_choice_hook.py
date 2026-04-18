"""Hook that presents a multiple choice question to agents at invoke start."""

from __future__ import annotations

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import AddInstructionPrompt, AddReplInput, Effect
from jaz.hooks.events import InvokeEnter
from jaz.interactive_objects import Choice, MultipleChoice


class MultipleChoiceHook(Hook):
    """Hook that presents a multiple choice question at invoke start.

    This hook injects a MultipleChoice object into the REPL and adds
    a prompt instructing the agent to make a selection before proceeding.

    Attributes:
        prompt: The question to present to the agent
        choices: List of Choice objects representing available options
        variable_name: Name of the variable in the REPL (default: "choice")
        choice_object: The MultipleChoice object after creation (for inspection)

    Example:
        >>> hook = MultipleChoiceHook(
        ...     prompt="How would you like to approach this task?",
        ...     choices=[
        ...         Choice("quick", "Quick and simple - minimal code"),
        ...         Choice("thorough", "Thorough and detailed - comprehensive"),
        ...     ],
        ... )
        >>> with hook:
        ...     result = invoke(MyAgent, prompt="Solve this problem")
        >>> print(hook.choice_object.selected_key)
        'quick'
    """

    def __init__(
        self,
        prompt: str,
        choices: list[Choice],
        variable_name: str = "choice",
    ):
        """Initialize the hook.

        Args:
            prompt: The question to present to the agent
            choices: List of Choice objects representing available options
            variable_name: Name of the variable in the REPL (default: "choice")
        """
        self.prompt = prompt
        self.choices = choices
        self.variable_name = variable_name
        self.choice_object: MultipleChoice | None = None

    def on_event(self, event: Event) -> list[Effect]:
        """Handle events and inject the multiple choice on InvokeEnter."""
        if not isinstance(event, InvokeEnter):
            return []

        # Create the MultipleChoice object
        self.choice_object = MultipleChoice(
            prompt=self.prompt,
            choices=self.choices,
        )

        # Build prompt addition using str() to show full descriptions
        prompt_text = f"""
IMPORTANT: Before doing anything else, you must make a selection.

{str(self.choice_object)}

Call `{self.variable_name}.select(index)` or `{self.variable_name}.select("key")` to make your choice, then proceed with your task.
"""

        return [
            AddReplInput(
                key=self.variable_name,
                value=self.choice_object,
                reason="Multiple choice selection for agent",
            ),
            AddInstructionPrompt(
                text=prompt_text,
            ),
        ]
