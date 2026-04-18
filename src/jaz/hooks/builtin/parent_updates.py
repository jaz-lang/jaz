"""Hook that enables child agents to send formatted updates to parent's stdout."""

from dataclasses import dataclass

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.hooks.effects import AddReplInput, AddSystemPrompt, Effect
from jaz.hooks.events import InvokeEnter
from jaz.parent_output import get_parent_output_channel


@dataclass
class ParentUpdatesHook(Hook):
    """Hook that enables child agents to send updates to their parent's stdout.

    When active, this hook:
    1. Adds a system prompt encouraging the agent to report progress
    2. Injects an update(message) function into the REPL

    The update() function formats messages according to message_format and
    prints them to the parent's REPL output.

    Attributes:
        message_format: Format string for messages. Placeholders:
            - {message}: The update message
            - {invoke_id}: The child's invoke ID
        system_prompt: Text added to system prompt (empty string to disable)
        function_name: Name of the update function in REPL (default: "update")

    Example:
        with ParentUpdatesHook():
            result = jaz.invoke("Complex task", return_type=str)

        # In child agent's REPL:
        update("Starting phase 1...")
        # Parent sees: "Jaz invoke abc-123 update: Starting phase 1..."
    """

    message_format: str = "Jaz invoke {invoke_id} update: {message}"
    system_prompt: str = (
        "\n\nWhen working on complex or multi-step tasks, use the `update(message)` "
        "function to report your progress. This sends status updates to the parent "
        "context and helps track nested agent execution."
    )
    function_name: str = "update"

    def on_event(self, event: Event) -> list[Effect]:
        if not isinstance(event, InvokeEnter):
            return []

        effects: list[Effect] = []

        # Add system prompt if configured
        if self.system_prompt:
            effects.append(AddSystemPrompt(text=self.system_prompt))

        # Capture for closure
        invoke_id = event.invoke_id
        msg_format = self.message_format

        # CRITICAL: Capture the parent channel NOW, before child REPL overwrites it
        parent_channel = get_parent_output_channel()

        def update(message: str) -> None:
            """Send an update message to the parent agent's output."""
            formatted = msg_format.format(message=message, invoke_id=invoke_id)
            # Use the captured channel directly, not parent_print()
            if parent_channel is not None:
                parent_channel(formatted)
            else:
                print(formatted)

        effects.append(
            AddReplInput(
                key=self.function_name,
                value=update,
                reason="Parent update function for child-to-parent communication",
            )
        )

        return effects
