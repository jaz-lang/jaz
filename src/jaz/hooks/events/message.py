"""Message event.

This is a simple observational event with no enter/exit pair.
"""

from dataclasses import dataclass

from jaz.providers import MessageDict

from ..base import Event, ExecutionContext


@dataclass
class Message(Event):
    """Fired when a message is added to the conversation.

    This is observational only - hooks can record metrics but cannot
    influence execution or modify messages.
    """

    message: MessageDict


@dataclass
class MessageContext(ExecutionContext):
    """Context for message events.

    This is a read-only context - messages are informational.
    Hooks can only record metrics, not halt or modify.
    """

    pass
