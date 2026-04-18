"""Interactive objects for agent communication during REPL execution.

This module provides reusable interactive objects that hooks can inject
into the REPL environment, enabling bidirectional communication between
hooks and agents.
"""

from .checkbox import Checkbox
from .multi_select import MultiSelect, Option
from .multiple_choice import Choice, MultipleChoice

__all__ = [
    "Checkbox",
    "Choice",
    "MultipleChoice",
    "MultiSelect",
    "Option",
]
