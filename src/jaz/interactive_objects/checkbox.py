"""Checkbox interactive object for simple yes/no agent decisions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Checkbox:
    """A simple checkbox that agents can check by calling it.

    This provides a minimal interface for yes/no decisions. The agent
    checks the box by calling the object as a function. If the agent
    doesn't call it, the box remains unchecked.

    Attributes:
        prompt: Description shown to the agent explaining what checking means

    Example:
        >>> explore = Checkbox("Check to explore multiple approaches")
        >>> explore.is_checked
        False
        >>> explore()  # Agent calls this to check the box
        >>> explore.is_checked
        True
    """

    prompt: str
    _checked: bool = field(default=False, repr=False)

    def __call__(self) -> None:
        """Check the checkbox.

        Call this method to indicate a positive decision.
        """
        self._checked = True
        print(f"Checked: {self.prompt}")

    @property
    def is_checked(self) -> bool:
        """Whether the checkbox has been checked."""
        return self._checked

    def reset(self) -> None:
        """Reset the checkbox to unchecked state."""
        self._checked = False

    def __str__(self) -> str:
        """Human-readable representation."""
        marker = "[x]" if self._checked else "[ ]"
        return f"{marker} {self.prompt}"

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return f"Checkbox(prompt={self.prompt!r}, checked={self._checked})"
