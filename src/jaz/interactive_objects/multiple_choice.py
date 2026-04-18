"""Multiple choice interactive object for agent interaction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Choice:
    """A single choice option.

    Attributes:
        key: Short identifier for selection (e.g., "quick")
        description: Longer explanation shown to agent (e.g., "Quick and simple approach")
    """

    key: str
    description: str


@dataclass
class MultipleChoice:
    """A multiple choice prompt supporting selection by index or key.

    This object can be injected into a REPL via AddReplInput effect,
    allowing agents to make selections during execution.

    Attributes:
        prompt: The question or instruction shown to the agent
        choices: List of Choice objects representing available options

    Example:
        >>> mc = MultipleChoice(
        ...     prompt="Select an approach",
        ...     choices=[
        ...         Choice("quick", "Quick and simple"),
        ...         Choice("thorough", "Thorough and detailed"),
        ...     ]
        ... )
        >>> mc.select(1)  # Select by index
        'quick'
        >>> mc.select("thorough")  # Select by key
        'thorough'
    """

    prompt: str
    choices: list[Choice]
    _selected: Choice | None = field(default=None, repr=False)

    def __str__(self) -> str:
        """Human-readable representation with descriptions."""
        lines = [f"MultipleChoice: {self.prompt}"]
        for i, choice in enumerate(self.choices, 1):
            marker = ">" if choice == self._selected else " "
            lines.append(f"  {marker} {i}. [{choice.key}] {choice.description}")
        if self._selected:
            lines.append(f"\nCurrently selected: {self._selected.key}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        keys = [c.key for c in self.choices]
        selected = self._selected.key if self._selected else None
        return (
            f"MultipleChoice(prompt={self.prompt!r}, keys={keys}, selected={selected})"
        )

    def select(self, choice: int | str) -> str:
        """Select an option by 1-based index or key string.

        Args:
            choice: Either a 1-based index (int) or the key string

        Returns:
            The key of the selected choice

        Raises:
            ValueError: If the index is out of range or key is invalid
        """
        if isinstance(choice, int):
            if not (1 <= choice <= len(self.choices)):
                raise ValueError(
                    f"Index {choice} out of range. Use 1-{len(self.choices)}"
                )
            selected = self.choices[choice - 1]
        else:
            selected = next((c for c in self.choices if c.key == choice), None)
            if selected is None:
                valid_keys = [c.key for c in self.choices]
                raise ValueError(f"Invalid key '{choice}'. Valid keys: {valid_keys}")

        self._selected = selected
        print(f"Selected: [{selected.key}] {selected.description}")
        return selected.key

    @property
    def selected(self) -> Choice | None:
        """The currently selected Choice object, or None if no selection."""
        return self._selected

    @property
    def selected_key(self) -> str | None:
        """The key of the currently selected choice, or None if no selection."""
        return self._selected.key if self._selected else None
