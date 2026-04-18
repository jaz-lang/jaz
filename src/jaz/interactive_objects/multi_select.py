"""Multi-select interactive object for agent interaction with combinable options."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Option:
    """A single option for multi-select.

    Attributes:
        key: Short identifier for selection (e.g., "search")
        description: Longer explanation shown to agent
    """

    key: str
    description: str


@dataclass
class MultiSelect:
    """A multi-select prompt supporting selection of multiple non-exclusive options.

    This object can be injected into a REPL via AddReplInput effect,
    allowing agents to make multiple selections during execution.

    Key behavior:
    - Selecting an exclusive option (e.g., "simple") clears all other selections
    - Selecting a non-exclusive option when an exclusive is selected clears the exclusive
    - Multiple non-exclusive options can be combined

    Attributes:
        prompt: The question or instruction shown to the agent
        options: List of Option objects representing available options
        exclusive_keys: Set of keys that are mutually exclusive with all others

    Example:
        >>> ms = MultiSelect(
        ...     prompt="Select strategies for this task:",
        ...     options=[
        ...         Option("simple", "No special handling"),
        ...         Option("search", "Try multiple approaches"),
        ...         Option("decompose", "Break into subtasks"),
        ...     ],
        ...     exclusive_keys=frozenset({"simple"}),
        ... )
        >>> ms.select("search", "decompose")  # Both selected
        ['search', 'decompose']
        >>> ms.select("simple")  # Clears others
        ['simple']
        >>> ms.select("search")  # Clears simple
        ['search']
    """

    prompt: str
    options: list[Option]
    exclusive_keys: frozenset[str] = frozenset({"simple"})
    _selected: set[str] = field(default_factory=set)
    _confirmed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate that exclusive_keys exist in options."""
        valid_keys = {o.key for o in self.options}
        invalid = self.exclusive_keys - valid_keys
        if invalid:
            raise ValueError(f"Exclusive keys not in options: {invalid}")

    def _resolve_choice(self, choice: int | str) -> str:
        """Resolve a choice (index or key) to a key string."""
        if isinstance(choice, int):
            if not (1 <= choice <= len(self.options)):
                raise ValueError(
                    f"Index {choice} out of range. Use 1-{len(self.options)}"
                )
            return self.options[choice - 1].key
        else:
            valid_keys = [o.key for o in self.options]
            if choice not in valid_keys:
                raise ValueError(f"Invalid key '{choice}'. Valid keys: {valid_keys}")
            return choice

    def select(self, *choices: int | str) -> list[str]:
        """Select one or more options by 1-based index or key string.

        If an exclusive option is selected, all other selections are cleared.
        If a non-exclusive option is selected when an exclusive is active, the
        exclusive option is cleared.

        Args:
            *choices: One or more 1-based indices (int) or key strings

        Returns:
            List of currently selected keys after the operation

        Raises:
            ValueError: If any index is out of range or key is invalid
        """
        keys_to_add = [self._resolve_choice(c) for c in choices]

        for key in keys_to_add:
            if key in self.exclusive_keys:
                # Exclusive option clears all others
                self._selected.clear()
                self._selected.add(key)
            else:
                # Non-exclusive option clears any exclusive options
                self._selected -= self.exclusive_keys
                self._selected.add(key)

        return self.selected_keys

    def deselect(self, *choices: int | str) -> list[str]:
        """Deselect one or more options by 1-based index or key string.

        Args:
            *choices: One or more 1-based indices (int) or key strings

        Returns:
            List of currently selected keys after the operation

        Raises:
            ValueError: If any index is out of range or key is invalid
        """
        keys_to_remove = [self._resolve_choice(c) for c in choices]
        for key in keys_to_remove:
            self._selected.discard(key)

        return self.selected_keys

    def toggle(self, choice: int | str) -> list[str]:
        """Toggle selection state of an option.

        Args:
            choice: A 1-based index (int) or key string

        Returns:
            List of currently selected keys after the operation

        Raises:
            ValueError: If index is out of range or key is invalid
        """
        key = self._resolve_choice(choice)
        if key in self._selected:
            return self.deselect(key)
        else:
            return self.select(key)

    def confirm(self) -> list[str]:
        """Confirm the current selection.

        Returns:
            List of confirmed selected keys

        Raises:
            ValueError: If no options are selected
        """
        if not self._selected:
            raise ValueError(
                "Cannot confirm with no selections. Select at least one option."
            )
        self._confirmed = True
        result = self.selected_keys
        print(f"Confirmed selection: {result}")
        return result

    @property
    def selected_keys(self) -> list[str]:
        """List of currently selected keys in option order."""
        # Return in the order options were defined
        return [o.key for o in self.options if o.key in self._selected]

    @property
    def is_confirmed(self) -> bool:
        """Whether the selection has been confirmed."""
        return self._confirmed

    def __str__(self) -> str:
        """Human-readable representation with descriptions."""
        lines = [f"MultiSelect: {self.prompt}"]
        for i, option in enumerate(self.options, 1):
            marker = "[x]" if option.key in self._selected else "[ ]"
            exclusive = " (exclusive)" if option.key in self.exclusive_keys else ""
            lines.append(
                f"  {marker} {i}. [{option.key}] {option.description}{exclusive}"
            )
        if self._selected:
            lines.append(f"\nCurrently selected: {self.selected_keys}")
        if self._confirmed:
            lines.append("(Selection confirmed)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        keys = [o.key for o in self.options]
        return (
            f"MultiSelect(prompt={self.prompt!r}, keys={keys}, "
            f"selected={self.selected_keys}, confirmed={self._confirmed})"
        )
