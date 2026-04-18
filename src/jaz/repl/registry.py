"""REPL registration system.

This module provides a decorator-based system for registering REPL implementations.
Users can register custom REPLs that will be available to the agent system.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import REPL


# Global mapping from language names to REPL classes
REPL_LANGUAGE_MAP: dict[str, type["REPL"]] = {}

# Character class for valid REPL language names, shared with REPL_INPUT_REGEX in agent.py
REPL_LANG_NAME_PAT = r"[a-zA-Z0-9_]+"
REPL_LANGUAGE_NAME_RE = re.compile(rf"^{REPL_LANG_NAME_PAT}$")


def register_repl(language: str):
    """Decorator to register a REPL implementation for a specific language.

    This decorator allows users to register custom REPL implementations that can
    be used by the agent. The registered REPL will be available via the language
    name in REPL_LANGUAGE_MAP.

    Args:
        language: The language name for this REPL (e.g., "python", "javascript")

    Returns:
        Decorator function that registers the REPL class

    Example:
        from jaz.repl import REPL, register_repl

        @register_repl("javascript")
        class JavaScriptREPL(REPL):
            @classmethod
            def initialize(cls, inputs, libraries, allowed_imports, ...):
                # Implementation
                ...

            def exec(self, src, input_id, ...):
                # Implementation
                ...

        # Now "javascript" is available in REPL_LANGUAGE_MAP
        # and can be used with Agent(repls=["python", "javascript"])
    """

    if not REPL_LANGUAGE_NAME_RE.match(language):
        raise ValueError(
            f"Invalid REPL language name: '{language}'. "
            "REPL language names must match the regex [a-zA-Z0-9_]+"
        )

    def decorator[C: type["REPL"]](repl_class: C) -> C:
        """Register the REPL class in the global map."""
        # Import here to avoid circular dependency
        from .base import REPL

        if not issubclass(repl_class, REPL):
            raise TypeError(
                f"REPL class {repl_class.__name__} must inherit from REPL base class"
            )
        if language in REPL_LANGUAGE_MAP:
            raise ValueError(
                f"REPL for language '{language}' is already registered "
                f"({REPL_LANGUAGE_MAP[language].__name__})"
            )
        REPL_LANGUAGE_MAP[language] = repl_class
        return repl_class

    return decorator
