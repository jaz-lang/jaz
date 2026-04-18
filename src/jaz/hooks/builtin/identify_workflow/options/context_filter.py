"""Decision Point 3: Options for context filtering (each step excluding 0th)."""

from jaz.interactive_objects import Option

# Options for decision point 3 (context filter)
# This is effectively a yes/no decision
CONTEXT_FILTER_OPTIONS = [
    Option(
        key="context_filter",
        description="Hand off to jaz.invoke with summarized context and filtered REPL history.",
    ),
    Option(
        key="no_filter",
        description="Continue with full context (no filtering).",
    ),
]
