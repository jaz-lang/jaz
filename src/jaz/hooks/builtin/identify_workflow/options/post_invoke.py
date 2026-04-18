"""Decision Point 4: Options after jaz.invoke completes."""

from jaz.interactive_objects import Option

# Options for decision point 4 (post invoke)
# This is effectively a yes/no decision
POST_INVOKE_OPTIONS = [
    Option(
        key="refinement",
        description="Inspect results/trajectory, improve prompt/tools, and redo if needed.",
    ),
    Option(
        key="accept",
        description="Accept the result as-is without refinement.",
    ),
]
