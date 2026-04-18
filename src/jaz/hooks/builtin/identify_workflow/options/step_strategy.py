"""Decision Point 2: Options for each step (including the 0th step)."""

from jaz.interactive_objects import Option

# Options for decision point 2 (step strategy)
# simple is exclusive; others are combinable
STEP_STRATEGY_OPTIONS = [
    Option(
        key="simple",
        description="Execute this step directly without calling jaz.invoke.",
    ),
    Option(
        key="search",
        description="Try multiple approaches to executing this step with jaz.invoke, and select the best result.",
    ),
    Option(
        key="extra_guidance",
        description="Call jaz.invoke with specialized prompts and/or tools to guide the subagent.",
    ),
]
