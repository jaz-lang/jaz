"""Decision Point 1: Options for the beginning of jaz.invoke."""

from jaz.interactive_objects import Option

# Options for decision point 1 (invoke start)
# simple is exclusive; others are combinable
INVOKE_START_OPTIONS = [
    Option(
        key="simple",
        description="Solve the task directly without any jaz.invoke calls.",
    ),
    Option(
        key="decompose",
        description="Break the task into independent subtasks and solve each one with jaz.invoke.",
    ),
    Option(
        key="search",
        description="Try multiple approaches to solving the (sub)task with jaz.invoke, and select the best result.",
    ),
    Option(
        key="extra_guidance",
        description="Call jaz.invoke with specialized prompts and/or tools to guide the subagent.",
    ),
    Option(
        key="refinement",
        description="Make multiple attempts at solving the (sub)task with jaz.invoke, refining and improving the approach each time.",
    ),
]
