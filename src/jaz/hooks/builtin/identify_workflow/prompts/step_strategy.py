"""Prompts for Decision Point 2: Strategy for each step."""

# TODO: Improve these prompts

# Map of option keys to their corresponding strategy prompts
STEP_STRATEGY_PROMPTS = {
    "simple": """
Execute this step directly without any recursive jaz.invoke calls.
""",
    "search": """
## Step Search Strategy

For this step, try multiple approaches before committing:
1. Identify 2-3 different ways to accomplish this step
2. Execute each approach
3. Compare results and select the best one
""",
    "extra_guidance": """
## Step Guidance Strategy

For this step, provide extra guidance to any subagent calls:
1. Prepare detailed context specific to this step
2. Create any helper functions that would be useful
3. Include constraints and expected output format
""",
}
