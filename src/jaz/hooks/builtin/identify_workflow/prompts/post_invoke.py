"""Prompts for Decision Point 4: Post-invoke refinement."""


# TODO: Implement this and improve prompt

# Map of option keys to their corresponding prompts
POST_INVOKE_PROMPTS = {
    "refinement": """
## Post-Invoke Refinement

After the jaz.invoke completes, analyze the result:
1. Inspect the trajectory/output
2. Identify any issues or areas for improvement
3. If needed, create an improved prompt and redo the invocation

This is useful when the initial result is close but not quite right.
""",
    "accept": """
Accept the result from jaz.invoke as-is.
""",
}
