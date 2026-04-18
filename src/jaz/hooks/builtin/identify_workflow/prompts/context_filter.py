"""Prompts for Decision Point 3: Context filtering."""

# Map of option keys to their corresponding prompts
CONTEXT_FILTER_PROMPTS = {
    "context_filter": """
## Context Filtering

Summarize the context and filter the REPL history:
1. Identify which parts of the current context are relevant to the remainder of the workflow
2. Create a concise summary of necessary information
3. Pass only the filtered context to jaz.invoke, including the relevant parts of your
   REPL history and relevant variables you've defined (or given to you as inputs)

Example:

<repl_input lang="python">
# Summarize relevant context
context_summary = \"\"\"\\
...
\"\"\"

# Filter history to only include relevant steps
filtered_history = __repl_history__[...]

# Pass the summarized context and filtered history to jaz.invoke to complete the rest of the task
jaz.invoke(
    __task__,
    return_type=...,
    context_summary=context_summary,
    previous_history=filtered_history,
    # Also pass in relevant variables you've defined
    var1=var1,
    var2=var2,
    ...,
    # Also pass in the inputs you've been given if relevant, or a subset thereof
    **__inputs__,
)
</repl_input>
""",
    "no_filter": """
Continue with the full context available.
""",
}
