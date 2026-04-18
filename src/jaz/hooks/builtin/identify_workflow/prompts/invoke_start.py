"""Prompts for Decision Point 1: Beginning of jaz.invoke."""

# Map of option keys to their corresponding strategy prompts
INVOKE_START_PROMPTS = {
    "simple": """
Solve the task without calling jaz.invoke.
""",
    "search": """
## Search Strategy

There are a few different versions of search.
1. Try different approaches
2. Try different initial steps
3. Try different approaches as well as different initial steps

### Try different approaches

Try multiple different approaches to solving the (sub)task. For each approach:
1. Call `jaz.invoke` with a prompt describing that specific approach
2. Evaluate the quality of the result
3. Track which approach produces the best outcome

Notes:
1. Don't forget to undo any side effects from previous attempts if needed
2. Don't forget to pass to the `jaz.invoke` call in relevant context such as inputs you've been given,
   as well as any variables you've defined if relevant.

Example pattern:

<repl_input lang="python">
# Optional initial steps, possibly over multiple REPL iterations
...
</repl_input>

<repl_output>
...  # results of initial steps
</repl_output>

<repl_input lang="python">
# Save checkpoint of any current state if applicable (to revert to if needed)
...

# Instructions for approach 1
approach1 = \"\"\"\\
...
\"\"\"

# Try approach 1
jaz.invoke(
    __task__,  # OR subtask
    return_type=...,
    approach=approach1,
    **__inputs__,  # Better to explicitly pass in relevant inputs with their variable names
)
</repl_input>

<repl_output>
...  # result of approach 1
</repl_output>

<repl_input lang="python">
# Restore checkpoint if applicable
...

# Instructions for approach 2
approach2 = \"\"\"\\
...
\"\"\"

# Try approach 2
jaz.invoke(
    __task__,  # OR subtask
    return_type=...,
    approach=approach2,
    **__inputs__,  # Better to explicitly pass in relevant inputs with their variable names
)
</repl_input>

<repl_output>
...  # result of approach 2
</repl_output>

...


### Try different initial steps

Try multiple different ways to take the initial step(s). For each way:
1. Execute the initial step(s)
2. Call `jaz.invoke` with the result(s) of the initial step(s) to complete the next step(s)
3. Evaluate the quality of the result
4. Track which initial step(s) produce the best outcome
5. Don't forget to undo any side effects from previous attempts if needed

Example pattern:

<repl_input lang="python">
# Save checkpoint of any current state if applicable (to revert to if needed)
...

# Execute initial step(s), possibly over multiple REPL iterations
...
initial_steps_result = ...
print(initial_steps_result)
</repl_input>

<repl_output>
...  # result of initial steps
</repl_output>

<repl_input lang="python">
jaz.invoke(
    __task__,  # OR subtask
    return_type=...,
    initial_steps_result=initial_steps_result,
    **__inputs__,  # Better to explicitly pass in relevant inputs with their variable names
)
</repl_input>

<repl_output>
...  # result of first attempt
</repl_output>

<repl_input lang="python">
# Restore checkpoint if applicable
...

# Execute initial steps differently, possibly over multiple REPL iterations
...
different_initial_steps_result = ...
print(different_initial_steps_result)
</repl_input>

<repl_output>
...  # result of different initial steps
</repl_output>

<repl_input lang="python">
jaz.invoke(
    __task__,  # OR subtask
    return_type=...,
    initial_steps_result=different_initial_steps_result,
    **__inputs__,  # Better to explicitly pass in relevant inputs with their variable names
)
</repl_input>

<repl_output>
...  # result of second attempt
</repl_output>

...


### Try different approaches as well as different initial steps

Try multiple different approaches to solving the (sub)task, as well as multiple different ways to take the initial step(s).
This combines the two patterns above.
""",
    "extra_guidance": """
## Extra Guidance Strategy

Before calling `jaz.invoke`, create specialized guidance for the subagent:
1. Write a detailed prompt that provides specific context and instructions
2. Optionally create helper functions or tools the subagent can use
3. Pass these as part of the `jaz.invoke` call

Example pattern:

<repl_input lang="python">
# Optional initial steps, possibly over multiple REPL iterations
...
</repl_input>

<repl_output>
...  # result of initial steps
</repl_output>

<repl_input lang="python">
# Create specialized guidance
guidance_prompt = f\"\"\"\\
Context: {relevant_context}
Specific instructions: {detailed_instructions}
Constraints: {constraints}
\"\"\"

# Optionally create helper tools
def helper_tool(arg: InputType) -> ReturnType:
    \"\"\"Helper tool for ...\"\"\"
    ...
    return result

...  # more helper tools

library_desc = \"\"\"\\
my_lib is a tool library for ...
It has the following tools:

helper_tool(arg: InputType) -> ReturnType: ...
    Helper tool for ...
    Args:
    - arg (InputType): ...
    Returns:
    - result (ReturnType): ...

...
\"\"\"

my_library = jaz.Library(
    name="My Tool Library",
    desc=library_desc,
    modules=[("my_lib", "my_lib is a module containing my tools")],
    tools=[("my_lib.helper_tool", helper_tool), ...],
)

# Call jaz.invoke with specialized guidance and tools
jaz.invoke(
    __task__,  # OR subtask
    return_type=...,
    guidance_prompt=guidance_prompt,
    additional_libraries=[my_library],
    **__inputs__,  # Better to explicitly pass in relevant inputs with their variable names
)
</repl_input>

<repl_output>
...  # result of jaz.invoke called with specialized guidance and tools
</repl_output>
""",
    "decompose": """
## Decomposition Strategy

Break the task down into smaller, independent subtasks that can be solved separately:
1. Identify the logical independent subtasks
2. Call `jaz.invoke` for each subtask
3. Synthesize the results into a final solution

Example pattern:

<repl_input lang="python">
# Optional initial steps, possibly over multiple REPL iterations
...
</repl_input>

<repl_output>
...  # result of initial steps
</repl_output>

<repl_input lang="python">
# Subtask 1
subtask1 = f\"\"\"\\
## Parent task

{__task__}

## Your task

...  (describe the first subtask)
\"\"\"

jaz.invoke(
    subtask1,
    return_type=...,
    **__inputs__,  # Better to explicitly pass in relevant inputs with their variable names
)
</repl_input>

<repl_output>
...  # result of subtask 1
</repl_output>

<repl_input lang="python">
# Optional intermediate steps, possibly over multiple REPL iterations
...
</repl_input>

<repl_output>
...  # result of intermediate steps
</repl_output>

<repl_input lang="python">
# Subtask 2
subtask2 = f\"\"\"\\
## Parent task

{__task__}

## Your task

...  (describe the second subtask)
\"\"\"

jaz.invoke(
    subtask2,
    return_type=...,
    **__inputs__,  # Better to explicitly pass in relevant inputs with their variable names
)
</repl_input>

<repl_output>
...  # result of subtask 2
</repl_output>

...
""",
    "refinement": """
## Refinement Strategy

After attempting the task, inspect the trajectory and improve:
1. First attempt: solve the task, asking jaz.invoke to return its __repl_history__.
2. Analyze what worked and what didn't
3. Based on learnings, create an improved prompt/approach and possibly a tool library.
4. Re-attempt with improvements

Example pattern:

<repl_input lang="python">
# Optional initial steps, possibly over multiple REPL iterations
...
</repl_input>

<repl_output>
...  # result of initial steps
</repl_output>

<repl_input lang="python">
# First attempt - return history for inspection
prompt = f\"\"\"\\
{__task__}

Additional instructions: ...

You must also return the list `__repl_history__` when you're done, i.e.,
&lt;repl_input lang="python"&gt;
RETURN final_result, __repl_history__
&lt;/repl_input&gt;
\"\"\"

def return_validator(return_value):
    assert isinstance(return_value[1], type(__repl_history__)), f"You must also return the list `__repl_history__` when you're done"

first_attempt_result, first_attempt_history = jaz.invoke(
    prompt,
    return_type=...,
    return_validator=return_validator,
    **__inputs__,  # Better to explicitly pass in relevant inputs with their variable names
)
print(first_attempt_result)
print(first_attempt_history)
</repl_input>

<repl_output>
...  # result of first attempt
...  # history of first attempt
</repl_output>

<repl_input lang="python">
# Refined attempt with learnings
refined_prompt = f\"\"\"\\
{__task__}

Additional instructions: ...
Common mistakes and how to avoid them: ...

You must also return the list `__repl_history__` when you're done, i.e.,
&lt;repl_input lang="python"&gt;
RETURN final_result, __repl_history__
&lt;/repl_input&gt;
\"\"\"

# If common errors are due to lack of certain tools, create them
def helper_tool(arg: InputType) -> ReturnType:
    \"\"\"Helper tool for ...\"\"\"
    ...
    return result

library_desc = \"\"\"\\
my_lib is a tool library for ...
It has the following tools:

helper_tool(arg: InputType) -> ReturnType: ...
    Args:
    - arg (InputType): ...
    Returns:
    - result (ReturnType): ...
    Helper tool for ...

...
\"\"\"

my_library = jaz.Library(
    name="My Tool Library",
    desc=library_desc,
    modules=[("my_lib", "my_lib is a module containing my tools")],
    tools=[("my_lib.helper_tool", helper_tool)],
    ...
)

# Call jaz.invoke with refined prompt and tools
second_attempt_result, second_attempt_history = jaz.invoke(
    refined_prompt,
    return_type=...,
    return_validator=return_validator,
    additional_libraries=[my_library],
    **__inputs__,  # Better to explicitly pass in relevant inputs with their variable names
)
print(second_attempt_result)
print(second_attempt_history)
</repl_input>

<repl_output>
...  # result of second attempt
...  # history of second attempt
</repl_output>

...
""",
}
