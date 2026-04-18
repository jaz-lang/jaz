import copy
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

from typing_extensions import TypeForm

from jaz.budget import CostTracker

from .agent import Agent
from .config import Config, get_config
from .library import Library, get_jaz_library

# Common misspellings or misuses of invoke() kwargs.
# Keys are invalid kwarg names that users may pass; values are suggestions.
_INVOKE_KWARG_TYPOS: dict[str, str] = {
    "max_repl_iterations": (
        "max_iterations (or use jaz.config_override(max_repl_iterations=...))"
    ),
    "model": "jaz.configure(model_config={'model': ...})",
    "temperature": "jaz.config_override(model_config={'temperature': ...})",
    "max_tokens": "jaz.config_override(model_config={'max_tokens': ...})",
}


@dataclass(kw_only=True, frozen=True)
class PrehookOutput:
    repl_recursion_depth: int
    parent_cost_tracker: CostTracker | None = None
    parent_invoke_id: str | None = None
    parent_repl_iteration: int | None = None
    parent_max_recursion_depth: int | None = None


class Prehook:
    """Callable prehook that tracks parent invoke context.

    The parent_repl_iteration field is mutable and updated by the Agent
    at each REPL iteration, so nested invokes know which iteration spawned them.
    """

    def __init__(
        self,
        repl_recursion_depth: int,
        parent_cost_tracker: CostTracker | None = None,
        parent_invoke_id: str | None = None,
        parent_max_recursion_depth: int | None = None,
    ) -> None:
        self.repl_recursion_depth = repl_recursion_depth
        self.parent_cost_tracker = parent_cost_tracker
        self.parent_invoke_id = parent_invoke_id
        self.parent_repl_iteration: int | None = None
        self.parent_max_recursion_depth: int | None = parent_max_recursion_depth

    def __call__(self) -> PrehookOutput:
        return PrehookOutput(
            repl_recursion_depth=self.repl_recursion_depth,
            parent_cost_tracker=self.parent_cost_tracker,
            parent_invoke_id=self.parent_invoke_id,
            parent_repl_iteration=self.parent_repl_iteration,
            parent_max_recursion_depth=self.parent_max_recursion_depth,
        )


def _invoke[ReturnT](
    task: str,
    *,
    return_type: TypeForm[ReturnT] | None,
    return_validator: Callable[[ReturnT], None] | None = None,
    repl_input_validator: Callable[[str], None] | None = None,
    repls: list[str] | None = None,
    max_iterations: int | None = None,
    max_invoke_calls: int | None = None,
    libraries: list[Library] | None = None,
    allowed_imports: list[str] | None = None,
    max_recursion_depth: int | None = None,
    max_cost_budget: float | None = None,
    max_llm_calls_budget: int | None = None,
    task_name: str = "main",
    input_descriptions: dict[str, str | None] | None = None,
    prehook: Callable[[], PrehookOutput],
    config: Config,
    **inputs: object,
) -> ReturnT | None:
    """
    Same as public `invoke` but with the prehook specified explicitly
    to track recursion depth and invoke calls.
    """
    # Validate task length
    if len(task) > config.max_task_length:
        raise ValueError(
            f"Task string length ({len(task)}) exceeds max_task_length ({config.max_task_length})"
        )

    # Run prehook
    prehook_output = prehook()

    # Resolve depth-specific LLM configuration
    effective_llm_client, effective_model_config = config.resolve_for_depth(
        prehook_output.repl_recursion_depth
    )
    if (
        effective_llm_client != config.llm_client
        or effective_model_config != config.model_config
    ):
        config = copy.copy(config)
        config.llm_client = effective_llm_client
        # Copy the dict so downstream mutations don't leak into the source
        # (either the original Config.model_config or the override's dict).
        config.model_config = dict(effective_model_config)

    # Generate invoke_id for this invocation
    invoke_id = str(uuid.uuid4())
    parent_invoke_id = prehook_output.parent_invoke_id
    parent_repl_iteration = prehook_output.parent_repl_iteration

    # Get config values
    if max_iterations is None:
        max_iterations = config.max_repl_iterations
    max_iterations_buffer = config.max_repl_iterations_buffer

    # Get max invoke calls if not specified
    if max_invoke_calls is None:
        max_invoke_calls = config.max_repl_invoke_calls

    # Create cost tracker
    parent_cost_tracker = prehook_output.parent_cost_tracker
    cost_tracker = None
    if parent_cost_tracker is None:
        if prehook_output.repl_recursion_depth == 1:
            # Create top-level cost tracker if any budget tracking is enabled
            # Use provided budget or fall back to config
            # TODO how to explicitly specify no budget, then?
            cost_budget = (
                max_cost_budget
                if max_cost_budget is not None
                else config.max_cost_budget
            )
            llm_calls_budget = (
                max_llm_calls_budget
                if max_llm_calls_budget is not None
                else config.max_llm_calls_budget
            )
            # Create tracker if any budget is set (cost, LLM calls, invoke calls, or JSON tracking)
            if (
                cost_budget is not None
                or llm_calls_budget is not None
                or max_invoke_calls is not None
                or config.cost_tracking_json_path is not None
            ):
                cost_tracker = CostTracker(
                    llm_cost_budget=cost_budget,
                    llm_cost_buffer=config.max_cost_buffer,
                    llm_calls_budget=llm_calls_budget,
                    llm_calls_buffer=config.max_llm_calls_buffer,
                    invoke_calls_budget=max_invoke_calls,
                    recursion_depth=1,
                )

    else:
        # Create child tracker with the budget specified for this invocation
        # If no budget specified, add_invoke_call will use parent's remaining budget
        cost_tracker = parent_cost_tracker.add_invoke_call(
            invoke_calls_budget=max_invoke_calls,
            llm_cost_budget=max_cost_budget,
            llm_calls_budget=max_llm_calls_budget,
        )

    # Get max recursion depth if not specified
    if max_recursion_depth is None:
        max_recursion_depth = prehook_output.parent_max_recursion_depth
        if max_recursion_depth is None:  # No parent specified, use config
            max_recursion_depth = config.max_repl_recursion

    # Get default allowed imports if not specified
    if allowed_imports is None:
        allowed_imports = config.allowed_imports

    # Compute REPLs for this invocation (default to python if not specified)
    agent_repls = repls if repls is not None else ["python"]
    if not agent_repls:
        raise ValueError("At least one REPL language must be specified")

    # Add JAZ library to `libraries`
    libraries = libraries or []
    new_prehook: Prehook | None = None
    if prehook_output.repl_recursion_depth < max_recursion_depth:
        # Allowed to make recursive jaz.invoke calls.
        # Set up prehook to track these calls and recursion depth.
        # The Prehook's parent_repl_iteration will be updated by Agent at each iteration.
        new_prehook = Prehook(
            repl_recursion_depth=prehook_output.repl_recursion_depth + 1,
            parent_cost_tracker=cost_tracker,
            parent_invoke_id=invoke_id,
            parent_max_recursion_depth=max_recursion_depth,
        )

        jaz_library = get_jaz_library(
            _invoke,
            new_prehook,
            include_library=config.include_library,
            invoke_usage=config.invoke_usage,
            jaz_library_description_template=config.jaz_library_description_template,
            simplify_recursive_invoke=config.simplify_recursive_invoke,
            parent_libraries=libraries,
            parent_allowed_imports=allowed_imports,
            parent_repls=agent_repls,
            config=config,
        )
        libraries = libraries + [jaz_library]
    # TODO: Right now JAZ library is omitted if recursion limit is reached.
    # If there's more stuff in JAZ library then JAZ library should still be added.
    # else:
    # Recursion limit reached -- no more recursive jaz.invoke calls allowed.
    # jaz_library = get_jaz_library()

    # Start timing if cost tracker is available
    if cost_tracker is not None:
        cost_tracker.start()

    try:
        # Invoke agent in REPL
        agent = Agent(
            repls=agent_repls,
            cost_tracker=cost_tracker,
            prehook=new_prehook,
            config=config,
        )
        # NOTE: For static type checking, we need to call the overloads explicitly
        # TODO: Once pyright gets better, we no longer need separate calls because overloads will no longer be needed
        if return_type is None:
            if return_validator is not None:
                raise ValueError(
                    "return_validator must be None when return_type is None"
                )
            return agent.invoke(
                task,
                return_type=return_type,
                return_validator=return_validator,
                repl_input_validator=repl_input_validator,
                libraries=libraries,
                max_iterations=max_iterations,
                max_iterations_buffer=max_iterations_buffer,
                cur_recursion_depth=prehook_output.repl_recursion_depth,
                max_recursion_depth=max_recursion_depth,
                allowed_imports=allowed_imports,
                task_name=task_name,
                invoke_id=invoke_id,
                parent_invoke_id=parent_invoke_id,
                parent_repl_iteration=parent_repl_iteration,
                input_descriptions=input_descriptions,
                **inputs,
            )

        return agent.invoke(
            task,
            return_type=return_type,
            return_validator=return_validator,
            repl_input_validator=repl_input_validator,
            libraries=libraries,
            max_iterations=max_iterations,
            max_iterations_buffer=max_iterations_buffer,
            cur_recursion_depth=prehook_output.repl_recursion_depth,
            max_recursion_depth=max_recursion_depth,
            allowed_imports=allowed_imports,
            task_name=task_name,
            invoke_id=invoke_id,
            parent_invoke_id=parent_invoke_id,
            parent_repl_iteration=parent_repl_iteration,
            input_descriptions=input_descriptions,
            **inputs,
        )

    finally:
        # End timing if cost tracker is available
        if cost_tracker is not None:
            cost_tracker.end()

        # Save cost tracking data if this is the top-level invocation
        # TODO: What if user makes multiple top-level `invoke` calls?
        # don't want to ovverride the file each time
        if (
            cost_tracker is not None
            and prehook_output.repl_recursion_depth == 1
            and config.cost_tracking_json_path is not None
        ):
            cost_tracker.save_to_json(config.cost_tracking_json_path)


@overload
def invoke[ReturnT](
    task: str,
    *,
    return_type: TypeForm[ReturnT],
    return_validator: Callable[[ReturnT], None] | None = None,
    repl_input_validator: Callable[[str], None] | None = None,
    repls: list[str] | None = None,
    max_iterations: int | None = None,
    max_invoke_calls: int | None = None,
    libraries: list[Library] | None = None,
    allowed_imports: list[str] | None = None,
    max_recursion_depth: int | None = None,
    max_cost_budget: float | None = None,
    max_llm_calls_budget: int | None = None,
    task_name: str = "main",
    input_descriptions: dict[str, str | None] | None = None,
    **inputs: object,
) -> ReturnT: ...


@overload
def invoke(
    task: str,
    *,
    return_type: None,
    return_validator: None = None,
    repl_input_validator: Callable[[str], None] | None = None,
    repls: list[str] | None = None,
    max_iterations: int | None = None,
    max_invoke_calls: int | None = None,
    libraries: list[Library] | None = None,
    allowed_imports: list[str] | None = None,
    max_recursion_depth: int | None = None,
    max_cost_budget: float | None = None,
    max_llm_calls_budget: int | None = None,
    task_name: str = "main",
    input_descriptions: dict[str, str | None] | None = None,
    **inputs: object,
) -> None: ...


def invoke[ReturnT](
    task: str,
    *,
    return_type: TypeForm[ReturnT] | None,
    return_validator: Callable[[ReturnT], None] | None = None,
    repl_input_validator: Callable[[str], None] | None = None,
    repls: list[str] | None = None,
    max_iterations: int | None = None,
    max_invoke_calls: int | None = None,
    libraries: list[Library] | None = None,
    allowed_imports: list[str] | None = None,
    max_recursion_depth: int | None = None,
    max_cost_budget: float | None = None,
    max_llm_calls_budget: int | None = None,
    task_name: str = "main",
    input_descriptions: dict[str, str | None] | None = None,
    **inputs: object,
) -> ReturnT | None:
    """
    Invoke the default agent with the task `task`
    to enter a REPL session and return the response

    Arguments:
        task: The task to invoke the language model with
        return_type: The type of the return value. Use `None` to specify no return
            value.
        return_validator: Validates the return value. Should raise an exception iff the
            validation fails. Validators may optionally accept
            ``repl_history=...`` to inspect prior REPL iterations.
        repl_input_validator: Validates REPL input code before execution. Should raise
            an exception iff the validation fails. Only applies to PythonREPL.
            Note: passing None here does NOT disable a validator set via
            config (``jaz.configure(repl_input_validator=fn)``); the config-level
            validator is used as a fallback when this argument is None.
        repls: The REPLs to make available to the agent. Defaults to ["python"].
        max_iterations: The maximum number of iterations to invoke the language model.
            Uses the config value if not specified.
        max_invoke_calls: The maximum number of `jaz.invoke` calls allowed in this REPL
            session. (This has no effect if `jaz.invoke` calls are not allowed because
            the recursion limit has been reached.)
        libraries: Tool libraries available to the agent. Default is the JAZ core
            library.
        allowed_imports: The set of allowed imports in the REPL. Defaults to
            {"math", "random", "datetime", "re", "string", "itertools", "functools",
            "collections"}
        max_recursion_depth: The maximum recursion depth for nested `jaz.invoke`
        max_cost_budget: Maximum cost budget in USD. Overrides config setting if
            provided.
        max_llm_calls_budget: Maximum number of LLM API calls. Overrides config
            setting if provided.
        input_descriptions: Optional mapping from input key to a custom description
            string. When provided, the description replaces the default stringified
            value shown in the user prompt for that input. A value of `None` means
            "skip rendering this input in the prompt entirely" — the input is still
            passed to the REPL session as a Python variable, but the agent doesn't
            see it in its prompt header.
        inputs: The inputs to pass to the agent

    Returns:
        The return value from the REPL session

    Note:
        Hooks are managed via context managers, not parameters. Use the `with` statement
        to add hooks before calling invoke. See jaz.hooks for details.
    """

    # Catch common kwarg misspellings before they silently become input variables
    for key in inputs:
        if key in _INVOKE_KWARG_TYPOS:
            raise TypeError(
                f"invoke() got unexpected keyword argument '{key}'. "
                f"Did you mean: {_INVOKE_KWARG_TYPOS[key]}"
            )

    # Resolve config once at the public entry point
    config = get_config()

    # Define top-level prehook
    def top_level_prehook() -> PrehookOutput:
        return PrehookOutput(repl_recursion_depth=1)

    # Call _invoke with prehook
    return _invoke(
        task,
        return_type=return_type,
        return_validator=return_validator,
        repl_input_validator=repl_input_validator,
        repls=repls,
        max_iterations=max_iterations,
        max_invoke_calls=max_invoke_calls,
        libraries=libraries,
        allowed_imports=allowed_imports,
        max_recursion_depth=max_recursion_depth,
        max_cost_budget=max_cost_budget,
        max_llm_calls_budget=max_llm_calls_budget,
        task_name=task_name,
        input_descriptions=input_descriptions,
        prehook=top_level_prehook,
        config=config,
        **inputs,
    )
