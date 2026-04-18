from __future__ import annotations

from collections.abc import Callable
from enum import Flag, auto
from typing import TYPE_CHECKING, Any

from typing_extensions import TypeForm

from jaz.template_loader import _jinja_env

from .core import Library

if TYPE_CHECKING:
    from jaz.config import Config


class InvokeUsage(Flag):
    CONTEXT_FILTER = auto()
    SEARCH = auto()
    SELF_IMPROVEMENT = auto()
    SUBTASK = auto()

    ALL = CONTEXT_FILTER | SEARCH | SELF_IMPROVEMENT | SUBTASK


def _dedupe_libraries_by_root_module(libraries: list[Library]) -> list[Library]:
    """Deduplicate libraries by root module name, keeping the last one.

    Recursive `jaz.invoke(..., additional_libraries=[...])` calls should follow
    the same namespace-replacement semantics as the agent prompt/repl path: if
    a later library provides the same root module as an earlier one, the later
    library wins.
    """
    seen_root_modules: set[str] = set()
    deduped_reversed: list[Library] = []
    for library in reversed(libraries):
        root_modules = set(getattr(library, "_root_modules", {}).keys())
        if root_modules & seen_root_modules:
            continue
        deduped_reversed.append(library)
        seen_root_modules.update(root_modules)
    deduped_reversed.reverse()
    return deduped_reversed


def parse_invoke_usage(value: str | list[str]) -> InvokeUsage:
    """Parse an InvokeUsage flag from a string or list of strings.

    This is useful for parsing command-line arguments or deserializing from JSON.

    Args:
        value: One of:
            - A string: "all", "none", or comma-separated flags like "SUBTASK,SEARCH"
            - A list of strings: ["SUBTASK", "SEARCH"]

    Returns:
        Combined InvokeUsage flag.

    Raises:
        ValueError: If any flag name is invalid.

    Examples:
        >>> parse_invoke_usage("SUBTASK,SEARCH")
        <InvokeUsage.SEARCH|SUBTASK: 10>
        >>> parse_invoke_usage(["SUBTASK", "SEARCH"])
        <InvokeUsage.SEARCH|SUBTASK: 10>
        >>> parse_invoke_usage("all")
        <InvokeUsage.SUBTASK|SELF_IMPROVEMENT|SEARCH|CONTEXT_FILTER: 15>
        >>> parse_invoke_usage("none")
        <InvokeUsage: 0>
    """
    # Convert string to list
    if isinstance(value, str):
        value_upper = value.strip().upper()
        if value_upper == "ALL":
            return InvokeUsage.ALL
        if value_upper == "NONE":
            return InvokeUsage(0)
        # Split comma-separated string into list
        names = [s.strip() for s in value.split(",") if s.strip()]
    else:
        names = value

    if not names:
        return InvokeUsage(0)

    result = InvokeUsage(0)
    for name in names:
        name_upper = name.strip().upper()
        try:
            result |= InvokeUsage[name_upper]
        except KeyError:
            valid_flags = ", ".join(f.name for f in InvokeUsage if f.name)
            raise ValueError(
                f"Invalid invoke_usage flag: '{name}'. "
                f"Valid flags are: {valid_flags}, 'all', or 'none'"
            ) from None
    return result


def _build_jaz_library_desc(
    invoke_usage: InvokeUsage,
    include_library: bool = False,
    template_name: str = "jaz_library_description.jinja2",
) -> str:
    """Build JAZ library description dynamically based on enabled usage patterns."""
    template = _jinja_env.get_template(template_name)
    return template.render(
        include_library=include_library,
        subtask=InvokeUsage.SUBTASK in invoke_usage,
        context_filter=InvokeUsage.CONTEXT_FILTER in invoke_usage,
        search=InvokeUsage.SEARCH in invoke_usage,
        self_improvement=InvokeUsage.SELF_IMPROVEMENT in invoke_usage,
    )


def get_jaz_library(
    invoke_function: Callable | None = None,
    prehook: Callable | None = None,
    config: Config | None = None,
    include_library: bool = False,
    invoke_usage: InvokeUsage = InvokeUsage.ALL,
    jaz_library_description_template: str = "jaz_library_description.jinja2",
    simplify_recursive_invoke: bool = False,
    parent_libraries: list[Library] | None = None,
    parent_allowed_imports: list[str] | None = None,
    parent_repls: list[str] | None = None,
) -> Library:  # TODO: proper type hinting
    if invoke_function is not None:
        assert prehook is not None
        assert config is not None

        def invoke[ReturnT](
            prompt: str,
            *,
            return_type: TypeForm[ReturnT] | None = None,
            return_validator: Callable[[ReturnT], None] | None = None,
            max_iterations: int | None = None,
            max_invoke_calls: int | None = None,
            additional_libraries: list[Library] | None = None,
            additional_repls: list[str] | None = None,
            max_recursion_depth: int | None = None,
            max_cost_budget: float | None = None,
            max_llm_calls_budget: int | None = None,
            input_descriptions: dict[str, str | None] | None = None,
            **inputs: object,
        ) -> ReturnT | None:
            # TODO: Remove `| None` once pyright gets better
            """Invoke the default agent with the prompt `prompt`
            to enter a REPL session and return the response

            Arguments:
                prompt: The prompt to invoke the language model with
                return_type: The type of the return value. Use `None` to specify no
                    return value.
                return_validator: Validates the return value. Should raise an exception
                    iff the validation fails. Validators may optionally accept
                    ``repl_history=...`` to inspect prior REPL iterations. Defaults
                    to no validation (other than type checking) if not specified.
                max_iterations: The maximum number of iterations to invoke the language
                    model. Defaults to the config value if not specified.
                max_invoke_calls: The maximum number of `jaz.invoke` calls allowed in
                    this REPL session. (This has no effect if `jaz.invoke` calls are not
                    allowed because the recursion limit has been reached.)
                    Defaults to the config value if not specified.
                additional_libraries: Additional tool libraries to make available to the
                    agent, in addition to libraries currently available in this REPL session.
                additional_repls: Additional REPLs to make available to the agent,
                    in addition to REPLs currently available in this REPL session.
                max_recursion_depth: The maximum recursion depth for nested `jaz.invoke`.
                    Must be greater than the current recursion depth of this REPL session.
                    Defaults to the maximum recursion depth specified in this REPL session
                    if not specified.
                max_cost_budget: Maximum cost budget in USD. Defaults to remaining
                    budget for this REPL session if not specified.
                max_llm_calls_budget: Maximum number of LLM API calls. Defaults to remaining
                    budget for this REPL session if not specified.
                input_descriptions: Optional mapping from input key to a custom description
                    string. When provided, the description replaces the default stringified
                    value shown in the user prompt for that input. A value of `None` means
                    "skip rendering this input in the prompt entirely" — the input is still
                    passed to the REPL session as a Python variable, but the agent doesn't
                    see it in its prompt header.
                **inputs: The inputs to pass to the agent

            Returns:
                The return value from the REPL session
            """
            # Merge additional libraries/imports/repls with parent's
            merged_libraries: list[Library] | None = None
            if parent_libraries or additional_libraries:
                merged_libraries = _dedupe_libraries_by_root_module(
                    (parent_libraries or []) + (additional_libraries or [])
                )

            # Merge REPLs: use set to deduplicate, but preserve order from parent first
            merged_repls: list[str] | None = None
            if parent_repls or additional_repls:
                seen: set[str] = set()
                merged_repls = []
                for repl in (parent_repls or []) + (additional_repls or []):
                    if repl not in seen:
                        seen.add(repl)
                        merged_repls.append(repl)

            return invoke_function(
                prompt,
                return_type=return_type,
                return_validator=return_validator,
                repls=merged_repls,
                max_iterations=max_iterations,
                max_invoke_calls=max_invoke_calls,
                libraries=merged_libraries,
                allowed_imports=parent_allowed_imports,
                max_recursion_depth=max_recursion_depth,
                max_cost_budget=max_cost_budget,
                max_llm_calls_budget=max_llm_calls_budget,
                input_descriptions=input_descriptions,
                prehook=prehook,
                config=config,
                **inputs,
            )

        if simplify_recursive_invoke:
            # Wrap the full invoke in a simplified signature for sub-agents
            _full_invoke = invoke

            def invoke_simple[ReturnT](
                prompt: str,
                *,
                return_type: TypeForm[ReturnT] | None = None,
                return_validator: Callable[[ReturnT], None] | None = None,
                **inputs: Any,
            ) -> ReturnT | None:
                """Invoke a sub-agent to solve a task in its own REPL session.

                Arguments:
                    prompt: The task description for the sub-agent
                    return_type: The expected return type. Use `None` for no return value.
                    return_validator: Optional callable to validate the return value.
                    **inputs: Named inputs to pass to the sub-agent

                Returns:
                    The return value from the sub-agent's REPL session
                """
                return _full_invoke(
                    prompt,
                    return_type=return_type,
                    return_validator=return_validator,
                    **inputs,
                )

            invoke_name_and_func = [("jaz.invoke", invoke_simple)]
        else:
            invoke_name_and_func = [("jaz.invoke", invoke)]
    else:
        # assert prehook is None
        # invoke_name_and_func = []
        raise ValueError(
            "Currently does not support getting jaz.invoke-less JAZ library."
        )

    # Build list of tools to include
    library_name_and_class = [("jaz.Library", Library)] if include_library else []
    all_tools = library_name_and_class + invoke_name_and_func

    # Build module description
    module_contents = []
    if include_library:
        module_contents.append("`jaz.Library`")
    if invoke_name_and_func:
        module_contents.append("`jaz.invoke`")
    module_desc = "Contains " + " and ".join(module_contents) + "."

    jaz_library = Library(
        "JAZ tool library",
        _build_jaz_library_desc(
            invoke_usage, include_library, jaz_library_description_template
        ),
        [("jaz", module_desc)],
        all_tools,
    )
    return jaz_library
