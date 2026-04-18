from collections.abc import Iterable, Mapping, Sequence

from typing_extensions import TypeForm

from .library import Library
from .repl.registry import REPL_LANGUAGE_MAP
from .string_utils import abbreviate_string, backtickify
from .template_loader import _jinja_env

# TODO: move `prompt_additions_str` out of the task block?


def get_system_prompt(
    *,
    system_prompt_template: str = "system_prompt.jinja2",
    libraries: Sequence[Library],
    forbidden_names: Iterable[str],
    forbidden_attributes: Iterable[str] | None = None,
    allowed_imports: Iterable[str],
    cur_recursion_depth: int,
    max_recursion_depth: int,
    max_iterations: int,
    available_repls: list[str],
    add_external_thinking: bool = False,
    system_prompt_additions: list[str] | None = None,
    repl_configs: dict[str, Mapping[str, object]] | None = None,
    show_tools_in_prompt: bool = True,
    full_docstrings_in_prompt: bool = False,
) -> str:
    # Generate REPL description header
    repl_list = ", ".join(f"`{lang}`" for lang in available_repls)
    repl_summary = f"You have access to the following REPLs: {repl_list}. "
    repl_example_lang = available_repls[0]

    # Get REPL configs - use provided or empty dict
    if repl_configs is None:
        repl_configs = {}

    # Collect REPL-specific instructions from each available REPL
    repl_specific_instructions_parts = []
    for lang in available_repls:
        if lang in REPL_LANGUAGE_MAP:
            repl_class = REPL_LANGUAGE_MAP[lang]
            # Get config for this language from Config
            repl_config = repl_configs.get(lang, {})
            # Get description for this REPL with its configuration
            description = repl_class.get_description(repl_config)
            repl_specific_instructions_parts.append(
                f'<repl_description lang="{lang}">\n{description}\n</repl_description>'
            )
        else:
            raise ValueError(
                f"REPL language `{lang}` is not available. "
                "Did you misspell or forget to register it?"
            )

    repl_specific_instructions = "\n\n".join(repl_specific_instructions_parts)

    # Build safety constraints section conditionally
    safety_constraints_parts = []
    forbidden_names_list = list(forbidden_names)
    allowed_imports_list = list(allowed_imports)

    if forbidden_names_list:
        forbidden_names_str = ", ".join(map(backtickify, forbidden_names_list))
        safety_constraints_parts.append(
            f"- You are FORBIDDEN from referencing these names:\n"
            f"  {forbidden_names_str}\n"
            f"  Referencing them will raise a SyntaxError and fail the task."
        )

    forbidden_attributes_list = (
        list(forbidden_attributes) if forbidden_attributes else []
    )
    if forbidden_attributes_list:
        forbidden_attributes_str = ", ".join(
            map(backtickify, forbidden_attributes_list)
        )
        safety_constraints_parts.append(
            f"- You are FORBIDDEN from accessing these attributes of ANY object:\n"
            f"  {forbidden_attributes_str}\n"
            f"  Accessing them on any object will raise a SyntaxError and fail the task."
        )

    if allowed_imports_list:
        allowed_imports_str = ", ".join(map(backtickify, allowed_imports_list))
        safety_constraints_parts.append(
            f"- You may ONLY import the following Python modules:\n"
            f"  {allowed_imports_str}\n"
            f"  Importing anything else will raise an ImportError and fail the task."
        )
    else:
        safety_constraints_parts.append(
            "- You are NOT allowed to import any Python modules.\n"
            "  Importing anything will raise an ImportError and fail the task."
        )

    safety_constraints_str = "IMPORTANT SAFETY CONSTRAINTS:\n\n" + "\n\n".join(
        safety_constraints_parts
    )

    show_recursion_depth = cur_recursion_depth < max_recursion_depth

    # Choose template based on whether we need external thinking tags
    if add_external_thinking:
        additional_plan_guidelines = (
            "Reason carefully about what's the best next step to take."
        )
    else:
        additional_plan_guidelines = ""

    # Build system prompt additions string
    system_prompt_additions_str = ""
    if system_prompt_additions:
        system_prompt_additions_str = "\n".join(system_prompt_additions)

    rendered_libraries = [
        lib.render_prompt_description(full_docstrings=full_docstrings_in_prompt)
        for lib in libraries
    ]

    template = _jinja_env.get_template(system_prompt_template)
    prompt = template.render(
        repl_summary=repl_summary,
        repl_example_lang=repl_example_lang,
        additional_plan_guidelines=additional_plan_guidelines,
        repl_specific_instructions=repl_specific_instructions,
        libraries=rendered_libraries,
        safety_constraints_str=safety_constraints_str,
        show_recursion_depth=show_recursion_depth,
        cur_recursion_depth=cur_recursion_depth,
        max_recursion_depth=max_recursion_depth,
        max_iterations=max_iterations,
        system_prompt_additions_str=system_prompt_additions_str,
        show_tools_in_prompt=show_tools_in_prompt,
    )

    return prompt


def get_user_prompt[ReturnT](
    *,
    user_prompt_template: str = "user_prompt.jinja2",
    task: str,
    inputs: Mapping[str, object],
    return_type: TypeForm[ReturnT] | None,
    max_input_length: int,
    budget_status: str | None = None,
    instruction_prompt_additions: list[str] | None = None,
    input_truncation_advice_template: str | None = None,
    task_preamble: str = "",
    inputs_preamble: str = "",
    input_descriptions: Mapping[str, str | None] | None = None,
    prefix_ratio: float = 0.8,
) -> str:
    truncated_variable_names: list[str] = []
    inputs_str = ""
    for key, value in inputs.items():
        if input_descriptions is not None and key in input_descriptions:
            description = input_descriptions[key]
            if description is None:
                # `None` means: skip rendering this input in the prompt
                # entirely. The input is still passed to the REPL session
                # as a Python variable, but the agent doesn't see it in
                # its prompt header.
                continue
            # TODO: Still apply truncation to custom descriptions
            abbreviated_value = description
            was_truncated = False
        else:
            value_str = str(value)
            abbreviated_value, was_truncated = abbreviate_string(
                value_str, max_input_length, prefix_ratio=prefix_ratio
            )
        if was_truncated:
            truncated_variable_names.append(key)
        inputs_str += (
            f"- {key} ({type(value).__name__}):\n"
            f"<{key}>\n{abbreviated_value}\n</{key}>\n"
        )

    truncation_advice_str = ""
    if input_truncation_advice_template and truncated_variable_names:
        template = _jinja_env.get_template(input_truncation_advice_template)
        rendered_advice = template.render(
            truncated_variable_names=truncated_variable_names,
        )
        truncation_advice_str = (
            "\n<truncation_advice>\n" + rendered_advice + "\n</truncation_advice>\n"
        )

    return_type_str = (
        return_type.__name__ if isinstance(return_type, type) else str(return_type)
    )
    if instruction_prompt_additions:
        prompt_additions_str = (
            "\n<additional_instructions>\n\n"
            + "\n".join(instruction_prompt_additions)
            + "\n\n</additional_instructions>\n"
        )
    else:
        prompt_additions_str = ""
    budget_str = budget_status if budget_status else ""
    template = _jinja_env.get_template(user_prompt_template)
    return template.render(
        task=task,
        inputs_str=inputs_str,
        return_type_str=return_type_str,
        budget_str=budget_str,
        prompt_additions_str=prompt_additions_str,
        truncation_advice_str=truncation_advice_str,
        task_preamble=task_preamble,
        inputs_preamble=inputs_preamble,
    )
