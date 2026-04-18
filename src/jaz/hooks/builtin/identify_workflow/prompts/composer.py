"""Compose multiple prompts into a single coherent prompt."""

from __future__ import annotations


def compose_prompts(prompts_dict: dict[str, str], selected_keys: list[str]) -> str:
    """Compose prompts for selected options into a single prompt.

    Prompts are composed in a specific order for consistency:
    decompose → search → extra_guidance → refinement → context_filter

    Args:
        prompts_dict: Mapping of option keys to their prompt strings
        selected_keys: List of selected option keys

    Returns:
        Combined prompt string, or empty string if no valid selections
    """
    # Define the composition order (most structural strategies first)
    COMPOSITION_ORDER = [
        "decompose",
        "search",
        "extra_guidance",
        "refinement",
        "context_filter",
        # simple/accept/no_filter are typically standalone
        "simple",
        "accept",
        "no_filter",
    ]

    for k in prompts_dict:
        if k not in COMPOSITION_ORDER:
            raise ValueError(
                f"prompts_dict key {k} cannot be composed. Must be in {COMPOSITION_ORDER}"
            )

    for k in selected_keys:
        if k not in prompts_dict:
            raise ValueError(
                f"Selected key {k} not found in prompts_dict. Must be in {list(prompts_dict.keys())}"
            )

    # Reorder selected_keys based on COMPOSITION_ORDER
    selected_keys = [k for k in COMPOSITION_ORDER if k in selected_keys]

    # Combine selected prompts
    sections = []
    for key in selected_keys:
        prompt = prompts_dict[key].strip()
        if prompt:
            sections.append(prompt)

    return "\n\n---\n\n".join(sections)
