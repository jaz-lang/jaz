"""Model pricing and metadata from bundled LiteLLM data.

This module loads pricing data from the bundled model_prices.json file
(sourced from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
and provides functions for cost calculation and model info retrieval.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

# Load bundled JSON at import time
_PRICING_DATA: dict[str, Any] = json.loads(
    files("jaz.providers").joinpath("model_prices.json").read_text()
)


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float | None:
    """Compute cost in USD from token counts.

    Cached tokens are billed at different rates than regular input tokens.
    The prompt_tokens count from the API includes cached tokens, so we
    subtract them before applying the regular input rate, then add the
    cached token costs at their respective rates.

    Args:
        model: The model identifier (e.g., "gpt-4o", "claude-3-opus-20240229").
        prompt_tokens: Number of input/prompt tokens (includes cached tokens).
        completion_tokens: Number of output/completion tokens.
        cache_creation_input_tokens: Tokens written to cache (Anthropic only).
        cache_read_input_tokens: Tokens read from cache (both providers).

    Returns:
        Cost in USD, or None if the model is not found in the pricing database.
    """
    info = _PRICING_DATA.get(model)
    if not info:
        return None

    input_cost_per_token = info.get("input_cost_per_token", 0) or 0
    output_cost_per_token = info.get("output_cost_per_token", 0) or 0
    cache_creation_cost = info.get("cache_creation_input_token_cost", 0) or 0
    cache_read_cost = info.get("cache_read_input_token_cost", 0) or 0

    # Regular input tokens = total prompt tokens minus cached tokens
    regular_input_tokens = (
        prompt_tokens - cache_creation_input_tokens - cache_read_input_tokens
    )
    if regular_input_tokens < 0:
        regular_input_tokens = 0

    input_cost = (
        input_cost_per_token * regular_input_tokens
        + cache_creation_cost * cache_creation_input_tokens
        + cache_read_cost * cache_read_input_tokens
    )
    output_cost = output_cost_per_token * completion_tokens

    return input_cost + output_cost


def cost_per_token(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> tuple[float, float]:
    """Compute separate input and output costs.

    This function mirrors litellm.cost_calculator.cost_per_token() API.

    Args:
        model: The model identifier.
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.

    Returns:
        Tuple of (prompt_cost, completion_cost) in USD.
        Returns (0.0, 0.0) if the model is not found.
    """
    info = _PRICING_DATA.get(model)
    if not info:
        return (0.0, 0.0)

    input_cost_per_token = info.get("input_cost_per_token", 0) or 0
    output_cost_per_token = info.get("output_cost_per_token", 0) or 0

    prompt_cost = input_cost_per_token * prompt_tokens
    completion_cost = output_cost_per_token * completion_tokens

    return (prompt_cost, completion_cost)


def get_model_info(model: str) -> dict[str, Any]:
    """Get model metadata (context window, capabilities, etc.).

    Args:
        model: The model identifier.

    Returns:
        A dictionary of model metadata, or an empty dict if not found.
        Keys include:
        - max_input_tokens: Maximum context window size
        - max_output_tokens: Maximum output tokens
        - input_cost_per_token: Cost per input token in USD
        - output_cost_per_token: Cost per output token in USD
    """
    return dict(_PRICING_DATA.get(model, {}))


def get_all_models() -> list[str]:
    """Get a list of all known model names.

    Returns:
        List of model identifiers in the pricing database.
    """
    return list(_PRICING_DATA.keys())
