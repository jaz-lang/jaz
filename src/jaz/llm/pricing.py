"""Model pricing and metadata from bundled LiteLLM data.

This module loads pricing data from the bundled model_prices.json file
(sourced from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
and provides functions for cost calculation and model info retrieval.

The bundled JSON is filtered to ``litellm_provider in {openai, anthropic}``
because those are the only backends that ship built in.
To refresh from upstream and re-filter::

    curl -sSL https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json \\
      | python -c "import json,sys; d=json.load(sys.stdin); \\
        json.dump({k:v for k,v in d.items() \\
          if isinstance(v,dict) and v.get('litellm_provider') in {'openai','anthropic'}}, \\
        sys.stdout, indent=2)" \\
      > src/jaz/llm/model_prices.json

    (Upstream's ``sample_spec`` template entry is intentionally dropped — it is
    not a real model and would otherwise leak into ``get_all_models()`` and be
    accepted by ``compute_cost``.)

When wiring up a new provider (Gemini, vLLM, etc.), add its
``litellm_provider`` tag to the filter set above and regenerate.

Cost calculation supports:

* **Service tiers** — OpenAI/Azure/Gemini publish per-tier rates
  (``priority``, ``flex``, ``batch``). Pass ``service_tier`` to
  ``compute_cost`` to pick the matching rate; falls back to the
  standard rate when a tier-specific key isn't published for that
  model.
* **Long-context overage** — Anthropic charges 2× rates when a prompt
  exceeds 200k tokens; Gemini has similar 128k/256k/272k thresholds.
  Detected automatically from the prompt size.
* **Prompt caching** — cache reads and cache writes are billed at
  different rates than regular input tokens.

Caller passes ``prompt_tokens`` as the total INCLUDING cached tokens
(OpenAI/ATIF convention). We subtract the cache buckets to find the
"new" non-cached portion priced at the regular input rate.
"""

# TODO(#1027): nothing detects that the bundled snapshot has gone stale. The refresh command in
# the docstring above is entirely manual, so every new model family reopens a window in which its
# cost cannot be computed and a `cost_budget` set on it cannot be enforced. That used to degrade
# silently — BudgetPool asserted, the dispatcher swallowed it, and one run reached 164 turns under
# a cap that never applied. It now fails loudly instead, but failing at snapshot level is cheaper
# than failing at enforcement level: a scheduled CI job that re-runs the command above and flags a
# diff against the bundled file would close the window rather than report it.

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any

# Load bundled JSON at import time
_PRICING_DATA: dict[str, Any] = json.loads(
    files("jaz.llm").joinpath("model_prices.json").read_text()
)

# Map our public ``service_tier`` argument to LiteLLM JSON key suffixes.
# There is deliberately no ``batch`` entry: batch pricing is billed through
# OpenAI's separate Batch API endpoint, not selected by ``service_tier`` (the
# SDK's is Literal["auto","default","flex","scale","priority"]). A former
# ``"batch": "batches"`` mapping applied the Batch API's ~50% discount to
# standard-rate requests, under-reporting cost — the direction ``cost_budget``
# cannot tolerate. See design/design_features/litellm_sole_backend_v1.md (fix 1).
# Any unrecognized tier falls back to "" (standard) via ``.get(tier, "")``.
_TIER_SUFFIX = {
    "priority": "priority",
    "flex": "flex",
    "standard": "",
    "default": "",
    "auto": "",
    None: "",
}

# Match overage keys like ``input_cost_per_token_above_200k_tokens``.
_OVERAGE_RE = re.compile(r"_above_(\d+)k_tokens(?:$|_)")


def _detect_overage_suffix(info: dict[str, Any], prompt_tokens: int) -> str:
    """Pick the long-context overage suffix that applies for ``prompt_tokens``.

    Returns a suffix like ``"above_200k_tokens"`` for use in key lookups,
    or ``""`` when no overage tier applies (no thresholds in the model
    info, or prompt is below all of them).

    When multiple thresholds are published (rare), the highest one that
    ``prompt_tokens`` exceeds wins.
    """
    thresholds = set()
    for key in info:
        match = _OVERAGE_RE.search(key)
        if match:
            thresholds.add(int(match.group(1)))
    applicable = [t for t in thresholds if prompt_tokens > t * 1000]
    if not applicable:
        return ""
    return f"above_{max(applicable)}k_tokens"


def _resolve_rate(
    info: dict[str, Any],
    base_key: str,
    tier_suffix: str,
    overage_suffix: str,
) -> float:
    """Find the most specific published rate for a token bucket.

    Lookup order (most specific → most generic):

        {base}_{overage}_{tier}    e.g. input_cost_per_token_above_200k_tokens_priority
        {base}_{overage}           e.g. input_cost_per_token_above_200k_tokens
        {base}_{tier}              e.g. input_cost_per_token_priority
        {base}                     e.g. input_cost_per_token

    Returns 0 if none of the candidates is present (rate not published
    for this bucket on this model).
    """
    # Precedence order (most specific first). When the combined
    # {base}_{overage}_{tier} key isn't published, the overage rate is
    # deliberately preferred over the tier rate: past the 200k threshold the
    # overage multiplier dominates, so it's the safer (higher) estimate.
    candidates: list[str] = []
    if overage_suffix and tier_suffix:
        candidates.append(f"{base_key}_{overage_suffix}_{tier_suffix}")
    if overage_suffix:
        candidates.append(f"{base_key}_{overage_suffix}")
    if tier_suffix:
        candidates.append(f"{base_key}_{tier_suffix}")
    candidates.append(base_key)
    for key in candidates:
        rate = info.get(key)
        if rate is not None:
            return rate
    return 0


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    service_tier: str | None = None,
) -> float | None:
    """Compute cost in USD from token counts.

    Args:
        model: The model identifier (e.g., ``"gpt-5"``, ``"claude-sonnet-4-5"``).
        prompt_tokens: Total input tokens INCLUDING cached tokens (ATIF
            convention). We subtract the cache buckets to find the
            "new" non-cached portion priced at the regular input rate.
        completion_tokens: Number of output/completion tokens.
        cache_creation_input_tokens: Tokens written to cache (Anthropic;
            zero for OpenAI/Gemini, which have no separate creation surcharge).
        cache_read_input_tokens: Tokens read from cache.
        service_tier: Optional API service tier — one of ``"flex"``,
            ``"standard"``, ``"priority"``, ``"batch"``, or ``None`` (treated
            as standard). Picks the matching ``_priority`` / ``_flex`` /
            ``_batches`` rate keys when published; falls back to standard
            rates otherwise.

    Returns:
        Cost in USD, or ``None`` if the model is unknown.

    Notes:
        * Anthropic's 1-hour cache write rate is not yet modeled — all
          cache_creation tokens are priced at the (5-minute) standard
          rate. Splitting requires the 5m/1h breakdown from
          ``usage.extra["cache_creation"]``.
        * Long-context overage (Anthropic 200k+, Gemini 272k+) is
          detected automatically and applies the higher rate to ALL
          tokens in the request (Anthropic's documented behavior).
    """
    info = _PRICING_DATA.get(model)
    if not info:
        return None

    tier_suffix = _TIER_SUFFIX.get(service_tier, "")
    overage_suffix = _detect_overage_suffix(info, prompt_tokens)

    input_rate = _resolve_rate(
        info, "input_cost_per_token", tier_suffix, overage_suffix
    )
    output_rate = _resolve_rate(
        info, "output_cost_per_token", tier_suffix, overage_suffix
    )
    cache_creation_rate = _resolve_rate(
        info, "cache_creation_input_token_cost", tier_suffix, overage_suffix
    )
    cache_read_rate = _resolve_rate(
        info, "cache_read_input_token_cost", tier_suffix, overage_suffix
    )

    # Regular (non-cached) input tokens = total prompt minus cache buckets.
    regular_input_tokens = max(
        0,
        prompt_tokens - cache_creation_input_tokens - cache_read_input_tokens,
    )

    input_cost = (
        input_rate * regular_input_tokens
        + cache_creation_rate * cache_creation_input_tokens
        + cache_read_rate * cache_read_input_tokens
    )
    output_cost = output_rate * completion_tokens

    return input_cost + output_cost


def cost_per_token(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    service_tier: str | None = None,
) -> tuple[float, float]:
    """Compute separate input and output costs.

    Mirrors LiteLLM's ``cost_calculator.cost_per_token()`` shape. Honors
    ``service_tier`` the same way ``compute_cost`` does.

    Returns:
        ``(prompt_cost, completion_cost)`` in USD; ``(0.0, 0.0)`` for
        unknown models.
    """
    info = _PRICING_DATA.get(model)
    if not info:
        return (0.0, 0.0)

    tier_suffix = _TIER_SUFFIX.get(service_tier, "")
    overage_suffix = _detect_overage_suffix(info, prompt_tokens)

    input_rate = _resolve_rate(
        info, "input_cost_per_token", tier_suffix, overage_suffix
    )
    output_rate = _resolve_rate(
        info, "output_cost_per_token", tier_suffix, overage_suffix
    )

    return (input_rate * prompt_tokens, output_rate * completion_tokens)


def get_model_info(model: str) -> dict[str, Any]:
    """Get model metadata (context window, capabilities, etc.).

    Args:
        model: The model identifier.

    Returns:
        A dictionary of model metadata, or an empty dict if not found.
        Keys include:
        - max_input_tokens: Maximum context window size
        - max_output_tokens: Maximum output tokens
        - input_cost_per_token: Cost per input token in USD (standard tier)
        - output_cost_per_token: Cost per output token in USD (standard tier)
        - input_cost_per_token_priority / _flex / _batches: tier rates
          (present only for models that support service tiers)
    """
    return dict(_PRICING_DATA.get(model, {}))


def get_all_models() -> list[str]:
    """Get a list of all known model names.

    Returns:
        List of model identifiers in the pricing database.
    """
    return list(_PRICING_DATA.keys())
