"""Per-field merge rules for ``LLMResponse.extra`` dicts.

The TailCallDelegationClient merges extras from two LLM calls into a
single ``LLMResponse``. Most fields today are additive token counts
(e.g. ``reasoning_tokens``, ``cache_creation_input_tokens``) or dicts
of additive counts (e.g. Anthropic's ``cache_creation`` 5m/1h
sub-buckets), and the default type-based dispatch handles these.

When a provider introduces a field where summation is wrong — e.g. vLLM
``completion_token_ids`` should concatenate, Gemini
``promptTokensDetails`` is a list keyed by modality — register a
custom rule at provider module import time::

    from jaz.providers.extras_merge import register_extra_merge_rule

    register_extra_merge_rule("prompt_token_ids", list.__add__)

Registry lookup is by top-level field name only. Nested fields inside
a dict use type-based dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

MergeRule = Callable[[Any, Any], Any]

_EXTRA_MERGE_RULES: dict[str, MergeRule] = {}


def register_extra_merge_rule(field_name: str, rule: MergeRule) -> None:
    """Register a merge rule for a specific ``extra`` field name.

    The rule receives ``(existing_value, new_value)`` and returns the
    merged value. Overrides the default type-based dispatch for that
    field name. Re-registering replaces the previous rule.
    """
    _EXTRA_MERGE_RULES[field_name] = rule


def get_extra_merge_rule(field_name: str) -> MergeRule | None:
    """Return the registered rule for ``field_name``, or None if unregistered."""
    return _EXTRA_MERGE_RULES.get(field_name)


def _default_merge_value(a: Any, b: Any) -> Any:
    """Type-based fallback merge (no registry lookup).

    * int + int → sum
    * dict + dict → recursive merge of sub-fields (type-based only)
    * list + list → concatenate
    * anything else → last-wins (``b``)
    """
    if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
        return a + b
    if isinstance(a, dict) and isinstance(b, dict):
        return _merge_dicts_typeonly(a, b)
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    return b


def _merge_dicts_typeonly(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge using only type-based dispatch (no registry)."""
    merged: dict[str, Any] = {**a}
    for key, value in b.items():
        if key in merged:
            merged[key] = _default_merge_value(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_extras(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two ``LLMResponse.extra`` dicts.

    For each key present in both inputs, applies the registered rule if
    one exists; otherwise falls back to ``_default_merge_value``. Keys
    present in only one input pass through unchanged.
    """
    merged: dict[str, Any] = {**a}
    for key, value in b.items():
        if key not in merged:
            merged[key] = value
            continue
        rule = _EXTRA_MERGE_RULES.get(key)
        if rule is not None:
            merged[key] = rule(merged[key], value)
        else:
            merged[key] = _default_merge_value(merged[key], value)
    return merged
