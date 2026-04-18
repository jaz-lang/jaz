from __future__ import annotations

import copy
import json
import numbers
import os
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .library.jaz import InvokeUsage
    from .llm_client import LLMClient


@dataclass
class DepthConfigOverride:
    """Per-depth override of llm_client and/or model_config.

    Fields left as None inherit the top-level Config.llm_client /
    Config.model_config.
    """

    llm_client: str | LLMClient | None = None
    model_config: dict[str, Any] | None = None


def _validate_depth_config_overrides(
    value: object,
) -> list[DepthConfigOverride] | None:
    """Validate and normalize depth_config_overrides.

    Accepts either a list of DepthConfigOverride instances or a list of dicts
    (from YAML/JSON); dicts are coerced to DepthConfigOverride. Returns the
    normalized list (or None)."""
    from .llm_client import LLMClient as _LLMClient

    if value is None:
        return None
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError("depth_config_overrides must be a non-empty list or None")

    allowed_keys = {"llm_client", "model_config"}
    normalized: list[DepthConfigOverride] = []
    for i, entry in enumerate(value):
        if isinstance(entry, DepthConfigOverride):
            override = entry
        elif isinstance(entry, dict):
            unknown = set(entry.keys()) - allowed_keys
            if unknown:
                raise ValueError(
                    f"depth_config_overrides[{i}] has unknown keys: {unknown}. "
                    f"Allowed keys: {allowed_keys}"
                )
            override = DepthConfigOverride(**entry)
        else:
            raise ValueError(
                f"depth_config_overrides[{i}] must be a DepthConfigOverride "
                f"or dict, got {type(entry).__name__}"
            )
        if override.llm_client is not None and not isinstance(
            override.llm_client, (str, _LLMClient)
        ):
            raise ValueError(
                f"depth_config_overrides[{i}].llm_client must be a string "
                f"or LLMClient instance"
            )
        if override.model_config is not None and not isinstance(
            override.model_config, dict
        ):
            raise ValueError(f"depth_config_overrides[{i}].model_config must be a dict")
        normalized.append(override)
    return normalized


@dataclass
class Config:
    """JAZ framework configuration"""

    # Whether to add external <thinking> tags to the prompt for step-by-step reasoning.
    # If True, prompts require explicit <thinking> tags.
    # If False (default), the model is assumed to have built-in reasoning (e.g., o1, o3).
    add_external_thinking: bool = False

    # LLM client type for making completion calls (default: "litellm")
    # Use "rlm" for RLM-based clients. Client is constructed by Agent at init.
    llm_client: str | LLMClient = "litellm"

    # LLM completion/construction parameters.
    # Passed as kwargs to the provider's completion call (e.g., temperature, max_tokens).
    # For RLM: used as construction params (backend, backend_kwargs, max_depth, etc.).
    # Only None and unset keys use the provider's default.
    model_config: dict[str, Any] = field(
        default_factory=lambda: {"model": "openai/gpt-5-mini"}
    )

    # Per-depth LLM configuration overrides.
    # A list of DepthConfigOverride instances; index 0 is used for depth 1,
    # index 1 for depth 2, etc. Beyond the last entry, the last entry is reused.
    # Unset fields (None) on an entry inherit from the top-level llm_client/model_config.
    # When the whole list is None (default), the top-level fields are used for all depths.
    depth_config_overrides: list[DepthConfigOverride] | None = None

    # LLM retry settings
    max_llm_attempts: int = 10
    llm_retry_wait_multiplier: float = 1.0
    llm_retry_wait_min: float = 4.0
    llm_retry_wait_max: float = 60.0

    # REPL settings
    max_repl_output_length: int = 10000
    max_repl_iterations: int = 10
    max_repl_iterations_buffer: int = 3
    max_repl_invoke_calls: int = 5
    max_repl_recursion: int = 1
    repl_exec_timeout: float | None = 30.0  # TODO: should default be None?
    delegate_exec_timeout: float | None = (
        None  # Timeout shown in delegate template (None = don't show timeout attribute)
    )
    # REPL configurations per language (e.g., {"python": {"multi_statement": False}})
    # If empty, REPL defaults are used
    repl_configs: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    # Optional callable to validate REPL input source code before execution.
    # Propagates to all nested jaz.invoke() calls automatically via config.
    # Set via jaz.configure(repl_input_validator=fn) — cannot be set in YAML.
    repl_input_validator: Callable[[str], None] | None = None

    # JAZ library invoke usage patterns (controls jaz.invoke documentation)
    # Env var JAZ_INVOKE_USAGE: comma-separated flags (CONTEXT_FILTER,SEARCH,SELF_IMPROVEMENT,SUBTASK)
    # or "all" for all patterns, "none" for no patterns
    invoke_usage: InvokeUsage = field(
        default_factory=lambda: __import__(
            "jaz.library.jaz", fromlist=["InvokeUsage"]
        ).InvokeUsage(0)
    )
    # Include jaz.Library class in the JAZ library (for creating custom tool libraries)
    # Env var JAZ_INCLUDE_LIBRARY: "true"/"1"/"yes" or "false"/"0"/"no"
    include_library: bool = False

    # Budget forcing - refuse early RETURN/RAISE to encourage more reasoning
    # Number of times to refuse finishing before any budget limit is hit (0 = disabled)
    budget_forcing_refusals: int = 0

    # Budget settings
    max_cost_budget: float | None = None  # Maximum cost in USD (None = no limit)
    max_cost_buffer: float = 0.10  # Buffer amount in USD before hard error
    max_llm_calls_budget: int | None = None  # Maximum LLM calls (None = no limit)
    max_llm_calls_buffer: int = 5  # Buffer LLM calls before hard error
    cost_tracking_json_path: str | None = None  # Path to save cost tracking JSON

    # Context window warning
    context_window_fraction: float | None = (
        None  # e.g., 0.9 = warn at 90% of max_input_tokens
    )

    # Display settings
    show_tools_in_prompt: bool = True  # Whether to show tool libraries in system prompt
    simplify_recursive_invoke: bool = False  # If True, sub-agents see a simplified jaz.invoke(prompt, return_type, return_validator, **inputs)
    full_docstrings_in_prompt: bool = False  # Whether to show full docstrings (vs. first line only) in tool library descriptions
    input_truncation_advice_template: str | None = (
        None  # Jinja2 template for input truncation advice (None = disabled)
    )
    repl_output_truncation_advice_template: str | None = (
        "repl_output_truncation_advice.jinja2"  # Jinja2 template for REPL output truncation advice (None = disabled)
    )
    # Maximum length for input strings in prompt inputs and task string
    max_input_length: int = 10000  # Maximum length for input strings in prompt inputs
    max_task_length: int = 10000  # Maximum task string length
    truncation_prefix_ratio: float = (
        0.8  # Ratio of prefix to total in truncated output (0.0 to 1.0)
    )

    # Template paths (relative to jaz/templates/)
    system_prompt_template: str = "system_prompt.jinja2"
    user_prompt_template: str = "user_prompt.jinja2"
    jaz_library_description_template: str = "jaz_library_description.jinja2"
    must_exit_warning_template: str = "must_exit_warning.jinja2"

    # Security - REPL sandboxing
    forbidden_names: list[str] = field(
        default_factory=lambda: ["exit", "quit", "SystemExit"]
    )
    forbidden_attributes: list[str] = field(
        default_factory=lambda: [
            "__closure__",
            "__getattribute__",
            "__getattr__",
            "__setattr__",
        ]
    )
    # See https://docs.python.org/3/library/index.html
    # for standard library (not comprehensive)
    allowed_imports: list[str] = field(
        default_factory=lambda: [
            "time",
            "math",
            "random",
            "datetime",
            "re",
            "string",
            "itertools",
            "functools",
            "collections",
        ]
    )

    def __post_init__(self):
        """Load environment variable overrides and validate"""
        update_dict = {}
        # Environment variable loading
        if val := os.environ.get("JAZ_ADD_EXTERNAL_THINKING"):
            val_lower = val.lower()
            if val_lower in ("auto", "none"):
                update_dict["add_external_thinking"] = None
            else:
                update_dict["add_external_thinking"] = val_lower in ("true", "1", "yes")
        if val := os.environ.get("JAZ_MODEL_CONFIG"):
            try:
                update_dict["model_config"] = json.loads(val)
            except json.JSONDecodeError as e:
                raise ValueError(f"JAZ_MODEL_CONFIG must be valid JSON: {e}") from e
        if val := os.environ.get("JAZ_DEPTH_CONFIG_OVERRIDES"):
            try:
                update_dict["depth_config_overrides"] = json.loads(val)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JAZ_DEPTH_CONFIG_OVERRIDES must be valid JSON: {e}"
                ) from e
        if val := os.environ.get("JAZ_MAX_LLM_ATTEMPTS"):
            update_dict["max_llm_attempts"] = int(val)
        if val := os.environ.get("JAZ_LLM_RETRY_WAIT_MULTIPLIER"):
            update_dict["llm_retry_wait_multiplier"] = float(val)
        if val := os.environ.get("JAZ_LLM_RETRY_WAIT_MIN"):
            update_dict["llm_retry_wait_min"] = float(val)
        if val := os.environ.get("JAZ_LLM_RETRY_WAIT_MAX"):
            update_dict["llm_retry_wait_max"] = float(val)
        if val := os.environ.get("JAZ_MAX_REPL_OUTPUT_LENGTH"):
            update_dict["max_repl_output_length"] = int(val)
        if val := os.environ.get("JAZ_MAX_REPL_ITERATIONS"):
            update_dict["max_repl_iterations"] = int(val)
        if val := os.environ.get("JAZ_MAX_REPL_ITERATIONS_BUFFER"):
            update_dict["max_repl_iterations_buffer"] = int(val)
        if val := os.environ.get("JAZ_MAX_REPL_INVOKE_CALLS"):
            update_dict["max_repl_invoke_calls"] = int(val)
        if val := os.environ.get("JAZ_MAX_REPL_RECURSION"):
            update_dict["max_repl_recursion"] = int(val)
        if val := os.environ.get("JAZ_REPL_EXEC_TIMEOUT"):
            update_dict["repl_exec_timeout"] = float(val)
        if val := os.environ.get("JAZ_DELEGATE_EXEC_TIMEOUT"):
            update_dict["delegate_exec_timeout"] = float(val)
        if val := os.environ.get("JAZ_BUDGET_FORCING_REFUSALS"):
            update_dict["budget_forcing_refusals"] = int(val)
        if val := os.environ.get("JAZ_MAX_COST_BUDGET"):
            update_dict["max_cost_budget"] = float(val)
        if val := os.environ.get("JAZ_MAX_COST_BUFFER"):
            update_dict["max_cost_buffer"] = float(val)
        if val := os.environ.get("JAZ_MAX_LLM_CALLS_BUDGET"):
            update_dict["max_llm_calls_budget"] = int(val)
        if val := os.environ.get("JAZ_MAX_LLM_CALLS_BUFFER"):
            update_dict["max_llm_calls_buffer"] = int(val)
        if val := os.environ.get("JAZ_COST_TRACKING_JSON_PATH"):
            update_dict["cost_tracking_json_path"] = val
        if val := os.environ.get("JAZ_CONTEXT_WINDOW_FRACTION"):
            update_dict["context_window_fraction"] = float(val)
        if val := os.environ.get("JAZ_INVOKE_USAGE"):
            from .library.jaz import parse_invoke_usage

            update_dict["invoke_usage"] = parse_invoke_usage(val)
        if val := os.environ.get("JAZ_INCLUDE_LIBRARY"):
            update_dict["include_library"] = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("JAZ_SHOW_TOOLS_IN_PROMPT"):
            update_dict["show_tools_in_prompt"] = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("JAZ_FULL_DOCSTRINGS_IN_PROMPT"):
            update_dict["full_docstrings_in_prompt"] = val.lower() in (
                "true",
                "1",
                "yes",
            )
        if val := os.environ.get("JAZ_MAX_INPUT_LENGTH"):
            update_dict["max_input_length"] = int(val)
        if val := os.environ.get("JAZ_MAX_TASK_LENGTH"):
            update_dict["max_task_length"] = int(val)
        if val := os.environ.get("JAZ_SYSTEM_PROMPT_TEMPLATE"):
            update_dict["system_prompt_template"] = val
        if val := os.environ.get("JAZ_USER_PROMPT_TEMPLATE"):
            update_dict["user_prompt_template"] = val
        if val := os.environ.get("JAZ_JAZ_LIBRARY_DESCRIPTION_TEMPLATE"):
            update_dict["jaz_library_description_template"] = val
        if val := os.environ.get("JAZ_MUST_EXIT_WARNING_TEMPLATE"):
            update_dict["must_exit_warning_template"] = val
        if val := os.environ.get("JAZ_INPUT_TRUNCATION_ADVICE_TEMPLATE"):
            update_dict["input_truncation_advice_template"] = val
        if val := os.environ.get("JAZ_REPL_OUTPUT_TRUNCATION_ADVICE_TEMPLATE"):
            update_dict["repl_output_truncation_advice_template"] = val
        if val := os.environ.get("JAZ_REPL_CONFIGS"):
            try:
                update_dict["repl_configs"] = json.loads(val)
            except json.JSONDecodeError as e:
                raise ValueError(f"JAZ_REPL_CONFIGS must be valid JSON: {e}") from e
        # Update fields
        self.update(**update_dict)

    def _validate(self, update_dict):
        """Validate configuration values"""
        # LLM client validation
        if "llm_client" in update_dict:
            from .llm_client import LLMClient as _LLMClient

            if not isinstance(update_dict["llm_client"], (str, _LLMClient)):
                raise ValueError(
                    "llm_client must be a string (e.g., 'litellm', 'rlm') "
                    "or an LLMClient instance"
                )

        # LLM completion parameters validation
        if "model_config" in update_dict:
            if not isinstance(update_dict["model_config"], dict):
                raise ValueError("model_config must be a dict")

        # Per-depth LLM config validation. Dict entries (from YAML/JSON) are
        # coerced to DepthConfigOverride in-place so the stored field is always
        # a list of dataclasses.
        if "depth_config_overrides" in update_dict:
            update_dict["depth_config_overrides"] = _validate_depth_config_overrides(
                update_dict["depth_config_overrides"]
            )

        # Other validation
        if update_dict.get("max_llm_attempts", 1) <= 0:
            raise ValueError("max_llm_attempts must be positive")
        if "llm_retry_wait_multiplier" in update_dict:
            if (
                not isinstance(update_dict["llm_retry_wait_multiplier"], numbers.Real)
                or update_dict["llm_retry_wait_multiplier"] <= 0
            ):
                raise ValueError("llm_retry_wait_multiplier must be positive")
        if "llm_retry_wait_min" in update_dict:
            if (
                not isinstance(update_dict["llm_retry_wait_min"], numbers.Real)
                or update_dict["llm_retry_wait_min"] < 0
            ):
                raise ValueError("llm_retry_wait_min must be non-negative")
        if "llm_retry_wait_max" in update_dict:
            if (
                not isinstance(update_dict["llm_retry_wait_max"], numbers.Real)
                or update_dict["llm_retry_wait_max"] <= 0
            ):
                raise ValueError("llm_retry_wait_max must be positive")
        if update_dict.get("max_repl_output_length", 1) <= 0:
            raise ValueError("max_repl_output_length must be positive")
        if update_dict.get("max_repl_iterations", 1) <= 0:
            raise ValueError("max_repl_iterations must be positive")
        if update_dict.get("max_repl_iterations_buffer", 1) <= 0:
            raise ValueError("max_repl_iterations_buffer must be positive")
        if update_dict.get("max_repl_invoke_calls", 1) <= 0:
            raise ValueError("max_repl_invoke_calls must be positive")
        if update_dict.get("max_repl_recursion", 1) <= 0:
            raise ValueError("max_repl_recursion must be positive")
        if "repl_exec_timeout" in update_dict:
            timeout = update_dict["repl_exec_timeout"]
            if timeout is not None and (
                not isinstance(timeout, numbers.Real) or timeout <= 0
            ):
                raise ValueError("repl_exec_timeout must be positive or None")
        if "delegate_exec_timeout" in update_dict:
            timeout = update_dict["delegate_exec_timeout"]
            if timeout is not None and (
                not isinstance(timeout, numbers.Real) or timeout <= 0
            ):
                raise ValueError("delegate_exec_timeout must be positive or None")
        if update_dict.get("max_input_length", 1) <= 0:
            raise ValueError("max_input_length must be positive")
        if "max_task_length" in update_dict:
            max_task_length = update_dict["max_task_length"]
            if not isinstance(max_task_length, int) or max_task_length <= 0:
                raise ValueError("max_task_length must be a positive integer")
        if "truncation_prefix_ratio" in update_dict:
            pr = update_dict["truncation_prefix_ratio"]
            if not isinstance(pr, (int, float)) or pr < 0.0 or pr > 1.0:
                raise ValueError(
                    "truncation_prefix_ratio must be a float between 0.0 and 1.0"
                )
        if "budget_forcing_refusals" in update_dict and (
            not isinstance(update_dict["budget_forcing_refusals"], int)
            or update_dict["budget_forcing_refusals"] < 0
        ):
            raise ValueError("budget_forcing_refusals must be a non-negative integer")
        if any(
            not name.isidentifier() for name in update_dict.get("forbidden_names", [])
        ):
            raise ValueError("All forbidden_names must be valid Python identifiers")
        if any(
            not name.isidentifier()
            for name in update_dict.get("forbidden_attributes", [])
        ):
            raise ValueError(
                "All forbidden_attributes must be valid Python identifiers"
            )
        if any(
            not name.isidentifier() for name in update_dict.get("allowed_imports", [])
        ):
            raise ValueError("All allowed_imports must be valid Python identifiers")
        if "max_cost_budget" in update_dict:
            budget = update_dict["max_cost_budget"]
            if budget is not None and (
                not isinstance(budget, numbers.Real) or budget <= 0
            ):
                raise ValueError("max_cost_budget must be positive or None")
        if "max_cost_buffer" in update_dict:
            buffer = update_dict["max_cost_buffer"]
            if not isinstance(buffer, numbers.Real) or buffer < 0:
                raise ValueError("max_cost_buffer must be non-negative")
        if "max_llm_calls_budget" in update_dict:
            budget = update_dict["max_llm_calls_budget"]
            if budget is not None and (not isinstance(budget, int) or budget <= 0):
                raise ValueError(
                    "max_llm_calls_budget must be a positive integer or None"
                )
        if "max_llm_calls_buffer" in update_dict:
            buffer = update_dict["max_llm_calls_buffer"]
            if not isinstance(buffer, int) or buffer < 0:
                raise ValueError("max_llm_calls_buffer must be a non-negative integer")
        if "context_window_fraction" in update_dict:
            val = update_dict["context_window_fraction"]
            if val is not None and (
                not isinstance(val, numbers.Real) or not (0.0 < float(val) < 1.0)
            ):
                raise ValueError(
                    "context_window_fraction must be between 0 and 1 (exclusive), or None"
                )
        if "invoke_usage" in update_dict:
            from .library.jaz import InvokeUsage

            usage = update_dict["invoke_usage"]
            if not isinstance(usage, InvokeUsage):
                raise ValueError(
                    "invoke_usage must be an InvokeUsage flag. "
                    "To combine flags, use the '|' operator, e.g., InvokeUsage.SUBTASK | InvokeUsage.SEARCH."
                )
        if "include_library" in update_dict:
            if not isinstance(update_dict["include_library"], bool):
                raise ValueError("include_library must be a boolean")
        if "show_tools_in_prompt" in update_dict:
            if not isinstance(update_dict["show_tools_in_prompt"], bool):
                raise ValueError("show_tools_in_prompt must be a boolean")
        if "simplify_recursive_invoke" in update_dict:
            if not isinstance(update_dict["simplify_recursive_invoke"], bool):
                raise ValueError("simplify_recursive_invoke must be a boolean")
        if "full_docstrings_in_prompt" in update_dict:
            if not isinstance(update_dict["full_docstrings_in_prompt"], bool):
                raise ValueError("full_docstrings_in_prompt must be a boolean")
        for template_field in (
            "system_prompt_template",
            "user_prompt_template",
            "jaz_library_description_template",
            "must_exit_warning_template",
        ):
            if template_field in update_dict:
                val = update_dict[template_field]
                if not isinstance(val, str) or not val:
                    raise ValueError(f"{template_field} must be a non-empty string")
        for optional_template_field in (
            "input_truncation_advice_template",
            "repl_output_truncation_advice_template",
        ):
            if optional_template_field in update_dict:
                val = update_dict[optional_template_field]
                if val is not None and (not isinstance(val, str) or not val):
                    raise ValueError(
                        f"{optional_template_field} must be a non-empty string or None"
                    )
        if "repl_configs" in update_dict:
            repl_configs = update_dict["repl_configs"]
            if not isinstance(repl_configs, dict):
                raise ValueError("repl_configs must be a dict")
            for lang, config in repl_configs.items():
                if not isinstance(lang, str):
                    raise ValueError(
                        "repl_configs keys must be strings (language names)"
                    )
                if not isinstance(config, dict):
                    raise ValueError(
                        f"repl_configs['{lang}'] must be a dict, got {type(config).__name__}"
                    )

    def resolve_for_depth(self, depth: int) -> tuple[str | LLMClient, dict[str, Any]]:
        """Return (llm_client, model_config) for the given recursion depth.

        If depth_config_overrides is set, uses the entry at index (depth - 1),
        clamped to the last entry. Fields that are None on the override fall
        back to self.llm_client / self.model_config.

        If depth_config_overrides is None, returns (self.llm_client, self.model_config).
        """
        if self.depth_config_overrides is None:
            return self.llm_client, self.model_config

        idx = min(depth - 1, len(self.depth_config_overrides) - 1)
        override = self.depth_config_overrides[idx]

        llm_client = (
            override.llm_client if override.llm_client is not None else self.llm_client
        )
        model_config = (
            override.model_config
            if override.model_config is not None
            else self.model_config
        )
        return llm_client, model_config

    def update(self, **kwargs):
        """Update multiple config values at once"""
        # Validate updates
        self._validate(kwargs)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown config option: {key}")
            setattr(self, key, value)

    @staticmethod
    def _serialize_invoke_usage(usage: InvokeUsage) -> list[str]:
        """Serialize InvokeUsage flag to a list of flag names for JSON."""
        from .library.jaz import InvokeUsage

        return [flag.name for flag in InvokeUsage if flag in usage and flag.name]

    def to_dict(self) -> dict:
        """Convert config to a JSON-serializable dictionary."""
        return {
            "llm_client": self.llm_client
            if isinstance(self.llm_client, str)
            else type(self.llm_client).__name__,
            "add_external_thinking": self.add_external_thinking,
            "model_config": dict(self.model_config),
            "depth_config_overrides": (
                [
                    {
                        k: v
                        for k, v in {
                            "llm_client": override.llm_client
                            if isinstance(override.llm_client, str)
                            or override.llm_client is None
                            else type(override.llm_client).__name__,
                            "model_config": dict(override.model_config)
                            if override.model_config is not None
                            else None,
                        }.items()
                        if v is not None
                    }
                    for override in self.depth_config_overrides
                ]
                if self.depth_config_overrides is not None
                else None
            ),
            "max_llm_attempts": self.max_llm_attempts,
            "llm_retry_wait_multiplier": self.llm_retry_wait_multiplier,
            "llm_retry_wait_min": self.llm_retry_wait_min,
            "llm_retry_wait_max": self.llm_retry_wait_max,
            "max_repl_output_length": self.max_repl_output_length,
            "max_repl_iterations": self.max_repl_iterations,
            "max_repl_iterations_buffer": self.max_repl_iterations_buffer,
            "max_repl_invoke_calls": self.max_repl_invoke_calls,
            "max_repl_recursion": self.max_repl_recursion,
            "repl_exec_timeout": self.repl_exec_timeout,
            "delegate_exec_timeout": self.delegate_exec_timeout,
            "repl_configs": self.repl_configs,
            "show_tools_in_prompt": self.show_tools_in_prompt,
            "simplify_recursive_invoke": self.simplify_recursive_invoke,
            "full_docstrings_in_prompt": self.full_docstrings_in_prompt,
            "max_input_length": self.max_input_length,
            "max_task_length": self.max_task_length,
            "truncation_prefix_ratio": self.truncation_prefix_ratio,
            "invoke_usage": self._serialize_invoke_usage(self.invoke_usage),
            "include_library": self.include_library,
            "budget_forcing_refusals": self.budget_forcing_refusals,
            "max_cost_budget": self.max_cost_budget,
            "max_cost_buffer": self.max_cost_buffer,
            "max_llm_calls_budget": self.max_llm_calls_budget,
            "max_llm_calls_buffer": self.max_llm_calls_buffer,
            "cost_tracking_json_path": self.cost_tracking_json_path,
            "context_window_fraction": self.context_window_fraction,
            "system_prompt_template": self.system_prompt_template,
            "user_prompt_template": self.user_prompt_template,
            "jaz_library_description_template": self.jaz_library_description_template,
            "must_exit_warning_template": self.must_exit_warning_template,
            "input_truncation_advice_template": self.input_truncation_advice_template,
            "repl_output_truncation_advice_template": self.repl_output_truncation_advice_template,
            "forbidden_names": self.forbidden_names,
            "forbidden_attributes": self.forbidden_attributes,
            "allowed_imports": self.allowed_imports,
        }


# Global default configuration (loaded from environment variables)
# This is a singleton that can be modified via configure_defaults()
_default_config = Config()

# Context-local configuration overrides
# When set, get_config() returns this instead of the global default
_config_override: ContextVar[Config | None] = ContextVar(
    "jaz_config_override", default=None
)


def get_config() -> Config:
    """
    Get the current configuration.

    Returns the context-local override if one is set (via config_override),
    otherwise returns the global default configuration.

    Returns:
        The current Config instance
    """
    override = _config_override.get()
    if override is not None:
        return override
    return _default_config


def configure(**kwargs) -> None:
    """
    Configure the global default JAZ framework settings.

    This modifies the global default configuration that is used when no
    context-local override is active.

    Args:
        add_external_thinking: Whether to add external `<thinking>` tags to prompts.
            If True, prompts require explicit `<thinking>` tags.
            If False (default), the model is assumed to have built-in reasoning.
        llm_client: LLM client for making completion calls. Either a string
            (e.g., ``"litellm"``, ``"rlm"``) or an ``LLMClient`` instance
            (e.g., ``MockLLMClient(fn=...)``). Default: ``"litellm"``.
        model_config: Dict of completion parameters passed to the provider's API
            (e.g.,
            `"temperature"`, `"max_tokens"`, `"reasoning_effort"`).
            Only non-None values are passed; unset keys use the provider's default.
            Env var: `JAZ_MODEL_CONFIG` (JSON string, e.g.,
            `'{"temperature": 0.7, "max_tokens": 1000}'`)
        max_llm_attempts: Maximum LLM call attempts
        llm_retry_wait_multiplier: Multiplier for exponential backoff between retries
        llm_retry_wait_min: Minimum wait time in seconds between retries
        llm_retry_wait_max: Maximum wait time in seconds between retries
        max_repl_output_length: Maximum REPL output length in characters
        max_repl_invoke_calls: Maximum number of `jaz.invoke` calls in REPL
        max_repl_iterations: Maximum REPL iterations
        max_repl_iterations_buffer: Grace period iterations after max_iterations
        repl_exec_timeout: Timeout for exec/eval operations in seconds (`None` = no timeout)
        repl_configs: Per-language REPL configurations. Maps language names to config
            dicts (e.g., `{"python": {"multi_statement": False, "allow_raise": True}}`).
            Empty dict uses REPL defaults.
        budget_forcing_refusals: Number of times to refuse early RETURN/RAISE
            to encourage more reasoning (0 = disabled)
        forbidden_names: List of forbidden variable/function names in REPL
        forbidden_attributes: List of forbidden attribute names in REPL
        allowed_imports: List of allowed Python imports in REPL
        max_cost_budget: Maximum cost budget in USD (`None` = no limit)
        max_cost_buffer: Buffer amount in USD before hard error
        max_llm_calls_budget: Maximum number of LLM API calls (`None` = no limit)
        max_llm_calls_buffer: Buffer LLM calls before hard error
        cost_tracking_json_path: Path to save cost tracking JSON
        context_window_fraction: Fraction of `max_input_tokens` at which to warn the
            agent that the context window is nearly full (e.g., 0.9 = 90%). `None`
            (default) disables the warning. Env var: `JAZ_CONTEXT_WINDOW_FRACTION`
        invoke_usage: Controls which `jaz.invoke` usage patterns are documented.
            Use `InvokeUsage` flags combined with `|` (e.g.,
            `InvokeUsage.SUBTASK | InvokeUsage.SEARCH`), or `InvokeUsage.ALL`.
            Env var: `JAZ_INVOKE_USAGE` (comma-separated flags or "all"/"none")
        max_input_length: Maximum length for input strings in prompt inputs. Default 10000.
            Env var: `JAZ_MAX_INPUT_LENGTH`
        max_task_length: Maximum allowed length for the task string. Default 10000.
            Env var: `JAZ_MAX_TASK_LENGTH`
        include_library: Whether to include `jaz.Library` class in the JAZ library
            for creating custom tool libraries. Default `False`.
            Env var: `JAZ_INCLUDE_LIBRARY` ("true"/"false")

    Example:
        jaz.configure(model="openai/gpt-4", max_repl_iterations=20)
        jaz.configure(model_config={"temperature": 0.7, "top_p": 0.9})
        jaz.configure(invoke_usage=InvokeUsage.SUBTASK | InvokeUsage.SEARCH)
        jaz.configure(include_library=True)
    """
    _default_config.update(**kwargs)


@contextmanager
def config_override(**kwargs) -> Generator[Config, None, None]:
    """
    Context manager for temporary configuration overrides.

    Creates a deep copy of the current configuration (either the context-local
    override or the global default) and applies the specified overrides.
    The overridden configuration is active only within the context.

    This is the recommended way to use different configurations in different
    threads or contexts, as each context gets its own Config instance.

    Args:
        **kwargs: Configuration overrides (same as configure())

    Example:
        # Global default
        jaz.configure(max_repl_iterations=10)

        # Temporary override in a specific context
        with jaz.config_override(max_repl_iterations=5):
            # This context uses max_repl_iterations=5
            pass

        # Outside the context, max_repl_iterations is back to 10

    Example with threads:
        from concurrent.futures import ThreadPoolExecutor

        def worker():
            with jaz.config_override(max_repl_iterations=20):
                # This worker uses max_repl_iterations=20
                config = jaz.get_config()
                return config.max_repl_iterations

        with ThreadPoolExecutor() as executor:
            # Each worker gets its own config copy
            futures = [executor.submit(worker) for _ in range(10)]

    Example with invoke_usage and include_library:
        from jaz.library import InvokeUsage

        # Override invoke patterns for a specific invocation
        with jaz.config_override(invoke_usage=InvokeUsage.SUBTASK):
            jaz.invoke(...)

        # Enable Library class for a specific invocation
        with jaz.config_override(include_library=True):
            jaz.invoke(...)
    """
    # Get the current config (could be override or default)
    current_config = get_config()

    # Create a deep copy to avoid shared mutable state
    new_config = copy.deepcopy(current_config)

    # Apply the overrides
    if kwargs:
        new_config.update(**kwargs)

    # Set the context-local override
    token = _config_override.set(new_config)
    try:
        yield new_config
    finally:
        _config_override.reset(token)
