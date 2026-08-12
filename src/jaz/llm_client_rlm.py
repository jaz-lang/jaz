from __future__ import annotations

import logging
from typing import Any

from ._llm_client import LLM, LLMResponse
from .providers import cost_per_token
from .providers.registry import register_llm

# TODO: clean up and consolidate logging (logging module vs. our logger hooks)
logger = logging.getLogger(__name__)


@register_llm("rlm")
class RLMClient(LLM):
    """LLM client that wraps the RLM framework.

    Settings ride the base class's ``**request_defaults`` tail, so they are ordinary keyword
    arguments here. Those accepted by ``RLM.__init__`` are passed straight to it; ``log_dir``
    and ``root_prompt`` are consumed by this client (see below); any remaining key (e.g.
    ``reasoning_effort``) is forwarded into ``backend_kwargs`` as a convenience for
    backend-specific parameters — put backend params under ``backend_kwargs`` explicitly if you
    prefer to avoid that fallback.

    Usage::

        jaz.configure(
            llm=RLMClient(
                backend="openai",
                backend_kwargs={"api_key": "..."},
                max_depth=2,
            )
        )
    """

    def __init__(self, **retry: Any) -> None:
        super().__init__(**retry)
        try:
            import rlm  # pyright: ignore[reportMissingImports]  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "RLMClient requires the 'rlm' package. Install it with: pip install rlm"
            ) from e

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        import inspect

        from rlm import RLM  # pyright: ignore[reportMissingImports]
        from rlm.logger.rlm_logger import (  # pyright: ignore[reportMissingImports]
            RLMLogger,
        )

        # Derive the accepted RLM constructor keys from the signature so the set
        # stays in sync automatically as RLM evolves ('logger' is handled below).
        # NOTE: this assumes RLM.__init__ keeps a concrete signature. If RLM ever
        # switches to (self, **kwargs), this set becomes empty and every key falls
        # through into backend_kwargs. test_llm_client_rlm.py pins that assumption.
        rlm_keys = set(inspect.signature(RLM.__init__).parameters) - {"self", "logger"}

        # Split incoming kwargs: RLM constructor keys, this client's own
        # log_dir/root_prompt, and backend-specific extras (-> backend_kwargs).
        config: dict[str, Any] = {}
        extra_backend_kwargs: dict[str, Any] = {}
        log_dir: str | None = None
        root_prompt_override: str | None = None
        for k, v in kwargs.items():
            if k == "log_dir":
                log_dir = v
            elif k == "root_prompt":
                root_prompt_override = v
            elif k == "logger":
                raise TypeError(
                    "RLMClient manages its own logger from `log_dir`; pass "
                    "`log_dir=...` instead of `logger=...`."
                )
            elif k in rlm_keys:
                config[k] = v
            elif k != "model":  # skip dummy model key
                extra_backend_kwargs[k] = v

        # Ensure backend_kwargs has model_name
        if "backend_kwargs" not in config:
            config["backend_kwargs"] = {"model_name": model}
        elif "model_name" not in config["backend_kwargs"]:
            config["backend_kwargs"]["model_name"] = model

        # Merge extra kwargs (e.g. reasoning_effort) into backend_kwargs
        if extra_backend_kwargs:
            config["backend_kwargs"] = {
                **config["backend_kwargs"],
                **extra_backend_kwargs,
            }

        # (log_dir is taken from model_config['log_dir'] only; the old inference
        # from config.cost_tracking_json_path was dropped when cost tracking moved
        # to the opt-in BudgetPool.)

        # Track the exact JSONL path this call's logger writes to, so we convert
        # *this* call's log rather than guessing the latest file in the directory
        # (robust under concurrent invocations sharing a log_dir).
        log_file_path: str | None = None
        if log_dir is not None:
            import os

            os.makedirs(log_dir, exist_ok=True)
            rlm_logger = RLMLogger(log_dir=log_dir)
            config["logger"] = rlm_logger
            log_file_path = rlm_logger.log_file_path

        rlm = RLM(**config)

        root_prompt = (
            root_prompt_override
            if root_prompt_override is not None
            else self._default_root_prompt()
        )
        # rlms types `prompt` as str|dict, but accepts an OpenAI-style message
        # list at runtime (which is what JAZ passes); the stub is just narrow.
        result = rlm.completion(prompt=messages, root_prompt=root_prompt)  # pyright: ignore[reportArgumentType]

        # Trim this call's log to metadata + final iteration (RLM re-logs the
        # full growing prompt each iteration, so earlier lines are redundant
        # prefixes), then convert the trimmed log to readable Markdown.
        if log_file_path is not None:
            self._truncate_log(log_file_path)
            self._convert_log(log_file_path)

        # At depth=0 RLM returns a plain str; at depth>=1 an RLMChatCompletion
        if isinstance(result, str):
            import warnings

            # No usage info is available for a depth=0 string result, so cost
            # accounting and budget enforcement are disabled for this call.
            warnings.warn(
                "RLM returned a plain string (depth=0): no token/cost usage is "
                "available, so cost tracking and budget enforcement are disabled "
                "for this call. Configure max_depth>=1 for cost accounting.",
                stacklevel=2,
            )
            return LLMResponse(
                content=result,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                cost=None,
                raw_response=result,
            )

        # RLMChatCompletion — aggregate usage across all models
        prompt_tokens = 0
        completion_tokens = 0
        for model_usage in result.usage_summary.model_usage_summaries.values():
            prompt_tokens += model_usage.total_input_tokens
            completion_tokens += model_usage.total_output_tokens

        cost = self._compute_cost(result.usage_summary)

        return LLMResponse(
            content=result.response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            raw_response=result,
        )

    @staticmethod
    def _default_root_prompt() -> str:
        """Render the default RLM root prompt.

        The root prompt tells the RLM how to use ``context`` (the agent's
        conversation history) to produce the agent's next action. It lives in
        ``templates/rlm_root_prompt.jinja2`` so it can be tuned without editing
        source; pass ``root_prompt=...`` in ``model_config`` to override it.
        """
        from .template_loader import _jinja_env

        return _jinja_env.get_template("rlm_root_prompt.jinja2").render()

    @staticmethod
    def _truncate_log(jsonl_path: str) -> None:
        """Trim an RLM JSONL log to its first and last lines.

        RLM appends one entry per inner iteration, each re-logging the full
        growing prompt, so the file is highly redundant: the last line already
        holds the most complete snapshot. Keeping the first line (run metadata)
        plus the last line (final iteration) preserves a self-describing record
        at a fraction of the size. Intermediate iterations are intentionally
        discarded (from both the JSONL and the Markdown rendered after this).
        """
        import os

        if not os.path.exists(jsonl_path):
            # Logger never wrote (e.g. completion failed before any log entry).
            return
        with open(jsonl_path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if len(lines) <= 2:
            return  # already minimal (metadata + at most one iteration)
        with open(jsonl_path, "w") as f:
            f.write(lines[0] + "\n")
            f.write(lines[-1] + "\n")

    @staticmethod
    def _convert_log(jsonl_path: str) -> None:
        """Convert a specific RLM JSONL log to a sibling ``.md``.

        Takes the exact path the logger wrote (not "the latest file in the
        dir"), so it converts the correct log even when multiple invocations
        share a log_dir concurrently.
        """
        import os

        if not os.path.exists(jsonl_path):
            # Logger never wrote (e.g. completion failed before any log entry).
            return
        md_path = jsonl_path.rsplit(".", 1)[0] + ".md"

        from .utils.rlm_log_viewer import convert_log

        try:
            md = convert_log(jsonl_path)
            with open(md_path, "w") as f:
                f.write(md)
        except Exception:
            # Don't let viewer errors break the agent, but surface them so log
            # schema drift / malformed entries are debuggable.
            logger.warning(
                "RLMClient: failed to convert RLM log %r to Markdown",
                jsonl_path,
                exc_info=True,
            )

    def get_model(self, model_config: dict[str, object]) -> str:
        # Prefer backend_kwargs.model_name, fall back to top-level model key
        backend_kwargs = model_config.get("backend_kwargs")
        if isinstance(backend_kwargs, dict) and "model_name" in backend_kwargs:
            model = backend_kwargs["model_name"]
        else:
            model = model_config.get("model", "")
        if not isinstance(model, str) or not model:
            raise ValueError(
                "RLMClient requires a model name. Set "
                "model_config['backend_kwargs']['model_name'] or "
                "model_config['model']."
            )
        return model

    def get_model_info(self, model_config: dict[str, object]) -> dict[str, Any]:
        from .providers import get_model_info as _get_model_info

        model_name = self.get_model(model_config)
        info = _get_model_info(model_name) or {}
        # RLM handles context overflow internally via recursive decomposition,
        # so remove max_input_tokens to prevent JAZ from enforcing a context
        # window limit. The aggregated prompt_tokens from RLM sub-calls can
        # far exceed any single model's context window.
        info.pop("max_input_tokens", None)
        return info

    @staticmethod
    def _compute_cost(usage_summary: Any) -> float:
        """Compute total USD cost from RLM UsageSummary using bundled pricing data."""
        total_cost = 0.0
        for model_name, model_usage in usage_summary.model_usage_summaries.items():
            try:
                prompt_cost, completion_cost = cost_per_token(
                    model=model_name,
                    prompt_tokens=model_usage.total_input_tokens,
                    completion_tokens=model_usage.total_output_tokens,
                )
                total_cost += prompt_cost + completion_cost
            except Exception:
                pass  # Model not in pricing DB
        return total_cost
