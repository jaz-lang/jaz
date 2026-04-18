from __future__ import annotations

from typing import Any

from .llm_client import LLMClient, LLMResponse
from .providers import cost_per_token


class RLMClient(LLMClient):
    """LLM client that wraps the RLM framework.

    All ``model_config`` keys are passed directly to the ``RLM()`` constructor.

    Usage::

        jaz.configure(
            llm_client="rlm",
            model_config={
                "backend": "openai",
                "backend_kwargs": {"api_key": "..."},
                "max_depth": 2,
            },
        )
    """

    def __init__(self) -> None:
        try:
            import rlm  # pyright: ignore[reportMissingImports]  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "RLMClient requires the 'rlm' package. Install it with: pip install rlm"
            ) from e

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        from rlm import RLM  # pyright: ignore[reportMissingImports]

        config = {**kwargs}
        config["backend_kwargs"] = {"model_name": model}
        rlm = RLM(**config)

        prompt = self._format_messages(messages)
        result = rlm.completion(prompt=prompt)

        # Aggregate usage across all models in usage_summary
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

    def get_model(self, model_config: dict[str, object]) -> str:
        backend_kwargs = model_config["backend_kwargs"]
        assert isinstance(backend_kwargs, dict)
        model = backend_kwargs["model_name"]
        assert isinstance(model, str)
        return model

    def get_model_info(self, model_config: dict[str, object]) -> dict[str, Any]:
        # TODO: Implement this
        raise NotImplementedError("RLM does not support model info")

    @staticmethod
    def _format_messages(messages: list[Any]) -> str:
        """Format OpenAI-style messages into a single role-tagged string."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user") if isinstance(msg, dict) else "user"
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

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
