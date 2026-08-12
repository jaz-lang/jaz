"""Hook that emits ATIF v1.7 (Agent Trajectory Interchange Format) traces.

Produces Harbor-compatible trajectory JSON directly from JAZ execution events,
without an intermediate conversion step.

Usage:
    with ATIFTrace(output_path="./trace.atif.json"):
        result = jaz.invoke(jaz.ReturnType(str), task="Do something")

    # Creates: ./trace.atif.json validated by harbor.utils.trajectory_validator

See:
    https://www.harborframework.com/docs/agents/trajectory-format
    https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jaz.hooks.dispatcher import Hook
from jaz.hooks.events.invoke import InvokeEnter, InvokeExit
from jaz.hooks.events.llm_query import LLMQueryEnter, LLMQueryExit
from jaz.repl.types import Raise, Return

SCHEMA_VERSION = "ATIF-v1.7"


def _serialize_namespace(namespace: dict[str, Any]) -> dict[str, Any]:
    """Serialize a name→value namespace (inputs or scope) to the ATIF extra form.

    Each value becomes ``{"type": <class name>, "repr_prefix_10000": <repr, truncated>}``.
    """
    return {
        k: {"type": type(v).__name__, "repr_prefix_10000": repr(v)[:10000]}
        for k, v in namespace.items()
    }


@dataclass
class _InvokeState:
    """Tracks per-invoke state for ATIF step building."""

    invoke_id: str
    parent_invoke_id: str | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    step_counter: int = 0
    model_name: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost: float = 0.0
    total_repl_iterations: int = 0
    current_iteration: int | None = None
    _last_message_count: int = 0
    children: list[_InvokeState] = field(default_factory=list)
    parent_repl_iteration: int | None = None
    depth: int | None = None
    # Metadata captured from InvokeEnter/Exit for ATIF extra fields. There is no separate
    # `prompt` field anymore (#538): `task` is an ordinary input, captured in `inputs` below.
    # Explicit `**inputs` kwargs and resolved ambient `jaz.scope`, kept as SEPARATE provenance
    # channels (#727) — emitted as distinct ``extra.inputs`` / ``extra.scope`` blocks.
    inputs: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    # Serialized ``to_dict()`` dicts of the hooks active for this invoke (baseline ‖ propagating ‖
    # local), captured at receipt from the live ``event.hooks`` (#727).
    hooks: tuple[dict, ...] = ()
    result_type: str | None = None
    result_value: Any = None
    result_exception: str | None = None
    result_exception_type: str | None = None

    def next_step_id(self) -> int:
        self.step_counter += 1
        return self.step_counter


class ATIFTrace(Hook):
    """Write the run as an ATIF v1.7 trajectory JSON file.

    A pure observer: it emits no effects, so it never influences the run.

    Under ``with`` the trace covers invokes nested inside. Passed positionally, only that
    invoke is traced and its sub-invokes are absent from the output.

    Args:
        output_path: Path to write the ATIF JSON output.
        indent: JSON indentation level (default: 2).
    """

    # Builds ATIF steps directly as events fire, unlike ``ConversationHistory`` +
    # ``trace_to_atif``, which converts an intermediate JSON.

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        indent: int | None = 2,
    ):
        self.output_path = Path(output_path) if output_path else None
        self.indent = indent
        self._invokes: dict[str, _InvokeState] = {}
        self._root_invokes: list[_InvokeState] = []

    def on_invoke_enter(self, event: InvokeEnter) -> list:
        state = _InvokeState(
            invoke_id=event.invoke_id,
            parent_invoke_id=event.parent_invoke_id,
            parent_repl_iteration=event.parent_repl_iteration,
            depth=event.depth,
            # Explicit kwargs and resolved scope kept separate (#727) — see the fields above.
            # SHALLOW snapshot: dict() pins the key set / top-level bindings at receipt, but nested
            # MUTABLE values are shared with the live namespace, so a value the agent mutates during
            # execution is reflected in the trace serialized later (below, at InvokeExit). This is
            # the same deep/shallow ceiling as event.inputs itself — pinning nested values would
            # need eager deep serialization here, which is costly and can fail on non-copyable
            # inputs, so it is a documented limitation (#831), not a bug.
            inputs=dict(event.inputs),
            scope=dict(event.scope),
            # Serialize the live active-hook set to dicts here, at receipt, so the trace pins the
            # governance active AT InvokeEnter (not whatever a live peer mutates to later) (#727).
            hooks=tuple(h.to_dict() for h in event.hooks),
        )
        self._invokes[event.invoke_id] = state
        if event.parent_invoke_id and event.parent_invoke_id in self._invokes:
            self._invokes[event.parent_invoke_id].children.append(state)
        else:
            self._root_invokes.append(state)
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list:
        state = self._invokes.get(event.invoke_id)
        if not state:
            return []

        # Track the iteration boundary here: #481 removed the REPLIteration*
        # events, and each iteration issues exactly one LLMQuery, so LLMQueryEnter
        # is the per-iteration boundary. current_iteration tags the steps emitted
        # this turn; total_repl_iterations feeds total_steps.
        state.current_iteration = event.iteration
        state.total_repl_iterations += 1

        # Emit new messages added since the last iteration.
        # - First iteration: system + user (task prompt)
        # - Subsequent iterations: user (the REPL output exactly as the agent sees it — since #928
        #   an observation carries no framing at all, so the message content *is* the output)
        # - Skip assistant messages (duplicates of previous LLMQueryExit)
        new_messages = event.messages[state._last_message_count :]
        state._last_message_count = len(event.messages)

        # Timestamp for system/user steps: captured here at LLMQueryEnter
        # (just before the LLM API call). Agent steps use LLMQueryExit
        # end_time instead (when the LLM response arrives).
        now = datetime.now(UTC).isoformat()
        for msg in new_messages:
            role = (
                msg.get("role", "user")
                if isinstance(msg, dict)
                else getattr(msg, "role", "user")
            )
            if role == "assistant":
                continue  # Already emitted via LLMQueryExit
            assert role in ("system", "user"), (
                f"Unexpected role {role!r} in LLMQueryEnter messages "
                f"(expected 'system' or 'user' after filtering 'assistant')"
            )
            content = (
                msg.get("content", "")
                if isinstance(msg, dict)
                else getattr(msg, "content", "")
            )
            step_data: dict[str, Any] = {
                "step_id": state.next_step_id(),
                "source": role,  # "system" or "user"
                "message": content,
                "timestamp": now,
                "llm_call_count": 0,
            }
            if state.current_iteration is not None:
                step_data["extra"] = {"iteration": state.current_iteration}
            state.steps.append(step_data)
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list:
        response = event.response
        response_content = response.content
        state = self._invokes.get(event.invoke_id)
        if not state or not response_content:
            return []

        if event.model:
            state.model_name = event.model

        prompt_tokens = response.prompt_tokens or 0
        completion_tokens = response.completion_tokens or 0
        # Keep the raw value (None = provider reported no caching) distinct from
        # the summing value (None coerced to 0).
        raw_cached_tokens = response.cached_tokens
        cached_tokens = raw_cached_tokens or 0
        cost = response.cost or 0.0

        state.total_prompt_tokens += prompt_tokens
        state.total_completion_tokens += completion_tokens
        state.total_cached_tokens += cached_tokens
        state.total_cost += cost

        # Use end_time from the event if available, else now
        end_time = getattr(event, "end_time", None)
        ts = (
            end_time.isoformat()
            if end_time is not None
            else datetime.now(UTC).isoformat()
        )

        metrics: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
        }
        # Write an explicit 0 when the model reports caching but had no hit (raw
        # == 0); omit only when caching is unreported (raw is None). Matches the
        # convention in trace_to_atif and conversation_history.
        if raw_cached_tokens is not None:
            metrics["cached_tokens"] = raw_cached_tokens
        if response.extra:
            metrics["extra"] = dict(response.extra)
        step: dict[str, Any] = {
            "step_id": state.next_step_id(),
            "source": "agent",
            "message": response_content,
            "timestamp": ts,
            "llm_call_count": 1,
            "metrics": metrics,
        }
        if state.current_iteration is not None:
            step["extra"] = {"iteration": state.current_iteration}
        state.steps.append(step)
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list:
        state = self._invokes.get(event.invoke_id)
        result = event.result
        if state and result is not None:
            state.result_type = type(result).__name__
            # InvokeExit yields either Return (terminal RETURN) or Raise (terminal RAISE) — both
            # terminal, so neither carries ``output`` (#903); capture the value / exception instead.
            match result:
                case Return(return_value=rv) if rv is not None:
                    # Always uses repr (truncated) for safety/bounded size.
                    # The trace_to_atif converter instead preserves the
                    # structured JSON from ConversationHistory.
                    state.result_value = {
                        "type": type(rv).__name__,
                        "repr_prefix_10000": repr(rv)[:10000],
                    }
                case Raise(exception=exc):
                    state.result_exception = str(exc)
                    state.result_exception_type = type(exc).__name__
        return []

    def _build_atif(self, state: _InvokeState) -> dict[str, Any]:
        """Build an ATIF document from an invoke state."""
        atif: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "agent": {
                "name": "jaz",
                "version": __import__("jaz").__version__,
            },
            "steps": state.steps,
            "final_metrics": {
                "total_prompt_tokens": state.total_prompt_tokens,
                "total_completion_tokens": state.total_completion_tokens,
                "total_cached_tokens": state.total_cached_tokens,
                "total_cost_usd": round(state.total_cost, 6),
                "total_steps": state.total_repl_iterations,
            },
            "notes": (
                "total_steps is the number of REPL iterations (think-act-observe cycles), "
                "not len(steps). Each REPL iteration corresponds to one LLM call."
            ),
        }

        if state.model_name:
            atif["agent"]["model_name"] = state.model_name
        if state.invoke_id:
            atif["trajectory_id"] = state.invoke_id

        # Extra metadata not in core ATIF schema but needed for full coverage. The former
        # `extra["prompt"]` is gone (#538); the prompt lives in `extra["inputs"]` now.
        extra: dict[str, Any] = {}
        if state.depth is not None:
            extra["depth"] = state.depth
        if state.parent_repl_iteration is not None:
            # Which REPL iteration of the parent invoke spawned this subagent.
            extra["parent_repl_iteration"] = state.parent_repl_iteration
        if state.inputs:
            # Serialize inputs: try JSON-serializable value, fall back to repr
            # TODO: revisit — should we use the serialization shown to the agent?
            extra["inputs"] = _serialize_namespace(state.inputs)
        if state.scope:
            # Resolved ambient jaz.scope, logged as its own provenance channel (#727).
            extra["scope"] = _serialize_namespace(state.scope)
        if state.hooks:
            # Governance active for this invoke: the resolved active hooks' serialized to_dict() dicts (#727).
            extra["hooks"] = list(state.hooks)
        if state.result_type:
            extra["result_type"] = state.result_type
        if state.result_value is not None:
            extra["result_value"] = state.result_value
        if state.result_exception is not None:
            extra["result_exception"] = state.result_exception
        if state.result_exception_type is not None:
            extra["result_exception_type"] = state.result_exception_type
        if extra:
            atif["extra"] = extra

        # Nested invokes: store in top-level subagent_trajectories[]
        if state.children:
            atif["subagent_trajectories"] = [
                self._build_atif(child) for child in state.children
            ]

        return atif

    def get_trajectories(self) -> list[dict[str, Any]]:
        """Get all root-level ATIF trajectories."""
        return [self._build_atif(state) for state in self._root_invokes]

    def _write_output(self) -> None:
        """Write ATIF JSON to the output file."""
        if not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        trajectories = self.get_trajectories()
        # Single root → write as object; multiple → write as array
        data = trajectories[0] if len(trajectories) == 1 else trajectories
        self.output_path.write_text(json.dumps(data, indent=self.indent))

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context and write output file."""
        self._write_output()
        if hasattr(self, "_token"):
            return super().__exit__(exc_type, exc_value, traceback)

    def reset(self) -> None:
        """Clear state for reuse."""
        self._invokes.clear()
        self._root_invokes.clear()


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``). This hook is absent from every ``__all__``,
#: so that deep-path import is the only way it was ever reachable.
ATIFTraceHook = ATIFTrace
