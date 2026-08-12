"""Logger hooks that write to files or terminal.

These hooks provide logging functionality similar to the old listener system,
but using the new hook architecture.
"""

import logging
import sys
from pathlib import Path

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.provenance import to_wire_messages
from jaz.string_utils import abbrev_repr


class PrintLogger(Hook):
    """Log every event to the terminal, with colors.

    A pure observer: it emits no effects, so it never influences the run.

    Under ``with`` it logs invokes nested inside too. Passed positionally, it logs only that
    invoke's events.

    Args:
        level: Logging level for the emitted records.
        max_field_length: Truncate any rendered field longer than this. ``None`` disables
            truncation.
    """

    def __init__(self, *, level: int = logging.INFO, max_field_length: int = 1000):
        self._max_field_length = max_field_length
        # invoke_ids whose initial prompt has already been logged. The prompt is logged
        # once, at the invoke's first LLMQueryEnter (later queries re-send the growing
        # buffer; re-dumping it every turn would be noise).
        self._logged_prompt_invokes: set[str] = set()
        name = f"jaz.hooks.print_logger.{id(self)}"
        self._logger = logging.Logger(name, level=level)
        self._logger.propagate = False
        # Bind to the host's live stdout captured *at construction* (`sys.stdout`), NOT
        # `sys.__stdout__`. Two reasons:
        #  - Capture-immunity: this is a saved reference distinct from the REPL's per-exec
        #    output capture, so diagnostics never land in an agent's observation buffer —
        #    including the subagent case, where a nested invoke's logger fires while the
        #    parent's exec-scoped capture is active. (Hooks are constructed before the
        #    invoke, so `sys.stdout` here is the host sink, not the capture proxy.)
        #  - Right destination: `sys.stdout` is where the human is actually looking — an
        #    ipykernel `OutStream` in a notebook, pytest's capture object under a test, the
        #    tty in a shell. `sys.__stdout__` is only "the console" in a bare terminal;
        #    empirically it is invisible in Jupyter under `capture_fd_output=False` and
        #    escapes pytest's `capsys`.
        #
        # Capture-immunity assumes construction happens *outside* a REPL exec — the normal
        # `with PrintLogger(): jaz.invoke(...)` case, and nested invokes (the instance is
        # reused via contextvars). The unenforced edge: an agent that constructs a
        # PrintLogger *inside* its own REPL exec binds to the live per-exec capture buffer,
        # reintroducing the leak this avoids. Also a forward seam: a ContextVar-routed
        # `sys.stdout` proxy (#649) that routes by contextvar *at write time* must keep this
        # binding pointed at the host sink (snapshot the underlying stream, or bind before
        # the proxy installs), or a logger write during an exec would route into the capture
        # buffer despite construction happening first.
        original_stdout = sys.stdout
        assert original_stdout is not None
        self._color_enabled = original_stdout.isatty()

        def _color(text, code: str) -> str:
            if not self._color_enabled:
                return str(text)
            return f"\033[{code}m{text}\033[0m"

        self._color = _color

        formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler(original_stdout)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

    def on_any(self, event: Event):
        """Log event to terminal."""
        from jaz.hooks.events import (
            InvokeEnter,
            InvokeExit,
            LLMQueryEnter,
            LLMQueryExit,
            LLMQueryRetry,
            REPLExecEnter,
            REPLExecExit,
        )

        match event:
            case InvokeEnter(
                inputs=inp,
                scope=sc,
                depth=d,
            ):
                # Explicit `**inputs` kwargs and resolved ambient `jaz.scope` are logged as SEPARATE
                # fields (#727); scope is shown only when active, to keep the common line clean.
                scope_frag = (
                    f"scope={self._color(abbrev_repr(dict(sc), self._max_field_length), '34')}, "
                    if sc
                    else ""
                )
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('invoke enter', '32')}: "
                    # The task is an ordinary input now (#538) — it appears in `inputs=`
                    # below, so there is no separate `prompt=` field to log.
                    f"inputs={self._color(abbrev_repr(dict(inp), self._max_field_length), '34')}, "
                    f"{scope_frag}"
                    f"depth={self._color(d, '36')}, "
                    # Governance active for this invoke (incl. this logger), #727. `event.hooks` is
                    # the live active set; serialize each via to_dict() at this edge.
                    f"hooks={self._color(abbrev_repr([h.to_dict() for h in event.hooks], self._max_field_length), '36')}"
                )
                self._logger.info(message)

            case InvokeExit(result=result, depth=d):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('invoke exit', '31')}: "
                    f"result={self._color(abbrev_repr(result, self._max_field_length), '35')}, "
                    f"depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case REPLExecEnter(iteration=i, code=code, depth=d):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('repl iteration', '32')} "
                    f"{self._color(i, '33')} "
                    f"{self._color('input', '32')}: "
                    f"code={self._color(repr(code), '34')}, "
                    f"depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case REPLExecExit(
                iteration=i,
                exec_result=result,
                depth=d,
            ):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('repl iteration', '32')} "
                    f"{self._color(i, '33')} "
                    f"{self._color('output', '32')}: "
                    f"exec_result={self._color(abbrev_repr(result, self._max_field_length), '35')}, "
                    f"depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case LLMQueryEnter(invoke_id=invoke_id, messages=messages, model=model):
                # Emit `[Agent] message:` lines for conversation content — the source of
                # message-content logging now the Message event is gone, and what
                # log_to_markdown reconstructs the transcript from (every role, not just
                # assistant). First query of an invoke: the whole initial prompt. Later
                # queries: only messages[-1] — the observation appended since the previous
                # query. Earlier turns were already logged and the assistant reply is logged
                # at LLMQueryExit, so this logs each message exactly once (without it,
                # observations would never appear — the gap that dropped tool results from
                # log_to_markdown output).
                if invoke_id not in self._logged_prompt_invokes:
                    self._logged_prompt_invokes.add(invoke_id)
                    new_wire = to_wire_messages(messages)
                else:
                    new_wire = to_wire_messages([messages[-1]]) if messages else []
                for wire in new_wire:
                    self._logger.info(
                        f"{self._color('[Agent]', '36')} "
                        f"{self._color('message', '32')}: "
                        f"{self._color(wire, '33')}"
                    )
                message = (
                    f"{self._color('[LLM]', '35')} {self._color('query enter', '32')}: "
                    f"model={self._color(model, '36')}, "
                    f"messages={self._color(len(messages), '33')} messages"
                )
                self._logger.info(message)

            case LLMQueryExit(response=resp, model=model):
                cost = resp.cost
                message = (
                    f"{self._color('[LLM]', '35')} {self._color('query exit', '32')}: "
                    f"model={self._color(model, '36')}, "
                    f"cost={self._color(f'${cost:.4f}' if cost is not None else 'N/A', '33')}"
                )
                self._logger.info(message)
                # Log the assistant reply as a message line — the source of assistant-content
                # logging now that the Message event is gone. Kept in the exact
                # `[Agent] message: {dict}` form that Replay.from_log_and_costs parses
                # back (role=="assistant"); see replay.py / TODO(#712).
                assistant_msg = to_wire_messages(
                    [{"role": "assistant", "content": resp.content}]
                )[0]
                self._logger.info(
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('message', '32')}: "
                    f"{self._color(assistant_msg, '33')}"
                )

            case LLMQueryRetry(
                model=model,
                attempt_number=attempt,
                exception=exc,
                wait_seconds=wait,
            ):
                message = (
                    f"{self._color('[LLM]', '35')} {self._color('retry', '33')}: "
                    f"model={self._color(model, '36')}, "
                    f"attempt={self._color(attempt, '33')}, "
                    f"error={self._color(f'{type(exc).__name__}: {exc}', '31')}, "
                    f"wait={self._color(f'{wait:.1f}s', '33')}"
                )
                self._logger.info(message)

        return []

    def teardown(self, exc: BaseException | None = None) -> None:
        # Release the per-invoke dedup set (grows one invoke_id per invoke). Unlike
        # FileLogger, don't touch the handler — it wraps the host's live stdout, which
        # this logger borrows but does not own.
        self._logged_prompt_invokes.clear()


class FileLogger(Hook):
    """Log every event to a file, without colors.

    A pure observer: it emits no effects, so it never influences the run.

    Under ``with`` it logs invokes nested inside too. Passed positionally, it logs only that
    invoke's events.

    Args:
        file_path: Destination file; opened on activation and closed when the scope exits.
        level: Logging level for the emitted records.
        max_field_length: Truncate any rendered field longer than this. ``None`` disables
            truncation.
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        level: int = logging.INFO,
        max_field_length: int = 1000,
    ):
        self._max_field_length = max_field_length
        # invoke_ids whose initial prompt has already been logged (see PrintLogger).
        self._logged_prompt_invokes: set[str] = set()
        self.file_path = Path(file_path)
        name = f"jaz.hooks.file_logger.{id(self)}"
        self._logger = logging.Logger(name, level=level)
        self._logger.propagate = False

        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(self.file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def on_any(self, event: Event):
        """Log event to file."""
        from jaz.hooks.events import (
            InvokeEnter,
            InvokeExit,
            LLMQueryEnter,
            LLMQueryExit,
            LLMQueryRetry,
            REPLExecEnter,
            REPLExecExit,
        )

        match event:
            case InvokeEnter(
                inputs=inp,
                scope=sc,
                depth=d,
            ):
                # Explicit `**inputs` kwargs and resolved ambient `jaz.scope` are logged as SEPARATE
                # fields (#727); scope is shown only when active, to keep the common line clean.
                scope_frag = (
                    f"scope={abbrev_repr(dict(sc), self._max_field_length)}, "
                    if sc
                    else ""
                )
                message = (
                    # The task is an ordinary input now (#538) — it appears in `inputs=`
                    # below, so there is no separate `prompt=` field to log.
                    f"[Agent] invoke enter: "
                    f"inputs={abbrev_repr(dict(inp), self._max_field_length)}, "
                    f"{scope_frag}"
                    f"depth={d}, "
                    # Governance active for this invoke (incl. this logger), #727. `event.hooks` is
                    # the live active set; serialize each via to_dict() at this edge.
                    f"hooks={abbrev_repr([h.to_dict() for h in event.hooks], self._max_field_length)}"
                )
                self._logger.info(message)

            case InvokeExit(result=result, depth=d):
                message = f"[Agent] invoke exit: result={abbrev_repr(result, self._max_field_length)}, depth={d}"
                self._logger.info(message)

            case REPLExecEnter(iteration=i, code=code, depth=d):
                message = (
                    f"[Agent] repl iteration {i} input: code={repr(code)}, depth={d}"
                )
                self._logger.info(message)

            case REPLExecExit(
                iteration=i,
                exec_result=result,
                depth=d,
            ):
                message = (
                    f"[Agent] repl iteration {i} output: "
                    f"exec_result={abbrev_repr(result, self._max_field_length)}, depth={d}"
                )
                self._logger.info(message)

            case LLMQueryEnter(invoke_id=invoke_id, messages=messages, model=model):
                # First query of an invoke: log the whole initial prompt. Later queries: only
                # messages[-1] — the observation appended since the previous query. Each
                # message is thus logged exactly once (assistant replies at LLMQueryExit),
                # which keeps observations in the `[Agent] message:` stream log_to_markdown
                # reconstructs from. See PrintLogger for the full rationale.
                if invoke_id not in self._logged_prompt_invokes:
                    self._logged_prompt_invokes.add(invoke_id)
                    new_wire = to_wire_messages(messages)
                else:
                    new_wire = to_wire_messages([messages[-1]]) if messages else []
                for wire in new_wire:
                    self._logger.info(f"[Agent] message: {wire}")
                message = (
                    f"[LLM] query enter: "
                    f"model={model}, "
                    f"messages={len(messages)} messages"
                )
                self._logger.info(message)

            case LLMQueryExit(response=resp, model=model):
                cost = resp.cost
                message = f"[LLM] query exit: model={model}, " + (
                    f"cost=${cost:.4f}" if cost is not None else "cost=N/A"
                )
                self._logger.info(message)
                # Log the assistant reply as a message line — the source of assistant-content
                # logging now that the Message event is gone. This exact
                # `[Agent] message: {dict}` line is parsed back by
                # Replay.from_log_and_costs via ast.literal_eval (role=="assistant");
                # a non-literal field would break it. TODO(#712): ast.literal_eval
                # round-tripping a human-readable log line is fragile — a structural fix
                # replaces it with a real re-entry point.
                assistant_msg = to_wire_messages(
                    [{"role": "assistant", "content": resp.content}]
                )[0]
                self._logger.info(f"[Agent] message: {assistant_msg}")

            case LLMQueryRetry(
                model=model,
                attempt_number=attempt,
                exception=exc,
                wait_seconds=wait,
            ):
                message = (
                    f"[LLM] retry: model={model}, "
                    f"attempt={attempt}, "
                    f"error={type(exc).__name__}: {exc}, "
                    f"wait={wait:.1f}s"
                )
                self._logger.info(message)

        return []

    def teardown(self, exc: BaseException | None = None) -> None:
        """Flush and close file handlers."""
        # Release the per-invoke dedup set — it only grows over a logger's lifetime
        # (one invoke_id per invoke), so drop it once logging is done.
        self._logged_prompt_invokes.clear()
        for handler in self._logger.handlers:
            handler.flush()
            handler.close()
