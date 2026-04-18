"""Logger hooks that write to files or terminal.

These hooks provide logging functionality similar to the old listener system,
but using the new hook architecture.
"""

import logging
import sys
from pathlib import Path

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.string_utils import abbrev_repr


class PrintLogger(Hook):
    """Hook that logs events to the terminal with colors."""

    def __init__(self, *, level: int = logging.INFO, max_field_length: int = 1000):
        self._max_field_length = max_field_length
        name = f"jaz.hooks.print_logger.{id(self)}"
        self._logger = logging.Logger(name, level=level)
        self._logger.propagate = False
        # Use __stdout__ to get the original stdout, not the potentially redirected one
        # This ensures that PrintLogger output from nested jaz.invoke calls goes to
        # the real terminal, not into the parent REPL's output buffer
        original_stdout = sys.__stdout__
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

    def on_event(self, event: Event):
        """Log event to terminal."""
        from jaz.hooks.events import (
            BudgetUpdate,
            InvokeEnter,
            InvokeExit,
            LLMQueryEnter,
            LLMQueryExit,
            LLMRetry,
            Message,
            ReplExecutionEnter,
            ReplExecutionExit,
        )

        match event:
            case InvokeEnter(
                prompt=p,
                return_type=rt,
                inputs=inp,
                max_iterations=mi,
                cur_recursion_depth=d,
            ):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('invoke enter', '32')}: "
                    f"prompt={self._color(abbrev_repr(p, self._max_field_length), '33')}, "
                    f"return_type={self._color(rt, '35')}, "
                    f"inputs={self._color(abbrev_repr(inp, self._max_field_length), '34')}, "
                    f"max_iterations={self._color(mi, '36')}, "
                    f"cur_recursion_depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case InvokeExit(result=result, cur_recursion_depth=d):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('invoke exit', '31')}: "
                    f"result={self._color(abbrev_repr(result, self._max_field_length), '35')}, "
                    f"cur_recursion_depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case ReplExecutionEnter(
                iteration=i, max_iterations=mi, code=code, cur_recursion_depth=d
            ):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('repl iteration', '32')} "
                    f"{self._color(i, '33')}/{self._color(mi, '33')} "
                    f"{self._color('input', '32')}: "
                    f"code={self._color(repr(code), '34')}, "
                    f"cur_recursion_depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case ReplExecutionExit(
                iteration=i,
                max_iterations=mi,
                exec_result=result,
                cur_recursion_depth=d,
            ):
                message = (
                    f"{self._color('[Agent]', '36')} "
                    f"{self._color('repl iteration', '32')} "
                    f"{self._color(i, '33')}/{self._color(mi, '33')} "
                    f"{self._color('output', '32')}: "
                    f"exec_result={self._color(abbrev_repr(result, self._max_field_length), '35')}, "
                    f"cur_recursion_depth={self._color(d, '36')}"
                )
                self._logger.info(message)

            case LLMQueryEnter(messages=messages, model=model):
                message = (
                    f"{self._color('[LLM]', '35')} {self._color('query enter', '32')}: "
                    f"model={self._color(model, '36')}, "
                    f"messages={self._color(len(messages), '33')} messages"
                )
                self._logger.info(message)

            case LLMQueryExit(response_content=_, model=model, cost=cost):
                message = (
                    f"{self._color('[LLM]', '35')} {self._color('query exit', '32')}: "
                    f"model={self._color(model, '36')}, "
                    f"cost={self._color(f'${cost:.4f}' if cost is not None else 'N/A', '33')}"
                )
                self._logger.info(message)

            case LLMRetry(
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

            case Message(message=msg):
                log_message = (
                    f"{self._color('[Agent]', '36')} {self._color('message', '32')}: "
                    f"{self._color(msg, '33')}"
                )
                self._logger.info(log_message)

            case BudgetUpdate(budget_status=status, cur_recursion_depth=d):
                depth_str = ""
                if d is not None:
                    depth_str = (
                        f" {self._color('(recursion depth', '90')} "
                        f"{self._color(d, '36')}{self._color(')', '90')}"
                    )
                message = (
                    f"{self._color('[Budget', '35')}{depth_str}{self._color(']', '35')} "
                    f"{self._color(status, '33')}"
                )
                self._logger.info(message)

        return []


class FileLogger(Hook):
    """Hook that logs events to a file without colors.

    Can be used as a context manager.
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        level: int = logging.INFO,
        max_field_length: int = 1000,
    ):
        self._max_field_length = max_field_length
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

    def on_event(self, event: Event):
        """Log event to file."""
        from jaz.hooks.events import (
            BudgetUpdate,
            InvokeEnter,
            InvokeExit,
            LLMQueryEnter,
            LLMQueryExit,
            LLMRetry,
            Message,
            ReplExecutionEnter,
            ReplExecutionExit,
        )

        match event:
            case InvokeEnter(
                prompt=p,
                return_type=rt,
                inputs=inp,
                max_iterations=mi,
                cur_recursion_depth=d,
            ):
                message = (
                    f"[Agent] invoke enter: "
                    f"prompt={abbrev_repr(p, self._max_field_length)}, "
                    f"return_type={rt}, "
                    f"inputs={abbrev_repr(inp, self._max_field_length)}, "
                    f"max_iterations={mi}, "
                    f"cur_recursion_depth={d}"
                )
                self._logger.info(message)

            case InvokeExit(result=result, cur_recursion_depth=d):
                message = f"[Agent] invoke exit: result={abbrev_repr(result, self._max_field_length)}, cur_recursion_depth={d}"
                self._logger.info(message)

            case ReplExecutionEnter(
                iteration=i, max_iterations=mi, code=code, cur_recursion_depth=d
            ):
                message = (
                    f"[Agent] repl iteration {i}/{mi} input: "
                    f"code={repr(code)}, cur_recursion_depth={d}"
                )
                self._logger.info(message)

            case ReplExecutionExit(
                iteration=i,
                max_iterations=mi,
                exec_result=result,
                cur_recursion_depth=d,
            ):
                message = (
                    f"[Agent] repl iteration {i}/{mi} output: "
                    f"exec_result={abbrev_repr(result, self._max_field_length)}, cur_recursion_depth={d}"
                )
                self._logger.info(message)

            case LLMQueryEnter(messages=messages, model=model):
                message = (
                    f"[LLM] query enter: "
                    f"model={model}, "
                    f"messages={len(messages)} messages"
                )
                self._logger.info(message)

            case LLMQueryExit(response_content=_, model=model, cost=cost):
                message = f"[LLM] query exit: model={model}, " + (
                    f"cost=${cost:.4f}" if cost is not None else "cost=N/A"
                )
                self._logger.info(message)

            case LLMRetry(
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

            case Message(message=msg):
                log_message = f"[Agent] message: {msg}"
                self._logger.info(log_message)

            case BudgetUpdate(budget_status=status, cur_recursion_depth=d):
                depth_str = ""
                if d is not None:
                    depth_str = f" (recursion depth {d})"
                message = f"[Budget{depth_str}] {status}"
                self._logger.info(message)

        return []

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context and cleanup file handlers."""
        # Call parent to reset hook context
        result = super().__exit__(exc_type, exc_value, traceback)

        # Flush and close handlers
        for handler in self._logger.handlers:
            handler.flush()
            handler.close()

        return result
