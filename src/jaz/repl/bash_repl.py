import re
import subprocess
import traceback
import uuid
from collections.abc import Callable
from typing import Any, Self, cast, override

import tree_sitter_bash as ts_bash
from tree_sitter import Language, Parser
from typing_extensions import TypedDict, TypeForm

from jaz.exceptions import BashExitException, BudgetForcingException, CommandSyntaxError
from jaz.library import Library
from jaz.template_loader import _jinja_env

from .base import REPL
from .registry import register_repl
from .types import ErrorResult, ExecResult, ExecuteResult, RaiseResult, ReturnResult


class BashREPLConfig(TypedDict, total=False):
    """Configuration dictionary for BashREPL."""

    multi_statement: bool
    repl_description_template: str


# Initialize tree-sitter bash parser
_BASH_LANGUAGE = Language(ts_bash.language())
_BASH_PARSER = Parser(_BASH_LANGUAGE)


class MultipleStatementsError(SyntaxError):
    """Raised when REPL input contains multiple statements in single-statement mode."""

    pass


def _check_single_statement(src: str) -> None:
    """Check that source contains exactly one bash statement.

    Uses tree-sitter for robust bash parsing. Exactly one non-comment
    statement is required. This allows:
    - Simple commands: `ls -l`
    - Pipelines: `ls | grep foo`
    - AND/OR lists: `mkdir dir && cd dir`
    - Line continuations: `echo hello \\`
                          `world`
    - Here-documents (count as single statement)

    Empty input, comment-only input, and multiple statements are rejected.

    Args:
        src: The source code to validate.

    Raises:
        MultipleStatementsError: If the source does not contain exactly one statement.
    """
    tree = _BASH_PARSER.parse(src.encode())

    # Get all named nodes excluding comments
    statements = [
        child
        for child in tree.root_node.children
        if child.is_named and child.type != "comment"
    ]

    if len(statements) != 1:
        raise MultipleStatementsError(
            f"REPL input must contain exactly 1 statement, "
            f"but found {len(statements)} statements. "
            f"Please split your code into multiple REPL inputs."
        )


@register_repl("bash")
class BashREPL(REPL):
    """REPL implementation that executes bash commands.

    Supports two modes controlled by the `multi_statement` config flag:
    - multi_statement=True (default): Allows multiline bash code
    - multi_statement=False: Enforces single-statement mode using tree-sitter
    """

    # Unified regex patterns: group(1) = preceding code, group(2) = value/error
    RETURN_CMD_RE = re.compile(r"^(.*\n)?RETURN\s*(?:\s(.*))?$", re.DOTALL)
    RAISE_CMD_RE = re.compile(r"^(.*\n)?RAISE\s*(?:\s(.*))?$", re.DOTALL)

    MAGIC_STRING = str(uuid.uuid4())

    # Budget forcing error messages
    BUDGET_FORCING_RETURN_MESSAGE = (
        "Budget forcing active: You cannot finish yet.\n\n"
        "Before returning a final answer, please:\n"
        "- Double-check your reasoning\n"
        "- Verify your solution is correct\n"
        "- Consider edge cases or alternative approaches\n"
        "- Test your answer if possible"
    )

    BUDGET_FORCING_RAISE_MESSAGE = (
        "Budget forcing active: You cannot raise an error yet.\n\n"
        "Before raising an error, please:\n"
        "- Reconsider whether the task is truly impossible\n"
        "- Try alternative approaches\n"
        "- Check if you misunderstood the requirements\n"
        "- Verify any assumptions you made"
    )

    @property
    def description(self) -> str:
        """Return the appropriate description based on configuration."""
        return self.get_description(
            BashREPLConfig(
                multi_statement=self.multi_statement,
                repl_description_template=self.repl_description_template,
            )
        )

    def __init__(
        self,
        repl_exec_timeout: float | None,
        return_type: TypeForm[Any] | None = None,
        return_validator: Callable[[Any], None] | None = None,
        multi_statement: bool = True,
        repl_description_template: str = "bash_repl_description.jinja2",
    ) -> None:
        """Initialize BashREPL with configuration.

        Args:
            repl_exec_timeout: Timeout in seconds for bash command execution.
            return_type: The expected return type for RETURN commands.
            return_validator: Optional callable to validate the return value.
            multi_statement: If True (default), allows multiline bash code.
                If False, enforces single-statement mode using tree-sitter.
            repl_description_template: Jinja2 template name for the REPL description.
        """
        self.return_type = return_type
        self.return_validator = return_validator
        self.multi_statement = multi_statement
        self.repl_exec_timeout = repl_exec_timeout
        self.repl_description_template = repl_description_template

    @classmethod
    def get_description(cls, config: BashREPLConfig | None = None) -> str:
        """Get the REPL description for the given configuration.

        Args:
            config: Configuration dict. Recognized keys:
                - multi_statement: bool (default True)
                - repl_description_template: str (default "bash_repl_description.jinja2")

        Returns:
            The appropriate description string for this configuration.
        """
        config = config or {}
        template_name = config.get(
            "repl_description_template", "bash_repl_description.jinja2"
        )
        template = _jinja_env.get_template(template_name)
        return template.render(
            multi_statement=config.get("multi_statement", True),
        )

    @override
    @classmethod
    def initialize[ReturnT](
        cls: type[Self],
        task: str,
        return_type: TypeForm[ReturnT] | None,
        inputs: dict[str, object],
        libraries: list[Library],
        allowed_imports: list[str],
        repl_exec_timeout: float | None,
        forbidden_names: list[str],
        forbidden_attributes: list[str] | None = None,
        allowed_builtins: dict[str, object] | None = None,
        session_id: str = "",
        config: BashREPLConfig | None = None,
        return_validator: Callable[[ReturnT], None] | None = None,
        repl_input_validator: Callable[[str], None] | None = None,
    ) -> Self:
        """Initialize the Bash REPL.

        Note: task, inputs, allowed_imports, allowed_builtins,
        forbidden_names, and forbidden_attributes are ignored for BashREPL
        as there is no state to maintain - state is in the surrounding bash
        environment.

        Arguments:
            task: Ignored for BashREPL.
            return_type: The expected return type for RETURN commands.
            inputs: Ignored for BashREPL.
            libraries: List of library objects available in the REPL.
            allowed_imports: Ignored for BashREPL.
            allowed_builtins: Ignored for BashREPL.
            session_id: Unique session identifier for this REPL instance.
            config: Configuration dict with optional keys:
                - multi_statement: bool (default True) - whether to allow multiline code.
            repl_exec_timeout: Timeout in seconds for bash command execution.
            forbidden_names: Ignored for BashREPL.
            forbidden_attributes: Ignored for BashREPL.
            return_validator: Optional callable to validate the return value.
        Returns:
            The initialized REPL instance.
        """
        return cls(
            repl_exec_timeout=repl_exec_timeout,
            return_type=return_type,
            return_validator=return_validator,
            **(config or {}),
        )

    @override
    def add_inputs(self, inputs: dict[str, object]) -> None:
        """No-op for BashREPL as it doesn't support code objects.

        BashREPL has no persistent Python state, so arbitrary Python objects
        cannot be added. State is managed through shell environment variables
        and the working directory.

        Args:
            inputs: Ignored
        """
        if inputs:
            # Log a warning if someone tries to add inputs to BashREPL
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"BashREPL.add_inputs() called with {len(inputs)} inputs, "
                "but BashREPL does not support adding arbitrary Python objects"
            )

    @staticmethod
    def _format_error(exception: BaseException, stdout: str = "") -> tuple[str, str]:
        """Format an exception for ERROR results.

        Arguments:
            exception: The exception to format.
            stdout: Any stdout output that occurred before the exception.

        Returns:
            Tuple of (formatted_output, error_summary) where:
            - formatted_output: Full output including stdout and traceback
            - error_summary: Short description of the error for the LLM
        """
        tb_str = "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )
        error_summary = f"{type(exception).__name__}: {str(exception)}"
        formatted_output = stdout + tb_str
        return (formatted_output, error_summary)

    def _run_bash(self, cmd: str, timeout: float | None) -> tuple[str, int]:
        """Run a bash command and return output and return code.

        Arguments:
            cmd: The bash command to execute.

        Returns:
            Tuple of (output, returncode)

        Raises:
            Exception: If subprocess.run fails
        """
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return (result.stdout, result.returncode)

    def _make_error_if_not_finish_result(
        self,
        error_if_not_finish: Exception,
        raise_if_not_finish: bool,
    ) -> ErrorResult | RaiseResult:
        """Create an error result when the command should finish but doesn't."""
        if raise_if_not_finish:
            return RaiseResult(
                output="",
                exception=error_if_not_finish,
            )
        formatted_output, error_summary = self._format_error(error_if_not_finish)
        return ErrorResult(
            output=formatted_output,
            error_summary=error_summary,
            exception=error_if_not_finish,
        )

    def _handle_return_command(
        self,
        return_match: re.Match,
        timeout: float | None,
    ) -> ReturnResult[Any] | ErrorResult:
        # TODO: Fix static types
        """Handle the RETURN command."""
        exec_src = return_match.group(1)
        rv_src = return_match.group(2)

        # In single-statement mode, RETURN must be standalone (no preceding code)
        if exec_src and not self.multi_statement:
            error = CommandSyntaxError(
                "Incorrect usage of `RETURN` command. "
                "`RETURN <value>` must be the only content in the REPL input."
            )
            formatted_output, error_summary = self._format_error(error)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )

        if not rv_src:
            return_value_str = ""
        else:
            # Build a script that executes everything together in one subprocess
            # This preserves state (variables, functions, etc.) across commands
            if exec_src is not None and exec_src.strip():
                # Execute preceding code to preserve state (variables, functions)
                # Use a temp file to capture and check for output violations
                script = f"""_tmpfile=$(mktemp)
{{
{exec_src}
}} > "$_tmpfile" 2>&1
if [ -s "$_tmpfile" ]; then
    cat "$_tmpfile"
    rm -f "$_tmpfile"
    echo '{self.MAGIC_STRING}'
    exit 1
fi
rm -f "$_tmpfile"
echo {rv_src}"""
            else:
                script = f"echo {rv_src}"

            try:
                output, returncode = self._run_bash(script, timeout)
            except Exception as e:  # e.g. utf-8 encoding/decoding error
                formatted_output, error_summary = self._format_error(e)
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=e,
                )

            # Check if this was due to output from preceding code
            if output.endswith(self.MAGIC_STRING + "\n"):
                error_summary = (
                    "Your `RETURN` command was not executed because your code produced "
                    "printed output. Please review the output below. If you are "
                    "confident and wish to finish, re-run only the RETURN command.\n"
                    "⚠️ You cannot continue working in any way on this task after "
                    "returning, so make sure you are confident with your work before "
                    "returning."
                )
                return ErrorResult(
                    # Remove magic string from output
                    output=output[: -len(self.MAGIC_STRING) - 1],
                    error_summary=error_summary,
                )
            elif returncode != 0:
                error_summary = f"RETURN command exited with return code {returncode}"
                return ErrorResult(
                    output=output,
                    error_summary=error_summary,
                )

            # TODO: Detect trailing commands after RETURN command
            # Right now the stdout/stderr from those will just get
            # included as part of return value.

            return_value_str = output

        # Convert return value to expected type
        return_value_str = return_value_str.rstrip("\n")
        if self.return_type is None:
            if return_value_str.lower().strip() not in ["", "none"]:
                error_summary = (
                    f"Expected to return nothing. Instead got {return_value_str!r}."
                )
                return ErrorResult(
                    output="",
                    error_summary=error_summary,
                )
            return ReturnResult(
                output="",
                return_value=None,
            )

        else:
            try:
                # TODO: Also apply check_type here for more soundness
                assert callable(self.return_type)
                # TODO: Fix static types
                return_value = cast(Any, self.return_type(return_value_str))
            except Exception as e:
                return_type_name = (
                    self.return_type.__name__
                    if isinstance(self.return_type, type)
                    else str(self.return_type)
                )
                # TODO: Include traceback in output?
                error_summary = (
                    f"Failed to convert return value {return_value_str!r} to type "
                    f"{return_type_name}. {e.__class__.__name__}: {e}\n"
                    "Try to use the Python REPL instead of the Bash REPL to return "
                    "objects of this type."
                )
                return ErrorResult(
                    output="",
                    error_summary=error_summary,
                )

            # Validate return value if validator is provided
            if self.return_validator is not None:
                try:
                    # TODO: Should stdout be captured here?
                    self.run_return_validator(return_value, repl_history=None)
                except Exception as e:
                    formatted_output, error_summary = self._format_error(e)
                    return ErrorResult(
                        output=formatted_output,
                        error_summary=error_summary,
                        exception=e,
                    )

            return ReturnResult(
                output="",
                return_value=return_value,
            )

    def _handle_raise_command(
        self, raise_match: re.Match, timeout: float | None
    ) -> RaiseResult | ErrorResult:
        """Handle the RAISE command."""
        exec_src = raise_match.group(1)
        err_src = raise_match.group(2)

        # In single-statement mode, RAISE must be standalone (no preceding code)
        if exec_src and not self.multi_statement:
            error = CommandSyntaxError(
                "Incorrect usage of `RAISE` command. "
                "`RAISE <error>` must be the only content in the REPL input."
            )
            formatted_output, error_summary = self._format_error(error)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )

        if err_src is None or not err_src.strip():
            error_summary = "Command `RAISE` must be called with an error message."
            return ErrorResult(
                output="",
                error_summary=error_summary,
            )

        # Build a script that executes everything together in one subprocess
        # This preserves state (variables, functions, etc.) across commands
        if exec_src:
            # Execute preceding code to preserve state (variables, functions)
            # Use a temp file to capture and check for output violations
            script = f"""_tmpfile=$(mktemp)
{{
{exec_src}
}} > "$_tmpfile" 2>&1
if [ -s "$_tmpfile" ]; then
    cat "$_tmpfile"
    rm -f "$_tmpfile"
    echo '{self.MAGIC_STRING}'
    exit 1
fi
rm -f "$_tmpfile"
echo {err_src}"""
        else:
            script = f"echo {err_src}"

        try:
            output, returncode = self._run_bash(script, timeout)
        except Exception as e:  # e.g. utf-8 encoding/decoding error
            formatted_output, error_summary = self._format_error(e)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=e,
            )

        if returncode != 0:
            # Check if this was due to output from preceding code
            if output.endswith(self.MAGIC_STRING + "\n"):
                error_summary = (
                    "Your `RAISE` command was not executed because your code produced "
                    "printed output. Please review the output below. If you are "
                    "confident and wish to finish, re-run only the RAISE command.\n"
                    "⚠️ You cannot continue working in any way on this task afterwards, "
                    "so make sure you are confident with your work before finishing."
                )
                return ErrorResult(
                    output=output[: -len(self.MAGIC_STRING) - 1],  # Remove magic string
                    error_summary=error_summary,
                )
            else:
                error_summary = f"RAISE command exited with return code {returncode}"
                return ErrorResult(
                    output=output,
                    error_summary=error_summary,
                )

        # TODO: Detect trailing commands after RAISE command
        # Right now the stdout/stderr from those will just get
        # included as part of the error message.

        error_message = output.rstrip("\n")

        return RaiseResult(
            output="",
            exception=BashExitException(error_message),
        )

    def _execute_regular_bash(
        self, src: str, timeout: float | None
    ) -> ExecuteResult | ErrorResult:
        """Execute regular bash code (no special commands)."""
        # In single-statement mode, validate that src contains exactly one statement
        if not self.multi_statement:
            try:
                _check_single_statement(src)
            except MultipleStatementsError as e:
                formatted_output, error_summary = self._format_error(e)
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=e,
                )

        try:
            output, returncode = self._run_bash(src, timeout)
        except Exception as e:  # e.g. utf-8 encoding/decoding error
            formatted_output, error_summary = self._format_error(e)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=e,
            )

        # Return EXECUTE if success (return code 0), ERROR otherwise
        if returncode == 0:
            return ExecuteResult(
                output=output,
            )
        else:
            return ErrorResult(
                output=output,
                error_summary=f"Command exited with return code {returncode}",
            )

    def _apply_budget_forcing[ReturnT](
        self,
        exec_result: ExecResult[ReturnT],
    ) -> ExecResult[ReturnT]:
        """Apply budget forcing - refuse RETURN/RAISE and return ErrorResult."""
        if isinstance(exec_result, ReturnResult):
            error = BudgetForcingException(self.BUDGET_FORCING_RETURN_MESSAGE)
            formatted_output, error_summary = self._format_error(error)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )
        elif isinstance(exec_result, RaiseResult):
            error = BudgetForcingException(self.BUDGET_FORCING_RAISE_MESSAGE)
            formatted_output, error_summary = self._format_error(error)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )
        return exec_result

    def _exec(
        self,
        src: str,
        error_if_not_finish: Exception | None,
        raise_if_not_finish: bool,
        timeout: float | None,
    ) -> ExecResult[Any]:
        # TODO: Fix static types
        """Internal execution logic for bash commands."""
        return_match = self.RETURN_CMD_RE.match(src)
        raise_match = self.RAISE_CMD_RE.match(src)
        assert (return_match is not None) + (raise_match is not None) <= 1

        # Check if we need to finish but don't have RETURN or RAISE
        if error_if_not_finish is not None and (
            return_match is None and raise_match is None
        ):
            return self._make_error_if_not_finish_result(
                error_if_not_finish, raise_if_not_finish
            )

        if return_match:
            return self._handle_return_command(return_match, timeout)

        if raise_match:
            return self._handle_raise_command(raise_match, timeout)

        # Normal bash command execution (no RETURN or RAISE)
        return self._execute_regular_bash(src, timeout)

    @override
    def exec(
        self,
        src: str,
        input_id: str,
        error_if_not_finish: Exception | None = None,
        raise_if_not_finish: bool = False,
        budget_forcing_active: bool = False,
        exec_timeout_override: float | None = None,
    ) -> ExecResult[Any]:
        # TODO: Fix static types
        """Executes a bash command.

        Note: BashREPL maintains no state - state is in the surrounding bash
        environment. Every command is executed in a new shell subprocess.

        ``return_type`` and ``return_validator`` are read from instance attributes
        set during initialization.

        Arguments:
            src: The bash command to execute.
            input_id: The unique identifier for the REPL input.
            error_if_not_finish: Error to return if command doesn't finish with
                RETURN/RAISE.
            raise_if_not_finish: Whether to raise error if command doesn't finish.
            budget_forcing_active: If True, RETURN and RAISE commands are refused
                and an ErrorResult is returned instead, encouraging the agent to
                do more reasoning before finishing.

        Returns:
            exec_result: A tuple containing:
                result_category: EXECUTE, ERROR, RETURN, or RAISE.
                output: The combined stdout/stderr output of the command.
                Additional fields depending on result category.
        """
        timeout = (
            exec_timeout_override
            if exec_timeout_override is not None
            else self.repl_exec_timeout
        )
        exec_result = self._exec(
            src,
            error_if_not_finish,
            raise_if_not_finish,
            timeout,
        )

        # Budget forcing: refuse RETURN/RAISE if active
        if budget_forcing_active:
            exec_result = self._apply_budget_forcing(exec_result)

        # TODO: Better way of combining ErrorResult and error_if_not_finish
        if raise_if_not_finish and not isinstance(
            exec_result, ReturnResult | RaiseResult
        ):
            assert error_if_not_finish is not None
            exec_result = RaiseResult(
                output="",
                exception=error_if_not_finish,
            )

        return exec_result
