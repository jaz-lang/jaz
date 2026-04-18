# TODO: Eventually deprecate and replace with NativeReturnPythonREPL

import ast
import re
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from types import ModuleType
from typing import Any, Self, TypeGuard, override

from beartype.door import is_bearable
from typing_extensions import TypedDict, TypeForm

from jaz.exceptions import (
    BudgetForcingException,
    CommandSyntaxError,
    CommandTypeError,
    _JazInternalError,
)
from jaz.library import Library
from jaz.template_loader import _jinja_env

from ._timeout_utils import eval_with_timeout, exec_with_timeout
from .base import REPL
from .compiler import (
    check_json_literal,
    secure_compile,
    validate_builtin_type_annotation,
)
from .inspect_object import inspect_object
from .permissions import ReplPermissionPolicy, build_allowed_builtins
from .registry import register_repl
from .types import ErrorResult, ExecResult, ExecuteResult, RaiseResult, ReturnResult


class PythonREPLConfig(TypedDict, total=False):
    """Configuration dictionary for PythonREPL."""

    multi_statement: bool
    allow_raise: bool
    allow_tools_command: bool
    allow_inspect_commands: bool
    repl_history: bool  # True (default): +RH; False: -RH
    expose_inputs_in_repl: bool  # True (default): +IN; False: -IN
    repl_description_template: str
    restrict_return_value: bool  # False (default): +RET; True: -RET
    restrict_exec: bool  # False (default): +EXEC; True: tool-call-only exec-mode code
    restrict_invoke_args: bool  # False (default): +R; True: 0R
    allow_eval_exec: bool  # False (default): deny eval()/exec()
    allow_file_writes: bool  # False (default): deny write-capable open() modes
    allowed_file_roots: list[str] | None  # None (default): no path root restrictions
    log_file_access: bool  # False (default): no file access logging
    # for -R set max_repl_recursion to 1
    check_output_on_exit: (
        bool  # True (default): block RETURN/RAISE if output was printed
    )
    max_ast_nodes: int | None

    # TODO: Implement the RAISE counterpart to restrict_return_value, or have the flag apply to both at the same time


def _check_type[ReturnT](
    return_value: object,
    return_type: TypeForm[ReturnT] | None,
    # TODO: Remove `| None` once pyright gets better
) -> TypeGuard[ReturnT]:
    """Check if the return value matches the expected type annotation."""
    return is_bearable(return_value, return_type)  # type: ignore


class MultipleStatementsError(SyntaxError):
    """Raised when REPL input contains multiple statements."""

    pass


class ReactToolCallError(SyntaxError):
    """Raised when REPL input is not a valid tool call with constant args."""

    pass


@dataclass(frozen=True)
class ReplHistoryInit:
    """Initial REPL state stored at __repl_history__[0]."""

    task: str
    inputs: dict[str, object]


@dataclass(frozen=True)
class ReplHistoryEntry:
    """REPL history entry for iteration N >= 1."""

    repl_input: str
    repl_output: str
    error: str | None


def _make_print(output: StringIO):
    """Create a print function that writes to the given StringIO."""

    def custom_print(
        *args: object,
        sep: str = " ",
        end: str = "\n",
        file: Any = None,
        flush: bool = False,
    ) -> None:
        if file is None:
            file = output
        print(*args, sep=sep, end=end, file=file, flush=flush)

    return custom_print


@register_repl("python")
class PythonREPL(REPL):
    """REPL implementation that enforces compile-time safety policies.

    Configuration flags:
        multi_statement: If True (default), allows multiple statements per input.
            If False, enforces single-statement inputs via AST validation.
        allow_raise: If True (default), allows RAISE command to exit with exception.
            If False, only RETURN is available for finishing.
        allow_tools_command: If True (default), allows TOOLS? command.
            If False, TOOLS? is not available.
        allow_inspect_commands: If True (default), allows INSPECT and INSPECT_SOURCE.
            If False, INSPECT and INSPECT_SOURCE are not available.
        restrict_exec: If True, restricts any exec-mode code to a single tool call
            with JSON-like literal arguments.
    """

    repl_state_locals: dict[str, object]
    libraries: list[Library]
    session_id: str | None
    multi_statement: bool
    allow_raise: bool
    allow_tools_command: bool
    allow_inspect_commands: bool
    repl_history: bool
    expose_inputs_in_repl: bool
    restrict_exec: bool

    @property
    def description(self) -> str:
        """Return the appropriate description based on configuration."""
        return self.get_description(
            PythonREPLConfig(
                multi_statement=self.multi_statement,
                allow_raise=self.allow_raise,
                allow_tools_command=self.allow_tools_command,
                allow_inspect_commands=self.allow_inspect_commands,
                repl_history=self.repl_history,
                repl_description_template=self.repl_description_template,
                expose_inputs_in_repl=self.expose_inputs_in_repl,
                restrict_return_value=self.restrict_return_value,
                restrict_exec=self.restrict_exec,
            )
        )

    @classmethod
    def get_description(cls, config: PythonREPLConfig | None = None) -> str:
        """Return the appropriate description based on configuration.

        This class method allows getting the description without instantiating the REPL,
        which is useful for displaying in system prompts.

        Args:
            config: Configuration dict with optional keys:
                - multi_statement: bool (default True)
                - allow_raise: bool (default True)
                - allow_tools_command: bool (default True)
                - allow_inspect_commands: bool (default True)
                - repl_history: bool (default True)
                - repl_description_template: str (default "python_repl_description.jinja2")
                - restrict_exec: bool (default False)

        Returns:
            The appropriate REPL description string.
        """
        config = config or {}
        default_template = "python_repl_description.jinja2"
        template_name = config.get("repl_description_template", default_template)
        template = _jinja_env.get_template(template_name)
        return template.render(
            multi_statement=config.get("multi_statement", True),
            allow_raise=config.get("allow_raise", True),
            allow_tools_command=config.get("allow_tools_command", True),
            allow_inspect_commands=config.get("allow_inspect_commands", True),
            repl_history=config.get("repl_history", True),
            expose_inputs_in_repl=config.get("expose_inputs_in_repl", True),
            restrict_return_value=config.get("restrict_return_value", False),
            restrict_exec=config.get("restrict_exec", False),
        )

    @override
    def get_task_preamble(self) -> str:
        if self.expose_inputs_in_repl:
            return (
                "The following is your task. If you need access to the task as a string object,\n"
                "use the `__task__` magic variable instead of copying "
                "the task string into your code."
            )
        return "The following is your task."

    @override
    def get_inputs_preamble(self) -> str:
        if self.expose_inputs_in_repl:
            return (
                "The following inputs are available as Python variables in the REPL.\n"
                "Use them directly by name (e.g., `RETURN my_var` or `print(my_var)`).\n"
                "Do NOT copy their contents into your code."
            )
        return (
            "The following inputs were provided. They are NOT available as variables "
            "in the REPL.\n"
            "If you wish to access them, you should explicitly copy their contents "
            "into your code."
        )

    @override
    def get_must_exit_warning(
        self,
        *,
        template_name: str,
        can_delegate: bool,
        max_input_length: int,
        delegate_exec_timeout: float | None,
    ) -> str:
        # TODO: Get template from REPL config
        return _jinja_env.get_template(template_name).render(
            allow_raise=self.allow_raise,
            repl_history=self.repl_history,
            expose_inputs_in_repl=self.expose_inputs_in_repl,
            can_delegate=can_delegate,
            max_input_length=max_input_length,
            timeout=delegate_exec_timeout,
        )

    @override
    def get_truncation_advice(
        self,
        *,
        template_name: str,
        iteration: int,
        output_truncated: bool,
        error_truncated: bool,
    ) -> str:
        # TODO: Get template from REPL config
        return _jinja_env.get_template(template_name).render(
            repl_history=self.repl_history,
            iteration=iteration,
            output_truncated=output_truncated,
            error_truncated=error_truncated,
        )

    @override
    def get_finish_command_hint(self) -> str:
        if self.allow_raise:
            return "`RETURN ...` or `RAISE ...`"
        return "`RETURN ...`"

    TOOLS_CMD_RE = re.compile(r"^TOOLS\?\s*(.*)$", re.DOTALL)
    # INSPECT ...: regex allows no argument to let us handle the error explicitly
    INSPECT_CMD_RE = re.compile(r"^INSPECT\s*(?:\s(.*))?$", re.DOTALL)
    # INSPECT_SOURCE ...: regex allows no argument to let us handle the error explicitly
    INSPECT_SOURCE_CMD_RE = re.compile(r"^INSPECT_SOURCE\s*(?:\s(.*))?$", re.DOTALL)
    # RETURN and RAISE command patterns
    # Group 1: preceding code (optional, ends with newline)
    # Group 2: value/expression (optional)
    RETURN_CMD_RE = re.compile(r"^(.*\n)?RETURN(?:\s+(.*))?$", re.DOTALL)
    RAISE_CMD_RE = re.compile(r"^(.*\n)?RAISE(?:\s+(.*))?$", re.DOTALL)

    # Common incorrect usage of RETURN and RAISE commands
    RETURN_CMD_LAX_RE = re.compile(r"^\s*RETURN", re.MULTILINE)
    RAISE_CMD_LAX_RE = re.compile(r"^\s*RAISE", re.MULTILINE)

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
        # Filter out internal JAZ stack frames
        # Keep only frames from REPL code (filename doesn't end with .py)
        tb_filtered = exception.__traceback__
        while tb_filtered is not None:
            filename = tb_filtered.tb_frame.f_code.co_filename
            # REPL frames have filenames like "<repl_session_id_input_id>"
            if not filename.endswith(".py"):
                break
            tb_filtered = tb_filtered.tb_next

        # Format filtered traceback
        tb_str = "".join(
            traceback.format_exception(type(exception), exception, tb_filtered)
        )

        # Create error summary
        error_summary = f"{type(exception).__name__}: {str(exception)}"

        # Combine stdout and traceback
        formatted_output = stdout + tb_str

        return (formatted_output, error_summary)

    def __init__(
        self,
        repl_state_locals: dict[str, object],
        libraries: list[Library],
        repl_exec_timeout: float | None,
        forbidden_names: list[str],
        return_type: TypeForm[Any] | None = None,  # TODO: Fix static types
        return_validator: Callable[[Any], None] | None = None,
        repl_input_validator: Callable[[str], None] | None = None,
        forbidden_attributes: list[str] | None = None,
        session_id: str | None = None,
        multi_statement: bool = True,
        allow_raise: bool = True,
        allow_tools_command: bool = True,
        allow_inspect_commands: bool = True,
        repl_history: bool = True,
        repl_description_template: str = "python_repl_description.jinja2",
        restrict_invoke_args: bool = False,
        restrict_return_value: bool = False,
        restrict_exec: bool = False,
        expose_inputs_in_repl: bool = True,
        check_output_on_exit: bool = True,
        max_ast_nodes: int | None = None,
    ) -> None:
        """Initialize a PythonREPL instance with the given state.

        Arguments:
            repl_state_locals: Dictionary of local variables in the REPL.
            libraries: List of library objects available in the REPL.
            session_id: Unique session identifier for linecache filenames.
            return_type: The expected return type for RETURN commands.
            return_validator: Optional callable to validate the return value.
            repl_input_validator: Optional callable to validate REPL input source code
                before execution. Called with the source string; should raise ValueError
                with a descriptive message if the input is rejected.
            multi_statement: If True (default), allows multiple statements per input.
                If False, enforces single-statement inputs via AST validation.
            allow_raise: If True (default), allows RAISE command to exit with exception.
                If False, only RETURN is available for finishing.
            allow_tools_command: If True (default), allows TOOLS? command.
                If False, TOOLS? is not available.
            allow_inspect_commands: If True (default), allows INSPECT and
                INSPECT_SOURCE commands. If False, they are not available.
            repl_exec_timeout: Timeout in seconds for exec/eval operations.
            forbidden_names: List of forbidden variable/function names.
            forbidden_attributes: List of forbidden attribute names.
            repl_history: If True (default), maintains __repl_history__ builtin.
                If False, __repl_history__ is not created or updated.
            repl_description_template: Jinja2 template name for the REPL description.
            restrict_exec: If True, restricts any exec-mode code to a single tool call
                with JSON-like literal arguments.
            expose_inputs_in_repl: If True (default), exposes __task__ and __inputs__
                magic variables and adds input variables to repl_state_locals.
                If False, these are not created.
            check_output_on_exit: If True (default), blocks RETURN/RAISE if output
                was printed. If False, allows finishing even with printed output.
            max_ast_nodes: If not None, the maximum number of AST nodes allowed in
                a single REPL input. Exceeding this raises an error telling the
                agent to break its code into smaller steps.
        """
        self.repl_state_locals = repl_state_locals
        self.libraries = libraries
        self.session_id = session_id
        self.return_type = return_type
        self.return_validator = return_validator
        self.repl_input_validator = repl_input_validator
        self.multi_statement = multi_statement
        self.allow_raise = allow_raise
        self.allow_tools_command = allow_tools_command
        self.allow_inspect_commands = allow_inspect_commands
        self.repl_history = repl_history
        self.expose_inputs_in_repl = expose_inputs_in_repl
        self.repl_exec_timeout = repl_exec_timeout
        self.forbidden_names = forbidden_names
        self.forbidden_attributes = forbidden_attributes or []
        self.repl_description_template = repl_description_template
        self.restrict_invoke_args = restrict_invoke_args
        self.restrict_return_value = restrict_return_value
        self.restrict_exec = restrict_exec
        self.check_output_on_exit = check_output_on_exit
        self.max_ast_nodes = max_ast_nodes

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
        config: PythonREPLConfig | None = None,
        return_validator: Callable[[ReturnT], None] | None = None,
        repl_input_validator: Callable[[str], None] | None = None,
    ) -> Self:
        """Initialize the REPL with given inputs and configuration.

        The config dict can contain:
            multi_statement: bool (default True) - allow multiple statements per input
            allow_raise: bool (default True) - allow RAISE command
            repl_history: bool (default True) - maintain __repl_history__ builtin
            restrict_exec: bool (default False) - restrict exec-mode code to
                a single tool call with literal arguments
            allow_eval_exec: bool (default False) - allow eval()/exec() builtins
            allow_file_writes: bool (default False) - allow write-capable open() modes
            allowed_file_roots: list[str] | None (default None) - restrict file access to listed roots
            log_file_access: bool (default False) - log file access allow/deny decisions
            check_output_on_exit: bool (default True) - block RETURN/RAISE if output was printed
        """
        config = config or {}
        multi_statement = config.get("multi_statement", True)
        allow_raise = config.get("allow_raise", True)
        allow_tools_command = config.get("allow_tools_command", True)
        allow_inspect_commands = config.get("allow_inspect_commands", True)
        repl_history = config.get("repl_history", True)
        expose_inputs_in_repl = config.get("expose_inputs_in_repl", True)
        restrict_return_value = config.get("restrict_return_value", False)
        restrict_exec = config.get("restrict_exec", False)
        restrict_invoke_args = config.get("restrict_invoke_args", False)
        allow_eval_exec = config.get("allow_eval_exec", False)
        allow_file_writes = config.get("allow_file_writes", False)
        allowed_file_roots = config.get("allowed_file_roots")
        log_file_access = config.get("log_file_access", False)
        check_output_on_exit = config.get("check_output_on_exit", True)
        max_ast_nodes = config.get("max_ast_nodes")
        repl_description_template = config.get(
            "repl_description_template", "python_repl_description.jinja2"
        )

        # Validate return_type is a built-in type when restrict_return_value is on
        if restrict_return_value:
            try:
                validate_builtin_type_annotation(return_type)
            except TypeError:
                raise TypeError(
                    f"Only built-in return types are supported (e.g. int, bool, list, dict[str, float], None). "
                    f"Got: {return_type}"
                ) from None

        permission_policy = ReplPermissionPolicy(
            allow_eval_exec=allow_eval_exec,
            allow_file_writes=allow_file_writes,
            allowed_file_roots=allowed_file_roots,
            log_file_access=log_file_access,
        )
        allowed_builtins = build_allowed_builtins(
            base_builtins=allowed_builtins,
            policy=permission_policy,
        )

        def controlled_import(name: str, *args, **kwargs) -> ModuleType:
            # Check root module for dotted names (e.g. "concurrent.futures")
            root = name.split(".")[0]
            if root not in allowed_imports:
                raise ImportError(
                    f"Import of module '{name}' is forbidden in the REPL."
                )
            return __import__(name, *args, **kwargs)

        allowed_builtins["__import__"] = controlled_import

        # Initialize magic variables for accessing initialization info
        if expose_inputs_in_repl:
            allowed_builtins["__task__"] = task
            allowed_builtins["__inputs__"] = inputs.copy()
        if not restrict_return_value:
            allowed_builtins["__return_type__"] = return_type
            allowed_builtins["__return_validator__"] = return_validator

        # Initialize __repl_history__ only when enabled
        if repl_history:
            if expose_inputs_in_repl:
                # Index 0 contains initialization info, N >= 1 contains iteration N results
                allowed_builtins["__repl_history__"] = [
                    ReplHistoryInit(task=task, inputs=inputs.copy())
                ]
            else:
                allowed_builtins["__repl_history__"] = [None]

        init_repl_state_locals: dict[str, object] = {"__builtins__": allowed_builtins}
        for library in libraries:
            library.add_self_to_program_state(init_repl_state_locals)
        # TODO: If PEP annotation, recurse to find all types that need to be added
        if (
            return_type is not None
            and isinstance(return_type, type)
            and not restrict_return_value
        ):
            init_repl_state_locals[return_type.__name__] = return_type
        if expose_inputs_in_repl:
            init_repl_state_locals.update(inputs)

        return cls(
            repl_state_locals=init_repl_state_locals,
            libraries=libraries,
            repl_exec_timeout=repl_exec_timeout,
            forbidden_names=forbidden_names,
            return_type=return_type,
            return_validator=return_validator,
            repl_input_validator=repl_input_validator,
            forbidden_attributes=forbidden_attributes,
            session_id=session_id,
            multi_statement=multi_statement,
            allow_raise=allow_raise,
            allow_tools_command=allow_tools_command,
            allow_inspect_commands=allow_inspect_commands,
            repl_history=repl_history,
            repl_description_template=repl_description_template,
            restrict_invoke_args=restrict_invoke_args,
            restrict_return_value=restrict_return_value,
            restrict_exec=restrict_exec,
            expose_inputs_in_repl=expose_inputs_in_repl,
            check_output_on_exit=check_output_on_exit,
            max_ast_nodes=max_ast_nodes,
        )

    @override
    def add_inputs(self, inputs: dict[str, object]) -> None:
        """Add inputs to the REPL environment.

        Updates repl_state_locals directly. Does NOT update __inputs__ magic
        variable as per design decision - that remains a snapshot of initialization.

        Args:
            inputs: Dictionary of variable name -> value to add
        """
        self.repl_state_locals.update(inputs)

    @contextmanager
    def _capture_print(self, string_io: StringIO) -> Generator[None, None, None]:
        """Context manager that redirects print() to write to the given StringIO.

        Also sets up the parent output channel so that child jaz.invoke() calls
        can send messages back to this REPL's output via parent_print().
        """
        from jaz.parent_output import (
            reset_parent_output_channel,
            set_parent_output_channel,
        )

        builtins = self.repl_state_locals["__builtins__"]
        assert isinstance(builtins, dict)
        if "print" in builtins:
            old_print = builtins["print"]
            old_print_exists = True
        else:
            old_print = None
            old_print_exists = False
        new_print = _make_print(string_io)
        builtins["print"] = new_print

        # Set parent output channel so child invokes can send messages here
        output_token = set_parent_output_channel(lambda msg: new_print(msg))

        try:
            yield
        finally:
            reset_parent_output_channel(output_token)
            if old_print_exists:
                builtins["print"] = old_print

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
        # TODO: Also pass in previous exec output into self._format_error()
        formatted_output, error_summary = self._format_error(error_if_not_finish)
        return ErrorResult(
            output=formatted_output,
            error_summary=error_summary,
            exception=error_if_not_finish,
        )

    def _handle_tools_command(
        self, tools_match: re.Match
    ) -> ExecuteResult | ErrorResult:
        """Handle the TOOLS? command."""
        if tools_match.group(1):
            error = CommandSyntaxError("Command `TOOLS?` does not take any arguments.")
            formatted_output, error_summary = self._format_error(error)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )
        return ExecuteResult(
            output="\n".join(map(str, self.libraries)),
        )

    def _handle_inspect_command(
        self,
        inspect_match: re.Match | None,
        inspect_source_match: re.Match | None,
        input_id: str,
        filename: str,
        timeout: float | None,
    ) -> ExecuteResult | ErrorResult:
        """Handle the INSPECT or INSPECT_SOURCE command."""
        if inspect_match:
            obj_src = inspect_match.group(1)
            detail_level = 0
        else:
            assert inspect_source_match is not None
            obj_src = inspect_source_match.group(1)
            detail_level = 1

        if not obj_src:
            error = CommandSyntaxError("Command `INSPECT` requires an object name.")
            formatted_output, error_summary = self._format_error(error)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )

        obj_ast = ast.parse(obj_src, mode="eval").body
        if not isinstance(obj_ast, ast.Name | ast.Attribute):
            error = CommandSyntaxError(
                "Command `INSPECT` requires a valid object name. "
                "This is either a variable name (e.g., `my_var`) or "
                "an attribute access (e.g., `my_obj.my_attr`, "
                "`my_package.my_module`)."
            )
            formatted_output, error_summary = self._format_error(error)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )

        string_io = StringIO()
        try:
            compiled_obj_src = secure_compile(
                obj_src,
                input_id,
                forbidden_names=self.forbidden_names,
                filename=filename,
                mode="eval",
                forbidden_attributes=self.forbidden_attributes,
                restrict_invoke_args=self.restrict_invoke_args,
            )
            with self._capture_print(string_io):
                obj = eval_with_timeout(
                    compiled_obj_src, self.repl_state_locals, timeout
                )
        except BaseException as e:
            formatted_output, error_summary = self._format_error(
                e, string_io.getvalue()
            )
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=e,
            )

        obj_info_str = inspect_object(obj, obj_src.strip(), detail_level=detail_level)
        output_str = string_io.getvalue() + obj_info_str
        return ExecuteResult(
            output=output_str,
        )

    def _is_regular_python_code(self, src: str) -> bool:
        """Check if source is regular Python code (not a RETURN/RAISE command).

        Tries to parse the source as Python. If it parses without errors AND
        the last statement doesn't match RETURN or RAISE command patterns,
        returns True. Otherwise returns False.

        Args:
            src: The source code to check.

        Returns:
            True if the source is regular Python code, False if it's a special
            command or has syntax errors.
        """
        try:
            parsed = ast.parse(src)
            if parsed.body:
                last_stmt = parsed.body[-1]
                # Get the line number (1-indexed) and extract src from that line onwards
                last_stmt_lineno = last_stmt.lineno
                src_lines = src.splitlines(keepends=True)
                last_stmt_src = "".join(src_lines[last_stmt_lineno - 1 :])
                # If it matches RETURN command (without preceding code), treat as special
                return_match = self.RETURN_CMD_RE.match(last_stmt_src)
                if return_match and not return_match.group(1):
                    return False
                # If RAISE is allowed and matches RAISE command (without preceding code)
                if self.allow_raise:
                    raise_match = self.RAISE_CMD_RE.match(last_stmt_src)
                    if raise_match and not raise_match.group(1):
                        return False
                return True
            return True
        except SyntaxError:
            # Code has syntax errors -- should be RETURN or RAISE command
            return False

    def _execute_regular_code(
        self,
        src: str,
        input_id: str,
        filename: str,
        timeout: float | None,
    ) -> ExecuteResult | ErrorResult:
        """Execute regular Python code (no special commands)."""
        # Validate single statement if required
        if not self.multi_statement:
            try:
                tree = ast.parse(src)
                if len(tree.body) > 1:
                    error = MultipleStatementsError(
                        f"REPL input must contain exactly 1 statement, "
                        f"but found {len(tree.body)} statements. "
                        f"Please split your code into multiple REPL inputs."
                    )
                    formatted_output, error_summary = self._format_error(error)
                    return ErrorResult(
                        output=formatted_output,
                        error_summary=error_summary,
                        exception=error,
                    )

            except SyntaxError:
                pass

        string_io = StringIO()
        try:
            compiled_src = secure_compile(
                src,
                input_id,
                forbidden_names=self.forbidden_names,
                filename=filename,
                mode="exec",
                store_last_expr=self.repl_history,
                forbidden_attributes=self.forbidden_attributes,
                restrict_invoke_args=self.restrict_invoke_args,
                restrict_exec=self.restrict_exec,
            )
            with self._capture_print(string_io):
                exec_with_timeout(compiled_src, self.repl_state_locals, timeout)
        except BaseException as e:
            if self.restrict_exec and isinstance(e, SyntaxError):
                e = ReactToolCallError(str(e))
            formatted_output, error_summary = self._format_error(
                e, string_io.getvalue()
            )
            exception_to_use = e
            # In single-statement mode, provide helpful hints for RETURN/RAISE misuse
            # NOTE: This is best-effort check only
            if not self.multi_statement and isinstance(e, SyntaxError | NameError):
                if "RETURN" in str(e) or self.RETURN_CMD_LAX_RE.search(src):
                    error = CommandSyntaxError(
                        "Incorrect usage of `RETURN` command. "
                        "`RETURN <value>` must be the only content in the REPL input."
                    )
                    formatted_output, error_summary = self._format_error(
                        error, string_io.getvalue()
                    )
                    exception_to_use = error
                elif self.allow_raise and (
                    "RAISE" in str(e) or self.RAISE_CMD_LAX_RE.search(src)
                ):
                    error = CommandSyntaxError(
                        "Incorrect usage of `RAISE` command. "
                        "`RAISE <error>` must be the only content in the REPL input."
                    )
                    formatted_output, error_summary = self._format_error(
                        error, string_io.getvalue()
                    )
                    exception_to_use = error
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=exception_to_use,
            )
        return ExecuteResult(
            output=string_io.getvalue(),
        )

    def _handle_return_command(
        self,
        src: str,
        return_match: re.Match,
        input_id: str,
        filename: str,
        timeout: float | None,
    ) -> ReturnResult[Any] | ErrorResult:
        # TODO: Fix static types
        """Handle the RETURN command.

        Regex groups: group(1) = preceding code, group(2) = return value
        """
        string_io = StringIO()

        # Execute code before RETURN if present (multi-statement mode only)
        exec_src = return_match.group(1)
        if exec_src:
            # Reject preceding code when restrict_return_value is enabled
            if self.restrict_return_value:
                error = CommandSyntaxError(
                    "If a RETURN statement is present, it must be the only statement "
                    "in the REPL input; preceding code is not allowed."
                )
                formatted_output, error_summary = self._format_error(error)
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=error,
                )
            if not self.multi_statement:
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
            try:
                compiled_exec_src = secure_compile(
                    exec_src,
                    input_id,
                    filename=filename,
                    mode="exec",
                    forbidden_names=self.forbidden_names,
                    store_last_expr=self.repl_history,
                    forbidden_attributes=self.forbidden_attributes,
                    restrict_invoke_args=self.restrict_invoke_args,
                    restrict_exec=self.restrict_exec,
                )
                with self._capture_print(string_io):
                    exec_with_timeout(
                        compiled_exec_src,
                        self.repl_state_locals,
                        timeout,
                    )
            except BaseException as e:
                if self.restrict_exec and isinstance(e, SyntaxError):
                    e = ReactToolCallError(str(e))
                # TODO: Not the most robust check, e.g.,
                # RETURN command is correct and there's a syntax error elsewhere
                if isinstance(e, SyntaxError) and self.RETURN_CMD_LAX_RE.search(
                    exec_src
                ):
                    e = CommandSyntaxError(
                        "Incorrect usage of `RETURN` command. "
                        "Your REPL input must contain at most one `RETURN` command, "
                        "and if present it must be the last line in your input and must NOT be indented (i.e., it must NOT be nested inside an if block or for/while loop)."
                    )
                formatted_output, error_summary = self._format_error(
                    e, string_io.getvalue()
                )
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=e,
                )

        # Execute RETURN expression
        rv_src = return_match.group(2)

        # Validate RETURN expression is a JSON literal when restricted
        if self.restrict_return_value and rv_src:
            try:
                tree = ast.parse(rv_src, mode="eval")
            except SyntaxError:
                pass  # Let parent code handle parse errors
            else:
                try:
                    check_json_literal(tree.body)
                except SyntaxError:
                    error = CommandSyntaxError(
                        "RETURN value must be a JSON-like literal "
                        "(literal string, number, bool, None, list, dict)."
                    )
                    formatted_output, error_summary = self._format_error(error)
                    return ErrorResult(
                        output=formatted_output,
                        error_summary=error_summary,
                        exception=error,
                    )

        if not rv_src:
            return_value = None
        else:
            try:
                compiled_rv_src = secure_compile(
                    rv_src,
                    input_id,
                    forbidden_names=self.forbidden_names,
                    filename=filename,
                    mode="eval",
                    forbidden_attributes=self.forbidden_attributes,
                    restrict_invoke_args=self.restrict_invoke_args,
                )
                with self._capture_print(string_io):
                    return_value = eval_with_timeout(
                        compiled_rv_src, self.repl_state_locals, timeout
                    )
            except BaseException as e:
                formatted_output, error_summary = self._format_error(
                    e, string_io.getvalue()
                )
                # TODO: Not the most robust check, e.g.,
                # RETURN command is correct and there's a syntax error elsewhere
                if isinstance(e, SyntaxError):
                    if self.multi_statement:
                        error = CommandSyntaxError(
                            "Incorrect usage of `RETURN` command. "
                            "Your REPL input must contain at most one `RETURN` command, "
                            "and if present it must be the last line in your input and must NOT be indented (i.e., it must NOT be nested inside an if block or for/while loop)."
                        )
                    else:
                        error = CommandSyntaxError(
                            "Incorrect usage of `RETURN` command. "
                            "`RETURN <value>` must be the only content in the REPL input."
                        )
                    formatted_output, error_summary = self._format_error(
                        error, string_io.getvalue()
                    )
                    return ErrorResult(
                        output=formatted_output,
                        error_summary=error_summary,
                        exception=error,
                    )
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=e,
                )

        # Check that no output was printed before or during RETURN
        if self.check_output_on_exit and string_io.getvalue():
            error = RuntimeError(
                "Your `RETURN` command was not executed because your code produced "
                "printed output. Please review the output below. If you are "
                "confident and wish to finish, re-run only the RETURN command.\n"
                "⚠️ You cannot continue working in any way on this task after "
                "returning, so make sure you are confident with your work before "
                "returning."
            )
            formatted_output, error_summary = self._format_error(
                error, string_io.getvalue()
            )
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )

        # Check return value is of the correct type
        if not _check_type(return_value, self.return_type):
            return_type_name = (
                self.return_type.__name__
                if isinstance(self.return_type, type)
                else str(self.return_type)
            )
            error = CommandTypeError(
                f"Expected to return object of type {return_type_name}. "
                f"Got object of type {type(return_value)} instead."
            )
            formatted_output, error_summary = self._format_error(
                error, string_io.getvalue()
            )
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )

        # Validate return value if validator is provided
        if self.return_validator is not None:
            try:
                repl_history = None
                if self.repl_history:
                    repl_state_builtins = self.repl_state_locals["__builtins__"]
                    assert isinstance(repl_state_builtins, dict)
                    history_obj = repl_state_builtins.get("__repl_history__")
                    assert isinstance(history_obj, list)
                    repl_history = list(history_obj) + [
                        ReplHistoryEntry(
                            repl_input=src,
                            repl_output=string_io.getvalue(),
                            error=None,
                        )
                    ]
                self.run_return_validator(return_value, repl_history=repl_history)
            except Exception as e:
                formatted_output, error_summary = self._format_error(
                    e, string_io.getvalue() + "\nRETURN value validation failed:\n"
                )
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=e,
                )

        return ReturnResult(
            output=string_io.getvalue(),
            return_value=return_value,
        )

    def _handle_raise_command(
        self,
        raise_match: re.Match,
        input_id: str,
        filename: str,
        timeout: float | None,
    ) -> RaiseResult | ErrorResult:
        """Handle the RAISE command.

        Regex groups: group(1) = preceding code, group(2) = error expression
        """
        string_io = StringIO()

        # Execute code before RAISE if present (multi-statement mode only)
        exec_src = raise_match.group(1)
        if exec_src:
            if not self.multi_statement:
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
            try:
                compiled_exec_src = secure_compile(
                    exec_src,
                    input_id,
                    filename=filename,
                    forbidden_names=self.forbidden_names,
                    mode="exec",
                    store_last_expr=self.repl_history,
                    forbidden_attributes=self.forbidden_attributes,
                    restrict_invoke_args=self.restrict_invoke_args,
                    restrict_exec=self.restrict_exec,
                )
                with self._capture_print(string_io):
                    exec_with_timeout(
                        compiled_exec_src,
                        self.repl_state_locals,
                        timeout,
                    )
            except BaseException as e:
                if self.restrict_exec and isinstance(e, SyntaxError):
                    e = ReactToolCallError(str(e))
                # TODO: Not the most robust check, e.g.,
                # RAISE command is correct and there's a syntax error elsewhere
                if isinstance(e, SyntaxError) and self.RAISE_CMD_LAX_RE.search(
                    exec_src
                ):
                    e = CommandSyntaxError(
                        "Incorrect usage of `RAISE` command. "
                        "Your REPL input must contain at most one `RAISE` command, "
                        "and if present it must be the very last line of your input."
                    )
                formatted_output, error_summary = self._format_error(
                    e, string_io.getvalue()
                )
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=e,
                )

        # Execute RAISE expression
        err_src = raise_match.group(2)
        if not err_src:
            error = CommandSyntaxError(
                "Command `RAISE` must be called with an exception to raise."
            )
            formatted_output, error_summary = self._format_error(
                error, string_io.getvalue()
            )
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )

        try:
            compiled_err_src = secure_compile(
                err_src,
                input_id,
                filename=filename,
                forbidden_names=self.forbidden_names,
                mode="eval",
                forbidden_attributes=self.forbidden_attributes,
                restrict_invoke_args=self.restrict_invoke_args,
            )
            with self._capture_print(string_io):
                evaluated_error = eval_with_timeout(
                    compiled_err_src, self.repl_state_locals, timeout
                )
        except BaseException as e:
            formatted_output, error_summary = self._format_error(
                e, string_io.getvalue()
            )
            if isinstance(e, SyntaxError):
                if self.multi_statement:
                    error = CommandSyntaxError(
                        "Incorrect usage of `RAISE` command. "
                        "Your REPL input must contain at most one `RAISE` command, "
                        "and if present it must be the very last line of your input."
                    )
                else:
                    error = CommandSyntaxError(
                        "Incorrect usage of `RAISE` command. "
                        "`RAISE <error>` must be the only content in the REPL input."
                    )
                formatted_output, error_summary = self._format_error(
                    error, string_io.getvalue()
                )
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=error,
                )
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=e,
            )

        # Check that no output was printed before or during RAISE
        if self.check_output_on_exit and string_io.getvalue():
            error = RuntimeError(
                "Your `RAISE` command was not executed because your code produced "
                "printed output. Please review the output below. If you are "
                "confident and wish to finish, re-run only the RAISE command.\n"
                "⚠️ You cannot continue working in any way on this task afterwards, "
                "so make sure you are confident with your work before finishing."
            )
            formatted_output, error_summary = self._format_error(
                error, string_io.getvalue()
            )
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )

        if isinstance(evaluated_error, BaseException):
            return RaiseResult(
                output=string_io.getvalue(),
                exception=evaluated_error,
            )
        else:
            error = CommandTypeError(
                "Command `RAISE` must be called with an exception to raise."
            )
            formatted_output, error_summary = self._format_error(
                error, string_io.getvalue()
            )
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )

    def _handle_syntax_error(
        self, src: str, input_id: str, filename: str
    ) -> ErrorResult:
        """Handle syntax errors in the source code."""
        try:
            secure_compile(
                src,
                input_id,
                filename=filename,
                mode="exec",
                forbidden_names=self.forbidden_names,
                store_last_expr=self.repl_history,
                forbidden_attributes=self.forbidden_attributes,
                restrict_invoke_args=self.restrict_invoke_args,
            )
        except SyntaxError as e:
            formatted_output, error_summary = self._format_error(e)
            # Provide helpful hints if they tried to use special commands incorrectly
            if self.RETURN_CMD_LAX_RE.search(src):
                if self.multi_statement:
                    error = CommandSyntaxError(
                        "Incorrect usage of `RETURN` command. "
                        "Your REPL input must contain at most one `RETURN` command, "
                        "and if present it must be the last line in your input and must NOT be indented (i.e., it must NOT be nested inside an if block or for/while loop)."
                    )
                else:
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
            if self.allow_raise and self.RAISE_CMD_LAX_RE.search(src):
                if self.multi_statement:
                    error = CommandSyntaxError(
                        "Incorrect usage of `RAISE` command. "
                        "Your REPL input must contain at most one `RAISE` command, "
                        "and if present it must be the very last line of your input. "
                        "The `RAISE` command also requires an argument."
                    )
                else:
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
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=e,
            )

        # This should be unreachable - we already know src has a syntax error
        raise _JazInternalError("Unreachable code in PythonREPL._handle_syntax_error")

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

    def _update_repl_history[ReturnT](
        self,
        src: str,
        exec_result: ExecResult[ReturnT],
    ) -> None:
        """Update __repl_history__ with the result of this execution."""
        error_summary = (
            exec_result.error_summary if isinstance(exec_result, ErrorResult) else None
        )
        history_entry = ReplHistoryEntry(
            repl_input=src,
            repl_output=exec_result.output,
            error=error_summary,
        )
        repl_state_builtins = self.repl_state_locals["__builtins__"]
        assert isinstance(repl_state_builtins, dict)
        repl_state_repl_history = repl_state_builtins["__repl_history__"]
        assert isinstance(repl_state_repl_history, list)
        repl_state_repl_history.append(history_entry)

    def _exec(
        self,
        src: str,
        input_id: str,
        error_if_not_finish: Exception | None,
        raise_if_not_finish: bool,
        timeout: float | None,
    ) -> ExecResult[Any]:
        # TODO: Fix static types
        # Create filename for linecache - include session_id if available to avoid
        # collisions
        if self.session_id is not None:
            filename = f"<repl_{self.session_id}_{input_id}>"
        else:
            filename = f"<repl_{input_id}>"

        # Enforce max AST node count if configured.
        # TODO: Consider removing max_ast_nodes in favor of repl_input_validator,
        # which can implement the same check and is more flexible.
        # TODO: This doesn't even work when there's special commands like RETURN/RAISE.
        if self.max_ast_nodes is not None:
            try:
                import ast

                tree = ast.parse(src)
                node_count = sum(1 for _ in ast.walk(tree))
                if node_count > self.max_ast_nodes:
                    error_summary = "You are trying to do too much work in a single REPL iteration. Do only the first step in the current REPL iteration and defer the rest to future iterations."
                    return ErrorResult(
                        output=error_summary,
                        error_summary=error_summary,
                        exception=ValueError(error_summary),
                    )
            except SyntaxError:
                pass  # Let downstream parsing handle syntax errors

        # Run repl_input_validator if configured.
        if self.repl_input_validator is not None:
            try:
                self.repl_input_validator(src)
            except ValueError as e:
                error_summary = str(e)
                return ErrorResult(
                    output=error_summary,
                    error_summary=error_summary,
                    exception=e,
                )

        # Check for TOOLS? and INSPECT commands first
        tools_match = self.TOOLS_CMD_RE.match(src) if self.allow_tools_command else None
        inspect_match = (
            self.INSPECT_CMD_RE.match(src) if self.allow_inspect_commands else None
        )
        inspect_source_match = (
            self.INSPECT_SOURCE_CMD_RE.match(src)
            if self.allow_inspect_commands
            else None
        )

        if error_if_not_finish is not None and (
            tools_match or inspect_match or inspect_source_match
        ):
            return self._make_error_if_not_finish_result(
                error_if_not_finish, raise_if_not_finish
            )

        if tools_match:
            return self._handle_tools_command(tools_match)

        if inspect_match or inspect_source_match:
            return self._handle_inspect_command(
                inspect_match, inspect_source_match, input_id, filename, timeout
            )

        # Check if this is regular Python code (not a RETURN/RAISE command)
        if self._is_regular_python_code(src):
            # Valid Python code that doesn't end with RETURN/RAISE - execute directly
            if error_if_not_finish is not None:
                return self._make_error_if_not_finish_result(
                    error_if_not_finish, raise_if_not_finish
                )
            return self._execute_regular_code(src, input_id, filename, timeout)

        # Code has syntax errors or ends with RETURN/RAISE - check for RETURN/RAISE
        return_match = self.RETURN_CMD_RE.match(src)
        raise_match = self.RAISE_CMD_RE.match(src) if self.allow_raise else None

        # Check if we need to finish but don't have RETURN or RAISE
        if error_if_not_finish is not None and (
            return_match is None and raise_match is None
        ):
            return self._make_error_if_not_finish_result(
                error_if_not_finish, raise_if_not_finish
            )

        if return_match:
            return self._handle_return_command(
                src, return_match, input_id, filename, timeout
            )

        if raise_match:
            return self._handle_raise_command(raise_match, input_id, filename, timeout)

        # No special command matched - re-compile to get the original syntax error
        return self._handle_syntax_error(src, input_id, filename)

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
        """Executes a REPL command and mutates state in place.

        If the command is "TOOLS?", returns a string representation of the available
        tool libraries.
        If the command is "INSPECT ...", returns information about the object.
        If the command is "INSPECT_SOURCE ...", returns detailed information about the
        object.
        If the command is "RETURN ...", exit the REPL with the return value of the
        command.
        If the command is "RAISE ...", exists the REPL with the error raised by the
        command.
        Otherwise, executes the Python code in the REPL.
        If an error occurs during execution, then the error is captured and returned
        back into the REPL, with result category ERROR.

        ``return_type`` and ``return_validator`` are read from instance attributes
        set during initialization.

        Arguments:
            src: The REPL command to execute
            input_id: The unique identifier for the REPL input
            error_if_not_finish: The error to return if the command does not
                end the REPL session. `None` means we don't require the command
                to end the REPL session.
            raise_if_not_finish: Whether to raise an error (i.e., result category is
                RAISE instead of ERROR) if we require the command to end the REPL
                session but it does not.
            budget_forcing_active: If True, RETURN and RAISE commands are refused
                and an ErrorResult is returned instead, encouraging the agent to
                do more reasoning before finishing.

        Returns:
            exec_result: A tuple containing:
                result_category: The category of the result (i.e., EXECUTE, ERROR,
                RETURN, RAISE).
                output: The output of the REPL command that was executed
                The REPL state is mutated in place.
        """
        # TODO: Rename input_id to (1-based) iteration index?
        timeout = (
            exec_timeout_override
            if exec_timeout_override is not None
            else self.repl_exec_timeout
        )
        exec_result = self._exec(
            src,
            input_id,
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

        if self.repl_history:
            self._update_repl_history(src, exec_result)

        return exec_result
