import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Self

from typing_extensions import TypeForm

from jaz.library.core import Library

from .types import ExecResult


class REPL(ABC):
    """Base contract for REPL implementations used by LM agents.

    Subclasses store ``return_type`` and ``return_validator`` as instance
    attributes set at initialization time so that ``exec()`` does not need
    to receive them on every call.
    """

    # REPL-specific description shown in the system prompt
    # Subclasses should override this with their specific instructions
    description: str

    # Set at initialization; used by exec() to validate RETURN commands.
    return_type: TypeForm[Any] | None
    return_validator: Callable[[Any], None] | None

    @classmethod
    def get_description(cls, config: Mapping[str, Any] | None = None) -> str:
        """Get the REPL description for the given configuration.

        Subclasses can override this to return different descriptions based on
        the configuration (e.g., PythonREPL returns different descriptions for
        single-statement vs multi-statement mode).

        Args:
            config: Optional REPL-specific configuration dict.

        Returns:
            The REPL description string to show in the system prompt.
        """
        return cls.description

    @classmethod
    @abstractmethod
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
        config: Mapping[str, Any] | None = None,
        return_validator: Callable[[ReturnT], None] | None = None,
        repl_input_validator: Callable[[str], None] | None = None,
    ) -> Self:
        """Initialize the REPL with given inputs and allowed built-ins.

        Arguments:
            task: The task that initiated this REPL session.
            return_type: The expected return type of the REPL session.
            inputs: A dictionary of initial inputs to the REPL.
            libraries: List of library objects available in the REPL.
            allowed_imports: List of allowed import module names.
            allowed_builtins: A dictionary of allowed built-in functions and variables.
            session_id: Unique session identifier for this REPL instance.
            config: Optional REPL-specific configuration dict. Used to configure
                REPL behavior (e.g., multi_statement, allow_raise for PythonREPL).
            repl_exec_timeout: Timeout in seconds for exec/eval operations.
            forbidden_names: List of forbidden variable/function names.
            forbidden_attributes: List of forbidden attribute names.
            return_validator: Optional callable to validate the return value.
        Returns:
            The initialized REPL instance.
        """

    @abstractmethod
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
        """Execute REPL input and mutate state in place.

        This function should *never* raise an error. Instead, all errors should
        be captured and returned, displayed to the user of the REPL. The result
        category in the case of an error is EXECUTE.

        ``return_type`` and ``return_validator`` are read from the instance
        attributes set during initialization.

        Arguments:
            src: The REPL input to execute.
            input_id: The unique identifier for the REPL input.
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
            exec_result: A tuple with result_category, output, and optional
            return value or error. The REPL state is mutated in place.
        """

    def get_task_preamble(self) -> str:
        """Extra text appended after 'The following is your task.' in user prompt."""
        return ""

    def get_inputs_preamble(self) -> str:
        """Text describing how inputs are available in this REPL."""
        return ""

    def get_must_exit_warning(
        self,
        *,
        template_name: str,
        can_delegate: bool,
        max_input_length: int,
        delegate_exec_timeout: float | None,
    ) -> str:
        """Rendered must-exit warning for this REPL type."""
        return ""

    def get_truncation_advice(
        self,
        *,
        template_name: str,
        iteration: int,
        output_truncated: bool,
        error_truncated: bool,
    ) -> str:
        """Rendered truncation advice for this REPL type."""
        return ""

    def get_finish_command_hint(self) -> str:
        """Text like '`RETURN ...`' or '`RETURN ...` or `RAISE ...`'."""
        return "`RETURN ...`"

    def run_return_validator(
        self,
        return_value: object,
        *,
        repl_history: object | None = None,
    ) -> None:
        """Run the configured return validator.

        Validators may use either the legacy single-argument signature
        ``validator(return_value)`` or opt into REPL history inspection with
        ``validator(return_value, *, repl_history=...)``.
        """
        if self.return_validator is None:
            return

        validator = self.return_validator

        try:
            signature = inspect.signature(validator)
        except (TypeError, ValueError):
            validator(return_value)
            return

        accepts_repl_history = False
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_repl_history = True
                break
            if parameter.name == "repl_history" and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepts_repl_history = True
                break

        if accepts_repl_history:
            validator(return_value, repl_history=repl_history)  # type: ignore[call-arg]  # runtime-inspected kwarg
        else:
            validator(return_value)

    @abstractmethod
    def add_inputs(self, inputs: dict[str, object]) -> None:
        """Add inputs to the REPL environment after initialization.

        This method allows dynamically adding variables, functions, or other
        Python objects to the REPL state after the REPL has been initialized.
        This is useful for hooks to inject context or utilities into the REPL
        at different stages of execution (invoke, iteration, or execution level).

        Args:
            inputs: Dictionary of variable name -> value to add to REPL state

        Note:
            This method should be idempotent - calling it multiple times
            with the same key should update the value (last write wins).
            The behavior is implementation-specific - some REPLs may not
            support this operation (e.g., BashREPL).
        """
