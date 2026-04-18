from collections.abc import Callable
from typing import Self, override

from typing_extensions import TypedDict, TypeForm

from jaz.library import Library

from .python_repl import PythonREPL, PythonREPLConfig
from .registry import register_repl


class CodeactREPLConfig(TypedDict, total=False):
    """Configuration dictionary for CodeactREPL."""

    restrict_invoke_args: bool


@register_repl("codeact")
class CodeactREPL(PythonREPL):
    """Python REPL for CodeAct-style agents.

    Like PythonREPL but with the following restrictions:
    - No RAISE command (only RETURN for finishing)
    - Inputs are not exposed as variables in the REPL
    - RETURN expression must be a JSON-like literal (string, number, bool, None, list, dict)
      and not have preceding code
    - No TOOLS?, INSPECT, INSPECT_SOURCE commands
    - No magic variables (__task__, __return_type__, __return_validator__, __inputs__, __repl_history__)
    - Return type must be a JSON-like type (None, int, str, float, bool, list, dict)

    Retained from PythonREPL:
    - Arbitrary multi-statement Python code
    """

    FORCED_CONFIG: PythonREPLConfig = PythonREPLConfig(
        multi_statement=True,
        allow_raise=False,
        allow_tools_command=False,
        allow_inspect_commands=False,
        repl_history=False,
        expose_inputs_in_repl=False,
        restrict_return_value=True,
    )

    @property
    @override
    def description(self) -> str:
        return self.get_description()

    @override
    @classmethod
    def get_description(cls, config: PythonREPLConfig | None = None) -> str:
        assert config is None
        return super().get_description(cls.FORCED_CONFIG)

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
        config: CodeactREPLConfig | None = None,
        return_validator: Callable[[ReturnT], None] | None = None,
        repl_input_validator: Callable[[str], None] | None = None,
    ) -> Self:
        """Initialize a CodeactREPL with restricted configuration.

        Raises:
            TypeError: If return_type is not a built-in type
                (enforced by PythonREPL via restrict_return_value=True).
        """
        merged_config: PythonREPLConfig = {**cls.FORCED_CONFIG, **(config or {})}

        return super().initialize(
            task=task,
            return_type=return_type,  # type: ignore[reportArgumentType]  # TODO: Handle this properly
            inputs=inputs,
            libraries=libraries,
            allowed_imports=allowed_imports,
            repl_exec_timeout=repl_exec_timeout,
            forbidden_names=forbidden_names,
            forbidden_attributes=forbidden_attributes,
            allowed_builtins=allowed_builtins,
            session_id=session_id,
            config=merged_config,
            return_validator=return_validator,
            repl_input_validator=repl_input_validator,
        )
