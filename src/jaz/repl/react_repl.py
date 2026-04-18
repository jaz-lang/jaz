import ast
from collections.abc import Callable
from typing import Self, override

from typing_extensions import TypeForm

from jaz.exceptions import CommandSyntaxError
from jaz.library import Library
from jaz.template_loader import _jinja_env

from .compiler import check_json_literal, secure_compile
from .python_repl import PythonREPL, PythonREPLConfig, ReactToolCallError
from .registry import register_repl

__all__ = ["ReactREPL", "ReactToolCallError"]


@register_repl("react")
class ReactREPL(PythonREPL):
    """Compatibility wrapper for ReAct-style tool-call-only REPL mode.

    This class delegates behavior to :class:`PythonREPL` with an explicit
    forced config:
    - CodeAct-style flags (no RAISE/tools/inspect, hidden inputs/history,
      literal-only RETURN)
    - ``restrict_exec=True`` for tool-call-only exec-mode code.
    """

    FORCED_CONFIG: PythonREPLConfig = PythonREPLConfig(
        multi_statement=True,
        allow_raise=False,
        allow_tools_command=False,
        allow_inspect_commands=False,
        repl_history=False,
        expose_inputs_in_repl=False,
        restrict_return_value=True,
        restrict_exec=True,
    )

    @override
    @classmethod
    def get_description(cls, config: PythonREPLConfig | None = None) -> str:
        config = config or {}
        template_name = config.get(
            "repl_description_template", "react_repl_description.jinja2"
        )
        template = _jinja_env.get_template(template_name)
        return template.render()

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
        config = config or {}
        merged_config: PythonREPLConfig = {
            **config,
            **cls.FORCED_CONFIG,
        }
        merged_config.setdefault(
            "repl_description_template", "react_repl_description.jinja2"
        )
        return super().initialize(
            task=task,
            return_type=return_type,
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

    @staticmethod
    def _validate_tool_call(src: str) -> None:
        """Validate tool-call-only input."""
        try:
            secure_compile(
                src,
                "validate_tool_call",
                forbidden_names=[],
                mode="exec",
                store_last_expr=False,
                restrict_exec=True,
            )
        except SyntaxError as e:
            raise ReactToolCallError(str(e)) from e

    @staticmethod
    def _validate_return_expression(expr: str) -> None:
        """Validate that a RETURN expression is a constant."""
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise CommandSyntaxError(
                f"RETURN value must be a constant. Parse error: {e}"
            ) from e

        try:
            check_json_literal(tree.body)
        except SyntaxError:
            raise CommandSyntaxError(
                "RETURN value must be a JSON-like literal (literal string, number, bool, None, list, dict)."
            ) from None
