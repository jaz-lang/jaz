import ast
import sys
from io import StringIO
from typing import Any, override

from jaz.exceptions import CommandSyntaxError, CommandTypeError
from jaz.template_loader import _jinja_env

from ._timeout_utils import exec_with_timeout
from .compiler import (
    check_json_literal,
    secure_compile,
)
from .python_repl import PythonREPL, PythonREPLConfig, _check_type
from .registry import register_repl
from .types import ErrorResult, ExecResult, ExecuteResult, RaiseResult, ReturnResult

# ---------------------------------------------------------------------------
# Sentinel exceptions — derive from BaseException so that
# `except Exception:` does NOT swallow them.
# ---------------------------------------------------------------------------


class _JazReturnSignal(BaseException):
    """Sentinel raised when user code executes a top-level ``return``."""

    def __init__(self, value: object) -> None:
        self.value = value


class _JazRaiseSignal(BaseException):
    """Sentinel raised when user code executes a top-level ``raise``."""

    def __init__(self, exception: object, cause: BaseException | None = None) -> None:
        self.exception = exception
        self.cause = cause


# ---------------------------------------------------------------------------
# AST transformer
# ---------------------------------------------------------------------------


class _NativeReturnRaiseTransformer(ast.NodeTransformer):
    """Replace top-level ``return`` / ``raise`` with sentinel raises.

    * ``return <value>`` → ``raise _JazReturnSignal(<value>)``
    * ``raise <exc>``    → ``raise _JazRaiseSignal(<exc>)``   (if *allow_raise*)

    Only transforms nodes at nesting depth 0 (i.e. *not* inside user-defined
    function / async-function / class bodies).
    """

    def __init__(self, allow_raise: bool) -> None:
        self.allow_raise = allow_raise
        self._depth = 0
        self.return_nodes: list[ast.Return] = []
        self.raise_transformed = 0

    # -- depth tracking for nested defs/classes --

    def _visit_nested(self, node: ast.AST) -> ast.AST:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_nested(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._visit_nested(node)

    # -- transformations at depth 0 --

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if self._depth > 0:
            return node
        self.return_nodes.append(node)
        # return <value> → raise _JazReturnSignal(<value>)
        value = node.value if node.value is not None else ast.Constant(value=None)
        new_node = ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="_JazReturnSignal", ctx=ast.Load()),
                args=[value],
                keywords=[],
            ),
            cause=None,
        )
        return ast.copy_location(new_node, node)

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        if self._depth > 0:
            return node
        if not self.allow_raise:
            return node
        self.raise_transformed += 1
        if node.exc is None:
            # bare `raise` → raise _JazRaiseSignal(_jaz_exc_info()[1])
            args: list[ast.expr] = [
                ast.Subscript(
                    value=ast.Call(
                        func=ast.Name(id="_jaz_exc_info", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                    slice=ast.Constant(value=1),
                    ctx=ast.Load(),
                )
            ]
        else:
            # raise <exc>          → raise _JazRaiseSignal(<exc>)
            # raise <exc> from <c> → raise _JazRaiseSignal(<exc>, <c>)
            args = [node.exc]
            if node.cause is not None:
                args.append(node.cause)
        new_node = ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="_JazRaiseSignal", ctx=ast.Load()),
                args=args,
                keywords=[],
            ),
            cause=None,
        )
        return ast.copy_location(new_node, node)


# ---------------------------------------------------------------------------
# Line-number fixup
# ---------------------------------------------------------------------------


def _adjust_col_offsets(node: ast.AST, delta: int) -> None:
    """Walk *node* and shift every ``col_offset`` / ``end_col_offset`` by *delta*."""
    for child in ast.walk(node):
        col = getattr(child, "col_offset", None)
        if col is not None:
            setattr(child, "col_offset", max(0, col + delta))  # noqa: B010
        end_col = getattr(child, "end_col_offset", None)
        if end_col is not None:
            setattr(child, "end_col_offset", max(0, end_col + delta))  # noqa: B010


# ---------------------------------------------------------------------------
# NativeReturnPythonREPL
# ---------------------------------------------------------------------------


@register_repl("python_native")
class NativeReturnPythonREPL(PythonREPL):
    """Python REPL variant that uses native ``return`` / ``raise`` syntax.

    Top-level ``return`` / ``raise`` statements (not inside a nested function
    or class) are reinterpreted as REPL exit points.  They work inside
    ``if`` / ``for`` / ``while`` / ``try`` blocks, just like in a real
    function body.

    No special commands: no ``RETURN``, ``RAISE``, ``TOOLS?``, ``INSPECT``,
    ``INSPECT_SOURCE``.
    """

    FORCED_CONFIG: PythonREPLConfig = PythonREPLConfig(
        allow_tools_command=False,
        allow_inspect_commands=False,
        multi_statement=True,
    )

    @classmethod
    @override
    def get_description(cls, config: PythonREPLConfig | None = None) -> str:
        config = config or {}
        template_name = config.get(
            "repl_description_template",
            "native_return_repl_description_simplified.jinja2",
        )
        template = _jinja_env.get_template(template_name)
        return template.render(
            allow_raise=config.get("allow_raise", True),
            repl_history=config.get("repl_history", True),
            expose_inputs_in_repl=config.get("expose_inputs_in_repl", True),
            restrict_return_value=config.get("restrict_return_value", False),
        )

    @classmethod
    @override
    def initialize(cls, *args, config=None, **kwargs):  # type: ignore[override]
        merged = dict(cls.FORCED_CONFIG)
        if config:
            merged.update(config)
        # Force our overrides back on top. TODO: handle this properly
        merged.update(cls.FORCED_CONFIG)
        return super().initialize(*args, config=merged, **kwargs)  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Apply forced config values
        for key, value in self.FORCED_CONFIG.items():
            kwargs[key] = value
        kwargs.setdefault(
            "repl_description_template",
            "native_return_repl_description_simplified.jinja2",
        )
        super().__init__(**kwargs)
        # Inject sentinel classes into builtins
        builtins = self.repl_state_locals["__builtins__"]
        assert isinstance(builtins, dict)
        # TODO: Should these be forbidden names to prevent LLM from using them or overriding them?
        builtins["_JazReturnSignal"] = _JazReturnSignal
        builtins["_JazRaiseSignal"] = _JazRaiseSignal
        builtins["_jaz_exc_info"] = sys.exc_info

    @override
    def get_finish_command_hint(self) -> str:
        if self.allow_raise:
            return "`return ...` or `raise ...`"
        return "`return ...`"

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
                "Use them directly by name (e.g., `return my_var` or `print(my_var)`).\n"
                "Do NOT copy their contents into your code."
            )
        return (
            "The following inputs were provided. They are NOT available as variables "
            "in the REPL.\n"
            "If you wish to access them, you should explicitly copy their contents "
            "into your code."
        )

    # ------------------------------------------------------------------
    # Core execution override
    # ------------------------------------------------------------------

    def _parse_and_transform(
        self, src: str
    ) -> tuple[ast.Module, _NativeReturnRaiseTransformer] | None:
        """Parse *src* and transform top-level ``return`` / ``raise``.

        Returns ``(tree, transformer)`` on success, ``None`` on syntax error.
        """
        transformer = _NativeReturnRaiseTransformer(allow_raise=self.allow_raise)

        # Fast path: no top-level return → ast.parse succeeds directly
        try:
            tree = ast.parse(src)
            transformer.visit(tree)
            ast.fix_missing_locations(tree)
            return tree, transformer
        except SyntaxError:
            # Slow path: wrap in a function to parse top-level return
            indented = "\n".join("    " + line for line in src.splitlines())
            wrapped = f"def __f__():\n{indented}\n"

            try:
                wrapped_tree = ast.parse(wrapped)
            except SyntaxError:
                return None

        # Extract function body
        func_def = wrapped_tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        body = func_def.body

        # Build a Module from the unwrapped body
        tree = ast.Module(body=body, type_ignores=[])

        # Transform
        transformer = _NativeReturnRaiseTransformer(allow_raise=self.allow_raise)
        transformer.visit(tree)

        # Fix line numbers: decrement lineno by 1 (wrapper added one line),
        # adjust col_offset by -4 (4-space indent)
        ast.increment_lineno(tree, -1)
        _adjust_col_offsets(tree, -4)

        ast.fix_missing_locations(tree)
        return tree, transformer

    @override
    def _exec(
        self,
        src: str,
        input_id: str,
        error_if_not_finish: Exception | None,
        raise_if_not_finish: bool,
        timeout: float | None,
    ) -> ExecResult[Any]:
        if self.session_id is not None:
            filename = f"<repl_{self.session_id}_{input_id}>"
        else:
            filename = f"<repl_{input_id}>"

        # ---- Parse and transform ----
        result = self._parse_and_transform(src)
        if result is None:
            return self._handle_syntax_error(src, input_id, filename)

        tree, transformer = result

        # ---- restrict_return_value checks ----
        if self.restrict_return_value and transformer.return_nodes:
            # return must be the only statement — no preceding code, no
            # nesting inside if/for/while/try.  After transformation the
            # Return has become a Raise, so tree.body must be a single Raise.
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Raise):
                error = CommandSyntaxError(
                    "If a `return` statement is present, it must be the only statement "
                    "in the REPL input; preceding code is not allowed."
                )
                formatted_output, error_summary = self._format_error(error)
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=error,
                )

            # return value must be a JSON-like literal
            for ret_node in transformer.return_nodes:
                val_node = ret_node.value
                if val_node is not None:
                    try:
                        check_json_literal(val_node)
                    except SyntaxError:
                        error = CommandSyntaxError(
                            "`return` value must be a JSON-like literal "
                            "(literal string, number, bool, None, list, dict)."
                        )
                        formatted_output, error_summary = self._format_error(error)
                        return ErrorResult(
                            output=formatted_output,
                            error_summary=error_summary,
                            exception=error,
                        )

        # ---- Static finish check ----
        code_might_finish = (
            len(transformer.return_nodes) > 0 or transformer.raise_transformed > 0
        )

        if error_if_not_finish is not None and not code_might_finish:
            return self._make_error_if_not_finish_result(
                error_if_not_finish, raise_if_not_finish
            )

        # ---- Apply policy visitors and compile ----
        try:
            compiled = secure_compile(
                src,
                input_id,
                forbidden_names=self.forbidden_names,
                filename=filename,
                mode="exec",
                store_last_expr=self.repl_history,
                forbidden_attributes=self.forbidden_attributes,
                restrict_invoke_args=self.restrict_invoke_args,
                tree=tree,
            )
        except SyntaxError as e:
            formatted_output, error_summary = self._format_error(e)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=e,
            )

        # ---- Execute ----
        string_io = StringIO()
        try:
            with self._capture_print(string_io):
                exec_with_timeout(compiled, self.repl_state_locals, timeout)
        except _JazReturnSignal as sig:
            return self._handle_native_return(sig.value, string_io)
        except _JazRaiseSignal as sig:
            return self._handle_native_raise(sig.exception, string_io, sig.cause)
        except BaseException as e:
            formatted_output, error_summary = self._format_error(
                e, string_io.getvalue()
            )
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=e,
            )

        # Normal completion (no return/raise executed at runtime)
        if error_if_not_finish is not None:
            return self._make_error_if_not_finish_result(
                error_if_not_finish, raise_if_not_finish
            )
        return ExecuteResult(output=string_io.getvalue())

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _handle_native_return(
        self, value: object, string_io: StringIO
    ) -> ReturnResult[Any] | ErrorResult:
        # Type check
        if not _check_type(value, self.return_type):
            return_type_name = (
                self.return_type.__name__
                if isinstance(self.return_type, type)
                else str(self.return_type)
            )
            error = CommandTypeError(
                f"Expected to return object of type {return_type_name}. "
                f"Got object of type {type(value)} instead."
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
                    builtins = self.repl_state_locals["__builtins__"]
                    assert isinstance(builtins, dict)
                    history_obj = builtins.get("__repl_history__")
                    if isinstance(history_obj, list):
                        repl_history = list(history_obj)
                self.run_return_validator(value, repl_history=repl_history)
            except Exception as e:
                formatted_output, error_summary = self._format_error(
                    e, string_io.getvalue() + "\nReturn value validation failed:\n"
                )
                return ErrorResult(
                    output=formatted_output,
                    error_summary=error_summary,
                    exception=e,
                )

        return ReturnResult(
            output=string_io.getvalue(),
            return_value=value,
        )

    def _handle_native_raise(
        self,
        exception: object,
        string_io: StringIO,
        cause: BaseException | None = None,
    ) -> RaiseResult | ErrorResult:
        if not isinstance(exception, BaseException):
            error = CommandTypeError(
                "`raise` must be called with an exception instance."
            )
            formatted_output, error_summary = self._format_error(
                error, string_io.getvalue()
            )
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=error,
            )
        exception.__cause__ = cause
        return RaiseResult(
            output=string_io.getvalue(),
            exception=exception,
        )

    # ------------------------------------------------------------------
    # Syntax error handling (override to avoid old-style command hints)
    # ------------------------------------------------------------------

    @override
    def _handle_syntax_error(
        self, src: str, input_id: str, filename: str
    ) -> ErrorResult:
        """Handle syntax errors without old-style command hints."""
        # TODO: Improve handling of syntax errors
        from jaz.exceptions import _JazInternalError

        try:
            # Compile to get the real syntax error
            compile(src, filename, "exec")
        except SyntaxError as e:
            formatted_output, error_summary = self._format_error(e)
            return ErrorResult(
                output=formatted_output,
                error_summary=error_summary,
                exception=e,
            )

        raise _JazInternalError(
            "Unreachable code in NativeReturnPythonREPL._handle_syntax_error"
        )
