import ast
import linecache
import types
import typing
from collections.abc import Container
from typing import Literal, cast

ALLOWED_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "tuple",
        "None",
    }
)

ALLOWED_BUILTIN_TYPES: frozenset[type] = frozenset(
    {
        int,
        str,
        float,
        bool,
        list,
        dict,
        tuple,
        type(None),
    }
)


_JSON_LITERAL_NODES: frozenset[type] = frozenset(
    {ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Load, ast.USub}
)


def check_json_literal(node: ast.AST) -> None:
    """Validate that *node* is a JSON-like literal expression.

    Allowed constructs: ``ast.Constant``, ``ast.List``, ``ast.Tuple``,
    ``ast.Dict``, ``ast.Set``, and ``ast.UnaryOp(USub, Constant(int|float))``.
    Everything else (names, binary ops, calls, etc.) is rejected.

    Raises:
        SyntaxError: If the expression contains non-literal constructs.
    """
    for child in ast.walk(node):
        if type(child) in _JSON_LITERAL_NODES:
            continue
        if isinstance(child, ast.UnaryOp):
            if (
                isinstance(child.op, ast.USub)
                and isinstance(child.operand, ast.Constant)
                and isinstance(child.operand.value, (int, float))
            ):
                continue
        lineno = getattr(child, "lineno", None)
        raise SyntaxError(
            f"{type(child).__name__} is not allowed; "
            f"only JSON-like literal values are permitted"
            + (f" (line {lineno})" if lineno is not None else "")
        )


def check_json_type_hint(node: ast.AST) -> None:
    """Validate that *node* is a type hint for JSON types.

    Allowed leaf types: ``str``, ``int``, ``float``, ``bool``, ``list``,
    ``dict``, and ``None``.  These may be composed with subscripts
    (e.g. ``list[str]``, ``dict[str, int]``) and ``|`` unions
    (e.g. ``str | int``, ``list[str] | None``).

    Raises:
        SyntaxError: If the expression uses disallowed types or constructs.
    """
    if isinstance(node, ast.Constant):
        if node.value is not None:
            raise SyntaxError(
                f"Only None is allowed as a constant in type hints (line {node.lineno})"
            )
        return
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_BUILTIN_NAMES:
            raise SyntaxError(
                f"Type name {node.id!r} is not allowed; "
                f"only built-in type names are permitted "
                f"(line {node.lineno})"
            )
        return
    if isinstance(node, ast.Subscript):
        check_json_type_hint(node.value)
        # node.slice is a Tuple for multi-arg generics like dict[str, int]
        if isinstance(node.slice, ast.Tuple):
            for elt in node.slice.elts:
                check_json_type_hint(elt)
        else:
            check_json_type_hint(node.slice)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ast.BitOr):
            raise SyntaxError(
                f"Only | (union) is allowed in type hints (line {node.lineno})"
            )
        check_json_type_hint(node.left)
        check_json_type_hint(node.right)
        return
    lineno = getattr(node, "lineno", None)
    raise SyntaxError(
        "Invalid type hint expression"
        + (f" (line {lineno})" if lineno is not None else "")
    )


def validate_builtin_type_annotation(
    annotation: object, *, _allow_nonetype: bool = False
) -> None:
    """Recursively validate that a runtime type annotation uses only built-in types.

    ``None`` (the value) is accepted as a type annotation.  ``NoneType``
    (i.e. ``type(None)``) is only allowed as a member of a union
    (e.g. ``str | None``), not as a top-level annotation.  We can't reject
    ``NoneType`` inside unions because Python represents ``str | None`` as
    ``UnionType(str, NoneType)`` at runtime.

    Raises:
        TypeError: If the annotation contains non-builtin types.
    """
    if annotation is None:
        return

    if annotation is type(None):
        if _allow_nonetype:
            return
        raise TypeError(
            "Use None instead of NoneType for type annotations. "
            "For optional types, use e.g. str | None."
        )

    if isinstance(annotation, type):
        if annotation not in ALLOWED_BUILTIN_TYPES:
            raise TypeError(f"Only built-in types are supported. Got: {annotation}")
        return

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Handle Union types (str | int, Optional[str], Union[str, int])
    if origin is types.UnionType or origin is typing.Union:
        for arg in args:
            validate_builtin_type_annotation(arg, _allow_nonetype=True)
        return

    # Handle generic aliases (list[str], dict[str, int])
    if origin is not None:
        if origin not in ALLOWED_BUILTIN_TYPES:
            raise TypeError(f"Only built-in types are supported. Got origin: {origin}")
        for arg in args:
            if arg is Ellipsis:
                continue
            validate_builtin_type_annotation(arg, _allow_nonetype=True)
        return

    raise TypeError(f"Only built-in types are supported. Got: {annotation}")


class ForbiddenNamesChecker(ast.NodeVisitor):
    def __init__(self, forbidden_names: Container[str]):
        self.forbidden_names = forbidden_names

    def visit_Name(self, node):
        if node.id in self.forbidden_names:
            raise SyntaxError(
                f"Reserved name {node.id!r} cannot be used (line {node.lineno})"
            )
        return self.generic_visit(node)


class ForbiddenAttributesChecker(ast.NodeVisitor):
    def __init__(self, forbidden_attributes: Container[str]):
        self.forbidden_attributes = forbidden_attributes

    def visit_Attribute(self, node):
        if node.attr in self.forbidden_attributes:
            raise SyntaxError(
                f"Attribute access {node.attr!r} is forbidden (line {node.lineno})"
            )
        return self.generic_visit(node)


class InvokeArgsChecker(ast.NodeVisitor):
    """Ensures any statement containing jaz.invoke() is a single
    Expr(Call(...)) with all-constant args."""

    @staticmethod
    def _is_jaz_invoke(node: ast.AST) -> bool:
        """Check if an AST node is a call to jaz.invoke(...)."""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "invoke"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "jaz"
        )

    @staticmethod
    def _check_call_args(call: ast.Call) -> None:
        """Validate that all args/kwargs of a jaz.invoke() call are literals.

        Positional args and most keyword args must be JSON-like literals.
        The ``return_type`` keyword arg must be a valid built-in type hint.
        """
        for arg in call.args:
            try:
                # This also handles *args, which is an ast.Starred node.
                check_json_literal(arg)
            except SyntaxError:
                raise SyntaxError(
                    "jaz.invoke() calls must use literal arguments only; "
                    f"found non-constant positional argument (line {arg.lineno})"
                ) from None
        for kw in call.keywords:
            if kw.arg is None:
                raise SyntaxError(
                    "jaz.invoke() calls must use literal arguments only; "
                    "**kwargs is not allowed"
                )
            checker = (
                check_json_type_hint if kw.arg == "return_type" else check_json_literal
            )
            try:
                checker(kw.value)
            except SyntaxError:
                raise SyntaxError(
                    "jaz.invoke() calls must use literal arguments only; "
                    f"found non-constant keyword argument {kw.arg!r} "
                    f"(line {kw.value.lineno})"
                ) from None

    def _find_invoke_call(self, node: ast.AST) -> ast.Call | None:
        """Walk the subtree looking for a jaz.invoke() call."""
        if self._is_jaz_invoke(node):
            assert isinstance(node, ast.Call)
            return node
        for child in ast.walk(node):
            if child is not node and self._is_jaz_invoke(child):
                assert isinstance(child, ast.Call)
                return child
        return None

    def _check_invoke_expr(self, expr_value: ast.AST) -> None:
        """Check that an expression is exactly a jaz.invoke() call with constant args."""
        invoke_call = self._find_invoke_call(expr_value)
        if invoke_call is None:
            return
        # The expression must be *exactly* the jaz.invoke() call, not nested
        if expr_value is not invoke_call:
            raise SyntaxError(
                "jaz.invoke() must be a standalone expression, "
                "not nested inside another expression"
            )
        self._check_call_args(invoke_call)

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            invoke_call = self._find_invoke_call(stmt)
            if invoke_call is None:
                continue
            # Statement contains jaz.invoke() — enforce constraints
            if len(node.body) != 1:
                raise SyntaxError(
                    "jaz.invoke() must be the only statement in the REPL input"
                )
            if not isinstance(stmt, ast.Expr):
                raise SyntaxError(
                    "Statements containing jaz.invoke() must be bare expressions, "
                    "not assignments, loops, etc."
                )
            if stmt.value is not invoke_call:
                raise SyntaxError(
                    "jaz.invoke() must be a standalone expression, "
                    "not nested inside another expression"
                )
            self._check_call_args(invoke_call)

    def visit_Expression(self, node: ast.Expression) -> None:
        self._check_invoke_expr(node.body)


class RestrictExecChecker(ast.NodeVisitor):
    """Enforces tool-call-only input for restrict_exec mode."""

    @staticmethod
    def _check_call_args(call: ast.Call) -> None:
        for i, arg in enumerate(call.args):
            try:
                check_json_literal(arg)
            except SyntaxError:
                raise SyntaxError(
                    f"Positional argument {i} must be a JSON-like literal (literal string, number, bool, None, list, dict)."
                ) from None

        for kw in call.keywords:
            checker = (
                check_json_type_hint if kw.arg == "return_type" else check_json_literal
            )
            try:
                checker(kw.value)
            except SyntaxError:
                if kw.arg == "return_type":
                    raise SyntaxError(
                        "Keyword argument 'return_type' must be a valid JSON type hint (e.g. str, list[int], str | None)."
                    ) from None
                raise SyntaxError(
                    f"Keyword argument '{kw.arg}' must be a JSON-like literal (literal string, number, bool, None, list, dict)."
                ) from None

    def _check_tool_call_expr(self, expr: ast.AST) -> None:
        if not isinstance(expr, ast.Call):
            raise SyntaxError(f"Input must be a tool call, not {type(expr).__name__}.")
        if not isinstance(expr.func, ast.Name | ast.Attribute):
            raise SyntaxError(
                "Function must be a name or attribute access, "
                f"not {type(expr.func).__name__}."
            )
        self._check_call_args(expr)

    def visit_Module(self, node: ast.Module) -> None:
        if len(node.body) != 1:
            raise SyntaxError(
                f"Input must be a single tool call. Got {len(node.body)} statements."
            )
        stmt = node.body[0]
        if not isinstance(stmt, ast.Expr):
            raise SyntaxError(
                f"Input must be a tool call expression, not {type(stmt).__name__}."
            )
        self._check_tool_call_expr(stmt.value)

    def visit_Expression(self, node: ast.Expression) -> None:
        self._check_tool_call_expr(node.body)


class PrintLastExprTransformer(ast.NodeTransformer):
    def __init__(self, input_id: str, store_last_expr: bool = True):
        # This is the ID of the REPL input.
        # Usually this is just `str(n)` where this is the `n`th input
        # in a REPL session.
        self.input_id = input_id
        self.store_last_expr = store_last_expr

    def visit_Module(self, node: ast.Module):
        if len(node.body) == 0 or not isinstance(node.body[-1], ast.Expr):
            return node
        last_expr_value = node.body[-1].value
        if self.store_last_expr:
            # (Suppose for example that self.input_id == 7)
            # ```
            # x = EXPR
            # if x is not None:
            #     print(repr(x))
            #     __builtins__['_'] = x
            #     __builtins__['_7'] = x
            # ```
            # Note that `_` is in `__builtins__` instead of `globals()` --
            # what Python REPL does. Similarly for `_7`, so that if user does
            # `_7 = ...` and later `del _7`, `_7` becomes accessible again.
            new_last_expr = ast.parse(
                "(lambda x: "
                "    None if x is None else "
                "        print(repr(x)) or "
                "        __builtins__.__setitem__('_', x) or "
                f"       __builtins__.__setitem__('_{self.input_id}', x)"
                ")(EXPR)"
            ).body[0]
        else:
            # Print the result but don't store in _ or _N.
            new_last_expr = ast.parse(
                "(lambda x: None if x is None else print(repr(x)))(EXPR)"
            ).body[0]
        assert isinstance(new_last_expr, ast.Expr)
        new_last_expr_value = new_last_expr.value
        assert isinstance(new_last_expr_value, ast.Call)
        new_last_expr_value.args[0] = last_expr_value
        node.body[-1].value = new_last_expr_value
        return node


def apply_policies[TreeT: ast.AST](
    policies: list[ast.NodeVisitor], tree: TreeT
) -> TreeT:
    for policy in policies:
        policy.visit(tree)
    ast.fix_missing_locations(tree)
    return tree


type CompileMode = Literal["exec", "eval", "single"]


def secure_compile(
    src: str,
    input_id: str,
    forbidden_names: list[str],
    filename: str = "<repl>",
    mode: CompileMode = "exec",
    store_last_expr: bool = True,
    forbidden_attributes: list[str] | None = None,
    restrict_invoke_args: bool = False,
    restrict_exec: bool = False,
    tree: ast.AST | None = None,
):
    if tree is None:
        tree = ast.parse(src, filename=filename, mode=mode)
    match mode:
        case "exec":
            tree = cast(ast.Module, tree)
        case "eval":
            tree = cast(ast.Expression, tree)
        case "single":
            tree = cast(ast.Interactive, tree)
        case _:
            raise ValueError(f"Unsupported compile mode {mode!r}.")

    policies: list[ast.NodeVisitor] = [
        ForbiddenNamesChecker(forbidden_names),
    ]
    if forbidden_attributes:
        policies.append(ForbiddenAttributesChecker(forbidden_attributes))
    if restrict_invoke_args:
        policies.append(InvokeArgsChecker())
    if restrict_exec:
        policies.append(RestrictExecChecker())
    if mode == "exec":
        policies.append(PrintLastExprTransformer(input_id, store_last_expr))

    # 1) enforce rules and transform calls
    tree = apply_policies(policies, tree)

    # 2) Add source to linecache so tracebacks can display source code
    # Split source into lines, preserving newlines for linecache
    lines = [line + "\n" for line in src.splitlines()]
    if lines and not src.endswith("\n"):
        # Remove the trailing newline from the last line if original didn't have it
        lines[-1] = lines[-1].rstrip("\n")
    # linecache.cache format: (size, mtime, lines, fullname)
    linecache.cache[filename] = (len(src), None, lines, filename)

    # 3) normal CPython compile
    return compile(tree, filename, mode=mode)
