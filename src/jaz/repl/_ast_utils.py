"""Shared AST helpers for the native-return REPL and the eval/delegation layers.

Previously the "parse source tolerating a module-level ``return``/``raise``" fallback was
copy-pasted in three places (``PythonREPL._parse_and_transform``,
``jaz.llm_client_delegate``, and ``evals/repl_config_hooks``). Consolidated here (#552) so the
one tricky parse dance lives in a single place.
"""

import ast


def _adjust_col_offsets(node: ast.AST, delta: int) -> None:
    """Walk *node* and shift every ``col_offset`` / ``end_col_offset`` by *delta*."""
    for child in ast.walk(node):
        col = getattr(child, "col_offset", None)
        if col is not None:
            setattr(child, "col_offset", max(0, col + delta))  # noqa: B010
        end_col = getattr(child, "end_col_offset", None)
        if end_col is not None:
            setattr(child, "end_col_offset", max(0, end_col + delta))  # noqa: B010


def parse_allowing_toplevel_return(src: str) -> ast.Module | None:
    """Parse *src* as a module, tolerating a module-level ``return`` / ``raise``.

    Those are valid finishes in the native-return REPL (#485) but historically ``ast.parse``
    rejected a bare module-level ``return``. Returns the parsed module, or ``None`` on a
    *genuine* syntax error so callers can defer to the REPL's own (better-positioned) error.

    The def-wrap branch is a defensive fallback: on supported Pythons (3.12+) ``ast.parse``
    already accepts a top-level ``return`` / ``yield`` / ``await`` (only ``compile()`` enforces
    return-outside-function), so the fast path handles them and this branch is effectively
    unreachable. It is kept for any construct that parses solely inside a function body; when it
    *is* hit, line/column offsets are corrected back so tracebacks still point at the real
    source. (Caveat: the 4-space indent it adds would corrupt a multi-line string literal — the
    reason this stays a last resort rather than the primary path.)
    """
    try:
        return ast.parse(src)
    except SyntaxError:
        pass
    indented = "\n".join("    " + line for line in src.splitlines())
    try:
        wrapped = ast.parse(f"def __jaz_toplevel_return_wrap__():\n{indented}\n")
    except SyntaxError:
        return None
    func = wrapped.body[0]
    if not isinstance(func, ast.FunctionDef):
        return None
    module = ast.Module(body=func.body, type_ignores=[])
    # Undo the wrapper's one added line + 4-space indent so locations match the original source.
    ast.increment_lineno(module, -1)
    _adjust_col_offsets(module, -4)
    ast.fix_missing_locations(module)
    return module
