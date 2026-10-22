"""AST helpers for the native-return REPL."""

# Consolidated here by #552: the same top-level-return parsing had been copy-pasted across the
# REPL, a since-removed delegate client and a host's hooks. The in-package caller is
# `PythonREPL._parse_and_transform`.

import ast


def parse_or_none(src: str) -> ast.Module | None:
    """Parse *src* as a module, returning ``None`` on a syntax error instead of raising.

    Callers use ``None`` to defer to the REPL, which positions the error against the source
    the agent actually submitted.
    """
    # The `None` contract is the whole reason this helper exists. The native-return finishes
    # #485 introduced need nothing else: `ast.parse` already accepts a module-level
    # `return`/`yield`/`await` — only `compile()` enforces return-outside-function.
    #
    # Deliberately no def-wrap fallback: re-parsing the source indented inside a synthesized
    # `def` accepts INDENTATION errors, and the indent it adds lands inside multi-line string
    # literals — '    s = """a\nb"""\n    print(repr(s))' ran, printing 'a\n    b'.
    try:
        return ast.parse(src)
    except SyntaxError:
        return None
