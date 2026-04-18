from __future__ import annotations

import ast
import re
from typing import Any

from .agent import REPL_INPUT_REGEX as _REPL_INPUT_RE
from .llm_client import LiteLLMClient, LLMClient, LLMResponse
from .template_loader import _jinja_env

# Matches a trailing RETURN or RAISE command (last non-blank line).
# TODO: This is currently specific to PythonREPL
_RETURN_RE = re.compile(r"^RETURN(?:\s+.*)?$", re.MULTILINE)
_RAISE_RE = re.compile(r"^RAISE(?:\s+.*)?$", re.MULTILINE)

# model_config key that holds the template name
_TEMPLATE_KEY = "delegate_transform_template"
_SLIM_TRANSFORM_KEY = "delegate_slim_transform"
_PROGRAMMATIC_KEY = "delegate_programmatic_transform"
_INCLUDE_PLAN_KEY = "delegate_include_plan"
_INCLUDE_INSTRUCTION_KEY = "delegate_include_instruction"

_DELEGATE_INSTRUCTION = (
    "The `history` input above contains your previous REPL history"
    " -- for example, its last entry contains the last REPL input"
    " you executed and its output. Plan your next step based on the `history`"
    " of what you've done so far."
)

# Matches <plan>...</plan> blocks in LLM responses
_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)


def _sum_optional(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def _sum_optional_float(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    return (a or 0.0) + (b or 0.0)


def _repl_input_already_finishes(content: str) -> bool:
    """Return True if the last <repl_input> block already ends with RETURN/RAISE."""
    match = _REPL_INPUT_RE.search(content)
    if match is None:
        return False
    body = match.group(3).strip()
    last_line = body.rsplit("\n", 1)[-1].strip()
    if _RETURN_RE.match(last_line) or _RAISE_RE.match(last_line):
        return True
    return False


def _render_transform_prompt(template_name: str) -> str:
    """Render the delegation transform template from model_config values."""
    return _jinja_env.get_template(template_name).render()


class _PrintCaptureTransformer(ast.NodeTransformer):
    """Replace ``print(...)`` calls with ``__captured_output += ...`` statements.

    Standalone ``print(x)`` expression statements become
    ``__captured_output += str(x) + '\\n'``.

    Nested ``print(x)`` inside larger expressions (e.g. ternaries) become
    ``(__captured_output := __captured_output + str(x) + '\\n') and None``
    which captures the output and evaluates to ``None`` (matching print's
    return value).
    """

    @staticmethod
    def _build_print_args_expr(call: ast.Call) -> ast.expr | None:
        """Build the string expression for print args.

        Returns None if the call cannot be transformed (keyword args like
        sep=, end=, file=, flush= change print semantics).
        """
        if call.keywords:
            return None

        args = call.args
        if not args:
            return ast.Constant(value="\n")

        has_starred = any(isinstance(a, ast.Starred) for a in args)

        if has_starred:
            # Use ' '.join(map(str, [a, *args, b])) + '\n'
            list_node = ast.List(elts=list(args), ctx=ast.Load())
            map_call = ast.Call(
                func=ast.Name(id="map", ctx=ast.Load()),
                args=[ast.Name(id="str", ctx=ast.Load()), list_node],
                keywords=[],
            )
            join_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Constant(value=" "),
                    attr="join",
                    ctx=ast.Load(),
                ),
                args=[map_call],
                keywords=[],
            )
            return ast.BinOp(
                left=join_call, op=ast.Add(), right=ast.Constant(value="\n")
            )

        # Static case: no starred args
        parts: list[ast.expr] = []
        for i, arg in enumerate(args):
            str_call = ast.Call(
                func=ast.Name(id="str", ctx=ast.Load()),
                args=[arg],
                keywords=[],
            )
            parts.append(str_call)
            if i < len(args) - 1:
                parts.append(ast.Constant(value=" "))
        parts.append(ast.Constant(value="\n"))
        rhs = parts[0]
        for part in parts[1:]:
            rhs = ast.BinOp(left=rhs, op=ast.Add(), right=part)
        return rhs

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        # Standalone print(...) statement → __captured_output += ...
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
        ):
            rhs = self._build_print_args_expr(node.value)
            if rhs is None:
                # TODO: May want to revisit deleting untransformable print statements
                return None
            new_node = ast.AugAssign(
                target=ast.Name(id="__captured_output", ctx=ast.Store()),
                op=ast.Add(),
                value=rhs,
            )
            return ast.copy_location(new_node, node)
        # Visit children so nested print() calls in expressions are transformed
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # First, visit children (handles nested print inside print args, etc.)
        visited = self.generic_visit(node)
        assert isinstance(visited, ast.Call)
        # Transform print(...) nested in expressions →
        # (__captured_output := __captured_output + ...) and None
        if not (isinstance(visited.func, ast.Name) and visited.func.id == "print"):
            return visited
        rhs = self._build_print_args_expr(visited)
        if rhs is None:
            # TODO: May want to revisit replacing untransformable print calls with None
            return ast.copy_location(ast.Constant(value=None), visited)
        walrus = ast.NamedExpr(
            target=ast.Name(id="__captured_output", ctx=ast.Store()),
            value=ast.BinOp(
                left=ast.Name(id="__captured_output", ctx=ast.Load()),
                op=ast.Add(),
                right=rhs,
            ),
        )
        # walrus evaluates to the new __captured_output (non-empty string, truthy),
        # so `walrus and None` evaluates to None, matching print()'s return value.
        new_node = ast.BoolOp(op=ast.And(), values=[walrus, ast.Constant(value=None)])
        return ast.copy_location(new_node, visited)


def _programmatic_delegate_transform(
    repl_input_code: str,
    *,
    include_plan: bool = False,
    plan: str | None = None,
    include_instruction: bool = False,
) -> str:
    """Programmatically transform code to capture output and add tail-call delegation.

    Returns transformed code or original code if the code cannot be parsed.
    """
    try:
        tree = ast.parse(repl_input_code)
    except SyntaxError:
        # No transformation if syntax error so that syntax error can be reported back to LLM
        return repl_input_code

    # Replace print(...) with __captured_output += ...
    transformer = _PrintCaptureTransformer()
    tree = transformer.visit(tree)

    # Prepend __captured_output = ""
    init_node = ast.parse("__captured_output = ''").body[0]
    tree.body.insert(0, init_node)
    ast.fix_missing_locations(tree)

    transformed = ast.unparse(tree)

    repr_code = repr(repl_input_code)
    if include_plan:
        repr_plan = repr(plan)
        entry = (
            '{"plan": '
            + repr_plan
            + ', "code": '
            + repr_code
            + ', "output": __captured_output}'
        )
    else:
        entry = '{"code": ' + repr_code + ', "output": __captured_output}'
    invoke_args = (
        "    __task__,\n"
        "    return_type=__return_type__,\n"
        "    return_validator=__return_validator__,\n"
        "    history=history,\n"
    )
    # Note: both `prompt` and `task` must be excluded. `prompt` is the
    # agent-facing parameter name on `invoke_simple`/`invoke`, while `task`
    # is the internal parameter name on `_invoke()` (the positional argument
    # gets renamed deeper in the call chain). Either one could trigger a
    # "got multiple values for argument" TypeError on **unpacking. (And
    # `task` is the name of a campus library module the agent has in its
    # globals, so this collision actually triggers.)
    explicit_kwarg_keys = [
        "prompt",
        "task",
        "jaz",
        "return_type",
        "return_validator",
        "history",
    ]
    if include_instruction:
        invoke_args += "    instruction=" + repr(_DELEGATE_INSTRUCTION) + ",\n"
        explicit_kwarg_keys.append("instruction")
    # `input_descriptions={k: None for k in __delegate_kwargs__}` tells the
    # next sub-agent's prompt builder to skip rendering all the threaded
    # state in the prompt header. The values are still bound as Python
    # variables in the new REPL — only the prompt display is suppressed.
    # This keeps the prompt size bounded as state accumulates across many
    # delegations, while still letting the agent access the full values
    # programmatically.
    invoke_args += (
        "    input_descriptions={__k: None for __k in __delegate_kwargs__},\n"
    )
    invoke_args += "    **__delegate_kwargs__,\n"

    # Thread the agent's REPL state forward by passing every binding in
    # globals() as a kwarg to the next sub-agent. Filtered out:
    #   - Dunder names (`__foo__`): Python module dunders and the framework's
    #     magic vars (`__task__`, `__return_type__`, `__captured_output`, etc.)
    #     and the transformer's own scratch names.
    #   - Names of jaz.invoke parameters that are filled explicitly below
    #     (would otherwise raise TypeError on **unpacking).
    # TODO: Instead of silently filtering out, we should raise an error if
    # the user tries to assign to these reserved names.
    delegate_kwargs_setup = (
        "\n__delegate_kwargs__ = {\n"
        "    __k: __v for __k, __v in globals().items()\n"
        "    if not __k.startswith('__')\n"
        "    and __k not in " + repr(explicit_kwarg_keys) + "\n"
        "}\n"
    )

    suffix = (
        "\nhistory = globals().get('history', []) + ["
        + entry
        + "]\n"
        + delegate_kwargs_setup
        + "\nRETURN jaz.invoke(\n"
        + invoke_args
        + ")\n"
    )

    code = transformed + suffix
    return code


class TailCallDelegationClient(LLMClient):
    """LLM client that makes two calls per complete(): regular + delegation transform.

    The first call is a normal completion (delegated to *inner*).  If the
    response contains a ``<repl_input>`` block **and** ``model_config``
    contains a ``delegate_transform_template`` key, a second call is made
    with the conversation extended by the first response and a user message
    asking the model to rewrite its REPL input with tail-call delegation.

    The template is rendered using ``allow_raise`` and ``repl_history`` from
    ``model_config`` (defaulting to ``True``).

    Token counts and costs from both calls are aggregated in the returned
    ``LLMResponse``.
    """

    def __init__(self, inner: LLMClient | None = None) -> None:
        self.inner = inner or LiteLLMClient()

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        # Pop the transform keys so they aren't forwarded to the inner client
        template_name: str | None = kwargs.pop(
            _TEMPLATE_KEY, "delegate_transform.jinja2"
        )
        slim_transform: bool = kwargs.pop(_SLIM_TRANSFORM_KEY, False)
        programmatic: bool = kwargs.pop(_PROGRAMMATIC_KEY, False)
        include_plan: bool = kwargs.pop(_INCLUDE_PLAN_KEY, False)
        include_instruction: bool = kwargs.pop(_INCLUDE_INSTRUCTION_KEY, False)

        if include_plan and not programmatic:
            raise ValueError(
                f"{_INCLUDE_PLAN_KEY} requires {_PROGRAMMATIC_KEY} to be enabled"
            )
        if include_instruction and not programmatic:
            raise ValueError(
                f"{_INCLUDE_INSTRUCTION_KEY} requires {_PROGRAMMATIC_KEY} to be enabled"
            )

        # Call 1: regular completion
        response1 = self.inner.complete(model=model, messages=messages, **kwargs)

        # Skip transformation when there is nothing to rewrite, or the REPL
        # input already ends with RETURN or RAISE.
        if (
            (not template_name and not programmatic)
            or not response1.content
            or not _REPL_INPUT_RE.search(response1.content)
            or _repl_input_already_finishes(response1.content)
        ):
            return response1

        # Extract the <repl_input> block from the first response
        repl_input_match = _REPL_INPUT_RE.search(response1.content)
        assert repl_input_match is not None  # guarded above
        repl_input_block = repl_input_match.group(0)
        repl_input_code = repl_input_match.group(3)
        repl_input_lang = repl_input_match.group(1)

        # Programmatic mode: transform without a second LLM call
        if programmatic:
            plan: str | None = None
            if include_plan:
                plan_match = _PLAN_RE.search(response1.content)
                plan = plan_match.group(1).strip() if plan_match else None
            transformed_code = _programmatic_delegate_transform(
                repl_input_code,
                include_plan=include_plan,
                plan=plan,
                include_instruction=include_instruction,
            )
            content2 = f'<repl_input lang="{repl_input_lang}">\n{transformed_code}\n</repl_input>'
            return LLMResponse(
                content=content2,
                prompt_tokens=response1.prompt_tokens,
                completion_tokens=response1.completion_tokens,
                total_tokens=response1.total_tokens,
                cost=response1.cost,
                raw_response=response1.raw_response,
            )

        # Render the transform prompt from the template
        assert template_name is not None
        transform_prompt = _render_transform_prompt(template_name)

        if slim_transform:
            # Slim mode: send only the repl_input block + transform prompt
            # in a single user message (no message history).
            transform_messages: list[Any] = [
                {
                    "role": "user",
                    "content": f"{repl_input_block}\n\n{transform_prompt}",
                },
            ]
        else:
            # Full mode: send the entire message history + first response
            transform_messages = [
                *messages,
                {"role": "assistant", "content": response1.content},
                {"role": "user", "content": transform_prompt},
            ]

        response2 = self.inner.complete(
            model=model, messages=transform_messages, **kwargs
        )

        # Replace __repl_input__ with the repr of the original repl_input code.
        content2 = response2.content
        if content2 and "__repl_input__" in content2:
            content2 = content2.replace("__repl_input__", repr(repl_input_code))

        # Merge token counts / cost from both calls
        return LLMResponse(
            content=content2,
            prompt_tokens=_sum_optional(
                response1.prompt_tokens, response2.prompt_tokens
            ),
            completion_tokens=_sum_optional(
                response1.completion_tokens, response2.completion_tokens
            ),
            total_tokens=_sum_optional(response1.total_tokens, response2.total_tokens),
            cost=_sum_optional_float(response1.cost, response2.cost),
            raw_response=response2.raw_response,
        )

    @property
    def non_retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        return self.inner.non_retryable_exceptions

    def get_model(self, model_config: dict[str, Any]) -> str:
        return self.inner.get_model(model_config)

    def get_model_info(self, model_config: dict[str, object]) -> dict[str, Any]:
        return self.inner.get_model_info(model_config)
