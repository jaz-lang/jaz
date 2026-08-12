"""Workflow replay hook that materializes Jaz execution as executable Python programs.

This hook captures the execution trace of a Jaz workflow and transforms it into
a structured Python package with:
- Hierarchical directory structure mirroring the call graph
- Executable programs with error handling logic
- LLM reasoning preserved as comments
"""

import ast
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryExit,
    REPLExecEnter,
    REPLExecExit,
)
from jaz.repl.types import Continue, ExecResult

# Global counter for naming invocations (shared across all contexts)
_global_counter: ContextVar[int] = ContextVar("workflow_replay_counter", default=0)


def _increment_counter() -> int:
    """Increment and return the new counter value."""
    new_val = _global_counter.get() + 1
    _global_counter.set(new_val)
    return new_val


def _write_indented(f: Any, code: str, indent: str) -> None:
    """Write each line of ``code`` prefixed with ``indent`` (blank lines stay blank)."""
    for line in code.split("\n"):
        f.write(f"{indent}{line}\n" if line.strip() else "\n")


@dataclass
class REPLIteration:
    """Record of a single REPL iteration."""

    iteration: int
    code: str
    exec_result: ExecResult
    llm_response: str | None = None  # Full LLM response including reasoning


@dataclass
class ChildInvocation:
    """Metadata about a child invocation."""

    invoke_id: str  # UUID from event
    counter_id: int  # Sequential counter for naming
    task_name: str
    function_name: str  # task_name + counter
    module_name: str  # function_name (without .py)
    has_children: bool  # Whether this invoke made nested calls


@dataclass
class InvocationContext:
    """Context for a single invoke execution."""

    invoke_id: str  # UUID from event
    parent_id: str | None  # Parent's invoke_id UUID (None for root)
    counter_id: int  # Sequential counter for naming
    task_name: str
    inputs: dict[str, object]
    depth: int
    output_path: Path  # Where this invocation's output will be written

    # Accumulated during execution
    repl_iterations: list[REPLIteration] = field(default_factory=list)
    child_invocations: list[ChildInvocation] = field(default_factory=list)

    # Per-turn record assembly (#481): the removed REPLIterationExit once carried a turn's
    # code + exec_result + llm_response in one event. Those now arrive on three separate
    # events — LLMQueryExit (response), REPLExecEnter (code), REPLExecExit (result) — so we
    # stash the first two here and write the REPLIteration record at REPLExecExit. A
    # parse-failure turn fires no REPLExec events, so it writes no record (the accepted
    # "no hook point on a parse-failure turn" tradeoff); its stashed response is overwritten
    # by the next turn's query.
    _pending_llm_response: str | None = None
    _pending_code: str | None = None

    def get_function_name(self) -> str:
        """Get the function name for this invocation."""
        return f"{self.task_name}_{self.counter_id}"

    def get_module_name(self) -> str:
        """Get the module/directory name for this invocation."""
        return f"{self.task_name}_{self.counter_id}"

    def is_root(self) -> bool:
        """Check if this is the root invocation."""
        return self.parent_id is None

    def has_children(self) -> bool:
        """Check if this invocation has child invocations."""
        return len(self.child_invocations) > 0


class WorkflowReplay(Hook):
    """Materialize a run as an executable Python package that replays it.

    Under ``with`` the generated program covers invokes nested inside. Passed positionally,
    only that invoke is materialized.

    Usage:
        with WorkflowReplay(output_dir="./workflows"):
            result = invoke(ReturnType(dict), task="Analyze data")

        # Creates: ./workflows/analyze_data_1/...

    Args:
        output_dir: Root directory for materialized workflows
    """

    # Reads the per-invoke ``task_name`` label off the blackboard (set by a
    # ``MetaData`` carrier hook); declaring it here lets that seed validate. Absent →
    # ``"main"``.
    blackboard_consumes = {"task_name": "Human label naming the replay dir/function."}

    def __init__(self, output_dir: str = "./workflows"):
        """Initialize the workflow replay hook.

        Args:
            output_dir: Directory where workflow packages will be written
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Map invoke_id (UUID) to InvocationContext
        self.contexts: dict[str, InvocationContext] = {}

    def on_any(self, event: Event) -> list:
        """Process events and build workflow materialization.

        Args:
            event: The hook event

        Returns:
            Empty list (this is a passive observer hook)
        """

        match event:
            case InvokeEnter(
                invoke_id=invoke_id,
                parent_invoke_id=parent_invoke_id,
                inputs=inputs,
                depth=depth,
            ):
                # task_name is per-invoke metadata carried on the blackboard (seeded
                # by a MetaData hook), not a core event field. Absent → "main".
                task_name = str(event.blackboard.get("task_name", "main"))
                # Increment global counter for naming this invocation
                counter_id = _increment_counter()

                # Get parent context using parent_invoke_id from event
                parent_ctx = (
                    self.contexts.get(parent_invoke_id) if parent_invoke_id else None
                )

                # Determine output path
                if parent_ctx is None:
                    # Root invocation: create directory under output_dir
                    output_path = self.output_dir / f"{task_name}_{counter_id}"
                else:
                    # Child invocation: will be under parent's directory
                    # We'll determine if it's a subdirectory or file later
                    output_path = parent_ctx.output_path / f"{task_name}_{counter_id}"

                # Create context
                ctx = InvocationContext(
                    invoke_id=invoke_id,
                    parent_id=parent_invoke_id,
                    counter_id=counter_id,
                    task_name=task_name,
                    # Full bound namespace: explicit inputs ∪ resolved scope (disjoint; #727 split
                    # them into separate InvokeEnter fields). Unlike the observability hooks
                    # (atif_trace / conversation_history / loggers / otel_tracing), which record the
                    # two as SEPARATE provenance channels, WorkflowReplay is a *codegen* hook: it
                    # lowers this namespace into the replay function's PARAMETERS and reconstructs it
                    # in the __main__ call. Replay reproduces the namespace the agent saw, so it needs
                    # every name regardless of provenance — hence the deliberate merge here (do NOT
                    # "fix" it into a split; splitting would require emitting `with jaz.scope(...)`
                    # wrappers in the generated code, a separate codegen change).
                    inputs={**event.scope, **inputs},
                    depth=depth,
                    output_path=output_path,
                )

                # Store context by invoke_id
                self.contexts[invoke_id] = ctx

                # Register with parent if not root
                if parent_ctx is not None:
                    parent_ctx.child_invocations.append(
                        ChildInvocation(
                            invoke_id=invoke_id,
                            counter_id=counter_id,
                            task_name=task_name,
                            function_name=ctx.get_function_name(),
                            module_name=ctx.get_module_name(),
                            has_children=False,  # Will be updated at InvokeExit
                        )
                    )

                # Create file and write function header immediately
                program_file = self._get_program_file(ctx)
                self._write_function_header(ctx, program_file)

            case LLMQueryExit(invoke_id=invoke_id, response=response):
                # Stash this turn's LLM response; the record is written at REPLExecExit
                # (below), where code + exec_result also become available.
                ctx = self.contexts.get(invoke_id)
                if ctx:
                    ctx._pending_llm_response = response.content

            case REPLExecEnter(invoke_id=invoke_id, code=code):
                # Stash this turn's code (only present on the runnable-code branch).
                ctx = self.contexts.get(invoke_id)
                if ctx:
                    ctx._pending_code = code

            case REPLExecExit(invoke_id=invoke_id, iteration=i, exec_result=result):
                # Assemble + write this turn's record now that all three parts are known
                # (response from the LLMQueryExit above, code from REPLExecEnter, result
                # here). REPLExec only fires on the runnable-code branch, so a parse-failure
                # turn writes no record (accepted #481 tradeoff — see InvocationContext).
                ctx = self.contexts.get(invoke_id)
                if not ctx:
                    return []

                repl_iteration = REPLIteration(
                    iteration=i,
                    code=ctx._pending_code or "",
                    exec_result=result,
                    llm_response=ctx._pending_llm_response,
                )
                ctx.repl_iterations.append(repl_iteration)
                ctx._pending_code = None
                ctx._pending_llm_response = None

                # Write this iteration to file
                self._write_iteration_to_file(ctx, repl_iteration)

            case InvokeExit(invoke_id=invoke_id):
                # Get context from dict
                ctx = self.contexts.get(invoke_id)
                if not ctx:
                    return []

                # Update parent's child info about whether we have children
                if ctx.parent_id is not None:
                    parent_ctx = self.contexts.get(ctx.parent_id)
                    if parent_ctx:
                        for child in parent_ctx.child_invocations:
                            if child.invoke_id == ctx.invoke_id:
                                child.has_children = ctx.has_children()
                                break

                # Finalize the workflow (add imports, etc.)
                self._finalize_workflow(ctx)

        return []

    def _get_program_file(self, ctx: InvocationContext) -> Path:
        """Get the program file path for this context.

        Args:
            ctx: The invocation context

        Returns:
            Path to the program file (either .py or __init__.py)
        """
        # We won't know if it has children until InvokeExit, so for now
        # assume it's a leaf and create a .py file. We'll move it later if needed.
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        return ctx.output_path.with_suffix(".py")

    def _write_iteration_to_file(
        self, ctx: InvocationContext, iteration: REPLIteration
    ) -> None:
        """Write a single REPL iteration to the program file.

        Args:
            ctx: The invocation context
            iteration: The REPL iteration to write
        """
        program_file = self._get_program_file(ctx)

        # Append iteration code to file
        with program_file.open("a") as f:
            if iteration.code:
                # Add LLM reasoning as comment if available
                if iteration.llm_response:
                    # Extract reasoning from LLM response (simplified)
                    # TODO: Parse reasoning more carefully
                    f.write(f"    # Iteration {iteration.iteration}\n")

                # Get the code, possibly with RETURN replaced
                code = iteration.code

                # Handle different execution result types
                match iteration.exec_result:
                    case Continue(exception=exception) if exception is not None:
                        # Recoverable error the agent observed and continued past: wrap the
                        # code in a try/except that swallows it, so the replayed program
                        # likewise continues. This is the ONLY case that wraps. Guard on
                        # `exception is not None`: a clean `Continue` (exception=None) must
                        # fall through to write-as-is, not get a bogus `except NoneType`
                        # wrapper (#566 step C).
                        exc_type = type(exception).__name__
                        f.write("    try:\n")
                        _write_indented(f, code, "        ")
                        f.write(f"    except {exc_type}:\n")
                        f.write("        pass\n")

                    case _:
                        # Everything else — clean `Continue`, `Return`, and terminal
                        # `Raise` — is written as-is. Crucially, a terminal `Raise` is NOT
                        # swallowed (#710): writing it as-is lets the replay re-raise and
                        # terminate where the original did. The `RETURN`/`RAISE` sentinels
                        # in the code are lowered to `return`/`raise` at finalization.
                        _write_indented(f, code, "    ")

    def _write_function_header(
        self, ctx: InvocationContext, program_file: Path
    ) -> None:
        """Write the function signature and docstring to the file.

        Args:
            ctx: The invocation context
            program_file: Path to the program file
        """
        lines = []

        # Generate parameter list from inputs with type hints
        if ctx.inputs:
            param_parts = []
            for name, value in ctx.inputs.items():
                # Get type hint from the value
                type_hint = type(value).__name__
                param_parts.append(f"{name}: {type_hint}")
            params = ", ".join(param_parts)
        else:
            params = ""

        # No docstring: the invoke's prompt is not a distinct field (#538) and the `task`
        # input — the only prompt-like candidate — is neither guaranteed to exist nor
        # contractually a string suitable as a docstring, so we omit it rather than emit a
        # misleading or malformed one. The materialized body is the REPL iterations below.
        # No return annotation: the return type is a ReturnType(...) hook now, not a per-invoke
        # field this codegen hook can see (#568).
        lines.append(f"def {ctx.get_function_name()}({params}):")
        lines.append("")

        program_file.write_text("\n".join(lines))

    def _finalize_workflow(self, ctx: InvocationContext) -> None:
        """Finalize the workflow by adding imports and handling directory structure.

        Args:
            ctx: The invocation context to finalize
        """
        program_file = self._get_program_file(ctx)

        # If this invocation has children, we need to restructure as a directory
        if ctx.has_children():
            # Create directory
            ctx.output_path.mkdir(parents=True, exist_ok=True)
            # Move the file to __init__.py
            new_file = ctx.output_path / "__init__.py"
            if program_file.exists():
                program_file.rename(new_file)
            program_file = new_file

        # Read current content
        if program_file.exists():
            current_content = program_file.read_text()
        else:
            current_content = ""

        # Generate imports
        import_lines = []
        if ctx.child_invocations:
            for child in ctx.child_invocations:
                import_lines.append(
                    f"from .{child.module_name} import {child.function_name}"
                )
            import_lines.append("")

            # Add invoke replacement boilerplate
            import_lines.append("# Global counter for invoke replacement")
            import_lines.append("_invoke_functions = [")
            for child in ctx.child_invocations:
                import_lines.append(f"    {child.function_name},")
            import_lines.append("]")
            import_lines.append("_invoke_idx = 0")
            import_lines.append("")
            import_lines.append("def _get_next_invoke_fn():")
            import_lines.append("    global _invoke_idx")
            import_lines.append("    fn = _invoke_functions[_invoke_idx]")
            import_lines.append("    _invoke_idx += 1")
            import_lines.append("    return fn")
            import_lines.append("")

        # Lower the RETURN/RAISE REPL sentinels to real Python FIRST — both are invalid
        # Python as-is and must go before the AST parse in _replace_invoke_calls (mirrors
        # python_repl.py lowering both). Without the RAISE pass, a terminal `RAISE ...`
        # iteration leaves a literal `RAISE ...` that fails ast.parse and silently aborts
        # finalization for the invoke (#710). Crude str.replace (not `\bRAISE\b`) matches
        # the existing RETURN handling.
        current_content = current_content.replace("RETURN", "return")
        current_content = current_content.replace("RAISE", "raise")

        # Replace jaz.invoke() calls in the function body if we have children
        if ctx.child_invocations:
            current_content = self._replace_invoke_calls_in_file(current_content)

        # Combine imports and content
        final_content = "\n".join(import_lines) + current_content

        # Add __main__ block for standalone execution
        main_block = self._generate_main_block(ctx)
        final_content += "\n" + main_block

        # Write final content
        program_file.write_text(final_content)

        # If this is a package (has children), also create __main__.py for module execution
        if ctx.has_children():
            self._create_main_py(ctx)

    def _generate_main_block(self, ctx: InvocationContext) -> str:
        """Generate a __main__ block for standalone execution.

        Args:
            ctx: The invocation context

        Returns:
            The __main__ block code as a string
        """
        lines = []
        lines.append("")
        lines.append('if __name__ == "__main__":')

        # Generate function call with the original inputs
        func_name = ctx.get_function_name()
        if ctx.inputs:
            # Format the inputs as arguments
            args = []
            for name, value in ctx.inputs.items():
                args.append(f"{name}={repr(value)}")
            args_str = ", ".join(args)
            lines.append(f"    result = {func_name}({args_str})")
        else:
            lines.append(f"    result = {func_name}()")

        lines.append("    print(result)")

        return "\n".join(lines) + "\n"

    def _create_main_py(self, ctx: InvocationContext) -> None:
        """Create a __main__.py file for package execution with python -m.

        Args:
            ctx: The invocation context
        """
        main_py_path = ctx.output_path / "__main__.py"

        lines = []
        lines.append(f'"""Execute the {ctx.get_function_name()} workflow."""')
        lines.append("")
        lines.append(f"from . import {ctx.get_function_name()}")
        lines.append("")
        lines.append('if __name__ == "__main__":')

        # Generate function call with the original inputs
        func_name = ctx.get_function_name()
        if ctx.inputs:
            # Format the inputs as arguments
            args = []
            for name, value in ctx.inputs.items():
                args.append(f"{name}={repr(value)}")
            args_str = ", ".join(args)
            lines.append(f"    result = {func_name}({args_str})")
        else:
            lines.append(f"    result = {func_name}()")

        lines.append("    print(result)")
        lines.append("")

        main_py_path.write_text("\n".join(lines))

    def _replace_invoke_calls_in_file(self, source_code: str) -> str:
        """Replace jaz.invoke() calls in the source code.

        Args:
            source_code: The source code with potential jaz.invoke() calls

        Returns:
            Modified source code with invoke calls replaced
        """
        # This is a simplified version - just replace the invoke calls
        # The original _replace_invoke_calls method had AST manipulation
        # For now, let's keep the same logic
        return self._replace_invoke_calls(source_code)

    def _replace_invoke_calls(self, source_code: str) -> str:
        """Replace jaz.invoke() calls with _get_next_invoke_fn()() calls.

        This uses AST manipulation to find and replace invoke calls while
        preserving all other code structure.

        Args:
            source_code: Original Python source code

        Returns:
            Modified source code with invoke calls replaced
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # If parsing fails, return original code
            return source_code

        # Transformer to replace jaz.invoke calls
        class InvokeReplacer(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call) -> ast.AST:
                # Check if this is a jaz.invoke() call
                is_jaz_invoke = False

                # Match: jaz.invoke(...)
                if isinstance(node.func, ast.Attribute):
                    if (
                        node.func.attr == "invoke"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "jaz"
                    ):
                        is_jaz_invoke = True

                if is_jaz_invoke:
                    # Replace with: _get_next_invoke_fn()(**data_inputs)
                    #
                    # The recorded workflow already captures the original return value, so we
                    # strip every "control" arg from the call and forward ONLY the data inputs.
                    # The public signature is `invoke(ReturnType(...), *local_hooks, task=...,
                    # config_override=None, **inputs)`: data inputs are keyword-only, so EVERY
                    # positional is a control hook (ReturnType / ConfigOverride / MetaData / …)
                    # and none should be forwarded — drop them all. (The former `node.args[2:]`
                    # was a relic of the 2-positional `invoke(prompt, config_override)` shape: it
                    # under-stripped a lone leading hook and forwarded any positional past index
                    # 2 — e.g. `invoke(H1(), H2(), H3(), task=...)` leaked `H3` into the replay.)
                    new_args: list[ast.expr] = []
                    _CONTROL_KWARGS = {
                        "return_type",
                        "prompt",
                        "task",
                        "task_name",
                        "config_override",
                    }
                    new_keywords = [
                        kw for kw in node.keywords if kw.arg not in _CONTROL_KWARGS
                    ]

                    # Create: _get_next_invoke_fn()
                    get_fn_call = ast.Call(
                        func=ast.Name(id="_get_next_invoke_fn", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    )

                    # Create: _get_next_invoke_fn()(new_args, new_keywords)
                    new_call = ast.Call(
                        func=get_fn_call, args=new_args, keywords=new_keywords
                    )

                    return ast.copy_location(new_call, node)

                # Recursively visit child nodes
                return self.generic_visit(node)

        # Apply transformation
        replacer = InvokeReplacer()
        new_tree = replacer.visit(tree)
        ast.fix_missing_locations(new_tree)

        # Convert back to source code
        try:
            return ast.unparse(new_tree)
        except Exception:
            # If unparsing fails, return original
            return source_code

    def teardown(self, exc: BaseException | None = None) -> None:
        """Reset counters and clear per-workflow state."""
        # Reset global counter for next workflow
        _global_counter.set(0)

        # Clear the contexts dict
        self.contexts.clear()


#: Deprecated alias for the pre-rename spelling — see the rationale block in
#: ``jaz/hooks/__init__.py``. Every renamed hook carries this alias at its definition
#: site so the deep-path import keeps working and so the alias map stays checkable
#: (``test_every_renamed_hook_has_an_alias``).
WorkflowReplayHook = WorkflowReplay
