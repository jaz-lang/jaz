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

from typing_extensions import TypeForm

from jaz.hooks.base import Event
from jaz.hooks.dispatcher import Hook
from jaz.hooks.events import (
    InvokeEnter,
    InvokeExit,
    ReplIterationExit,
)
from jaz.repl.types import ErrorResult, ExecResult, RaiseResult, ReturnResult

# Global counter for naming invocations (shared across all contexts)
_global_counter: ContextVar[int] = ContextVar("workflow_replay_counter", default=0)


def _increment_counter() -> int:
    """Increment and return the new counter value."""
    new_val = _global_counter.get() + 1
    _global_counter.set(new_val)
    return new_val


@dataclass
class ReplIteration:
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
    prompt: str
    return_type: TypeForm[Any] | None
    has_children: bool  # Whether this invoke made nested calls


@dataclass
class InvocationContext:
    """Context for a single invoke execution."""

    invoke_id: str  # UUID from event
    parent_id: str | None  # Parent's invoke_id UUID (None for root)
    counter_id: int  # Sequential counter for naming
    task_name: str
    prompt: str
    return_type: TypeForm[Any] | None
    inputs: dict[str, object]
    recursion_depth: int
    output_path: Path  # Where this invocation's output will be written

    # Accumulated during execution
    repl_iterations: list[ReplIteration] = field(default_factory=list)
    child_invocations: list[ChildInvocation] = field(default_factory=list)

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


class WorkflowReplayHook(Hook):
    """Hook that materializes Jaz workflows as executable Python programs.

    This hook tracks all invocations and REPL iterations, then generates
    structured Python packages that replay the workflow execution.

    Usage:
        with WorkflowReplayHook(output_dir="./workflows"):
            result = invoke("Analyze data", return_type=dict)

        # Creates: ./workflows/analyze_data_1/...

    Args:
        output_dir: Root directory for materialized workflows
    """

    def __init__(self, output_dir: str = "./workflows"):
        """Initialize the workflow replay hook.

        Args:
            output_dir: Directory where workflow packages will be written
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Map invoke_id (UUID) to InvocationContext
        self.contexts: dict[str, InvocationContext] = {}

    def on_event(self, event: Event) -> list:
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
                prompt=prompt,
                task_name=task_name,
                return_type=return_type,
                inputs=inputs,
                cur_recursion_depth=depth,
            ):
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
                    prompt=prompt,
                    return_type=return_type,
                    inputs=inputs,
                    recursion_depth=depth,
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
                            prompt=prompt,
                            return_type=return_type,
                            has_children=False,  # Will be updated at InvokeExit
                        )
                    )

                # Create file and write function header immediately
                program_file = self._get_program_file(ctx)
                self._write_function_header(ctx, program_file)

            case ReplIterationExit(invoke_id=invoke_id, iteration=i, result=result):
                # Write this iteration to file immediately
                ctx = self.contexts.get(invoke_id)
                if not ctx:
                    return []

                # Create ReplIteration record
                repl_iteration = ReplIteration(
                    iteration=i,
                    code=result.code or "",
                    exec_result=result.exec_result,
                    llm_response=result.response_content,
                )
                ctx.repl_iterations.append(repl_iteration)

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
        self, ctx: InvocationContext, iteration: ReplIteration
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
                    case (
                        ErrorResult(exception=exception)
                        | RaiseResult(exception=exception)
                    ):
                        # Wrap in try-except block
                        exc_type = type(exception).__name__
                        f.write("    try:\n")
                        for line in code.split("\n"):
                            if line.strip():
                                f.write(f"        {line}\n")
                            else:
                                f.write("\n")
                        f.write(f"    except {exc_type}:\n")
                        f.write("        pass\n")

                    case ReturnResult():
                        # Write code as-is, RETURN will be replaced at InvokeExit
                        for line in code.split("\n"):
                            if line.strip():
                                f.write(f"    {line}\n")
                            else:
                                f.write("\n")

                    case _:
                        # ExecuteResult or other - write code as-is
                        for line in code.split("\n"):
                            if line.strip():
                                f.write(f"    {line}\n")
                            else:
                                f.write("\n")

    def _write_function_header(
        self, ctx: InvocationContext, program_file: Path
    ) -> None:
        """Write the function signature and docstring to the file.

        Args:
            ctx: The invocation context
            program_file: Path to the program file
        """
        lines = []

        # Generate function signature
        return_type_str = ""
        if ctx.return_type:
            type_name = getattr(ctx.return_type, "__name__", None)
            return_type_str = f" -> {type_name or ctx.return_type}"

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

        lines.append(f"def {ctx.get_function_name()}({params}){return_type_str}:")
        lines.append('    """')
        lines.append(f"    {ctx.prompt}")
        lines.append('    """')
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

        # Replace all RETURN with return in the final code FIRST
        # (Must be done before AST parsing for invoke replacement, since RETURN is invalid Python)
        current_content = current_content.replace("RETURN", "return")

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
                    # Replace with: _get_next_invoke_fn()(args, kwargs)
                    # Original: jaz.invoke(prompt, arg1, arg2, return_type, **kwargs)
                    # New: _get_next_invoke_fn()(arg1, arg2, **kwargs)
                    # We need to skip the prompt and return_type arguments

                    # Extract actual arguments (skip prompt and return_type)
                    # Typically: jaz.invoke(prompt, *args, return_type, **kwargs)
                    new_args = []
                    new_keywords = []

                    # Skip first argument (prompt)
                    if len(node.args) > 1:
                        new_args = node.args[1:]

                    # Filter out return_type, prompt, and task_name from keywords
                    for kw in node.keywords:
                        if kw.arg not in ("return_type", "prompt", "task_name"):
                            new_keywords.append(kw)

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

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context and reset counters."""
        # Reset global counter for next workflow
        _global_counter.set(0)

        # Clear the contexts dict
        self.contexts.clear()

        # Call parent to reset hook context
        return super().__exit__(exc_type, exc_value, traceback)
