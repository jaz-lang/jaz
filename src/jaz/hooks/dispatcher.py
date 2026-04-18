"""HookDispatcher orchestrates hooks and composes their effects into execution contexts."""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager

from jaz.exceptions import DuplicateReplInputError

from .base import Event, HaltableContext
from .context import _reset_hook_context, _set_hook_context, get_current_hooks
from .effects import (
    AddInstructionPrompt,
    AddLibrary,
    AddReplInput,
    AddSystemPrompt,
    Effect,
    HaltExecution,
    ModifyIterationLimit,
)
from .events import (
    BudgetUpdate,
    BudgetUpdateContext,
    InvokeContext,
    InvokeEnter,
    InvokeExit,
    InvokeSpan,
    LLMQueryContext,
    LLMQueryEnter,
    LLMQueryExit,
    LLMQuerySpan,
    LLMRetry,
    LLMRetryContext,
    Message,
    MessageContext,
    ReplExecutionContext,
    ReplExecutionEnter,
    ReplExecutionExit,
    ReplExecutionSpan,
    ReplIterationContext,
    ReplIterationEnter,
    ReplIterationExit,
    ReplIterationSpan,
)

logger = logging.getLogger(__name__)


class Hook(ABC):
    """Base class for hooks that respond to events by returning effects.

    Hooks are the extension points where users can observe and influence
    agent execution. Each hook receives events and returns a list of effects
    that express how it wants to modify behavior.

    Context Manager Protocol
    ------------------------
    Hooks support the context manager protocol, allowing them to be used with
    the `with` statement to automatically add themselves to the hook context:

        with MyHook():
            invoke(...)  # This invoke (and any nested invokes) will have MyHook active

    Multiple hooks can be activated using Python's native multi-context-manager syntax:

        with Hook1(), Hook2(), Hook3():
            invoke(...)  # All three hooks active

    Note: Activating a dynamic list of hooks (e.g., ``with_hooks(*hook_list)``) is not
    currently supported. Python's ``with`` statement requires a fixed number of context
    managers, and a naive wrapper cannot correctly propagate exceptions to each hook's
    ``__exit__`` method. Use the native syntax for static sets of hooks.

    Automatic Propagation
    ---------------------
    Once activated, hooks automatically propagate to:
    - Nested invoke() calls
    - Functions called within the hook context
    - Async code (contextvars are async-safe)
    - Threads (each thread gets its own context copy)

    Example
    -------
        from jaz.hooks import Hook, Event, Effect, ReplExecutionEnter, HaltExecution

        class MyHook(Hook):
            def on_event(self, event: Event) -> list[Effect]:
                # Hooks see all events - filter to what you care about
                if isinstance(event, ReplExecutionEnter) and event.iteration > 50:
                    return [HaltExecution(error=RuntimeError("Limit exceeded"))]
                return []

        # Use it
        with MyHook():
            result = invoke("task", return_type=str)

    See Also
    --------
    - disable_hook(): Temporarily disable specific hook types
    - isolated_hooks(): Clear all parent hooks for clean slate
    - hooks/README.md: Comprehensive documentation
    """

    @abstractmethod
    def on_event(self, event: Event) -> list[Effect]:
        """Process an event and return effects.

        Args:
            event: The event that occurred

        Returns:
            List of effects expressing how to influence execution.
            Return empty list if no effects needed.
        """
        ...

    def __enter__(self):
        """Enter hook context - adds this hook to active hooks.

        If the current context is the implicit default (PrintLogger only),
        we replace it with just this hook instead of adding to it.
        This ensures PrintLogger is only active when no user hooks are present.
        """
        from .context import HookContext

        current = get_current_hooks()

        # If current context is implicit default, start fresh without default hooks
        # TODO: Sometimes we don't want to replace the implicit default with a new hook
        # We should support the ability for a Hook subclass to specify whether or not
        # it should replace the implicit default (PrintLogger) when engaged.
        if current.is_implicit_default:
            new_context = HookContext(hooks=(self,))
        else:
            new_context = current.with_hook(self)

        self._token = _set_hook_context(new_context)
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Exit hook context - restores previous context."""
        _reset_hook_context(self._token)
        return False


class HookDispatcher:
    """Stateless orchestrator that reads hooks from context and composes their effects.

    The HookDispatcher is responsible for:
    1. Reading active hooks from the current context variable
    2. Emitting events to active hooks
    3. Collecting effects from all hooks
    4. Composing effects into a typed ExecutionContext with explicit rules
    5. Detecting and resolving conflicts (e.g., multiple halt requests)
    6. Managing span lifecycle for enter/exit event pairs

    Design principles:
    - HookDispatcher is stateless - all hook state comes from context variables
    - Hooks are called in registration order (first registered, first called)
    - Composition rules are explicit and documented
    - Type system prevents invalid operations (e.g., can't halt at InvokeExit)
    - Each event type has a specific ExecutionContext type

    Note: The HookDispatcher is a singleton. Use get_dispatcher() to access it.

    TODO: Add setting to change how dispatcher handles invalid effects (warn vs. error)
    """

    def emit(self, event: Event) -> list[Effect]:
        """Emit an event to all active hooks in current context and collect effects.

        Args:
            event: The event to emit

        Returns:
            List of all effects returned by hooks (in hook registration order)
        """
        effects: list[Effect] = []
        hook_ctx = get_current_hooks()

        for hook in hook_ctx.get_active_hooks():
            try:
                hook_effects = hook.on_event(event)
                effects.extend(hook_effects)
            except Exception as e:
                logger.error(
                    f"Hook {hook.__class__.__name__} raised exception "
                    f"processing {type(event).__name__}: {e}",
                    exc_info=True,
                )
                # Continue processing other hooks
        return effects

    # Context composition helpers (shared across context types)

    def _apply_halt_effects(self, ctx: HaltableContext, effects: list[Effect]) -> None:
        """Apply HaltExecution effects to context.

        Composition rule: If ANY hook requests halt, execution halts.
        All halt errors are collected; a single error is raised directly,
        multiple errors are raised as an ExceptionGroup.
        """
        halt_effects = [e for e in effects if isinstance(e, HaltExecution)]
        if halt_effects:
            ctx.halt_errors.extend(h.error for h in halt_effects)

    # Context-specific composition methods

    def _compose_repl_iteration(self, effects: list[Effect]) -> ReplIterationContext:
        """Compose effects for REPL iteration events.

        Composition rules:
        - Halt: ANY halt effect → halt (all errors collected) - halts entire invoke
        - Prompt additions: ALL are collected and sorted for deterministic ordering
        - REPL inputs: ALL are added (last write wins for same key)
        """
        ctx = ReplIterationContext()

        # Apply universal effects
        self._apply_halt_effects(ctx, effects)

        # Apply iteration-specific effects
        for effect in effects:
            match effect:
                case AddInstructionPrompt(text=text):
                    ctx.instruction_prompt_additions.append(text)

                case AddReplInput(key=key, value=value):
                    if key in ctx.repl_inputs:
                        raise DuplicateReplInputError(
                            f"REPL input '{key}' already exists in ReplIterationContext. "
                            f"Cannot add duplicate input."
                        )
                    ctx.repl_inputs[key] = value

        # Warn about invalid effects for REPL iteration
        for effect in effects:
            if isinstance(
                effect,
                ModifyIterationLimit,
            ):
                logger.warning(
                    f"{type(effect).__name__} not valid for ReplIteration, ignoring"
                )

        # Sort prompt additions for deterministic ordering regardless of hook order
        ctx.instruction_prompt_additions.sort()

        return ctx

    def _compose_repl_execution(self, effects: list[Effect]) -> ReplExecutionContext:
        """Compose effects for REPL execution events.

        Composition rules:
        - Halt: ANY halt effect → halt (all errors collected)
        - Iteration limit: MINIMUM of all ModifyIterationLimit effects
        - REPL inputs: ALL are added (last write wins for same key)
        """
        ctx = ReplExecutionContext()

        # Apply universal effects
        self._apply_halt_effects(ctx, effects)

        # Apply REPL-specific effects
        for effect in effects:
            match effect:
                case ModifyIterationLimit(new_max=new_max):
                    if ctx.max_iterations_override is None:
                        ctx.max_iterations_override = new_max
                    else:
                        # Take minimum
                        ctx.max_iterations_override = min(
                            ctx.max_iterations_override, new_max
                        )

                case AddReplInput(key=key, value=value):
                    if key in ctx.repl_inputs:
                        raise DuplicateReplInputError(
                            f"REPL input '{key}' already exists in ReplExecutionContext. "
                            f"Cannot add duplicate input."
                        )
                    ctx.repl_inputs[key] = value

                case AddInstructionPrompt():
                    logger.warning(
                        f"{type(effect).__name__} not valid for ReplExecution, ignoring"
                    )

        return ctx

    def _compose_llm_query(self, effects: list[Effect]) -> LLMQueryContext:
        """Compose effects for LLM query events.

        Composition rules:
        - Halt: ANY halt effect → halt (all errors collected)
        """
        ctx = LLMQueryContext()

        # Apply universal effects
        self._apply_halt_effects(ctx, effects)

        # Warn about invalid effects for LLM query
        for effect in effects:
            match effect:
                case AddInstructionPrompt() | ModifyIterationLimit():
                    logger.warning(
                        f"{type(effect).__name__} not valid for LLMQuery, ignoring"
                    )

        return ctx

    def _compose_invoke(self, effects: list[Effect]) -> InvokeContext:
        """Compose effects for invoke enter events.

        Composition rules:
        - Halt: ANY halt effect → halt (all errors collected)
        - Prompt additions: ALL are collected and sorted for deterministic ordering
        - Libraries: ALL are added
        - REPL inputs: ALL are added (last write wins for same key)
        """
        ctx = InvokeContext()

        # Apply universal effects
        self._apply_halt_effects(ctx, effects)

        # Apply invoke-specific effects
        for effect in effects:
            match effect:
                case AddInstructionPrompt(text=text):
                    ctx.instruction_prompt_additions.append(text)

                case AddSystemPrompt(text=text):
                    ctx.system_prompt_additions.append(text)

                case AddLibrary(library=lib):
                    ctx.additional_libraries.append(lib)

                case AddReplInput(key=key, value=value):
                    if key in ctx.repl_inputs:
                        raise DuplicateReplInputError(
                            f"REPL input '{key}' already exists in InvokeContext. "
                            f"Cannot add duplicate input."
                        )
                    ctx.repl_inputs[key] = value

        # Sort prompt additions for deterministic ordering regardless of hook order
        ctx.instruction_prompt_additions.sort()
        ctx.system_prompt_additions.sort()

        return ctx

    def _compose_message(self, effects: list[Effect]) -> MessageContext:
        """Compose effects for message events.

        This is a read-only context - messages are informational.
        No effects are allowed.
        """
        ctx = MessageContext()

        # Warn about invalid effects
        for effect in effects:
            if isinstance(effect, HaltExecution):
                logger.warning("Cannot halt on Message, ignoring HaltExecution")
            else:
                logger.warning(
                    f"{type(effect).__name__} not valid for Message, ignoring"
                )

        return ctx

    def _compose_budget_update(self, effects: list[Effect]) -> BudgetUpdateContext:
        """Compose effects for budget update events.

        This is a read-only context - budget updates are informational.
        Halting should happen at the next action event (e.g., ReplExecutionEnter).
        No effects are allowed.
        """
        ctx = BudgetUpdateContext()

        # Warn about invalid effects
        for effect in effects:
            if isinstance(effect, HaltExecution):
                logger.warning(
                    "Cannot halt on BudgetUpdate (halt at next action event instead), "
                    "ignoring HaltExecution"
                )
            else:
                logger.warning(
                    f"{type(effect).__name__} not valid for BudgetUpdate, ignoring"
                )

        return ctx

    def _compose_llm_retry(self, effects: list[Effect]) -> LLMRetryContext:
        """Compose effects for LLM retry events.

        This is a read-only context - LLM retries are informational.
        No effects are allowed.
        """
        ctx = LLMRetryContext()

        # Warn about invalid effects
        for effect in effects:
            if isinstance(effect, HaltExecution):
                logger.warning("Cannot halt on LLMRetry, ignoring HaltExecution")
            else:
                logger.warning(
                    f"{type(effect).__name__} not valid for LLMRetry, ignoring"
                )

        return ctx

    # Span-based context managers for enter/exit event pairs

    @contextmanager
    def span_repl_iteration(self, enter_event: ReplIterationEnter):
        """Context manager for REPL iteration span.

        Usage:
            enter_event = ReplIterationEnter(...)
            with dispatcher.span_repl_iteration(enter_event) as span:
                if span.ctx.should_halt:
                    _raise_halt_errors(span.ctx.halt_errors)  # ExceptionGroup if multiple

                result = do_one_repl_iteration(...)
                span.complete(result=result)

        The span must be completed before exiting, or an error is raised.
        """
        # Fire enter event and compose context
        effects = self.emit(enter_event)
        ctx = self._compose_repl_iteration(effects)

        # Create span
        span = ReplIterationSpan(ctx=ctx)

        try:
            yield span
        finally:
            # Ensure span was completed
            if not span.is_completed():
                logger.error(
                    f"ReplIteration span for iteration {enter_event.iteration} was not "
                    "completed. You must call span.complete(result=...) before exiting."
                )
                # Don't return - let exceptions propagate
            else:
                # Fire exit event only if span was completed
                exit_event = ReplIterationExit(
                    invoke_id=enter_event.invoke_id,
                    iteration=enter_event.iteration,
                    max_iterations=enter_event.max_iterations,
                    cur_recursion_depth=enter_event.cur_recursion_depth,
                    max_recursion_depth=enter_event.max_recursion_depth,
                    result=span.get_result(),
                )
                # Emit to hooks (they can observe but not influence)
                self.emit(exit_event)

    @contextmanager
    def span_repl_execution(self, enter_event: ReplExecutionEnter):
        """Context manager for REPL execution span.

        Usage:
            enter_event = ReplExecutionEnter(...)
            with dispatcher.span_repl_execution(enter_event) as span:
                if span.ctx.should_halt:
                    raise span.ctx.halt_errors[0]

                # Use span.ctx.prompt_warnings, etc.
                exec_result = repl.exec(...)
                span.complete(exec_result=exec_result)

        The span must be completed before exiting, or an error is raised.
        """
        # Fire enter event and compose context
        effects = self.emit(enter_event)
        ctx = self._compose_repl_execution(effects)

        # Create span
        span = ReplExecutionSpan(ctx=ctx)

        try:
            yield span
        finally:
            # Ensure span was completed
            if not span.is_completed():
                logger.error(
                    f"ReplExecution span for execution {enter_event.iteration} was not "
                    "completed. You must call span.complete(exec_result=...) before "
                    "exiting."
                )
                # Don't return - let exceptions propagate
            else:
                # Fire exit event only if span was completed
                exit_event = ReplExecutionExit(
                    invoke_id=enter_event.invoke_id,
                    iteration=enter_event.iteration,
                    max_iterations=enter_event.max_iterations,
                    exec_result=span.get_exec_result(),
                    cur_recursion_depth=enter_event.cur_recursion_depth,
                )
                # Emit to hooks (they can observe but not influence)
                self.emit(exit_event)

    @contextmanager
    def span_llm_query(self, enter_event: LLMQueryEnter):
        """Context manager for LLM query span.

        Usage:
            enter_event = LLMQueryEnter(...)
            with dispatcher.span_llm_query(enter_event) as span:
                if span.ctx.should_halt:
                    raise span.ctx.halt_errors[0]

                response = llm.query(messages, **base_config)
                span.complete(
                    response_content=response,
                    prompt_tokens=...,
                    completion_tokens=...,
                    cost=...
                )
        """
        # Fire enter event and compose context
        effects = self.emit(enter_event)
        ctx = self._compose_llm_query(effects)

        # Create span
        span = LLMQuerySpan(ctx=ctx)

        try:
            yield span
        finally:
            # Ensure span was completed
            if not span.is_completed():
                logger.error(
                    "LLMQuery span was not completed. "
                    "You must call span.complete(...) before exiting."
                )
                # Don't return - let exceptions propagate
            else:
                # Fire exit event only if span was completed
                exit_event = LLMQueryExit(
                    invoke_id=enter_event.invoke_id,
                    response_content=span.get_response_content(),
                    model=enter_event.model,
                    prompt_tokens=span.get_prompt_tokens(),
                    completion_tokens=span.get_completion_tokens(),
                    cost=span.get_cost(),
                    iteration=span.get_iteration(),
                    cur_recursion_depth=enter_event.cur_recursion_depth,
                    start_time=span.get_start_time(),
                    end_time=span.get_end_time(),
                )
                self.emit(exit_event)

    @contextmanager
    def span_invoke(self, enter_event: InvokeEnter):
        """Context manager for invoke span.

        Usage:
            enter_event = InvokeEnter(...)
            with dispatcher.span_invoke(enter_event) as span:
                if span.ctx.should_halt:
                    raise span.ctx.halt_errors[0]

                # Use span.ctx.instruction_prompt_additions, additional_libraries, etc.
                result = agent._invoke_internal(...)
                span.complete(result=result)
        """
        # Fire enter event and compose context
        effects = self.emit(enter_event)
        ctx = self._compose_invoke(effects)

        # Create span
        span = InvokeSpan(ctx=ctx)

        try:
            yield span
        finally:
            # Ensure span was completed
            if not span.is_completed():
                logger.error(
                    "Invoke span was not completed. "
                    "You must call span.complete(result=...) before exiting."
                )
                # Don't return - let exceptions propagate
            else:
                # Fire exit event only if span was completed
                exit_event = InvokeExit(
                    invoke_id=enter_event.invoke_id,
                    result=span.get_result(),
                    cur_recursion_depth=enter_event.cur_recursion_depth,
                    max_recursion_depth=enter_event.max_recursion_depth,
                )
                self.emit(exit_event)

    # Simple events (no enter/exit pair, just fire and compose)

    def process_message(self, event: Message) -> MessageContext:
        """Process a message event.

        Args:
            event: The message event to process

        Returns:
            MessageContext (read-only, can only record metrics)
        """
        effects = self.emit(event)
        return self._compose_message(effects)

    def process_budget_update(self, event: BudgetUpdate) -> BudgetUpdateContext:
        """Process a budget update event.

        Args:
            event: The budget update event to process

        Returns:
            BudgetUpdateContext (read-only, can only record metrics)
        """
        effects = self.emit(event)
        return self._compose_budget_update(effects)

    def process_llm_retry(self, event: LLMRetry) -> LLMRetryContext:
        """Process an LLM retry event.

        Args:
            event: The LLM retry event to process

        Returns:
            LLMRetryContext (read-only, can only record metrics)
        """
        effects = self.emit(event)
        return self._compose_llm_retry(effects)


# Singleton dispatcher instance
_dispatcher = HookDispatcher()


def get_dispatcher() -> HookDispatcher:
    """Get the singleton hook dispatcher instance.

    Since HookDispatcher is stateless (all hooks come from context variables),
    we use a single global instance.
    """
    return _dispatcher
