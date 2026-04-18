# Hook System

The hook system provides a type-safe, composable way to observe and influence agent execution through events, effects, and contexts.

## Design Philosophy

The hook system was redesigned around **context-based hook management** to achieve several key goals:

### Why Context Variables?

1. **Automatic Propagation**: Hooks naturally propagate to nested `invoke()` calls without explicit parameter passing
2. **Scope-Based Lifecycle**: Hooks are active exactly within their `with` block - no manual cleanup needed
3. **Thread-Safe**: Python's `contextvars` module provides thread-safe and async-safe context management
4. **Composability**: Easy to add, remove, or temporarily disable hooks in nested scopes
5. **Cleaner API**: No need to pass `dispatcher` parameters through function calls

### Core Principles

1. **Hooks are order-independent** - Can run in parallel, composition is deterministic
2. **Type system prevents invalid operations** - Can't halt at InvokeExit, can't modify prompt in LLMQuery
3. **Composition rules are explicit** - No hidden behavior, easy to reason about
4. **Context managers for lifecycle** - Pythonic way to manage hook activation/deactivation
5. **Stateless dispatcher** - All state lives in context variables, dispatcher just orchestrates

### Design Trade-offs

**What we gained:**
- Automatic hook propagation to nested invokes
- No dispatcher parameter pollution
- Natural scoping via `with` statements
- Thread and async safety

**What we gave up:**
- Per-event-type hook registration (now all hooks see all events)
- Explicit hook ordering control (order is implicit via nesting)
- Multiple dispatchers (now singleton - all state in context)

The trade-offs were worth it for the cleaner API and better composability.

## Key Concepts

### Events
Discrete points in the agent's execution lifecycle where hooks can observe and influence behavior.

Examples:
- `ReplIterationEnter` - Before executing REPL code
- `LLMQueryEnter` - Before making an LLM API call
- `InvokeEnter` - When `jaz.invoke()` is called
- `Message` - When a message is processed
- `BudgetUpdate` - When cost budget status changes

### Effects
Typed outputs from hooks that express how they want to influence execution. Effects are order-independent and composed by the dispatcher.

Examples:
- `HaltExecution` - Stop execution immediately
- `AddInstructionPrompt` - Add instruction text to the agent's prompt
- `ModifyIterationLimit` - Change max REPL iterations
- `OverrideLLMClient` - Override the LLM model for a query

### Contexts
Typed contracts between the dispatcher and agent containing composed effects. Each event type has a specific context type.

Examples:
- `ReplIterationContext` - Can halt, modify prompts, adjust iterations
- `LLMQueryContext` - Can halt, override model parameters
- `MessageContext` - Read-only, can only record metrics
- `InvokeContext` - Can halt, add prompts, add libraries

### Hooks
Extension points that process events and return effects. Hooks inherit from the `Hook` base class and support the context manager protocol.

```python
from jaz.dispatcher import Hook, Event, Effect

class MyHook(Hook):
    def on_event(self, event: Event) -> list[Effect]:
        """Process event and return effects."""
        # Hooks see all events - filter to what you care about
        if isinstance(event, ReplIterationEnter):
            return [AddInstructionPrompt("Be concise.")]
        return []
```

### HookDispatcher
Stateless singleton that orchestrates hooks by:
1. Reading active hooks from the current context variable
2. Emitting events to all active hooks
3. Collecting effects from all hooks
4. Applying composition rules to create typed ExecutionContext

## Context-Based Hook Management

### Basic Usage

Hooks are activated using Python's `with` statement:

```python
from jaz import invoke
from jaz.hooks import PrintLogger, BudgetControlHook
from jaz.dispatcher import disable_hook, isolated_hooks
import logging

# Simple hook activation
with PrintLogger(level=logging.INFO):
    result = invoke("Calculate 2+2", return_type=int)
    # PrintLogger is active during this invoke

# Nested hooks - both active
with PrintLogger(level=logging.INFO):
    with BudgetControlHook(cost_tracker):
        result = invoke("Complex task", return_type=str)
        # Both PrintLogger and BudgetControlHook are active

# Disable specific hooks
with PrintLogger(level=logging.INFO):
    invoke("task 1", return_type=str)  # PrintLogger active

    with disable_hook(PrintLogger):
        invoke("task 2", return_type=str)  # PrintLogger disabled

    invoke("task 3", return_type=str)  # PrintLogger active again

# Isolated context (no parent hooks)
with PrintLogger():
    with isolated_hooks():
        invoke("clean slate", return_type=str)  # No hooks at all
```

### Automatic Propagation

Hooks automatically propagate to nested `invoke()` calls via `contextvars`:

```python
from jaz import invoke
from jaz.hooks import PrintLogger

with PrintLogger(level=logging.INFO):
    # This invoke has PrintLogger
    result = invoke(
        """
        Call invoke again with prompt: 'What is 5 * 5?'
        """,
        return_type=int
    )
    # The nested invoke() call also has PrintLogger automatically!
```

This works across any level of nesting - hooks propagate through:
- Nested `invoke()` calls
- Function calls within the hook context
- Async functions (contextvars are async-safe)
- Threads (each thread gets its own context copy)

### Default Global Context

By default, all invokes have `PrintLogger(level=logging.INFO)` active. To start with a clean slate:

```python
from jaz.dispatcher import isolated_hooks

with isolated_hooks():
    # No default PrintLogger - completely clean context
    invoke("silent task", return_type=str)
```

### Helper Functions

```python
from jaz.dispatcher import disable_hook, isolated_hooks

# Disable specific hook types
with disable_hook(PrintLogger, MetricsHook):
    invoke(...)  # These hook types are disabled

# Activate multiple hooks with native Python syntax
with Hook1(), Hook2(), Hook3():
    invoke(...)  # All three hooks active

# Isolated context
with isolated_hooks():
    invoke(...)  # No parent hooks, clean slate
```

## Architecture

### Context Variable Flow

```
┌─────────────────────────────────────────────────────────┐
│ Global Default Context                                   │
│ hooks: (PrintLogger(INFO),)                             │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────┐
           │ with MyHook():              │
           │ hooks: (PrintLogger, MyHook)│
           └─────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────────────────────────┐
              │ invoke() calls                           │
              │ HookDispatcher reads current context     │
              │ and emits to active hooks                │
              └──────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Nested invoke()      │
              │ Same hooks active!   │
              └──────────────────────┘
```

### Event Flow

```
┌─────────────┐
│   Agent     │
│             │
│  Emits ───────────┐
│  Events     │     │
└─────────────┘     │
                    ▼
              ┌────────────────────┐
              │  HookDispatcher    │ ◄── Reads hooks from
              │  (Singleton)       │     context variable
              │                    │
              │  emit() ───────────┼───► Hook 1 ───► [Effect, Effect]
              │                    │
              │                    ┼───► Hook 2 ───► [Effect]
              │                    │
              │                    │◄─── Hook 3 ───► [Effect, Effect]
              └────────────────────┘
                    │
                    │ Composition Rules:
                    │ • ANY halt → halt (all errors collected)
                    │ • ALL prompts → concatenate
                    │ • MIN iteration limit
                    │ • ALL metrics → record
                    │
                    ▼
          ┌──────────────────────┐
          │ ExecutionContext     │
          │ (Typed)              │
          │                      │
          │ • halt_errors: list  │
          │ • should_halt: bool  │
          │ • instruction_prompt │
          │ • max_iterations...  │
          └──────────────────────┘
                    │
                    ▼
              ┌─────────────┐
              │   Agent     │
              │  Uses       │
              │  Context    │
              └─────────────┘
```

## Usage Examples

### 1. Create a Hook

```python
from jaz.dispatcher import Hook, Event, Effect, ReplIterationEnter, HaltExecution, AddInstructionPrompt
from jaz.budget import CostTracker

class BudgetControlHook(Hook):
    """Hook that enforces budget limits."""

    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker

    def on_event(self, event: Event) -> list[Effect]:
        effects = []

        # Only act on REPL iteration events
        if isinstance(event, ReplIterationEnter):
            if self.cost_tracker.is_budget_exhausted():
                effects.append(AddInstructionPrompt(
                    text="Budget exhausted! Exit immediately.",
                ))

                if self.cost_tracker.is_buffer_exhausted():
                    effects.append(HaltExecution(
                        error=BudgetExhaustedError(),
                    ))

        return effects
```

### 2. Use Hook as Context Manager

```python
from jaz import invoke

# Activate hook for specific scope
with BudgetControlHook(cost_tracker):
    result = invoke("Expensive task", return_type=str)
    # BudgetControlHook is active during this invoke
```

### 3. Combine Multiple Hooks

```python
from jaz.hooks import PrintLogger, SafetyFilterHook
import logging

# Both hooks active
with PrintLogger(level=logging.INFO):
    with BudgetControlHook(cost_tracker):
        with SafetyFilterHook():
            result = invoke("Task with all protections", return_type=str)
```

### 4. Conditional Hook Activation

```python
def my_function(enable_logging: bool = True):
    if enable_logging:
        with PrintLogger(level=logging.INFO):
            return invoke("task", return_type=str)
    else:
        with isolated_hooks():  # No default PrintLogger
            return invoke("silent task", return_type=str)
```

### 5. Hook Disabling Pattern

```python
# Override PrintLogger level temporarily
with PrintLogger(level=logging.INFO):
    invoke("normal logging", return_type=str)

    # Temporarily use DEBUG level
    with disable_hook(PrintLogger):
        with PrintLogger(level=logging.DEBUG):
            invoke("verbose logging", return_type=str)

    # Back to INFO level
    invoke("normal logging again", return_type=str)
```

### 6. Use in Agent (Internal)

```python
from jaz.dispatcher import get_dispatcher

# In Agent.do_one_repl_iteration()
dispatcher = get_dispatcher()

with dispatcher.span_repl_iteration(
    iteration=i,
    max_iterations=max_iterations,
    code=code,
    cur_recursion_depth=cur_recursion_depth
) as span:
    # Check if hooks want to halt
    if span.ctx.should_halt:
        _raise_halt_errors(span.ctx.halt_errors)  # ExceptionGroup if multiple

    # Apply prompt modifications from hooks
    if span.ctx.instruction_prompt_additions:
        prompt += "\n" + "\n".join(span.ctx.instruction_prompt_additions)

    # Use iteration limit override if provided
    max_iters = span.ctx.max_iterations_override or max_iterations

    # Execute REPL code
    exec_result = repl.exec(code, ...)

    # Complete the span
    span.complete(exec_result=exec_result)
```

## Composition Rules

The dispatcher applies explicit composition rules when combining effects from multiple hooks:

### Halting
- **Rule**: If ANY hook returns `HaltExecution`, execution halts
- **Multiple halts**: All errors are collected. A single error is raised directly; multiple errors are raised as an `ExceptionGroup`

### Prompt Modifications
- **Rule**: ALL `AddInstructionPrompt` effects are concatenated
- **Order**: Concatenated in hook registration order
- **Example**: Budget warning + safety note both appear in prompt

### Iteration Limits
- **Rule**: MINIMUM of all `ModifyIterationLimit` effects
- **Rationale**: Most restrictive limit wins (safety-first)
- **Example**: Hook A says max=10, Hook B says max=5 → use 5

### Model Overrides
- **Rule**: LAST `OverrideLLMClient` effect wins
- **Order**: Determined by hook registration order
- **Note**: Last hook registered has final say

## Type Safety

The dispatcher uses typed contexts to prevent invalid operations:

```python
# Type checker knows ctx is ReplIterationContext!
with dispatcher.span_repl_iteration(...) as span:
    span.ctx.instruction_prompt_additions  # ✓ Exists
    span.ctx.llm_client_override  # ✗ Type error - doesn't exist on ReplIterationContext

# Type checker knows ctx is LLMQueryContext!
with dispatcher.span_llm_query(...) as span:
    span.ctx.llm_client_override  # ✓ Exists
    span.ctx.model_config_overrides  # ✓ Exists
    span.ctx.instruction_prompt_additions  # ✗ Type error - doesn't exist on LLMQueryContext
```

Read-only contexts prevent invalid operations:

```python
# MessageContext is read-only - can't halt during message processing
ctx = dispatcher.process_message(message)
ctx.should_halt  # AttributeError — ExecutionContext has no halt fields
# Hooks that return HaltExecution for Message events get logged as warnings
```

## Hook Coordination

For complex scenarios where hooks need to coordinate, create a "meta-hook" that makes coordinated decisions internally:

```python
class AdaptiveBudgetHook(Hook):
    """Single hook that coordinates budget + iteration adjustment."""

    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker

    def on_event(self, event: Event) -> list[Effect]:
        effects = []

        if isinstance(event, ReplIterationEnter):
            utilization = (
                self.cost_tracker.total_llm_cost /
                self.cost_tracker.llm_cost_budget
            )

            # Coordinated decision
            if utilization > 0.8:
                # Reduce iterations
                effects.append(ModifyIterationLimit(
                    new_max=max(3, event.max_iterations // 2),
                    reason="budget_conservation"
                ))
                # AND add explanation to prompt
                effects.append(AddInstructionPrompt(
                    text=f"Budget at {utilization:.1%}, reducing iterations",
                ))

        return effects
```

This is cleaner than having two separate hooks that might conflict.

## Advanced Patterns

### Context-Aware Hook

Hook behavior that changes based on context:

```python
class AdaptiveLogger(Hook):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def on_event(self, event: Event) -> list[Effect]:
        if self.verbose:
            # Log everything
            return [AddInstructionPrompt(f"Event: {type(event).__name__}")]
        elif isinstance(event, (ReplIterationEnter, LLMQueryEnter)):
            # Only log important events
            return [AddInstructionPrompt(f"Event: {type(event).__name__}")]
        return []

# Use different verbosity in different scopes
with AdaptiveLogger(verbose=False):
    invoke("production task", return_type=str)

with AdaptiveLogger(verbose=True):
    invoke("debug task", return_type=str)
```

### Conditional Hook Wrapper

```python
from contextlib import contextmanager

@contextmanager
def maybe_hook(hook: Hook, enabled: bool):
    """Conditionally activate a hook."""
    if enabled:
        with hook:
            yield
    else:
        yield

# Use it
with maybe_hook(PrintLogger(level=logging.DEBUG), enable_debug):
    invoke("task", return_type=str)
```

### Hook Stack Inspection

```python
from jaz.dispatcher import get_current_hooks

def print_active_hooks():
    """Debug utility to see what hooks are active."""
    hook_ctx = get_current_hooks()
    print(f"Active hooks: {[type(h).__name__ for h in hook_ctx.get_active_hooks()]}")
    print(f"Disabled types: {hook_ctx.disabled_types}")
```

## Testing Hooks

The context-based design makes hooks easy to test in isolation:

```python
from jaz.dispatcher import get_dispatcher, isolated_hooks

def test_my_hook():
    """Test hook in isolation."""
    # Use isolated_hooks to avoid default PrintLogger
    with isolated_hooks():
        with MyHook():
            dispatcher = get_dispatcher()

            # Emit test event
            event = ReplIterationEnter(1, 10, "test", 0)
            effects = dispatcher.emit(event)

            # Assert expected effects
            assert len(effects) == 1
            assert isinstance(effects[0], AddInstructionPrompt)
```

## Extension Points

To add new capabilities to the hook system:

### 1. New Event Type

```python
# events/my_event.py
@dataclass
class MyCustomEvent(Event):
    """My custom event."""
    data: str
    timestamp: float

# dispatcher.py - add span method
@contextmanager
def span_my_custom(self, data: str):
    enter_event = MyCustomEvent(data=data, timestamp=time.time())
    effects = self.emit(enter_event)
    ctx = self._compose_my_custom(effects)
    span = Span(ctx=ctx)
    try:
        yield span
    finally:
        if span.is_completed():
            exit_event = MyCustomEventExit(...)
            self.emit(exit_event)
```

### 2. New Effect Type

```python
# effects.py
@dataclass
class MyCustomEffect(Effect):
    """My custom effect."""
    action: str
    data: dict

# dispatcher.py - handle in composition method
def _compose_my_custom(self, effects: list[Effect]) -> MyCustomContext:
    ctx = MyCustomContext()
    for effect in effects:
        match effect:
            case MyCustomEffect(action=action, data=data):
                ctx.actions.append((action, data))
    return ctx
```

### 3. New Hook

Just implement the `Hook` base class:

```python
class MyCustomHook(Hook):
    def on_event(self, event: Event) -> list[Effect]:
        if isinstance(event, MyCustomEvent):
            return [MyCustomEffect(action="process", data={})]
        return []
```

## Migration from Old API

If you have code using the old dispatcher API:

```python
# OLD WAY (deprecated)
dispatcher = HookDispatcher()
dispatcher.add_hook(ReplIterationEnter, MyHook())
agent = Agent(repls=["python"], dispatcher=dispatcher)
result = agent.invoke(...)

# NEW WAY
with MyHook():
    result = invoke(...)  # MyHook automatically active
```

The old API methods (`add_hook`, `add_hook_for_all`, `dispatcher` parameter) have been removed.

## FAQ

**Q: Why can't I register hooks for specific event types anymore?**

A: Hooks now see all events and filter internally using `isinstance()`. This simplifies the API and makes hook behavior more explicit.

**Q: How do I control hook ordering?**

A: Hooks are called in registration order (outer `with` to inner `with`). If you need strict ordering, use a single meta-hook that coordinates internally.

**Q: What if I need different hooks for different invoke calls?**

A: Use nested `with` statements to change the active hooks:

```python
with Hook1():
    invoke("task 1", return_type=str)  # Has Hook1

with Hook2():
    invoke("task 2", return_type=str)  # Has Hook2
```

**Q: Are hooks thread-safe?**

A: Yes! `contextvars` automatically creates per-thread contexts, so each thread has its own set of active hooks.

**Q: Can I use hooks with async code?**

A: Yes! `contextvars` are async-safe, so hooks work correctly across `await` boundaries.

**Q: How do I see what hooks are currently active?**

A: Use `get_current_hooks()` from `jaz.dispatcher`:

```python
from jaz.dispatcher import get_current_hooks

hook_ctx = get_current_hooks()
print([type(h).__name__ for h in hook_ctx.get_active_hooks()])
```
