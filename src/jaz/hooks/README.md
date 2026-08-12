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
- `LLMQueryEnter` - Before making an LLM API call (the always-present per-turn boundary)
- `REPLExecEnter` - Before executing parsed REPL code (conditional — only on runnable code)
- `InvokeEnter` - When `jaz.invoke()` is called

**Events are immutable — treat them as completely read-only.** An `Event` is a frozen dataclass (rebinding a field raises `FrozenInstanceError`), `inputs`/`scope` are read-only mappings, and `hooks` is an immutable tuple. The **only** way a hook influences execution — or communicates with another hook — is by **returning effects**. The same event object is dispatched to every hook, so mutating it would let an earlier-ordered hook silently change what a later-ordered hook sees at the same event; effects avoid this because the dispatcher applies them only *after* the whole hook loop, so one hook's output is invisible to the others at the same event. (Some referenced objects — `config`, the live `hooks` — are deliberately live so you can *read* real runtime state; reading is fine, mutating through them is unsupported.)

### Effects
Typed outputs from hooks that express how they want to influence execution. Effects are order-independent and composed by the dispatcher.

Examples:
- `Abort` - Terminate the invoke with a `Raise` (loop/budget hard-stops). Valid at **every live event**.
- `OverrideResult` - Supply an `ExecResult` at `REPLExecEnter`, skipping execution.
- `ModifyResult` - Transform the `ExecResult` at `REPLExecExit` (e.g. budget-forcing a refusal).
- `AddMessages` - Add message(s) (e.g. instruction text) to the query at `LLMQueryEnter`
- `OverrideResponse` - Supply a pre-computed LLM response, skipping the API call

The two exec-result effects each carry a full `ExecResult`: `OverrideResult` *supplies* one at
`REPLExecEnter` (execution is skipped; multiple must **agree**), `ModifyResult` *transforms* the
result at `REPLExecExit` (multiple **fold** — carried `Continue`s concat output + group
exceptions). `Abort` is *termination* — un-bundled from these (#481) — and is valid at **every
live event**, so loop/budget control has an always-present home. Emitting an exec-result effect
at any other event is ignored with a warning.

**Conditional vs unconditional events (a contract every enforcement hook must
know):** within a span, enter/exit pairing is guaranteed **with one carve-out — an
`Abort` at `LLMQueryEnter` skips the paired `LLMQueryExit`** (the query never happens; the
loop short-circuits to termination). So an observer hook that opens a resource at
`LLMQueryEnter` and closes it at `LLMQueryExit` (e.g. a tracing span) leaks on every
hard-stopped turn unless it *also* cleans up at `InvokeExit` — which is where the invoke
ends and every still-open per-turn resource for that `invoke_id` should be released. The
in-repo `OTelTracing` does exactly this (ends any orphaned `llm_query` child span at
`InvokeExit`). Consumers that merely *read* `LLMQueryExit` (budget/cost, history, loggers)
need no carve-out: an aborted turn ran no query, so there is correctly nothing to record.
Child spans are also **not** guaranteed to open at all — REPL *execution* events only fire
when the LLM response parses to runnable code; a turn whose response fails to parse never
opens an execution span. So never use
execution events as a proxy for "once per turn" — a hard stop hung there silently never
fires for perpetually-unparseable output. Rule of thumb: **liveness / per-turn
enforcement → `Abort` at `LLMQueryEnter`** (the always-present per-turn boundary — it fires
unconditionally once per turn, before the query); **result-composed nudges → execution
exit** (they intentionally compose with what the code actually did). This is how
`IterationLimit` and `BudgetPool` place their hard stops vs soft force-finish nudges.

### Contexts
Typed contracts between the dispatcher and agent containing composed effects. Each event type has a specific context type.

Examples:
- `REPLExecContext` - REPL execution *enter*: can supply a result (OverrideResult) or terminate (Abort)
- `REPLExecExitContext` - REPL execution *exit*: can transform the result (ModifyResult) or terminate (Abort)
- `LLMQueryContext` - LLM query *enter*: can edit the prompt (AddMessages/DropMessages), override the response (OverrideResponse), or terminate (Abort)
- `InvokeContext` - Can add REPL inputs, or terminate (Abort)

### Hooks
Extension points that process events and return effects. Hooks inherit from the `Hook` base class and support the context manager protocol.

```python
from jaz.hooks import Hook, Effect, AddMessages
from jaz.hooks.events import LLMQueryEnter

class MyHook(Hook):
    # Override a typed per-event handler for just the events you care about — typed
    # attribute access, no isinstance/match, and unimplemented events default to no-op.
    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Prompt text is added as an AddMessages at LLMQueryEnter (the context
        # where message edits compose).
        return [
            AddMessages(
                messages=[{"role": "user", "content": "Be concise."}],
                index=len(event.messages),
            )
        ]
```

A cross-cutting observer that treats **every** event uniformly (logging, tracing) overrides
`on_any(event)` instead — the dispatcher calls it *in addition to* the matched typed handler,
so it composes with (never replaces) the per-event dispatch, and one hook may use both.
Don't override `_dispatch_event`: it's the framework's private internal event router
(#597, renamed from `on_event` in #740 so the router no longer looks like an `on_<event>`
override point). Override the typed `on_<event>` handlers or `on_any` instead.

### Hook serialization (`to_dict` / `from_dict`)

`Hook.to_dict()` serializes a hook to a JSON-safe `{"class", "qualified_name", "params"}` dict,
and `Hook.from_dict(d)` reconstructs it. Two uses: recording *what governance applied* in an
observability trace (the loggers/tracers serialize each active hook at `InvokeEnter`), and
round-tripping `Config.baseline_hooks` through `Config.to_dict()` → reconstruct.

**Make a hook a dataclass and serialization is free** — no boilerplate:

```python
from dataclasses import dataclass
from jaz.hooks import Hook

@dataclass(eq=False)          # eq=False: hooks are identity objects (deduped by `is`, not value)
class MyHook(Hook):
    threshold: float = 0.5

MyHook(threshold=0.9).to_dict()   # {"class": "MyHook", "qualified_name": ..., "params": {"threshold": 0.9}}
Hook.from_dict(d)                 # -> MyHook(threshold=0.9)
```

Rules for the dataclass path:
- Each `init` field becomes a param; `field(init=False)` runtime state (counters, per-invoke
  dicts, open handles) is **excluded**.
- A field whose value isn't JSON-safe and has no encoder is **omitted** — it round-trips to its
  constructor default rather than crashing serialization.
- For a non-JSON-safe value you *do* want preserved, declare a per-field
  `field(metadata={"to_dict": encode, "from_dict": decode})` pair (an encoder returning `None`
  omits the key). This is how `IterationLimit.must_exit_warning` (a callable/str/None)
  round-trips.
- Put validation in `__post_init__` (e.g. a range check), and heavy `__init__` work / derived
  state there too.

**Reconstruction never imports by name.** `from_dict` matches `class`/`qualified_name` against a
registry auto-populated when each `Hook` subclass is defined (`__init_subclass__`), so any
*imported* subclass **resolves** — built-ins are covered because importing `jaz.hooks` imports
them all; a custom hook is resolvable once its module is imported. An unknown/unimported class
raises `ValueError`. A short-name collision requires a matching `qualified_name` to disambiguate.
A full *round-trip* additionally needs the params, so it holds for any imported **dataclass** hook
(or one overriding `_to_dict_params`) — a plain-class hook with required constructor args
serializes to empty params and fails to reconstruct.

A hook that can't reasonably be a dataclass (live objects, `**kwargs`, credentials) stays a plain
class and overrides `_to_dict_params()` (the explicit escape hatch), or simply doesn't serialize
its params.

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
from jaz import ReturnType, invoke
from jaz.hooks import PrintLogger, BudgetPool
import logging

# Simple hook activation
with PrintLogger(level=logging.INFO):
    result = invoke(ReturnType(int), task="Calculate 2+2")
    # PrintLogger is active during this invoke

# Nested hooks - both active
with PrintLogger(level=logging.INFO):
    with BudgetPool(cost_budget=1.0, calls_budget=50):
        result = invoke(ReturnType(str), task="Complex task")
        # Both PrintLogger and BudgetPool are active
```

### Automatic Propagation

Hooks automatically propagate to nested `invoke()` calls via `contextvars`:

```python
from jaz import ReturnType, invoke
from jaz.hooks import PrintLogger

with PrintLogger(level=logging.INFO):
    # This invoke has PrintLogger
    result = invoke(
        ReturnType(int),
        task="""
        Call invoke again with prompt: 'What is 5 * 5?'
        """,
    )
    # The nested invoke() call also has PrintLogger automatically!
```

This works across any level of nesting - hooks propagate through:
- Nested `invoke()` calls
- Function calls within the hook context
- Async functions (contextvars are async-safe)
- Threads started *inside* the hook context (each inherits a copy)

> **Thread pool caveat**: `ThreadPoolExecutor` workers do **not** inherit the
> calling thread's hooks — Python's `contextvars` only propagates to threads
> that are *started* inside the active context, not to pre-existing pool
> workers.
>
> Inside a **`jaz.invoke` sub-invoke** run in such a worker this is largely
> mitigated (#727): the ancestor hook chain is closure-threaded on the prehook and
> **re-based** onto the worker — composed *under* any hooks the worker activates
> locally (`ancestor + local`, so both fire), rather than the old either/or that
> dropped the ancestor whenever the worker had its own hook. What is **not**
> recovered automatically is a hook the agent activated **mid-REPL** and then a raw
> pool worker: `prehook.parent_hooks` was snapshotted before the REPL ran, and
> contextvars can't cross the pool boundary. To carry the full live context
> (including mid-REPL `with Hook()` activations) into a worker, wrap the call in
> `contextvars.copy_context()`:
>
> ```python
> import contextvars
> from concurrent.futures import ThreadPoolExecutor
>
> ctx = contextvars.copy_context()  # snapshots the live hook + config context
> with ThreadPoolExecutor() as ex:
>     ex.submit(ctx.run, lambda: jaz.invoke(jaz.ReturnType(str), task="subtask")).result()
> ```
>
> `copy_context()` carries the *established* context across, so the worker sees the
> full parent state and no re-base is needed. Outside a sub-invoke (a bare hook, no
> `jaz.invoke`), the caveat is unchanged — activate the hook *inside* the worker or
> use `copy_context()`.

### Default Global Context

The default hook context is **empty**. Hooks are opt-in: nothing runs until
you activate a hook via `with HookClass():`.

Future essential built-in behaviors (budget enforcement, permissions, etc.)
will be auto-installed by `jaz.invoke()`. Each such built-in will ship with
its own dedicated, behavior-specific opt-out (e.g., a hypothetical
`bypass_budget()`) — **not** a generic disable/clear mechanism.

There is deliberately **no** generic way to disable or clear hooks — neither a
class-keyed `disable_hook(SomeClass)` nor a blanket `clear_all_hooks()` (#466
removed both). Hook lifecycle is managed purely by lexical `with` scoping:

1. Multiple instances of the same `Hook` class can be active simultaneously
   (e.g., two `PrintLogger`s at different levels). A class-keyed disable
   conflates class with instance identity and silently kills all of them.
2. A blanket clear is a footgun: it silently strips governance / safety
   baseline hooks (e.g. the loop's only termination guarantee), and its effect
   can't even be honored in raw worker threads (they restore the ancestor
   context) — a false sense of a clean slate.
3. Named, behavior-specific opt-outs document intent at the call site and are
   far easier to audit than a generic kill switch.

### Helper Functions

```python
# Activate multiple hooks with native Python syntax
with Hook1(), Hook2(), Hook3():
    invoke(...)  # All three hooks active
```

## Architecture

### Context Variable Flow

```
┌─────────────────────────────────────────────────────────┐
│ Global Default Context                                   │
│ hooks: ()  # empty — hooks are opt-in                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────┐
           │ with MyHook():              │
           │ hooks: (MyHook,)            │
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
                    │ • Abort → Raise (supersedes ModifyResult)
                    │ • ModifyResult → transforms the ExecResult
                    │ • ALL prompts → concatenate
                    │ • MIN iteration limit
                    │ • ALL metrics → record
                    │
                    ▼
          ┌──────────────────────┐
          │ ExecutionContext     │
          │ (Typed)              │
          │                      │
          │ • error_effects: list│
          │ • raise_effects: list│
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
from jaz.hooks import (
    Hook, Event, Effect, LLMQueryEnter, Abort, AddMessages,
)

class WrapUpHook(Hook):
    """Example custom hook: nudge the agent to finish after `soft` REPL
    iterations, then hard-stop after `hard`.

    Illustrates the two effect kinds you'll reach for most — a soft nudge via
    AddMessages (prompt modification, emitted at LLMQueryEnter) and a terminal abort
    via Abort — plus the fact that a hook may hold its own state across events.

    Both live on ``LLMQueryEnter``, the always-present per-turn boundary: the nudge
    because that's the only context where message edits compose, and the hard stop because
    ``Abort`` there fires once per turn *before* the query (so it never wastes a call and
    never silently skips a parse-failure turn the way a conditional execution event would).

    Note ``_iterations`` here is **tree-wide**: one hook instance propagates to
    nested ``invoke()`` calls via contextvars, so the count spans the whole
    invoke tree rather than each invoke separately. For per-invoke state, key your
    tracking by ``event.invoke_id`` — which is exactly why the built-in
    BudgetPool keys its trackers by invoke_id. For real LLM cost /
    call-count budgets, use that built-in rather than rolling your own.
    """

    def __init__(self, soft: int = 5, hard: int = 8):
        self.soft = soft
        self.hard = hard
        self._iterations = 0

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # LLMQueryEnter fires once per turn, before the query — count here.
        self._iterations += 1

        # Once past `hard`, terminate the invoke: Abort resolves to a Raise that
        # propagates out of invoke().
        if self._iterations > self.hard:
            return [Abort(error=RuntimeError("Iteration limit exceeded"))]

        # Once past `soft`, nudge the agent to finish. The nudge is an AddMessages
        # appended to the end of this query.
        if self._iterations > self.soft:
            return [
                AddMessages(
                    messages=[{"role": "user", "content": "Wrap up and RETURN soon."}],
                    index=len(event.messages),
                )
            ]

        return []
```

### 2. Use Hook as Context Manager

```python
from jaz import ReturnType, invoke

# Activate hook for specific scope
with WrapUpHook():
    result = invoke(ReturnType(str), task="Expensive task")
    # WrapUpHook is active during this invoke
```

### 3. Combine Multiple Hooks

```python
from jaz.hooks import PrintLogger, BudgetPool
import logging

# All three hooks active; they compose and propagate together
with PrintLogger(level=logging.INFO):
    with BudgetPool(cost_budget=1.0, calls_budget=50):
        with WrapUpHook():
            result = invoke(ReturnType(str), task="Task with all protections")
```

### 4. Conditional Hook Activation

```python
def my_function(enable_logging: bool = True):
    if enable_logging:
        with PrintLogger(level=logging.INFO):
            return invoke(ReturnType(str), task="task")
    else:
        return invoke(ReturnType(str), task="silent task")  # No logging hook
```

### 5. Use in Agent (Internal)

```python
from jaz.hooks import get_dispatcher

# In Agent.do_one_repl_iteration()
dispatcher = get_dispatcher()

with dispatcher.span_repl_exec(enter_event) as span:
    if span.enter_override is not None:
        # An enter-time OverrideResult (supply) / Abort short-circuits execution.
        exec_result = span.enter_override
    else:
        # Execute REPL code (REPL inputs are injected once at InvokeEnter, not here —
        # AddInput is InvokeEnter-only, #481)
        exec_result = repl.exec(code, ...)

    # Complete the span
    span.complete(exec_result=exec_result)

# Apply any exit-time ModifyResult transform / Abort.
exec_result = span.get_final_exec_result()
```

## Composition Rules

The dispatcher applies explicit composition rules when combining effects from multiple hooks:

### Exec-result / terminate effects (`OverrideResult` / `ModifyResult` / `Abort`)
`OverrideResult` is valid only at `REPLExecEnter` (supply), `ModifyResult` only at `REPLExecExit`
(transform); `Abort` is valid at every live event. Where they co-occur (the REPL execution
boundaries) they compose into the REPL result types (`Continue` / `Return` / `Raise`):

- **Enter (`OverrideResult`)**: execution is skipped and the supplied result is used. Multiple
  `OverrideResult`s **fold** among themselves (no `exec_result` to fold onto yet) by the same
  precedence as the exit boundary — carried `Continue`s concatenate their `output` and group their
  exceptions, so two hooks vetoing the same input (e.g. a `ValidateREPLInput` and the evals
  `RestrictReturnValue`) compose into one recoverable rejection; distinct carried `Return` *values*
  still raise `ReturnValueConflictError`. `Abort` supersedes them.
- **Exit (`ModifyResult`)**: the composed result supersedes the actual `exec_result`; multiple
  **fold** by carried-result kind precedence `Raise` > `Return` > `Continue`. Carried `Continue`s
  concatenate their `output` onto the original's and group their exceptions into an
  `ExceptionGroup`; carried `Return`s cannot merge (two distinct values raise
  `ReturnValueConflictError`, identical ones coalesce); carried `Raise`s group their exceptions. The original's
  exception is folded in only when the outcome is of the same final type; the original `output` is
  preserved.

Both boundaries share one fold (`_fold_carried_results`); enter passes `original=None`, exit passes
the executed result.
- **`Abort` supersedes** the exec-result effects at either boundary (termination trumps
  supply/transform) → `Raise`. Multiple `Abort`s group their exceptions into an `ExceptionGroup`.

### Prompt Modifications
- **Rule**: prompt text is added via `AddMessages` at `LLMQueryEnter`; all adds compose into the query through the message-edit fold (`apply_message_edits`)
- **Order**: deterministic and hook-order-independent — same-slot adds are ordered by `(sort_key, canonical content)`
- **Example**: Budget warning + safety note both appear in the query

## Type Safety

The dispatcher uses typed contexts to prevent invalid operations:

```python
# Type checker knows ctx is REPLExecContext!
with dispatcher.span_repl_exec(...) as span:
    span.ctx.override_effects  # ✓ Exists
    span.ctx.message_edits  # ✗ Type error - doesn't exist on REPLExecContext

# Type checker knows ctx is LLMQueryContext!
with dispatcher.span_llm_query(...) as span:
    span.ctx.message_edits  # ✓ Exists
    span.ctx.override_response  # ✓ Exists
    span.ctx.override_effects  # ✗ Type error - doesn't exist on LLMQueryContext
```

Read-only contexts prevent invalid operations:

```python
# An OverrideResult / ModifyResult emitted at an event that can't compose it
# (e.g. LLMQueryEnter) is logged as a warning and ignored — but Abort is
# valid there and terminates the invoke.
```

## Hook Coordination

For complex scenarios where hooks need to coordinate, create a "meta-hook" that makes coordinated decisions internally:

```python
class AdaptiveBudgetHook(Hook):
    """Single hook that coordinates a budget warning + abort decision."""

    def __init__(self, cost_tracker: CostTracker, budget: float):
        # CostTracker is a budget-less accounting node now; the meta-hook holds its
        # own budget (mirrors how BudgetTrackingHook owns the one pool budget).
        self.cost_tracker = cost_tracker
        self.budget = budget

    def _utilization(self) -> float:
        return self.cost_tracker.total_llm_cost / self.budget

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Effect]:
        # Both decisions live on LLMQueryEnter (once per turn, before the query).
        # Check the hard stop first so an exhausted budget terminates rather than
        # merely warning.
        if self._utilization() >= 1.0:
            # Abort is valid at every live event and resolves to a terminal Raise.
            return [Abort(error=BudgetExhaustedError())]

        # Otherwise, warn the agent to wrap up (prompt modification, composed at the query).
        if self._utilization() > 0.8:
            return [AddMessages(
                messages=[{
                    "role": "user",
                    "content": f"Budget at {self._utilization():.1%}; finish and RETURN soon.",
                }],
                index=len(event.messages),
            )]

        return []
```

This is cleaner than having two separate hooks that might conflict.

## Advanced Patterns

### Context-Aware Hook

Hook behavior that changes based on context:

```python
class AdaptiveLogger(Hook):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def on_any(self, event: Event) -> list[Effect]:
        # Prompt additions only compose at LLMQueryEnter; add more detail when verbose.
        if isinstance(event, LLMQueryEnter):
            note = "verbose context note" if self.verbose else "brief context note"
            return [
                AddMessages(
                    messages=[{"role": "user", "content": note}],
                    index=len(event.messages),
                )
            ]
        return []

# Use different verbosity in different scopes
with AdaptiveLogger(verbose=False):
    invoke(ReturnType(str), task="production task")

with AdaptiveLogger(verbose=True):
    invoke(ReturnType(str), task="debug task")
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
    invoke(ReturnType(str), task="task")
```

### Hook Stack Inspection

```python
from jaz.hooks.context import get_current_hooks

def print_active_hooks():
    """Debug utility to see what hooks are active."""
    hook_ctx = get_current_hooks()
    print(f"Active hooks: {[type(h).__name__ for h in hook_ctx.hooks]}")
```

## Testing Hooks

The context-based design makes hooks easy to test in isolation:

```python
from jaz.hooks import get_dispatcher

def test_my_hook():
    """Test hook in isolation."""
    # The default context is empty, so MyHook is the only active hook.
    with MyHook():
        dispatcher = get_dispatcher()

        # Emit test event. Every Event carries the invoke's effective `config`
        # (the dispatcher resolves baseline_hooks from it — see #463); in a test,
        # `get_config()` is the convenient default.
        event = LLMQueryEnter(
            config=get_config(),
            invoke_id="test",
            messages=[],
            model="test-model",
            iteration=1,
            depth=0,
            can_recurse=True,
        )
        effects = dispatcher.emit(event)

        # Assert expected effects (a prompt addition is an AddMessages at LLMQueryEnter).
        assert len(effects) == 1
        assert isinstance(effects[0], AddMessages)
```

## Extension Points

To add new capabilities to the hook system:

### 1. New Event Type

```python
# events/my_event.py
@dataclass
class MyCustomEvent(Event):
    """My custom event. Inherits the required `config` field from Event (#463)."""
    data: str
    timestamp: float

# dispatcher.py - add span method. The Agent passes the invoke's effective config
# into the enter event (self.config); the dispatcher copies it onto the exit event.
@contextmanager
def span_my_custom(self, config: "Config", data: str):
    enter_event = MyCustomEvent(config=config, data=data, timestamp=time.time())
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
    def on_any(self, event: Event) -> list[Effect]:
        if isinstance(event, MyCustomEvent):
            return [MyCustomEffect(action="process", data={})]
        return []
```

## Migration from Old API

If you have code using the old dispatcher API:

```python
# OLD WAY (deprecated)
dispatcher = HookDispatcher()
dispatcher.add_hook(LLMQueryEnter, MyHook())
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
    invoke(ReturnType(str), task="task 1")  # Has Hook1

with Hook2():
    invoke(ReturnType(str), task="task 2")  # Has Hook2
```

**Q: Are hooks thread-safe?**

A: Yes! `contextvars` automatically creates per-thread contexts, so each thread has its own set of active hooks.

**Q: Can I use hooks with async code?**

A: Yes! `contextvars` are async-safe, so hooks work correctly across `await` boundaries.

**Q: How do I see what hooks are currently active?**

A: Use `get_current_hooks()` from `jaz.hooks.context`:

```python
from jaz.hooks.context import get_current_hooks

hook_ctx = get_current_hooks()
print([type(h).__name__ for h in hook_ctx.hooks])
```
